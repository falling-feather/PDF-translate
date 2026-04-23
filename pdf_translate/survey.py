from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from pdf_translate.config import AppConfig
from pdf_translate.translators.http_retry import call_with_http_retry


@dataclass
class ChunkSurveyResult:
    """巡视输出：术语草稿 + 任务标签（后续识图/排版消费）。"""

    figure_heavy: bool = False
    text_ratio: float = 0.8
    needs_ocr: bool = False
    needs_vlm: bool = False
    draft_terms: list[dict[str, str]] = field(default_factory=list)
    notes: str = ""
    raw_response: str = ""
    skipped: bool = False
    error: str | None = None


SURVEY_SYSTEM = """你是学术 PDF 分块分析助手。用户会提供：页码范围、该块纯文本（可能不完整）、以及该块在 PDF 中的图片数量与超链接数量统计。
注意：未提供页面截图，请仅根据文本与统计做**保守估计**。
你必须只输出一个 JSON 对象，不要 Markdown 围栏，不要解释。JSON 字段如下：
{
  "figure_heavy": boolean,
  "text_ratio": number,
  "needs_ocr": boolean,
  "needs_vlm": boolean,
  "draft_terms": [ {"en": "英文术语或专名", "zh": "建议中文译法"} ],
  "notes": "可选简短说明"
}
要求：draft_terms 只列本块明显重要的专名/术语（通常不超过 25 条）；若无则 []。"""


def _truncate(s: str, max_chars: int) -> str:
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 20] + "\n…(截断)…"


def _extract_json_object(text: str) -> dict[str, Any]:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", t)
    if m:
        t = m.group(1).strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    i0 = t.find("{")
    i1 = t.rfind("}")
    if i0 >= 0 and i1 > i0:
        try:
            obj = json.loads(t[i0 : i1 + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {}


def _normalize_terms(raw: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        en = str(item.get("en", "")).strip()
        zh = str(item.get("zh", "")).strip()
        if en and zh:
            out.append({"en": en, "zh": zh})
    return out


def run_chunk_survey(
    cfg: AppConfig,
    *,
    chunk_text: str,
    chunk_id: str,
    pages_1based: tuple[int, int],
    image_count: int,
    link_count: int,
) -> ChunkSurveyResult:
    """译前巡视：未开启或缺少密钥时返回 skipped 结果。"""
    if not cfg.survey_enabled:
        return ChunkSurveyResult(skipped=True)
    key = (cfg.siliconflow_api_key or "").strip()
    if not key:
        return ChunkSurveyResult(skipped=True, error="SILICONFLOW_API_KEY 未设置")
    model = (cfg.siliconflow_survey_model or "").strip()
    if not model:
        return ChunkSurveyResult(skipped=True, error="SILICONFLOW_SURVEY_MODEL 未设置")
    if "deepseek" in model.lower():
        return ChunkSurveyResult(skipped=True, error="硅基巡视模型禁止使用 DeepSeek；请改为 Qwen/Kimi 等模型。")

    p0, p1 = pages_1based
    body_text = _truncate(chunk_text, cfg.survey_max_text_chars)
    user_msg = (
        f"chunk_id: {chunk_id}\n"
        f"pages_1based: {p0}–{p1}\n"
        f"image_count (from PDF): {image_count}\n"
        f"link_count (from PDF): {link_count}\n\n"
        f"--- 块文本 ---\n{body_text}"
    )

    url = f"{cfg.siliconflow_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    req_body: dict[str, Any] = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": SURVEY_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
    }
    read_s = max(60.0, float(cfg.http_timeout_s))
    timeout = httpx.Timeout(connect=45.0, read=read_s, write=120.0, pool=45.0)

    def _do() -> str:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, headers=headers, json=req_body)
            r.raise_for_status()
            data = r.json()
        try:
            return str(data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Unexpected survey API response: {json.dumps(data, ensure_ascii=False)[:600]}") from e

    try:
        raw = call_with_http_retry(_do, context="SiliconFlow survey")
    except Exception as e:
        return ChunkSurveyResult(skipped=True, error=str(e), raw_response="")

    parsed = _extract_json_object(raw)
    terms = _normalize_terms(parsed.get("draft_terms"))
    fig = bool(parsed.get("figure_heavy"))
    try:
        tr = float(parsed.get("text_ratio", 0.8))
    except (TypeError, ValueError):
        tr = 0.8
    tr = max(0.0, min(1.0, tr))
    notes = str(parsed.get("notes") or "").strip()

    # 元数据兜底：图较多但模型未标 heavy 时标记为 figure_heavy（后续识图管线可消费）
    needs_ocr = bool(parsed.get("needs_ocr"))
    needs_vlm = bool(parsed.get("needs_vlm"))
    if image_count >= 2 and not fig:
        fig = True

    return ChunkSurveyResult(
        figure_heavy=fig,
        text_ratio=tr,
        needs_ocr=needs_ocr,
        needs_vlm=needs_vlm,
        draft_terms=terms,
        notes=notes,
        raw_response=raw,
        skipped=False,
    )


def survey_result_to_jsonable(r: ChunkSurveyResult) -> dict[str, Any]:
    return {
        "figure_heavy": r.figure_heavy,
        "text_ratio": r.text_ratio,
        "needs_ocr": r.needs_ocr,
        "needs_vlm": r.needs_vlm,
        "draft_terms": r.draft_terms,
        "notes": r.notes,
        "skipped": r.skipped,
        "error": r.error,
    }
