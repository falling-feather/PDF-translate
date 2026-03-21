from __future__ import annotations

import hashlib
import json
from typing import Any

import httpx

from pdf_translate.deferral_markers import DEFERRAL_PROTOCOL_USER_BLOCK
from pdf_translate.translators.base import TranslationRequest
from pdf_translate.translators.http_retry import call_with_http_retry


SYSTEM_PROMPT_VERSION = "v3"

SYSTEM_PROMPT = """你是学术文献翻译助手。将用户给出的英文论文片段译为简体中文。
要求：
1. 保留数学公式、符号、编号与原样可辨认的格式（用 LaTeX 或原文保留）。
2. 严格遵守术语表中的译法；术语表未出现的专有名词首次可音译或保留英文并在括号内注明。
3. 与前文摘要中的叙事线索保持一致，指代清晰。
4. 若给出「上一块译文结尾」，仅用于语气与指代衔接：不要把它逐字重复进本块输出；若原文与上一块重叠，不要重复已译过的等价句子。
5. 若用户消息含「段尾顺延协议」，须严格按协议结构输出（含标识符与未译英文尾段）；否则只输出译文正文。
6. 不要输出解释性文字。"""


def _build_user_message(req: TranslationRequest) -> str:
    parts = []
    if req.style_notes.strip():
        parts.append("【风格说明】\n" + req.style_notes.strip())
    if req.glossary_excerpt.strip():
        parts.append("【术语表】\n" + req.glossary_excerpt.strip())
    if req.prior_summaries.strip():
        parts.append("【前文摘要】\n" + req.prior_summaries.strip())
    if req.continuation_hint.strip():
        parts.append("【衔接要求】\n" + req.continuation_hint.strip())
    if req.prior_tail_zh.strip():
        parts.append("【上一块译文结尾（仅供衔接，请勿逐字重复）】\n" + req.prior_tail_zh.strip())
    if req.prior_untranslated_continuation.strip():
        parts.append(
            "【紧接上段未译英文（须先与此处衔接译出，再接【待译正文】；勿重复已给出的中文）】\n"
            + req.prior_untranslated_continuation.strip()
        )
    if req.defer_source_tail_protocol:
        parts.append(DEFERRAL_PROTOCOL_USER_BLOCK)
    parts.append("【待译正文】\n" + req.source_text)
    return "\n\n".join(parts)


POLISH_SYSTEM_PROMPT = """你是中文学术编辑。输入为机器翻译得到的中文草稿。请在尽量不改变原意的前提下：
统一术语（对照术语表）、理顺指代与语序、保留公式与符号。
只输出修订后的中文正文，不要解释。"""


class OpenAICompatibleTranslator:
    """兼容 OpenAI Chat Completions 的 HTTP API（含多数代理与 Azure OpenAI 变体，见 SETUP_MANUAL）。"""

    name = "openai_compatible"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_s: float = 120.0,
        system_prompt: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = max(30.0, timeout_s)
        self.system_prompt = system_prompt or SYSTEM_PROMPT

    def translate(self, req: TranslationRequest) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": _build_user_message(req)},
            ],
        }
        read_s = max(60.0, float(self.timeout_s))
        timeout = httpx.Timeout(connect=45.0, read=read_s, write=120.0, pool=45.0)

        def _do() -> str:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(url, headers=headers, json=body)
                r.raise_for_status()
                data = r.json()
            try:
                return data["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError) as e:
                raise RuntimeError(
                    f"Unexpected API response: {json.dumps(data, ensure_ascii=False)[:800]}"
                ) from e

        return call_with_http_retry(_do, context="Chat Completions")


def prompt_fingerprint() -> str:
    return hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12]
