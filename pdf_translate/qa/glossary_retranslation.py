from __future__ import annotations

import json
import hashlib
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pdf_translate.chunking import TextChunk
from pdf_translate.deferral_markers import finalize_merged_translation_markdown, strip_yaml_front_matter
from pdf_translate.memory_store import MemoryStore, _pending_dedupe_key, _term_scope_matches
from pdf_translate.translators.base import TranslationRequest, Translator
from pdf_translate.translators.openai_compatible import SYSTEM_PROMPT_VERSION, prompt_fingerprint

SCHEMA_VERSION = "glossary-retranslation-plan-v1"
EXECUTION_SCHEMA_VERSION = "glossary-retranslation-execution-v1"
PUBLISH_SCHEMA_VERSION = "glossary-retranslation-publish-v1"
ROLLBACK_SCHEMA_VERSION = "glossary-retranslation-rollback-v1"


def _read_json_dict(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        return dict(default or {})
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else dict(default or {})


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _unique_texts(values: Any) -> list[str]:
    raw_values: list[Any]
    if isinstance(values, list):
        raw_values = values
    elif values is None:
        raw_values = []
    else:
        raw_values = [values]
    out: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        text = str(value or "").strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _normalized_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _contains_source_term(text: str, term: str) -> bool:
    normalized_term = _normalized_space(term)
    if not normalized_term:
        return False
    escaped = re.escape(normalized_term).replace(r"\ ", r"\s+")
    pattern = re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)
    return bool(pattern.search(text or ""))


def _contains_translation(text: str, phrase: str) -> bool:
    phrase = str(phrase or "").strip()
    if not phrase:
        return False
    if phrase in text:
        return True
    return _normalized_space(phrase) in _normalized_space(text)


def _strip_markdown_frontmatter(text: str) -> str:
    if not text.startswith("---"):
        return text
    parts = text.split("---", 2)
    if len(parts) == 3:
        return parts[2].lstrip()
    return text


def _read_chunk_translation(chunk_dir: Path, chunk_id: str) -> tuple[str, str]:
    path = chunk_dir / f"{chunk_id}.md"
    if not path.is_file() or path.stat().st_size == 0:
        return "", ""
    rel_path = f"output/chunks/{chunk_id}.md"
    return _strip_markdown_frontmatter(path.read_text(encoding="utf-8")), rel_path


def _entry_source_text(entry: dict[str, Any], output_dir: Path) -> tuple[str, str]:
    """Return source text and a stable relative path when the manifest points to one."""
    inline_text = str(entry.get("text") or "")
    if inline_text:
        return inline_text, "chunks_manifest.text"

    raw_path = str(entry.get("source_text_path") or entry.get("source_path") or "").strip()
    if not raw_path:
        return "", ""
    work_dir = output_dir.parent.resolve()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = work_dir / raw_path
    try:
        resolved = candidate.resolve()
        resolved.relative_to(work_dir)
    except (OSError, ValueError):
        return "", ""
    if not resolved.is_file():
        return "", raw_path.replace("\\", "/")
    try:
        rel = resolved.relative_to(work_dir).as_posix()
    except ValueError:
        rel = raw_path.replace("\\", "/")
    return resolved.read_text(encoding="utf-8"), rel


def _confirmed_review_terms(memory_dir: Path) -> list[dict[str, Any]]:
    pending = _read_json_dict(memory_dir / "pending_review.json", {"items": []})
    glossary = _read_json_dict(memory_dir / "glossary.json", {"terms": []})
    terms: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in pending.get("items") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "glossary_conflict" or item.get("status") != "confirmed":
            continue
        en = str(item.get("en") or "").strip()
        confirmed_zh = str(item.get("confirmed_zh") or item.get("candidate_zh") or "").strip()
        if not en or not confirmed_zh:
            continue
        review_id = str(item.get("dedupe_key") or _pending_dedupe_key(item))
        previous_zh = _unique_texts(item.get("existing_zh"))
        original_candidate = str(item.get("original_candidate_zh") or "").strip()
        if original_candidate and original_candidate != confirmed_zh:
            previous_zh.append(original_candidate)
        previous_zh = [value for value in _unique_texts(previous_zh) if value != confirmed_zh]
        term = {
            "review_id": review_id,
            "source": "pending_review",
            "en": en,
            "confirmed_zh": confirmed_zh,
            "previous_zh": previous_zh,
            "first_page": item.get("first_page"),
            "section_scope": str(item.get("section_scope") or "").strip(),
            "reviewed_at": item.get("reviewed_at"),
            "reviewed_by": item.get("reviewed_by"),
            "confidence": item.get("confidence"),
        }
        terms.append(term)
        seen.add(f"{en.lower()}->{confirmed_zh}")

    for item in glossary.get("terms") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("status") or "").lower() != "confirmed":
            continue
        if item.get("review_decision") != "confirm_candidate":
            continue
        en = str(item.get("en") or "").strip()
        confirmed_zh = str(item.get("zh") or "").strip()
        if not en or not confirmed_zh or f"{en.lower()}->{confirmed_zh}" in seen:
            continue
        previous_zh = _unique_texts(item.get("previous_zh"))
        original_candidate = str(item.get("original_candidate_zh") or "").strip()
        if original_candidate:
            previous_zh.append(original_candidate)
        terms.append(
            {
                "review_id": f"glossary:{en.lower()}:{confirmed_zh}",
                "source": "glossary",
                "en": en,
                "confirmed_zh": confirmed_zh,
                "previous_zh": [value for value in _unique_texts(previous_zh) if value != confirmed_zh],
                "first_page": item.get("first_page"),
                "section_scope": str(item.get("section_scope") or "").strip(),
                "reviewed_at": item.get("reviewed_at"),
                "reviewed_by": item.get("reviewed_by"),
                "confidence": item.get("confidence"),
            }
        )

    return terms


def _entry_pages(entry: dict[str, Any]) -> list[int]:
    pages: list[int] = []
    for item in entry.get("pages_1based") or []:
        try:
            page_no = int(item)
        except (TypeError, ValueError):
            continue
        if page_no > 0:
            pages.append(page_no)
    return pages


def _entry_structure_types(entry: dict[str, Any]) -> list[str]:
    block_types = entry.get("block_types")
    if isinstance(block_types, dict):
        return [str(key) for key, value in block_types.items() if int(value or 0) > 0]
    return []


def _entry_section_scopes(entry: dict[str, Any]) -> list[str]:
    raw = entry.get("section_scopes")
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item or "").strip()]
    single = str(entry.get("section_scope") or "").strip()
    return [single] if single else []


def _entry_block_ids(entry: dict[str, Any]) -> list[str]:
    raw = entry.get("block_ids")
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item or "").strip()]
    return []


def _term_scope_matches_entry(term: dict[str, Any], entry: dict[str, Any]) -> bool:
    return _term_scope_matches(
        {"section_scope": term.get("section_scope")},
        section_scope=_entry_section_scopes(entry),
        structure_types=_entry_structure_types(entry),
        block_ids=_entry_block_ids(entry),
    )


def build_glossary_retranslation_plan(
    output_dir: Path,
    memory_dir: Path,
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a local plan describing chunks affected by confirmed glossary decisions."""
    output_dir = output_dir.resolve()
    memory_dir = memory_dir.resolve()
    entries = _read_json_list(output_dir / "chunks_manifest.json")
    chunk_dir = output_dir / "chunks"
    confirmed_terms = _confirmed_review_terms(memory_dir)
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()

    chunk_records: dict[str, dict[str, Any]] = {}
    term_impacts: list[dict[str, Any]] = []

    for term in confirmed_terms:
        matched_chunk_ids: list[str] = []
        stale_chunk_ids: list[str] = []
        ready_chunk_ids: list[str] = []
        en = term["en"]
        confirmed_zh = term["confirmed_zh"]
        previous_zh = list(term.get("previous_zh") or [])

        for entry in entries:
            chunk_id = str(entry.get("chunk_id") or "").strip()
            if not chunk_id or not _term_scope_matches_entry(term, entry):
                continue
            source_text, source_text_path = _entry_source_text(entry, output_dir)
            if not _contains_source_term(source_text, en):
                continue

            translation_text, translation_path = _read_chunk_translation(chunk_dir, chunk_id)
            translation_missing = not translation_path
            confirmed_present = _contains_translation(translation_text, confirmed_zh)
            previous_present = [
                value for value in previous_zh if _contains_translation(translation_text, value)
            ]
            stale_reasons: list[str] = []
            if translation_missing:
                stale_reasons.append("translation_chunk_missing")
            if previous_present:
                stale_reasons.append("contains_previous_translation")
            if not translation_missing and not confirmed_present:
                stale_reasons.append("missing_confirmed_translation")
            recommended_action = "retranslate_chunk" if stale_reasons else "keep_translation"

            record = chunk_records.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "pages_1based": _entry_pages(entry),
                    "section_scopes": _entry_section_scopes(entry),
                    "structure_types": _entry_structure_types(entry),
                    "block_ids": _entry_block_ids(entry),
                    "source_text_path": source_text_path,
                    "translation_path": translation_path,
                    "matched_terms": [],
                    "recommended_action": "keep_translation",
                    "stale_reasons": [],
                },
            )
            term_match = {
                "review_id": term["review_id"],
                "en": en,
                "confirmed_zh": confirmed_zh,
                "previous_zh_present": previous_present,
                "translation_contains_confirmed_zh": confirmed_present,
                "translation_missing": translation_missing,
                "stale_reasons": stale_reasons,
                "recommended_action": recommended_action,
            }
            record["matched_terms"].append(term_match)
            if stale_reasons:
                record["recommended_action"] = "retranslate_chunk"
                for reason in stale_reasons:
                    if reason not in record["stale_reasons"]:
                        record["stale_reasons"].append(reason)

            matched_chunk_ids.append(chunk_id)
            if stale_reasons:
                stale_chunk_ids.append(chunk_id)
            else:
                ready_chunk_ids.append(chunk_id)

        term_impacts.append(
            {
                "review_id": term["review_id"],
                "source": term.get("source"),
                "en": en,
                "confirmed_zh": confirmed_zh,
                "previous_zh": previous_zh,
                "first_page": term.get("first_page"),
                "section_scope": term.get("section_scope") or "",
                "confidence": term.get("confidence"),
                "reviewed_at": term.get("reviewed_at"),
                "reviewed_by": term.get("reviewed_by"),
                "matched_chunk_ids": sorted(set(matched_chunk_ids)),
                "stale_chunk_ids": sorted(set(stale_chunk_ids)),
                "ready_chunk_ids": sorted(set(ready_chunk_ids)),
            }
        )

    chunks = sorted(chunk_records.values(), key=lambda item: item["chunk_id"])
    stale_chunks = [item for item in chunks if item["recommended_action"] == "retranslate_chunk"]
    ready_chunks = [item for item in chunks if item["recommended_action"] == "keep_translation"]
    status = "ready"
    if not entries:
        status = "missing_chunks_manifest"
    elif not confirmed_terms:
        status = "no_confirmed_glossary_review"
    elif stale_chunks:
        status = "needs_retranslation"

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "summary": {
            "status": status,
            "confirmed_review_count": len(confirmed_terms),
            "manifest_chunk_count": len(entries),
            "matched_chunk_count": len(chunks),
            "stale_chunk_count": len(stale_chunks),
            "ready_chunk_count": len(ready_chunks),
            "terms_with_stale_chunks": sum(1 for item in term_impacts if item["stale_chunk_ids"]),
            "recommended_action": (
                "retranslate_stale_chunks" if stale_chunks else "no_retranslation_needed"
            ),
        },
        "terms": term_impacts,
        "chunks": chunks,
    }


def _md_cell(value: Any) -> str:
    text = str(value if value is not None else "").replace("\n", " ").strip()
    return text.replace("|", "\\|") or "-"


def glossary_retranslation_plan_to_markdown(plan: dict[str, Any]) -> str:
    summary = plan.get("summary") if isinstance(plan.get("summary"), dict) else {}
    lines = [
        "# 术语确认重译计划",
        "",
        "## 摘要",
        "",
        f"- 状态：{summary.get('status') or '-'}",
        f"- 已确认术语：{summary.get('confirmed_review_count', 0)}",
        f"- 命中分块：{summary.get('matched_chunk_count', 0)}",
        f"- 建议重译分块：{summary.get('stale_chunk_count', 0)}",
        f"- 无需重译分块：{summary.get('ready_chunk_count', 0)}",
        f"- 建议动作：{summary.get('recommended_action') or '-'}",
        "",
        "## 术语影响",
        "",
        "| 术语 | 确认译名 | 旧译名线索 | 作用域 | 命中 chunk | 建议重译 chunk |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for term in plan.get("terms") or []:
        if not isinstance(term, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(term.get("en")),
                    _md_cell(term.get("confirmed_zh")),
                    _md_cell(" / ".join(term.get("previous_zh") or [])),
                    _md_cell(term.get("section_scope")),
                    _md_cell(", ".join(term.get("matched_chunk_ids") or [])),
                    _md_cell(", ".join(term.get("stale_chunk_ids") or [])),
                ]
            )
            + " |"
        )
    if not plan.get("terms"):
        lines.append("| - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## 分块重译建议",
            "",
            "| chunk | 页码 | 章节 | 命中术语 | 风险原因 | 建议 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for chunk in plan.get("chunks") or []:
        if not isinstance(chunk, dict):
            continue
        term_labels = [
            f"{item.get('en')}->{item.get('confirmed_zh')}"
            for item in chunk.get("matched_terms") or []
            if isinstance(item, dict)
        ]
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(chunk.get("chunk_id")),
                    _md_cell(", ".join(str(item) for item in chunk.get("pages_1based") or [])),
                    _md_cell(" / ".join(chunk.get("section_scopes") or [])),
                    _md_cell(", ".join(term_labels)),
                    _md_cell(", ".join(chunk.get("stale_reasons") or [])),
                    _md_cell(chunk.get("recommended_action")),
                ]
            )
            + " |"
        )
    if not plan.get("chunks"):
        lines.append("| - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## 使用说明",
            "",
            "- `retranslate_chunk` 表示该分块命中已确认术语，且译文缺少确认译名、仍含旧译名，或缺少分块译文。",
            "- 该报告只生成重译计划，不会自动覆盖当前译文；自动重译执行器会生成候选重译稿与执行报告。",
        ]
    )
    return "\n".join(lines) + "\n"


def _load_structure_hints_by_chunk(output_dir: Path) -> dict[str, str]:
    raw = _read_json_dict(output_dir / "structure_hints_manifest.json", {"chunks": []})
    out: dict[str, str] = {}
    for item in raw.get("chunks") or []:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id") or "").strip()
        if chunk_id:
            out[chunk_id] = str(item.get("hint_text") or "")
    return out


def _entry_pages_0based(entry: dict[str, Any]) -> list[int]:
    pages = _entry_pages(entry)
    if len(pages) >= 2:
        start = max(1, pages[0])
        end = max(start, pages[-1])
        return list(range(start - 1, end))
    if len(pages) == 1:
        return [max(0, pages[0] - 1)]
    return [0]


def _text_chunk_from_entry(entry: dict[str, Any], source_text: str) -> TextChunk:
    chunk = TextChunk(
        chunk_id=str(entry.get("chunk_id") or "").strip(),
        pages_0based=_entry_pages_0based(entry),
        text=source_text,
        link_count=int(entry.get("link_count") or 0),
        image_count=int(entry.get("image_count") or 0),
    )
    chunk.block_ids = _entry_block_ids(entry)  # type: ignore[attr-defined]
    chunk.block_types = entry.get("block_types") if isinstance(entry.get("block_types"), dict) else {}  # type: ignore[attr-defined]
    return chunk


def _chunk_body_and_meta(
    chunk: TextChunk,
    zh: str,
    translator_name: str,
    *,
    extra_meta: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    p0 = chunk.pages_0based[0] + 1
    p1 = chunk.pages_0based[-1] + 1
    meta: dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "pages_1based": [p0, p1],
        "link_count": chunk.link_count,
        "image_count": chunk.image_count,
        "translator": translator_name,
        "prompt_version": SYSTEM_PROMPT_VERSION,
        "prompt_fingerprint": prompt_fingerprint(),
        "glossary_retranslation": True,
    }
    if extra_meta:
        meta.update(extra_meta)
    body = f"---\n{json.dumps(meta, ensure_ascii=False, indent=2)}\n---\n\n{zh}\n"
    return body, meta


def _merge_candidate_full(
    output_dir: Path,
    entries: list[dict[str, Any]],
    retranslated_dir: Path,
    target: Path,
) -> dict[str, Any]:
    base_dir = output_dir / "chunks"
    parts: list[str] = []
    used_retranslated: list[str] = []
    used_base: list[str] = []
    missing: list[str] = []
    for entry in entries:
        chunk_id = str(entry.get("chunk_id") or "").strip()
        if not chunk_id:
            continue
        candidate = retranslated_dir / f"{chunk_id}.md"
        base = base_dir / f"{chunk_id}.md"
        source = candidate if candidate.is_file() else base
        if not source.is_file():
            missing.append(chunk_id)
            continue
        body = strip_yaml_front_matter(source.read_text(encoding="utf-8")).strip()
        if body:
            parts.append(body)
        if source == candidate:
            used_retranslated.append(chunk_id)
        else:
            used_base.append(chunk_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    merged = finalize_merged_translation_markdown("\n\n".join(parts))
    target.write_text(merged, encoding="utf-8")
    return {
        "target_path": "output/glossary_retranslated_full.md",
        "used_retranslated_chunk_ids": used_retranslated,
        "used_base_chunk_ids": used_base,
        "missing_chunk_ids": missing,
    }


def glossary_retranslation_execution_to_markdown(result: dict[str, Any]) -> str:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
    lines = [
        "# 术语确认重译执行报告",
        "",
        "## 摘要",
        "",
        f"- 状态：{summary.get('status') or '-'}",
        f"- 后端：{summary.get('backend') or '-'}",
        f"- 翻译器：{summary.get('translator') or '-'}",
        f"- 请求分块：{summary.get('requested_chunk_count', 0)}",
        f"- 已重译分块：{summary.get('executed_chunk_count', 0)}",
        f"- 失败分块：{summary.get('failed_chunk_count', 0)}",
        f"- 候选全文：{artifacts.get('retranslated_full_path') or '-'}",
        "",
        "## 分块结果",
        "",
        "| chunk | 页码 | 状态 | 命中术语 | 确认译名覆盖 | 输出 | 错误 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for chunk in result.get("chunks") or []:
        if not isinstance(chunk, dict):
            continue
        terms = [
            f"{item.get('en')}->{item.get('confirmed_zh')}"
            for item in chunk.get("matched_terms") or []
            if isinstance(item, dict)
        ]
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(chunk.get("chunk_id")),
                    _md_cell(", ".join(str(item) for item in chunk.get("pages_1based") or [])),
                    _md_cell(chunk.get("status")),
                    _md_cell(", ".join(terms)),
                    _md_cell(f"{chunk.get('confirmed_term_present_count', 0)}/{len(terms)}"),
                    _md_cell(chunk.get("retranslated_path")),
                    _md_cell(chunk.get("error")),
                ]
            )
            + " |"
        )
    if not result.get("chunks"):
        lines.append("| - | - | - | - | - | - | - |")
    lines.extend(
        [
            "",
            "## 使用说明",
            "",
            "- 本执行器默认生成候选重译稿，不覆盖原始 `output/chunks/` 和 `translated_full.md`。",
            "- 候选全文由重译 chunk 与原始 chunk 按原 manifest 顺序合并，可用于人工验收、对比实验和后续显式发布。",
        ]
    )
    return "\n".join(lines) + "\n"


def execute_glossary_retranslation(
    output_dir: Path,
    memory_dir: Path,
    translator: Translator,
    *,
    backend: str = "",
    mode: str = "stale_only",
    chunk_ids: list[str] | None = None,
    max_chunks: int | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Retranslate chunks affected by confirmed glossary decisions into safe candidate artifacts."""
    output_dir = output_dir.resolve()
    memory_dir = memory_dir.resolve()
    if mode not in {"stale_only", "selected"}:
        raise ValueError("mode must be 'stale_only' or 'selected'")
    if mode == "selected" and not chunk_ids:
        raise ValueError("selected mode requires chunk_ids")

    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = write_glossary_retranslation_plan(output_dir, memory_dir)
    entries = _read_json_list(output_dir / "chunks_manifest.json")
    entries_by_id = {
        str(entry.get("chunk_id") or "").strip(): entry
        for entry in entries
        if str(entry.get("chunk_id") or "").strip()
    }
    plan_chunks = [
        item for item in plan.get("chunks") or [] if isinstance(item, dict) and str(item.get("chunk_id") or "")
    ]
    plan_by_id = {str(item.get("chunk_id")): item for item in plan_chunks}
    requested_ids = [
        str(item.get("chunk_id"))
        for item in plan_chunks
        if item.get("recommended_action") == "retranslate_chunk"
    ]
    if mode == "selected":
        requested_ids = [
            str(item).strip()
            for item in chunk_ids or []
            if str(item or "").strip()
        ]
    requested_ids = list(dict.fromkeys(requested_ids))
    skipped_by_limit: list[str] = []
    if max_chunks is not None and max_chunks > 0 and len(requested_ids) > max_chunks:
        skipped_by_limit = requested_ids[max_chunks:]
        requested_ids = requested_ids[:max_chunks]

    mem = MemoryStore(memory_dir)
    style_text = ""
    if mem.style_path.is_file():
        style_text = mem.style_path.read_text(encoding="utf-8").strip()
    hints_by_chunk = _load_structure_hints_by_chunk(output_dir)
    retranslated_dir = output_dir / "glossary_retranslated_chunks"
    retranslated_dir.mkdir(parents=True, exist_ok=True)

    chunk_results: list[dict[str, Any]] = []
    for chunk_id in requested_ids:
        entry = entries_by_id.get(chunk_id)
        plan_chunk = plan_by_id.get(chunk_id, {})
        if not entry:
            chunk_results.append(
                {
                    "chunk_id": chunk_id,
                    "status": "failed",
                    "error": "chunk_not_found_in_manifest",
                    "matched_terms": plan_chunk.get("matched_terms") or [],
                }
            )
            continue
        source_text, source_text_path = _entry_source_text(entry, output_dir)
        if not source_text.strip():
            chunk_results.append(
                {
                    "chunk_id": chunk_id,
                    "pages_1based": _entry_pages(entry),
                    "status": "failed",
                    "error": "source_text_missing",
                    "source_text_path": source_text_path,
                    "matched_terms": plan_chunk.get("matched_terms") or [],
                }
            )
            continue
        chunk = _text_chunk_from_entry(entry, source_text)
        pages = _entry_pages(entry)
        start_page = pages[0] if pages else chunk.pages_0based[0] + 1
        end_page = pages[-1] if pages else chunk.pages_0based[-1] + 1
        glossary_excerpt = mem.glossary_snippet_for_pages(
            start_page,
            end_page,
            section_scope=_entry_section_scopes(entry),
            structure_types=_entry_structure_types(entry),
            block_ids=_entry_block_ids(entry),
        )
        request = TranslationRequest(
            source_text=source_text,
            glossary_excerpt=glossary_excerpt,
            prior_summaries="",
            style_notes=style_text,
            structure_hints=hints_by_chunk.get(chunk_id, ""),
        )
        started = time.perf_counter()
        try:
            translated = translator.translate(request).strip()
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            matched_terms = [
                item for item in plan_chunk.get("matched_terms") or [] if isinstance(item, dict)
            ]
            confirmed_present_count = sum(
                1
                for item in matched_terms
                if _contains_translation(translated, str(item.get("confirmed_zh") or ""))
            )
            body, meta = _chunk_body_and_meta(
                chunk,
                translated,
                getattr(translator, "name", "translator"),
                extra_meta={
                    "source_text_path": source_text_path,
                    "base_translation_path": f"output/chunks/{chunk_id}.md",
                    "matched_terms": matched_terms,
                    "retranslation_mode": mode,
                },
            )
            out_path = retranslated_dir / f"{chunk_id}.md"
            out_path.write_text(body, encoding="utf-8")
            chunk_results.append(
                {
                    "chunk_id": chunk_id,
                    "pages_1based": _entry_pages(entry),
                    "status": "executed",
                    "source_text_path": source_text_path,
                    "base_translation_path": f"output/chunks/{chunk_id}.md",
                    "retranslated_path": f"output/glossary_retranslated_chunks/{chunk_id}.md",
                    "matched_terms": matched_terms,
                    "confirmed_term_present_count": confirmed_present_count,
                    "source_char_count": len(source_text),
                    "translated_char_count": len(translated),
                    "elapsed_ms": elapsed_ms,
                    "metadata": meta,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive around external translators
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            chunk_results.append(
                {
                    "chunk_id": chunk_id,
                    "pages_1based": _entry_pages(entry),
                    "status": "failed",
                    "source_text_path": source_text_path,
                    "matched_terms": plan_chunk.get("matched_terms") or [],
                    "elapsed_ms": elapsed_ms,
                    "error": str(exc),
                }
            )

    for chunk_id in skipped_by_limit:
        chunk_results.append(
            {
                "chunk_id": chunk_id,
                "status": "skipped",
                "error": "max_chunks_limit",
                "matched_terms": (plan_by_id.get(chunk_id) or {}).get("matched_terms") or [],
            }
        )

    executed_count = sum(1 for item in chunk_results if item.get("status") == "executed")
    failed_count = sum(1 for item in chunk_results if item.get("status") == "failed")
    skipped_count = sum(1 for item in chunk_results if item.get("status") == "skipped")
    if not requested_ids:
        status = "nothing_to_retranslate"
    elif failed_count and executed_count:
        status = "partial_failed"
    elif failed_count:
        status = "failed"
    else:
        status = "executed"

    merge_info = _merge_candidate_full(
        output_dir,
        entries,
        retranslated_dir,
        output_dir / "glossary_retranslated_full.md",
    )
    result = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "generated_at": generated_at,
        "summary": {
            "status": status,
            "backend": backend,
            "translator": getattr(translator, "name", "translator"),
            "mode": mode,
            "requested_chunk_count": len(requested_ids),
            "executed_chunk_count": executed_count,
            "failed_chunk_count": failed_count,
            "skipped_chunk_count": skipped_count,
            "plan_status": (plan.get("summary") or {}).get("status") if isinstance(plan.get("summary"), dict) else "",
        },
        "artifacts": {
            "plan_json_path": "output/glossary_retranslation_plan.json",
            "plan_md_path": "output/glossary_retranslation_plan.md",
            "retranslated_chunk_dir": "output/glossary_retranslated_chunks",
            "retranslated_full_path": merge_info["target_path"],
            "result_json_path": "output/glossary_retranslation_result.json",
            "result_md_path": "output/glossary_retranslation_result.md",
            "merge": merge_info,
        },
        "requested_chunk_ids": requested_ids,
        "chunks": chunk_results,
    }
    (output_dir / "glossary_retranslation_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "glossary_retranslation_result.md").write_text(
        glossary_retranslation_execution_to_markdown(result), encoding="utf-8"
    )
    return result


def write_glossary_retranslation_plan(
    output_dir: Path,
    memory_dir: Path,
    json_path: Path | None = None,
    md_path: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_path or output_dir / "glossary_retranslation_plan.json"
    md_path = md_path or output_dir / "glossary_retranslation_plan.md"
    plan = build_glossary_retranslation_plan(output_dir, memory_dir)
    json_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(glossary_retranslation_plan_to_markdown(plan), encoding="utf-8")
    return plan


def _file_sha256(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_text_atomic(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f".{target_path.name}.tmp")
    temp_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    temp_path.replace(target_path)


def build_glossary_retranslation_publish(
    execution_result: dict[str, Any],
    *,
    confirm: bool = False,
    candidate_full_path: Path | None = None,
    published_full_path: Path | None = None,
    original_full_path: Path | None = None,
) -> dict[str, Any]:
    """Create an explicit publication copy from the glossary retranslation candidate."""
    summary = (
        execution_result.get("summary")
        if isinstance(execution_result.get("summary"), dict)
        else {}
    )
    artifacts = (
        execution_result.get("artifacts")
        if isinstance(execution_result.get("artifacts"), dict)
        else {}
    )
    candidate_path_text = str(artifacts.get("retranslated_full_path") or "")
    candidate_path = candidate_full_path or (Path(candidate_path_text) if candidate_path_text else None)
    target_path = published_full_path
    original_path = original_full_path

    requested_count = int(summary.get("requested_chunk_count") or 0)
    executed_count = int(summary.get("executed_chunk_count") or 0)
    failed_count = int(summary.get("failed_chunk_count") or 0)
    skipped_count = int(summary.get("skipped_chunk_count") or 0)
    open_issue_count = failed_count + skipped_count
    warnings: list[str] = []
    if failed_count:
        warnings.append(f"{failed_count} 个术语重译分块执行失败。")
    if skipped_count:
        warnings.append(f"{skipped_count} 个术语重译分块被跳过。")
    if requested_count and executed_count < requested_count:
        warnings.append("候选全文中仍包含部分原始译文分块。")

    status = "pending_confirmation"
    reason = "需要显式人工确认后才生成术语重译发布副本。"
    published = False
    if confirm:
        if candidate_path is None or not candidate_path.is_file():
            status = "blocked_missing_candidate_full"
            reason = "未找到可发布的术语候选重译全文。"
        elif target_path is None:
            status = "blocked_missing_publish_target"
            reason = "未提供术语重译发布副本输出路径。"
        elif executed_count <= 0:
            status = "blocked_no_retranslated_chunks"
            reason = "没有已执行的术语重译分块，未生成发布副本。"
        else:
            _copy_text_atomic(candidate_path, target_path)
            published = True
            status = "published_with_warnings" if open_issue_count else "published"
            reason = "已生成人工确认后的术语重译发布副本。"

    candidate_hash = _file_sha256(candidate_path)
    published_hash = _file_sha256(target_path)
    original_hash = _file_sha256(original_path)
    return {
        "schema_version": PUBLISH_SCHEMA_VERSION,
        "summary": {
            "execution_schema_version": execution_result.get("schema_version"),
            "confirmed": bool(confirm),
            "published": published,
            "publish_status": status,
            "reason": reason,
            "requested_chunk_count": requested_count,
            "executed_chunk_count": executed_count,
            "failed_chunk_count": failed_count,
            "skipped_chunk_count": skipped_count,
            "open_issue_count": open_issue_count,
            "candidate_full_path": candidate_path.as_posix() if candidate_path else "",
            "published_full_path": target_path.as_posix() if target_path else "",
            "original_full_path": original_path.as_posix() if original_path else "",
            "candidate_sha256": candidate_hash,
            "published_sha256": published_hash,
            "original_sha256": original_hash,
            "published_matches_candidate": bool(
                published_hash and candidate_hash and published_hash == candidate_hash
            ),
            "rollback_available": bool(original_path and original_path.is_file()),
            "warnings": warnings,
        },
        "source": {
            "execution_summary": summary,
            "execution_artifacts": artifacts,
            "requested_chunk_ids": execution_result.get("requested_chunk_ids") or [],
        },
    }


def glossary_retranslation_publish_to_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    warnings = summary.get("warnings") if isinstance(summary.get("warnings"), list) else []
    lines = [
        "# 术语候选重译发布确认",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| 状态 | `{summary.get('publish_status') or '-'}` |",
        f"| 已请求发布确认 | {summary.get('confirmed', False)} |",
        f"| 已发布副本 | {summary.get('published', False)} |",
        f"| 请求重译分块 | {summary.get('requested_chunk_count', 0)} |",
        f"| 已重译分块 | {summary.get('executed_chunk_count', 0)} |",
        f"| 失败分块 | {summary.get('failed_chunk_count', 0)} |",
        f"| 跳过分块 | {summary.get('skipped_chunk_count', 0)} |",
        f"| 候选全文 | `{summary.get('candidate_full_path') or '-'}` |",
        f"| 发布副本 | `{summary.get('published_full_path') or '-'}` |",
        f"| 原始译文 | `{summary.get('original_full_path') or '-'}` |",
        f"| 发布副本匹配候选全文 | {summary.get('published_matches_candidate', False)} |",
        f"| 可回滚 | {summary.get('rollback_available', False)} |",
        "",
        summary.get("reason") or "",
    ]
    if warnings:
        lines.extend(["", "## 警告", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines).rstrip() + "\n"


def write_glossary_retranslation_publish(
    execution_result: dict[str, Any],
    json_path: Path,
    markdown_path: Path,
    *,
    confirm: bool = False,
    candidate_full_path: Path | None = None,
    published_full_path: Path | None = None,
    original_full_path: Path | None = None,
) -> dict[str, Any]:
    report = build_glossary_retranslation_publish(
        execution_result,
        confirm=confirm,
        candidate_full_path=candidate_full_path,
        published_full_path=published_full_path or json_path.parent / "glossary_retranslation_published_full.md",
        original_full_path=original_full_path,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(glossary_retranslation_publish_to_markdown(report), encoding="utf-8")
    return report


def build_glossary_retranslation_rollback(
    publish_report: dict[str, Any],
    *,
    confirm: bool = False,
    original_full_path: Path | None = None,
    published_full_path: Path | None = None,
    rollback_full_path: Path | None = None,
) -> dict[str, Any]:
    """Create an auditable rollback drill for glossary retranslation publication."""
    summary = (
        publish_report.get("summary")
        if isinstance(publish_report.get("summary"), dict)
        else {}
    )
    original_path_text = str(summary.get("original_full_path") or "")
    published_path_text = str(summary.get("published_full_path") or "")
    original_path = original_full_path or (Path(original_path_text) if original_path_text else None)
    published_path = published_full_path or (Path(published_path_text) if published_path_text else None)
    target_path = rollback_full_path

    published = bool(summary.get("published"))
    original_exists = bool(original_path and original_path.is_file())
    published_exists = bool(published_path and published_path.is_file())
    rollback_available = original_exists and published_exists and published
    warnings: list[str] = []
    if not published:
        warnings.append("术语重译发布稿尚未生成，暂不能演练回滚。")
    if not original_exists:
        warnings.append("原始译文缺失，无法作为回滚基线。")
    if not published_exists:
        warnings.append("术语重译发布稿缺失，无法验证回滚目标。")

    status = "pending_confirmation" if rollback_available else "not_ready"
    reason = "需要显式确认后才生成术语重译回滚演练副本。"
    rollback_applied = False
    published_hash_before = _file_sha256(published_path)
    if confirm:
        if not published:
            status = "blocked_unpublished"
            reason = "术语重译发布稿尚未生成，未执行回滚演练。"
        elif not original_exists:
            status = "blocked_missing_original"
            reason = "原始译文缺失，未执行回滚演练。"
        elif not published_exists:
            status = "blocked_missing_published"
            reason = "术语重译发布稿缺失，未执行回滚演练。"
        elif target_path is None:
            status = "blocked_missing_rollback_target"
            reason = "未提供术语重译回滚演练副本输出路径。"
        else:
            _copy_text_atomic(original_path, target_path)
            rollback_applied = True
            status = "rolled_back"
            reason = "已生成术语重译回滚演练副本；发布稿未被覆盖。"
    elif not rollback_available:
        reason = "当前缺少发布稿或原始译文基线，暂不能演练回滚。"

    original_hash = _file_sha256(original_path)
    published_hash = _file_sha256(published_path)
    rollback_hash = _file_sha256(target_path) if rollback_applied else ""
    return {
        "schema_version": ROLLBACK_SCHEMA_VERSION,
        "summary": {
            "publish_schema_version": publish_report.get("schema_version"),
            "confirmed": bool(confirm),
            "rollback_available": rollback_available,
            "rollback_applied": rollback_applied,
            "rollback_status": status,
            "reason": reason,
            "publish_status": summary.get("publish_status") or "",
            "published": published,
            "original_full_path": original_path.as_posix() if original_path else "",
            "published_full_path": published_path.as_posix() if published_path else "",
            "rollback_full_path": target_path.as_posix() if target_path else "",
            "original_sha256": original_hash,
            "published_sha256": published_hash,
            "rollback_sha256": rollback_hash,
            "rollback_matches_original": bool(rollback_hash and original_hash and rollback_hash == original_hash),
            "published_preserved": bool(
                published_hash_before and published_hash and published_hash_before == published_hash
            ),
            "warnings": warnings,
        },
        "source": {
            "publish_summary": summary,
        },
    }


def glossary_retranslation_rollback_to_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    warnings = summary.get("warnings") if isinstance(summary.get("warnings"), list) else []
    lines = [
        "# 术语候选重译回滚演练",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| 状态 | `{summary.get('rollback_status') or '-'}` |",
        f"| 已请求回滚演练 | {summary.get('confirmed', False)} |",
        f"| 可回滚 | {summary.get('rollback_available', False)} |",
        f"| 已生成回滚副本 | {summary.get('rollback_applied', False)} |",
        f"| 发布状态 | `{summary.get('publish_status') or '-'}` |",
        f"| 原始译文 | `{summary.get('original_full_path') or '-'}` |",
        f"| 发布稿 | `{summary.get('published_full_path') or '-'}` |",
        f"| 回滚演练副本 | `{summary.get('rollback_full_path') or '-'}` |",
        f"| 回滚副本匹配原始译文 | {summary.get('rollback_matches_original', False)} |",
        f"| 发布稿保持不变 | {summary.get('published_preserved', False)} |",
        "",
        summary.get("reason") or "",
    ]
    if warnings:
        lines.extend(["", "## 警告", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines).rstrip() + "\n"


def write_glossary_retranslation_rollback(
    publish_report: dict[str, Any],
    json_path: Path,
    markdown_path: Path,
    *,
    confirm: bool = False,
    original_full_path: Path | None = None,
    published_full_path: Path | None = None,
    rollback_full_path: Path | None = None,
) -> dict[str, Any]:
    report = build_glossary_retranslation_rollback(
        publish_report,
        confirm=confirm,
        original_full_path=original_full_path,
        published_full_path=published_full_path,
        rollback_full_path=rollback_full_path or json_path.parent / "glossary_retranslation_rollback_full.md",
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(glossary_retranslation_rollback_to_markdown(report), encoding="utf-8")
    return report
