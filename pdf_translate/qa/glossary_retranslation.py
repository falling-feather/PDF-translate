from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pdf_translate.memory_store import _pending_dedupe_key, _term_scope_matches

SCHEMA_VERSION = "glossary-retranslation-plan-v1"


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
            source_text = str(entry.get("text") or "")
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
            "- 该报告只生成重译计划，不会自动覆盖当前译文；后续自动重译可直接消费 `chunks[].chunk_id`。",
        ]
    )
    return "\n".join(lines) + "\n"


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
