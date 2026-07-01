from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "ocr-candidate-promotion-v1"
PROMOTABLE_STATUS = "candidate"


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0


def _text(value: Any) -> str:
    return str(value or "").strip()


def _pages(document_ir: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for page in document_ir.get("pages") or []:
        if not isinstance(page, dict):
            continue
        page_no = _as_int(page.get("page_no"))
        if page_no > 0:
            out[page_no] = page
    return out


def _blocks(document_ir: dict[str, Any]) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    out: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for page in document_ir.get("pages") or []:
        if not isinstance(page, dict):
            continue
        for block in page.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            block_id = str(block.get("block_id") or "")
            if block_id:
                out[block_id] = (page, block)
    return out


def _candidate_list(target: dict[str, Any]) -> list[dict[str, Any]]:
    meta = target.get("meta") if isinstance(target.get("meta"), dict) else {}
    candidates = meta.get("ocr_candidates") if isinstance(meta, dict) else []
    if not isinstance(candidates, list):
        return []
    return [item for item in candidates if isinstance(item, dict)]


def _find_source_candidate(
    document_ir: dict[str, Any],
    qa_item: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    page_index = _pages(document_ir)
    block_index = _blocks(document_ir)
    block_id = str(qa_item.get("block_id") or "")
    if block_id:
        found = block_index.get(block_id)
        if not found:
            return "block", {}, None
        _page, block = found
        target = block
        target_kind = "block"
    else:
        target = page_index.get(_as_int(qa_item.get("page_no")), {})
        target_kind = "page"

    candidates = _candidate_list(target)
    target_index = _as_int(qa_item.get("target_index"))
    if 0 <= target_index < len(candidates):
        candidate = candidates[target_index]
        task_matches = str(candidate.get("task_id") or "") == str(qa_item.get("task_id") or "")
        text_matches = _text(candidate.get("text")) == _text(qa_item.get("preview"))
        if task_matches or text_matches:
            return target_kind, target, candidate

    task_id = str(qa_item.get("task_id") or "")
    for candidate in candidates:
        if str(candidate.get("task_id") or "") == task_id:
            return target_kind, target, candidate
    return target_kind, target, None


def _promotion_meta(qa_item: dict[str, Any], source_candidate: dict[str, Any]) -> dict[str, Any]:
    text = _text(source_candidate.get("text") or qa_item.get("preview"))
    meta: dict[str, Any] = {
        "source": "ocr_candidate_promotion",
        "qa_schema_version": "ocr-candidate-qa-v1",
        "task_id": str(qa_item.get("task_id") or ""),
        "candidate_status": str(qa_item.get("status") or ""),
        "candidate_target_index": _as_int(qa_item.get("target_index")),
        "confidence": source_candidate.get("confidence", qa_item.get("confidence")),
        "engine": str(source_candidate.get("engine") or qa_item.get("engine") or ""),
        "language": str(source_candidate.get("language") or qa_item.get("language") or ""),
        "input_path": str(source_candidate.get("input_path") or qa_item.get("input_path") or ""),
        "text_char_count": len(text),
    }
    target_structure_type = str(
        source_candidate.get("target_structure_type") or qa_item.get("target_structure_type") or ""
    )
    if target_structure_type:
        meta["target_structure_type"] = target_structure_type
    for key in ("table_context", "subtarget", "structure_contract"):
        value = source_candidate.get(key)
        if not isinstance(value, dict):
            value = qa_item.get(key)
        if isinstance(value, dict):
            meta[key] = _json_copy(value)
    return meta


def _attach_structure_trace(record: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    for key in ("target_structure_type", "table_context", "subtarget"):
        value = meta.get(key)
        if isinstance(value, dict):
            record[key] = _json_copy(value)
        elif isinstance(value, str) and value:
            record[key] = value
    return record


def _append_text(existing: Any, text: str) -> str:
    current = str(existing or "").strip()
    if not current:
        return text
    if text in current:
        return current
    return current + "\n\n" + text


def _add_page_text(page: dict[str, Any], text: str) -> None:
    page["text"] = _append_text(page.get("text"), text)


def _record_promotion(target: dict[str, Any], meta: dict[str, Any]) -> None:
    target_meta = target.setdefault("meta", {})
    if not isinstance(target_meta, dict):
        target_meta = {}
        target["meta"] = target_meta
    promotions = target_meta.setdefault("ocr_promotions", [])
    if not isinstance(promotions, list):
        promotions = []
        target_meta["ocr_promotions"] = promotions
    promotions.append(meta)


def _bbox_for_synthetic_block(page: dict[str, Any], candidate: dict[str, Any]) -> list[float]:
    bbox = candidate.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        out: list[float] = []
        for item in bbox[:4]:
            if isinstance(item, bool):
                break
            if isinstance(item, (int, float)):
                out.append(float(item))
            elif isinstance(item, str):
                try:
                    out.append(float(item))
                except ValueError:
                    break
        if len(out) == 4:
            return out
    width = page.get("width") if isinstance(page.get("width"), (int, float)) else 0
    height = page.get("height") if isinstance(page.get("height"), (int, float)) else 0
    return [0.0, 0.0, float(width), float(height)]


def _next_synthetic_block_id(page: dict[str, Any]) -> str:
    page_no = _as_int(page.get("page_no"))
    blocks = page.get("blocks") if isinstance(page.get("blocks"), list) else []
    existing = {
        str(block.get("block_id") or "")
        for block in blocks
        if isinstance(block, dict) and str(block.get("block_id") or "")
    }
    idx = 0
    while True:
        block_id = f"p{page_no}-ocr{idx:04d}"
        if block_id not in existing:
            return block_id
        idx += 1


def _max_order(page: dict[str, Any]) -> int:
    blocks = page.get("blocks") if isinstance(page.get("blocks"), list) else []
    orders = [_as_int(block.get("order")) for block in blocks if isinstance(block, dict)]
    return max(orders) if orders else -1


def _increment_block_type_count(page: dict[str, Any], block_type: str) -> None:
    meta = page.setdefault("meta", {})
    if not isinstance(meta, dict):
        meta = {}
        page["meta"] = meta
    counts = meta.setdefault("block_type_counts", {})
    if not isinstance(counts, dict):
        counts = {}
        meta["block_type_counts"] = counts
    counts[block_type] = _as_int(counts.get(block_type)) + 1


def _promote_to_block(
    page: dict[str, Any],
    block: dict[str, Any],
    qa_item: dict[str, Any],
    source_candidate: dict[str, Any],
) -> dict[str, Any]:
    text = _text(source_candidate.get("text") or qa_item.get("preview"))
    meta = _promotion_meta(qa_item, source_candidate)
    block["text"] = _append_text(block.get("text"), text)
    _add_page_text(page, text)
    _record_promotion(block, meta)
    return _attach_structure_trace(
        {
            "task_id": meta["task_id"],
            "page_no": _as_int(qa_item.get("page_no")),
            "block_id": str(block.get("block_id") or ""),
            "source_target": str(qa_item.get("target") or ""),
            "promotion_target": "document_ir.block.text",
            "action": "set_or_append_block_text",
            "block_type": str(block.get("type") or ""),
            "text_char_count": meta["text_char_count"],
            "confidence": meta["confidence"],
            "engine": meta["engine"],
        },
        meta,
    )


def _promote_to_page_block(
    page: dict[str, Any],
    qa_item: dict[str, Any],
    source_candidate: dict[str, Any],
) -> dict[str, Any]:
    text = _text(source_candidate.get("text") or qa_item.get("preview"))
    meta = _promotion_meta(qa_item, source_candidate)
    block_id = _next_synthetic_block_id(page)
    block = {
        "block_id": block_id,
        "page_no": _as_int(page.get("page_no")),
        "type": "paragraph",
        "text": text,
        "bbox": _bbox_for_synthetic_block(page, source_candidate),
        "order": _max_order(page) + 1,
        "parent_id": None,
        "locked_tokens": [],
        "meta": {
            "synthetic_source": "ocr_candidate_promotion",
            "ocr_promotions": [meta],
        },
    }
    blocks = page.setdefault("blocks", [])
    if not isinstance(blocks, list):
        blocks = []
        page["blocks"] = blocks
    blocks.append(block)
    _add_page_text(page, text)
    _increment_block_type_count(page, "paragraph")
    return _attach_structure_trace(
        {
            "task_id": meta["task_id"],
            "page_no": _as_int(page.get("page_no")),
            "block_id": block_id,
            "source_target": str(qa_item.get("target") or ""),
            "promotion_target": "document_ir.page.blocks.synthetic",
            "action": "create_synthetic_ocr_block",
            "block_type": "paragraph",
            "text_char_count": meta["text_char_count"],
            "confidence": meta["confidence"],
            "engine": meta["engine"],
        },
        meta,
    )


def build_ocr_candidate_promotion(
    document_ir_ocr: dict[str, Any] | None,
    ocr_candidate_qa: dict[str, Any] | None,
) -> dict[str, Any]:
    promoted_ir = _json_copy(document_ir_ocr or {})
    qa_items = (
        ocr_candidate_qa.get("candidates")
        if isinstance(ocr_candidate_qa, dict) and isinstance(ocr_candidate_qa.get("candidates"), list)
        else []
    )
    promotions: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    skip_reason_counts: Counter[str] = Counter()
    block_type_counts: Counter[str] = Counter()

    page_index = _pages(promoted_ir)
    block_index = _blocks(promoted_ir)
    for raw_item in qa_items:
        if not isinstance(raw_item, dict):
            continue
        status = str(raw_item.get("status") or "unknown")
        status_counts[status] += 1
        block_type_counts[str(raw_item.get("block_type") or "unknown")] += 1
        base = {
            "task_id": str(raw_item.get("task_id") or ""),
            "page_no": _as_int(raw_item.get("page_no")),
            "block_id": str(raw_item.get("block_id") or ""),
            "status": status,
        }
        if status != PROMOTABLE_STATUS:
            skipped.append({**base, "reason": "status_not_promotable"})
            skip_reason_counts["status_not_promotable"] += 1
            continue

        target_kind, target, source_candidate = _find_source_candidate(promoted_ir, raw_item)
        text = _text((source_candidate or {}).get("text") or raw_item.get("preview"))
        if not target:
            skipped.append({**base, "reason": "target_missing"})
            skip_reason_counts["target_missing"] += 1
            continue
        if source_candidate is None:
            skipped.append({**base, "reason": "candidate_missing"})
            skip_reason_counts["candidate_missing"] += 1
            continue
        if not text:
            skipped.append({**base, "reason": "empty_text"})
            skip_reason_counts["empty_text"] += 1
            continue

        if target_kind == "block":
            found = block_index.get(str(target.get("block_id") or ""))
            page = found[0] if found else page_index.get(_as_int(raw_item.get("page_no")), {})
            promotion = _promote_to_block(page, target, raw_item, source_candidate)
        else:
            promotion = _promote_to_page_block(target, raw_item, source_candidate)
        promotions.append(promotion)

    promoted_text_char_count = sum(_as_int(item.get("text_char_count")) for item in promotions)
    return {
        "schema_version": SCHEMA_VERSION,
        "doc_id": str((promoted_ir or {}).get("doc_id") or (ocr_candidate_qa or {}).get("doc_id") or ""),
        "summary": {
            "candidate_count": len([item for item in qa_items if isinstance(item, dict)]),
            "eligible_candidate_count": status_counts.get(PROMOTABLE_STATUS, 0),
            "promoted_candidate_count": len(promotions),
            "skipped_candidate_count": len(skipped),
            "block_promotion_count": sum(1 for item in promotions if item.get("promotion_target") == "document_ir.block.text"),
            "page_promotion_count": sum(
                1 for item in promotions if item.get("promotion_target") == "document_ir.page.blocks.synthetic"
            ),
            "promoted_text_char_count": promoted_text_char_count,
            "candidate_status_counts": dict(status_counts),
            "skip_reason_counts": dict(skip_reason_counts),
            "block_type_counts": dict(block_type_counts),
        },
        "promotion_policy": {
            "eligible_status": PROMOTABLE_STATUS,
            "source": "Only OCR candidates that passed OCR candidate QA may enter the promoted DocumentIR.",
            "original_ir_policy": "Original document_ir.json and document_ir_ocr.json are not mutated.",
        },
        "promotions": promotions,
        "skipped_candidates": skipped,
        "promoted_document_ir": promoted_ir,
    }


def ocr_candidate_promotion_to_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# OCR Candidate Promotion",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Candidate count | {summary.get('candidate_count', 0)} |",
        f"| Eligible candidates | {summary.get('eligible_candidate_count', 0)} |",
        f"| Promoted candidates | {summary.get('promoted_candidate_count', 0)} |",
        f"| Skipped candidates | {summary.get('skipped_candidate_count', 0)} |",
        f"| Promoted text chars | {summary.get('promoted_text_char_count', 0)} |",
        "",
        "## Promotions",
        "",
    ]
    promotions = report.get("promotions") if isinstance(report.get("promotions"), list) else []
    if promotions:
        for item in promotions:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- task `{item.get('task_id')}` page {item.get('page_no')} "
                f"block `{item.get('block_id') or '-'}` -> `{item.get('promotion_target')}`"
            )
    else:
        lines.append("No OCR candidates were promoted.")
    lines.extend(["", "## Skipped Candidates", ""])
    skipped = report.get("skipped_candidates") if isinstance(report.get("skipped_candidates"), list) else []
    if skipped:
        for item in skipped:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- task `{item.get('task_id')}` page {item.get('page_no')} "
                f"block `{item.get('block_id') or '-'}`: {item.get('reason')}"
            )
    else:
        lines.append("No OCR candidates were skipped after eligibility checks.")
    return "\n".join(lines).rstrip() + "\n"


def _artifact_rel(path: Path) -> str:
    if path.parent.name == "output":
        return f"output/{path.name}"
    return path.as_posix()


def write_ocr_candidate_promotion(
    document_ir_ocr: dict[str, Any] | None,
    ocr_candidate_qa: dict[str, Any] | None,
    report_path: Path,
    markdown_path: Path,
    promoted_ir_path: Path,
) -> dict[str, Any]:
    payload = build_ocr_candidate_promotion(document_ir_ocr, ocr_candidate_qa)
    promoted_ir = payload.pop("promoted_document_ir")
    payload["artifacts"] = {"promoted_document_ir": _artifact_rel(promoted_ir_path)}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    promoted_ir_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(ocr_candidate_promotion_to_markdown(payload), encoding="utf-8")
    promoted_ir_path.write_text(json.dumps(promoted_ir, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
