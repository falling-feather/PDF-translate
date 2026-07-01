from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "ocr-candidate-qa-v1"
DEFAULT_REVIEW_CONFIDENCE = 0.75
MIN_USEFUL_CHAR_RATIO = 0.45
MIN_TEXT_CHARS = 3
STRUCTURE_REVIEW_BLOCK_TYPES = {"table", "formula"}
STRUCTURED_RESULT_FIELDS = (
    "structured_cells",
    "cell_bboxes",
    "merged_cell_candidates",
    "table_footnotes",
)


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _structured_payload(value: Any) -> Any | None:
    if isinstance(value, (dict, list)):
        return _json_copy(value)
    return None


def _structured_result_fields(source: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in STRUCTURED_RESULT_FIELDS:
        value = _structured_payload(source.get(key))
        if value is not None:
            out[key] = value
    return out


def _item_count(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        return len(value)
    return 0


def _useful_char_ratio(text: str) -> float:
    visible = [ch for ch in text if not ch.isspace()]
    if not visible:
        return 0.0
    useful = [ch for ch in visible if ch.isalnum() or "\u4e00" <= ch <= "\u9fff"]
    return round(len(useful) / len(visible), 4)


def _text_overlap(candidate: str, source: str) -> bool:
    candidate_norm = _normalized_text(candidate).casefold()
    source_norm = _normalized_text(source).casefold()
    if len(candidate_norm) < 8 or len(source_norm) < 8:
        return False
    return candidate_norm in source_norm or source_norm in candidate_norm


def _iter_page_candidates(page: dict[str, Any]) -> list[dict[str, Any]]:
    meta = page.get("meta") if isinstance(page.get("meta"), dict) else {}
    candidates = meta.get("ocr_candidates") if isinstance(meta, dict) else []
    if not isinstance(candidates, list):
        return []
    out: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        if isinstance(candidate, dict):
            out.append(
                {
                    "candidate": candidate,
                    "target_kind": "page",
                    "target_index": idx,
                    "target_text": str(page.get("text") or ""),
                    "block_type": "page",
                }
            )
    return out


def _iter_block_candidates(page: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    blocks = page.get("blocks") if isinstance(page.get("blocks"), list) else []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        meta = block.get("meta") if isinstance(block.get("meta"), dict) else {}
        candidates = meta.get("ocr_candidates") if isinstance(meta, dict) else []
        if not isinstance(candidates, list):
            continue
        for idx, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            out.append(
                {
                    "candidate": candidate,
                    "target_kind": "block",
                    "target_index": idx,
                    "target_text": str(block.get("text") or ""),
                    "block_type": str(block.get("type") or ""),
                }
            )
    return out


def _iter_candidates(document_ir_ocr: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(document_ir_ocr, dict):
        return []
    out: list[dict[str, Any]] = []
    pages = document_ir_ocr.get("pages") if isinstance(document_ir_ocr.get("pages"), list) else []
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_no = int(page.get("page_no") or 0)
        for item in _iter_page_candidates(page) + _iter_block_candidates(page):
            candidate = item["candidate"]
            record = {
                "task_id": str(candidate.get("task_id") or ""),
                "page_no": int(candidate.get("page_no") or page_no),
                "block_id": str(candidate.get("block_id") or ""),
                "scope": str(candidate.get("scope") or item["target_kind"]),
                "target": f"document_ir.{item['target_kind']}.meta.ocr_candidates",
                "target_index": int(item["target_index"]),
                "block_type": item["block_type"],
                "target_structure_type": str(candidate.get("target_structure_type") or item["block_type"]),
                "text": str(candidate.get("text") or ""),
                "confidence": _as_float(candidate.get("confidence")),
                "engine": str(candidate.get("engine") or ""),
                "language": str(candidate.get("language") or ""),
                "input_path": str(candidate.get("input_path") or ""),
                "warnings": [str(value) for value in candidate.get("warnings") or [] if str(value)],
                "target_text": item["target_text"],
                "table_context": _json_copy(candidate.get("table_context"))
                if isinstance(candidate.get("table_context"), dict)
                else {},
                "subtarget": _json_copy(candidate.get("subtarget"))
                if isinstance(candidate.get("subtarget"), dict)
                else {},
                "structure_contract": _json_copy(candidate.get("structure_contract"))
                if isinstance(candidate.get("structure_contract"), dict)
                else {},
            }
            record.update(_structured_result_fields(candidate))
            out.append(record)
    return out


def _assessment(item: dict[str, Any], *, review_confidence: float) -> dict[str, Any]:
    text = str(item.get("text") or "").strip()
    confidence = _as_float(item.get("confidence"))
    useful_ratio = _useful_char_ratio(text)
    block_type = str(item.get("block_type") or "")
    table_context = item.get("table_context") if isinstance(item.get("table_context"), dict) else {}
    subtarget = item.get("subtarget") if isinstance(item.get("subtarget"), dict) else {}
    structure_contract = (
        item.get("structure_contract") if isinstance(item.get("structure_contract"), dict) else {}
    )
    structured_result_fields = _structured_result_fields(item)
    reasons: list[str] = []
    blockers: list[str] = []

    if len(text) < MIN_TEXT_CHARS:
        blockers.append("too_short")
    if useful_ratio < MIN_USEFUL_CHAR_RATIO:
        blockers.append("low_useful_char_ratio")
    if _text_overlap(text, str(item.get("target_text") or "")):
        reasons.append("duplicate_source_text")
    if confidence < review_confidence:
        reasons.append("needs_confidence_review")
    if block_type in STRUCTURE_REVIEW_BLOCK_TYPES:
        reasons.append(f"needs_{block_type}_structure_review")
    if item.get("warnings"):
        reasons.append("engine_warnings_present")

    if blockers:
        status = "blocked"
    elif reasons:
        status = "needs_review"
    else:
        status = "candidate"

    assessment: dict[str, Any] = {
        "task_id": item["task_id"],
        "page_no": item["page_no"],
        "block_id": item["block_id"],
        "scope": item["scope"],
        "target": item["target"],
        "target_index": item["target_index"],
        "block_type": block_type,
        "target_structure_type": str(item.get("target_structure_type") or block_type),
        "status": status,
        "reasons": reasons,
        "blockers": blockers,
        "text_char_count": len(text),
        "useful_char_ratio": useful_ratio,
        "confidence": round(confidence, 4),
        "engine": item["engine"],
        "language": item["language"],
        "input_path": item["input_path"],
        "warnings": item["warnings"],
        "preview": text[:160],
    }
    if table_context:
        assessment["table_context"] = _json_copy(table_context)
    if subtarget:
        assessment["subtarget"] = _json_copy(subtarget)
    if structure_contract:
        assessment["structure_contract"] = _json_copy(structure_contract)
    for key, value in structured_result_fields.items():
        assessment[key] = _json_copy(value)
    return assessment


def build_ocr_candidate_qa(
    document_ir_ocr: dict[str, Any] | None,
    ocr_writeback: dict[str, Any] | None = None,
    *,
    review_confidence: float = DEFAULT_REVIEW_CONFIDENCE,
) -> dict[str, Any]:
    candidates = _iter_candidates(document_ir_ocr)
    assessments = [_assessment(item, review_confidence=review_confidence) for item in candidates]
    status_counts = Counter(str(item.get("status") or "unknown") for item in assessments)
    issue_counts: Counter[str] = Counter()
    engine_counts = Counter(str(item.get("engine") or "unknown") for item in assessments)
    block_type_counts = Counter(str(item.get("block_type") or "unknown") for item in assessments)
    scope_counts = Counter(str(item.get("scope") or "unknown") for item in assessments)
    text_char_count = sum(int(item.get("text_char_count") or 0) for item in assessments)
    table_context_candidate_count = sum(1 for item in assessments if isinstance(item.get("table_context"), dict))
    structured_contract_candidate_count = sum(
        1 for item in assessments if isinstance(item.get("structure_contract"), dict)
    )
    subtarget_candidate_count = sum(1 for item in assessments if isinstance(item.get("subtarget"), dict))
    structured_result_candidate_count = sum(
        1
        for item in assessments
        if any(isinstance(item.get(key), (dict, list)) for key in STRUCTURED_RESULT_FIELDS)
    )
    structured_result_field_counts = {
        key: sum(1 for item in assessments if isinstance(item.get(key), (dict, list)))
        for key in STRUCTURED_RESULT_FIELDS
    }
    structured_result_item_counts = {
        key: sum(_item_count(item.get(key)) for item in assessments if isinstance(item.get(key), (dict, list)))
        for key in STRUCTURED_RESULT_FIELDS
    }
    for item in assessments:
        for reason in item.get("reasons") or []:
            issue_counts[str(reason)] += 1
        for blocker in item.get("blockers") or []:
            issue_counts[str(blocker)] += 1

    writeback_summary = (
        ocr_writeback.get("summary")
        if isinstance(ocr_writeback, dict) and isinstance(ocr_writeback.get("summary"), dict)
        else {}
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "doc_id": str((document_ir_ocr or {}).get("doc_id") or (ocr_writeback or {}).get("doc_id") or ""),
        "review_confidence": review_confidence,
        "summary": {
            "candidate_count": len(assessments),
            "promotable_candidate_count": status_counts.get("candidate", 0),
            "needs_review_candidate_count": status_counts.get("needs_review", 0),
            "blocked_candidate_count": status_counts.get("blocked", 0),
            "candidate_text_char_count": text_char_count,
            "table_context_candidate_count": table_context_candidate_count,
            "structured_contract_candidate_count": structured_contract_candidate_count,
            "subtarget_candidate_count": subtarget_candidate_count,
            "structured_result_candidate_count": structured_result_candidate_count,
            "structured_result_field_counts": structured_result_field_counts,
            "structured_result_item_counts": structured_result_item_counts,
            "structured_cells_candidate_count": structured_result_field_counts["structured_cells"],
            "cell_bboxes_candidate_count": structured_result_field_counts["cell_bboxes"],
            "merged_cell_candidates_candidate_count": structured_result_field_counts["merged_cell_candidates"],
            "table_footnotes_candidate_count": structured_result_field_counts["table_footnotes"],
            "structured_cell_count": structured_result_item_counts["structured_cells"],
            "cell_bbox_count": structured_result_item_counts["cell_bboxes"],
            "result_merged_cell_candidate_count": structured_result_item_counts["merged_cell_candidates"],
            "result_table_footnote_count": structured_result_item_counts["table_footnotes"],
            "writeback_accepted_result_count": int(writeback_summary.get("accepted_result_count") or 0),
            "status_counts": dict(status_counts),
            "issue_counts": dict(issue_counts),
            "engine_counts": dict(engine_counts),
            "block_type_counts": dict(block_type_counts),
            "scope_counts": dict(scope_counts),
        },
        "gate_policy": {
            "candidate": "May enter manual or downstream promotion review.",
            "needs_review": "Requires confidence, duplicate, or structure review before promotion.",
            "blocked": "Must not enter structure chunks or formal translation.",
        },
        "candidates": assessments,
    }


def ocr_candidate_qa_to_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# OCR Candidate QA",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Candidate count | {summary.get('candidate_count', 0)} |",
        f"| Promotable candidates | {summary.get('promotable_candidate_count', 0)} |",
        f"| Needs review | {summary.get('needs_review_candidate_count', 0)} |",
        f"| Blocked | {summary.get('blocked_candidate_count', 0)} |",
        f"| Candidate text chars | {summary.get('candidate_text_char_count', 0)} |",
        "",
        "## Issue Counts",
        "",
    ]
    issue_counts = summary.get("issue_counts") if isinstance(summary.get("issue_counts"), dict) else {}
    if issue_counts:
        lines.extend(["| Issue | Count |", "| --- | --- |"])
        for issue, count in sorted(issue_counts.items()):
            lines.append(f"| `{issue}` | {count} |")
    else:
        lines.append("No OCR candidate gate issues.")
    lines.extend(["", "## Candidate Details", ""])
    for item in report.get("candidates") or []:
        if not isinstance(item, dict):
            continue
        reasons = ", ".join(
            [str(value) for value in (item.get("reasons") or []) + (item.get("blockers") or [])]
        )
        reasons = reasons or "-"
        lines.append(
            f"- `{item.get('status')}` task `{item.get('task_id')}` "
            f"page {item.get('page_no')} block `{item.get('block_id') or '-'}`: {reasons}"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_ocr_candidate_qa(
    document_ir_ocr: dict[str, Any] | None,
    ocr_writeback: dict[str, Any] | None,
    json_path: Path,
    markdown_path: Path,
    *,
    review_confidence: float = DEFAULT_REVIEW_CONFIDENCE,
) -> dict[str, Any]:
    report = build_ocr_candidate_qa(
        document_ir_ocr,
        ocr_writeback,
        review_confidence=review_confidence,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(ocr_candidate_qa_to_markdown(report), encoding="utf-8")
    return report
