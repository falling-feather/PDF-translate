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
    "formula_latex",
    "formula_tokens",
    "equation_labels",
    "formula_confidence",
)
STRUCTURED_TABLE_PASS = "passed"
STRUCTURED_TABLE_REVIEW = "needs_review"
STRUCTURED_TABLE_BLOCKED = "blocked"
STRUCTURED_TABLE_NOT_APPLICABLE = "not_applicable"
STRUCTURED_FORMULA_PASS = "passed"
STRUCTURED_FORMULA_REVIEW = "needs_review"
STRUCTURED_FORMULA_BLOCKED = "blocked"
STRUCTURED_FORMULA_NOT_APPLICABLE = "not_applicable"
TRACE_ONLY_WARNINGS = {
    "structured_table_inferred_from_text",
    "cell_bboxes_estimated_from_region",
}


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
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
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
    if isinstance(value, str):
        return 1 if value.strip() else 0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return 1
    return 0


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
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


def _tokens_missing(tokens: Any, haystack: str) -> list[str]:
    if not isinstance(tokens, list):
        return []
    normalized = _normalized_text(haystack).casefold()
    missing: list[str] = []
    for token in tokens:
        item = _normalized_text(token)
        if item and item.casefold() not in normalized:
            missing.append(item)
    return missing


def _string_items(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_normalized_text(item) for item in value if _normalized_text(item)]
    if isinstance(value, str) and value.strip():
        return [_normalized_text(value)]
    return []


def _actionable_warnings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item) and str(item) not in TRACE_ONLY_WARNINGS]


def _cell_text(cell: Any) -> str:
    if isinstance(cell, dict):
        return _normalized_text(cell.get("text") or cell.get("value") or "")
    return _normalized_text(cell)


def _cell_position(cell: Any) -> tuple[int, int] | None:
    if not isinstance(cell, dict):
        return None
    row_value = cell.get("row", cell.get("row_index"))
    col_value = cell.get("col", cell.get("column", cell.get("column_index")))
    if row_value is None or col_value is None:
        return None
    row = _as_int(row_value)
    col = _as_int(col_value)
    if row < 0 or col < 0:
        return None
    return row, col


def _valid_bbox(value: Any) -> bool:
    bbox = value.get("bbox") if isinstance(value, dict) else value
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    for item in bbox:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return False
    return True


def _structured_table_gate(item: dict[str, Any]) -> dict[str, Any]:
    target_structure_type = str(item.get("target_structure_type") or item.get("block_type") or "")
    block_type = str(item.get("block_type") or "")
    structured_cells = item.get("structured_cells")
    if target_structure_type != "table" and block_type != "table":
        return {"status": STRUCTURED_TABLE_NOT_APPLICABLE}
    if not isinstance(structured_cells, list):
        return {
            "status": STRUCTURED_TABLE_REVIEW,
            "issues": ["missing_structured_cells"],
            "cell_count": 0,
            "row_count": 0,
            "column_count": 0,
        }

    positions: list[tuple[int, int]] = []
    cell_texts: list[str] = []
    malformed_cell_count = 0
    for cell in structured_cells:
        position = _cell_position(cell)
        if position is None:
            malformed_cell_count += 1
        else:
            positions.append(position)
        cell_text = _cell_text(cell)
        if cell_text:
            cell_texts.append(cell_text)

    row_count = max((row for row, _col in positions), default=-1) + 1
    column_count = max((col for _row, col in positions), default=-1) + 1
    table_context = item.get("table_context") if isinstance(item.get("table_context"), dict) else {}
    expected_row_count = _as_int(table_context.get("row_count"))
    expected_column_count = _as_int(table_context.get("column_count"))
    locked_tokens = table_context.get("locked_tokens") if isinstance(table_context, dict) else []
    missing_locked_tokens = _tokens_missing(locked_tokens, " ".join(cell_texts) or str(item.get("text") or ""))
    cell_bboxes = item.get("cell_bboxes")
    cell_bbox_count = _item_count(cell_bboxes)
    malformed_cell_bbox_count = (
        sum(1 for bbox in cell_bboxes if not _valid_bbox(bbox)) if isinstance(cell_bboxes, list) else 0
    )

    issues: list[str] = []
    blockers: list[str] = []
    if not structured_cells:
        blockers.append("structured_table_empty_cells")
    if malformed_cell_count:
        blockers.append("structured_table_malformed_cells")
    if not positions:
        blockers.append("structured_table_missing_cell_coordinates")
    if expected_row_count and row_count and row_count != expected_row_count:
        issues.append("structured_table_row_count_mismatch")
    if expected_column_count and column_count and column_count != expected_column_count:
        issues.append("structured_table_column_count_mismatch")
    if locked_tokens and missing_locked_tokens:
        issues.append("structured_table_missing_locked_tokens")
    if isinstance(cell_bboxes, list) and cell_bbox_count < len(structured_cells):
        issues.append("structured_table_incomplete_cell_bboxes")
    elif cell_bboxes is None:
        issues.append("structured_table_missing_cell_bboxes")
    if malformed_cell_bbox_count:
        issues.append("structured_table_malformed_cell_bboxes")

    if blockers:
        status = STRUCTURED_TABLE_BLOCKED
    elif issues:
        status = STRUCTURED_TABLE_REVIEW
    else:
        status = STRUCTURED_TABLE_PASS

    return {
        "status": status,
        "issues": issues,
        "blockers": blockers,
        "cell_count": len(structured_cells),
        "cell_bbox_count": cell_bbox_count,
        "row_count": row_count,
        "column_count": column_count,
        "expected_row_count": expected_row_count,
        "expected_column_count": expected_column_count,
        "missing_locked_tokens": missing_locked_tokens,
        "malformed_cell_count": malformed_cell_count,
        "malformed_cell_bbox_count": malformed_cell_bbox_count,
    }


def _structured_formula_gate(item: dict[str, Any], *, review_confidence: float) -> dict[str, Any]:
    target_structure_type = str(item.get("target_structure_type") or item.get("block_type") or "")
    block_type = str(item.get("block_type") or "")
    if target_structure_type != "formula" and block_type != "formula":
        return {"status": STRUCTURED_FORMULA_NOT_APPLICABLE}

    formula_context = item.get("formula_context") if isinstance(item.get("formula_context"), dict) else {}
    text = str(item.get("text") or "")
    latex = str(item.get("formula_latex") or "").strip()
    formula_tokens = _string_items(item.get("formula_tokens"))
    equation_labels = _string_items(item.get("equation_labels"))
    source_tokens = _string_items(formula_context.get("source_tokens"))
    locked_tokens = _string_items(formula_context.get("locked_tokens"))
    expected_equation_labels = [
        token for token in source_tokens if token.startswith("(") and token.endswith(")")
    ]
    haystack = " ".join([text, latex, " ".join(formula_tokens), " ".join(equation_labels)])
    missing_locked_tokens = _tokens_missing(locked_tokens, haystack)
    missing_equation_labels = _tokens_missing(expected_equation_labels, haystack)
    formula_confidence = _as_float(item.get("formula_confidence"))

    issues: list[str] = []
    blockers: list[str] = []
    if not latex:
        issues.append("structured_formula_missing_latex")
    if not formula_tokens:
        issues.append("structured_formula_missing_tokens")
    if expected_equation_labels and missing_equation_labels:
        issues.append("structured_formula_missing_equation_labels")
    if locked_tokens and missing_locked_tokens:
        issues.append("structured_formula_missing_locked_tokens")
    if item.get("formula_confidence") is not None and formula_confidence < review_confidence:
        issues.append("structured_formula_low_confidence")
    if not text.strip() and not latex:
        blockers.append("structured_formula_empty_text")

    if blockers:
        status = STRUCTURED_FORMULA_BLOCKED
    elif issues:
        status = STRUCTURED_FORMULA_REVIEW
    else:
        status = STRUCTURED_FORMULA_PASS

    return {
        "status": status,
        "issues": issues,
        "blockers": blockers,
        "formula_token_count": len(formula_tokens),
        "equation_label_count": len(equation_labels),
        "expected_equation_label_count": len(expected_equation_labels),
        "missing_equation_labels": missing_equation_labels,
        "missing_locked_tokens": missing_locked_tokens,
        "formula_confidence": round(formula_confidence, 4),
    }


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
                "formula_context": _json_copy(candidate.get("formula_context"))
                if isinstance(candidate.get("formula_context"), dict)
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
    formula_context = item.get("formula_context") if isinstance(item.get("formula_context"), dict) else {}
    subtarget = item.get("subtarget") if isinstance(item.get("subtarget"), dict) else {}
    structure_contract = (
        item.get("structure_contract") if isinstance(item.get("structure_contract"), dict) else {}
    )
    structured_result_fields = _structured_result_fields(item)
    structured_table_gate = _structured_table_gate(item)
    structured_formula_gate = _structured_formula_gate(item, review_confidence=review_confidence)
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
    if structured_table_gate.get("status") == STRUCTURED_TABLE_BLOCKED:
        blockers.extend([str(value) for value in structured_table_gate.get("blockers") or []])
        reasons.extend([str(value) for value in structured_table_gate.get("issues") or []])
    elif structured_table_gate.get("status") == STRUCTURED_TABLE_REVIEW:
        reasons.extend([str(value) for value in structured_table_gate.get("issues") or []])
        if "missing_structured_cells" in structured_table_gate.get("issues", []):
            reasons.append("needs_table_structure_review")
        else:
            reasons.append("needs_structured_table_review")
    elif structured_table_gate.get("status") == STRUCTURED_TABLE_NOT_APPLICABLE and block_type in STRUCTURE_REVIEW_BLOCK_TYPES:
        if block_type != "formula":
            reasons.append(f"needs_{block_type}_structure_review")
    if structured_formula_gate.get("status") == STRUCTURED_FORMULA_BLOCKED:
        blockers.extend([str(value) for value in structured_formula_gate.get("blockers") or []])
        reasons.extend([str(value) for value in structured_formula_gate.get("issues") or []])
    elif structured_formula_gate.get("status") == STRUCTURED_FORMULA_REVIEW:
        reasons.extend([str(value) for value in structured_formula_gate.get("issues") or []])
        reasons.append("needs_structured_formula_review")
    elif (
        structured_formula_gate.get("status") == STRUCTURED_FORMULA_NOT_APPLICABLE
        and block_type == "formula"
    ):
        reasons.append("needs_formula_structure_review")
    if _actionable_warnings(item.get("warnings")):
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
    if formula_context:
        assessment["formula_context"] = _json_copy(formula_context)
    if subtarget:
        assessment["subtarget"] = _json_copy(subtarget)
    if structure_contract:
        assessment["structure_contract"] = _json_copy(structure_contract)
    for key, value in structured_result_fields.items():
        assessment[key] = _json_copy(value)
    if structured_table_gate.get("status") != STRUCTURED_TABLE_NOT_APPLICABLE:
        assessment["structured_table_gate"] = _json_copy(structured_table_gate)
    if structured_formula_gate.get("status") != STRUCTURED_FORMULA_NOT_APPLICABLE:
        assessment["structured_formula_gate"] = _json_copy(structured_formula_gate)
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
    formula_context_candidate_count = sum(1 for item in assessments if isinstance(item.get("formula_context"), dict))
    structured_contract_candidate_count = sum(
        1 for item in assessments if isinstance(item.get("structure_contract"), dict)
    )
    subtarget_candidate_count = sum(1 for item in assessments if isinstance(item.get("subtarget"), dict))
    structured_result_candidate_count = sum(
        1
        for item in assessments
        if any(_structured_payload(item.get(key)) is not None for key in STRUCTURED_RESULT_FIELDS)
    )
    structured_result_field_counts = {
        key: sum(1 for item in assessments if _structured_payload(item.get(key)) is not None)
        for key in STRUCTURED_RESULT_FIELDS
    }
    structured_result_item_counts = {
        key: sum(_item_count(item.get(key)) for item in assessments if _structured_payload(item.get(key)) is not None)
        for key in STRUCTURED_RESULT_FIELDS
    }
    structured_table_gate_counts = Counter(
        str(item.get("structured_table_gate", {}).get("status") or STRUCTURED_TABLE_NOT_APPLICABLE)
        for item in assessments
    )
    structured_formula_gate_counts = Counter(
        str(item.get("structured_formula_gate", {}).get("status") or STRUCTURED_FORMULA_NOT_APPLICABLE)
        for item in assessments
    )
    structured_table_gate_issue_counts: Counter[str] = Counter()
    structured_formula_gate_issue_counts: Counter[str] = Counter()
    structured_table_missing_locked_token_count = 0
    structured_formula_missing_locked_token_count = 0
    structured_formula_token_count = 0
    structured_formula_equation_label_count = 0
    for item in assessments:
        gate = item.get("structured_table_gate") if isinstance(item.get("structured_table_gate"), dict) else {}
        for issue in gate.get("issues") or []:
            structured_table_gate_issue_counts[str(issue)] += 1
        for blocker in gate.get("blockers") or []:
            structured_table_gate_issue_counts[str(blocker)] += 1
        structured_table_missing_locked_token_count += len(gate.get("missing_locked_tokens") or [])
        formula_gate = (
            item.get("structured_formula_gate") if isinstance(item.get("structured_formula_gate"), dict) else {}
        )
        for issue in formula_gate.get("issues") or []:
            structured_formula_gate_issue_counts[str(issue)] += 1
        for blocker in formula_gate.get("blockers") or []:
            structured_formula_gate_issue_counts[str(blocker)] += 1
        structured_formula_missing_locked_token_count += len(formula_gate.get("missing_locked_tokens") or [])
        structured_formula_token_count += int(formula_gate.get("formula_token_count") or 0)
        structured_formula_equation_label_count += int(formula_gate.get("equation_label_count") or 0)
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
            "formula_context_candidate_count": formula_context_candidate_count,
            "structured_contract_candidate_count": structured_contract_candidate_count,
            "subtarget_candidate_count": subtarget_candidate_count,
            "structured_result_candidate_count": structured_result_candidate_count,
            "structured_result_field_counts": structured_result_field_counts,
            "structured_result_item_counts": structured_result_item_counts,
            "structured_cells_candidate_count": structured_result_field_counts["structured_cells"],
            "cell_bboxes_candidate_count": structured_result_field_counts["cell_bboxes"],
            "merged_cell_candidates_candidate_count": structured_result_field_counts["merged_cell_candidates"],
            "table_footnotes_candidate_count": structured_result_field_counts["table_footnotes"],
            "formula_latex_candidate_count": structured_result_field_counts["formula_latex"],
            "formula_tokens_candidate_count": structured_result_field_counts["formula_tokens"],
            "equation_labels_candidate_count": structured_result_field_counts["equation_labels"],
            "formula_confidence_candidate_count": structured_result_field_counts["formula_confidence"],
            "structured_cell_count": structured_result_item_counts["structured_cells"],
            "cell_bbox_count": structured_result_item_counts["cell_bboxes"],
            "result_merged_cell_candidate_count": structured_result_item_counts["merged_cell_candidates"],
            "result_table_footnote_count": structured_result_item_counts["table_footnotes"],
            "result_formula_latex_count": structured_result_item_counts["formula_latex"],
            "result_formula_token_count": structured_result_item_counts["formula_tokens"],
            "result_equation_label_count": structured_result_item_counts["equation_labels"],
            "structured_table_gate_counts": dict(structured_table_gate_counts),
            "structured_table_gate_issue_counts": dict(structured_table_gate_issue_counts),
            "structured_table_candidate_count": (
                structured_table_gate_counts.get(STRUCTURED_TABLE_PASS, 0)
                + structured_table_gate_counts.get(STRUCTURED_TABLE_REVIEW, 0)
                + structured_table_gate_counts.get(STRUCTURED_TABLE_BLOCKED, 0)
            ),
            "structured_table_gate_passed_count": structured_table_gate_counts.get(STRUCTURED_TABLE_PASS, 0),
            "structured_table_gate_review_count": structured_table_gate_counts.get(STRUCTURED_TABLE_REVIEW, 0),
            "structured_table_gate_blocked_count": structured_table_gate_counts.get(STRUCTURED_TABLE_BLOCKED, 0),
            "structured_table_missing_locked_token_count": structured_table_missing_locked_token_count,
            "structured_formula_gate_counts": dict(structured_formula_gate_counts),
            "structured_formula_gate_issue_counts": dict(structured_formula_gate_issue_counts),
            "structured_formula_candidate_count": (
                structured_formula_gate_counts.get(STRUCTURED_FORMULA_PASS, 0)
                + structured_formula_gate_counts.get(STRUCTURED_FORMULA_REVIEW, 0)
                + structured_formula_gate_counts.get(STRUCTURED_FORMULA_BLOCKED, 0)
            ),
            "structured_formula_gate_passed_count": structured_formula_gate_counts.get(STRUCTURED_FORMULA_PASS, 0),
            "structured_formula_gate_review_count": structured_formula_gate_counts.get(STRUCTURED_FORMULA_REVIEW, 0),
            "structured_formula_gate_blocked_count": structured_formula_gate_counts.get(STRUCTURED_FORMULA_BLOCKED, 0),
            "structured_formula_missing_locked_token_count": structured_formula_missing_locked_token_count,
            "structured_formula_token_count": structured_formula_token_count,
            "structured_formula_equation_label_count": structured_formula_equation_label_count,
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
