from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pdf_translate.extractors.document_ir import DocumentIR

SCHEMA_VERSION = "ocr-writeback-v1"
OCR_RESULTS_SCHEMA_VERSION = "ocr-results-v1"
DEFAULT_MIN_CONFIDENCE = 0.5
SUCCESS_STATUSES = {"ok", "success", "succeeded", "completed", "done"}
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
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?")


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _tasks(ocr_tasks: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(ocr_tasks, dict):
        return []
    raw = ocr_tasks.get("tasks")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _results(ocr_results: dict[str, Any] | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if isinstance(ocr_results, list):
        raw = ocr_results
    elif isinstance(ocr_results, dict):
        raw = ocr_results.get("results")
    else:
        raw = []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _page_index(document_ir: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for page in document_ir.get("pages") or []:
        if not isinstance(page, dict):
            continue
        try:
            page_no = int(page.get("page_no") or 0)
        except (TypeError, ValueError):
            continue
        if page_no > 0:
            out[page_no] = page
    return out


def _block_index(document_ir: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for page in document_ir.get("pages") or []:
        if not isinstance(page, dict):
            continue
        for block in page.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            block_id = str(block.get("block_id") or "")
            if block_id:
                out[block_id] = block
    return out


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


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        parsed = int(value)
        return parsed if parsed >= 0 else None
    if isinstance(value, str):
        try:
            parsed = int(float(value.strip()))
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None


def _normalized_bbox(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[float] = []
    for item in value:
        if isinstance(item, bool):
            return []
        if isinstance(item, (int, float)):
            out.append(float(item))
        elif isinstance(item, str):
            try:
                out.append(float(item))
            except ValueError:
                return []
        else:
            return []
    return out


def _normalized_warnings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _structured_payload(value: Any) -> Any | None:
    if isinstance(value, (dict, list)):
        return _json_copy(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def _normalised_covered_cells(value: Any) -> list[dict[str, int]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, int]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        row = _nonnegative_int(item.get("row_index", item.get("row")))
        col = _nonnegative_int(item.get("column_index", item.get("col", item.get("column"))))
        if row is None or col is None:
            continue
        out.append({"row": row, "col": col, "row_index": row, "column_index": col})
    return out


def _normalise_merged_cell_candidate(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    out = _json_copy(value)
    span_type = str(out.get("span_type") or out.get("type") or "").strip()
    if span_type:
        out.setdefault("span_type", span_type)
        out.setdefault("type", span_type)

    rows = out.get("rows") if isinstance(out.get("rows"), list) else []
    cols = out.get("cols") if isinstance(out.get("cols"), list) else []
    row_index = _nonnegative_int(out.get("row_index", out.get("row")))
    if row_index is None and rows:
        row_index = _nonnegative_int(rows[0])
    column_index = _nonnegative_int(out.get("column_index", out.get("col", out.get("column"))))
    if column_index is None and cols:
        column_index = _nonnegative_int(cols[0])
    if row_index is not None:
        out.setdefault("row", row_index)
        out.setdefault("row_index", row_index)
    if column_index is not None:
        out.setdefault("col", column_index)
        out.setdefault("column_index", column_index)

    row_span = _positive_int(out.get("row_span"))
    column_span = _positive_int(out.get("column_span"))
    normalised_rows = [_nonnegative_int(item) for item in rows]
    normalised_rows = [item for item in normalised_rows if item is not None]
    normalised_cols = [_nonnegative_int(item) for item in cols]
    normalised_cols = [item for item in normalised_cols if item is not None]
    if row_span <= 0 and row_index is not None and normalised_rows:
        row_span = max(normalised_rows) - row_index + 1
    if column_span <= 0 and column_index is not None and normalised_cols:
        column_span = max(normalised_cols) - column_index + 1
    if row_span <= 0:
        row_span = 1
    if column_span <= 0:
        column_span = 1
    out.setdefault("row_span", row_span)
    out.setdefault("column_span", column_span)

    covered_cells = _normalised_covered_cells(out.get("covered_cells"))
    if not covered_cells and row_index is not None and column_index is not None:
        if span_type == "colspan" and column_span > 1:
            covered_cells = [
                _covered_cell(row_index, col_index)
                for col_index in range(column_index + 1, column_index + column_span)
            ]
        elif span_type == "rowspan" and row_span > 1:
            covered_cells = [
                _covered_cell(row_index + offset, column_index)
                for offset in range(1, row_span)
            ]
    if covered_cells:
        out["covered_cells"] = covered_cells
    out.setdefault("candidate_status", "candidate")
    candidate_bbox = _normalized_bbox(out.get("bbox") or out.get("anchor_bbox") or out.get("span_bbox"))
    if "visual_evidence_level" not in out:
        out["visual_evidence_level"] = "visual_span_bbox" if len(candidate_bbox) == 4 else "none"
    if not isinstance(out.get("bbox_evidence"), dict):
        out["bbox_evidence"] = {
            "status": "span_reported" if len(candidate_bbox) == 4 else "missing",
            "source": "candidate_bbox" if len(candidate_bbox) == 4 else "not_provided",
        }
    return out


def _normalise_merged_cell_candidates(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalise_merged_cell_candidate(item) for item in value]
    return _normalise_merged_cell_candidate(value)


def _structured_result_fields(source: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in STRUCTURED_RESULT_FIELDS:
        raw_value = source.get(key)
        if key == "merged_cell_candidates":
            raw_value = _normalise_merged_cell_candidates(raw_value)
        value = _structured_payload(raw_value)
        if value is not None:
            out[key] = value
    return out


def _is_table_task(task: dict[str, Any]) -> bool:
    return (
        str(task.get("target_structure_type") or "") == "table"
        or str(task.get("block_type") or "") == "table"
        or str(task.get("layout_scope") or "") == "table_region"
        or isinstance(task.get("table_context"), dict)
    )


def _pipe_table_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if "|" not in line:
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if not parts:
            continue
        if all(part and set(part) <= {"-", ":", " "} for part in parts):
            continue
        rows.append(parts)
    return rows if len(rows) >= 2 else []


def _whitespace_table_rows(text: str, expected_columns: int) -> list[list[str]]:
    if expected_columns < 2:
        return []
    rows: list[list[str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "\t" in line:
            parts = [part.strip() for part in line.split("\t") if part.strip()]
        else:
            parts = [part.strip() for part in re.split(r"\s{2,}", line) if part.strip()]
            if len(parts) < expected_columns:
                parts = [part.strip() for part in line.split() if part.strip()]
        if len(parts) != expected_columns:
            return []
        rows.append(parts)
    return rows if len(rows) >= 2 else []


def _positive_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value if value > 0 else 0
    if isinstance(value, float):
        return int(value) if value > 0 else 0
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            return 0
        return parsed if parsed > 0 else 0
    return 0


def _normalise_rows(rows: list[list[str]], column_count: int) -> list[list[str]]:
    if column_count <= 0:
        column_count = max((len(row) for row in rows), default=0)
    return [row + [""] * max(0, column_count - len(row)) for row in rows]


def _table_rows_from_text(text: str, expected_columns: int) -> tuple[list[list[str]], list[list[str]]]:
    raw_rows = _pipe_table_rows(text)
    if not raw_rows:
        raw_rows = _whitespace_table_rows(text, expected_columns)
    if not raw_rows:
        return [], []
    column_count = max(expected_columns, max((len(row) for row in raw_rows), default=0))
    return raw_rows, _normalise_rows(raw_rows, column_count)


def _covered_cell(row_index: int, column_index: int) -> dict[str, int]:
    return {
        "row": row_index,
        "col": column_index,
        "row_index": row_index,
        "column_index": column_index,
    }


def _merged_cell_candidate(
    *,
    span_type: str,
    row_index: int,
    column_index: int,
    row_span: int,
    column_span: int,
    text: str,
    reason: str,
    confidence: str,
    covered_cells: list[dict[str, int]],
) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "type": span_type,
        "span_type": span_type,
        "row": row_index,
        "col": column_index,
        "row_index": row_index,
        "column_index": column_index,
        "row_span": row_span,
        "column_span": column_span,
        "text": text,
        "reason": reason,
        "confidence": confidence,
        "covered_cells": covered_cells,
        "source": "local_text_table_parser",
        "candidate_status": "candidate",
        "visual_evidence_level": "none",
        "bbox_evidence": {"status": "missing", "source": "text_heuristic"},
    }
    if span_type == "colspan":
        candidate["cols"] = list(range(column_index, column_index + column_span))
    if span_type == "rowspan":
        candidate["rows"] = list(range(row_index, row_index + row_span))
    return candidate


def _merged_cell_candidates_from_rows(rows: list[list[str]], column_count: int) -> list[dict[str, Any]]:
    if column_count < 2 or not rows:
        return []
    normalised = _normalise_rows(rows, column_count)
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def add(candidate: dict[str, Any]) -> None:
        covered = tuple(
            (cell["row_index"], cell["column_index"])
            for cell in candidate.get("covered_cells") or []
        )
        key = (
            candidate.get("span_type"),
            candidate.get("row_index"),
            candidate.get("column_index"),
            candidate.get("row_span"),
            candidate.get("column_span"),
            candidate.get("reason"),
            covered,
        )
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)

    for row_index, row in enumerate(normalised):
        raw_row = rows[row_index] if row_index < len(rows) else row
        padded_columns = set(range(max(0, len(raw_row)), column_count))
        nonempty_columns = [index for index, value in enumerate(row) if str(value).strip()]
        if len(raw_row) < column_count and nonempty_columns:
            if len(nonempty_columns) == 1:
                anchor_column = nonempty_columns[0]
                confidence = "medium" if row_index == 0 and not NUMBER_RE.search(row[anchor_column]) else "low"
                add(
                    _merged_cell_candidate(
                        span_type="colspan",
                        row_index=row_index,
                        column_index=anchor_column,
                        row_span=1,
                        column_span=max(1, column_count - anchor_column),
                        text=row[anchor_column],
                        reason="single_cell_ragged_row",
                        confidence=confidence,
                        covered_cells=[
                            _covered_cell(row_index, column_index)
                            for column_index in range(anchor_column + 1, column_count)
                        ],
                    )
                )
            else:
                anchor_column = max(nonempty_columns)
                if anchor_column < column_count - 1:
                    add(
                        _merged_cell_candidate(
                            span_type="colspan",
                            row_index=row_index,
                            column_index=anchor_column,
                            row_span=1,
                            column_span=column_count - anchor_column,
                            text=row[anchor_column],
                            reason="ragged_row_trailing_span",
                            confidence="low",
                            covered_cells=[
                                _covered_cell(row_index, column_index)
                                for column_index in range(anchor_column + 1, column_count)
                            ],
                        )
                    )

        for column_index, text in enumerate(row):
            if column_index in padded_columns or str(text).strip():
                continue
            if row_index > 0 and str(normalised[row_index - 1][column_index]).strip():
                confidence = "medium" if column_index == 0 else "low"
                add(
                    _merged_cell_candidate(
                        span_type="rowspan",
                        row_index=row_index - 1,
                        column_index=column_index,
                        row_span=2,
                        column_span=1,
                        text=normalised[row_index - 1][column_index],
                        reason="empty_cell_below_nonempty_anchor",
                        confidence=confidence,
                        covered_cells=[_covered_cell(row_index, column_index)],
                    )
                )
            if column_index > 0 and str(row[column_index - 1]).strip():
                add(
                    _merged_cell_candidate(
                        span_type="colspan",
                        row_index=row_index,
                        column_index=column_index - 1,
                        row_span=1,
                        column_span=2,
                        text=row[column_index - 1],
                        reason="empty_cell_right_of_nonempty_anchor",
                        confidence="low",
                        covered_cells=[_covered_cell(row_index, column_index)],
                    )
                )

    return candidates


def _estimated_cell_bboxes(rows: list[list[str]], bbox: list[float]) -> list[dict[str, Any]]:
    if len(bbox) != 4 or not rows:
        return []
    row_count = len(rows)
    column_count = max((len(row) for row in rows), default=0)
    if row_count <= 0 or column_count <= 0:
        return []
    x0, y0, x1, y1 = bbox
    width = (x1 - x0) / column_count
    height = (y1 - y0) / row_count
    out: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        for col_index, _ in enumerate(row):
            out.append(
                {
                    "row": row_index,
                    "col": col_index,
                    "bbox": [
                        round(x0 + width * col_index, 2),
                        round(y0 + height * row_index, 2),
                        round(x0 + width * (col_index + 1), 2),
                        round(y0 + height * (row_index + 1), 2),
                    ],
                    "estimated": True,
                }
            )
    return out


def _infer_structured_table_from_text(text: str, task: dict[str, Any], bbox: list[float]) -> dict[str, Any]:
    if not _is_table_task(task):
        return {}
    table_context = task.get("table_context") if isinstance(task.get("table_context"), dict) else {}
    expected_columns = _positive_int(table_context.get("column_count"))
    raw_rows, rows = _table_rows_from_text(text, expected_columns)
    if not rows:
        return {}
    column_count = max(len(row) for row in rows)
    structured_cells: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            cell: dict[str, Any] = {
                "row": row_index,
                "col": col_index,
                "text": value,
                "source": "local_text_table_parser",
            }
            if row_index == 0:
                cell["role"] = "header"
            structured_cells.append(cell)
    inferred: dict[str, Any] = {
        "structured_cells": structured_cells,
        "row_count": len(rows),
        "column_count": column_count,
    }
    cell_bboxes = _estimated_cell_bboxes(rows, bbox)
    if cell_bboxes:
        inferred["cell_bboxes"] = cell_bboxes
    merged_cell_candidates = _merged_cell_candidates_from_rows(raw_rows, column_count)
    if merged_cell_candidates:
        inferred["merged_cell_candidates"] = merged_cell_candidates
    return inferred


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


def _reject(result: dict[str, Any], task: dict[str, Any] | None, reason: str) -> dict[str, Any]:
    return {
        "task_id": str(result.get("task_id") or ""),
        "page_no": int((task or {}).get("page_no") or 0),
        "block_id": str((task or {}).get("block_id") or ""),
        "status": str(result.get("status") or "succeeded"),
        "engine": str(result.get("engine") or (task or {}).get("recommended_engine") or ""),
        "reason": reason,
        "text_char_count": len(str(result.get("text") or "")),
        "confidence": _as_float(result.get("confidence")),
    }


def _pending_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": str(task.get("task_id") or ""),
        "page_no": int(task.get("page_no") or 0),
        "block_id": str(task.get("block_id") or ""),
        "scope": str(task.get("scope") or ""),
        "status": str(task.get("status") or ""),
        "priority": str(task.get("priority") or ""),
        "recommended_engine": str(task.get("recommended_engine") or ""),
        "input_path": str(task.get("input_path") or ""),
    }


def _candidate(
    result: dict[str, Any],
    task: dict[str, Any],
    *,
    confidence: float,
    text: str,
) -> dict[str, Any]:
    bbox = _normalized_bbox(result.get("bbox")) or _normalized_bbox(task.get("bbox"))
    candidate: dict[str, Any] = {
        "source": "ocr_result",
        "task_id": str(task.get("task_id") or ""),
        "page_no": int(task.get("page_no") or 0),
        "block_id": str(task.get("block_id") or ""),
        "scope": str(task.get("scope") or ""),
        "block_type": str(task.get("block_type") or ""),
        "target_structure_type": str(task.get("target_structure_type") or ""),
        "text": text,
        "confidence": round(confidence, 4),
        "engine": str(result.get("engine") or task.get("recommended_engine") or ""),
        "language": str(result.get("language") or "unknown"),
        "bbox": bbox,
        "warnings": _normalized_warnings(result.get("warnings")),
        "input_path": str(task.get("input_path") or ""),
        "result_status": str(result.get("status") or "succeeded"),
    }
    for key in ("table_context", "formula_context", "structure_contract"):
        value = task.get(key)
        if isinstance(value, dict):
            candidate[key] = _json_copy(value)
    writeback = task.get("writeback") if isinstance(task.get("writeback"), dict) else {}
    subtarget = writeback.get("subtarget") if isinstance(writeback.get("subtarget"), dict) else {}
    if subtarget:
        candidate["subtarget"] = _json_copy(subtarget)
    candidate.update(_structured_result_fields(result))
    if "structured_cells" not in candidate:
        inferred = _infer_structured_table_from_text(text, task, bbox)
        if inferred:
            candidate.update(inferred)
            candidate["warnings"].append("structured_table_inferred_from_text")
            if "cell_bboxes" in inferred:
                candidate["warnings"].append("cell_bboxes_estimated_from_region")
            if "merged_cell_candidates" in inferred:
                candidate["warnings"].append("merged_cell_candidates_inferred_from_text")
    return candidate


def build_empty_ocr_results(ocr_tasks: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "schema_version": OCR_RESULTS_SCHEMA_VERSION,
        "doc_id": str((ocr_tasks or {}).get("doc_id") or ""),
        "results": [],
    }


def load_ocr_results(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) or isinstance(payload, list):
        return payload
    raise ValueError("OCR results must be a JSON object or a list of result objects.")


def build_ocr_results_payload(
    ocr_tasks: dict[str, Any] | None,
    ocr_results: dict[str, Any] | list[dict[str, Any]] | None = None,
    *,
    source_path: str = "",
) -> dict[str, Any]:
    if ocr_results is None:
        payload = build_empty_ocr_results(ocr_tasks)
        payload["source"] = "not_provided"
    elif isinstance(ocr_results, dict):
        payload = _json_copy(ocr_results)
        payload["schema_version"] = str(payload.get("schema_version") or OCR_RESULTS_SCHEMA_VERSION)
        payload["doc_id"] = str(payload.get("doc_id") or (ocr_tasks or {}).get("doc_id") or "")
        payload["source"] = str(payload.get("source") or "provided_file")
    elif isinstance(ocr_results, list):
        payload = {
            "schema_version": OCR_RESULTS_SCHEMA_VERSION,
            "doc_id": str((ocr_tasks or {}).get("doc_id") or ""),
            "source": "provided_list",
            "results": _json_copy(ocr_results),
        }
    else:
        raise ValueError("OCR results must be a JSON object or a list of result objects.")

    raw_results = payload.get("results")
    raw_result_count = len(raw_results) if isinstance(raw_results, list) else 0
    normalized_results = _results(payload)
    status_counts = Counter(str(item.get("status") or "succeeded") for item in normalized_results)
    engine_counts = Counter(str(item.get("engine") or "unknown") for item in normalized_results)
    payload["results"] = normalized_results
    payload["summary"] = {
        "result_count": len(normalized_results),
        "invalid_result_count": raw_result_count - len(normalized_results),
        "status_counts": dict(status_counts),
        "engine_counts": dict(engine_counts),
    }
    if source_path:
        payload["source_path"] = source_path
    return payload


def write_ocr_results_payload(
    ocr_tasks: dict[str, Any] | None,
    path: Path,
    ocr_results: dict[str, Any] | list[dict[str, Any]] | None = None,
    *,
    source_path: Path | str | None = None,
) -> dict[str, Any]:
    payload = build_ocr_results_payload(
        ocr_tasks,
        ocr_results,
        source_path=str(source_path or ""),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_ocr_writeback(
    doc_ir: DocumentIR,
    ocr_tasks: dict[str, Any] | None,
    ocr_results: dict[str, Any] | list[dict[str, Any]] | None = None,
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> dict[str, Any]:
    """Append accepted OCR result candidates to a copied DocumentIR payload."""
    augmented_ir = _json_copy(doc_ir.to_json_dict())
    pages = _page_index(augmented_ir)
    blocks = _block_index(augmented_ir)
    task_list = _tasks(ocr_tasks)
    task_index = {str(task.get("task_id") or ""): task for task in task_list if str(task.get("task_id") or "")}
    result_list = _results(ocr_results)

    writebacks: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    tasks_with_results: set[str] = set()
    result_status_counts: Counter[str] = Counter()
    accepted_engine_counts: Counter[str] = Counter()
    rejection_reason_counts: Counter[str] = Counter()
    block_writeback_count = 0
    page_writeback_count = 0
    table_context_writeback_count = 0
    formula_context_writeback_count = 0
    structured_result_writeback_count = 0
    structured_result_field_counts: Counter[str] = Counter()
    structured_result_item_counts: Counter[str] = Counter()

    for result in result_list:
        task_id = str(result.get("task_id") or "")
        status = str(result.get("status") or "succeeded")
        result_status_counts[status] += 1
        task = task_index.get(task_id)
        if task is None:
            item = _reject(result, None, "unknown_task")
            rejected.append(item)
            rejection_reason_counts[item["reason"]] += 1
            continue

        tasks_with_results.add(task_id)
        task_status = str(task.get("status") or "")
        if task_status != "pending_engine":
            item = _reject(result, task, "task_not_ready_for_writeback")
            rejected.append(item)
            rejection_reason_counts[item["reason"]] += 1
            continue
        if status not in SUCCESS_STATUSES:
            item = _reject(result, task, "result_not_succeeded")
            rejected.append(item)
            rejection_reason_counts[item["reason"]] += 1
            continue

        text = str(result.get("text") or "").strip()
        if not text:
            item = _reject(result, task, "empty_text")
            rejected.append(item)
            rejection_reason_counts[item["reason"]] += 1
            continue
        confidence = _as_float(result.get("confidence"))
        if confidence < min_confidence:
            item = _reject(result, task, "low_confidence")
            rejected.append(item)
            rejection_reason_counts[item["reason"]] += 1
            continue

        block_id = str(task.get("block_id") or "")
        page_no = int(task.get("page_no") or 0)
        target = blocks.get(block_id) if block_id else None
        target_kind = "block" if target is not None else "page"
        if target is None:
            target = pages.get(page_no)
        if target is None:
            item = _reject(result, task, "writeback_target_missing")
            rejected.append(item)
            rejection_reason_counts[item["reason"]] += 1
            continue

        candidate = _candidate(result, task, confidence=confidence, text=text)
        meta = target.setdefault("meta", {})
        candidates = meta.setdefault("ocr_candidates", [])
        candidates.append(candidate)
        candidate_index = len(candidates) - 1
        accepted_engine_counts[candidate["engine"] or "unknown"] += 1
        if target_kind == "block":
            block_writeback_count += 1
        else:
            page_writeback_count += 1
        if isinstance(candidate.get("table_context"), dict):
            table_context_writeback_count += 1
        if isinstance(candidate.get("formula_context"), dict):
            formula_context_writeback_count += 1
        structured_fields = {
            key: candidate[key]
            for key in STRUCTURED_RESULT_FIELDS
            if _structured_payload(candidate.get(key)) is not None
        }
        if structured_fields:
            structured_result_writeback_count += 1
            for key, value in structured_fields.items():
                structured_result_field_counts[key] += 1
                structured_result_item_counts[key] += _item_count(value)
        writeback_record = {
            "task_id": task_id,
            "page_no": page_no,
            "block_id": block_id,
            "target": f"document_ir.{target_kind}.meta.ocr_candidates",
            "candidate_index": candidate_index,
            "text_char_count": len(text),
            "confidence": candidate["confidence"],
            "engine": candidate["engine"],
        }
        for key in ("table_context", "formula_context", "subtarget"):
            value = candidate.get(key)
            if isinstance(value, dict):
                writeback_record[key] = _json_copy(value)
        if structured_fields:
            writeback_record["structured_result_fields"] = sorted(structured_fields)
            writeback_record["structured_result_item_counts"] = {
                key: _item_count(value) for key, value in sorted(structured_fields.items())
            }
        writebacks.append(writeback_record)

    pending = [_pending_task(task) for task in task_list if str(task.get("task_id") or "") not in tasks_with_results]
    unknown_task_result_count = rejection_reason_counts.get("unknown_task", 0)

    return {
        "schema_version": SCHEMA_VERSION,
        "doc_id": doc_ir.doc_id,
        "source_pdf": doc_ir.source_pdf,
        "min_confidence": min_confidence,
        "summary": {
            "task_count": len(task_list),
            "result_count": len(result_list),
            "accepted_result_count": len(writebacks),
            "rejected_result_count": len(rejected),
            "pending_task_count": len(pending),
            "missing_result_task_count": len(pending),
            "unknown_task_result_count": unknown_task_result_count,
            "block_writeback_count": block_writeback_count,
            "page_writeback_count": page_writeback_count,
            "table_context_writeback_count": table_context_writeback_count,
            "formula_context_writeback_count": formula_context_writeback_count,
            "structured_result_writeback_count": structured_result_writeback_count,
            "structured_result_field_counts": dict(structured_result_field_counts),
            "structured_result_item_counts": dict(structured_result_item_counts),
            "result_status_counts": dict(result_status_counts),
            "accepted_engine_counts": dict(accepted_engine_counts),
            "rejection_reason_counts": dict(rejection_reason_counts),
        },
        "writebacks": writebacks,
        "rejected_results": rejected,
        "pending_tasks": pending,
        "augmented_document_ir": augmented_ir,
    }


def _artifact_rel(path: Path) -> str:
    if path.parent.name == "output":
        return f"output/{path.name}"
    return path.as_posix()


def write_ocr_writeback(
    doc_ir: DocumentIR,
    ocr_tasks: dict[str, Any] | None,
    report_path: Path,
    augmented_ir_path: Path,
    ocr_results: dict[str, Any] | list[dict[str, Any]] | None = None,
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> dict[str, Any]:
    payload = build_ocr_writeback(
        doc_ir,
        ocr_tasks,
        ocr_results,
        min_confidence=min_confidence,
    )
    augmented_ir = payload.pop("augmented_document_ir")
    payload["artifacts"] = {
        "augmented_document_ir": _artifact_rel(augmented_ir_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    augmented_ir_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    augmented_ir_path.write_text(json.dumps(augmented_ir, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
