from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pdf_translate.extractors.document_ir import BlockIR, DocumentIR

SCHEMA_VERSION = "ocr-task-manifest-v1"
STRUCTURE_CONTRACT_SCHEMA_VERSION = "ocr-structure-contract-v1"
OCR_ROUTE_ACTIONS = {"local_ocr", "vlm_review"}
_FORMULA_TOKEN_RE = re.compile(
    r"(?:\([0-9]{1,3}[A-Za-z]?\)|[A-Za-z](?:_\{?[A-Za-z0-9,+\-]+\}?|\^\{?[A-Za-z0-9,+\-]+\}?)+|"
    r"\\[A-Za-z]+|[=+\-*/<>≤≥≈±∑∫√])"
)


def _route_page_index(vision_route: dict[str, Any]) -> dict[int, dict[str, Any]]:
    pages = vision_route.get("pages") if isinstance(vision_route, dict) else None
    out: dict[int, dict[str, Any]] = {}
    if not isinstance(pages, list):
        return out
    for page in pages:
        if not isinstance(page, dict):
            continue
        try:
            page_no = int(page.get("page_no") or 0)
        except (TypeError, ValueError):
            continue
        if page_no > 0:
            out[page_no] = page
    return out


def _block_index(doc_ir: DocumentIR) -> dict[str, BlockIR]:
    return {block.block_id: block for page in doc_ir.pages for block in page.blocks}


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


def _string_list(value: Any, *, limit: int = 80) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _target_structure_type(block_type: str) -> str:
    if block_type in {"table", "formula", "image", "caption"}:
        return block_type
    return "text"


def _layout_scope(scope: str, block_type: str) -> str:
    if scope == "page":
        return "page"
    if block_type in {"table", "formula", "image", "caption"}:
        return f"{block_type}_region"
    return "text_region"


def _table_context(block: BlockIR | None) -> dict[str, Any]:
    if block is None or block.type != "table":
        return {}
    meta = block.meta if isinstance(block.meta, dict) else {}
    table = meta.get("table") if isinstance(meta.get("table"), dict) else {}
    rows = table.get("rows") if isinstance(table.get("rows"), list) else []
    row_count = _as_int(table.get("row_count")) or len(rows)
    column_count = _as_int(table.get("column_count"))
    row_lengths = [len(row) for row in rows if isinstance(row, list)]
    if not column_count and row_lengths:
        column_count = max(row_lengths)
    table_scope = (
        "continued"
        if meta.get("table_continuation") or meta.get("continued_table_group_id")
        else "primary"
    )
    return {
        "table_id": block.block_id,
        "table_scope": table_scope,
        "table_block_type": block.type,
        "source_page_no": block.page_no,
        "source_block_bbox": [round(float(value), 2) for value in block.bbox],
        "row_count": row_count,
        "column_count": column_count,
        "cell_count": row_count * column_count if row_count and column_count else 0,
        "header": _string_list(table.get("header"), limit=40),
        "numeric_tokens": _string_list(table.get("numeric_tokens"), limit=120),
        "locked_tokens": _string_list(block.locked_tokens, limit=120),
        "source_confidence": str(table.get("confidence") or ""),
        "warnings": _string_list(table.get("warnings"), limit=80),
    }


def _table_subtarget(table_context: dict[str, Any]) -> dict[str, Any]:
    if not table_context:
        return {}
    return {
        "type": "table_block",
        "table_id": str(table_context.get("table_id") or ""),
        "table_scope": str(table_context.get("table_scope") or "primary"),
        "expected_granularity": "rows_and_cells",
    }


def _formula_context(block: BlockIR | None) -> dict[str, Any]:
    if block is None or block.type != "formula":
        return {}
    source_text = str(block.text or "").strip()
    return {
        "formula_id": block.block_id,
        "formula_block_type": block.type,
        "source_page_no": block.page_no,
        "source_block_bbox": [round(float(value), 2) for value in block.bbox],
        "source_text": source_text[:500],
        "source_tokens": _string_list(_FORMULA_TOKEN_RE.findall(source_text), limit=120),
        "locked_tokens": _string_list(block.locked_tokens, limit=120),
    }


def _formula_subtarget(formula_context: dict[str, Any]) -> dict[str, Any]:
    if not formula_context:
        return {}
    return {
        "type": "formula_block",
        "formula_id": str(formula_context.get("formula_id") or ""),
        "expected_granularity": "formula_text_latex_and_tokens",
    }


def _base_structure_contract(target_structure_type: str, expected_output: str) -> dict[str, Any]:
    return {
        "schema_version": STRUCTURE_CONTRACT_SCHEMA_VERSION,
        "target_structure_type": target_structure_type,
        "expected_output": expected_output,
        "required_result_fields": [
            "task_id",
            "text",
            "confidence",
            "engine",
            "language",
            "bbox",
            "warnings",
        ],
    }


def _structure_contract(
    block_type: str,
    table_context: dict[str, Any],
    formula_context: dict[str, Any],
) -> dict[str, Any]:
    if block_type == "table" and table_context:
        return {
            **_base_structure_contract("table", "plain_text_with_structured_table_cells"),
            "optional_result_fields": [
                "structured_cells",
                "row_count",
                "column_count",
                "cell_bboxes",
                "merged_cell_candidates",
                "table_footnotes",
            ],
            "qa_gates": [
                "preserve_row_column_grid",
                "preserve_locked_tokens",
                "preserve_numeric_tokens",
                "flag_merged_or_ragged_cells",
                "keep_table_footnotes_attached",
            ],
        }
    if block_type == "formula" and formula_context:
        return {
            **_base_structure_contract("formula", "plain_text_with_formula_latex_and_tokens"),
            "optional_result_fields": [
                "formula_latex",
                "formula_tokens",
                "equation_labels",
                "formula_confidence",
            ],
            "qa_gates": [
                "preserve_formula_tokens",
                "preserve_equation_labels",
                "flag_low_confidence_formula",
            ],
        }
    return {}


def _priority(action: str, risk_level: str, block_type: str) -> str:
    if action == "local_ocr" and (risk_level == "high" or block_type in {"table", "formula"}):
        return "P0"
    if action == "vlm_review" or risk_level == "medium" or block_type in {"image", "caption"}:
        return "P1"
    return "P2"


def _recommended_engine(block_type: str) -> str:
    if block_type == "table":
        return "local_table_ocr"
    if block_type == "formula":
        return "local_formula_ocr"
    return "local_ocr"


def _fallback_engine(action: str) -> str:
    return "vlm_review" if action == "vlm_review" else ""


def _writeback_target(page_no: int, block_id: str, subtarget: dict[str, Any] | None = None) -> dict[str, Any]:
    if block_id:
        target = {
            "target": "document_ir.block.meta.ocr_candidates",
            "page_no": page_no,
            "block_id": block_id,
            "merge_policy": "append_only_until_qa",
        }
    else:
        target = {
            "target": "document_ir.page.meta.ocr_candidates",
            "page_no": page_no,
            "block_id": "",
            "merge_policy": "append_only_until_qa",
        }
    if subtarget:
        target["subtarget"] = subtarget
    return target


def _base_task_payload(
    *,
    scope: str,
    page_no: int,
    block_id: str,
    block_type: str,
    block: BlockIR | None,
) -> dict[str, Any]:
    table_context = _table_context(block)
    formula_context = _formula_context(block)
    subtarget = _table_subtarget(table_context) or _formula_subtarget(formula_context)
    contract = _structure_contract(block_type, table_context, formula_context)
    payload: dict[str, Any] = {
        "layout_scope": _layout_scope(scope, block_type),
        "target_structure_type": _target_structure_type(block_type),
        "writeback": _writeback_target(page_no, block_id, subtarget),
    }
    if table_context:
        payload["table_context"] = table_context
    if formula_context:
        payload["formula_context"] = formula_context
    if contract:
        payload["structure_contract"] = contract
    return payload


def _page_writeback_target(page_no: int) -> dict[str, Any]:
    return {
        "target": "document_ir.page.meta.ocr_candidates",
        "page_no": page_no,
        "block_id": "",
        "merge_policy": "append_only_until_qa",
    }


def _task_status(input_path: str) -> str:
    return "pending_engine" if input_path else "blocked_missing_visual_evidence"


def _region_task(
    *,
    doc_id: str,
    page: dict[str, Any],
    crop: dict[str, Any],
    task_index: int,
    blocks_by_id: dict[str, BlockIR],
) -> dict[str, Any]:
    page_no = int(page.get("page_no") or 0)
    evidence = page.get("evidence") if isinstance(page.get("evidence"), dict) else {}
    block_id = str(crop.get("block_id") or "")
    block_type = str(crop.get("block_type") or "region")
    block = blocks_by_id.get(block_id)
    action = str(page.get("action") or "")
    risk_level = str(page.get("risk_level") or "low")
    input_path = str(crop.get("crop_path") or "")
    task = {
        "task_id": f"ocr-p{page_no:04d}-r{task_index:03d}",
        "doc_id": doc_id,
        "page_no": page_no,
        "scope": "region",
        "status": _task_status(input_path),
        "priority": _priority(action, risk_level, block_type),
        "route_action": action,
        "risk_level": risk_level,
        "risk_score": page.get("risk_score", 0),
        "reasons": list(page.get("reasons") or []),
        "recommended_engine": _recommended_engine(block_type),
        "fallback_engine": _fallback_engine(action),
        "input_path": input_path,
        "page_preview_path": str(evidence.get("page_preview_path") or ""),
        "block_id": block_id,
        "block_type": block_type,
        "block_known_in_document_ir": block_id in blocks_by_id,
        "bbox": list(crop.get("bbox") or []),
        "crop_width": int(crop.get("crop_width") or 0),
        "crop_height": int(crop.get("crop_height") or 0),
    }
    task.update(
        _base_task_payload(
            scope="region",
            page_no=page_no,
            block_id=block_id,
            block_type=block_type,
            block=block,
        )
    )
    return task


def _page_task(
    *,
    doc_id: str,
    page: dict[str, Any],
    task_index: int,
) -> dict[str, Any]:
    page_no = int(page.get("page_no") or 0)
    evidence = page.get("evidence") if isinstance(page.get("evidence"), dict) else {}
    action = str(page.get("action") or "")
    risk_level = str(page.get("risk_level") or "low")
    input_path = str(evidence.get("page_preview_path") or "")
    return {
        "task_id": f"ocr-p{page_no:04d}-page-{task_index:03d}",
        "doc_id": doc_id,
        "page_no": page_no,
        "scope": "page",
        "layout_scope": "page",
        "status": _task_status(input_path),
        "priority": _priority(action, risk_level, "page"),
        "route_action": action,
        "risk_level": risk_level,
        "risk_score": page.get("risk_score", 0),
        "reasons": list(page.get("reasons") or []),
        "recommended_engine": "local_ocr",
        "fallback_engine": _fallback_engine(action),
        "input_path": input_path,
        "page_preview_path": input_path,
        "block_id": "",
        "block_type": "page",
        "target_structure_type": "page",
        "block_known_in_document_ir": False,
        "bbox": [],
        "crop_width": 0,
        "crop_height": 0,
        "writeback": _page_writeback_target(page_no),
    }


def build_ocr_task_manifest(
    doc_ir: DocumentIR,
    vision_route: dict[str, Any],
) -> dict[str, Any]:
    """Turn OCR/VLM route evidence into engine-agnostic pending OCR tasks."""
    route_pages = _route_page_index(vision_route)
    blocks_by_id = _block_index(doc_ir)
    tasks: list[dict[str, Any]] = []
    skipped_pages: list[dict[str, Any]] = []

    for page_no in sorted(route_pages):
        page = route_pages[page_no]
        action = str(page.get("action") or "")
        if action not in OCR_ROUTE_ACTIONS:
            skipped_pages.append(
                {
                    "page_no": page_no,
                    "action": action,
                    "reason": "route_action_does_not_require_ocr_task",
                }
            )
            continue
        evidence = page.get("evidence") if isinstance(page.get("evidence"), dict) else {}
        crops = [item for item in evidence.get("region_crops") or [] if isinstance(item, dict)]
        if crops:
            for crop in crops:
                tasks.append(
                    _region_task(
                        doc_id=doc_ir.doc_id,
                        page=page,
                        crop=crop,
                        task_index=len(tasks),
                        blocks_by_id=blocks_by_id,
                    )
                )
        else:
            tasks.append(
                _page_task(
                    doc_id=doc_ir.doc_id,
                    page=page,
                    task_index=len(tasks),
                )
            )

    scope_counts = Counter(str(task.get("scope") or "unknown") for task in tasks)
    status_counts = Counter(str(task.get("status") or "unknown") for task in tasks)
    priority_counts = Counter(str(task.get("priority") or "unknown") for task in tasks)
    engine_counts = Counter(str(task.get("recommended_engine") or "unknown") for task in tasks)
    block_type_counts = Counter(str(task.get("block_type") or "unknown") for task in tasks)
    route_action_counts = Counter(str(task.get("route_action") or "unknown") for task in tasks)
    structure_target_counts = Counter(str(task.get("target_structure_type") or "unknown") for task in tasks)
    fallback_count = sum(1 for task in tasks if str(task.get("fallback_engine") or ""))
    structured_contract_count = sum(1 for task in tasks if isinstance(task.get("structure_contract"), dict))
    table_context_count = sum(1 for task in tasks if isinstance(task.get("table_context"), dict))
    table_context_ready_count = sum(
        1
        for task in tasks
        if isinstance(task.get("table_context"), dict) and str(task.get("status") or "") == "pending_engine"
    )
    formula_context_count = sum(1 for task in tasks if isinstance(task.get("formula_context"), dict))
    formula_context_ready_count = sum(
        1
        for task in tasks
        if isinstance(task.get("formula_context"), dict) and str(task.get("status") or "") == "pending_engine"
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "doc_id": doc_ir.doc_id,
        "source_pdf": doc_ir.source_pdf,
        "summary": {
            "task_count": len(tasks),
            "region_task_count": scope_counts.get("region", 0),
            "page_task_count": scope_counts.get("page", 0),
            "ready_task_count": status_counts.get("pending_engine", 0),
            "blocked_by_missing_evidence_count": status_counts.get("blocked_missing_visual_evidence", 0),
            "vlm_fallback_task_count": fallback_count,
            "structured_contract_task_count": structured_contract_count,
            "table_context_task_count": table_context_count,
            "table_context_ready_task_count": table_context_ready_count,
            "formula_context_task_count": formula_context_count,
            "formula_context_ready_task_count": formula_context_ready_count,
            "scope_counts": dict(scope_counts),
            "status_counts": dict(status_counts),
            "priority_counts": dict(priority_counts),
            "recommended_engine_counts": dict(engine_counts),
            "block_type_counts": dict(block_type_counts),
            "route_action_counts": dict(route_action_counts),
            "structure_target_counts": dict(structure_target_counts),
            "skipped_page_count": len(skipped_pages),
        },
        "result_writeback_contract": {
            "schema_version": "ocr-result-v1",
            "required_fields": [
                "task_id",
                "text",
                "confidence",
                "engine",
                "language",
                "bbox",
                "warnings",
            ],
            "optional_structured_fields": [
                "table_context",
                "formula_context",
                "subtarget",
                "structured_cells",
                "row_count",
                "column_count",
                "cell_bboxes",
                "merged_cell_candidates",
                "table_footnotes",
                "formula_latex",
                "formula_tokens",
                "equation_labels",
                "formula_confidence",
            ],
            "writeback_policy": "append OCR candidates to DocumentIR meta; do not replace source text before QA",
            "qa_gate": "OCR text must pass structure and token checks before entering translation chunks",
        },
        "tasks": tasks,
        "skipped_pages": skipped_pages,
    }


def write_ocr_task_manifest(
    doc_ir: DocumentIR,
    vision_route: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    manifest = build_ocr_task_manifest(doc_ir, vision_route)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
