from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from pdf_translate.extractors.document_ir import DocumentIR

SCHEMA_VERSION = "ocr-task-manifest-v1"
OCR_ROUTE_ACTIONS = {"local_ocr", "vlm_review"}


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


def _known_blocks(doc_ir: DocumentIR) -> set[str]:
    return {block.block_id for page in doc_ir.pages for block in page.blocks}


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


def _writeback_target(page_no: int, block_id: str) -> dict[str, Any]:
    if block_id:
        return {
            "target": "document_ir.block.meta.ocr_candidates",
            "page_no": page_no,
            "block_id": block_id,
            "merge_policy": "append_only_until_qa",
        }
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
    known_block_ids: set[str],
) -> dict[str, Any]:
    page_no = int(page.get("page_no") or 0)
    evidence = page.get("evidence") if isinstance(page.get("evidence"), dict) else {}
    block_id = str(crop.get("block_id") or "")
    block_type = str(crop.get("block_type") or "region")
    action = str(page.get("action") or "")
    risk_level = str(page.get("risk_level") or "low")
    input_path = str(crop.get("crop_path") or "")
    return {
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
        "block_known_in_document_ir": block_id in known_block_ids,
        "bbox": list(crop.get("bbox") or []),
        "crop_width": int(crop.get("crop_width") or 0),
        "crop_height": int(crop.get("crop_height") or 0),
        "writeback": _writeback_target(page_no, block_id),
    }


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
        "block_known_in_document_ir": False,
        "bbox": [],
        "crop_width": 0,
        "crop_height": 0,
        "writeback": _writeback_target(page_no, ""),
    }


def build_ocr_task_manifest(
    doc_ir: DocumentIR,
    vision_route: dict[str, Any],
) -> dict[str, Any]:
    """Turn OCR/VLM route evidence into engine-agnostic pending OCR tasks."""
    route_pages = _route_page_index(vision_route)
    known_block_ids = _known_blocks(doc_ir)
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
                        known_block_ids=known_block_ids,
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
    fallback_count = sum(1 for task in tasks if str(task.get("fallback_engine") or ""))

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
            "scope_counts": dict(scope_counts),
            "status_counts": dict(status_counts),
            "priority_counts": dict(priority_counts),
            "recommended_engine_counts": dict(engine_counts),
            "block_type_counts": dict(block_type_counts),
            "route_action_counts": dict(route_action_counts),
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
