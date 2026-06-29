from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from pdf_translate.extractors.document_ir import DocumentIR

SCHEMA_VERSION = "ocr-writeback-v1"
OCR_RESULTS_SCHEMA_VERSION = "ocr-results-v1"
DEFAULT_MIN_CONFIDENCE = 0.5
SUCCESS_STATUSES = {"ok", "success", "succeeded", "completed", "done"}


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
    return {
        "source": "ocr_result",
        "task_id": str(task.get("task_id") or ""),
        "page_no": int(task.get("page_no") or 0),
        "block_id": str(task.get("block_id") or ""),
        "scope": str(task.get("scope") or ""),
        "text": text,
        "confidence": round(confidence, 4),
        "engine": str(result.get("engine") or task.get("recommended_engine") or ""),
        "language": str(result.get("language") or "unknown"),
        "bbox": bbox,
        "warnings": _normalized_warnings(result.get("warnings")),
        "input_path": str(task.get("input_path") or ""),
        "result_status": str(result.get("status") or "succeeded"),
    }


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
        writebacks.append(
            {
                "task_id": task_id,
                "page_no": page_no,
                "block_id": block_id,
                "target": f"document_ir.{target_kind}.meta.ocr_candidates",
                "candidate_index": candidate_index,
                "text_char_count": len(text),
                "confidence": candidate["confidence"],
                "engine": candidate["engine"],
            }
        )

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
