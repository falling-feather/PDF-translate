from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pdf_translate.vision.ocr_writeback import OCR_RESULTS_SCHEMA_VERSION, STRUCTURED_RESULT_FIELDS

SCHEMA_VERSION = "vlm-fallback-review-v1"
VLM_REVIEW_RESULT_SOURCE = "vlm_fallback_review"
DEFAULT_ACCEPTED_CONFIDENCE = 0.85
STRUCTURED_REVIEW_FIELDS = (
    *STRUCTURED_RESULT_FIELDS,
    "row_count",
    "column_count",
    "layout_notes",
    "caption_or_image_text",
)
HUMAN_DECISIONS = {"", "accept_result", "mark_unusable", "needs_revision"}


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _tasks(vlm_tasks: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(vlm_tasks, dict):
        return []
    raw = vlm_tasks.get("tasks")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _string_list(value: Any, *, limit: int = 120) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return 0
    return 0


def _as_float(value: Any, *, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_vlm_fallback_review_human_decision(value: Any) -> str:
    decision = str(value or "").strip().lower()
    aliases = {
        "": "",
        "clear": "",
        "reset": "",
        "accept": "accept_result",
        "accepted": "accept_result",
        "approve": "accept_result",
        "approved": "accept_result",
        "confirm": "accept_result",
        "confirmed": "accept_result",
        "accept_result": "accept_result",
        "mark_unusable": "mark_unusable",
        "unusable": "mark_unusable",
        "reject": "mark_unusable",
        "rejected": "mark_unusable",
        "skip": "mark_unusable",
        "skipped": "mark_unusable",
        "needs_revision": "needs_revision",
        "manual_review_required": "needs_revision",
        "revise": "needs_revision",
        "revision": "needs_revision",
    }
    normalized = aliases.get(decision, decision)
    if normalized not in HUMAN_DECISIONS:
        raise ValueError(f"unsupported VLM fallback review decision: {value}")
    return normalized


def _normalized_confidence(value: Any) -> float:
    confidence = _as_float(value, default=DEFAULT_ACCEPTED_CONFIDENCE)
    if confidence < 0 or confidence > 1:
        raise ValueError("review confidence must be between 0 and 1")
    return round(confidence, 4)


def _structured_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, Any] = {}
    for key in STRUCTURED_REVIEW_FIELDS:
        raw = value.get(key)
        if raw is None or raw == "":
            continue
        if isinstance(raw, (dict, list)):
            out[key] = _json_copy(raw)
        elif isinstance(raw, (str, int, float)) and not isinstance(raw, bool):
            out[key] = raw
    return out


def _default_decision(task: dict[str, Any]) -> str:
    if str(task.get("status") or "") == "blocked_missing_visual_evidence":
        return "blocked_missing_visual_evidence"
    return "needs_manual_result"


def _review_status(review: dict[str, Any]) -> str:
    decision = str(review.get("human_decision") or "")
    if decision == "accept_result":
        return "accepted_result"
    if decision == "mark_unusable":
        return "marked_unusable"
    if decision == "needs_revision":
        return "needs_revision"
    if str(review.get("source_status") or "") == "blocked_missing_visual_evidence":
        return "blocked_missing_visual_evidence"
    return "pending_review"


def _ready_for_writeback(review: dict[str, Any]) -> bool:
    return (
        str(review.get("human_decision") or "") == "accept_result"
        and bool(str(review.get("review_text") or "").strip())
        and bool(str(review.get("source_ocr_task_id") or "").strip())
    )


def _review_from_task(task: dict[str, Any]) -> dict[str, Any]:
    review_id = str(task.get("task_id") or task.get("source_ocr_task_id") or "").strip()
    item: dict[str, Any] = {
        "review_id": review_id,
        "vlm_task_id": str(task.get("task_id") or ""),
        "source_ocr_task_id": str(task.get("source_ocr_task_id") or ""),
        "doc_id": str(task.get("doc_id") or ""),
        "page_no": _as_int(task.get("page_no")),
        "scope": str(task.get("scope") or ""),
        "layout_scope": str(task.get("layout_scope") or ""),
        "source_status": str(task.get("status") or ""),
        "priority": str(task.get("priority") or ""),
        "block_id": str(task.get("block_id") or ""),
        "block_type": str(task.get("block_type") or ""),
        "target_structure_type": str(task.get("target_structure_type") or ""),
        "trigger_reasons": _string_list(task.get("trigger_reasons")),
        "source_stages": _string_list(task.get("source_stages"), limit=40),
        "route_action": str(task.get("route_action") or ""),
        "route_reasons": _string_list(task.get("route_reasons")),
        "review_goals": _string_list(task.get("review_goals")),
        "expected_outputs": _string_list(task.get("expected_outputs")),
        "input_path": str(task.get("input_path") or ""),
        "page_preview_path": str(task.get("page_preview_path") or ""),
        "bbox": list(task.get("bbox") or []),
        "visual_evidence": _json_copy(task.get("visual_evidence") or {}),
        "source_evidence": _json_copy(task.get("source_evidence") or []),
        "default_decision": _default_decision(task),
        "human_decision": "",
        "human_comment": "",
        "reviewed_by": "",
        "reviewed_at": "",
        "review_text": "",
        "review_confidence": None,
        "review_language": "",
        "structured_result": {},
    }
    for key in ("table_context", "formula_context", "structure_contract", "writeback", "fallback_policy"):
        value = task.get(key)
        if isinstance(value, dict):
            item[key] = _json_copy(value)
    item["effective_status"] = _review_status(item)
    item["ready_for_writeback"] = _ready_for_writeback(item)
    return item


def _refresh_vlm_fallback_review_summary(report: dict[str, Any]) -> dict[str, Any]:
    reviews = [item for item in report.get("reviews") or [] if isinstance(item, dict)]
    status_counts: Counter[str] = Counter()
    priority_counts: Counter[str] = Counter()
    block_type_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    trigger_reason_counts: Counter[str] = Counter()
    expected_output_counts: Counter[str] = Counter()
    review_goal_counts: Counter[str] = Counter()
    ready_for_writeback_count = 0
    structured_result_count = 0
    human_reviewed_count = 0

    for review in reviews:
        review["effective_status"] = _review_status(review)
        review["ready_for_writeback"] = _ready_for_writeback(review)
        status = str(review.get("effective_status") or "unknown")
        decision = str(review.get("human_decision") or "")
        status_counts[status] += 1
        decision_counts[decision or "pending"] += 1
        priority_counts[str(review.get("priority") or "unknown")] += 1
        block_type_counts[str(review.get("block_type") or "unknown")] += 1
        trigger_reason_counts.update(str(item) for item in review.get("trigger_reasons") or [] if str(item))
        expected_output_counts.update(str(item) for item in review.get("expected_outputs") or [] if str(item))
        review_goal_counts.update(str(item) for item in review.get("review_goals") or [] if str(item))
        if review.get("ready_for_writeback"):
            ready_for_writeback_count += 1
        if isinstance(review.get("structured_result"), dict) and review.get("structured_result"):
            structured_result_count += 1
        if decision in {"accept_result", "mark_unusable", "needs_revision"}:
            human_reviewed_count += 1

    pending_review_count = status_counts.get("pending_review", 0)
    needs_revision_count = status_counts.get("needs_revision", 0)
    blocked_count = status_counts.get("blocked_missing_visual_evidence", 0)
    summary = {
        "review_count": len(reviews),
        "source_task_count": len(reviews),
        "review_required_count": pending_review_count + needs_revision_count,
        "pending_review_count": pending_review_count,
        "blocked_by_missing_visual_evidence_count": blocked_count,
        "human_reviewed_count": human_reviewed_count,
        "accepted_result_count": status_counts.get("accepted_result", 0),
        "marked_unusable_count": status_counts.get("marked_unusable", 0),
        "needs_revision_count": needs_revision_count,
        "ready_for_writeback_count": ready_for_writeback_count,
        "structured_result_count": structured_result_count,
        "result_schema_version": OCR_RESULTS_SCHEMA_VERSION,
        "status_counts": dict(status_counts),
        "decision_counts": dict(decision_counts),
        "priority_counts": dict(priority_counts),
        "block_type_counts": dict(block_type_counts),
        "trigger_reason_counts": dict(trigger_reason_counts),
        "expected_output_counts": dict(expected_output_counts),
        "review_goal_counts": dict(review_goal_counts),
    }
    report["summary"] = summary
    return summary


def build_vlm_fallback_review(vlm_tasks: dict[str, Any]) -> dict[str, Any]:
    reviews = [_review_from_task(task) for task in _tasks(vlm_tasks)]
    report = {
        "schema_version": SCHEMA_VERSION,
        "doc_id": str(vlm_tasks.get("doc_id") or ""),
        "source_pdf": str(vlm_tasks.get("source_pdf") or ""),
        "source_schema_version": str(vlm_tasks.get("schema_version") or ""),
        "result_contract": _json_copy(vlm_tasks.get("result_contract") or {}),
        "reviews": reviews,
    }
    _refresh_vlm_fallback_review_summary(report)
    return report


def _md_cell(value: Any) -> str:
    text = str(value if value is not None else "").replace("\n", " ").replace("|", "\\|").strip()
    return text or "-"


def vlm_fallback_review_to_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# VLM Fallback Review",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Review items | {summary.get('review_count', 0)} |",
        f"| Pending review | {summary.get('pending_review_count', 0)} |",
        f"| Needs revision | {summary.get('needs_revision_count', 0)} |",
        f"| Accepted results | {summary.get('accepted_result_count', 0)} |",
        f"| Ready for OCR writeback | {summary.get('ready_for_writeback_count', 0)} |",
        f"| Missing visual evidence | {summary.get('blocked_by_missing_visual_evidence_count', 0)} |",
        "",
        "| Review ID | Page | Type | Status | Decision | Source OCR task | Result excerpt | Notes |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for review in report.get("reviews") or []:
        if not isinstance(review, dict):
            continue
        notes = "; ".join(_string_list(review.get("trigger_reasons"), limit=6))
        result_excerpt = str(review.get("review_text") or "").strip()
        if len(result_excerpt) > 160:
            result_excerpt = result_excerpt[:157] + "..."
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(review.get("review_id")),
                    _md_cell(review.get("page_no")),
                    _md_cell(review.get("target_structure_type") or review.get("block_type")),
                    _md_cell(review.get("effective_status")),
                    _md_cell(review.get("human_decision") or review.get("default_decision")),
                    _md_cell(review.get("source_ocr_task_id")),
                    _md_cell(result_excerpt),
                    _md_cell(notes),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_vlm_fallback_review(
    vlm_tasks: dict[str, Any],
    json_path: Path,
    markdown_path: Path | None = None,
) -> dict[str, Any]:
    report = build_vlm_fallback_review(vlm_tasks)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(vlm_fallback_review_to_markdown(report), encoding="utf-8")
    return report


def _load_review_report(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("VLM fallback review report must be a JSON object")
    return raw


def _review_lookup(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for item in report.get("reviews") or []:
        if not isinstance(item, dict):
            continue
        review_id = str(item.get("review_id") or "").strip()
        if review_id:
            lookup[review_id] = item
    return lookup


def _normalise_review_ids(review_ids: Any) -> list[str]:
    if not isinstance(review_ids, list):
        raise ValueError("review_ids must be a list")
    out: list[str] = []
    seen: set[str] = set()
    for item in review_ids:
        review_id = str(item or "").strip()
        if review_id and review_id not in seen:
            seen.add(review_id)
            out.append(review_id)
    if not out:
        raise ValueError("review_ids must not be empty")
    if len(out) > 500:
        raise ValueError("at most 500 VLM fallback review items can be updated at once")
    return out


def apply_vlm_fallback_review_decision(
    report: dict[str, Any],
    review_id: str,
    *,
    decision: Any,
    reviewer: str = "",
    comment: str = "",
    text: Any = None,
    confidence: Any = None,
    language: Any = "",
    structured_result: Any = None,
    reviewed_at: str | None = None,
) -> dict[str, Any]:
    normalized = normalize_vlm_fallback_review_human_decision(decision)
    lookup = _review_lookup(report)
    item = lookup.get(str(review_id or "").strip())
    if item is None:
        raise KeyError(review_id)

    if not normalized:
        item["human_decision"] = ""
        item["human_comment"] = ""
        item["reviewed_by"] = ""
        item["reviewed_at"] = ""
        item["review_text"] = ""
        item["review_confidence"] = None
        item["review_language"] = ""
        item["structured_result"] = {}
        _refresh_vlm_fallback_review_summary(report)
        return report

    item["human_decision"] = normalized
    item["human_comment"] = str(comment or "").strip()
    item["reviewed_by"] = str(reviewer or "").strip()
    item["reviewed_at"] = reviewed_at or _now_iso()

    if normalized == "accept_result":
        review_text = str(text if text is not None else item.get("review_text") or "").strip()
        if not review_text:
            raise ValueError("review_text is required when accepting a VLM fallback result")
        item["review_text"] = review_text
        item["review_confidence"] = _normalized_confidence(confidence)
        item["review_language"] = str(language or item.get("review_language") or "unknown").strip() or "unknown"
        item["structured_result"] = _structured_result(structured_result)
    else:
        if text is not None:
            item["review_text"] = str(text or "").strip()
        if confidence is not None:
            item["review_confidence"] = _normalized_confidence(confidence)
        if language:
            item["review_language"] = str(language).strip()
        if structured_result is not None:
            item["structured_result"] = _structured_result(structured_result)

    _refresh_vlm_fallback_review_summary(report)
    return report


def apply_vlm_fallback_review_batch_decision(
    report: dict[str, Any],
    review_ids: Any,
    *,
    decision: Any,
    reviewer: str = "",
    comment: str = "",
    reviewed_at: str | None = None,
) -> dict[str, Any]:
    normalized = normalize_vlm_fallback_review_human_decision(decision)
    if normalized == "accept_result":
        raise ValueError("batch accept_result requires item-level review_text; update items one by one")
    ids = _normalise_review_ids(review_ids)
    lookup = _review_lookup(report)
    missing = [review_id for review_id in ids if review_id not in lookup]
    if missing:
        raise KeyError(missing[0])
    for review_id in ids:
        apply_vlm_fallback_review_decision(
            report,
            review_id,
            decision=normalized,
            reviewer=reviewer,
            comment=comment,
            reviewed_at=reviewed_at,
        )
    _refresh_vlm_fallback_review_summary(report)
    return report


def _write_report(report: dict[str, Any], json_path: Path, markdown_path: Path | None) -> dict[str, Any]:
    _refresh_vlm_fallback_review_summary(report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(vlm_fallback_review_to_markdown(report), encoding="utf-8")
    return report


def write_vlm_fallback_review_decision(
    json_path: Path,
    markdown_path: Path | None,
    review_id: str,
    *,
    decision: Any,
    reviewer: str = "",
    comment: str = "",
    text: Any = None,
    confidence: Any = None,
    language: Any = "",
    structured_result: Any = None,
) -> dict[str, Any]:
    report = _load_review_report(json_path)
    apply_vlm_fallback_review_decision(
        report,
        review_id,
        decision=decision,
        reviewer=reviewer,
        comment=comment,
        text=text,
        confidence=confidence,
        language=language,
        structured_result=structured_result,
    )
    return _write_report(report, json_path, markdown_path)


def write_vlm_fallback_review_batch_decision(
    json_path: Path,
    markdown_path: Path | None,
    review_ids: Any,
    *,
    decision: Any,
    reviewer: str = "",
    comment: str = "",
) -> dict[str, Any]:
    report = _load_review_report(json_path)
    apply_vlm_fallback_review_batch_decision(
        report,
        review_ids,
        decision=decision,
        reviewer=reviewer,
        comment=comment,
    )
    return _write_report(report, json_path, markdown_path)


def _result_from_review(review: dict[str, Any]) -> dict[str, Any] | None:
    if not _ready_for_writeback(review):
        return None
    structured = _structured_result(review.get("structured_result"))
    warnings = ["manual_vlm_review", f"source_vlm_review_id:{review.get('review_id') or ''}"]
    result: dict[str, Any] = {
        "task_id": str(review.get("source_ocr_task_id") or ""),
        "status": "succeeded",
        "text": str(review.get("review_text") or "").strip(),
        "confidence": _normalized_confidence(review.get("review_confidence")),
        "engine": "manual_vlm_review",
        "language": str(review.get("review_language") or "unknown").strip() or "unknown",
        "bbox": list(review.get("bbox") or []),
        "warnings": warnings,
        "page_no": _as_int(review.get("page_no")),
        "block_id": str(review.get("block_id") or ""),
        "input_path": str(review.get("input_path") or ""),
        "page_preview_path": str(review.get("page_preview_path") or ""),
    }
    result.update(structured)
    return result


def build_vlm_review_ocr_results(report: dict[str, Any]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    structured_field_counts: Counter[str] = Counter()
    for review in report.get("reviews") or []:
        if not isinstance(review, dict):
            continue
        result = _result_from_review(review)
        if result is None:
            continue
        results.append(result)
        for key in STRUCTURED_REVIEW_FIELDS:
            if key in result:
                structured_field_counts[key] += 1
    summary = {
        "result_count": len(results),
        "accepted_review_count": len(results),
        "source_review_count": len([item for item in report.get("reviews") or [] if isinstance(item, dict)]),
        "structured_result_field_counts": dict(structured_field_counts),
    }
    return {
        "schema_version": OCR_RESULTS_SCHEMA_VERSION,
        "doc_id": str(report.get("doc_id") or ""),
        "source": VLM_REVIEW_RESULT_SOURCE,
        "source_schema_version": str(report.get("schema_version") or ""),
        "summary": summary,
        "results": results,
    }


def write_vlm_review_ocr_results(report: dict[str, Any], path: Path) -> dict[str, Any]:
    payload = build_vlm_review_ocr_results(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
