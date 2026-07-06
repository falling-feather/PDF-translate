from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "vlm-fallback-tasks-v1"
VLM_RESULT_SCHEMA_VERSION = "ocr-results-v1"

OCR_REJECTION_REASONS = {
    "result_not_succeeded",
    "empty_text",
    "low_confidence",
    "writeback_target_missing",
    "task_not_ready_for_writeback",
}
VISUAL_REVIEW_BLOCK_TYPES = {"image", "caption", "table", "formula", "page"}
VISUAL_REVIEW_REASONS = {
    "image_caption_context",
    "possible_image_table",
    "formula_dense_low_text",
    "low_text_image_heavy_page",
}
STRUCTURED_GATE_STATUSES = {"needs_review", "blocked"}


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _tasks(ocr_tasks: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(ocr_tasks, dict):
        return []
    raw = ocr_tasks.get("tasks")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


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
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def _eligible_for_vlm(task: dict[str, Any]) -> bool:
    block_type = str(task.get("block_type") or "")
    target_structure_type = str(task.get("target_structure_type") or "")
    layout_scope = str(task.get("layout_scope") or "")
    route_reasons = set(_string_list(task.get("reasons")))
    if str(task.get("fallback_engine") or "") == "vlm_review":
        return True
    if block_type in VISUAL_REVIEW_BLOCK_TYPES or target_structure_type in VISUAL_REVIEW_BLOCK_TYPES:
        return True
    if layout_scope in {"table_region", "formula_region", "image_region", "caption_region", "page"}:
        return True
    return bool(route_reasons & VISUAL_REVIEW_REASONS)


def _expected_outputs(task: dict[str, Any]) -> list[str]:
    block_type = str(task.get("block_type") or "")
    target_structure_type = str(task.get("target_structure_type") or block_type)
    outputs = ["plain_text", "confidence", "warnings", "layout_notes"]
    if block_type == "table" or target_structure_type == "table":
        outputs.extend(["structured_cells", "row_count", "column_count", "cell_bboxes", "merged_cell_candidates"])
    if block_type == "formula" or target_structure_type == "formula":
        outputs.extend(["formula_latex", "formula_tokens", "equation_labels", "formula_confidence"])
    if block_type in {"image", "caption"} or "image_caption_context" in set(_string_list(task.get("reasons"))):
        outputs.append("caption_or_image_text")
    return list(dict.fromkeys(outputs))


def _review_goals(task: dict[str, Any], reasons: list[str]) -> list[str]:
    block_type = str(task.get("block_type") or "")
    target_structure_type = str(task.get("target_structure_type") or block_type)
    route_reasons = set(_string_list(task.get("reasons")))
    trigger_reasons = set(reasons)
    goals = ["recover_visible_text"]
    if block_type == "table" or target_structure_type == "table" or "possible_image_table" in route_reasons:
        goals.append("recover_table_grid_and_cells")
    if block_type == "formula" or target_structure_type == "formula" or "formula_dense_low_text" in route_reasons:
        goals.append("recover_formula_latex_and_tokens")
    if block_type in {"image", "caption"} or "image_caption_context" in route_reasons:
        goals.append("verify_image_caption_relationship")
    if trigger_reasons & {"low_confidence", "empty_text", "result_not_succeeded", "missing_ocr_result"}:
        goals.append("repair_local_ocr_failure")
    if any(reason.startswith("structured_table_") for reason in trigger_reasons):
        goals.append("repair_structured_table_gate")
    if any(reason.startswith("structured_formula_") for reason in trigger_reasons):
        goals.append("repair_structured_formula_gate")
    return list(dict.fromkeys(goals))


def _base_issue(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_ocr_task_id": str(task.get("task_id") or ""),
        "task": task,
        "trigger_reasons": [],
        "source_stages": [],
        "evidence": [],
    }


def _add_issue(
    issues: dict[str, dict[str, Any]],
    task: dict[str, Any],
    *,
    reason: str,
    source_stage: str,
    evidence: dict[str, Any] | None = None,
) -> None:
    if not _eligible_for_vlm(task):
        return
    task_id = str(task.get("task_id") or "")
    if not task_id:
        return
    issue = issues.setdefault(task_id, _base_issue(task))
    _append_unique(issue["trigger_reasons"], reason)
    _append_unique(issue["source_stages"], source_stage)
    if evidence:
        issue["evidence"].append(_json_copy(evidence))


def _iter_rejected_results(ocr_writeback: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(ocr_writeback, dict):
        return []
    raw = ocr_writeback.get("rejected_results")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _iter_pending_tasks(ocr_writeback: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(ocr_writeback, dict):
        return []
    raw = ocr_writeback.get("pending_tasks")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _iter_candidates(ocr_candidate_qa: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(ocr_candidate_qa, dict):
        return []
    raw = ocr_candidate_qa.get("candidates")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _candidate_gate_reasons(candidate: dict[str, Any]) -> list[str]:
    status = str(candidate.get("status") or "")
    reasons: list[str] = []
    if status == "needs_review":
        reasons.append("ocr_candidate_needs_review")
    elif status == "blocked":
        reasons.append("ocr_candidate_blocked")
    else:
        return []
    for key, prefix in (
        ("structured_table_gate", "structured_table"),
        ("structured_formula_gate", "structured_formula"),
    ):
        gate = candidate.get(key) if isinstance(candidate.get(key), dict) else {}
        gate_status = str(gate.get("status") or "")
        if gate_status in STRUCTURED_GATE_STATUSES:
            reasons.append(f"{prefix}_{gate_status}")
            reasons.extend(_string_list(gate.get("issues"), limit=40))
            reasons.extend(_string_list(gate.get("blockers"), limit=40))
    reasons.extend(_string_list(candidate.get("blockers"), limit=40))
    reasons.extend(_string_list(candidate.get("reasons"), limit=40))
    return list(dict.fromkeys(reasons))


def _visual_evidence(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_path": str(task.get("input_path") or ""),
        "page_preview_path": str(task.get("page_preview_path") or ""),
        "bbox": list(task.get("bbox") or []),
        "crop_width": _as_int(task.get("crop_width")),
        "crop_height": _as_int(task.get("crop_height")),
    }


def _vlm_task_id(source_task_id: str) -> str:
    safe = "".join(ch if ch.isalnum() else "-" for ch in source_task_id).strip("-")
    return f"vlm-{safe or 'task'}"


def _priority(task: dict[str, Any], reasons: list[str]) -> str:
    if any(
        reason in reasons
        for reason in (
            "result_not_succeeded",
            "empty_text",
            "low_confidence",
            "missing_ocr_result",
            "ocr_candidate_blocked",
            "structured_table_blocked",
            "structured_formula_blocked",
        )
    ):
        return "P0"
    if any(reason.endswith("_needs_review") or reason == "ocr_candidate_needs_review" for reason in reasons):
        return "P1"
    return str(task.get("priority") or "P1")


def _fallback_task(issue: dict[str, Any]) -> dict[str, Any]:
    task = issue["task"]
    reasons = _string_list(issue.get("trigger_reasons"), limit=120)
    has_visual_evidence = bool(str(task.get("input_path") or "") or str(task.get("page_preview_path") or ""))
    status = "pending_vlm" if has_visual_evidence else "blocked_missing_visual_evidence"
    item: dict[str, Any] = {
        "task_id": _vlm_task_id(str(task.get("task_id") or "")),
        "source_ocr_task_id": str(task.get("task_id") or ""),
        "doc_id": str(task.get("doc_id") or ""),
        "page_no": _as_int(task.get("page_no")),
        "scope": str(task.get("scope") or ""),
        "layout_scope": str(task.get("layout_scope") or ""),
        "status": status,
        "priority": _priority(task, reasons),
        "block_id": str(task.get("block_id") or ""),
        "block_type": str(task.get("block_type") or ""),
        "target_structure_type": str(task.get("target_structure_type") or ""),
        "trigger_reasons": reasons,
        "source_stages": _string_list(issue.get("source_stages"), limit=20),
        "route_action": str(task.get("route_action") or ""),
        "route_reasons": _string_list(task.get("reasons"), limit=80),
        "recommended_engine": "vlm_fallback",
        "input_path": str(task.get("input_path") or ""),
        "page_preview_path": str(task.get("page_preview_path") or ""),
        "bbox": list(task.get("bbox") or []),
        "visual_evidence": _visual_evidence(task),
        "review_goals": _review_goals(task, reasons),
        "expected_outputs": _expected_outputs(task),
        "fallback_policy": {
            "trigger": "after_ocr_gate",
            "invoke_when": [
                "local_ocr_result_missing",
                "local_ocr_result_rejected",
                "ocr_candidate_blocked_or_needs_review",
                "structured_gate_requires_visual_review",
            ],
            "result_schema_version": VLM_RESULT_SCHEMA_VERSION,
            "engine": "vlm_fallback",
            "writeback_policy": "emit an OCR result for source_ocr_task_id; reuse OCR writeback and candidate QA gates",
        },
        "source_evidence": issue.get("evidence") or [],
    }
    for key in ("table_context", "formula_context", "structure_contract", "writeback"):
        value = task.get(key)
        if isinstance(value, dict):
            item[key] = _json_copy(value)
    return item


def build_vlm_fallback_tasks(
    ocr_tasks: dict[str, Any] | None,
    ocr_results: dict[str, Any] | list[dict[str, Any]] | None = None,
    ocr_writeback: dict[str, Any] | None = None,
    ocr_candidate_qa: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize VLM review tasks after OCR execution and QA gates without calling a VLM."""
    task_list = _tasks(ocr_tasks)
    task_index = {str(task.get("task_id") or ""): task for task in task_list if str(task.get("task_id") or "")}
    issues: dict[str, dict[str, Any]] = {}

    for pending in _iter_pending_tasks(ocr_writeback):
        task_id = str(pending.get("task_id") or "")
        task = task_index.get(task_id)
        if task is None:
            continue
        has_visual_evidence = bool(str(task.get("input_path") or "") or str(task.get("page_preview_path") or ""))
        reason = (
            "missing_visual_evidence"
            if not has_visual_evidence or str(task.get("status") or "") != "pending_engine"
            else "missing_ocr_result"
        )
        _add_issue(
            issues,
            task,
            reason=reason,
            source_stage="ocr_missing_result",
            evidence={"pending_task": pending},
        )

    for rejected in _iter_rejected_results(ocr_writeback):
        task_id = str(rejected.get("task_id") or "")
        task = task_index.get(task_id)
        if task is None:
            continue
        reason = str(rejected.get("reason") or "ocr_result_rejected")
        if reason not in OCR_REJECTION_REASONS:
            continue
        _add_issue(
            issues,
            task,
            reason=reason,
            source_stage="ocr_writeback_rejection",
            evidence={"rejected_result": rejected},
        )

    for candidate in _iter_candidates(ocr_candidate_qa):
        task_id = str(candidate.get("task_id") or "")
        task = task_index.get(task_id)
        if task is None:
            continue
        reasons = _candidate_gate_reasons(candidate)
        if not reasons:
            continue
        if not _eligible_for_vlm(task):
            continue
        issue = issues.setdefault(task_id, _base_issue(task))
        _append_unique(issue["source_stages"], "ocr_candidate_gate")
        issue["evidence"].append({"candidate": _json_copy(candidate)})
        for reason in reasons:
            _append_unique(issue["trigger_reasons"], reason)

    tasks = [_fallback_task(issue) for _task_id, issue in sorted(issues.items())]
    status_counts = Counter(str(task.get("status") or "unknown") for task in tasks)
    priority_counts = Counter(str(task.get("priority") or "unknown") for task in tasks)
    block_type_counts = Counter(str(task.get("block_type") or "unknown") for task in tasks)
    reason_counts: Counter[str] = Counter()
    source_stage_counts: Counter[str] = Counter()
    review_goal_counts: Counter[str] = Counter()
    expected_output_counts: Counter[str] = Counter()
    for task in tasks:
        reason_counts.update(str(reason) for reason in task.get("trigger_reasons") or [] if str(reason))
        source_stage_counts.update(str(stage) for stage in task.get("source_stages") or [] if str(stage))
        review_goal_counts.update(str(goal) for goal in task.get("review_goals") or [] if str(goal))
        expected_output_counts.update(str(output) for output in task.get("expected_outputs") or [] if str(output))

    return {
        "schema_version": SCHEMA_VERSION,
        "doc_id": str((ocr_tasks or {}).get("doc_id") or (ocr_writeback or {}).get("doc_id") or ""),
        "source_pdf": str((ocr_tasks or {}).get("source_pdf") or ""),
        "summary": {
            "task_count": len(tasks),
            "after_ocr_gate_task_count": len(tasks),
            "ready_task_count": status_counts.get("pending_vlm", 0),
            "blocked_by_missing_visual_evidence_count": status_counts.get("blocked_missing_visual_evidence", 0),
            "ocr_missing_result_task_count": reason_counts.get("missing_ocr_result", 0),
            "ocr_missing_visual_evidence_task_count": reason_counts.get("missing_visual_evidence", 0),
            "ocr_failed_result_task_count": reason_counts.get("result_not_succeeded", 0),
            "ocr_empty_text_task_count": reason_counts.get("empty_text", 0),
            "ocr_low_confidence_task_count": reason_counts.get("low_confidence", 0),
            "ocr_writeback_rejection_task_count": source_stage_counts.get("ocr_writeback_rejection", 0),
            "ocr_candidate_gate_task_count": source_stage_counts.get("ocr_candidate_gate", 0),
            "ocr_candidate_needs_review_task_count": reason_counts.get("ocr_candidate_needs_review", 0),
            "ocr_candidate_blocked_task_count": reason_counts.get("ocr_candidate_blocked", 0),
            "structured_gate_task_count": sum(
                count
                for reason, count in reason_counts.items()
                if reason.startswith("structured_table_") or reason.startswith("structured_formula_")
            ),
            "status_counts": dict(status_counts),
            "priority_counts": dict(priority_counts),
            "block_type_counts": dict(block_type_counts),
            "reason_counts": dict(reason_counts),
            "source_stage_counts": dict(source_stage_counts),
            "review_goal_counts": dict(review_goal_counts),
            "expected_output_counts": dict(expected_output_counts),
        },
        "result_contract": {
            "schema_version": VLM_RESULT_SCHEMA_VERSION,
            "engine": "vlm_fallback",
            "task_id_policy": "Return results with source_ocr_task_id so existing OCR writeback can apply them.",
            "required_fields": ["task_id", "text", "confidence", "engine", "language", "bbox", "warnings"],
            "optional_structured_fields": [
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
        },
        "tasks": tasks,
    }


def write_vlm_fallback_tasks(
    ocr_tasks: dict[str, Any] | None,
    ocr_results: dict[str, Any] | list[dict[str, Any]] | None,
    ocr_writeback: dict[str, Any] | None,
    ocr_candidate_qa: dict[str, Any] | None,
    path: Path,
) -> dict[str, Any]:
    payload = build_vlm_fallback_tasks(
        ocr_tasks,
        ocr_results,
        ocr_writeback,
        ocr_candidate_qa,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
