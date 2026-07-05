from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "experiment-metrics-v1"

DEFAULT_EVIDENCE_FILES = {
    "structure_chunks_manifest": "output/structure_chunks_manifest.json",
    "structure_hints_manifest": "output/structure_hints_manifest.json",
    "structure_qa": "output/structure_qa.json",
    "table_reconstruction": "output/table_reconstruction.json",
    "vision_route": "output/vision_route.json",
    "ocr_tasks": "output/ocr_tasks.json",
    "ocr_results": "output/ocr_results.json",
    "ocr_writeback": "output/ocr_writeback.json",
    "ocr_candidate_qa": "output/ocr_candidate_qa.json",
    "ocr_candidate_promotion": "output/ocr_candidate_promotion.json",
    "document_ir_ocr": "output/document_ir_ocr.json",
    "document_ir_promoted": "output/document_ir_promoted.json",
    "chunk_boundary_qa": "output/chunk_boundary_qa.json",
    "chunk_strategy_comparison": "output/chunk_strategy_comparison.json",
    "translation_qa": "output/qa_report.json",
    "repair_plan": "output/repair_plan.json",
    "repair_requests": "output/repair_requests.json",
    "repair_results": "output/repair_results.json",
    "repair_validation": "output/repair_validation.json",
    "repair_merge": "output/repair_merge.json",
    "repair_patch_review": "output/repair_patch_review.json",
    "repair_merge_qa": "output/repair_merge_qa.json",
    "repair_publish": "output/repair_publish.json",
    "repair_published_full": "output/published_full.md",
    "translated_pdf": "output/translated_full.pdf",
    "translated_pdf_report": "output/translated_pdf_report.json",
    "run_metrics": "output/run_metrics.json",
    "run_log": "output/run_log.jsonl",
    "cost_estimate": "output/cost_estimate.json",
}


def _summary(report: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    summary = report.get("summary")
    return summary if isinstance(summary, dict) else {}


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


def _as_float(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _counter_dict(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): _as_int(value) for key, value in raw.items()}


def _rate(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def build_experiment_metrics(
    structure_qa: dict[str, Any] | None,
    vision_route: dict[str, Any] | None,
    translation_qa: dict[str, Any] | None,
    repair_plan: dict[str, Any] | None,
    *,
    doc_id: str | None = None,
    pipeline_variant: str | None = None,
    chunk_boundary_qa: dict[str, Any] | None = None,
    chunk_strategy_comparison: dict[str, Any] | None = None,
    structure_hints_manifest: dict[str, Any] | None = None,
    table_reconstruction: dict[str, Any] | None = None,
    ocr_tasks: dict[str, Any] | None = None,
    ocr_results: dict[str, Any] | None = None,
    ocr_writeback: dict[str, Any] | None = None,
    ocr_candidate_qa: dict[str, Any] | None = None,
    ocr_candidate_promotion: dict[str, Any] | None = None,
    repair_requests: dict[str, Any] | None = None,
    repair_results: dict[str, Any] | None = None,
    repair_validation: dict[str, Any] | None = None,
    repair_merge: dict[str, Any] | None = None,
    repair_patch_review: dict[str, Any] | None = None,
    repair_merge_qa: dict[str, Any] | None = None,
    repair_publish: dict[str, Any] | None = None,
    translated_pdf_report: dict[str, Any] | None = None,
    run_metrics: dict[str, Any] | None = None,
    cost_estimate: dict[str, Any] | None = None,
    evidence_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Aggregate pipeline QA artifacts into a patent-facing experiment summary."""
    structure_summary = _summary(structure_qa)
    table_reconstruction_summary = _summary(table_reconstruction)
    vision_summary = _summary(vision_route)
    ocr_task_summary = _summary(ocr_tasks)
    ocr_result_summary = _summary(ocr_results)
    ocr_execution = ocr_results.get("execution") if isinstance(ocr_results, dict) else None
    ocr_execution_summary = _summary(ocr_execution if isinstance(ocr_execution, dict) else None)
    ocr_writeback_summary = _summary(ocr_writeback)
    ocr_candidate_summary = _summary(ocr_candidate_qa)
    ocr_candidate_promotion_summary = _summary(ocr_candidate_promotion)
    chunk_boundary_summary = _summary(chunk_boundary_qa)
    chunk_strategy_summary = _summary(chunk_strategy_comparison)
    structure_hints_summary = _summary(structure_hints_manifest)
    translation_summary = _summary(translation_qa)
    repair_summary = _summary(repair_plan)
    repair_request_summary = _summary(repair_requests)
    repair_result_summary = _summary(repair_results)
    repair_validation_summary = _summary(repair_validation)
    repair_merge_summary = _summary(repair_merge)
    repair_patch_review_summary = _summary(repair_patch_review)
    repair_merge_qa_summary = _summary(repair_merge_qa)
    repair_publish_summary = _summary(repair_publish)
    translated_pdf_summary = _summary(translated_pdf_report)
    run_summary = _summary(run_metrics)
    cost_summary = _summary(cost_estimate)

    block_counts = _counter_dict(structure_summary.get("block_counts"))
    entity_type_counts = _counter_dict(structure_summary.get("entity_type_counts"))
    continuation_kind_counts = _counter_dict(structure_summary.get("continuation_kind_counts"))
    stitch_action_counts = _counter_dict(structure_summary.get("stitch_action_counts"))
    vision_action_counts = _counter_dict(vision_summary.get("action_counts"))
    vision_risk_counts = _counter_dict(vision_summary.get("risk_counts"))
    ocr_task_scope_counts = _counter_dict(ocr_task_summary.get("scope_counts"))
    ocr_task_status_counts = _counter_dict(ocr_task_summary.get("status_counts"))
    ocr_task_priority_counts = _counter_dict(ocr_task_summary.get("priority_counts"))
    ocr_task_engine_counts = _counter_dict(ocr_task_summary.get("recommended_engine_counts"))
    ocr_task_block_type_counts = _counter_dict(ocr_task_summary.get("block_type_counts"))
    ocr_task_structure_target_counts = _counter_dict(ocr_task_summary.get("structure_target_counts"))
    ocr_result_payload_status_counts = _counter_dict(ocr_result_summary.get("status_counts"))
    ocr_result_payload_engine_counts = _counter_dict(ocr_result_summary.get("engine_counts"))
    ocr_execution_status_counts = _counter_dict(ocr_execution_summary.get("status_counts"))
    ocr_writeback_status_counts = _counter_dict(ocr_writeback_summary.get("result_status_counts"))
    ocr_writeback_engine_counts = _counter_dict(ocr_writeback_summary.get("accepted_engine_counts"))
    ocr_writeback_rejection_counts = _counter_dict(ocr_writeback_summary.get("rejection_reason_counts"))
    ocr_writeback_structured_result_field_counts = _counter_dict(
        ocr_writeback_summary.get("structured_result_field_counts")
    )
    ocr_candidate_status_counts = _counter_dict(ocr_candidate_summary.get("status_counts"))
    ocr_candidate_issue_counts = _counter_dict(ocr_candidate_summary.get("issue_counts"))
    ocr_candidate_structured_result_field_counts = _counter_dict(
        ocr_candidate_summary.get("structured_result_field_counts")
    )
    ocr_candidate_structured_table_gate_counts = _counter_dict(
        ocr_candidate_summary.get("structured_table_gate_counts")
    )
    ocr_candidate_structured_table_gate_issue_counts = _counter_dict(
        ocr_candidate_summary.get("structured_table_gate_issue_counts")
    )
    ocr_candidate_structured_formula_gate_counts = _counter_dict(
        ocr_candidate_summary.get("structured_formula_gate_counts")
    )
    ocr_candidate_structured_formula_gate_issue_counts = _counter_dict(
        ocr_candidate_summary.get("structured_formula_gate_issue_counts")
    )
    ocr_candidate_promotion_status_counts = _counter_dict(
        ocr_candidate_promotion_summary.get("candidate_status_counts")
    )
    ocr_candidate_promotion_skip_counts = _counter_dict(ocr_candidate_promotion_summary.get("skip_reason_counts"))
    translation_issue_counts = _counter_dict(translation_summary.get("issue_counts"))
    translation_severity_counts = _counter_dict(translation_summary.get("severity_counts"))
    repair_action_counts = _counter_dict(repair_summary.get("action_counts"))
    repair_priority_counts = _counter_dict(repair_summary.get("priority_counts"))
    repair_scope_counts = _counter_dict(repair_summary.get("scope_counts"))
    run_breakdowns = run_metrics.get("breakdowns") if isinstance(run_metrics, dict) else {}
    stage_elapsed_ms = _counter_dict(run_summary.get("stage_elapsed_ms"))
    stage_counts = _counter_dict((run_breakdowns or {}).get("stage_counts"))
    translator_counts = _counter_dict((run_breakdowns or {}).get("translator_counts"))
    skip_reasons = _counter_dict((run_breakdowns or {}).get("skip_reasons"))
    run_error_code_counts = _counter_dict(run_summary.get("error_code_counts")) or _counter_dict(
        (run_breakdowns or {}).get("error_code_counts")
    )
    run_error_category_counts = _counter_dict(run_summary.get("error_category_counts")) or _counter_dict(
        (run_breakdowns or {}).get("error_category_counts")
    )

    page_count = _as_int(structure_summary.get("page_count")) or _as_int(vision_summary.get("page_count"))
    table_count = _as_int(structure_summary.get("table_count")) or _as_int(
        table_reconstruction_summary.get("table_count")
    )
    reconstructable_table_count = _as_int(table_reconstruction_summary.get("reconstructable_table_count"))
    low_confidence_table_count = _as_int(table_reconstruction_summary.get("low_confidence_table_count"))
    table_cell_count = _as_int(table_reconstruction_summary.get("cell_count"))
    table_empty_cell_count = _as_int(table_reconstruction_summary.get("empty_cell_count"))
    table_numeric_cell_count = _as_int(table_reconstruction_summary.get("numeric_cell_count"))
    table_numeric_token_count = _as_int(table_reconstruction_summary.get("numeric_token_count"))
    table_unit_token_count = _as_int(table_reconstruction_summary.get("unit_token_count"))
    table_significance_token_count = _as_int(table_reconstruction_summary.get("significance_token_count"))
    table_ragged_table_count = _as_int(table_reconstruction_summary.get("ragged_table_count"))
    table_ragged_row_count = _as_int(table_reconstruction_summary.get("ragged_row_count"))
    table_merged_cell_candidate_count = _as_int(
        table_reconstruction_summary.get("merged_cell_candidate_count")
    )
    table_merged_cell_candidate_type_counts = _counter_dict(
        table_reconstruction_summary.get("merged_cell_candidate_type_counts")
    )
    table_merged_cell_candidate_reason_counts = _counter_dict(
        table_reconstruction_summary.get("merged_cell_candidate_reason_counts")
    )
    table_merged_cell_candidate_status_counts = _counter_dict(
        table_reconstruction_summary.get("merged_cell_candidate_status_counts")
    )
    table_merged_cell_candidate_visual_evidence_counts = _counter_dict(
        table_reconstruction_summary.get("merged_cell_candidate_visual_evidence_counts")
    )
    table_merged_cell_candidate_bbox_evidence_counts = _counter_dict(
        table_reconstruction_summary.get("merged_cell_candidate_bbox_evidence_counts")
    )
    table_caption_linked_count = _as_int(table_reconstruction_summary.get("caption_linked_table_count"))
    table_footnote_linked_count = _as_int(table_reconstruction_summary.get("footnote_linked_table_count"))
    table_footnote_binding_count = _as_int(table_reconstruction_summary.get("table_footnote_binding_count"))
    table_footnote_cell_binding_count = _as_int(
        table_reconstruction_summary.get("table_footnote_cell_binding_count")
    )
    table_footnote_bound_cell_count = _as_int(table_reconstruction_summary.get("table_footnote_bound_cell_count"))
    table_footnote_unbound_count = _as_int(table_reconstruction_summary.get("table_footnote_unbound_count"))
    table_footnote_table_level_count = _as_int(
        table_reconstruction_summary.get("table_footnote_table_level_count")
    )
    continued_table_group_count = _as_int(table_reconstruction_summary.get("continued_table_group_count"))
    continued_table_segment_count = _as_int(table_reconstruction_summary.get("continued_table_segment_count"))
    continued_table_merged_cell_candidate_count = _as_int(
        table_reconstruction_summary.get("continued_table_merged_cell_candidate_count")
    )
    continued_table_reconstructable_group_count = _as_int(
        table_reconstruction_summary.get("continued_table_reconstructable_group_count")
    )
    continued_table_merged_row_count = _as_int(table_reconstruction_summary.get("continued_table_merged_row_count"))
    table_chain_candidate_count = _as_int(table_reconstruction_summary.get("table_chain_candidate_count"))
    table_chain_merged_count = _as_int(table_reconstruction_summary.get("table_chain_merged_count"))
    table_chain_reject_count = _as_int(table_reconstruction_summary.get("table_chain_reject_count"))
    table_chain_row_gain = _as_int(table_reconstruction_summary.get("table_chain_row_gain"))
    table_chain_warning_count = _as_int(table_reconstruction_summary.get("table_chain_warning_count"))
    table_chain_reject_reason_count = _as_int(
        table_reconstruction_summary.get("table_chain_reject_reason_count")
    )
    table_chain_warning_reason_count = _as_int(
        table_reconstruction_summary.get("table_chain_warning_reason_count")
    )
    table_chain_reject_reason_counts = _counter_dict(
        table_reconstruction_summary.get("table_chain_reject_reason_counts")
    )
    table_chain_reject_reason_category_counts = _counter_dict(
        table_reconstruction_summary.get("table_chain_reject_reason_category_counts")
    )
    table_chain_warning_reason_counts = _counter_dict(
        table_reconstruction_summary.get("table_chain_warning_reason_counts")
    )
    table_chain_warning_reason_category_counts = _counter_dict(
        table_reconstruction_summary.get("table_chain_warning_reason_category_counts")
    )
    table_reconstruction_ready_rate = _as_float(
        table_reconstruction_summary.get("table_reconstruction_ready_rate")
    )
    table_continuation_count = _as_int(structure_summary.get("table_continuation_count"))
    table_footnote_count = _as_int(structure_summary.get("table_footnote_count"))
    caption_count = _as_int(structure_summary.get("caption_count"))
    caption_orphan_count = _as_int(structure_summary.get("caption_orphan_count"))
    footnote_count = _as_int(structure_summary.get("footnote_count"))
    footnote_orphan_count = _as_int(structure_summary.get("footnote_orphan_count"))
    relationship_warning_count = _as_int(structure_summary.get("relationship_warning_count"))
    cross_page_relationship_count = _as_int(structure_summary.get("cross_page_relationship_count"))
    caption_cross_page_linked_count = _as_int(structure_summary.get("caption_cross_page_linked_count"))
    caption_cross_page_orphan_count = _as_int(structure_summary.get("caption_cross_page_orphan_count"))
    footnote_cross_page_linked_count = _as_int(structure_summary.get("footnote_cross_page_linked_count"))
    footnote_cross_page_orphan_count = _as_int(structure_summary.get("footnote_cross_page_orphan_count"))
    cross_page_parent_gap_max = _as_int(structure_summary.get("cross_page_parent_gap_max"))
    cross_page_parent_attempt_count = (
        cross_page_relationship_count
        + caption_cross_page_orphan_count
        + footnote_cross_page_orphan_count
    )
    entity_candidate_count = _as_int(structure_summary.get("entity_candidate_count"))
    entity_unique_count = _as_int(structure_summary.get("entity_unique_count"))
    translation_entity_candidate_count = _as_int(translation_summary.get("entity_candidate_count"))
    missing_entity_token_count = _as_int(translation_summary.get("missing_entity_token_count"))
    source_formula_token_count = _as_int(translation_summary.get("source_formula_token_count"))
    missing_formula_token_count = _as_int(translation_summary.get("missing_formula_token_count"))
    source_equation_label_count = _as_int(translation_summary.get("source_equation_label_count"))
    missing_equation_label_count = _as_int(translation_summary.get("missing_equation_label_count"))
    source_table_count = _as_int(translation_summary.get("source_table_count")) or table_count
    table_shape_error_count = _as_int(translation_summary.get("table_shape_error_count"))
    source_table_locked_token_count = _as_int(translation_summary.get("source_table_locked_token_count"))
    table_cell_token_error_count = _as_int(translation_summary.get("table_cell_token_error_count"))
    missing_table_locked_token_count = _as_int(translation_summary.get("missing_table_locked_token_count"))
    page_boundary_fragment_count = _as_int(structure_summary.get("page_boundary_fragment_count"))
    page_boundary_stitch_candidate_count = _as_int(
        structure_summary.get("page_boundary_stitch_candidate_count")
    )
    table_continuation_boundary_count = _as_int(structure_summary.get("table_continuation_boundary_count"))
    page_boundary_fragment_rate = _as_float(structure_summary.get("page_boundary_fragment_rate"))
    split_boundary_count = _as_int(chunk_boundary_summary.get("split_boundary_count"))
    protected_boundary_count = _as_int(chunk_boundary_summary.get("protected_boundary_count"))
    co_located_boundary_count = _as_int(chunk_boundary_summary.get("co_located_boundary_count"))
    high_risk_split_count = _as_int(chunk_boundary_summary.get("high_risk_split_count"))
    table_continuation_protected_count = _as_int(
        chunk_boundary_summary.get("table_continuation_protected_count")
    )
    table_continuation_split_count = _as_int(chunk_boundary_summary.get("table_continuation_split_count"))
    table_continuation_co_located_count = _as_int(
        chunk_boundary_summary.get("table_continuation_co_located_count")
    )
    budget_overflow_chunk_count = _as_int(chunk_boundary_summary.get("budget_overflow_chunk_count"))
    budget_overflow_char_total = _as_int(chunk_boundary_summary.get("budget_overflow_char_total"))
    structural_relation_protected_count = _as_int(
        chunk_boundary_summary.get("structural_relation_protected_count")
    )
    budget_split_reason_counts = _counter_dict(chunk_boundary_summary.get("budget_split_reason_counts"))
    budget_pressure_counts = _counter_dict(chunk_boundary_summary.get("budget_pressure_counts"))
    baseline_split_boundary_count = _as_int(chunk_strategy_summary.get("baseline_split_boundary_count"))
    active_split_boundary_count = _as_int(chunk_strategy_summary.get("active_split_boundary_count"))
    active_split_reduction_vs_baseline = _as_int(chunk_strategy_summary.get("active_split_reduction_vs_baseline"))
    active_split_reduction_rate_vs_baseline = _as_float(
        chunk_strategy_summary.get("active_split_reduction_rate_vs_baseline")
    )
    baseline_table_continuation_split_count = _as_int(
        chunk_strategy_summary.get("baseline_table_continuation_split_count")
    )
    active_table_continuation_split_count = _as_int(
        chunk_strategy_summary.get("active_table_continuation_split_count")
    )
    active_table_continuation_split_reduction_vs_baseline = _as_int(
        chunk_strategy_summary.get("active_table_continuation_split_reduction_vs_baseline")
    )
    active_table_continuation_split_reduction_rate_vs_baseline = _as_float(
        chunk_strategy_summary.get("active_table_continuation_split_reduction_rate_vs_baseline")
    )
    routed_page_count = _as_int(vision_summary.get("routed_page_count"))
    vision_preview_page_count = _as_int(vision_summary.get("preview_page_count"))
    vision_region_crop_count = _as_int(vision_summary.get("preview_crop_count"))
    ocr_task_count = _as_int(ocr_task_summary.get("task_count"))
    ocr_region_task_count = _as_int(ocr_task_summary.get("region_task_count"))
    ocr_page_task_count = _as_int(ocr_task_summary.get("page_task_count"))
    ocr_ready_task_count = _as_int(ocr_task_summary.get("ready_task_count"))
    ocr_blocked_task_count = _as_int(ocr_task_summary.get("blocked_by_missing_evidence_count"))
    ocr_vlm_fallback_task_count = _as_int(ocr_task_summary.get("vlm_fallback_task_count"))
    ocr_structured_contract_task_count = _as_int(ocr_task_summary.get("structured_contract_task_count"))
    ocr_table_context_task_count = _as_int(ocr_task_summary.get("table_context_task_count"))
    ocr_table_context_ready_task_count = _as_int(ocr_task_summary.get("table_context_ready_task_count"))
    ocr_formula_context_task_count = _as_int(ocr_task_summary.get("formula_context_task_count"))
    ocr_formula_context_ready_task_count = _as_int(ocr_task_summary.get("formula_context_ready_task_count"))
    ocr_result_payload_count = _as_int(ocr_result_summary.get("result_count"))
    ocr_invalid_result_count = _as_int(ocr_result_summary.get("invalid_result_count"))
    ocr_executor_attempted_task_count = _as_int(ocr_execution_summary.get("attempted_task_count"))
    ocr_executor_succeeded_task_count = _as_int(ocr_execution_summary.get("succeeded_task_count"))
    ocr_executor_failed_task_count = _as_int(ocr_execution_summary.get("failed_task_count"))
    ocr_executor_skipped_task_count = _as_int(ocr_execution_summary.get("skipped_task_count"))
    ocr_executor_available = _as_bool(ocr_execution_summary.get("engine_available"))
    ocr_result_count = _as_int(ocr_writeback_summary.get("result_count"))
    ocr_accepted_result_count = _as_int(ocr_writeback_summary.get("accepted_result_count"))
    ocr_rejected_result_count = _as_int(ocr_writeback_summary.get("rejected_result_count"))
    ocr_pending_task_count = _as_int(ocr_writeback_summary.get("pending_task_count"))
    ocr_missing_result_task_count = _as_int(ocr_writeback_summary.get("missing_result_task_count"))
    ocr_unknown_task_result_count = _as_int(ocr_writeback_summary.get("unknown_task_result_count"))
    ocr_block_writeback_count = _as_int(ocr_writeback_summary.get("block_writeback_count"))
    ocr_page_writeback_count = _as_int(ocr_writeback_summary.get("page_writeback_count"))
    ocr_table_context_writeback_count = _as_int(ocr_writeback_summary.get("table_context_writeback_count"))
    ocr_formula_context_writeback_count = _as_int(ocr_writeback_summary.get("formula_context_writeback_count"))
    ocr_structured_result_writeback_count = _as_int(
        ocr_writeback_summary.get("structured_result_writeback_count")
    )
    ocr_candidate_qa_count = _as_int(ocr_candidate_summary.get("candidate_count"))
    ocr_candidate_promotable_count = _as_int(ocr_candidate_summary.get("promotable_candidate_count"))
    ocr_candidate_needs_review_count = _as_int(ocr_candidate_summary.get("needs_review_candidate_count"))
    ocr_candidate_blocked_count = _as_int(ocr_candidate_summary.get("blocked_candidate_count"))
    ocr_candidate_text_char_count = _as_int(ocr_candidate_summary.get("candidate_text_char_count"))
    ocr_table_context_candidate_count = _as_int(ocr_candidate_summary.get("table_context_candidate_count"))
    ocr_formula_context_candidate_count = _as_int(ocr_candidate_summary.get("formula_context_candidate_count"))
    ocr_structured_contract_candidate_count = _as_int(
        ocr_candidate_summary.get("structured_contract_candidate_count")
    )
    ocr_subtarget_candidate_count = _as_int(ocr_candidate_summary.get("subtarget_candidate_count"))
    ocr_structured_result_candidate_count = _as_int(
        ocr_candidate_summary.get("structured_result_candidate_count")
    )
    ocr_structured_cells_candidate_count = _as_int(
        ocr_candidate_summary.get("structured_cells_candidate_count")
    )
    ocr_cell_bboxes_candidate_count = _as_int(ocr_candidate_summary.get("cell_bboxes_candidate_count"))
    ocr_merged_cell_candidates_candidate_count = _as_int(
        ocr_candidate_summary.get("merged_cell_candidates_candidate_count")
    )
    ocr_table_footnotes_candidate_count = _as_int(ocr_candidate_summary.get("table_footnotes_candidate_count"))
    ocr_formula_latex_candidate_count = _as_int(ocr_candidate_summary.get("formula_latex_candidate_count"))
    ocr_formula_tokens_candidate_count = _as_int(ocr_candidate_summary.get("formula_tokens_candidate_count"))
    ocr_equation_labels_candidate_count = _as_int(ocr_candidate_summary.get("equation_labels_candidate_count"))
    ocr_formula_confidence_candidate_count = _as_int(ocr_candidate_summary.get("formula_confidence_candidate_count"))
    ocr_structured_cell_count = _as_int(ocr_candidate_summary.get("structured_cell_count"))
    ocr_cell_bbox_count = _as_int(ocr_candidate_summary.get("cell_bbox_count"))
    ocr_result_merged_cell_candidate_count = _as_int(
        ocr_candidate_summary.get("result_merged_cell_candidate_count")
    )
    ocr_result_table_footnote_count = _as_int(ocr_candidate_summary.get("result_table_footnote_count"))
    ocr_result_formula_latex_count = _as_int(ocr_candidate_summary.get("result_formula_latex_count"))
    ocr_result_formula_token_count = _as_int(ocr_candidate_summary.get("result_formula_token_count"))
    ocr_result_equation_label_count = _as_int(ocr_candidate_summary.get("result_equation_label_count"))
    ocr_structured_table_candidate_count = _as_int(
        ocr_candidate_summary.get("structured_table_candidate_count")
    )
    ocr_structured_table_gate_passed_count = _as_int(
        ocr_candidate_summary.get("structured_table_gate_passed_count")
    )
    ocr_structured_table_gate_review_count = _as_int(
        ocr_candidate_summary.get("structured_table_gate_review_count")
    )
    ocr_structured_table_gate_blocked_count = _as_int(
        ocr_candidate_summary.get("structured_table_gate_blocked_count")
    )
    ocr_structured_table_missing_locked_token_count = _as_int(
        ocr_candidate_summary.get("structured_table_missing_locked_token_count")
    )
    ocr_structured_table_row_col_mismatch_count = (
        ocr_candidate_structured_table_gate_issue_counts.get("structured_table_row_count_mismatch", 0)
        + ocr_candidate_structured_table_gate_issue_counts.get("structured_table_column_count_mismatch", 0)
    )
    ocr_structured_table_missing_cell_bboxes_count = (
        ocr_candidate_structured_table_gate_issue_counts.get("structured_table_missing_cell_bboxes", 0)
        + ocr_candidate_structured_table_gate_issue_counts.get("structured_table_incomplete_cell_bboxes", 0)
    )
    ocr_structured_formula_candidate_count = _as_int(
        ocr_candidate_summary.get("structured_formula_candidate_count")
    )
    ocr_structured_formula_gate_passed_count = _as_int(
        ocr_candidate_summary.get("structured_formula_gate_passed_count")
    )
    ocr_structured_formula_gate_review_count = _as_int(
        ocr_candidate_summary.get("structured_formula_gate_review_count")
    )
    ocr_structured_formula_gate_blocked_count = _as_int(
        ocr_candidate_summary.get("structured_formula_gate_blocked_count")
    )
    ocr_structured_formula_missing_locked_token_count = _as_int(
        ocr_candidate_summary.get("structured_formula_missing_locked_token_count")
    )
    ocr_structured_formula_token_count = _as_int(
        ocr_candidate_summary.get("structured_formula_token_count")
    )
    ocr_structured_formula_equation_label_count = _as_int(
        ocr_candidate_summary.get("structured_formula_equation_label_count")
    )
    ocr_structured_formula_missing_equation_label_count = ocr_candidate_structured_formula_gate_issue_counts.get(
        "structured_formula_missing_equation_labels",
        0,
    )
    ocr_candidate_promotion_eligible_count = _as_int(ocr_candidate_promotion_summary.get("eligible_candidate_count"))
    ocr_candidate_promoted_count = _as_int(ocr_candidate_promotion_summary.get("promoted_candidate_count"))
    ocr_candidate_promotion_skipped_count = _as_int(ocr_candidate_promotion_summary.get("skipped_candidate_count"))
    ocr_candidate_block_promotion_count = _as_int(ocr_candidate_promotion_summary.get("block_promotion_count"))
    ocr_candidate_page_promotion_count = _as_int(ocr_candidate_promotion_summary.get("page_promotion_count"))
    ocr_canonical_structure_promotion_count = _as_int(
        ocr_candidate_promotion_summary.get("canonical_structure_promotion_count")
    )
    ocr_structured_table_promotion_count = _as_int(
        ocr_candidate_promotion_summary.get("structured_table_promotion_count")
    )
    ocr_structured_formula_promotion_count = _as_int(
        ocr_candidate_promotion_summary.get("structured_formula_promotion_count")
    )
    ocr_candidate_promoted_text_char_count = _as_int(
        ocr_candidate_promotion_summary.get("promoted_text_char_count")
    )
    ocr_candidate_page_count = (
        vision_action_counts.get("local_ocr", 0) + vision_action_counts.get("vlm_review", 0)
    )
    chunk_count = _as_int(translation_summary.get("chunk_count")) or _as_int(repair_summary.get("chunk_count"))
    structure_hint_chunk_count = _as_int(structure_hints_summary.get("structure_hint_chunk_count"))
    structure_hint_empty_chunk_count = _as_int(
        structure_hints_summary.get("structure_hint_empty_chunk_count")
    )
    structure_hint_char_count = _as_int(structure_hints_summary.get("structure_hint_char_count"))
    structure_hint_avg_char_count = _as_float(
        structure_hints_summary.get("structure_hint_avg_char_count")
    )
    structure_hint_max_char_count = _as_int(
        structure_hints_summary.get("structure_hint_max_char_count")
    )
    structure_hint_table_count = _as_int(structure_hints_summary.get("structure_hint_table_count"))
    structure_hint_continued_group_count = _as_int(
        structure_hints_summary.get("structure_hint_continued_group_count")
    )
    structure_hint_merged_cell_candidate_count = _as_int(
        structure_hints_summary.get("structure_hint_merged_cell_candidate_count")
    )
    structure_hint_merged_cell_candidate_type_counts = _counter_dict(
        structure_hints_summary.get("structure_hint_merged_cell_candidate_type_counts")
    )
    structure_hint_merged_cell_candidate_reason_counts = _counter_dict(
        structure_hints_summary.get("structure_hint_merged_cell_candidate_reason_counts")
    )
    structure_hint_merged_cell_candidate_status_counts = _counter_dict(
        structure_hints_summary.get("structure_hint_merged_cell_candidate_status_counts")
    )
    structure_hint_merged_cell_candidate_visual_evidence_counts = _counter_dict(
        structure_hints_summary.get("structure_hint_merged_cell_candidate_visual_evidence_counts")
    )
    structure_hint_merged_cell_candidate_bbox_evidence_counts = _counter_dict(
        structure_hints_summary.get("structure_hint_merged_cell_candidate_bbox_evidence_counts")
    )
    structure_hint_footnote_binding_count = _as_int(
        structure_hints_summary.get("structure_hint_footnote_binding_count")
    )
    structure_hint_locked_token_count = _as_int(structure_hints_summary.get("structure_hint_locked_token_count"))
    structure_hint_total_chunk_count = _as_int(structure_hints_summary.get("chunk_count"))
    translation_issue_count = _as_int(translation_summary.get("issue_count"))
    repair_item_count = _as_int(repair_summary.get("repair_item_count"))
    repair_request_count = _as_int(repair_request_summary.get("repair_request_count"))
    repair_backend_request_count = _as_int(repair_request_summary.get("ready_for_translation_backend_count"))
    repair_manual_request_count = _as_int(repair_request_summary.get("manual_review_request_count"))
    repair_executed_request_count = _as_int(repair_result_summary.get("executed_request_count"))
    repair_succeeded_count = _as_int(repair_result_summary.get("succeeded_count"))
    repair_failed_count = _as_int(repair_result_summary.get("failed_count"))
    repair_skipped_count = _as_int(repair_result_summary.get("skipped_count"))
    repair_validation_checked_count = _as_int(repair_validation_summary.get("validated_result_count"))
    repair_validation_passed_count = _as_int(repair_validation_summary.get("passed_count"))
    repair_validation_failed_count = _as_int(repair_validation_summary.get("failed_count"))
    repair_validation_unchecked_count = _as_int(repair_validation_summary.get("unchecked_count"))
    repair_validation_skipped_count = _as_int(repair_validation_summary.get("skipped_count"))
    repair_validation_checked_locked_token_count = _as_int(
        repair_validation_summary.get("checked_locked_token_count")
    )
    repair_validation_missing_locked_token_count = _as_int(
        repair_validation_summary.get("missing_locked_token_count")
    )
    repair_validation_table_shape_check_count = _as_int(repair_validation_summary.get("table_shape_check_count"))
    repair_validation_table_shape_passed_count = _as_int(repair_validation_summary.get("table_shape_passed_count"))
    repair_merge_candidate_count = _as_int(repair_merge_summary.get("merge_candidate_count"))
    repair_merge_applied_count = _as_int(repair_merge_summary.get("applied_count"))
    repair_merge_patched_chunk_count = _as_int(repair_merge_summary.get("patched_chunk_count"))
    repair_merge_skipped_count = _as_int(repair_merge_summary.get("skipped_count"))
    repair_merge_manual_required_count = _as_int(repair_merge_summary.get("manual_merge_required_count"))
    repair_merge_conflict_count = _as_int(repair_merge_summary.get("conflict_count"))
    repair_merge_table_targeted_patch_count = _as_int(repair_merge_summary.get("table_targeted_patch_count"))
    repair_merge_strategy_counts = _counter_dict(repair_merge_summary.get("strategy_counts"))
    repair_merge_applied_strategy_counts = _counter_dict(repair_merge_summary.get("applied_strategy_counts"))
    repair_patch_review_count = _as_int(repair_patch_review_summary.get("patch_count"))
    repair_patch_review_safe_count = _as_int(repair_patch_review_summary.get("auto_merge_safe_count"))
    repair_patch_review_effective_safe_count = _as_int(repair_patch_review_summary.get("effective_safe_count"))
    repair_patch_review_required_count = _as_int(repair_patch_review_summary.get("review_required_count"))
    repair_patch_review_blocking_count = _as_int(repair_patch_review_summary.get("publish_blocking_count"))
    repair_patch_review_human_reviewed_count = _as_int(repair_patch_review_summary.get("human_reviewed_count"))
    repair_patch_review_table_count = _as_int(repair_patch_review_summary.get("table_patch_review_count"))
    repair_patch_review_decision_counts = _counter_dict(
        repair_patch_review_summary.get("default_decision_counts")
    )
    repair_patch_review_effective_decision_counts = _counter_dict(
        repair_patch_review_summary.get("effective_decision_counts")
    )
    repair_patch_review_human_decision_counts = _counter_dict(
        repair_patch_review_summary.get("human_decision_counts")
    )
    repair_patch_review_risk_counts = _counter_dict(repair_patch_review_summary.get("risk_level_counts"))
    repair_publish_confirmed = _as_bool(repair_publish_summary.get("confirmed"))
    repair_publish_published = _as_bool(repair_publish_summary.get("published"))
    repair_publish_open_issue_count = _as_int(repair_publish_summary.get("open_merge_issue_count"))
    repair_publish_status = str(repair_publish_summary.get("publish_status") or "")
    post_repair_issue_count = _as_int(repair_merge_qa_summary.get("issue_count"))
    post_repair_table_shape_error_count = _as_int(repair_merge_qa_summary.get("table_shape_error_count"))
    post_repair_table_cell_token_error_count = _as_int(repair_merge_qa_summary.get("table_cell_token_error_count"))
    post_repair_missing_table_locked_token_count = _as_int(
        repair_merge_qa_summary.get("missing_table_locked_token_count")
    )
    translated_pdf_generated = _as_bool(translated_pdf_summary.get("generated"))
    translated_pdf_page_count = _as_int(translated_pdf_summary.get("page_count"))
    translated_pdf_chunk_count = _as_int(translated_pdf_summary.get("chunk_count"))
    translated_pdf_table_count = _as_int(translated_pdf_summary.get("table_count"))
    translated_pdf_qa_issue_count = _as_int(translated_pdf_summary.get("qa_issue_count"))
    translated_pdf_repair_item_count = _as_int(translated_pdf_summary.get("repair_item_count"))
    translated_pdf_warning_count = _as_int(translated_pdf_summary.get("warning_count"))
    translated_pdf_structure_context_chunk_count = _as_int(
        translated_pdf_summary.get("structure_context_chunk_count")
    )
    translated_pdf_source_table_reference_count = _as_int(
        translated_pdf_summary.get("source_table_reference_count")
    )
    translated_pdf_source_caption_reference_count = _as_int(
        translated_pdf_summary.get("source_caption_reference_count")
    )
    translated_pdf_source_footnote_reference_count = _as_int(
        translated_pdf_summary.get("source_footnote_reference_count")
    )
    translated_pdf_source_footnote_cell_binding_count = _as_int(
        translated_pdf_summary.get("source_footnote_cell_binding_count")
    )
    translated_pdf_continued_table_group_reference_count = _as_int(
        translated_pdf_summary.get("continued_table_group_reference_count")
    )
    translated_pdf_structural_relation_reference_count = _as_int(
        translated_pdf_summary.get("structural_relation_reference_count")
    )
    max_english_residual_ratio = _as_float(translation_summary.get("max_english_residual_ratio"))
    total_elapsed_ms = _as_int(run_summary.get("total_elapsed_ms"))
    translation_elapsed_ms = _as_int(run_summary.get("translation_elapsed_ms"))
    translation_request_count = _as_int(run_summary.get("translation_request_count"))
    http_attempt_count = _as_int(run_summary.get("http_attempt_count"))
    http_retry_count = _as_int(run_summary.get("http_retry_count"))
    http_failed_attempt_count = _as_int(run_summary.get("http_failed_attempt_count"))
    http_retryable_error_count = _as_int(run_summary.get("http_retryable_error_count"))
    http_fatal_error_count = _as_int(run_summary.get("http_fatal_error_count"))
    failed_event_count = _as_int(run_summary.get("failed_event_count"))
    skipped_chunk_count = _as_int(run_summary.get("skipped_chunk_count"))
    source_char_count = _as_int(run_summary.get("source_char_count"))
    context_char_count = _as_int(run_summary.get("context_char_count"))
    request_char_count = _as_int(run_summary.get("request_char_count"))
    translated_char_count = _as_int(run_summary.get("translated_char_count"))
    estimated_source_token_count = _as_int(run_summary.get("estimated_source_token_count"))
    estimated_context_token_count = _as_int(run_summary.get("estimated_context_token_count"))
    estimated_request_token_count = _as_int(run_summary.get("estimated_request_token_count"))
    estimated_translated_token_count = _as_int(run_summary.get("estimated_translated_token_count"))
    estimated_total_token_count = _as_int(run_summary.get("estimated_total_token_count"))
    avg_chunk_elapsed_ms = _as_float(run_summary.get("avg_chunk_elapsed_ms"))
    max_chunk_elapsed_ms = _as_int(run_summary.get("max_chunk_elapsed_ms"))
    request_chars_per_second = _as_float(run_summary.get("request_chars_per_second"))
    translated_chars_per_second = _as_float(run_summary.get("translated_chars_per_second"))
    estimated_total_cost = _as_float(cost_summary.get("estimated_total_cost"))
    input_token_cost = _as_float(cost_summary.get("input_token_cost"))
    output_token_cost = _as_float(cost_summary.get("output_token_cost"))
    input_char_cost = _as_float(cost_summary.get("input_char_cost"))
    output_char_cost = _as_float(cost_summary.get("output_char_cost"))
    request_cost = _as_float(cost_summary.get("request_cost"))
    cost_usage = cost_estimate.get("usage") if isinstance(cost_estimate, dict) else {}
    if not isinstance(cost_usage, dict):
        cost_usage = {}
    billable_request_count = _as_int(cost_usage.get("billable_request_count"))
    cost_configured = _as_bool((cost_estimate or {}).get("configured")) if isinstance(cost_estimate, dict) else False
    cost_currency = str((cost_estimate or {}).get("currency") or "")
    cost_profile_key = str((cost_estimate or {}).get("profile_key") or "")
    cost_warning_count = len((cost_estimate or {}).get("warnings") or []) if isinstance(cost_estimate, dict) else 0

    relationship_total = caption_count + footnote_count
    effective_entity_candidate_count = translation_entity_candidate_count or entity_candidate_count
    resolved_doc_id = (
        doc_id
        or (structure_qa or {}).get("doc_id")
        or (vision_route or {}).get("doc_id")
        or "unknown"
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "doc_id": resolved_doc_id,
        "pipeline_variant": pipeline_variant or "unknown",
        "quality": {
            "page_count": page_count,
            "chunk_count": chunk_count,
            "structure_hint_chunk_count": structure_hint_chunk_count,
            "structure_hint_empty_chunk_count": structure_hint_empty_chunk_count,
            "structure_hint_char_count": structure_hint_char_count,
            "structure_hint_avg_char_count": structure_hint_avg_char_count,
            "structure_hint_max_char_count": structure_hint_max_char_count,
            "structure_hint_table_count": structure_hint_table_count,
            "structure_hint_continued_group_count": structure_hint_continued_group_count,
            "structure_hint_merged_cell_candidate_count": structure_hint_merged_cell_candidate_count,
            "structure_hint_merged_cell_candidate_visual_supported_count": (
                structure_hint_merged_cell_candidate_status_counts.get("visually_supported", 0)
            ),
            "structure_hint_footnote_binding_count": structure_hint_footnote_binding_count,
            "structure_hint_locked_token_count": structure_hint_locked_token_count,
            "table_count": table_count,
            "reconstructable_table_count": reconstructable_table_count,
            "low_confidence_table_count": low_confidence_table_count,
            "table_cell_count": table_cell_count,
            "table_empty_cell_count": table_empty_cell_count,
            "table_numeric_cell_count": table_numeric_cell_count,
            "table_numeric_token_count": table_numeric_token_count,
            "table_unit_token_count": table_unit_token_count,
            "table_significance_token_count": table_significance_token_count,
            "table_ragged_table_count": table_ragged_table_count,
            "table_ragged_row_count": table_ragged_row_count,
            "table_merged_cell_candidate_count": table_merged_cell_candidate_count,
            "table_merged_cell_candidate_visual_supported_count": (
                table_merged_cell_candidate_status_counts.get("visually_supported", 0)
            ),
            "table_caption_linked_count": table_caption_linked_count,
            "table_footnote_linked_count": table_footnote_linked_count,
            "table_footnote_binding_count": table_footnote_binding_count,
            "table_footnote_cell_binding_count": table_footnote_cell_binding_count,
            "table_footnote_bound_cell_count": table_footnote_bound_cell_count,
            "table_footnote_unbound_count": table_footnote_unbound_count,
            "table_footnote_table_level_count": table_footnote_table_level_count,
            "continued_table_group_count": continued_table_group_count,
            "continued_table_segment_count": continued_table_segment_count,
            "continued_table_merged_cell_candidate_count": continued_table_merged_cell_candidate_count,
            "continued_table_reconstructable_group_count": continued_table_reconstructable_group_count,
            "continued_table_merged_row_count": continued_table_merged_row_count,
            "table_chain_candidate_count": table_chain_candidate_count,
            "table_chain_merged_count": table_chain_merged_count,
            "table_chain_reject_count": table_chain_reject_count,
            "table_chain_row_gain": table_chain_row_gain,
            "table_chain_warning_count": table_chain_warning_count,
            "table_chain_reject_reason_count": table_chain_reject_reason_count,
            "table_chain_warning_reason_count": table_chain_warning_reason_count,
            "table_continuation_count": table_continuation_count,
            "table_footnote_count": table_footnote_count,
            "caption_orphan_count": caption_orphan_count,
            "footnote_orphan_count": footnote_orphan_count,
            "relationship_warning_count": relationship_warning_count,
            "cross_page_relationship_count": cross_page_relationship_count,
            "caption_cross_page_linked_count": caption_cross_page_linked_count,
            "caption_cross_page_orphan_count": caption_cross_page_orphan_count,
            "footnote_cross_page_linked_count": footnote_cross_page_linked_count,
            "footnote_cross_page_orphan_count": footnote_cross_page_orphan_count,
            "cross_page_parent_gap_max": cross_page_parent_gap_max,
            "entity_candidate_count": entity_candidate_count,
            "entity_unique_count": entity_unique_count,
            "translation_entity_candidate_count": translation_entity_candidate_count,
            "missing_entity_token_count": missing_entity_token_count,
            "source_formula_token_count": source_formula_token_count,
            "missing_formula_token_count": missing_formula_token_count,
            "source_equation_label_count": source_equation_label_count,
            "missing_equation_label_count": missing_equation_label_count,
            "source_table_count": source_table_count,
            "table_shape_error_count": table_shape_error_count,
            "source_table_locked_token_count": source_table_locked_token_count,
            "table_cell_token_error_count": table_cell_token_error_count,
            "missing_table_locked_token_count": missing_table_locked_token_count,
            "page_boundary_fragment_count": page_boundary_fragment_count,
            "page_boundary_stitch_candidate_count": page_boundary_stitch_candidate_count,
            "page_boundary_fragment_rate": page_boundary_fragment_rate,
            "table_continuation_boundary_count": table_continuation_boundary_count,
            "split_boundary_count": split_boundary_count,
            "protected_boundary_count": protected_boundary_count,
            "co_located_boundary_count": co_located_boundary_count,
            "high_risk_split_count": high_risk_split_count,
            "table_continuation_protected_count": table_continuation_protected_count,
            "table_continuation_split_count": table_continuation_split_count,
            "table_continuation_co_located_count": table_continuation_co_located_count,
            "budget_overflow_chunk_count": budget_overflow_chunk_count,
            "budget_overflow_char_total": budget_overflow_char_total,
            "structural_relation_protected_count": structural_relation_protected_count,
            "baseline_split_boundary_count": baseline_split_boundary_count,
            "active_split_boundary_count": active_split_boundary_count,
            "active_split_reduction_vs_baseline": active_split_reduction_vs_baseline,
            "baseline_table_continuation_split_count": baseline_table_continuation_split_count,
            "active_table_continuation_split_count": active_table_continuation_split_count,
            "active_table_continuation_split_reduction_vs_baseline": (
                active_table_continuation_split_reduction_vs_baseline
            ),
            "routed_page_count": routed_page_count,
            "vision_preview_page_count": vision_preview_page_count,
            "vision_region_crop_count": vision_region_crop_count,
            "ocr_task_count": ocr_task_count,
            "ocr_region_task_count": ocr_region_task_count,
            "ocr_page_task_count": ocr_page_task_count,
            "ocr_ready_task_count": ocr_ready_task_count,
            "ocr_blocked_task_count": ocr_blocked_task_count,
            "ocr_vlm_fallback_task_count": ocr_vlm_fallback_task_count,
            "ocr_structured_contract_task_count": ocr_structured_contract_task_count,
            "ocr_table_context_task_count": ocr_table_context_task_count,
            "ocr_table_context_ready_task_count": ocr_table_context_ready_task_count,
            "ocr_formula_context_task_count": ocr_formula_context_task_count,
            "ocr_formula_context_ready_task_count": ocr_formula_context_ready_task_count,
            "ocr_result_payload_count": ocr_result_payload_count,
            "ocr_invalid_result_count": ocr_invalid_result_count,
            "ocr_executor_attempted_task_count": ocr_executor_attempted_task_count,
            "ocr_executor_succeeded_task_count": ocr_executor_succeeded_task_count,
            "ocr_executor_failed_task_count": ocr_executor_failed_task_count,
            "ocr_executor_skipped_task_count": ocr_executor_skipped_task_count,
            "ocr_executor_available": ocr_executor_available,
            "ocr_result_count": ocr_result_count,
            "ocr_accepted_result_count": ocr_accepted_result_count,
            "ocr_rejected_result_count": ocr_rejected_result_count,
            "ocr_pending_task_count": ocr_pending_task_count,
            "ocr_missing_result_task_count": ocr_missing_result_task_count,
            "ocr_unknown_task_result_count": ocr_unknown_task_result_count,
            "ocr_block_writeback_count": ocr_block_writeback_count,
            "ocr_page_writeback_count": ocr_page_writeback_count,
            "ocr_table_context_writeback_count": ocr_table_context_writeback_count,
            "ocr_formula_context_writeback_count": ocr_formula_context_writeback_count,
            "ocr_structured_result_writeback_count": ocr_structured_result_writeback_count,
            "ocr_candidate_qa_count": ocr_candidate_qa_count,
            "ocr_candidate_promotable_count": ocr_candidate_promotable_count,
            "ocr_candidate_needs_review_count": ocr_candidate_needs_review_count,
            "ocr_candidate_blocked_count": ocr_candidate_blocked_count,
            "ocr_candidate_text_char_count": ocr_candidate_text_char_count,
            "ocr_table_context_candidate_count": ocr_table_context_candidate_count,
            "ocr_formula_context_candidate_count": ocr_formula_context_candidate_count,
            "ocr_structured_contract_candidate_count": ocr_structured_contract_candidate_count,
            "ocr_subtarget_candidate_count": ocr_subtarget_candidate_count,
            "ocr_structured_result_candidate_count": ocr_structured_result_candidate_count,
            "ocr_structured_cells_candidate_count": ocr_structured_cells_candidate_count,
            "ocr_cell_bboxes_candidate_count": ocr_cell_bboxes_candidate_count,
            "ocr_merged_cell_candidates_candidate_count": ocr_merged_cell_candidates_candidate_count,
            "ocr_table_footnotes_candidate_count": ocr_table_footnotes_candidate_count,
            "ocr_formula_latex_candidate_count": ocr_formula_latex_candidate_count,
            "ocr_formula_tokens_candidate_count": ocr_formula_tokens_candidate_count,
            "ocr_equation_labels_candidate_count": ocr_equation_labels_candidate_count,
            "ocr_formula_confidence_candidate_count": ocr_formula_confidence_candidate_count,
            "ocr_structured_cell_count": ocr_structured_cell_count,
            "ocr_cell_bbox_count": ocr_cell_bbox_count,
            "ocr_result_merged_cell_candidate_count": ocr_result_merged_cell_candidate_count,
            "ocr_result_table_footnote_count": ocr_result_table_footnote_count,
            "ocr_result_formula_latex_count": ocr_result_formula_latex_count,
            "ocr_result_formula_token_count": ocr_result_formula_token_count,
            "ocr_result_equation_label_count": ocr_result_equation_label_count,
            "ocr_structured_table_candidate_count": ocr_structured_table_candidate_count,
            "ocr_structured_table_gate_passed_count": ocr_structured_table_gate_passed_count,
            "ocr_structured_table_gate_review_count": ocr_structured_table_gate_review_count,
            "ocr_structured_table_gate_blocked_count": ocr_structured_table_gate_blocked_count,
            "ocr_structured_table_missing_locked_token_count": ocr_structured_table_missing_locked_token_count,
            "ocr_structured_table_row_col_mismatch_count": ocr_structured_table_row_col_mismatch_count,
            "ocr_structured_table_missing_cell_bboxes_count": ocr_structured_table_missing_cell_bboxes_count,
            "ocr_structured_formula_candidate_count": ocr_structured_formula_candidate_count,
            "ocr_structured_formula_gate_passed_count": ocr_structured_formula_gate_passed_count,
            "ocr_structured_formula_gate_review_count": ocr_structured_formula_gate_review_count,
            "ocr_structured_formula_gate_blocked_count": ocr_structured_formula_gate_blocked_count,
            "ocr_structured_formula_missing_locked_token_count": (
                ocr_structured_formula_missing_locked_token_count
            ),
            "ocr_structured_formula_missing_equation_label_count": (
                ocr_structured_formula_missing_equation_label_count
            ),
            "ocr_structured_formula_token_count": ocr_structured_formula_token_count,
            "ocr_structured_formula_equation_label_count": ocr_structured_formula_equation_label_count,
            "ocr_candidate_promotion_eligible_count": ocr_candidate_promotion_eligible_count,
            "ocr_candidate_promoted_count": ocr_candidate_promoted_count,
            "ocr_candidate_promotion_skipped_count": ocr_candidate_promotion_skipped_count,
            "ocr_candidate_block_promotion_count": ocr_candidate_block_promotion_count,
            "ocr_candidate_page_promotion_count": ocr_candidate_page_promotion_count,
            "ocr_canonical_structure_promotion_count": ocr_canonical_structure_promotion_count,
            "ocr_structured_table_promotion_count": ocr_structured_table_promotion_count,
            "ocr_structured_formula_promotion_count": ocr_structured_formula_promotion_count,
            "ocr_candidate_promoted_text_char_count": ocr_candidate_promoted_text_char_count,
            "ocr_candidate_page_count": ocr_candidate_page_count,
            "translation_issue_count": translation_issue_count,
            "repair_item_count": repair_item_count,
            "repair_request_count": repair_request_count,
            "repair_backend_request_count": repair_backend_request_count,
            "repair_manual_request_count": repair_manual_request_count,
            "repair_executed_request_count": repair_executed_request_count,
            "repair_succeeded_count": repair_succeeded_count,
            "repair_failed_count": repair_failed_count,
            "repair_skipped_count": repair_skipped_count,
            "repair_validation_checked_count": repair_validation_checked_count,
            "repair_validation_passed_count": repair_validation_passed_count,
            "repair_validation_failed_count": repair_validation_failed_count,
            "repair_validation_unchecked_count": repair_validation_unchecked_count,
            "repair_validation_skipped_count": repair_validation_skipped_count,
            "repair_validation_checked_locked_token_count": repair_validation_checked_locked_token_count,
            "repair_validation_missing_locked_token_count": repair_validation_missing_locked_token_count,
            "repair_validation_table_shape_check_count": repair_validation_table_shape_check_count,
            "repair_validation_table_shape_passed_count": repair_validation_table_shape_passed_count,
            "repair_merge_candidate_count": repair_merge_candidate_count,
            "repair_merge_applied_count": repair_merge_applied_count,
            "repair_merge_patched_chunk_count": repair_merge_patched_chunk_count,
            "repair_merge_skipped_count": repair_merge_skipped_count,
            "repair_merge_manual_required_count": repair_merge_manual_required_count,
            "repair_merge_conflict_count": repair_merge_conflict_count,
            "repair_merge_table_targeted_patch_count": repair_merge_table_targeted_patch_count,
            "repair_patch_review_count": repair_patch_review_count,
            "repair_patch_review_safe_count": repair_patch_review_safe_count,
            "repair_patch_review_effective_safe_count": repair_patch_review_effective_safe_count,
            "repair_patch_review_required_count": repair_patch_review_required_count,
            "repair_patch_review_blocking_count": repair_patch_review_blocking_count,
            "repair_patch_review_human_reviewed_count": repair_patch_review_human_reviewed_count,
            "repair_patch_review_table_count": repair_patch_review_table_count,
            "repair_publish_confirmed": repair_publish_confirmed,
            "repair_publish_published": repair_publish_published,
            "repair_publish_open_issue_count": repair_publish_open_issue_count,
            "post_repair_issue_count": post_repair_issue_count,
            "post_repair_issue_delta": translation_issue_count - post_repair_issue_count,
            "post_repair_table_shape_error_count": post_repair_table_shape_error_count,
            "post_repair_table_cell_token_error_count": post_repair_table_cell_token_error_count,
            "post_repair_missing_table_locked_token_count": post_repair_missing_table_locked_token_count,
            "translated_pdf_generated": translated_pdf_generated,
            "translated_pdf_page_count": translated_pdf_page_count,
            "translated_pdf_chunk_count": translated_pdf_chunk_count,
            "translated_pdf_table_count": translated_pdf_table_count,
            "translated_pdf_qa_issue_count": translated_pdf_qa_issue_count,
            "translated_pdf_repair_item_count": translated_pdf_repair_item_count,
            "translated_pdf_warning_count": translated_pdf_warning_count,
            "translated_pdf_structure_context_chunk_count": translated_pdf_structure_context_chunk_count,
            "translated_pdf_source_table_reference_count": translated_pdf_source_table_reference_count,
            "translated_pdf_source_caption_reference_count": translated_pdf_source_caption_reference_count,
            "translated_pdf_source_footnote_reference_count": translated_pdf_source_footnote_reference_count,
            "translated_pdf_source_footnote_cell_binding_count": translated_pdf_source_footnote_cell_binding_count,
            "translated_pdf_continued_table_group_reference_count": (
                translated_pdf_continued_table_group_reference_count
            ),
            "translated_pdf_structural_relation_reference_count": (
                translated_pdf_structural_relation_reference_count
            ),
            "max_english_residual_ratio": max_english_residual_ratio,
        },
        "performance": {
            "total_elapsed_ms": total_elapsed_ms,
            "translation_elapsed_ms": translation_elapsed_ms,
            "translation_request_count": translation_request_count,
            "http_attempt_count": http_attempt_count,
            "http_retry_count": http_retry_count,
            "http_failed_attempt_count": http_failed_attempt_count,
            "http_retryable_error_count": http_retryable_error_count,
            "http_fatal_error_count": http_fatal_error_count,
            "failed_event_count": failed_event_count,
            "skipped_chunk_count": skipped_chunk_count,
            "source_char_count": source_char_count,
            "context_char_count": context_char_count,
            "request_char_count": request_char_count,
            "translated_char_count": translated_char_count,
            "estimated_source_token_count": estimated_source_token_count,
            "estimated_context_token_count": estimated_context_token_count,
            "estimated_request_token_count": estimated_request_token_count,
            "estimated_translated_token_count": estimated_translated_token_count,
            "estimated_total_token_count": estimated_total_token_count,
            "avg_chunk_elapsed_ms": avg_chunk_elapsed_ms,
            "max_chunk_elapsed_ms": max_chunk_elapsed_ms,
            "request_chars_per_second": request_chars_per_second,
            "translated_chars_per_second": translated_chars_per_second,
            "cost_profile_configured": cost_configured,
            "cost_profile_key": cost_profile_key,
            "cost_currency": cost_currency,
            "input_token_cost": input_token_cost,
            "output_token_cost": output_token_cost,
            "input_char_cost": input_char_cost,
            "output_char_cost": output_char_cost,
            "request_cost": request_cost,
            "billable_request_count": billable_request_count,
            "estimated_total_cost": estimated_total_cost,
            "cost_warning_count": cost_warning_count,
        },
        "rates": {
            "relationship_warning_rate": _rate(relationship_warning_count, relationship_total),
            "cross_page_relationship_rate": _rate(cross_page_relationship_count, relationship_total),
            "cross_page_parent_success_rate": _rate(
                cross_page_relationship_count,
                cross_page_parent_attempt_count,
            ),
            "caption_cross_page_link_rate": _rate(caption_cross_page_linked_count, caption_count),
            "footnote_cross_page_link_rate": _rate(footnote_cross_page_linked_count, footnote_count),
            "caption_orphan_rate": _rate(caption_orphan_count, caption_count),
            "footnote_orphan_rate": _rate(footnote_orphan_count, footnote_count),
            "structure_hint_chunk_rate": _rate(
                structure_hint_chunk_count,
                chunk_count or structure_hint_total_chunk_count,
            ),
            "structure_hint_table_per_chunk": _rate(structure_hint_table_count, structure_hint_chunk_count),
            "structure_hint_merged_cell_candidate_per_chunk": _rate(
                structure_hint_merged_cell_candidate_count,
                structure_hint_chunk_count,
            ),
            "structure_hint_merged_cell_candidate_visual_supported_rate": _rate(
                structure_hint_merged_cell_candidate_status_counts.get("visually_supported", 0),
                structure_hint_merged_cell_candidate_count,
            ),
            "structure_hint_footnote_binding_per_chunk": _rate(
                structure_hint_footnote_binding_count,
                structure_hint_chunk_count,
            ),
            "structure_hint_locked_token_per_chunk": _rate(
                structure_hint_locked_token_count,
                structure_hint_chunk_count,
            ),
            "table_reconstruction_ready_rate": table_reconstruction_ready_rate,
            "table_empty_cell_rate": _rate(table_empty_cell_count, table_cell_count),
            "table_ragged_table_rate": _rate(table_ragged_table_count, table_count),
            "table_merged_cell_candidate_rate": _rate(table_merged_cell_candidate_count, table_count),
            "table_merged_cell_candidate_visual_supported_rate": _rate(
                table_merged_cell_candidate_status_counts.get("visually_supported", 0),
                table_merged_cell_candidate_count,
            ),
            "table_numeric_cell_rate": _rate(table_numeric_cell_count, table_cell_count),
            "table_caption_link_rate": _rate(table_caption_linked_count, table_count),
            "table_footnote_binding_rate": _rate(table_footnote_linked_count, table_count),
            "table_footnote_cell_binding_rate": _rate(
                table_footnote_cell_binding_count,
                table_footnote_binding_count,
            ),
            "table_footnote_unbound_rate": _rate(table_footnote_unbound_count, table_footnote_binding_count),
            "table_continuation_rate": _rate(table_continuation_count, table_count),
            "continued_table_reconstruction_rate": _rate(
                continued_table_reconstructable_group_count,
                continued_table_group_count,
            ),
            "table_chain_merge_rate": _rate(table_chain_merged_count, table_chain_candidate_count),
            "table_chain_reject_rate": _rate(table_chain_reject_count, table_chain_candidate_count),
            "table_chain_reject_reason_per_rejected_chain": _rate(
                table_chain_reject_reason_count,
                table_chain_reject_count,
            ),
            "table_chain_warning_reason_per_candidate_chain": _rate(
                table_chain_warning_reason_count + table_chain_reject_reason_count,
                table_chain_candidate_count,
            ),
            "table_shape_error_rate": _rate(table_shape_error_count, source_table_count),
            "table_cell_token_error_rate": _rate(table_cell_token_error_count, table_numeric_cell_count),
            "table_locked_token_missing_rate": _rate(
                missing_table_locked_token_count,
                source_table_locked_token_count,
            ),
            "split_boundary_rate": _rate(split_boundary_count, page_boundary_fragment_count),
            "protected_boundary_rate": _rate(protected_boundary_count, page_boundary_fragment_count),
            "co_located_boundary_rate": _rate(co_located_boundary_count, page_boundary_fragment_count),
            "table_continuation_boundary_split_rate": _rate(
                table_continuation_split_count,
                table_continuation_boundary_count,
            ),
            "table_continuation_boundary_protected_rate": _rate(
                table_continuation_protected_count,
                table_continuation_boundary_count,
            ),
            "budget_overflow_chunk_rate": _rate(budget_overflow_chunk_count, chunk_count),
            "active_split_reduction_rate_vs_baseline": active_split_reduction_rate_vs_baseline,
            "active_table_continuation_split_reduction_rate_vs_baseline": (
                active_table_continuation_split_reduction_rate_vs_baseline
            ),
            "entity_missing_rate": _rate(missing_entity_token_count, effective_entity_candidate_count),
            "formula_token_missing_rate": _rate(missing_formula_token_count, source_formula_token_count),
            "equation_label_missing_rate": _rate(missing_equation_label_count, source_equation_label_count),
            "repair_item_per_chunk": _rate(repair_item_count, chunk_count),
            "repair_request_ready_rate": _rate(repair_backend_request_count, repair_request_count),
            "repair_execution_success_rate": _rate(repair_succeeded_count, repair_executed_request_count),
            "repair_validation_pass_rate": _rate(
                repair_validation_passed_count,
                repair_validation_passed_count + repair_validation_failed_count,
            ),
            "repair_locked_token_pass_rate": _rate(
                repair_validation_checked_locked_token_count - repair_validation_missing_locked_token_count,
                repair_validation_checked_locked_token_count,
            ),
            "repair_table_shape_validation_pass_rate": _rate(
                repair_validation_table_shape_passed_count,
                repair_validation_table_shape_check_count,
            ),
            "repair_merge_apply_rate": _rate(repair_merge_applied_count, repair_merge_candidate_count),
            "repair_merge_table_targeted_patch_rate": _rate(
                repair_merge_table_targeted_patch_count,
                repair_merge_applied_count,
            ),
            "repair_patch_review_safe_rate": _rate(repair_patch_review_safe_count, repair_patch_review_count),
            "repair_patch_review_effective_safe_rate": _rate(
                repair_patch_review_effective_safe_count,
                repair_patch_review_count,
            ),
            "repair_patch_review_human_reviewed_rate": _rate(
                repair_patch_review_human_reviewed_count,
                repair_patch_review_count,
            ),
            "repair_patch_review_required_rate": _rate(
                repair_patch_review_required_count,
                repair_patch_review_count,
            ),
            "repair_publish_rate": _rate(int(repair_publish_published), int(repair_publish_confirmed)),
            "post_repair_issue_reduction_rate": _rate(
                translation_issue_count - post_repair_issue_count,
                translation_issue_count,
            ),
            "ocr_candidate_page_rate": _rate(ocr_candidate_page_count, page_count),
            "routed_page_rate": _rate(routed_page_count, page_count),
            "vision_preview_page_rate": _rate(vision_preview_page_count, page_count),
            "vision_region_crop_per_routed_page": _rate(
                vision_region_crop_count,
                routed_page_count,
            ),
            "ocr_task_per_routed_page": _rate(ocr_task_count, routed_page_count),
            "ocr_region_task_rate": _rate(ocr_region_task_count, ocr_task_count),
            "ocr_ready_task_rate": _rate(ocr_ready_task_count, ocr_task_count),
            "ocr_structured_contract_task_rate": _rate(ocr_structured_contract_task_count, ocr_task_count),
            "ocr_table_context_task_rate": _rate(ocr_table_context_task_count, ocr_task_count),
            "ocr_table_context_ready_rate": _rate(
                ocr_table_context_ready_task_count,
                ocr_table_context_task_count,
            ),
            "ocr_formula_context_task_rate": _rate(ocr_formula_context_task_count, ocr_task_count),
            "ocr_formula_context_ready_rate": _rate(
                ocr_formula_context_ready_task_count,
                ocr_formula_context_task_count,
            ),
            "ocr_result_payload_valid_rate": _rate(
                ocr_result_payload_count,
                ocr_result_payload_count + ocr_invalid_result_count,
            ),
            "ocr_executor_success_rate": _rate(
                ocr_executor_succeeded_task_count,
                ocr_executor_attempted_task_count,
            ),
            "ocr_task_result_coverage_rate": _rate(ocr_result_count, ocr_task_count),
            "ocr_result_acceptance_rate": _rate(ocr_accepted_result_count, ocr_result_count),
            "ocr_writeback_apply_rate": _rate(
                ocr_block_writeback_count + ocr_page_writeback_count,
                ocr_task_count,
            ),
            "ocr_structured_result_writeback_rate": _rate(
                ocr_structured_result_writeback_count,
                ocr_accepted_result_count,
            ),
            "ocr_structured_result_candidate_rate": _rate(
                ocr_structured_result_candidate_count,
                ocr_candidate_qa_count,
            ),
            "ocr_structured_cells_candidate_rate": _rate(
                ocr_structured_cells_candidate_count,
                ocr_candidate_qa_count,
            ),
            "ocr_cell_bboxes_candidate_rate": _rate(
                ocr_cell_bboxes_candidate_count,
                ocr_candidate_qa_count,
            ),
            "ocr_structured_table_gate_pass_rate": _rate(
                ocr_structured_table_gate_passed_count,
                ocr_structured_table_candidate_count,
            ),
            "ocr_structured_table_gate_review_rate": _rate(
                ocr_structured_table_gate_review_count,
                ocr_structured_table_candidate_count,
            ),
            "ocr_structured_table_structure_review_rate": _rate(
                ocr_structured_table_gate_review_count + ocr_structured_table_gate_blocked_count,
                ocr_structured_table_candidate_count,
            ),
            "ocr_structured_table_row_col_match_rate": _rate(
                max(ocr_structured_table_candidate_count - ocr_structured_table_row_col_mismatch_count, 0),
                ocr_structured_table_candidate_count,
            ),
            "ocr_table_cell_bbox_coverage_rate": _rate(
                min(ocr_cell_bbox_count, ocr_structured_cell_count),
                ocr_structured_cell_count,
            ),
            "ocr_structured_formula_gate_pass_rate": _rate(
                ocr_structured_formula_gate_passed_count,
                ocr_structured_formula_candidate_count,
            ),
            "ocr_structured_formula_gate_review_rate": _rate(
                ocr_structured_formula_gate_review_count,
                ocr_structured_formula_candidate_count,
            ),
            "ocr_structured_formula_structure_review_rate": _rate(
                ocr_structured_formula_gate_review_count + ocr_structured_formula_gate_blocked_count,
                ocr_structured_formula_candidate_count,
            ),
            "ocr_formula_token_per_candidate": _rate(
                ocr_structured_formula_token_count,
                ocr_structured_formula_candidate_count,
            ),
            "ocr_formula_equation_label_per_candidate": _rate(
                ocr_structured_formula_equation_label_count,
                ocr_structured_formula_candidate_count,
            ),
            "ocr_candidate_promotable_rate": _rate(ocr_candidate_promotable_count, ocr_candidate_qa_count),
            "ocr_candidate_blocked_rate": _rate(ocr_candidate_blocked_count, ocr_candidate_qa_count),
            "ocr_candidate_promotion_rate": _rate(ocr_candidate_promoted_count, ocr_candidate_qa_count),
            "ocr_candidate_eligible_promotion_rate": _rate(
                ocr_candidate_promoted_count,
                ocr_candidate_promotion_eligible_count,
            ),
            "ocr_canonical_structure_promotion_rate": _rate(
                ocr_canonical_structure_promotion_count,
                ocr_candidate_promoted_count,
            ),
            "ocr_structured_table_promotion_rate": _rate(
                ocr_structured_table_promotion_count,
                ocr_structured_table_gate_passed_count,
            ),
            "ocr_structured_formula_promotion_rate": _rate(
                ocr_structured_formula_promotion_count,
                ocr_structured_formula_gate_passed_count,
            ),
            "qa_issue_per_chunk": _rate(translation_issue_count, chunk_count),
            "translated_pdf_structure_context_chunk_rate": _rate(
                translated_pdf_structure_context_chunk_count,
                translated_pdf_chunk_count or chunk_count,
            ),
            "translation_request_per_chunk": _rate(translation_request_count, chunk_count),
            "http_attempt_per_translation_request": _rate(
                http_attempt_count,
                translation_request_count,
            ),
            "http_retry_rate": _rate(http_retry_count, http_attempt_count),
            "billable_request_per_chunk": _rate(billable_request_count, chunk_count),
            "estimated_request_tokens_per_chunk": _rate(
                estimated_request_token_count,
                translation_request_count,
            ),
            "estimated_cost_per_chunk": _rate(estimated_total_cost, chunk_count),
        },
        "breakdowns": {
            "block_counts": block_counts,
            "entity_type_counts": entity_type_counts,
            "continuation_kind_counts": continuation_kind_counts,
            "stitch_action_counts": stitch_action_counts,
            "vision_action_counts": vision_action_counts,
            "vision_risk_counts": vision_risk_counts,
            "ocr_task_scope_counts": ocr_task_scope_counts,
            "ocr_task_status_counts": ocr_task_status_counts,
            "ocr_task_priority_counts": ocr_task_priority_counts,
            "ocr_task_engine_counts": ocr_task_engine_counts,
            "ocr_task_block_type_counts": ocr_task_block_type_counts,
            "ocr_task_structure_target_counts": ocr_task_structure_target_counts,
            "ocr_result_payload_status_counts": ocr_result_payload_status_counts,
            "ocr_result_payload_engine_counts": ocr_result_payload_engine_counts,
            "ocr_execution_status_counts": ocr_execution_status_counts,
            "ocr_writeback_status_counts": ocr_writeback_status_counts,
            "ocr_writeback_engine_counts": ocr_writeback_engine_counts,
            "ocr_writeback_rejection_counts": ocr_writeback_rejection_counts,
            "ocr_writeback_structured_result_field_counts": ocr_writeback_structured_result_field_counts,
            "ocr_candidate_status_counts": ocr_candidate_status_counts,
            "ocr_candidate_issue_counts": ocr_candidate_issue_counts,
            "ocr_candidate_structured_result_field_counts": ocr_candidate_structured_result_field_counts,
            "ocr_candidate_structured_table_gate_counts": ocr_candidate_structured_table_gate_counts,
            "ocr_candidate_structured_table_gate_issue_counts": ocr_candidate_structured_table_gate_issue_counts,
            "ocr_candidate_structured_formula_gate_counts": ocr_candidate_structured_formula_gate_counts,
            "ocr_candidate_structured_formula_gate_issue_counts": (
                ocr_candidate_structured_formula_gate_issue_counts
            ),
            "ocr_candidate_promotion_status_counts": ocr_candidate_promotion_status_counts,
            "ocr_candidate_promotion_skip_counts": ocr_candidate_promotion_skip_counts,
            "translation_issue_counts": translation_issue_counts,
            "translation_severity_counts": translation_severity_counts,
            "repair_action_counts": repair_action_counts,
            "repair_priority_counts": repair_priority_counts,
            "repair_scope_counts": repair_scope_counts,
            "repair_merge_strategy_counts": repair_merge_strategy_counts,
            "repair_merge_applied_strategy_counts": repair_merge_applied_strategy_counts,
            "repair_patch_review_default_decision_counts": repair_patch_review_decision_counts,
            "repair_patch_review_effective_decision_counts": repair_patch_review_effective_decision_counts,
            "repair_patch_review_human_decision_counts": repair_patch_review_human_decision_counts,
            "repair_patch_review_risk_counts": repair_patch_review_risk_counts,
            "repair_publish_status_counts": {repair_publish_status: 1} if repair_publish_status else {},
            "stage_elapsed_ms": stage_elapsed_ms,
            "stage_counts": stage_counts,
            "translator_counts": translator_counts,
            "skip_reasons": skip_reasons,
            "error_code_counts": run_error_code_counts,
            "error_category_counts": run_error_category_counts,
            "structure_hint_merged_cell_candidate_type_counts": structure_hint_merged_cell_candidate_type_counts,
            "structure_hint_merged_cell_candidate_reason_counts": structure_hint_merged_cell_candidate_reason_counts,
            "structure_hint_merged_cell_candidate_status_counts": structure_hint_merged_cell_candidate_status_counts,
            "structure_hint_merged_cell_candidate_visual_evidence_counts": (
                structure_hint_merged_cell_candidate_visual_evidence_counts
            ),
            "structure_hint_merged_cell_candidate_bbox_evidence_counts": (
                structure_hint_merged_cell_candidate_bbox_evidence_counts
            ),
            "table_merged_cell_candidate_type_counts": table_merged_cell_candidate_type_counts,
            "table_merged_cell_candidate_reason_counts": table_merged_cell_candidate_reason_counts,
            "table_merged_cell_candidate_status_counts": table_merged_cell_candidate_status_counts,
            "table_merged_cell_candidate_visual_evidence_counts": (
                table_merged_cell_candidate_visual_evidence_counts
            ),
            "table_merged_cell_candidate_bbox_evidence_counts": table_merged_cell_candidate_bbox_evidence_counts,
            "table_chain_reject_reason_counts": table_chain_reject_reason_counts,
            "table_chain_reject_reason_category_counts": table_chain_reject_reason_category_counts,
            "table_chain_warning_reason_counts": table_chain_warning_reason_counts,
            "table_chain_warning_reason_category_counts": table_chain_warning_reason_category_counts,
            "budget_split_reason_counts": budget_split_reason_counts,
            "budget_pressure_counts": budget_pressure_counts,
        },
        "evidence_files": dict(evidence_files or DEFAULT_EVIDENCE_FILES),
    }


def write_experiment_metrics(
    structure_qa: dict[str, Any] | None,
    vision_route: dict[str, Any] | None,
    translation_qa: dict[str, Any] | None,
    repair_plan: dict[str, Any] | None,
    path: Path,
    *,
    doc_id: str | None = None,
    pipeline_variant: str | None = None,
    chunk_boundary_qa: dict[str, Any] | None = None,
    chunk_strategy_comparison: dict[str, Any] | None = None,
    structure_hints_manifest: dict[str, Any] | None = None,
    table_reconstruction: dict[str, Any] | None = None,
    ocr_tasks: dict[str, Any] | None = None,
    ocr_results: dict[str, Any] | None = None,
    ocr_writeback: dict[str, Any] | None = None,
    ocr_candidate_qa: dict[str, Any] | None = None,
    ocr_candidate_promotion: dict[str, Any] | None = None,
    repair_requests: dict[str, Any] | None = None,
    repair_results: dict[str, Any] | None = None,
    repair_validation: dict[str, Any] | None = None,
    repair_merge: dict[str, Any] | None = None,
    repair_patch_review: dict[str, Any] | None = None,
    repair_merge_qa: dict[str, Any] | None = None,
    repair_publish: dict[str, Any] | None = None,
    translated_pdf_report: dict[str, Any] | None = None,
    run_metrics: dict[str, Any] | None = None,
    cost_estimate: dict[str, Any] | None = None,
    evidence_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    metrics = build_experiment_metrics(
        structure_qa,
        vision_route,
        translation_qa,
        repair_plan,
        doc_id=doc_id,
        pipeline_variant=pipeline_variant,
        chunk_boundary_qa=chunk_boundary_qa,
        chunk_strategy_comparison=chunk_strategy_comparison,
        structure_hints_manifest=structure_hints_manifest,
        table_reconstruction=table_reconstruction,
        ocr_tasks=ocr_tasks,
        ocr_results=ocr_results,
        ocr_writeback=ocr_writeback,
        ocr_candidate_qa=ocr_candidate_qa,
        ocr_candidate_promotion=ocr_candidate_promotion,
        repair_requests=repair_requests,
        repair_results=repair_results,
        repair_validation=repair_validation,
        repair_merge=repair_merge,
        repair_patch_review=repair_patch_review,
        repair_merge_qa=repair_merge_qa,
        repair_publish=repair_publish,
        translated_pdf_report=translated_pdf_report,
        run_metrics=run_metrics,
        cost_estimate=cost_estimate,
        evidence_files=evidence_files,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics
