from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "experiment-metrics-v1"

DEFAULT_EVIDENCE_FILES = {
    "structure_qa": "output/structure_qa.json",
    "vision_route": "output/vision_route.json",
    "translation_qa": "output/qa_report.json",
    "repair_plan": "output/repair_plan.json",
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
    evidence_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Aggregate pipeline QA artifacts into a patent-facing experiment summary."""
    structure_summary = _summary(structure_qa)
    vision_summary = _summary(vision_route)
    translation_summary = _summary(translation_qa)
    repair_summary = _summary(repair_plan)

    block_counts = _counter_dict(structure_summary.get("block_counts"))
    entity_type_counts = _counter_dict(structure_summary.get("entity_type_counts"))
    vision_action_counts = _counter_dict(vision_summary.get("action_counts"))
    vision_risk_counts = _counter_dict(vision_summary.get("risk_counts"))
    translation_issue_counts = _counter_dict(translation_summary.get("issue_counts"))
    translation_severity_counts = _counter_dict(translation_summary.get("severity_counts"))
    repair_action_counts = _counter_dict(repair_summary.get("action_counts"))
    repair_priority_counts = _counter_dict(repair_summary.get("priority_counts"))
    repair_scope_counts = _counter_dict(repair_summary.get("scope_counts"))

    page_count = _as_int(structure_summary.get("page_count")) or _as_int(vision_summary.get("page_count"))
    table_count = _as_int(structure_summary.get("table_count"))
    table_continuation_count = _as_int(structure_summary.get("table_continuation_count"))
    table_footnote_count = _as_int(structure_summary.get("table_footnote_count"))
    caption_count = _as_int(structure_summary.get("caption_count"))
    caption_orphan_count = _as_int(structure_summary.get("caption_orphan_count"))
    footnote_count = _as_int(structure_summary.get("footnote_count"))
    footnote_orphan_count = _as_int(structure_summary.get("footnote_orphan_count"))
    relationship_warning_count = _as_int(structure_summary.get("relationship_warning_count"))
    entity_candidate_count = _as_int(structure_summary.get("entity_candidate_count"))
    entity_unique_count = _as_int(structure_summary.get("entity_unique_count"))
    translation_entity_candidate_count = _as_int(translation_summary.get("entity_candidate_count"))
    missing_entity_token_count = _as_int(translation_summary.get("missing_entity_token_count"))
    page_boundary_fragment_count = _as_int(structure_summary.get("page_boundary_fragment_count"))
    page_boundary_fragment_rate = _as_float(structure_summary.get("page_boundary_fragment_rate"))
    routed_page_count = _as_int(vision_summary.get("routed_page_count"))
    ocr_candidate_page_count = (
        vision_action_counts.get("local_ocr", 0) + vision_action_counts.get("vlm_review", 0)
    )
    chunk_count = _as_int(translation_summary.get("chunk_count")) or _as_int(repair_summary.get("chunk_count"))
    translation_issue_count = _as_int(translation_summary.get("issue_count"))
    repair_item_count = _as_int(repair_summary.get("repair_item_count"))
    max_english_residual_ratio = _as_float(translation_summary.get("max_english_residual_ratio"))

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
            "table_count": table_count,
            "table_continuation_count": table_continuation_count,
            "table_footnote_count": table_footnote_count,
            "caption_orphan_count": caption_orphan_count,
            "footnote_orphan_count": footnote_orphan_count,
            "relationship_warning_count": relationship_warning_count,
            "entity_candidate_count": entity_candidate_count,
            "entity_unique_count": entity_unique_count,
            "translation_entity_candidate_count": translation_entity_candidate_count,
            "missing_entity_token_count": missing_entity_token_count,
            "page_boundary_fragment_count": page_boundary_fragment_count,
            "page_boundary_fragment_rate": page_boundary_fragment_rate,
            "routed_page_count": routed_page_count,
            "ocr_candidate_page_count": ocr_candidate_page_count,
            "translation_issue_count": translation_issue_count,
            "repair_item_count": repair_item_count,
            "max_english_residual_ratio": max_english_residual_ratio,
        },
        "rates": {
            "relationship_warning_rate": _rate(relationship_warning_count, relationship_total),
            "caption_orphan_rate": _rate(caption_orphan_count, caption_count),
            "footnote_orphan_rate": _rate(footnote_orphan_count, footnote_count),
            "table_continuation_rate": _rate(table_continuation_count, table_count),
            "entity_missing_rate": _rate(missing_entity_token_count, effective_entity_candidate_count),
            "repair_item_per_chunk": _rate(repair_item_count, chunk_count),
            "ocr_candidate_page_rate": _rate(ocr_candidate_page_count, page_count),
            "routed_page_rate": _rate(routed_page_count, page_count),
            "qa_issue_per_chunk": _rate(translation_issue_count, chunk_count),
        },
        "breakdowns": {
            "block_counts": block_counts,
            "entity_type_counts": entity_type_counts,
            "vision_action_counts": vision_action_counts,
            "vision_risk_counts": vision_risk_counts,
            "translation_issue_counts": translation_issue_counts,
            "translation_severity_counts": translation_severity_counts,
            "repair_action_counts": repair_action_counts,
            "repair_priority_counts": repair_priority_counts,
            "repair_scope_counts": repair_scope_counts,
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
    evidence_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    metrics = build_experiment_metrics(
        structure_qa,
        vision_route,
        translation_qa,
        repair_plan,
        doc_id=doc_id,
        pipeline_variant=pipeline_variant,
        evidence_files=evidence_files,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics
