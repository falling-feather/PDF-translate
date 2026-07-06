from __future__ import annotations

import csv
import json
import re
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import fitz

from pdf_translate import pipeline
from pdf_translate.config import AppConfig
from pdf_translate.extractors.document_ir import extract_entity_candidates

SCHEMA_VERSION = "batch-experiment-v1"
EVIDENCE_SCHEMA_VERSION = "batch-experiment-evidence-v1"
SAMPLE_MANIFEST_SCHEMA_VERSION = "experiment-sample-manifest-v1"
SAMPLE_COVERAGE_REQUIREMENTS = (
    {"category": "normal", "label": "普通英文论文", "minimum": 10},
    {"category": "table-heavy", "label": "表格密集论文", "minimum": 10},
    {"category": "formula-heavy", "label": "公式密集论文", "minimum": 5},
    {"category": "multi-column", "label": "多栏复杂论文", "minimum": 5},
    {"category": "scanned", "label": "扫描/低文本论文", "minimum": 5},
    {"category": "annotation-entity-heavy", "label": "注释/实体密集论文", "minimum": 5},
)

REVIEW_SCORE_FIELDS = [
    "human_score_markdown",
    "human_score_html",
    "human_score_pdf",
    "human_score_table_readability",
    "human_score_figure_footnote_layout",
    "human_score_terminology_consistency",
    "human_score_structure_coherence",
]

REVIEW_DECISION_FIELDS = [
    "include_in_patent_evidence",
    "patent_evidence_notes",
]

EVIDENCE_SCORE_FIELDS = ["human_score", *REVIEW_SCORE_FIELDS]

SUMMARY_FIELDS: dict[str, list[str]] = {
    "quality": [
        "page_count",
        "chunk_count",
        "structure_hint_chunk_count",
        "structure_hint_empty_chunk_count",
        "structure_hint_char_count",
        "structure_hint_avg_char_count",
        "structure_hint_max_char_count",
        "structure_hint_table_count",
        "structure_hint_merged_cell_candidate_count",
        "structure_hint_relationship_count",
        "structure_hint_relationship_cross_page_count",
        "structure_hint_entity_count",
        "table_count",
        "translation_issue_count",
        "source_formula_token_count",
        "missing_formula_token_count",
        "source_equation_label_count",
        "missing_equation_label_count",
        "table_shape_error_count",
        "table_cell_token_error_count",
        "missing_table_locked_token_count",
        "table_merged_cell_candidate_count",
        "table_ragged_table_count",
        "table_ragged_row_count",
        "table_empty_cell_count",
        "table_chain_reject_reason_count",
        "table_chain_warning_reason_count",
        "table_footnote_cell_binding_count",
        "table_footnote_unbound_count",
        "page_boundary_fragment_count",
        "split_boundary_count",
        "protected_boundary_count",
        "ocr_task_count",
        "ocr_structured_contract_task_count",
        "ocr_table_context_task_count",
        "ocr_table_context_ready_task_count",
        "ocr_formula_context_task_count",
        "ocr_formula_context_ready_task_count",
        "ocr_structured_result_writeback_count",
        "ocr_candidate_qa_count",
        "ocr_candidate_promotable_count",
        "ocr_candidate_needs_review_count",
        "ocr_candidate_blocked_count",
        "ocr_table_context_candidate_count",
        "ocr_structured_contract_candidate_count",
        "ocr_structured_result_candidate_count",
        "ocr_structured_cell_count",
        "ocr_cell_bbox_count",
        "ocr_structured_table_candidate_count",
        "ocr_structured_table_gate_passed_count",
        "ocr_structured_table_gate_review_count",
        "ocr_structured_table_gate_blocked_count",
        "ocr_structured_table_missing_locked_token_count",
        "ocr_structured_table_row_col_mismatch_count",
        "ocr_structured_table_missing_cell_bboxes_count",
        "ocr_structured_formula_candidate_count",
        "ocr_structured_formula_gate_passed_count",
        "ocr_structured_formula_gate_review_count",
        "ocr_structured_formula_gate_blocked_count",
        "ocr_structured_formula_token_count",
        "ocr_structured_formula_equation_label_count",
        "ocr_structured_formula_missing_equation_label_count",
        "ocr_candidate_promotion_eligible_count",
        "ocr_candidate_promoted_count",
        "ocr_candidate_promotion_skipped_count",
        "ocr_canonical_structure_promotion_count",
        "ocr_structured_table_promotion_count",
        "ocr_structured_formula_promotion_count",
        "repair_request_count",
        "repair_merge_applied_count",
        "repair_merge_table_targeted_patch_count",
        "repair_patch_review_count",
        "repair_patch_review_safe_count",
        "repair_patch_review_required_count",
        "repair_patch_review_blocking_count",
        "repair_patch_review_table_count",
        "repair_effectiveness_before_issue_count",
        "repair_effectiveness_after_issue_count",
        "repair_effectiveness_issue_delta",
        "repair_effectiveness_resolved_issue_count",
        "repair_effectiveness_persisted_issue_count",
        "repair_effectiveness_new_issue_count",
        "repair_effectiveness_improved_chunk_count",
        "repair_effectiveness_regressed_chunk_count",
        "repair_publish_confirmed",
        "repair_publish_published",
        "repair_publish_open_issue_count",
        "repair_rollback_available",
        "repair_rollback_confirmed",
        "repair_rollback_applied",
        "repair_rollback_matches_original",
        "repair_formal_replace_available",
        "repair_formal_replace_confirmed",
        "repair_formal_replace_replaced",
        "repair_formal_replace_matches_published",
        "repair_formal_replace_rollback_available",
        "repair_formal_rollback_available",
        "repair_formal_rollback_confirmed",
        "repair_formal_rollback_applied",
        "repair_formal_rollback_matches_backup",
        "translation_structure_relation_check_count",
        "translation_structure_relation_mismatch_count",
        "translation_structure_relation_missing_anchor_count",
        "translation_table_footnote_binding_check_count",
        "translation_table_footnote_binding_mismatch_count",
        "translation_table_footnote_binding_missing_cell_count",
        "post_repair_issue_count",
    ],
    "rates": [
        "table_reconstruction_ready_rate",
        "structure_hint_chunk_rate",
        "structure_hint_table_per_chunk",
        "structure_hint_merged_cell_candidate_per_chunk",
        "structure_hint_footnote_binding_per_chunk",
        "structure_hint_relationship_per_chunk",
        "structure_hint_entity_per_chunk",
        "structure_hint_locked_token_per_chunk",
        "table_merged_cell_candidate_rate",
        "table_ragged_table_rate",
        "table_empty_cell_rate",
        "table_footnote_cell_binding_rate",
        "table_footnote_unbound_rate",
        "table_chain_reject_reason_per_rejected_chain",
        "table_chain_warning_reason_per_candidate_chain",
        "table_cell_token_error_rate",
        "table_locked_token_missing_rate",
        "split_boundary_rate",
        "protected_boundary_rate",
        "active_split_reduction_rate_vs_baseline",
        "entity_missing_rate",
        "formula_token_missing_rate",
        "equation_label_missing_rate",
        "routed_page_rate",
        "ocr_ready_task_rate",
        "ocr_structured_contract_task_rate",
        "ocr_table_context_task_rate",
        "ocr_table_context_ready_rate",
        "ocr_formula_context_task_rate",
        "ocr_formula_context_ready_rate",
        "ocr_structured_result_writeback_rate",
        "ocr_structured_result_candidate_rate",
        "ocr_structured_table_gate_pass_rate",
        "ocr_structured_table_gate_review_rate",
        "ocr_structured_table_structure_review_rate",
        "ocr_structured_table_row_col_match_rate",
        "ocr_table_cell_bbox_coverage_rate",
        "ocr_structured_formula_gate_pass_rate",
        "ocr_structured_formula_gate_review_rate",
        "ocr_structured_formula_structure_review_rate",
        "ocr_candidate_promotable_rate",
        "ocr_candidate_blocked_rate",
        "ocr_candidate_promotion_rate",
        "ocr_candidate_eligible_promotion_rate",
        "ocr_canonical_structure_promotion_rate",
        "ocr_structured_table_promotion_rate",
        "ocr_structured_formula_promotion_rate",
        "qa_issue_per_chunk",
        "repair_merge_apply_rate",
        "repair_merge_table_targeted_patch_rate",
        "repair_patch_review_safe_rate",
        "repair_patch_review_required_rate",
        "repair_effectiveness_issue_reduction_rate",
        "repair_publish_rate",
        "repair_rollback_success_rate",
        "repair_formal_replace_success_rate",
        "repair_formal_rollback_success_rate",
        "translation_structure_relation_mismatch_rate",
        "translation_table_footnote_binding_mismatch_rate",
        "post_repair_issue_reduction_rate",
    ],
    "performance": [
        "total_elapsed_ms",
        "translation_elapsed_ms",
        "translation_request_count",
        "http_attempt_count",
        "http_retry_count",
        "estimated_request_token_count",
        "estimated_total_token_count",
        "estimated_total_cost",
        "estimated_cost_per_chunk",
    ],
    "breakdowns": [
        "table_chain_reject_reason_counts",
        "table_chain_reject_reason_category_counts",
        "table_chain_warning_reason_counts",
        "table_chain_warning_reason_category_counts",
        "structure_hint_merged_cell_candidate_type_counts",
        "structure_hint_merged_cell_candidate_reason_counts",
        "structure_hint_relationship_type_counts",
        "table_merged_cell_candidate_type_counts",
        "table_merged_cell_candidate_reason_counts",
        "ocr_task_structure_target_counts",
        "ocr_writeback_structured_result_field_counts",
        "ocr_candidate_status_counts",
        "ocr_candidate_issue_counts",
        "ocr_candidate_structured_result_field_counts",
        "ocr_candidate_structured_table_gate_counts",
        "ocr_candidate_structured_table_gate_issue_counts",
        "ocr_candidate_structured_formula_gate_counts",
        "ocr_candidate_structured_formula_gate_issue_counts",
        "ocr_candidate_promotion_status_counts",
        "ocr_candidate_promotion_skip_counts",
        "repair_merge_strategy_counts",
        "repair_merge_applied_strategy_counts",
        "repair_patch_review_default_decision_counts",
        "repair_patch_review_risk_counts",
        "repair_effectiveness_status_counts",
        "repair_publish_status_counts",
        "repair_rollback_status_counts",
        "repair_formal_replace_status_counts",
        "repair_formal_rollback_status_counts",
    ],
}

COMPARISON_FIELDS = [
    ("quality", "translation_issue_count"),
    ("quality", "missing_formula_token_count"),
    ("quality", "table_shape_error_count"),
    ("quality", "table_cell_token_error_count"),
    ("quality", "ocr_structured_table_gate_review_count"),
    ("rates", "split_boundary_rate"),
    ("rates", "protected_boundary_rate"),
    ("rates", "active_split_reduction_rate_vs_baseline"),
    ("rates", "structure_hint_relationship_per_chunk"),
    ("rates", "structure_hint_entity_per_chunk"),
    ("rates", "formula_token_missing_rate"),
    ("rates", "ocr_structured_table_gate_pass_rate"),
    ("rates", "ocr_structured_table_promotion_rate"),
    ("rates", "ocr_table_cell_bbox_coverage_rate"),
    ("rates", "ocr_structured_formula_gate_pass_rate"),
    ("rates", "ocr_structured_formula_promotion_rate"),
    ("rates", "repair_patch_review_required_rate"),
    ("rates", "repair_effectiveness_issue_reduction_rate"),
    ("rates", "repair_publish_rate"),
    ("rates", "repair_rollback_success_rate"),
    ("rates", "repair_formal_replace_success_rate"),
    ("rates", "repair_formal_rollback_success_rate"),
    ("rates", "translation_structure_relation_mismatch_rate"),
    ("rates", "translation_table_footnote_binding_mismatch_rate"),
    ("performance", "total_elapsed_ms"),
    ("performance", "translation_request_count"),
    ("performance", "estimated_total_cost"),
]

ORDERED_SAMPLE_METADATA_KEY = "__ordered_samples__"


@dataclass(frozen=True)
class ExperimentVariant:
    name: str
    chunk_strategy: Literal["page", "structure"]
    execute_ocr: bool = False
    execute_repair_requests: bool = False


@dataclass(frozen=True)
class ExperimentSample:
    source_pdf: Path
    sample_id: str
    pdf_type: str = ""
    tags: tuple[str, ...] = ()
    notes: str = ""
    include_in_patent_batch: str = ""
    reviewer: str = ""
    review_notes: str = ""


SAMPLE_MANIFEST_CSV_FIELDS = [
    "source_pdf",
    "sample_id",
    "pdf_type",
    "tags",
    "notes",
    "suggested_pdf_type",
    "suggested_tags",
    "confirmed_pdf_type",
    "confirmed_tags",
    "include_in_patent_batch",
    "reviewer",
    "review_notes",
]


def _safe_id(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "-", value.strip())
    text = re.sub(r"-{2,}", "-", text).strip("-._")
    return text or "item"


def _split_tags(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if not isinstance(value, str):
        return ()
    parts = re.split(r"[;,，；|]", value)
    return tuple(part.strip() for part in parts if part.strip())


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str):
            if value.strip():
                return value
        elif value:
            return value
    return ""


def _metadata_key(value: str | Path) -> str:
    return str(Path(value).expanduser()).replace("\\", "/").lower()


def load_sample_metadata(path: Path) -> dict[str, Any]:
    """Load sample metadata by path/name keys, with ordered fallback for pathless rows."""
    if not path.is_file():
        raise FileNotFoundError(path)
    base_dir = path.parent.resolve()
    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            rows = raw.get("samples", raw.get("items", []))
        else:
            rows = raw
        if not isinstance(rows, list):
            raise ValueError("sample metadata JSON must be a list or contain samples/items")
    elif suffix in (".csv", ".tsv"):
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f, delimiter=delimiter))
    else:
        raise ValueError("sample metadata must be .json, .csv, or .tsv")

    metadata: dict[str, Any] = {}
    ordered_items: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        source_pdf = row.get("source_pdf") or row.get("pdf") or row.get("path") or row.get("file")
        sample_id = str(row.get("sample_id") or "").strip()
        source_text = str(source_pdf).strip() if source_pdf else ""
        item = {
            "sample_id": sample_id,
            "pdf_type": str(
                _first_present(row, "confirmed_pdf_type", "pdf_type", "type", "suggested_pdf_type")
            ).strip(),
            "tags": _split_tags(_first_present(row, "confirmed_tags", "tags", "labels", "suggested_tags")),
            "notes": str(row.get("notes") or row.get("remark") or "").strip(),
            "suggested_pdf_type": str(row.get("suggested_pdf_type") or "").strip(),
            "suggested_tags": _split_tags(row.get("suggested_tags") or ""),
            "confirmed_pdf_type": str(row.get("confirmed_pdf_type") or "").strip(),
            "confirmed_tags": _split_tags(row.get("confirmed_tags") or ""),
            "include_in_patent_batch": str(row.get("include_in_patent_batch") or "").strip(),
            "reviewer": str(row.get("reviewer") or "").strip(),
            "review_notes": str(row.get("review_notes") or "").strip(),
            "_has_source_pdf": bool(source_text),
        }
        ordered_items.append(item)
        keys: set[str] = set()
        if source_text:
            source_path = Path(source_text)
            source_for_resolution = source_path if source_path.is_absolute() else base_dir / source_path
            keys.update(
                {
                    _metadata_key(source_text),
                    source_path.name.lower(),
                    source_path.stem.lower(),
                    _metadata_key(source_for_resolution),
                }
            )
            try:
                keys.add(str(source_for_resolution.resolve()).replace("\\", "/").lower())
            except OSError:
                pass
        if sample_id:
            keys.add(sample_id.lower())
        for key in keys:
            if key:
                metadata[key] = item
    if ordered_items:
        metadata[ORDERED_SAMPLE_METADATA_KEY] = {"items": ordered_items}
    return metadata


def _metadata_for_pdf(pdf: Path, metadata: dict[str, Any], index: int) -> dict[str, Any]:
    keys = [
        str(pdf.resolve()).replace("\\", "/").lower(),
        _metadata_key(pdf),
        pdf.name.lower(),
        pdf.stem.lower(),
    ]
    for key in keys:
        item = metadata.get(key)
        if isinstance(item, dict):
            return item
    ordered = metadata.get(ORDERED_SAMPLE_METADATA_KEY)
    if isinstance(ordered, dict):
        items = ordered.get("items")
        if isinstance(items, list) and index < len(items):
            item = items[index]
            if isinstance(item, dict) and not item.get("_has_source_pdf"):
                return item
    return {}


def _build_samples(pdfs: list[Path], sample_metadata: dict[str, Any] | None = None) -> list[ExperimentSample]:
    metadata = sample_metadata or {}
    samples: list[ExperimentSample] = []
    seen_ids: dict[str, int] = {}
    for index, pdf in enumerate(pdfs, start=1):
        item = _metadata_for_pdf(pdf, metadata, index - 1)
        base_id = str(item.get("sample_id") or f"{index:03d}-{_safe_id(pdf.stem)}").strip()
        sample_id = _safe_id(base_id)
        if sample_id in seen_ids:
            seen_ids[sample_id] += 1
            sample_id = f"{sample_id}-{seen_ids[sample_id]}"
        else:
            seen_ids[sample_id] = 1
        samples.append(
            ExperimentSample(
                source_pdf=pdf,
                sample_id=sample_id,
                pdf_type=str(item.get("pdf_type") or ""),
                tags=tuple(item.get("tags") or ()),
                notes=str(item.get("notes") or ""),
                include_in_patent_batch=str(item.get("include_in_patent_batch") or ""),
                reviewer=str(item.get("reviewer") or ""),
                review_notes=str(item.get("review_notes") or ""),
            )
        )
    return samples


def _sample_text_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _line_number_count(line: str) -> int:
    return len(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?", line))


def _table_keyword_count(text: str) -> int:
    return len(re.findall(r"\b(?:table|tab\.)\s*\d+|表\s*\d+", text, flags=re.IGNORECASE))


def _formula_marker_count(text: str) -> int:
    symbol_count = len(re.findall(r"[=±×÷≤≥≈∑∫√∞αβγδθλμσπΩΔ]", text))
    equation_labels = len(re.findall(r"(?:^|\s)\(\d{1,3}\)(?:\s|$)", text))
    return symbol_count + equation_labels


def _figure_caption_keyword_count(text: str) -> int:
    return len(re.findall(r"\bfig(?:ure)?\.?\s*\d+|图\s*\d+", text, flags=re.IGNORECASE))


def _annotation_marker_count(text: str) -> int:
    keyword_count = len(
        re.findall(
            r"\b(?:corresponding author|author contributions?|conflicts? of interest|"
            r"supplementary|present address|equal contribution|footnotes?|e-?mail|affiliations?)\b",
            text,
            flags=re.IGNORECASE,
        )
    )
    footnote_like = 0
    for line in _sample_text_lines(text):
        if re.match(
            r"^\s*(?:\d{1,2}|[*†‡])[\).]?\s+"
            r"(?:Department|School|College|University|Institute|Laborator|Corresponding|"
            r"Email|E-mail|These authors|Present address|Author contributions?|Conflict|"
            r"Supplementary|Note\b|p\s*[<=>])",
            line,
            flags=re.IGNORECASE,
        ):
            footnote_like += 1
    return keyword_count + footnote_like


def _multi_column_block_page(page: fitz.Page) -> bool:
    width = float(page.rect.width or 1)
    blocks = page.get_text("blocks") or []
    left = 0
    right = 0
    for block in blocks:
        if len(block) < 5:
            continue
        text = str(block[4] or "").strip()
        if len(text) < 30:
            continue
        x0 = float(block[0])
        x1 = float(block[2])
        center = (x0 + x1) / 2
        if center < width * 0.45:
            left += 1
        elif center > width * 0.55:
            right += 1
    return left >= 2 and right >= 2


def _sample_tags_from_metrics(metrics: dict[str, Any]) -> tuple[str, ...]:
    page_count = max(int(metrics.get("page_count") or 0), 1)
    low_text_page_count = int(metrics.get("low_text_page_count") or 0)
    image_page_count = int(metrics.get("image_page_count") or 0)
    table_keyword_count = int(metrics.get("table_keyword_count") or 0)
    table_like_row_count = int(metrics.get("table_like_row_count") or 0)
    formula_marker_count = int(metrics.get("formula_marker_count") or 0)
    multi_column_page_count = int(metrics.get("multi_column_page_count") or 0)
    text_char_count = int(metrics.get("text_char_count") or 0)
    annotation_marker_count = int(metrics.get("annotation_marker_count") or 0)
    figure_caption_count = int(metrics.get("figure_caption_count") or 0)
    entity_candidate_count = int(metrics.get("entity_candidate_count") or 0)
    organization_candidate_count = int(metrics.get("organization_candidate_count") or 0)
    person_candidate_count = int(metrics.get("person_candidate_count") or 0)
    model_dataset_candidate_count = int(metrics.get("model_dataset_candidate_count") or 0)

    tags: list[str] = []
    if text_char_count < page_count * 80 and (image_page_count or low_text_page_count >= page_count):
        tags.append("scanned")
    if table_keyword_count >= 2 or table_like_row_count >= max(3, page_count * 2):
        tags.append("table")
    if formula_marker_count >= max(8, page_count * 4):
        tags.append("formula")
    if multi_column_page_count >= max(1, round(page_count * 0.4)):
        tags.append("multi-column")
    if annotation_marker_count >= max(2, page_count) or figure_caption_count >= max(3, page_count * 2):
        tags.append("annotation")
    if entity_candidate_count >= max(6, page_count * 3) or (
        organization_candidate_count + person_candidate_count + model_dataset_candidate_count
    ) >= max(3, page_count * 2):
        tags.append("entity")
    if image_page_count:
        tags.append("image")
    if not tags:
        tags.append("normal")
    return tuple(dict.fromkeys(tags))


def _sample_type_from_tags(tags: tuple[str, ...]) -> str:
    if "scanned" in tags:
        return "scanned"
    if "table" in tags:
        return "table-heavy"
    if "formula" in tags:
        return "formula-heavy"
    if "multi-column" in tags:
        return "multi-column"
    if "annotation" in tags or "entity" in tags:
        return "annotation-entity-heavy"
    return "normal"


def _sample_matches_coverage(sample: dict[str, Any], category: str) -> bool:
    pdf_type = str(sample.get("pdf_type") or "")
    tags = {str(tag) for tag in sample.get("tags", []) or []}
    if category == "normal":
        return pdf_type == "normal"
    if category == "table-heavy":
        return pdf_type == "table-heavy" or "table" in tags
    if category == "formula-heavy":
        return pdf_type == "formula-heavy" or "formula" in tags
    if category == "multi-column":
        return pdf_type == "multi-column" or "multi-column" in tags
    if category == "scanned":
        return pdf_type == "scanned" or "scanned" in tags
    if category == "annotation-entity-heavy":
        return pdf_type == "annotation-entity-heavy" or "annotation" in tags or "entity" in tags
    return False


def _build_sample_coverage(samples: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        str(requirement["category"]): sum(
            1 for sample in samples if _sample_matches_coverage(sample, str(requirement["category"]))
        )
        for requirement in SAMPLE_COVERAGE_REQUIREMENTS
    }
    requirements: list[dict[str, Any]] = []
    for requirement in SAMPLE_COVERAGE_REQUIREMENTS:
        category = str(requirement["category"])
        minimum = int(requirement["minimum"])
        count = counts.get(category, 0)
        missing = max(0, minimum - count)
        requirements.append(
            {
                "category": category,
                "label": requirement["label"],
                "minimum": minimum,
                "count": count,
                "missing": missing,
                "status": "met" if missing == 0 else "missing",
            }
        )
    missing_counts = {item["category"]: item["missing"] for item in requirements if int(item["missing"]) > 0}
    return {
        "recommended_minimums": {
            str(requirement["category"]): int(requirement["minimum"])
            for requirement in SAMPLE_COVERAGE_REQUIREMENTS
        },
        "counts": counts,
        "missing_counts": missing_counts,
        "requirement_count": len(requirements),
        "met_requirement_count": sum(1 for item in requirements if item["status"] == "met"),
        "ready_for_patent_batch": not missing_counts,
        "requirements": requirements,
    }


def _markdown_cell(value: Any) -> str:
    text = str(value or "").replace("\n", " ").replace("\r", " ")
    return text.replace("|", "\\|")


def _sample_metric_summary(metrics: dict[str, Any]) -> str:
    parts = [
        f"pages={metrics.get('page_count', 0)}",
        f"chars={metrics.get('text_char_count', 0)}",
        f"tables={metrics.get('table_like_row_count', 0)}",
        f"formula={metrics.get('formula_marker_count', 0)}",
        f"low_text={metrics.get('low_text_page_count', 0)}",
        f"annotations={metrics.get('annotation_marker_count', 0)}",
        f"entities={metrics.get('entity_candidate_count', 0)}",
    ]
    return ", ".join(parts)


def write_sample_manifest_markdown(manifest: dict[str, Any], path: Path) -> Path:
    summary = manifest.get("summary", {}) if isinstance(manifest.get("summary"), dict) else {}
    coverage = summary.get("coverage", {}) if isinstance(summary.get("coverage"), dict) else {}
    lines = [
        "# 跑批样本覆盖度报告",
        "",
        f"- 生成时间：{manifest.get('created_at', '')}",
        f"- 样本数：{manifest.get('sample_count', 0)}",
    ]
    if coverage:
        lines.extend(
            [
                f"- 覆盖建议达成：{coverage.get('met_requirement_count', 0)}/"
                f"{coverage.get('requirement_count', 0)}",
                f"- 是否达到申请前建议样本量：{'是' if coverage.get('ready_for_patent_batch') else '否'}",
            ]
        )

    lines.extend(["", "## 覆盖度缺口", "", "| 类型 | 建议数量 | 当前数量 | 仍缺 | 状态 |", "| --- | ---: | ---: | ---: | --- |"])
    for item in (coverage.get("requirements", []) if isinstance(coverage.get("requirements"), list) else []):
        if not isinstance(item, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(item.get("label") or item.get("category")),
                    str(item.get("minimum", 0)),
                    str(item.get("count", 0)),
                    str(item.get("missing", 0)),
                    _markdown_cell("已满足" if item.get("status") == "met" else "需补样本"),
                ]
            )
            + " |"
        )

    lines.extend(["", "## 类型分布", "", "| 类型 | 数量 |", "| --- | ---: |"])
    for pdf_type, count in (summary.get("pdf_type_counts") or {}).items():
        lines.append(f"| {_markdown_cell(pdf_type)} | {count} |")

    lines.extend(["", "## 标签分布", "", "| 标签 | 数量 |", "| --- | ---: |"])
    for tag, count in (summary.get("tag_counts") or {}).items():
        lines.append(f"| {_markdown_cell(tag)} | {count} |")

    lines.extend(
        [
            "",
            "## 样本清单",
            "",
            "| 样本 ID | 文件 | 建议类型 | 确认类型 | 标签 | 纳入批量 | 关键指标 | 备注 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for sample in (manifest.get("samples", []) if isinstance(manifest.get("samples"), list) else []):
        if not isinstance(sample, dict):
            continue
        metrics = sample.get("metrics", {}) if isinstance(sample.get("metrics"), dict) else {}
        display_tags = sample.get("confirmed_tags") or sample.get("tags") or []
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(sample.get("sample_id")),
                    _markdown_cell(sample.get("source_pdf")),
                    _markdown_cell(sample.get("suggested_pdf_type") or sample.get("pdf_type")),
                    _markdown_cell(sample.get("confirmed_pdf_type")),
                    _markdown_cell(", ".join(display_tags)),
                    _markdown_cell(sample.get("include_in_patent_batch")),
                    _markdown_cell(_sample_metric_summary(metrics)),
                    _markdown_cell(sample.get("notes")),
                ]
            )
            + " |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def analyze_pdf_sample(pdf: Path, *, max_pages: int = 20) -> dict[str, Any]:
    doc = fitz.open(pdf)
    try:
        page_count = len(doc)
        inspected_pages = min(page_count, max_pages)
        text_char_count = 0
        low_text_page_count = 0
        image_page_count = 0
        image_count = 0
        table_keyword_total = 0
        table_like_row_count = 0
        formula_marker_total = 0
        multi_column_page_count = 0
        figure_caption_total = 0
        annotation_marker_total = 0
        entity_keys: set[tuple[str, str]] = set()
        entity_type_counts: dict[str, int] = {}

        for index in range(inspected_pages):
            page = doc[index]
            text = page.get_text("text") or ""
            lines = _sample_text_lines(text)
            text_chars = len(text.strip())
            text_char_count += text_chars
            if text_chars < 80:
                low_text_page_count += 1
            page_images = len(page.get_images(full=True))
            image_count += page_images
            if page_images:
                image_page_count += 1
            table_keyword_total += _table_keyword_count(text)
            table_like_row_count += sum(1 for line in lines if _line_number_count(line) >= 3)
            formula_marker_total += _formula_marker_count(text)
            figure_caption_total += _figure_caption_keyword_count(text)
            annotation_marker_total += _annotation_marker_count(text)
            for entity in extract_entity_candidates(text):
                entity_type = str(entity.get("type") or "unknown")
                entity_text = str(entity.get("text") or "").casefold()
                key = (entity_type, entity_text)
                if entity_text and key not in entity_keys:
                    entity_keys.add(key)
                    entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
            if _multi_column_block_page(page):
                multi_column_page_count += 1

        metrics = {
            "page_count": page_count,
            "inspected_page_count": inspected_pages,
            "text_char_count": text_char_count,
            "avg_text_chars_per_page": round(text_char_count / inspected_pages, 2) if inspected_pages else 0,
            "low_text_page_count": low_text_page_count,
            "image_page_count": image_page_count,
            "image_count": image_count,
            "table_keyword_count": table_keyword_total,
            "table_like_row_count": table_like_row_count,
            "formula_marker_count": formula_marker_total,
            "multi_column_page_count": multi_column_page_count,
            "figure_caption_count": figure_caption_total,
            "annotation_marker_count": annotation_marker_total,
            "entity_candidate_count": len(entity_keys),
            "entity_type_counts": dict(sorted(entity_type_counts.items())),
            "organization_candidate_count": entity_type_counts.get("organization", 0),
            "person_candidate_count": entity_type_counts.get("person", 0),
            "model_dataset_candidate_count": entity_type_counts.get("model_or_dataset", 0),
        }
        tags = _sample_tags_from_metrics(metrics)
        pdf_type = _sample_type_from_tags(tags)
        return {
            "source_pdf": str(pdf),
            "pdf_type": pdf_type,
            "tags": list(tags),
            "notes": (
                f"auto: pages={page_count}, inspected={inspected_pages}, "
                f"chars={text_char_count}, table_rows={table_like_row_count}, "
                f"formula_markers={formula_marker_total}, annotations={annotation_marker_total}, "
                f"entities={len(entity_keys)}, low_text_pages={low_text_page_count}"
            ),
            "metrics": metrics,
        }
    finally:
        doc.close()


def _relative_manifest_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def build_sample_manifest(
    pdfs: list[Path],
    *,
    base_dir: Path | None = None,
    max_pages: int = 20,
) -> dict[str, Any]:
    base = base_dir or Path.cwd()
    seen_ids: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    for index, pdf in enumerate(pdfs, start=1):
        analysis = analyze_pdf_sample(pdf, max_pages=max_pages)
        sample_id = _safe_id(f"{index:03d}-{pdf.stem}")
        if sample_id in seen_ids:
            seen_ids[sample_id] += 1
            sample_id = f"{sample_id}-{seen_ids[sample_id]}"
        else:
            seen_ids[sample_id] = 1
        samples.append(
            {
                "source_pdf": _relative_manifest_path(pdf, base),
                "sample_id": sample_id,
                "pdf_type": analysis["pdf_type"],
                "tags": analysis["tags"],
                "notes": analysis["notes"],
                "suggested_pdf_type": analysis["pdf_type"],
                "suggested_tags": analysis["tags"],
                "confirmed_pdf_type": "",
                "confirmed_tags": [],
                "include_in_patent_batch": "",
                "reviewer": "",
                "review_notes": "",
                "metrics": analysis["metrics"],
            }
        )

    type_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    for sample in samples:
        pdf_type = str(sample.get("pdf_type") or "unknown")
        type_counts[pdf_type] = type_counts.get(pdf_type, 0) + 1
        for tag in sample.get("tags", []) or []:
            tag_counts[str(tag)] = tag_counts.get(str(tag), 0) + 1

    return {
        "schema_version": SAMPLE_MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(samples),
        "summary": {
            "pdf_type_counts": dict(sorted(type_counts.items())),
            "tag_counts": dict(sorted(tag_counts.items())),
            "coverage": _build_sample_coverage(samples),
        },
        "samples": samples,
    }


def write_sample_manifest(
    pdfs: list[Path],
    path: Path,
    *,
    report_path: Path | None = None,
    markdown_path: Path | None = None,
    max_pages: int = 20,
) -> dict[str, Any]:
    manifest = build_sample_manifest(pdfs, base_dir=path.parent, max_pages=max_pages)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SAMPLE_MANIFEST_CSV_FIELDS)
        writer.writeheader()
        for sample in manifest["samples"]:
            writer.writerow(
                {
                    "source_pdf": sample["source_pdf"],
                    "sample_id": sample["sample_id"],
                    "pdf_type": sample["pdf_type"],
                    "tags": ";".join(sample.get("tags", []) or []),
                    "notes": sample["notes"],
                    "suggested_pdf_type": sample.get("suggested_pdf_type", ""),
                    "suggested_tags": ";".join(sample.get("suggested_tags", []) or []),
                    "confirmed_pdf_type": sample.get("confirmed_pdf_type", ""),
                    "confirmed_tags": ";".join(sample.get("confirmed_tags", []) or []),
                    "include_in_patent_batch": sample.get("include_in_patent_batch", ""),
                    "reviewer": sample.get("reviewer", ""),
                    "review_notes": sample.get("review_notes", ""),
                }
            )
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if markdown_path:
        write_sample_manifest_markdown(manifest, markdown_path)
    return manifest


def parse_variant_spec(spec: str) -> ExperimentVariant:
    raw = spec.strip().lower().replace("_", "-")
    if not raw:
        raise ValueError("empty experiment variant")
    parts = [part for part in raw.split("+") if part]
    base = parts[0]
    if base not in ("page", "structure"):
        raise ValueError(f"unknown experiment variant base: {base}")

    flags = set(parts[1:])
    known_flags = {"ocr", "execute-ocr", "repair", "repairs", "execute-repairs"}
    unknown_flags = flags - known_flags
    if unknown_flags:
        raise ValueError(f"unknown experiment variant flags: {', '.join(sorted(unknown_flags))}")

    execute_ocr = bool(flags & {"ocr", "execute-ocr"})
    execute_repair_requests = bool(flags & {"repair", "repairs", "execute-repairs"})
    name_parts = [base]
    if execute_ocr:
        name_parts.append("ocr")
    if execute_repair_requests:
        name_parts.append("repair")
    return ExperimentVariant(
        name="+".join(name_parts),
        chunk_strategy=base,  # type: ignore[arg-type]
        execute_ocr=execute_ocr,
        execute_repair_requests=execute_repair_requests,
    )


def parse_variant_specs(specs: str | list[str] | tuple[str, ...]) -> list[ExperimentVariant]:
    if isinstance(specs, str):
        raw_specs = [part.strip() for part in specs.split(",")]
    else:
        raw_specs = []
        for item in specs:
            raw_specs.extend(part.strip() for part in item.split(","))

    variants: list[ExperimentVariant] = []
    seen: set[str] = set()
    for raw in raw_specs:
        if not raw:
            continue
        variant = parse_variant_spec(raw)
        if variant.name not in seen:
            variants.append(variant)
            seen.add(variant.name)
    if not variants:
        raise ValueError("at least one experiment variant is required")
    return variants


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_subset(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for group, fields in SUMMARY_FIELDS.items():
        raw_group = metrics.get(group)
        if not isinstance(raw_group, dict):
            raw_group = {}
        summary[group] = {field: raw_group.get(field, 0) for field in fields}
    return summary


def _metric_value(record: dict[str, Any], group: str, field: str) -> Any:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return None
    raw_group = metrics.get(group)
    if not isinstance(raw_group, dict):
        return None
    return raw_group.get(field)


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _merge_counter_dicts(values: list[Any]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        for key, count in value.items():
            if isinstance(count, bool):
                continue
            if isinstance(count, (int, float)):
                merged[str(key)] = merged.get(str(key), 0) + int(count)
    return dict(sorted(merged.items()))


def _aggregate_records(records: list[dict[str, Any]], variants: list[ExperimentVariant]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for variant in variants:
        variant_records = [record for record in records if record.get("variant") == variant.name]
        succeeded = [record for record in variant_records if record.get("status") == "succeeded"]
        averages: dict[str, dict[str, float]] = {}
        for group, fields in SUMMARY_FIELDS.items():
            if group == "breakdowns":
                continue
            averages[group] = {}
            for field in fields:
                values = [
                    number
                    for record in succeeded
                    if (number := _numeric(_metric_value(record, group, field))) is not None
                ]
                averages[group][field] = _mean(values)
        breakdowns = {
            field: _merge_counter_dicts(
                [
                    _metric_value(record, "breakdowns", field)
                    for record in succeeded
                ]
            )
            for field in SUMMARY_FIELDS.get("breakdowns", [])
        }
        aggregates.append(
            {
                "variant": variant.name,
                "chunk_strategy": variant.chunk_strategy,
                "execute_ocr": variant.execute_ocr,
                "execute_repair_requests": variant.execute_repair_requests,
                "run_count": len(variant_records),
                "succeeded_count": len(succeeded),
                "failed_count": len(variant_records) - len(succeeded),
                "averages": averages,
                "breakdowns": breakdowns,
            }
        )
    return aggregates


def _compare_to_baseline(records: list[dict[str, Any]], baseline_variant: str) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_sample.setdefault(str(record.get("sample_id", "")), []).append(record)

    for sample_id, sample_records in sorted(by_sample.items()):
        baseline = next(
            (
                record
                for record in sample_records
                if record.get("variant") == baseline_variant and record.get("status") == "succeeded"
            ),
            None,
        )
        if baseline is None:
            continue
        for record in sample_records:
            if record.get("variant") == baseline_variant or record.get("status") != "succeeded":
                continue
            deltas: dict[str, float] = {}
            for group, field in COMPARISON_FIELDS:
                before = _numeric(_metric_value(baseline, group, field))
                after = _numeric(_metric_value(record, group, field))
                if before is None or after is None:
                    continue
                deltas[f"{group}.{field}"] = round(after - before, 4)
            comparisons.append(
                {
                    "sample_id": sample_id,
                    "baseline_variant": baseline_variant,
                    "variant": record.get("variant"),
                    "deltas": deltas,
                }
            )
    return comparisons


def _record_paths(work_dir: Path, output_dir: Path) -> dict[str, str]:
    files = {
        "experiment_metrics": work_dir / "output" / "experiment_metrics.json",
        "run_metrics": work_dir / "output" / "run_metrics.json",
        "cost_estimate": work_dir / "output" / "cost_estimate.json",
        "translated_full": work_dir / "output" / "translated_full.md",
        "translated_pdf": work_dir / "output" / "translated_full.pdf",
        "bilingual_html": work_dir / "output" / "bilingual.html",
        "repair_patch_review": work_dir / "output" / "repair_patch_review.json",
        "repair_patch_review_md": work_dir / "output" / "repair_patch_review.md",
        "repair_effectiveness": work_dir / "output" / "repair_effectiveness.json",
        "repair_effectiveness_md": work_dir / "output" / "repair_effectiveness.md",
        "repair_publish": work_dir / "output" / "repair_publish.json",
        "repair_publish_md": work_dir / "output" / "repair_publish.md",
        "repair_rollback": work_dir / "output" / "repair_rollback.json",
        "repair_rollback_md": work_dir / "output" / "repair_rollback.md",
        "repair_formal_replace": work_dir / "output" / "repair_formal_replace.json",
        "repair_formal_replace_md": work_dir / "output" / "repair_formal_replace.md",
        "repair_formal_rollback": work_dir / "output" / "repair_formal_rollback.json",
        "repair_formal_rollback_md": work_dir / "output" / "repair_formal_rollback.md",
    }
    optional_files = {
        "repair_published_full": work_dir / "output" / "published_full.md",
        "repair_rollback_full": work_dir / "output" / "rollback_full.md",
        "repair_formal_full": work_dir / "output" / "formal_full.md",
        "repair_formal_backup_full": work_dir / "output" / "formal_full.before_repair.md",
        "repair_formal_active_before_rollback_full": work_dir / "output" / "formal_full.repair_applied.md",
    }
    result: dict[str, str] = {}
    for key, path in files.items():
        try:
            result[key] = str(path.relative_to(output_dir)).replace("\\", "/")
        except ValueError:
            result[key] = str(path)
    for key, path in optional_files.items():
        if not path.is_file():
            result[key] = ""
            continue
        try:
            result[key] = str(path.relative_to(output_dir)).replace("\\", "/")
        except ValueError:
            result[key] = str(path)
    return result


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _format_counter(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return ""
    return "; ".join(f"{key}:{count}" for key, count in sorted(value.items()))


def _extend_sample_coverage_markdown(lines: list[str], coverage: Any, *, heading: str) -> None:
    if not isinstance(coverage, dict) or not coverage:
        return
    requirements = coverage.get("requirements", [])
    if not isinstance(requirements, list):
        requirements = []
    missing = _format_counter(coverage.get("missing_counts")) or "none"
    ready_label = "yes" if coverage.get("ready_for_patent_batch") else "no"
    lines.extend(
        [
            "",
            f"## {heading}",
            "",
            f"- Requirements met: {coverage.get('met_requirement_count', 0)}/{coverage.get('requirement_count', 0)}",
            f"- Ready for patent batch: {ready_label}",
            f"- Missing counts: {missing}",
            "",
            "| Category | Current samples | Suggested minimum | Missing | Status |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for item in requirements:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "")
        category = str(item.get("category") or "")
        display = category if not label or label == category else f"{category} ({label})"
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(display),
                    _format_number(item.get("count", 0)),
                    _format_number(item.get("minimum", 0)),
                    _format_number(item.get("missing", 0)),
                    str(item.get("status", "")),
                ]
            )
            + " |"
        )


def write_batch_experiment_markdown(report: dict[str, Any], path: Path) -> Path:
    sample_filter = report.get("sample_filter", {}) if isinstance(report.get("sample_filter"), dict) else {}
    lines = [
        "# 批量实验汇总",
        "",
        f"- 生成时间：{report.get('created_at')}",
        f"- 输入 PDF 数：{report.get('input_pdf_count', report.get('sample_count'))}",
        f"- 样本数：{report.get('sample_count')}",
        f"- 仅运行人工纳入样本：{'是' if sample_filter.get('patent_batch_only') else '否'}",
        f"- 跳过样本数：{sample_filter.get('skipped_sample_count', 0)}",
        f"- 运行数：{report.get('run_count')}",
        f"- 成功数：{report.get('succeeded_count')}",
        f"- 失败数：{report.get('failed_count')}",
        f"- 人工评分表：{report.get('review_file', '')}",
        "",
        "## 策略均值",
        "",
        "| 策略 | 成功/总数 | 平均 issue | 平均合并候选 | 平均边界切开率 | 平均边界保护率 | 平均续表拒绝原因 | 续表拒绝类别 | 平均耗时 ms | 平均估算成本 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in report.get("aggregates", []):
        averages = item.get("averages", {})
        quality = averages.get("quality", {})
        rates = averages.get("rates", {})
        performance = averages.get("performance", {})
        breakdowns = item.get("breakdowns", {}) if isinstance(item.get("breakdowns"), dict) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("variant", "")),
                    f"{item.get('succeeded_count', 0)}/{item.get('run_count', 0)}",
                    _format_number(quality.get("translation_issue_count", 0)),
                    _format_number(quality.get("table_merged_cell_candidate_count", 0)),
                    _format_number(rates.get("split_boundary_rate", 0)),
                    _format_number(rates.get("protected_boundary_rate", 0)),
                    _format_number(quality.get("table_chain_reject_reason_count", 0)),
                    _format_counter(breakdowns.get("table_chain_reject_reason_category_counts")),
                    _format_number(performance.get("total_elapsed_ms", 0)),
                    _format_number(performance.get("estimated_total_cost", 0)),
                ]
            )
            + " |"
        )

    _extend_sample_coverage_markdown(
        lines,
        sample_filter.get("selected_coverage"),
        heading="Confirmed patent-batch sample coverage",
    )

    lines.extend(
        [
            "",
            "## OCR structured table gate",
            "",
            "| Variant | Avg structured table candidates | Avg gate pass rate | Avg review count | Avg bbox coverage rate | Gate issues |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in report.get("aggregates", []):
        averages = item.get("averages", {})
        quality = averages.get("quality", {})
        rates = averages.get("rates", {})
        breakdowns = item.get("breakdowns", {}) if isinstance(item.get("breakdowns"), dict) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("variant", "")),
                    _format_number(quality.get("ocr_structured_table_candidate_count", 0)),
                    _format_number(rates.get("ocr_structured_table_gate_pass_rate", 0)),
                    _format_number(quality.get("ocr_structured_table_gate_review_count", 0)),
                    _format_number(rates.get("ocr_table_cell_bbox_coverage_rate", 0)),
                    _format_counter(breakdowns.get("ocr_candidate_structured_table_gate_issue_counts")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## OCR structured formula gate",
            "",
            "| Variant | Avg structured formula candidates | Avg gate pass rate | Avg review count | Avg formula tokens | Avg equation labels | Gate issues |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in report.get("aggregates", []):
        averages = item.get("averages", {})
        quality = averages.get("quality", {})
        rates = averages.get("rates", {})
        breakdowns = item.get("breakdowns", {}) if isinstance(item.get("breakdowns"), dict) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("variant", "")),
                    _format_number(quality.get("ocr_structured_formula_candidate_count", 0)),
                    _format_number(rates.get("ocr_structured_formula_gate_pass_rate", 0)),
                    _format_number(quality.get("ocr_structured_formula_gate_review_count", 0)),
                    _format_number(quality.get("ocr_structured_formula_token_count", 0)),
                    _format_number(quality.get("ocr_structured_formula_equation_label_count", 0)),
                    _format_counter(breakdowns.get("ocr_candidate_structured_formula_gate_issue_counts")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 样本标签",
            "",
            "| 样本 | 类型 | 标签 | 纳入批量 | 确认人 | 确认备注 | 备注 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for sample in report.get("samples", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(sample.get("sample_id", "")),
                    str(sample.get("pdf_type", "")),
                    ", ".join(sample.get("tags", []) or []),
                    str(sample.get("include_in_patent_batch", "")),
                    str(sample.get("reviewer", "")),
                    str(sample.get("review_notes", "")),
                    str(sample.get("notes", "")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 单次运行",
            "",
            "| 样本 | 策略 | 状态 | 工作目录 | 指标文件 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for record in report.get("records", []):
        files = record.get("files", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record.get("sample_id", "")),
                    str(record.get("variant", "")),
                    str(record.get("status", "")),
                    str(record.get("work_dir", "")),
                    str(files.get("experiment_metrics", "")),
                ]
            )
            + " |"
        )

    if report.get("comparisons"):
        lines.extend(
            [
                "",
                "## 相对基线差值",
                "",
                "差值为“当前策略 - 基线策略”。负数通常表示 issue、耗时或成本下降；正数对保护率和降幅类指标通常更好。",
                "",
                "| 样本 | 基线 | 策略 | issue 差值 | 边界切开率差值 | 边界保护率差值 | 耗时差值 ms |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for item in report["comparisons"]:
            deltas = item.get("deltas", {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(item.get("sample_id", "")),
                        str(item.get("baseline_variant", "")),
                        str(item.get("variant", "")),
                        _format_number(deltas.get("quality.translation_issue_count", "")),
                        _format_number(deltas.get("rates.split_boundary_rate", "")),
                        _format_number(deltas.get("rates.protected_boundary_rate", "")),
                        _format_number(deltas.get("performance.total_elapsed_ms", "")),
                    ]
                )
                + " |"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_batch_experiment_review_csv(report: dict[str, Any], path: Path) -> Path:
    fieldnames = [
        "sample_id",
        "source_pdf",
        "pdf_type",
        "tags",
        "sample_include_in_patent_batch",
        "sample_reviewer",
        "sample_review_notes",
        "variant",
        "status",
        "translation_issue_count",
        "source_formula_token_count",
        "missing_formula_token_count",
        "formula_token_missing_rate",
        "source_equation_label_count",
        "missing_equation_label_count",
        "equation_label_missing_rate",
        "table_shape_error_count",
        "table_cell_token_error_count",
        "table_merged_cell_candidate_count",
        "table_merged_cell_candidate_types",
        "table_chain_reject_reason_count",
        "table_chain_reject_reason_categories",
        "ocr_structured_contract_task_count",
        "ocr_table_context_task_count",
        "ocr_structured_result_candidate_count",
        "ocr_structured_table_candidate_count",
        "ocr_structured_table_gate_passed_count",
        "ocr_structured_table_gate_review_count",
        "ocr_structured_table_gate_pass_rate",
        "ocr_structured_table_promotion_count",
        "ocr_structured_table_promotion_rate",
        "ocr_table_cell_bbox_coverage_rate",
        "ocr_structured_formula_candidate_count",
        "ocr_structured_formula_gate_passed_count",
        "ocr_structured_formula_gate_review_count",
        "ocr_structured_formula_gate_pass_rate",
        "ocr_structured_formula_promotion_count",
        "ocr_structured_formula_promotion_rate",
        "ocr_structured_formula_token_count",
        "ocr_structured_formula_equation_label_count",
        "ocr_candidate_structured_fields",
        "ocr_structured_table_gate_issues",
        "ocr_structured_formula_gate_issues",
        "repair_patch_review_count",
        "repair_patch_review_safe_count",
        "repair_patch_review_required_count",
        "repair_patch_review_blocking_count",
        "repair_patch_review_safe_rate",
        "repair_patch_review_required_rate",
        "repair_patch_review_default_decision_counts",
        "repair_patch_review_risk_counts",
        "repair_patch_review_report",
        "repair_effectiveness_before_issue_count",
        "repair_effectiveness_after_issue_count",
        "repair_effectiveness_issue_delta",
        "repair_effectiveness_resolved_issue_count",
        "repair_effectiveness_persisted_issue_count",
        "repair_effectiveness_new_issue_count",
        "repair_effectiveness_improved_chunk_count",
        "repair_effectiveness_regressed_chunk_count",
        "repair_effectiveness_issue_reduction_rate",
        "repair_effectiveness_status_counts",
        "repair_effectiveness_report",
        "repair_publish_confirmed",
        "repair_publish_published",
        "repair_publish_open_issue_count",
        "repair_publish_rate",
        "repair_publish_status_counts",
        "repair_publish_report",
        "repair_published_full",
        "repair_rollback_available",
        "repair_rollback_confirmed",
        "repair_rollback_applied",
        "repair_rollback_matches_original",
        "repair_rollback_success_rate",
        "repair_rollback_status_counts",
        "repair_rollback_report",
        "repair_rollback_full",
        "repair_formal_replace_available",
        "repair_formal_replace_confirmed",
        "repair_formal_replace_replaced",
        "repair_formal_replace_matches_published",
        "repair_formal_replace_success_rate",
        "repair_formal_replace_status_counts",
        "repair_formal_replace_report",
        "repair_formal_full",
        "repair_formal_backup_full",
        "repair_formal_rollback_available",
        "repair_formal_rollback_confirmed",
        "repair_formal_rollback_applied",
        "repair_formal_rollback_matches_backup",
        "repair_formal_rollback_success_rate",
        "repair_formal_rollback_status_counts",
        "repair_formal_rollback_report",
        "repair_formal_active_before_rollback_full",
        "split_boundary_rate",
        "protected_boundary_rate",
        "total_elapsed_ms",
        "estimated_total_cost",
        "translated_full",
        "translated_pdf",
        "bilingual_html",
        "human_score",
        *REVIEW_SCORE_FIELDS,
        *REVIEW_DECISION_FIELDS,
        "reviewer",
        "review_notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in report.get("records", []):
            metrics = record.get("metrics", {})
            quality = metrics.get("quality", {}) if isinstance(metrics, dict) else {}
            rates = metrics.get("rates", {}) if isinstance(metrics, dict) else {}
            performance = metrics.get("performance", {}) if isinstance(metrics, dict) else {}
            breakdowns = metrics.get("breakdowns", {}) if isinstance(metrics, dict) else {}
            files = record.get("files", {})
            writer.writerow(
                {
                    "sample_id": record.get("sample_id", ""),
                    "source_pdf": record.get("source_pdf", ""),
                    "pdf_type": record.get("pdf_type", ""),
                    "tags": ";".join(record.get("tags", []) or []),
                    "sample_include_in_patent_batch": record.get("include_in_patent_batch", ""),
                    "sample_reviewer": record.get("reviewer", ""),
                    "sample_review_notes": record.get("review_notes", ""),
                    "variant": record.get("variant", ""),
                    "status": record.get("status", ""),
                    "translation_issue_count": quality.get("translation_issue_count", ""),
                    "source_formula_token_count": quality.get("source_formula_token_count", ""),
                    "missing_formula_token_count": quality.get("missing_formula_token_count", ""),
                    "formula_token_missing_rate": rates.get("formula_token_missing_rate", ""),
                    "source_equation_label_count": quality.get("source_equation_label_count", ""),
                    "missing_equation_label_count": quality.get("missing_equation_label_count", ""),
                    "equation_label_missing_rate": rates.get("equation_label_missing_rate", ""),
                    "table_shape_error_count": quality.get("table_shape_error_count", ""),
                    "table_cell_token_error_count": quality.get("table_cell_token_error_count", ""),
                    "table_merged_cell_candidate_count": quality.get("table_merged_cell_candidate_count", ""),
                    "table_merged_cell_candidate_types": _format_counter(
                        breakdowns.get("table_merged_cell_candidate_type_counts", {})
                    ),
                    "table_chain_reject_reason_count": quality.get("table_chain_reject_reason_count", ""),
                    "table_chain_reject_reason_categories": _format_counter(
                        breakdowns.get("table_chain_reject_reason_category_counts", {})
                    ),
                    "ocr_structured_contract_task_count": quality.get("ocr_structured_contract_task_count", ""),
                    "ocr_table_context_task_count": quality.get("ocr_table_context_task_count", ""),
                    "ocr_structured_result_candidate_count": quality.get(
                        "ocr_structured_result_candidate_count",
                        "",
                    ),
                    "ocr_structured_table_candidate_count": quality.get(
                        "ocr_structured_table_candidate_count",
                        "",
                    ),
                    "ocr_structured_table_gate_passed_count": quality.get(
                        "ocr_structured_table_gate_passed_count",
                        "",
                    ),
                    "ocr_structured_table_gate_review_count": quality.get(
                        "ocr_structured_table_gate_review_count",
                        "",
                    ),
                    "ocr_structured_table_gate_pass_rate": rates.get("ocr_structured_table_gate_pass_rate", ""),
                    "ocr_structured_table_promotion_count": quality.get(
                        "ocr_structured_table_promotion_count",
                        "",
                    ),
                    "ocr_structured_table_promotion_rate": rates.get(
                        "ocr_structured_table_promotion_rate",
                        "",
                    ),
                    "ocr_table_cell_bbox_coverage_rate": rates.get("ocr_table_cell_bbox_coverage_rate", ""),
                    "ocr_structured_formula_candidate_count": quality.get(
                        "ocr_structured_formula_candidate_count",
                        "",
                    ),
                    "ocr_structured_formula_gate_passed_count": quality.get(
                        "ocr_structured_formula_gate_passed_count",
                        "",
                    ),
                    "ocr_structured_formula_gate_review_count": quality.get(
                        "ocr_structured_formula_gate_review_count",
                        "",
                    ),
                    "ocr_structured_formula_gate_pass_rate": rates.get(
                        "ocr_structured_formula_gate_pass_rate",
                        "",
                    ),
                    "ocr_structured_formula_promotion_count": quality.get(
                        "ocr_structured_formula_promotion_count",
                        "",
                    ),
                    "ocr_structured_formula_promotion_rate": rates.get(
                        "ocr_structured_formula_promotion_rate",
                        "",
                    ),
                    "ocr_structured_formula_token_count": quality.get("ocr_structured_formula_token_count", ""),
                    "ocr_structured_formula_equation_label_count": quality.get(
                        "ocr_structured_formula_equation_label_count",
                        "",
                    ),
                    "ocr_candidate_structured_fields": _format_counter(
                        breakdowns.get("ocr_candidate_structured_result_field_counts", {})
                    ),
                    "ocr_structured_table_gate_issues": _format_counter(
                        breakdowns.get("ocr_candidate_structured_table_gate_issue_counts", {})
                    ),
                    "ocr_structured_formula_gate_issues": _format_counter(
                        breakdowns.get("ocr_candidate_structured_formula_gate_issue_counts", {})
                    ),
                    "repair_patch_review_count": quality.get("repair_patch_review_count", ""),
                    "repair_patch_review_safe_count": quality.get("repair_patch_review_safe_count", ""),
                    "repair_patch_review_required_count": quality.get("repair_patch_review_required_count", ""),
                    "repair_patch_review_blocking_count": quality.get("repair_patch_review_blocking_count", ""),
                    "repair_patch_review_safe_rate": rates.get("repair_patch_review_safe_rate", ""),
                    "repair_patch_review_required_rate": rates.get("repair_patch_review_required_rate", ""),
                    "repair_patch_review_default_decision_counts": _format_counter(
                        breakdowns.get("repair_patch_review_default_decision_counts", {})
                    ),
                    "repair_patch_review_risk_counts": _format_counter(
                        breakdowns.get("repair_patch_review_risk_counts", {})
                    ),
                    "repair_patch_review_report": files.get("repair_patch_review", ""),
                    "repair_effectiveness_before_issue_count": quality.get(
                        "repair_effectiveness_before_issue_count",
                        "",
                    ),
                    "repair_effectiveness_after_issue_count": quality.get(
                        "repair_effectiveness_after_issue_count",
                        "",
                    ),
                    "repair_effectiveness_issue_delta": quality.get(
                        "repair_effectiveness_issue_delta",
                        "",
                    ),
                    "repair_effectiveness_resolved_issue_count": quality.get(
                        "repair_effectiveness_resolved_issue_count",
                        "",
                    ),
                    "repair_effectiveness_persisted_issue_count": quality.get(
                        "repair_effectiveness_persisted_issue_count",
                        "",
                    ),
                    "repair_effectiveness_new_issue_count": quality.get(
                        "repair_effectiveness_new_issue_count",
                        "",
                    ),
                    "repair_effectiveness_improved_chunk_count": quality.get(
                        "repair_effectiveness_improved_chunk_count",
                        "",
                    ),
                    "repair_effectiveness_regressed_chunk_count": quality.get(
                        "repair_effectiveness_regressed_chunk_count",
                        "",
                    ),
                    "repair_effectiveness_issue_reduction_rate": rates.get(
                        "repair_effectiveness_issue_reduction_rate",
                        "",
                    ),
                    "repair_effectiveness_status_counts": _format_counter(
                        breakdowns.get("repair_effectiveness_status_counts", {})
                    ),
                    "repair_effectiveness_report": files.get("repair_effectiveness", ""),
                    "repair_publish_confirmed": quality.get("repair_publish_confirmed", ""),
                    "repair_publish_published": quality.get("repair_publish_published", ""),
                    "repair_publish_open_issue_count": quality.get("repair_publish_open_issue_count", ""),
                    "repair_publish_rate": rates.get("repair_publish_rate", ""),
                    "repair_publish_status_counts": _format_counter(
                        breakdowns.get("repair_publish_status_counts", {})
                    ),
                    "repair_publish_report": files.get("repair_publish", ""),
                    "repair_published_full": files.get("repair_published_full", ""),
                    "repair_rollback_available": quality.get("repair_rollback_available", ""),
                    "repair_rollback_confirmed": quality.get("repair_rollback_confirmed", ""),
                    "repair_rollback_applied": quality.get("repair_rollback_applied", ""),
                    "repair_rollback_matches_original": quality.get("repair_rollback_matches_original", ""),
                    "repair_rollback_success_rate": rates.get("repair_rollback_success_rate", ""),
                    "repair_rollback_status_counts": _format_counter(
                        breakdowns.get("repair_rollback_status_counts", {})
                    ),
                    "repair_rollback_report": files.get("repair_rollback", ""),
                    "repair_rollback_full": files.get("repair_rollback_full", ""),
                    "repair_formal_replace_available": quality.get("repair_formal_replace_available", ""),
                    "repair_formal_replace_confirmed": quality.get("repair_formal_replace_confirmed", ""),
                    "repair_formal_replace_replaced": quality.get("repair_formal_replace_replaced", ""),
                    "repair_formal_replace_matches_published": quality.get(
                        "repair_formal_replace_matches_published",
                        "",
                    ),
                    "repair_formal_replace_success_rate": rates.get("repair_formal_replace_success_rate", ""),
                    "repair_formal_replace_status_counts": _format_counter(
                        breakdowns.get("repair_formal_replace_status_counts", {})
                    ),
                    "repair_formal_replace_report": files.get("repair_formal_replace", ""),
                    "repair_formal_full": files.get("repair_formal_full", ""),
                    "repair_formal_backup_full": files.get("repair_formal_backup_full", ""),
                    "repair_formal_rollback_available": quality.get("repair_formal_rollback_available", ""),
                    "repair_formal_rollback_confirmed": quality.get("repair_formal_rollback_confirmed", ""),
                    "repair_formal_rollback_applied": quality.get("repair_formal_rollback_applied", ""),
                    "repair_formal_rollback_matches_backup": quality.get(
                        "repair_formal_rollback_matches_backup",
                        "",
                    ),
                    "repair_formal_rollback_success_rate": rates.get("repair_formal_rollback_success_rate", ""),
                    "repair_formal_rollback_status_counts": _format_counter(
                        breakdowns.get("repair_formal_rollback_status_counts", {})
                    ),
                    "repair_formal_rollback_report": files.get("repair_formal_rollback", ""),
                    "repair_formal_active_before_rollback_full": files.get(
                        "repair_formal_active_before_rollback_full",
                        "",
                    ),
                    "split_boundary_rate": rates.get("split_boundary_rate", ""),
                    "protected_boundary_rate": rates.get("protected_boundary_rate", ""),
                    "total_elapsed_ms": performance.get("total_elapsed_ms", ""),
                    "estimated_total_cost": performance.get("estimated_total_cost", ""),
                    "translated_full": files.get("translated_full", ""),
                    "translated_pdf": files.get("translated_pdf", ""),
                    "bilingual_html": files.get("bilingual_html", ""),
                    "human_score": "",
                    **{field: "" for field in REVIEW_SCORE_FIELDS},
                    "include_in_patent_evidence": "",
                    "patent_evidence_notes": "",
                    "reviewer": "",
                    "review_notes": "",
                }
            )
    return path


def read_batch_experiment_review_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _review_text(value: Any) -> str:
    return str(value or "").strip()


def _review_truthy(value: Any) -> bool:
    text = _review_text(value).lower()
    return text in {"1", "true", "yes", "y", "include", "included", "pass", "ok", "是", "纳入", "采纳", "通过"}


def _review_bool(value: Any) -> bool | None:
    text = _review_text(value).lower()
    if text in {"1", "true", "yes", "y", "pass", "ok", "是", "已确认", "已发布", "通过"}:
        return True
    if text in {"0", "false", "no", "n", "fail", "否", "未确认", "未发布", "不通过"}:
        return False
    return None


def _review_number(value: Any) -> float | None:
    text = _review_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _review_has_content(row: dict[str, Any]) -> bool:
    review_fields = [
        *EVIDENCE_SCORE_FIELDS,
        *REVIEW_DECISION_FIELDS,
        "reviewer",
        "review_notes",
    ]
    return any(_review_text(row.get(field)) for field in review_fields)


def _score_average(rows: list[dict[str, Any]], field: str) -> dict[str, float | int]:
    values = [number for row in rows if (number := _review_number(row.get(field))) is not None]
    return {"average": _mean(values), "count": len(values)}


def _parse_counter_text(value: Any) -> dict[str, int]:
    text = _review_text(value)
    if not text:
        return {}
    result: dict[str, int] = {}
    for part in text.split(";"):
        item = part.strip()
        if not item or ":" not in item:
            continue
        key, raw_count = item.rsplit(":", 1)
        number = _review_number(raw_count)
        if number is None:
            continue
        result[key.strip()] = result.get(key.strip(), 0) + int(number)
    return dict(sorted(result.items()))


def _merge_counter_texts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return _merge_counter_dicts([_parse_counter_text(row.get(field)) for row in rows])


def _review_group_summary(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = _review_text(row.get(field)) or "unknown"
        groups.setdefault(key, []).append(row)

    result: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items()):
        result.append(
            {
                field: key,
                "run_count": len(group_rows),
                "reviewed_count": sum(1 for row in group_rows if _review_has_content(row)),
                "included_count": sum(1 for row in group_rows if _review_truthy(row.get("include_in_patent_evidence"))),
                "human_score": _score_average(group_rows, "human_score"),
                "table_readability_score": _score_average(group_rows, "human_score_table_readability"),
                "structure_coherence_score": _score_average(group_rows, "human_score_structure_coherence"),
                "ocr_structured_table_gate_pass_rate": _score_average(
                    group_rows,
                    "ocr_structured_table_gate_pass_rate",
                ),
                "ocr_structured_formula_gate_pass_rate": _score_average(
                    group_rows,
                    "ocr_structured_formula_gate_pass_rate",
                ),
                "ocr_table_cell_bbox_coverage_rate": _score_average(
                    group_rows,
                    "ocr_table_cell_bbox_coverage_rate",
                ),
            }
        )
    return result


def _review_sum(rows: list[dict[str, Any]], field: str) -> int:
    total = 0
    for row in rows:
        number = _review_number(row.get(field))
        if number is not None:
            total += int(number)
    return total


def _review_bool_count(rows: list[dict[str, Any]], field: str) -> int:
    return sum(1 for row in rows if _review_bool(row.get(field)) is True)


def _review_record_key(row: dict[str, Any]) -> tuple[str, str]:
    return (_review_text(row.get("sample_id")), _review_text(row.get("variant")))


def _review_record_key_with_source(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _review_text(row.get("sample_id")),
        _review_text(row.get("variant")),
        _metadata_key(_review_text(row.get("source_pdf"))),
    )


def _selected_record_metrics(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics", {}) if isinstance(record, dict) else {}
    quality = metrics.get("quality", {}) if isinstance(metrics, dict) else {}
    rates = metrics.get("rates", {}) if isinstance(metrics, dict) else {}
    performance = metrics.get("performance", {}) if isinstance(metrics, dict) else {}
    return {
        "translation_issue_count": quality.get("translation_issue_count"),
        "table_shape_error_count": quality.get("table_shape_error_count"),
        "table_cell_token_error_count": quality.get("table_cell_token_error_count"),
        "ocr_structured_table_gate_pass_rate": rates.get("ocr_structured_table_gate_pass_rate"),
        "ocr_table_cell_bbox_coverage_rate": rates.get("ocr_table_cell_bbox_coverage_rate"),
        "ocr_structured_formula_gate_pass_rate": rates.get("ocr_structured_formula_gate_pass_rate"),
        "ocr_structured_formula_candidate_count": quality.get("ocr_structured_formula_candidate_count"),
        "repair_publish_confirmed": quality.get("repair_publish_confirmed"),
        "repair_publish_published": quality.get("repair_publish_published"),
        "repair_publish_open_issue_count": quality.get("repair_publish_open_issue_count"),
        "repair_publish_rate": rates.get("repair_publish_rate"),
        "repair_rollback_available": quality.get("repair_rollback_available"),
        "repair_rollback_applied": quality.get("repair_rollback_applied"),
        "repair_rollback_success_rate": rates.get("repair_rollback_success_rate"),
        "repair_formal_replace_replaced": quality.get("repair_formal_replace_replaced"),
        "repair_formal_replace_success_rate": rates.get("repair_formal_replace_success_rate"),
        "repair_formal_rollback_applied": quality.get("repair_formal_rollback_applied"),
        "repair_formal_rollback_success_rate": rates.get("repair_formal_rollback_success_rate"),
        "repair_patch_review_count": quality.get("repair_patch_review_count"),
        "repair_patch_review_required_count": quality.get("repair_patch_review_required_count"),
        "repair_patch_review_blocking_count": quality.get("repair_patch_review_blocking_count"),
        "repair_patch_review_required_rate": rates.get("repair_patch_review_required_rate"),
        "repair_effectiveness_issue_delta": quality.get("repair_effectiveness_issue_delta"),
        "repair_effectiveness_resolved_issue_count": quality.get("repair_effectiveness_resolved_issue_count"),
        "repair_effectiveness_new_issue_count": quality.get("repair_effectiveness_new_issue_count"),
        "repair_effectiveness_regressed_chunk_count": quality.get("repair_effectiveness_regressed_chunk_count"),
        "repair_effectiveness_issue_reduction_rate": rates.get("repair_effectiveness_issue_reduction_rate"),
        "split_boundary_rate": rates.get("split_boundary_rate"),
        "protected_boundary_rate": rates.get("protected_boundary_rate"),
        "total_elapsed_ms": performance.get("total_elapsed_ms"),
        "estimated_total_cost": performance.get("estimated_total_cost"),
    }


def build_batch_experiment_evidence(
    report: dict[str, Any],
    review_rows: list[dict[str, Any]],
    *,
    summary_file: str = "",
    review_file: str = "",
) -> dict[str, Any]:
    records = report.get("records", [])
    records_by_source_key = {
        (
            str(record.get("sample_id", "")),
            str(record.get("variant", "")),
            _metadata_key(str(record.get("source_pdf", ""))),
        ): record
        for record in records
        if isinstance(record, dict)
    }
    records_by_pair_key = {
        (str(record.get("sample_id", "")), str(record.get("variant", ""))): record
        for record in records
        if isinstance(record, dict)
    }
    run_failures = [
        {
            "sample_id": record.get("sample_id", ""),
            "variant": record.get("variant", ""),
            "source_pdf": record.get("source_pdf", ""),
            "status": record.get("status", ""),
            "error": record.get("error", ""),
            "work_dir": record.get("work_dir", ""),
        }
        for record in records
        if isinstance(record, dict) and record.get("status") != "succeeded"
    ]

    evidence_candidates: list[dict[str, Any]] = []
    for index, row in enumerate(review_rows, start=1):
        if not _review_truthy(row.get("include_in_patent_evidence")):
            continue
        record = records_by_source_key.get(_review_record_key_with_source(row))
        if record is None:
            record = records_by_pair_key.get(_review_record_key(row), {})
        status = _review_text(row.get("status")) or str(record.get("status", ""))
        if status and status != "succeeded":
            continue
        files = record.get("files", {}) if isinstance(record, dict) else {}
        scores = {
            field: number
            for field in EVIDENCE_SCORE_FIELDS
            if (number := _review_number(row.get(field))) is not None
        }
        evidence_candidates.append(
            {
                "evidence_id": f"{_safe_id(_review_text(row.get('sample_id')))}-{_safe_id(_review_text(row.get('variant')))}-{index:03d}",
                "sample_id": _review_text(row.get("sample_id")),
                "variant": _review_text(row.get("variant")),
                "source_pdf": _review_text(row.get("source_pdf")),
                "pdf_type": _review_text(row.get("pdf_type")),
                "tags": _split_tags(row.get("tags")),
                "status": status,
                "work_dir": record.get("work_dir", "") if isinstance(record, dict) else "",
                "scores": scores,
                "include_in_patent_evidence": _review_text(row.get("include_in_patent_evidence")),
                "patent_evidence_notes": _review_text(row.get("patent_evidence_notes")),
                "reviewer": _review_text(row.get("reviewer")),
                "review_notes": _review_text(row.get("review_notes")),
                "ocr": {
                    "structured_table_candidate_count": _review_number(
                        row.get("ocr_structured_table_candidate_count")
                    ),
                    "structured_table_gate_pass_rate": _review_number(
                        row.get("ocr_structured_table_gate_pass_rate")
                    ),
                    "table_cell_bbox_coverage_rate": _review_number(
                        row.get("ocr_table_cell_bbox_coverage_rate")
                    ),
                    "structured_table_gate_issues": _parse_counter_text(
                        row.get("ocr_structured_table_gate_issues")
                    ),
                    "structured_formula_candidate_count": _review_number(
                        row.get("ocr_structured_formula_candidate_count")
                    ),
                    "structured_formula_gate_pass_rate": _review_number(
                        row.get("ocr_structured_formula_gate_pass_rate")
                    ),
                    "structured_formula_token_count": _review_number(
                        row.get("ocr_structured_formula_token_count")
                    ),
                    "structured_formula_equation_label_count": _review_number(
                        row.get("ocr_structured_formula_equation_label_count")
                    ),
                    "structured_formula_gate_issues": _parse_counter_text(
                        row.get("ocr_structured_formula_gate_issues")
                    ),
                },
                "repair_publish": {
                    "confirmed": _review_bool(row.get("repair_publish_confirmed")),
                    "published": _review_bool(row.get("repair_publish_published")),
                    "open_issue_count": _review_number(row.get("repair_publish_open_issue_count")),
                    "publish_rate": _review_number(row.get("repair_publish_rate")),
                    "status_counts": _parse_counter_text(row.get("repair_publish_status_counts")),
                    "report_file": _review_text(row.get("repair_publish_report"))
                    or str(files.get("repair_publish", "")),
                    "published_full_file": _review_text(row.get("repair_published_full"))
                    or str(files.get("repair_published_full", "")),
                },
                "repair_rollback": {
                    "available": _review_bool(row.get("repair_rollback_available")),
                    "confirmed": _review_bool(row.get("repair_rollback_confirmed")),
                    "applied": _review_bool(row.get("repair_rollback_applied")),
                    "matches_original": _review_bool(row.get("repair_rollback_matches_original")),
                    "success_rate": _review_number(row.get("repair_rollback_success_rate")),
                    "status_counts": _parse_counter_text(row.get("repair_rollback_status_counts")),
                    "report_file": _review_text(row.get("repair_rollback_report"))
                    or str(files.get("repair_rollback", "")),
                    "rollback_full_file": _review_text(row.get("repair_rollback_full"))
                    or str(files.get("repair_rollback_full", "")),
                },
                "repair_formal_replace": {
                    "available": _review_bool(row.get("repair_formal_replace_available")),
                    "confirmed": _review_bool(row.get("repair_formal_replace_confirmed")),
                    "replaced": _review_bool(row.get("repair_formal_replace_replaced")),
                    "matches_published": _review_bool(row.get("repair_formal_replace_matches_published")),
                    "success_rate": _review_number(row.get("repair_formal_replace_success_rate")),
                    "status_counts": _parse_counter_text(row.get("repair_formal_replace_status_counts")),
                    "report_file": _review_text(row.get("repair_formal_replace_report"))
                    or str(files.get("repair_formal_replace", "")),
                    "formal_full_file": _review_text(row.get("repair_formal_full"))
                    or str(files.get("repair_formal_full", "")),
                    "backup_full_file": _review_text(row.get("repair_formal_backup_full"))
                    or str(files.get("repair_formal_backup_full", "")),
                },
                "repair_formal_rollback": {
                    "available": _review_bool(row.get("repair_formal_rollback_available")),
                    "confirmed": _review_bool(row.get("repair_formal_rollback_confirmed")),
                    "applied": _review_bool(row.get("repair_formal_rollback_applied")),
                    "matches_backup": _review_bool(row.get("repair_formal_rollback_matches_backup")),
                    "success_rate": _review_number(row.get("repair_formal_rollback_success_rate")),
                    "status_counts": _parse_counter_text(row.get("repair_formal_rollback_status_counts")),
                    "report_file": _review_text(row.get("repair_formal_rollback_report"))
                    or str(files.get("repair_formal_rollback", "")),
                    "active_before_rollback_file": _review_text(
                        row.get("repair_formal_active_before_rollback_full")
                    )
                    or str(files.get("repair_formal_active_before_rollback_full", "")),
                },
                "repair_patch_review": {
                    "patch_count": _review_number(row.get("repair_patch_review_count")),
                    "safe_count": _review_number(row.get("repair_patch_review_safe_count")),
                    "required_count": _review_number(row.get("repair_patch_review_required_count")),
                    "blocking_count": _review_number(row.get("repair_patch_review_blocking_count")),
                    "safe_rate": _review_number(row.get("repair_patch_review_safe_rate")),
                    "required_rate": _review_number(row.get("repair_patch_review_required_rate")),
                    "default_decision_counts": _parse_counter_text(
                        row.get("repair_patch_review_default_decision_counts")
                    ),
                    "risk_counts": _parse_counter_text(row.get("repair_patch_review_risk_counts")),
                    "report_file": _review_text(row.get("repair_patch_review_report"))
                    or str(files.get("repair_patch_review", "")),
                },
                "repair_effectiveness": {
                    "before_issue_count": _review_number(row.get("repair_effectiveness_before_issue_count")),
                    "after_issue_count": _review_number(row.get("repair_effectiveness_after_issue_count")),
                    "issue_delta": _review_number(row.get("repair_effectiveness_issue_delta")),
                    "resolved_issue_count": _review_number(row.get("repair_effectiveness_resolved_issue_count")),
                    "persisted_issue_count": _review_number(row.get("repair_effectiveness_persisted_issue_count")),
                    "new_issue_count": _review_number(row.get("repair_effectiveness_new_issue_count")),
                    "improved_chunk_count": _review_number(row.get("repair_effectiveness_improved_chunk_count")),
                    "regressed_chunk_count": _review_number(row.get("repair_effectiveness_regressed_chunk_count")),
                    "issue_reduction_rate": _review_number(row.get("repair_effectiveness_issue_reduction_rate")),
                    "status_counts": _parse_counter_text(row.get("repair_effectiveness_status_counts")),
                    "report_file": _review_text(row.get("repair_effectiveness_report"))
                    or str(files.get("repair_effectiveness", "")),
                    "markdown_file": str(files.get("repair_effectiveness_md", "")),
                },
                "metrics": _selected_record_metrics(record) if isinstance(record, dict) else {},
                "files": files if isinstance(files, dict) else {},
            }
        )

    reviewed_count = sum(1 for row in review_rows if _review_has_content(row))
    included_count = len(evidence_candidates)
    score_averages = {field: _score_average(review_rows, field) for field in EVIDENCE_SCORE_FIELDS}
    ocr_rows = [
        row
        for row in review_rows
        if _review_number(row.get("ocr_structured_table_candidate_count"))
        or _review_number(row.get("ocr_structured_table_gate_pass_rate")) is not None
        or _review_number(row.get("ocr_table_cell_bbox_coverage_rate")) is not None
        or _review_text(row.get("ocr_structured_table_gate_issues"))
    ]
    ocr_formula_rows = [
        row
        for row in review_rows
        if _review_number(row.get("ocr_structured_formula_candidate_count"))
        or _review_number(row.get("ocr_structured_formula_gate_pass_rate")) is not None
        or _review_number(row.get("ocr_structured_formula_token_count")) is not None
        or _review_number(row.get("ocr_structured_formula_equation_label_count")) is not None
        or _review_text(row.get("ocr_structured_formula_gate_issues"))
    ]
    repair_publish_rows = [
        row
        for row in review_rows
        if _review_bool(row.get("repair_publish_confirmed")) is not None
        or _review_bool(row.get("repair_publish_published")) is not None
        or _review_number(row.get("repair_publish_open_issue_count")) is not None
        or _review_number(row.get("repair_publish_rate")) is not None
        or _review_text(row.get("repair_publish_status_counts"))
    ]
    repair_rollback_rows = [
        row
        for row in review_rows
        if _review_bool(row.get("repair_rollback_available")) is not None
        or _review_bool(row.get("repair_rollback_confirmed")) is not None
        or _review_bool(row.get("repair_rollback_applied")) is not None
        or _review_bool(row.get("repair_rollback_matches_original")) is not None
        or _review_number(row.get("repair_rollback_success_rate")) is not None
        or _review_text(row.get("repair_rollback_status_counts"))
    ]
    repair_formal_replace_rows = [
        row
        for row in review_rows
        if _review_bool(row.get("repair_formal_replace_available")) is not None
        or _review_bool(row.get("repair_formal_replace_confirmed")) is not None
        or _review_bool(row.get("repair_formal_replace_replaced")) is not None
        or _review_bool(row.get("repair_formal_replace_matches_published")) is not None
        or _review_number(row.get("repair_formal_replace_success_rate")) is not None
        or _review_text(row.get("repair_formal_replace_status_counts"))
    ]
    repair_formal_rollback_rows = [
        row
        for row in review_rows
        if _review_bool(row.get("repair_formal_rollback_available")) is not None
        or _review_bool(row.get("repair_formal_rollback_confirmed")) is not None
        or _review_bool(row.get("repair_formal_rollback_applied")) is not None
        or _review_bool(row.get("repair_formal_rollback_matches_backup")) is not None
        or _review_number(row.get("repair_formal_rollback_success_rate")) is not None
        or _review_text(row.get("repair_formal_rollback_status_counts"))
    ]
    repair_patch_review_rows = [
        row
        for row in review_rows
        if _review_number(row.get("repair_patch_review_count")) is not None
        or _review_number(row.get("repair_patch_review_required_count")) is not None
        or _review_number(row.get("repair_patch_review_blocking_count")) is not None
        or _review_number(row.get("repair_patch_review_required_rate")) is not None
        or _review_text(row.get("repair_patch_review_default_decision_counts"))
        or _review_text(row.get("repair_patch_review_risk_counts"))
    ]
    repair_effectiveness_rows = [
        row
        for row in review_rows
        if _review_number(row.get("repair_effectiveness_before_issue_count")) is not None
        or _review_number(row.get("repair_effectiveness_after_issue_count")) is not None
        or _review_number(row.get("repair_effectiveness_issue_delta")) is not None
        or _review_number(row.get("repair_effectiveness_issue_reduction_rate")) is not None
        or _review_text(row.get("repair_effectiveness_status_counts"))
    ]
    sample_filter = report.get("sample_filter", {}) if isinstance(report.get("sample_filter"), dict) else {}
    patent_batch_coverage = (
        sample_filter.get("selected_coverage")
        if isinstance(sample_filter.get("selected_coverage"), dict)
        else {}
    )

    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_summary_file": summary_file,
        "source_review_file": review_file,
        "batch_schema_version": report.get("schema_version"),
        "sample_count": report.get("sample_count", 0),
        "sample_filter": sample_filter,
        "patent_batch_coverage": patent_batch_coverage,
        "run_count": report.get("run_count", len(review_rows)),
        "succeeded_count": report.get("succeeded_count", 0),
        "failed_count": report.get("failed_count", len(run_failures)),
        "variant_count": len(report.get("aggregates", []) if isinstance(report.get("aggregates"), list) else []),
        "review_row_count": len(review_rows),
        "reviewed_count": reviewed_count,
        "included_count": included_count,
        "score_averages": score_averages,
        "variant_summary": _review_group_summary(review_rows, "variant"),
        "pdf_type_summary": _review_group_summary(review_rows, "pdf_type"),
        "ocr_structured_table_gate_summary": {
            "row_count": len(ocr_rows),
            "structured_table_candidate_count_total": _review_sum(
                review_rows,
                "ocr_structured_table_candidate_count",
            ),
            "structured_table_gate_review_count_total": _review_sum(
                review_rows,
                "ocr_structured_table_gate_review_count",
            ),
            "structured_table_promotion_count_total": _review_sum(
                review_rows,
                "ocr_structured_table_promotion_count",
            ),
            "gate_pass_rate": _score_average(ocr_rows, "ocr_structured_table_gate_pass_rate"),
            "structure_promotion_rate": _score_average(ocr_rows, "ocr_structured_table_promotion_rate"),
            "bbox_coverage_rate": _score_average(ocr_rows, "ocr_table_cell_bbox_coverage_rate"),
            "gate_issue_counts": _merge_counter_texts(review_rows, "ocr_structured_table_gate_issues"),
        },
        "ocr_structured_formula_gate_summary": {
            "row_count": len(ocr_formula_rows),
            "structured_formula_candidate_count_total": _review_sum(
                review_rows,
                "ocr_structured_formula_candidate_count",
            ),
            "structured_formula_gate_review_count_total": _review_sum(
                review_rows,
                "ocr_structured_formula_gate_review_count",
            ),
            "structured_formula_token_count_total": _review_sum(
                review_rows,
                "ocr_structured_formula_token_count",
            ),
            "structured_formula_equation_label_count_total": _review_sum(
                review_rows,
                "ocr_structured_formula_equation_label_count",
            ),
            "structured_formula_promotion_count_total": _review_sum(
                review_rows,
                "ocr_structured_formula_promotion_count",
            ),
            "gate_pass_rate": _score_average(ocr_formula_rows, "ocr_structured_formula_gate_pass_rate"),
            "structure_promotion_rate": _score_average(ocr_formula_rows, "ocr_structured_formula_promotion_rate"),
            "gate_issue_counts": _merge_counter_texts(review_rows, "ocr_structured_formula_gate_issues"),
        },
        "repair_publish_summary": {
            "row_count": len(repair_publish_rows),
            "confirmed_count_total": _review_bool_count(review_rows, "repair_publish_confirmed"),
            "published_count_total": _review_bool_count(review_rows, "repair_publish_published"),
            "open_issue_count_total": _review_sum(review_rows, "repair_publish_open_issue_count"),
            "publish_rate": _score_average(repair_publish_rows, "repair_publish_rate"),
            "status_counts": _merge_counter_texts(review_rows, "repair_publish_status_counts"),
        },
        "repair_rollback_summary": {
            "row_count": len(repair_rollback_rows),
            "available_count_total": _review_bool_count(review_rows, "repair_rollback_available"),
            "confirmed_count_total": _review_bool_count(review_rows, "repair_rollback_confirmed"),
            "applied_count_total": _review_bool_count(review_rows, "repair_rollback_applied"),
            "matches_original_count_total": _review_bool_count(review_rows, "repair_rollback_matches_original"),
            "success_rate": _score_average(repair_rollback_rows, "repair_rollback_success_rate"),
            "status_counts": _merge_counter_texts(review_rows, "repair_rollback_status_counts"),
        },
        "repair_formal_replace_summary": {
            "row_count": len(repair_formal_replace_rows),
            "available_count_total": _review_bool_count(review_rows, "repair_formal_replace_available"),
            "confirmed_count_total": _review_bool_count(review_rows, "repair_formal_replace_confirmed"),
            "replaced_count_total": _review_bool_count(review_rows, "repair_formal_replace_replaced"),
            "matches_published_count_total": _review_bool_count(
                review_rows,
                "repair_formal_replace_matches_published",
            ),
            "success_rate": _score_average(repair_formal_replace_rows, "repair_formal_replace_success_rate"),
            "status_counts": _merge_counter_texts(review_rows, "repair_formal_replace_status_counts"),
        },
        "repair_formal_rollback_summary": {
            "row_count": len(repair_formal_rollback_rows),
            "available_count_total": _review_bool_count(review_rows, "repair_formal_rollback_available"),
            "confirmed_count_total": _review_bool_count(review_rows, "repair_formal_rollback_confirmed"),
            "applied_count_total": _review_bool_count(review_rows, "repair_formal_rollback_applied"),
            "matches_backup_count_total": _review_bool_count(
                review_rows,
                "repair_formal_rollback_matches_backup",
            ),
            "success_rate": _score_average(repair_formal_rollback_rows, "repair_formal_rollback_success_rate"),
            "status_counts": _merge_counter_texts(review_rows, "repair_formal_rollback_status_counts"),
        },
        "repair_patch_review_summary": {
            "row_count": len(repair_patch_review_rows),
            "patch_count_total": _review_sum(review_rows, "repair_patch_review_count"),
            "safe_count_total": _review_sum(review_rows, "repair_patch_review_safe_count"),
            "required_count_total": _review_sum(review_rows, "repair_patch_review_required_count"),
            "blocking_count_total": _review_sum(review_rows, "repair_patch_review_blocking_count"),
            "safe_rate": _score_average(repair_patch_review_rows, "repair_patch_review_safe_rate"),
            "required_rate": _score_average(repair_patch_review_rows, "repair_patch_review_required_rate"),
            "default_decision_counts": _merge_counter_texts(
                review_rows,
                "repair_patch_review_default_decision_counts",
            ),
            "risk_counts": _merge_counter_texts(review_rows, "repair_patch_review_risk_counts"),
        },
        "repair_effectiveness_summary": {
            "row_count": len(repair_effectiveness_rows),
            "before_issue_count_total": _review_sum(review_rows, "repair_effectiveness_before_issue_count"),
            "after_issue_count_total": _review_sum(review_rows, "repair_effectiveness_after_issue_count"),
            "issue_delta_total": _review_sum(review_rows, "repair_effectiveness_issue_delta"),
            "resolved_issue_count_total": _review_sum(review_rows, "repair_effectiveness_resolved_issue_count"),
            "persisted_issue_count_total": _review_sum(review_rows, "repair_effectiveness_persisted_issue_count"),
            "new_issue_count_total": _review_sum(review_rows, "repair_effectiveness_new_issue_count"),
            "improved_chunk_count_total": _review_sum(review_rows, "repair_effectiveness_improved_chunk_count"),
            "regressed_chunk_count_total": _review_sum(review_rows, "repair_effectiveness_regressed_chunk_count"),
            "issue_reduction_rate": _score_average(
                repair_effectiveness_rows,
                "repair_effectiveness_issue_reduction_rate",
            ),
            "status_counts": _merge_counter_texts(review_rows, "repair_effectiveness_status_counts"),
        },
        "run_failures": run_failures,
        "evidence_candidates": evidence_candidates,
    }


def write_batch_experiment_evidence_markdown(evidence: dict[str, Any], path: Path) -> Path:
    lines = [
        "# 批量实验专利证据摘要",
        "",
        f"- 生成时间：{evidence.get('created_at', '')}",
        f"- 评分行数：{evidence.get('review_row_count', 0)}",
        f"- 已填写评分/结论：{evidence.get('reviewed_count', 0)}",
        f"- 纳入专利证据：{evidence.get('included_count', 0)}",
        f"- 来源摘要：{evidence.get('source_summary_file', '')}",
        f"- 来源评分表：{evidence.get('source_review_file', '')}",
        "",
        "## 评分均值",
        "",
        "| 字段 | 均值 | 样本数 |",
        "| --- | --- | --- |",
    ]
    for field, item in evidence.get("score_averages", {}).items():
        if not isinstance(item, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(field),
                    _format_number(item.get("average", 0)),
                    _format_number(item.get("count", 0)),
                ]
            )
            + " |"
        )

    _extend_sample_coverage_markdown(
        lines,
        evidence.get("patent_batch_coverage"),
        heading="Confirmed patent-batch sample coverage",
    )

    lines.extend(
        [
            "",
            "## 策略汇总",
            "",
            "| 策略 | 运行数 | 已评审 | 纳入证据 | 总分均值 | 表格可读性 | 结构连贯性 | OCR 表格通过率 | OCR 公式通过率 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in evidence.get("variant_summary", []):
        if not isinstance(item, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("variant", "")),
                    _format_number(item.get("run_count", 0)),
                    _format_number(item.get("reviewed_count", 0)),
                    _format_number(item.get("included_count", 0)),
                    _format_number((item.get("human_score") or {}).get("average", 0)),
                    _format_number((item.get("table_readability_score") or {}).get("average", 0)),
                    _format_number((item.get("structure_coherence_score") or {}).get("average", 0)),
                    _format_number((item.get("ocr_structured_table_gate_pass_rate") or {}).get("average", 0)),
                    _format_number((item.get("ocr_structured_formula_gate_pass_rate") or {}).get("average", 0)),
                ]
            )
            + " |"
        )

    ocr_summary = evidence.get("ocr_structured_table_gate_summary", {})
    formula_ocr_summary = evidence.get("ocr_structured_formula_gate_summary", {})
    repair_summary = evidence.get("repair_publish_summary", {})
    rollback_summary = evidence.get("repair_rollback_summary", {})
    formal_replace_summary = evidence.get("repair_formal_replace_summary", {})
    formal_rollback_summary = evidence.get("repair_formal_rollback_summary", {})
    patch_review_summary = evidence.get("repair_patch_review_summary", {})
    effectiveness_summary = evidence.get("repair_effectiveness_summary", {})
    lines.extend(
        [
            "",
            "## OCR 结构门禁",
            "",
            f"- 结构化表格候选总数：{ocr_summary.get('structured_table_candidate_count_total', 0)}",
            f"- 表格门禁复核总数：{ocr_summary.get('structured_table_gate_review_count_total', 0)}",
            f"- 表格门禁通过率均值：{_format_number((ocr_summary.get('gate_pass_rate') or {}).get('average', 0))}",
            f"- 表格 bbox 覆盖率均值：{_format_number((ocr_summary.get('bbox_coverage_rate') or {}).get('average', 0))}",
            f"- 表格门禁问题分布：{_format_counter(ocr_summary.get('gate_issue_counts'))}",
            f"- 结构化公式候选总数：{formula_ocr_summary.get('structured_formula_candidate_count_total', 0)}",
            f"- 公式门禁复核总数：{formula_ocr_summary.get('structured_formula_gate_review_count_total', 0)}",
            f"- 公式门禁通过率均值：{_format_number((formula_ocr_summary.get('gate_pass_rate') or {}).get('average', 0))}",
            f"- 公式 token 回流总数：{formula_ocr_summary.get('structured_formula_token_count_total', 0)}",
            f"- 公式编号回流总数：{formula_ocr_summary.get('structured_formula_equation_label_count_total', 0)}",
            f"- 公式门禁问题分布：{_format_counter(formula_ocr_summary.get('gate_issue_counts'))}",
            "",
            "## 局部修复发布审核",
            "",
            f"- 记录行数：{repair_summary.get('row_count', 0)}",
            f"- 已请求发布确认：{repair_summary.get('confirmed_count_total', 0)}",
            f"- 已生成发布稿：{repair_summary.get('published_count_total', 0)}",
            f"- 开放合并问题总数：{repair_summary.get('open_issue_count_total', 0)}",
            f"- 发布率均值：{_format_number((repair_summary.get('publish_rate') or {}).get('average', 0))}",
            f"- 发布状态分布：{_format_counter(repair_summary.get('status_counts'))}",
            f"- 可回滚记录数：{rollback_summary.get('available_count_total', 0)}",
            f"- 已执行回滚演练：{rollback_summary.get('applied_count_total', 0)}",
            f"- 回滚副本匹配原始译文：{rollback_summary.get('matches_original_count_total', 0)}",
            f"- 回滚演练成功率均值：{_format_number((rollback_summary.get('success_rate') or {}).get('average', 0))}",
            f"- 回滚状态分布：{_format_counter(rollback_summary.get('status_counts'))}",
            f"- 正式替换已执行：{formal_replace_summary.get('replaced_count_total', 0)}",
            f"- 正式替换匹配发布稿：{formal_replace_summary.get('matches_published_count_total', 0)}",
            f"- 正式替换成功率均值：{_format_number((formal_replace_summary.get('success_rate') or {}).get('average', 0))}",
            f"- 正式替换状态分布：{_format_counter(formal_replace_summary.get('status_counts'))}",
            f"- 正式回滚已执行：{formal_rollback_summary.get('applied_count_total', 0)}",
            f"- 正式回滚匹配备份：{formal_rollback_summary.get('matches_backup_count_total', 0)}",
            f"- 正式回滚成功率均值：{_format_number((formal_rollback_summary.get('success_rate') or {}).get('average', 0))}",
            f"- 正式回滚状态分布：{_format_counter(formal_rollback_summary.get('status_counts'))}",
            f"- 补丁审核记录行数：{patch_review_summary.get('row_count', 0)}",
            f"- 补丁总数：{patch_review_summary.get('patch_count_total', 0)}",
            f"- 需要人工复核补丁数：{patch_review_summary.get('required_count_total', 0)}",
            f"- 阻断发布补丁数：{patch_review_summary.get('blocking_count_total', 0)}",
            f"- 默认审核结论分布：{_format_counter(patch_review_summary.get('default_decision_counts'))}",
            f"- 风险分布：{_format_counter(patch_review_summary.get('risk_counts'))}",
            f"- 修复效果对比记录行数：{effectiveness_summary.get('row_count', 0)}",
            f"- 修复前问题总数：{effectiveness_summary.get('before_issue_count_total', 0)}",
            f"- 修复后问题总数：{effectiveness_summary.get('after_issue_count_total', 0)}",
            f"- 问题减少总数：{effectiveness_summary.get('issue_delta_total', 0)}",
            f"- 问题减少率均值：{_format_number((effectiveness_summary.get('issue_reduction_rate') or {}).get('average', 0))}",
            f"- 已解决问题总数：{effectiveness_summary.get('resolved_issue_count_total', 0)}",
            f"- 新增问题总数：{effectiveness_summary.get('new_issue_count_total', 0)}",
            f"- 回归 chunk 总数：{effectiveness_summary.get('regressed_chunk_count_total', 0)}",
            f"- 修复效果状态分布：{_format_counter(effectiveness_summary.get('status_counts'))}",
            "",
            "## 纳入证据候选",
            "",
            "| 样本 | 策略 | 类型 | 总分 | 修复发布 | 修复效果 | 证据说明 | 译文 PDF |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in evidence.get("evidence_candidates", []):
        if not isinstance(item, dict):
            continue
        files = item.get("files", {}) if isinstance(item.get("files"), dict) else {}
        scores = item.get("scores", {}) if isinstance(item.get("scores"), dict) else {}
        repair_publish = item.get("repair_publish", {}) if isinstance(item.get("repair_publish"), dict) else {}
        repair_effectiveness = (
            item.get("repair_effectiveness", {}) if isinstance(item.get("repair_effectiveness"), dict) else {}
        )
        repair_status = (
            "已发布"
            if repair_publish.get("published") is True
            else "已请求" if repair_publish.get("confirmed") is True else "待确认"
        )
        repair_effect = (
            f"减少 {_format_number(repair_effectiveness.get('issue_delta', ''))}"
            if repair_effectiveness.get("issue_delta") is not None
            else ""
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("sample_id", "")),
                    str(item.get("variant", "")),
                    str(item.get("pdf_type", "")),
                    _format_number(scores.get("human_score", "")),
                    repair_status,
                    repair_effect,
                    str(item.get("patent_evidence_notes", "")),
                    str(files.get("translated_pdf", "")),
                ]
            )
            + " |"
        )
    if not evidence.get("evidence_candidates"):
        lines.append("|  |  |  |  |  |  | 暂无显式纳入专利证据的评分行 |  |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_batch_experiment_evidence(
    summary_path: Path,
    review_csv_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    report = _read_json(summary_path)
    review_rows = read_batch_experiment_review_csv(review_csv_path)
    target_dir = (output_dir or summary_path.parent).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    evidence = build_batch_experiment_evidence(
        report,
        review_rows,
        summary_file=str(summary_path),
        review_file=str(review_csv_path),
    )
    (target_dir / "batch_experiment_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_batch_experiment_evidence_markdown(evidence, target_dir / "batch_experiment_evidence.md")
    return evidence


def run_batch_experiment(
    pdfs: list[Path],
    output_dir: Path,
    cfg: AppConfig,
    *,
    variants: list[ExperimentVariant] | None = None,
    backend: str | None = None,
    pages_per_chunk: int = 3,
    overlap_pages: int = 1,
    max_chunks: int | None = None,
    tail_fallback: bool = False,
    translate_mode: Literal["serial", "parallel"] = "serial",
    parallel_workers: int = 4,
    resume: bool = False,
    stop_on_error: bool = False,
    sample_metadata: dict[str, Any] | None = None,
    patent_batch_only: bool = False,
) -> dict[str, Any]:
    if not pdfs:
        raise ValueError("at least one PDF is required")
    variants = variants or parse_variant_specs("page,structure")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_samples = _build_samples(pdfs, sample_metadata)
    samples = [
        sample
        for sample in all_samples
        if not patent_batch_only or _review_truthy(sample.include_in_patent_batch)
    ]
    if not samples:
        raise ValueError(
            "no samples selected for batch experiment; fill include_in_patent_batch "
            "or disable patent_batch_only"
        )
    selected_coverage = _build_sample_coverage(
        [
            {
                "sample_id": sample.sample_id,
                "source_pdf": str(sample.source_pdf),
                "pdf_type": sample.pdf_type,
                "tags": list(sample.tags),
                "include_in_patent_batch": sample.include_in_patent_batch,
            }
            for sample in samples
        ]
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend or cfg.default_translator,
        "input_pdf_count": len(pdfs),
        "sample_count": len(samples),
        "sample_filter": {
            "patent_batch_only": patent_batch_only,
            "input_sample_count": len(all_samples),
            "selected_sample_count": len(samples),
            "skipped_sample_count": len(all_samples) - len(samples),
            "selected_coverage": selected_coverage,
        },
        "pages_per_chunk": pages_per_chunk,
        "overlap_pages": overlap_pages,
        "max_chunks": max_chunks,
        "tail_fallback": tail_fallback,
        "translate_mode": translate_mode,
        "parallel_workers": parallel_workers,
        "variants": [variant.__dict__ for variant in variants],
        "samples": [
            {
                "sample_id": sample.sample_id,
                "source_pdf": str(sample.source_pdf.resolve()),
                "pdf_type": sample.pdf_type,
                "tags": list(sample.tags),
                "notes": sample.notes,
                "include_in_patent_batch": sample.include_in_patent_batch,
                "reviewer": sample.reviewer,
                "review_notes": sample.review_notes,
            }
            for sample in samples
        ],
    }
    (output_dir / "batch_experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    records: list[dict[str, Any]] = []
    for sample in samples:
        pdf = sample.source_pdf
        sample_id = sample.sample_id
        for variant in variants:
            work_dir = output_dir / "runs" / sample_id / _safe_id(variant.name)
            record: dict[str, Any] = {
                "sample_id": sample_id,
                "source_pdf": str(pdf.resolve()),
                "pdf_type": sample.pdf_type,
                "tags": list(sample.tags),
                "notes": sample.notes,
                "include_in_patent_batch": sample.include_in_patent_batch,
                "reviewer": sample.reviewer,
                "review_notes": sample.review_notes,
                "variant": variant.name,
                "chunk_strategy": variant.chunk_strategy,
                "execute_ocr": variant.execute_ocr,
                "execute_repair_requests": variant.execute_repair_requests,
                "work_dir": str(work_dir.relative_to(output_dir)).replace("\\", "/"),
                "status": "pending",
            }
            try:
                pipeline.init_workdir(work_dir)
                pipeline.run_split(pdf, work_dir, use_tail_if_no_heading=tail_fallback)
                pipeline.run_translate(
                    work_dir,
                    cfg,
                    backend=backend,
                    pages_per_chunk=pages_per_chunk,
                    overlap_pages=overlap_pages,
                    resume=resume,
                    max_chunks=max_chunks,
                    translate_mode=translate_mode,
                    parallel_workers=parallel_workers,
                    chunk_strategy=variant.chunk_strategy,
                    execute_repair_requests=variant.execute_repair_requests,
                    execute_ocr=variant.execute_ocr,
                )
                metrics_path = work_dir / "output" / "experiment_metrics.json"
                metrics = _read_json(metrics_path)
                record["status"] = "succeeded"
                record["metrics"] = _summary_subset(metrics)
                record["files"] = _record_paths(work_dir, output_dir)
            except Exception as exc:  # pragma: no cover - exercised through integration failures
                record["status"] = "failed"
                record["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(limit=20),
                }
                records.append(record)
                if stop_on_error:
                    raise
                continue
            records.append(record)

    aggregates = _aggregate_records(records, variants)
    baseline_variant = variants[0].name
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend or cfg.default_translator,
        "input_pdf_count": len(pdfs),
        "sample_count": len(samples),
        "sample_filter": manifest["sample_filter"],
        "patent_batch_coverage": selected_coverage,
        "variant_count": len(variants),
        "run_count": len(records),
        "succeeded_count": sum(1 for record in records if record.get("status") == "succeeded"),
        "failed_count": sum(1 for record in records if record.get("status") == "failed"),
        "baseline_variant": baseline_variant,
        "samples": manifest["samples"],
        "records": records,
        "aggregates": aggregates,
        "comparisons": _compare_to_baseline(records, baseline_variant),
        "manifest_file": "batch_experiment_manifest.json",
        "review_file": "batch_experiment_review.csv",
    }
    (output_dir / "batch_experiment_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_batch_experiment_review_csv(report, output_dir / "batch_experiment_review.csv")
    write_batch_experiment_markdown(report, output_dir / "batch_experiment_summary.md")
    return report
