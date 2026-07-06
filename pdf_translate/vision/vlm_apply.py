from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from pdf_translate.extractors.document_ir import document_ir_from_json_dict
from pdf_translate.qa.ocr_candidates import write_ocr_candidate_qa
from pdf_translate.vision.ocr_promotion import write_ocr_candidate_promotion
from pdf_translate.vision.ocr_writeback import (
    OCR_RESULTS_SCHEMA_VERSION,
    build_ocr_results_payload,
    write_ocr_writeback,
)
from pdf_translate.vision.vlm_retranslation import write_vlm_retranslation_plan

SCHEMA_VERSION = "vlm-results-apply-v1"
MERGED_OCR_RESULTS_SOURCE = "ocr_results_with_vlm_fallback_review"


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _load_json_dict(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return raw


def _optional_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    return _load_json_dict(path)


def _results(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    raw = payload.get("results")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _task_id(result: dict[str, Any]) -> str:
    return str(result.get("task_id") or "").strip()


def build_vlm_merged_ocr_results(
    ocr_tasks: dict[str, Any] | None,
    base_ocr_results: dict[str, Any] | None,
    vlm_results: dict[str, Any],
) -> dict[str, Any]:
    """Merge accepted VLM review results into the normal OCR result payload.

    VLM results win per task_id so a manual/visual review can replace an earlier
    low-confidence, failed, or empty OCR result without losing other OCR results.
    """
    base_payload = build_ocr_results_payload(ocr_tasks, base_ocr_results)
    vlm_payload = build_ocr_results_payload(ocr_tasks, vlm_results)
    base_results = _results(base_payload)
    vlm_result_items = _results(vlm_payload)

    merged_results: list[dict[str, Any]] = [_json_copy(item) for item in base_results]
    index_by_task_id: dict[str, int] = {}
    for idx, result in enumerate(merged_results):
        task_id = _task_id(result)
        if task_id and task_id not in index_by_task_id:
            index_by_task_id[task_id] = idx

    replaced_task_ids: list[str] = []
    appended_task_ids: list[str] = []
    for result in vlm_result_items:
        item = _json_copy(result)
        task_id = _task_id(item)
        if task_id and task_id in index_by_task_id:
            merged_results[index_by_task_id[task_id]] = item
            replaced_task_ids.append(task_id)
        else:
            merged_results.append(item)
            if task_id:
                appended_task_ids.append(task_id)

    merged_payload = {
        "schema_version": OCR_RESULTS_SCHEMA_VERSION,
        "doc_id": str(vlm_payload.get("doc_id") or base_payload.get("doc_id") or (ocr_tasks or {}).get("doc_id") or ""),
        "source": MERGED_OCR_RESULTS_SOURCE,
        "source_components": {
            "base_source": str(base_payload.get("source") or "not_provided"),
            "vlm_source": str(vlm_payload.get("source") or ""),
            "vlm_source_schema_version": str(vlm_results.get("source_schema_version") or ""),
        },
        "results": merged_results,
    }
    normalized = build_ocr_results_payload(ocr_tasks, merged_payload)
    status_counts = Counter(str(item.get("status") or "succeeded") for item in vlm_result_items)
    engine_counts = Counter(str(item.get("engine") or "unknown") for item in vlm_result_items)
    normalized["summary"].update(
        {
            "base_result_count": len(base_results),
            "vlm_result_count": len(vlm_result_items),
            "vlm_status_counts": dict(status_counts),
            "vlm_engine_counts": dict(engine_counts),
            "merged_result_count": len(_results(normalized)),
            "replaced_result_count": len(replaced_task_ids),
            "appended_result_count": len(appended_task_ids),
            "replaced_task_ids": replaced_task_ids,
            "appended_task_ids": appended_task_ids,
        }
    )
    return normalized


def vlm_results_apply_to_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# VLM Results Apply",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Status | {summary.get('status', '')} |",
        f"| Base OCR results | {summary.get('base_result_count', 0)} |",
        f"| VLM results | {summary.get('vlm_result_count', 0)} |",
        f"| Replaced results | {summary.get('replaced_result_count', 0)} |",
        f"| Appended results | {summary.get('appended_result_count', 0)} |",
        f"| Merged OCR results | {summary.get('merged_result_count', 0)} |",
        f"| OCR writeback accepted | {summary.get('writeback_accepted_result_count', 0)} |",
        f"| QA candidates | {summary.get('qa_candidate_count', 0)} |",
        f"| Promotable candidates | {summary.get('qa_promotable_candidate_count', 0)} |",
        f"| Promoted candidates | {summary.get('promoted_candidate_count', 0)} |",
        f"| Canonical structure promotions | {summary.get('canonical_structure_promotion_count', 0)} |",
        f"| Retranslation plan status | {summary.get('retranslation_plan_status', '')} |",
        f"| Retranslation chunks | {summary.get('retranslation_plan_retranslate_chunk_count', 0)} |",
        f"| Unmapped VLM tasks | {summary.get('retranslation_plan_unmapped_task_count', 0)} |",
        "",
        "## Artifacts",
        "",
    ]
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    for key, value in sorted(artifacts.items()):
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines).rstrip() + "\n"


def write_vlm_results_apply(
    work_dir: Path,
    *,
    report_path: Path | None = None,
    markdown_path: Path | None = None,
) -> dict[str, Any]:
    out_dir = work_dir / "output"
    document_ir_path = out_dir / "document_ir.json"
    ocr_tasks_path = out_dir / "ocr_tasks.json"
    ocr_results_path = out_dir / "ocr_results.json"
    vlm_results_path = out_dir / "vlm_results.json"
    ocr_writeback_path = out_dir / "ocr_writeback.json"
    document_ir_ocr_path = out_dir / "document_ir_ocr.json"
    ocr_candidate_qa_path = out_dir / "ocr_candidate_qa.json"
    ocr_candidate_qa_md_path = out_dir / "ocr_candidate_qa.md"
    ocr_candidate_promotion_path = out_dir / "ocr_candidate_promotion.json"
    ocr_candidate_promotion_md_path = out_dir / "ocr_candidate_promotion.md"
    promoted_ir_path = out_dir / "document_ir_promoted.json"
    vlm_retranslation_plan_path = out_dir / "vlm_retranslation_plan.json"
    vlm_retranslation_plan_md_path = out_dir / "vlm_retranslation_plan.md"
    report_path = report_path or out_dir / "vlm_apply.json"
    markdown_path = markdown_path or out_dir / "vlm_apply.md"

    document_ir_raw = _load_json_dict(document_ir_path)
    doc_ir = document_ir_from_json_dict(document_ir_raw)
    ocr_tasks = _load_json_dict(ocr_tasks_path)
    vlm_results = _load_json_dict(vlm_results_path)
    base_ocr_results = _optional_json_dict(ocr_results_path)
    merged_ocr_results = build_vlm_merged_ocr_results(ocr_tasks, base_ocr_results, vlm_results)

    ocr_results_path.write_text(json.dumps(merged_ocr_results, ensure_ascii=False, indent=2), encoding="utf-8")
    ocr_writeback = write_ocr_writeback(
        doc_ir,
        ocr_tasks,
        ocr_writeback_path,
        document_ir_ocr_path,
        merged_ocr_results,
    )
    document_ir_ocr = _load_json_dict(document_ir_ocr_path)
    ocr_candidate_qa = write_ocr_candidate_qa(
        document_ir_ocr,
        ocr_writeback,
        ocr_candidate_qa_path,
        ocr_candidate_qa_md_path,
    )
    ocr_candidate_promotion = write_ocr_candidate_promotion(
        document_ir_ocr,
        ocr_candidate_qa,
        ocr_candidate_promotion_path,
        ocr_candidate_promotion_md_path,
        promoted_ir_path,
    )

    merge_summary = merged_ocr_results.get("summary") if isinstance(merged_ocr_results.get("summary"), dict) else {}
    writeback_summary = ocr_writeback.get("summary") if isinstance(ocr_writeback.get("summary"), dict) else {}
    qa_summary = ocr_candidate_qa.get("summary") if isinstance(ocr_candidate_qa.get("summary"), dict) else {}
    promotion_summary = (
        ocr_candidate_promotion.get("summary")
        if isinstance(ocr_candidate_promotion.get("summary"), dict)
        else {}
    )
    vlm_retranslation_plan = write_vlm_retranslation_plan(
        out_dir,
        json_path=vlm_retranslation_plan_path,
        markdown_path=vlm_retranslation_plan_md_path,
        vlm_apply_report={
            "summary": {
                "canonical_structure_promotion_count": int(
                    promotion_summary.get("canonical_structure_promotion_count") or 0
                ),
                "structured_table_promotion_count": int(
                    promotion_summary.get("structured_table_promotion_count") or 0
                ),
                "structured_formula_promotion_count": int(
                    promotion_summary.get("structured_formula_promotion_count") or 0
                ),
            }
        },
    )
    retranslation_summary = (
        vlm_retranslation_plan.get("summary")
        if isinstance(vlm_retranslation_plan.get("summary"), dict)
        else {}
    )
    vlm_result_count = int(merge_summary.get("vlm_result_count") or 0)
    summary = {
        "status": "applied" if vlm_result_count > 0 else "no_vlm_results",
        "base_result_count": int(merge_summary.get("base_result_count") or 0),
        "vlm_result_count": vlm_result_count,
        "merged_result_count": int(merge_summary.get("merged_result_count") or 0),
        "replaced_result_count": int(merge_summary.get("replaced_result_count") or 0),
        "appended_result_count": int(merge_summary.get("appended_result_count") or 0),
        "writeback_accepted_result_count": int(writeback_summary.get("accepted_result_count") or 0),
        "writeback_rejected_result_count": int(writeback_summary.get("rejected_result_count") or 0),
        "writeback_pending_task_count": int(writeback_summary.get("pending_task_count") or 0),
        "qa_candidate_count": int(qa_summary.get("candidate_count") or 0),
        "qa_promotable_candidate_count": int(qa_summary.get("promotable_candidate_count") or 0),
        "qa_needs_review_candidate_count": int(qa_summary.get("needs_review_candidate_count") or 0),
        "qa_blocked_candidate_count": int(qa_summary.get("blocked_candidate_count") or 0),
        "promoted_candidate_count": int(promotion_summary.get("promoted_candidate_count") or 0),
        "skipped_candidate_count": int(promotion_summary.get("skipped_candidate_count") or 0),
        "canonical_structure_promotion_count": int(
            promotion_summary.get("canonical_structure_promotion_count") or 0
        ),
        "structured_table_promotion_count": int(promotion_summary.get("structured_table_promotion_count") or 0),
        "structured_formula_promotion_count": int(
            promotion_summary.get("structured_formula_promotion_count") or 0
        ),
        "retranslation_plan_status": str(retranslation_summary.get("status") or ""),
        "retranslation_plan_affected_chunk_count": int(
            retranslation_summary.get("affected_chunk_count") or 0
        ),
        "retranslation_plan_retranslate_chunk_count": int(
            retranslation_summary.get("retranslate_chunk_count") or 0
        ),
        "retranslation_plan_unmapped_task_count": int(
            retranslation_summary.get("unmapped_task_count") or 0
        ),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "doc_id": str(merged_ocr_results.get("doc_id") or doc_ir.doc_id),
        "source": "vlm_fallback_results_apply",
        "summary": summary,
        "merge_summary": _json_copy(merge_summary),
        "artifacts": {
            "merged_ocr_results": "output/ocr_results.json",
            "ocr_writeback": "output/ocr_writeback.json",
            "document_ir_ocr": "output/document_ir_ocr.json",
            "ocr_candidate_qa": "output/ocr_candidate_qa.json",
            "ocr_candidate_qa_md": "output/ocr_candidate_qa.md",
            "ocr_candidate_promotion": "output/ocr_candidate_promotion.json",
            "ocr_candidate_promotion_md": "output/ocr_candidate_promotion.md",
            "document_ir_promoted": "output/document_ir_promoted.json",
            "vlm_retranslation_plan": "output/vlm_retranslation_plan.json",
            "vlm_retranslation_plan_md": "output/vlm_retranslation_plan.md",
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(vlm_results_apply_to_markdown(report), encoding="utf-8")
    return report
