from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import fitz
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response

from pdf_translate.chunking import TextChunk
from pdf_translate.error_codes import PdfTranslateError, error_info_from_exception, make_error_info
from pdf_translate.exporters.translated_pdf import write_translated_pdf
from pdf_translate.export_filename import suggest_md_download_name, suggest_zip_bundle_name
from pdf_translate.memory_store import MemoryStore
from pdf_translate.pipeline_cancel import cancel_flag_path
from pdf_translate.qa.glossary_retranslation import (
    execute_glossary_retranslation,
    write_glossary_retranslation_plan,
)
from pdf_translate.qa.repair import (
    write_repair_formal_replace,
    write_repair_formal_rollback,
    write_repair_patch_review,
    write_repair_patch_review_batch_decision,
    write_repair_patch_review_decision,
    write_repair_publish,
    write_repair_rollback,
)
from pdf_translate.qa.table_reconstruction import (
    load_preferred_table_reconstruction,
    table_structure_publish_to_markdown,
    write_table_merged_cell_review,
    write_table_merged_cell_review_batch_decision,
    write_table_merged_cell_review_decision,
    write_table_structure_publish,
)

from pdf_translate.server.auth_deps import Principal, bearer_principal, mint_token, require_admin
from pdf_translate.server import database
from pdf_translate.server.jobs import JOB_STATUS_SCHEMA_VERSION, JobRecord, JobRegistry, start_job_thread, zip_job_outputs
from pdf_translate.server.runtime_state import require_data_dir
from pdf_translate.server.security_preflight import build_security_preflight, max_upload_mb
from pdf_translate.server import settings_service
from pdf_translate.translators.factory import build_translator
from pdf_translate.translators.registry import (
    backend_catalog,
    backend_ids,
    backend_ui_labels,
    custom_api_backend_ids,
    get_backend_spec,
    normalize_backend_id,
)

PDF_UPLOAD_MAGIC = b"%PDF-"
UPLOAD_READ_CHUNK_BYTES = 1024 * 1024


def _read_json_artifact(path: Path, *, missing_message: str, invalid_message: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise HTTPException(404, missing_message)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(409, invalid_message) from exc
    if not isinstance(raw, dict):
        raise HTTPException(409, invalid_message)
    return raw


def _optional_json_dict_artifact(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _optional_json_list_artifact(path: Path) -> list[Any] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, list) else None


def _load_chunks_for_pdf_refresh(out_dir: Path) -> list[TextChunk]:
    manifest = _optional_json_list_artifact(out_dir / "chunks_manifest.json") or []
    chunk_dir = out_dir / "chunks"
    chunks: list[TextChunk] = []
    for entry in manifest:
        if not isinstance(entry, dict):
            continue
        chunk_id = str(entry.get("chunk_id") or "").strip()
        if not chunk_id or not (chunk_dir / f"{chunk_id}.md").is_file():
            continue
        pages = []
        for page in entry.get("pages_1based") or []:
            try:
                page_index = int(page) - 1
            except (TypeError, ValueError):
                continue
            if page_index >= 0:
                pages.append(page_index)
        chunk = TextChunk(
            chunk_id=chunk_id,
            pages_0based=pages,
            text=str(entry.get("text") or ""),
            link_count=int(entry.get("link_count") or 0),
            image_count=int(entry.get("image_count") or 0),
        )
        chunk.block_ids = list(entry.get("block_ids") or [])
        chunk.block_types = entry.get("block_types") if isinstance(entry.get("block_types"), dict) else {}
        chunk.warnings = list(entry.get("warnings") or [])
        chunk.boundary_fragment_ids = list(entry.get("boundary_fragment_ids") or [])
        chunk.structural_relation_ids = list(entry.get("structural_relation_ids") or [])
        chunk.approx_tokens = int(entry.get("approx_tokens") or 0)
        budget = entry.get("budget") if isinstance(entry.get("budget"), dict) else {}
        chunk.budget_target_chars = int(budget.get("target_chars") or 0)
        chunk.budget_max_chars = int(budget.get("max_chars") or 0)
        chunk.split_reason = str(budget.get("split_reason") or "")
        chunk.budget_overflow_chars = int(budget.get("overflow_chars") or 0)
        chunk.budget_pressure = str(budget.get("pressure") or "")
        chunks.append(chunk)
    return chunks


def _refresh_translated_pdf_with_confirmed_tables_for_record(rec: JobRecord) -> dict[str, Any] | None:
    out_dir = rec.work_dir / "output"
    table_reconstruction = load_preferred_table_reconstruction(out_dir)
    summary = table_reconstruction.get("summary") if isinstance(table_reconstruction.get("summary"), dict) else {}
    if summary.get("table_structure_source") != "confirmed":
        return None
    chunks = _load_chunks_for_pdf_refresh(out_dir)
    if not chunks:
        return None
    source_pdf = rec.work_dir / "input.pdf"
    return write_translated_pdf(
        chunks,
        out_dir / "chunks",
        out_dir / "translated_full.pdf",
        qa_report=_optional_json_dict_artifact(out_dir / "qa_report.json"),
        repair_plan=_optional_json_dict_artifact(out_dir / "repair_plan.json"),
        structure_qa=_optional_json_dict_artifact(out_dir / "structure_qa.json"),
        table_reconstruction=table_reconstruction,
        title=f"{Path(rec.original_filename or 'translated').stem} structured translation",
        source_pdf=source_pdf if source_pdf.is_file() else None,
        report_path=out_dir / "translated_pdf_report.json",
    )


def _read_or_create_repair_patch_review(out_dir: Path, repair_merge: dict[str, Any]) -> dict[str, Any]:
    json_path = out_dir / "repair_patch_review.json"
    md_path = out_dir / "repair_patch_review.md"
    if json_path.is_file() and json_path.stat().st_size > 0:
        return _read_json_artifact(
            json_path,
            missing_message="局部修复补丁审核报告尚未生成",
            invalid_message="局部修复补丁审核报告无法解析",
        )
    return write_repair_patch_review(repair_merge, json_path, md_path)


def _read_or_create_table_merged_cell_review(
    out_dir: Path,
    table_reconstruction: dict[str, Any],
) -> dict[str, Any]:
    json_path = out_dir / "table_merged_cell_review.json"
    md_path = out_dir / "table_merged_cell_review.md"
    if json_path.is_file() and json_path.stat().st_size > 0:
        return _read_json_artifact(
            json_path,
            missing_message="表格合并候选确认清单尚未生成",
            invalid_message="表格合并候选确认清单无法解析",
        )
    return write_table_merged_cell_review(table_reconstruction, json_path, md_path)


def _read_or_create_table_structure_publish(
    out_dir: Path,
    table_reconstruction: dict[str, Any],
    table_merged_cell_review: dict[str, Any],
) -> dict[str, Any]:
    json_path = out_dir / "table_structure_publish.json"
    md_path = out_dir / "table_structure_publish.md"
    if json_path.is_file() and json_path.stat().st_size > 0:
        report = _read_json_artifact(
            json_path,
            missing_message="表格结构确认发布报告尚未生成",
            invalid_message="表格结构确认发布报告无法解析",
        )
        if not md_path.is_file() or md_path.stat().st_size == 0:
            md_path.write_text(table_structure_publish_to_markdown(report), encoding="utf-8")
        return report
    return write_table_structure_publish(
        table_reconstruction,
        table_merged_cell_review,
        json_path,
        md_path,
        confirm=False,
        published_reconstruction_path=out_dir / "table_reconstruction_confirmed.json",
    )


def _glossary_memory_store_for_record(rec: JobRecord) -> MemoryStore:
    return MemoryStore(rec.work_dir / "memory")


def _glossary_review_report_for_record(rec: JobRecord) -> dict[str, Any]:
    try:
        return _glossary_memory_store_for_record(rec).build_glossary_review_report()
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(409, "术语审核文件无法解析") from exc


def _glossary_review_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary")
    return summary if isinstance(summary, dict) else {}


def _glossary_review_lookup(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for item in report.get("pending_reviews") or []:
        if not isinstance(item, dict):
            continue
        review_id = str(item.get("review_id") or item.get("dedupe_key") or "").strip()
        if review_id:
            lookup[review_id] = item
    return lookup


def _normalize_glossary_review_ids(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        raise HTTPException(400, "review_ids 必须是非空数组")
    review_ids: list[str] = []
    seen: set[str] = set()
    for item in raw:
        review_id = str(item or "").strip()
        if not review_id or review_id in seen:
            continue
        seen.add(review_id)
        review_ids.append(review_id)
    if not review_ids:
        raise HTTPException(400, "review_ids 必须是非空数组")
    return review_ids


def _validate_glossary_review_ids(report: dict[str, Any], review_ids: list[str]) -> None:
    lookup = _glossary_review_lookup(report)
    for review_id in review_ids:
        item = lookup.get(review_id)
        if item is None:
            raise HTTPException(404, f"术语审核项不存在：{review_id}")
        if item.get("type") != "glossary_conflict":
            raise HTTPException(400, f"暂不支持处理该术语审核类型：{item.get('type') or 'unknown'}")
        if item.get("status", "pending") != "pending":
            raise HTTPException(400, f"术语审核项已处理：{review_id}")


def _normalize_glossary_review_batch_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    def normalize_decision(value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in {"confirm", "confirmed", "confirm_candidate", "accept", "approve"}:
            return "confirm_candidate"
        if text in {"reject", "rejected", "reject_candidate", "deny"}:
            return "reject_candidate"
        raise HTTPException(400, "decision must be confirm_candidate or reject_candidate")

    raw_items = payload.get("items")
    if raw_items is None:
        review_ids = _normalize_glossary_review_ids(
            payload.get("review_ids") if "review_ids" in payload else payload.get("pending_keys")
        )
        if "candidate_zh" in payload or "confirmed_zh" in payload:
            raise HTTPException(400, "批量接口不支持顶层统一改写候选译名，请使用 items[].candidate_zh")
        normalize_decision(payload.get("decision"))
        return [
            {
                "review_id": review_id,
                "payload": payload,
            }
            for review_id in review_ids
        ]
    if "candidate_zh" in payload or "confirmed_zh" in payload:
        raise HTTPException(400, "批量接口不支持顶层统一改写候选译名，请使用 items[].candidate_zh")
    if not isinstance(raw_items, list) or not raw_items:
        raise HTTPException(400, "items 必须是非空数组")

    default_decision = str(payload.get("decision") or "").strip()
    default_comment = str(payload.get("comment") or "").strip()
    default_section_scope = str(payload.get("section_scope") or "").strip()
    default_confidence = payload.get("confidence")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise HTTPException(400, "items[] 必须是对象")
        review_id = str(
            raw.get("review_id") or raw.get("pending_key") or raw.get("dedupe_key") or ""
        ).strip()
        if not review_id:
            raise HTTPException(400, "items[].review_id 不能为空")
        if review_id in seen:
            raise HTTPException(400, f"重复的术语审核项：{review_id}")
        seen.add(review_id)
        item_decision = normalize_decision(raw.get("decision") or default_decision)

        item_payload: dict[str, Any] = {
            "decision": item_decision,
            "comment": str(raw.get("comment") if raw.get("comment") is not None else default_comment),
            "section_scope": str(
                raw.get("section_scope")
                if raw.get("section_scope") is not None
                else default_section_scope
            ),
            "confidence": raw.get("confidence")
            if raw.get("confidence") is not None
            else default_confidence,
        }
        if "candidate_zh" in raw:
            item_payload["candidate_zh"] = raw.get("candidate_zh")
        if "confirmed_zh" in raw:
            item_payload["confirmed_zh"] = raw.get("confirmed_zh")
        normalized.append({"review_id": review_id, "payload": item_payload})
    return normalized


def _validate_glossary_review_batch_payloads(
    report: dict[str, Any],
    batch_items: list[dict[str, Any]],
) -> None:
    lookup = _glossary_review_lookup(report)
    for batch_item in batch_items:
        review_id = str(batch_item.get("review_id") or "")
        payload = batch_item.get("payload") if isinstance(batch_item.get("payload"), dict) else {}
        decision = str(payload.get("decision") or "").strip().lower()
        if decision in {"confirm", "confirmed", "confirm_candidate", "accept", "approve"}:
            item = lookup.get(review_id) or {}
            candidate_value = (
                payload.get("candidate_zh")
                if "candidate_zh" in payload
                else payload.get("confirmed_zh")
                if "confirmed_zh" in payload
                else item.get("candidate_zh")
            )
            candidate_text = str(candidate_value or "").strip()
            if not candidate_text:
                raise HTTPException(400, f"candidate_zh must not be empty: {review_id}")
            if len(candidate_text) > 120:
                raise HTTPException(400, f"candidate_zh must be at most 120 characters: {review_id}")
        confidence = payload.get("confidence")
        if confidence is None or confidence == "":
            continue
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, f"confidence must be between 0 and 1: {review_id}") from exc
        if confidence_value < 0 or confidence_value > 1:
            raise HTTPException(400, f"confidence must be between 0 and 1: {review_id}")


def _apply_glossary_review_decision(
    rec: JobRecord,
    review_id: str,
    payload: dict[str, Any],
    reviewer: str,
) -> dict[str, Any]:
    candidate_zh = None
    if "candidate_zh" in payload:
        candidate_zh = payload.get("candidate_zh")
    elif "confirmed_zh" in payload:
        candidate_zh = payload.get("confirmed_zh")
    try:
        return _glossary_memory_store_for_record(rec).apply_glossary_review_decision(
            review_id,
            str(payload.get("decision") or ""),
            reviewer=reviewer,
            comment=str(payload.get("comment") or ""),
            confidence=payload.get("confidence"),
            section_scope=str(payload.get("section_scope") or ""),
            candidate_zh=candidate_zh,
        )
    except ValueError as exc:
        message = str(exc)
        if "not found" in message:
            raise HTTPException(404, message) from exc
        raise HTTPException(400, message) from exc
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(409, "术语审核文件无法写入") from exc


def _refresh_glossary_retranslation_plan_for_record(rec: JobRecord) -> dict[str, Any]:
    try:
        return write_glossary_retranslation_plan(rec.work_dir / "output", rec.work_dir / "memory")
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(409, "术语重译计划无法生成") from exc


def _execute_glossary_retranslation_for_record(
    rec: JobRecord,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if rec.status != "done":
        raise HTTPException(409, "任务尚未完成，不能执行术语重译")
    cfg = settings_service.effective_app_config()
    try:
        backend = settings_service.assert_backend_allowed(
            str(payload.get("backend") or rec.backend or ""),
            cfg.default_translator,
        )
        translator = build_translator(backend, cfg)
        mode = str(payload.get("mode") or "stale_only").strip() or "stale_only"
        raw_chunk_ids = payload.get("chunk_ids")
        chunk_ids = (
            [str(item).strip() for item in raw_chunk_ids if str(item or "").strip()]
            if isinstance(raw_chunk_ids, list)
            else None
        )
        max_chunks: int | None = None
        if payload.get("max_chunks") is not None:
            max_chunks = max(1, int(payload.get("max_chunks")))
        return execute_glossary_retranslation(
            rec.work_dir / "output",
            rec.work_dir / "memory",
            translator,
            backend=backend,
            mode=mode,
            chunk_ids=chunk_ids,
            max_chunks=max_chunks,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(409, "术语重译执行产物无法生成") from exc


def _confirm_repair_publish_for_record(rec: JobRecord) -> dict[str, Any]:
    if rec.status != "done":
        raise HTTPException(409, "任务尚未完成，不能确认发布修复稿")
    out_dir = rec.work_dir / "output"
    repair_merge = _read_json_artifact(
        out_dir / "repair_merge.json",
        missing_message="局部修复合并报告尚未生成",
        invalid_message="局部修复合并报告无法解析",
    )
    repair_patch_review = _read_or_create_repair_patch_review(out_dir, repair_merge)
    report = write_repair_publish(
        repair_merge,
        out_dir / "repair_publish.json",
        out_dir / "repair_publish.md",
        confirm=True,
        source_full_path=out_dir / "repaired_full.md",
        published_full_path=out_dir / "published_full.md",
        original_full_path=out_dir / "translated_full.md",
        repair_patch_review=repair_patch_review,
    )
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    if not summary.get("published"):
        raise HTTPException(409, str(summary.get("reason") or "修复发布稿未生成"))
    return report


def _confirm_repair_rollback_for_record(rec: JobRecord) -> dict[str, Any]:
    if rec.status != "done":
        raise HTTPException(409, "任务尚未完成，不能确认回滚演练")
    out_dir = rec.work_dir / "output"
    repair_publish = _read_json_artifact(
        out_dir / "repair_publish.json",
        missing_message="局部修复发布确认报告尚未生成",
        invalid_message="局部修复发布确认报告无法解析",
    )
    report = write_repair_rollback(
        repair_publish,
        out_dir / "repair_rollback.json",
        out_dir / "repair_rollback.md",
        confirm=True,
        original_full_path=out_dir / "translated_full.md",
        published_full_path=out_dir / "published_full.md",
        rollback_full_path=out_dir / "rollback_full.md",
    )
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    if not summary.get("rollback_applied"):
        raise HTTPException(409, str(summary.get("reason") or "回滚演练副本未生成"))
    return report


def _confirm_repair_formal_replace_for_record(rec: JobRecord) -> dict[str, Any]:
    if rec.status != "done":
        raise HTTPException(409, "任务尚未完成，不能确认正式替换")
    out_dir = rec.work_dir / "output"
    repair_publish = _read_json_artifact(
        out_dir / "repair_publish.json",
        missing_message="局部修复发布确认报告尚未生成",
        invalid_message="局部修复发布确认报告无法解析",
    )
    report = write_repair_formal_replace(
        repair_publish,
        out_dir / "repair_formal_replace.json",
        out_dir / "repair_formal_replace.md",
        confirm=True,
        original_full_path=out_dir / "translated_full.md",
        published_full_path=out_dir / "published_full.md",
        formal_full_path=out_dir / "formal_full.md",
        backup_full_path=out_dir / "formal_full.before_repair.md",
    )
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    if not summary.get("replaced"):
        raise HTTPException(409, str(summary.get("reason") or "正式替换未执行"))
    return report


def _confirm_repair_formal_rollback_for_record(rec: JobRecord) -> dict[str, Any]:
    if rec.status != "done":
        raise HTTPException(409, "任务尚未完成，不能确认正式回滚")
    out_dir = rec.work_dir / "output"
    repair_formal_replace = _read_json_artifact(
        out_dir / "repair_formal_replace.json",
        missing_message="局部修复正式替换报告尚未生成",
        invalid_message="局部修复正式替换报告无法解析",
    )
    report = write_repair_formal_rollback(
        repair_formal_replace,
        out_dir / "repair_formal_rollback.json",
        out_dir / "repair_formal_rollback.md",
        confirm=True,
        formal_full_path=out_dir / "formal_full.md",
        backup_full_path=out_dir / "formal_full.before_repair.md",
        active_before_rollback_path=out_dir / "formal_full.repair_applied.md",
    )
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    if not summary.get("rollback_applied"):
        raise HTTPException(409, str(summary.get("reason") or "正式回滚未执行"))
    return report


def _confirm_table_structure_publish_for_record(rec: JobRecord) -> dict[str, Any]:
    if rec.status != "done":
        raise HTTPException(409, "任务尚未完成，不能确认发布表格结构副本")
    out_dir = rec.work_dir / "output"
    table_reconstruction = _read_json_artifact(
        out_dir / "table_reconstruction.json",
        missing_message="表格重建报告尚未生成",
        invalid_message="表格重建报告无法解析",
    )
    table_merged_cell_review = _read_or_create_table_merged_cell_review(out_dir, table_reconstruction)
    report = write_table_structure_publish(
        table_reconstruction,
        table_merged_cell_review,
        out_dir / "table_structure_publish.json",
        out_dir / "table_structure_publish.md",
        confirm=True,
        published_reconstruction_path=out_dir / "table_reconstruction_confirmed.json",
    )
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    if not summary.get("published"):
        raise HTTPException(409, str(summary.get("reason") or "表格结构副本未生成"))
    try:
        pdf_report = _refresh_translated_pdf_with_confirmed_tables_for_record(rec)
    except Exception as exc:  # noqa: BLE001 - optional PDF refresh must not block confirmation.
        pdf_report = None
        summary["translated_pdf_refresh_status"] = f"failed:{type(exc).__name__}"
    else:
        summary["translated_pdf_refresh_status"] = "refreshed" if pdf_report else "skipped_missing_translation_chunks"
    if isinstance(pdf_report, dict):
        pdf_summary = pdf_report.get("summary") if isinstance(pdf_report.get("summary"), dict) else {}
        summary["translated_pdf_table_reconstruction_source"] = pdf_report.get("table_reconstruction_source") or ""
        summary["translated_pdf_confirmed_candidate_reference_count"] = int(
            pdf_summary.get("confirmed_merged_cell_candidate_reference_count") or 0
        )
    (out_dir / "table_structure_publish.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "table_structure_publish.md").write_text(
        table_structure_publish_to_markdown(report),
        encoding="utf-8",
    )
    return report


def _normalise_preview_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _table_merged_cell_review_preview_bbox(review: dict[str, Any]) -> tuple[float, float, float, float] | None:
    evidence = review.get("bbox_evidence") if isinstance(review.get("bbox_evidence"), dict) else {}
    for key in ("span_bbox", "evidence_bbox", "candidate_bbox"):
        bbox = _normalise_preview_bbox(evidence.get(key))
        if bbox is not None:
            return bbox
    return None


def _clip_preview_bbox(
    bbox: tuple[float, float, float, float],
    page_rect: fitz.Rect,
) -> fitz.Rect | None:
    x0 = min(max(bbox[0], page_rect.x0), page_rect.x1)
    y0 = min(max(bbox[1], page_rect.y0), page_rect.y1)
    x1 = min(max(bbox[2], page_rect.x0), page_rect.x1)
    y1 = min(max(bbox[3], page_rect.y0), page_rect.y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return fitz.Rect(x0, y0, x1, y1)


def _render_table_merged_cell_review_preview_for_record(
    rec: JobRecord,
    review: dict[str, Any],
    *,
    scale: float = 1.6,
) -> bytes:
    input_pdf = rec.work_dir / "input.pdf"
    if not input_pdf.is_file() or input_pdf.stat().st_size == 0:
        raise HTTPException(404, "input.pdf is not available for preview")
    try:
        page_no = int(review.get("page_no") or 0)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, "review page_no is invalid") from exc
    if page_no <= 0:
        raise HTTPException(400, "review page_no is invalid")
    try:
        doc = fitz.open(input_pdf)
    except Exception as exc:  # pragma: no cover - pymupdf raises several concrete types
        raise HTTPException(409, "input.pdf cannot be rendered") from exc
    try:
        if page_no > doc.page_count:
            raise HTTPException(404, "review page is outside input.pdf")
        page = doc.load_page(page_no - 1)
        bbox = _table_merged_cell_review_preview_bbox(review)
        if bbox is not None:
            rect = _clip_preview_bbox(bbox, page.rect)
            if rect is not None:
                page.draw_rect(rect, color=(1, 0.2, 0.05), width=2.0, overlay=True)
        matrix = fitz.Matrix(max(0.5, min(float(scale), 3.0)), max(0.5, min(float(scale), 3.0)))
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


def _find_table_merged_cell_review_item(report: dict[str, Any], review_id: str) -> dict[str, Any]:
    for item in report.get("candidate_reviews") or []:
        if isinstance(item, dict) and str(item.get("review_id") or "") == review_id:
            return item
    raise KeyError(review_id)


def _is_deepseek_model_name(model_name: str | None) -> bool:
    return "deepseek" in str(model_name or "").strip().lower()


def client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for") or request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    if request.client:
        return request.client.host or ""
    return ""


async def _save_pdf_upload_streaming(
    file: UploadFile,
    dest: Path,
    *,
    max_bytes: int,
    max_mb: int,
) -> int:
    header = await file.read(len(PDF_UPLOAD_MAGIC))
    if header != PDF_UPLOAD_MAGIC:
        raise HTTPException(400, "上传文件不是有效 PDF")
    total = len(header)
    if total > max_bytes:
        raise HTTPException(400, f"文件超过 {max_mb}MB 上限")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dest.open("wb") as out:
            out.write(header)
            while True:
                chunk = await file.read(UPLOAD_READ_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(400, f"文件超过 {max_mb}MB 上限")
                out.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    return total


def _build_runtime_cfg_for_custom_api(
    *,
    cfg,
    backend: str,
    api_key: str | None,
    api_base_url: str | None,
    api_model: str | None,
):
    out = replace(cfg)
    b = normalize_backend_id(backend)
    key = (api_key or "").strip()
    base = (api_base_url or "").strip()
    model = (api_model or "").strip()
    spec = get_backend_spec(b)

    if not spec.supports_custom_api:
        raise PdfTranslateError(
            make_error_info(
                "CONFIG_INVALID_BACKEND",
                detail=f"{b} does not support per-job custom API settings.",
                source="api:create_job",
            )
        )

    if b in ("deepseek", "openai"):
        if not key:
            raise PdfTranslateError(
                make_error_info(
                    "CONFIG_MISSING_API_KEY",
                    detail=f"{b} custom API requires API Key.",
                    source="api:create_job",
                )
            )
        setattr(out, spec.api_key_attr or "", key)
        if base:
            setattr(out, spec.base_url_attr or "", base.rstrip("/"))
        if model:
            setattr(out, spec.model_attr or "", model)
        return out
    if b == "deepl":
        if not key:
            raise PdfTranslateError(
                make_error_info(
                    "CONFIG_MISSING_API_KEY",
                    detail="deepl custom API requires API Key.",
                    source="api:create_job",
                )
            )
        out.deepl_api_key = key
        if base:
            out.deepl_api_url = base
        return out
    if b == "ollama":
        if base:
            out.ollama_base_url = base.rstrip("/")
        if model:
            out.ollama_model = model
        return out
    raise PdfTranslateError(
        make_error_info(
            "CONFIG_INVALID_BACKEND",
            detail=f"Custom API translation does not support {backend}.",
            source="api:create_job",
        )
    )


def register_web_routes(app_registry: JobRegistry) -> APIRouter:
    api = APIRouter(prefix="/api")

    @api.get("/health")
    def health() -> dict:
        return {"ok": True}

    auth = APIRouter(prefix="/auth", tags=["auth"])

    @auth.post("/login")
    def login(request: Request, username: str = Form(...), password: str = Form(...)) -> dict:
        ip = client_ip(request)
        u = database.verify_user(username, password)
        if not u:
            database.log_audit(
                action="login_fail",
                ip=ip,
                user_id=None,
                username=username.strip(),
                detail={"reason": "bad_credentials"},
            )
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        database.log_audit(
            action="login_success",
            ip=ip,
            user_id=u["id"],
            username=u["username"],
            detail={"role": u["role"]},
        )
        token = mint_token(user_id=u["id"], username=u["username"], role=u["role"])
        return {"access_token": token, "token_type": "bearer", "role": u["role"], "username": u["username"]}

    @auth.post("/register")
    def register(request: Request, username: str = Form(...), password: str = Form(...)) -> dict:
        if not database.registration_open():
            raise HTTPException(400, "管理员已关闭自助注册")
        reserved = os.getenv("PDF_TRANSLATE_ADMIN_USERNAME", "falling-feather").strip()
        if username.strip() == reserved:
            raise HTTPException(400, "该用户名为系统保留")
        if len(password) < 6:
            raise HTTPException(400, "密码至少 6 位")
        ip = client_ip(request)
        try:
            uid = database.create_user(username=username, password=password, role="user")
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        database.log_audit(
            action="register",
            ip=ip,
            user_id=uid,
            username=username.strip(),
            detail={},
        )
        u = database.get_user_by_id(uid)
        assert u
        token = mint_token(user_id=u["id"], username=u["username"], role=u["role"])
        return {"access_token": token, "token_type": "bearer", "role": u["role"], "username": u["username"]}

    @auth.get("/me")
    def me(p: Principal = Depends(bearer_principal)) -> dict:
        return {"user_id": p.user_id, "username": p.username, "role": p.role}

    api.include_router(auth)

    def _can_access_job(p: Principal, rec: JobRecord | None) -> bool:
        if not rec:
            return False
        if p.role == "admin":
            return True
        return rec.owner_user_id is not None and rec.owner_user_id == p.user_id

    @api.get("/user/backends")
    def user_backends(p: Principal = Depends(bearer_principal)) -> dict:
        _ = p
        eb = settings_service.enabled_backends()
        cfg = settings_service.effective_app_config()
        enabled_ordered = [b for b in backend_ids() if b in eb]
        labels = backend_ui_labels()
        return {
            "enabled": enabled_ordered,
            "default_backend": cfg.default_translator,
            "labels": {k: labels.get(k, k) for k in enabled_ordered},
            "catalog": backend_catalog(),
            "custom_api_backends": custom_api_backend_ids(),
        }

    @api.get("/user/jobs")
    def my_jobs(p: Principal = Depends(bearer_principal), limit: int = 100) -> dict:
        rows = app_registry.merge_status_into_rows(database.list_jobs_for_user(p.user_id, limit=limit))
        favorites = app_registry.merge_status_into_rows(database.list_favorite_jobs_for_user(p.user_id, limit=limit))
        return {
            "jobs": rows,
            "favorites": favorites,
            "favorite_max": database.MAX_JOB_FAVORITES_PER_USER,
        }

    @api.post("/user/jobs/cleanup-stale")
    def cleanup_stale_jobs(
        p: Principal = Depends(bearer_principal),
        hours: int = 24,
    ) -> dict:
        if hours < 1 or hours > 168:
            raise HTTPException(400, "hours 应在 1–168 之间")
        stale = database.list_stale_job_ids_for_user(p.user_id, hours=hours)
        deleted: list[str] = []
        for jid in stale:
            rec = app_registry.get(jid)
            if rec and rec.status in ("queued", "running"):
                continue
            database.delete_job_meta_row(jid)
            app_registry.remove_job(jid)
            deleted.append(jid)
        return {"deleted": deleted, "hours": hours}

    # 收藏接口单独路径，避免与 GET /user/jobs/favorites 等产生「POST 命中仅 GET 路由 → 405」的歧义
    @api.post("/user/favorites/{job_id}")
    def favorite_job(job_id: str, p: Principal = Depends(bearer_principal)) -> dict:
        try:
            database.add_job_favorite(p.user_id, job_id)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return {"ok": True}

    @api.delete("/user/favorites/{job_id}")
    def unfavorite_job(job_id: str, p: Principal = Depends(bearer_principal)) -> dict:
        try:
            database.remove_job_favorite(p.user_id, job_id)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        ts = datetime.now(timezone.utc).isoformat()
        rec = app_registry.get(job_id)
        if rec:
            app_registry.update(job_id, created_at=ts)
        return {"ok": True}

    @api.post("/jobs")
    async def create_job(
        request: Request,
        p: Principal = Depends(bearer_principal),
        file: UploadFile = File(...),
        tail_fallback: bool = Form(False),
        pages_per_chunk: int = Form(3),
        overlap_pages: int = Form(1),
        backend: str | None = Form(None),
        max_chunks: str | None = Form(None),
        translate_mode: str = Form("serial"),
        parallel_max_workers: str | None = Form(None),
        use_custom_api: bool = Form(False),
        custom_backend: str | None = Form(None),
        custom_api_key: str | None = Form(None),
        custom_api_base_url: str | None = Form(None),
        custom_api_model: str | None = Form(None),
    ) -> dict:
        if p.role not in ("user", "admin"):
            raise HTTPException(403, "无权提交翻译任务")
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(400, "请上传 .pdf 文件")
        if pages_per_chunk < 1 or pages_per_chunk > 3:
            raise HTTPException(400, "pages_per_chunk 必须在 1–3 之间")
        if overlap_pages < 0 or overlap_pages >= pages_per_chunk:
            raise HTTPException(400, "overlap_pages 无效")

        cfg = settings_service.effective_app_config()
        runtime_cfg = cfg
        if use_custom_api:
            be = (custom_backend or "").strip().lower()
            if not be:
                raise HTTPException(400, "已启用 API翻译，请选择 API 后端")
            try:
                be = normalize_backend_id(be)
            except ValueError as e:
                raise HTTPException(400, f"不支持的 API 后端：{be}") from e
            if be not in custom_api_backend_ids():
                raise HTTPException(400, f"不支持的 API 后端：{be}")
            try:
                runtime_cfg = _build_runtime_cfg_for_custom_api(
                    cfg=cfg,
                    backend=be,
                    api_key=custom_api_key,
                    api_base_url=custom_api_base_url,
                    api_model=custom_api_model,
                )
            except PdfTranslateError as e:
                raise HTTPException(400, detail=e.error_info.to_dict()) from e
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
        else:
            try:
                be = settings_service.assert_backend_allowed(backend, cfg.default_translator)
            except ValueError as e:
                info = error_info_from_exception(e, source="api:create_job")
                raise HTTPException(400, detail=info.to_dict()) from e

        try:
            build_translator(be, runtime_cfg)
        except PdfTranslateError as e:
            raise HTTPException(400, detail=e.error_info.to_dict()) from e
        except ValueError as e:
            info = error_info_from_exception(e, source="api:create_job")
            raise HTTPException(400, detail=info.to_dict()) from e

        max_n: int | None = None
        if max_chunks not in (None, ""):
            try:
                max_n = int(max_chunks)
            except ValueError:
                raise HTTPException(400, "max_chunks 必须为整数") from None

        tm = (translate_mode or "serial").strip().lower()
        if tm not in ("serial", "parallel", "premium"):
            raise HTTPException(400, "translate_mode 须为 serial、parallel 或 premium（精品翻译）")
        if tm == "premium":
            cfg_sf = settings_service.effective_app_config()
            if not (cfg_sf.siliconflow_api_key or "").strip():
                raise HTTPException(400, "精品翻译需要管理员在后台配置硅基流动 API Key")
            if not (cfg_sf.siliconflow_survey_model or "").strip():
                raise HTTPException(400, "精品翻译需要管理员在后台配置硅基流动「巡视模型」")
            if _is_deepseek_model_name(cfg_sf.siliconflow_survey_model):
                raise HTTPException(400, "精品翻译的硅基巡视模型不能填写 DeepSeek，请改为 Qwen/Kimi 等模型；DeepSeek 请走 DeepSeek API。")
        pwm: int | None = None
        if parallel_max_workers not in (None, ""):
            try:
                pwm = int(parallel_max_workers)
            except ValueError:
                raise HTTPException(400, "parallel_max_workers 须为整数") from None
            if pwm < 1 or pwm > 32:
                raise HTTPException(400, "parallel_max_workers 应在 1–32 之间")

        rec = app_registry.create_job(
            owner_user_id=p.user_id,
            owner_username=p.username,
            original_filename=file.filename or "upload.pdf",
            translate_mode=tm,
            parallel_max_workers=pwm,
            backend=be,
            tail_fallback=tail_fallback,
            pages_per_chunk=pages_per_chunk,
            overlap_pages=overlap_pages,
            max_chunks=max_n,
            use_custom_api=use_custom_api,
        )
        dest = rec.work_dir / "input.pdf"
        max_mb = max_upload_mb()
        try:
            uploaded_bytes = await _save_pdf_upload_streaming(
                file,
                dest,
                max_bytes=max_mb * 1024 * 1024,
                max_mb=max_mb,
            )
        except HTTPException:
            app_registry.remove_job(rec.job_id)
            raise

        database.insert_job_meta(rec.job_id, p.user_id, p.username, file.filename or "upload.pdf")
        database.log_audit(
            action="job_submit",
            ip=client_ip(request),
            user_id=p.user_id,
            username=p.username,
            job_id=rec.job_id,
            detail={
                "status_schema_version": JOB_STATUS_SCHEMA_VERSION,
                "status": rec.status,
                "phase": rec.phase,
                "filename": file.filename,
                "backend": be,
                "use_custom_api": bool(use_custom_api),
                "translate_mode": tm,
                "parallel_max_workers": pwm,
                "pages_per_chunk": pages_per_chunk,
                "overlap_pages": overlap_pages,
                "max_chunks": max_n,
                "upload_bytes": uploaded_bytes,
                "upload_limit_mb": max_mb,
                "runtime_created_at": rec.created_at,
                "work_dir": str(rec.work_dir.resolve()),
            },
        )

        start_job_thread(
            app_registry,
            rec.job_id,
            tail_fallback=tail_fallback,
            pages_per_chunk=pages_per_chunk,
            overlap_pages=overlap_pages,
            backend=be,
            max_chunks=max_n,
            cfg=runtime_cfg,
        )
        return {"job_id": rec.job_id}

    def _job_dict(rec: JobRecord) -> dict:
        complete = rec.status == "done"
        suggested_name = suggest_md_download_name(
            original_filename=rec.original_filename,
            work_dir=rec.work_dir,
            complete=complete,
        )
        suggested_zip = suggest_zip_bundle_name(
            original_filename=rec.original_filename,
            work_dir=rec.work_dir,
            complete=complete,
        )
        d = app_registry.diagnostic_summary_for_record(rec)
        d["suggested_download_filename"] = suggested_name
        d["suggested_zip_filename"] = suggested_zip
        return d

    @api.get("/jobs/{job_id}")
    def get_job(job_id: str, p: Principal = Depends(bearer_principal)) -> dict:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        return _job_dict(rec)

    @api.post("/jobs/{job_id}/cancel")
    def cancel_job(request: Request, job_id: str, p: Principal = Depends(bearer_principal)) -> dict:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        if rec.status not in ("queued", "running"):
            raise HTTPException(400, "当前状态不可终止")
        previous_status = rec.status
        previous_phase = rec.phase
        previous_message = rec.message
        flag_path = cancel_flag_path(rec.work_dir)
        flag_path.write_text("1", encoding="utf-8")
        app_registry.update(job_id, message="已收到终止请求，将在当前块结束后停止…")
        database.log_audit(
            action="job_cancel_requested",
            ip=client_ip(request),
            user_id=p.user_id,
            username=p.username,
            job_id=job_id,
            detail={
                "previous_status": previous_status,
                "previous_phase": previous_phase,
                "previous_message": previous_message,
                "chunk_index": rec.chunk_index,
                "chunk_total": rec.chunk_total,
                "chunk_id": rec.chunk_id,
                "cancel_flag_path": str(flag_path.resolve()),
                "requested_by_user_id": p.user_id,
                "requested_by_username": p.username,
            },
        )
        return {"ok": True}

    @api.get("/jobs/{job_id}/download/full.md")
    def download_full(job_id: str, p: Principal = Depends(bearer_principal)) -> FileResponse:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        path = rec.work_dir / "output" / "translated_full.md"
        if not path.is_file() or path.stat().st_size == 0:
            raise HTTPException(404, "译文尚未生成或仍为空")
        disp_name = suggest_md_download_name(
            original_filename=rec.original_filename,
            work_dir=rec.work_dir,
            complete=rec.status == "done",
        )
        ascii_fallback = "translated.md"
        cd = (
            f'attachment; filename="{ascii_fallback}"; '
            f"filename*=UTF-8''{quote(disp_name)}"
        )
        return FileResponse(
            path,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": cd},
        )

    @api.get("/jobs/{job_id}/download/input.pdf")
    def download_input_pdf(job_id: str, p: Principal = Depends(bearer_principal)) -> FileResponse:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        path = rec.work_dir / "input.pdf"
        if not path.is_file() or path.stat().st_size == 0:
            raise HTTPException(404, "未找到上传原文件")
        disp_name = rec.original_filename or "input.pdf"
        ascii_fallback = "input.pdf"
        cd = (
            f'attachment; filename="{ascii_fallback}"; '
            f"filename*=UTF-8''{quote(disp_name)}"
        )
        return FileResponse(
            path,
            media_type="application/pdf",
            headers={"Content-Disposition": cd},
        )

    @api.get("/jobs/{job_id}/download/translated.pdf")
    def download_translated_pdf(job_id: str, p: Principal = Depends(bearer_principal)) -> FileResponse:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        path = rec.work_dir / "output" / "translated_full.pdf"
        if not path.is_file() or path.stat().st_size == 0:
            raise HTTPException(404, "PDF 译文尚未生成")
        ascii_fallback = "translated.pdf"
        disp_name = f"{Path(rec.original_filename or 'translated').stem}_translated.pdf"
        cd = f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{quote(disp_name)}'
        return FileResponse(
            path,
            media_type="application/pdf",
            headers={"Content-Disposition": cd},
        )

    @api.get("/jobs/{job_id}/download/repair-publish.md")
    def download_repair_publish(job_id: str, p: Principal = Depends(bearer_principal)) -> FileResponse:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        path = rec.work_dir / "output" / "repair_publish.md"
        if not path.is_file() or path.stat().st_size == 0:
            raise HTTPException(404, "局部修复发布确认报告尚未生成")
        ascii_fallback = "repair_publish.md"
        disp_name = f"{Path(rec.original_filename or 'translated').stem}_repair_publish.md"
        cd = f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{quote(disp_name)}'
        return FileResponse(
            path,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": cd},
        )

    @api.get("/jobs/{job_id}/download/repair-effectiveness.md")
    def download_repair_effectiveness(job_id: str, p: Principal = Depends(bearer_principal)) -> FileResponse:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        path = rec.work_dir / "output" / "repair_effectiveness.md"
        if not path.is_file() or path.stat().st_size == 0:
            raise HTTPException(404, "局部修复效果对比报告尚未生成")
        ascii_fallback = "repair_effectiveness.md"
        disp_name = f"{Path(rec.original_filename or 'translated').stem}_repair_effectiveness.md"
        cd = f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{quote(disp_name)}'
        return FileResponse(
            path,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": cd},
        )

    @api.get("/jobs/{job_id}/download/repair-rollback.md")
    def download_repair_rollback(job_id: str, p: Principal = Depends(bearer_principal)) -> FileResponse:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        path = rec.work_dir / "output" / "repair_rollback.md"
        if not path.is_file() or path.stat().st_size == 0:
            raise HTTPException(404, "局部修复回滚演练报告尚未生成")
        ascii_fallback = "repair_rollback.md"
        disp_name = f"{Path(rec.original_filename or 'translated').stem}_repair_rollback.md"
        cd = f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{quote(disp_name)}'
        return FileResponse(
            path,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": cd},
        )

    @api.get("/jobs/{job_id}/download/repair-formal-replace.md")
    def download_repair_formal_replace(job_id: str, p: Principal = Depends(bearer_principal)) -> FileResponse:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        path = rec.work_dir / "output" / "repair_formal_replace.md"
        if not path.is_file() or path.stat().st_size == 0:
            raise HTTPException(404, "局部修复正式替换报告尚未生成")
        ascii_fallback = "repair_formal_replace.md"
        disp_name = f"{Path(rec.original_filename or 'translated').stem}_repair_formal_replace.md"
        cd = f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{quote(disp_name)}'
        return FileResponse(
            path,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": cd},
        )

    @api.get("/jobs/{job_id}/download/repair-formal-rollback.md")
    def download_repair_formal_rollback(job_id: str, p: Principal = Depends(bearer_principal)) -> FileResponse:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        path = rec.work_dir / "output" / "repair_formal_rollback.md"
        if not path.is_file() or path.stat().st_size == 0:
            raise HTTPException(404, "局部修复正式回滚报告尚未生成")
        ascii_fallback = "repair_formal_rollback.md"
        disp_name = f"{Path(rec.original_filename or 'translated').stem}_repair_formal_rollback.md"
        cd = f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{quote(disp_name)}'
        return FileResponse(
            path,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": cd},
        )

    @api.get("/jobs/{job_id}/download/repair-patch-review.md")
    def download_repair_patch_review(job_id: str, p: Principal = Depends(bearer_principal)) -> FileResponse:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        path = rec.work_dir / "output" / "repair_patch_review.md"
        if not path.is_file() or path.stat().st_size == 0:
            raise HTTPException(404, "局部修复补丁审核清单尚未生成")
        ascii_fallback = "repair_patch_review.md"
        disp_name = f"{Path(rec.original_filename or 'translated').stem}_repair_patch_review.md"
        cd = f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{quote(disp_name)}'
        return FileResponse(
            path,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": cd},
        )

    @api.get("/jobs/{job_id}/download/table-merged-cell-review.md")
    def download_table_merged_cell_review(
        job_id: str,
        p: Principal = Depends(bearer_principal),
    ) -> FileResponse:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        path = rec.work_dir / "output" / "table_merged_cell_review.md"
        if not path.is_file() or path.stat().st_size == 0:
            raise HTTPException(404, "表格合并候选确认清单尚未生成")
        ascii_fallback = "table_merged_cell_review.md"
        disp_name = f"{Path(rec.original_filename or 'translated').stem}_table_merged_cell_review.md"
        cd = f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{quote(disp_name)}'
        return FileResponse(
            path,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": cd},
        )

    @api.get("/jobs/{job_id}/download/table-structure-publish.md")
    def download_table_structure_publish(
        job_id: str,
        p: Principal = Depends(bearer_principal),
    ) -> FileResponse:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        path = rec.work_dir / "output" / "table_structure_publish.md"
        if not path.is_file() or path.stat().st_size == 0:
            raise HTTPException(404, "表格结构确认发布报告尚未生成")
        ascii_fallback = "table_structure_publish.md"
        disp_name = f"{Path(rec.original_filename or 'translated').stem}_table_structure_publish.md"
        cd = f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{quote(disp_name)}'
        return FileResponse(
            path,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": cd},
        )

    @api.get("/jobs/{job_id}/download/table-reconstruction-confirmed.json")
    def download_confirmed_table_reconstruction(
        job_id: str,
        p: Principal = Depends(bearer_principal),
    ) -> FileResponse:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        path = rec.work_dir / "output" / "table_reconstruction_confirmed.json"
        if not path.is_file() or path.stat().st_size == 0:
            raise HTTPException(404, "确认后的表格结构副本尚未生成")
        ascii_fallback = "table_reconstruction_confirmed.json"
        disp_name = f"{Path(rec.original_filename or 'translated').stem}_table_reconstruction_confirmed.json"
        cd = f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{quote(disp_name)}'
        return FileResponse(
            path,
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": cd},
        )

    @api.get("/jobs/{job_id}/download/glossary-retranslation-plan.md")
    def download_glossary_retranslation_plan_md(
        job_id: str,
        p: Principal = Depends(bearer_principal),
    ) -> FileResponse:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        path = rec.work_dir / "output" / "glossary_retranslation_plan.md"
        if not path.is_file() or path.stat().st_size == 0:
            raise HTTPException(404, "术语确认重译计划尚未生成")
        ascii_fallback = "glossary_retranslation_plan.md"
        disp_name = f"{Path(rec.original_filename or 'translated').stem}_glossary_retranslation_plan.md"
        cd = f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{quote(disp_name)}'
        return FileResponse(
            path,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": cd},
        )

    @api.get("/jobs/{job_id}/download/glossary-retranslation-plan.json")
    def download_glossary_retranslation_plan_json(
        job_id: str,
        p: Principal = Depends(bearer_principal),
    ) -> FileResponse:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        path = rec.work_dir / "output" / "glossary_retranslation_plan.json"
        if not path.is_file() or path.stat().st_size == 0:
            raise HTTPException(404, "术语确认重译计划尚未生成")
        ascii_fallback = "glossary_retranslation_plan.json"
        disp_name = f"{Path(rec.original_filename or 'translated').stem}_glossary_retranslation_plan.json"
        cd = f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{quote(disp_name)}'
        return FileResponse(
            path,
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": cd},
        )

    @api.get("/jobs/{job_id}/download/glossary-retranslation-result.md")
    def download_glossary_retranslation_result_md(
        job_id: str,
        p: Principal = Depends(bearer_principal),
    ) -> FileResponse:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        path = rec.work_dir / "output" / "glossary_retranslation_result.md"
        if not path.is_file() or path.stat().st_size == 0:
            raise HTTPException(404, "术语重译执行报告尚未生成")
        ascii_fallback = "glossary_retranslation_result.md"
        disp_name = f"{Path(rec.original_filename or 'translated').stem}_glossary_retranslation_result.md"
        cd = f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{quote(disp_name)}'
        return FileResponse(
            path,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": cd},
        )

    @api.get("/jobs/{job_id}/download/glossary-retranslation-result.json")
    def download_glossary_retranslation_result_json(
        job_id: str,
        p: Principal = Depends(bearer_principal),
    ) -> FileResponse:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        path = rec.work_dir / "output" / "glossary_retranslation_result.json"
        if not path.is_file() or path.stat().st_size == 0:
            raise HTTPException(404, "术语重译执行报告尚未生成")
        ascii_fallback = "glossary_retranslation_result.json"
        disp_name = f"{Path(rec.original_filename or 'translated').stem}_glossary_retranslation_result.json"
        cd = f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{quote(disp_name)}'
        return FileResponse(
            path,
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": cd},
        )

    @api.get("/jobs/{job_id}/download/glossary-retranslated-full.md")
    def download_glossary_retranslated_full(
        job_id: str,
        p: Principal = Depends(bearer_principal),
    ) -> FileResponse:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        path = rec.work_dir / "output" / "glossary_retranslated_full.md"
        if not path.is_file() or path.stat().st_size == 0:
            raise HTTPException(404, "术语候选重译全文尚未生成")
        ascii_fallback = "glossary_retranslated_full.md"
        disp_name = f"{Path(rec.original_filename or 'translated').stem}_glossary_retranslated_full.md"
        cd = f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{quote(disp_name)}'
        return FileResponse(
            path,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": cd},
        )

    @api.post("/jobs/{job_id}/glossary-retranslation/execute")
    def execute_glossary_retranslation_for_job(
        job_id: str,
        request: Request,
        payload: dict[str, Any] | None = Body(default=None),
        p: Principal = Depends(bearer_principal),
    ) -> dict[str, Any]:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        result = _execute_glossary_retranslation_for_record(rec, payload or {})
        summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
        app_registry.update(
            job_id,
            message=f"术语候选重译已执行：{summary.get('executed_chunk_count', 0)} 个分块",
        )
        database.log_audit(
            action="job_glossary_retranslation_execute",
            ip=client_ip(request),
            user_id=p.user_id,
            username=p.username,
            job_id=job_id,
            detail={
                "backend": summary.get("backend"),
                "mode": summary.get("mode"),
                "status": summary.get("status"),
                "requested_chunk_count": summary.get("requested_chunk_count"),
                "executed_chunk_count": summary.get("executed_chunk_count"),
                "failed_chunk_count": summary.get("failed_chunk_count"),
                "skipped_chunk_count": summary.get("skipped_chunk_count"),
            },
        )
        updated = app_registry.get(job_id) or rec
        d = _job_dict(updated)
        d["glossary_retranslation_execution_summary"] = summary
        d["glossary_retranslation_execution"] = result
        return d

    @api.get("/jobs/{job_id}/glossary-review")
    def get_glossary_review(job_id: str, p: Principal = Depends(bearer_principal)) -> dict[str, Any]:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        return _glossary_review_report_for_record(rec)

    @api.post("/jobs/{job_id}/glossary-review/batch")
    def update_glossary_review_batch(
        job_id: str,
        request: Request,
        payload: dict[str, Any] = Body(...),
        p: Principal = Depends(bearer_principal),
    ) -> dict[str, Any]:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        batch_items = _normalize_glossary_review_batch_items(payload)
        review_ids = [item["review_id"] for item in batch_items]
        before_report = _glossary_review_report_for_record(rec)
        _validate_glossary_review_ids(before_report, review_ids)
        _validate_glossary_review_batch_payloads(before_report, batch_items)
        updated_items = [
            _apply_glossary_review_decision(rec, item["review_id"], item["payload"], p.username)
            for item in batch_items
        ]
        retranslation_plan = _refresh_glossary_retranslation_plan_for_record(rec)
        plan_summary = (
            retranslation_plan.get("summary")
            if isinstance(retranslation_plan.get("summary"), dict)
            else {}
        )
        report = _glossary_review_report_for_record(rec)
        summary = _glossary_review_summary(report)
        app_registry.update(job_id, message=f"已批量更新 {len(updated_items)} 个术语审核项")
        decisions = sorted(
            {
                str(item.get("review_decision") or item.get("status") or "")
                for item in updated_items
                if isinstance(item, dict)
            }
        )
        database.log_audit(
            action="job_glossary_review_batch_update",
            ip=client_ip(request),
            user_id=p.user_id,
            username=p.username,
            job_id=job_id,
            detail={
                "review_ids": review_ids,
                "updated_count": len(updated_items),
                "decision": decisions[0] if len(decisions) == 1 else "mixed",
                "batch_schema": "items" if payload.get("items") is not None else "review_ids",
                "candidate_zh_changed_count": sum(
                    1 for item in updated_items if item.get("edited_candidate_zh")
                ),
                "pending_count": summary.get("pending_count"),
                "confirmed_count": summary.get("confirmed_count"),
                "rejected_count": summary.get("rejected_count"),
                "glossary_retranslation_plan_status": plan_summary.get("status"),
                "glossary_retranslation_plan_stale_chunk_count": plan_summary.get("stale_chunk_count"),
            },
        )
        updated = app_registry.get(job_id) or rec
        d = _job_dict(updated)
        d["glossary_review_summary"] = summary
        d["glossary_review"] = report
        d["glossary_retranslation_plan_summary"] = plan_summary
        return d

    @api.post("/jobs/{job_id}/glossary-review/{review_id:path}")
    def update_glossary_review(
        job_id: str,
        review_id: str,
        request: Request,
        payload: dict[str, Any] = Body(...),
        p: Principal = Depends(bearer_principal),
    ) -> dict[str, Any]:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        before_report = _glossary_review_report_for_record(rec)
        _validate_glossary_review_ids(before_report, [review_id])
        updated_item = _apply_glossary_review_decision(rec, review_id, payload, p.username)
        retranslation_plan = _refresh_glossary_retranslation_plan_for_record(rec)
        plan_summary = (
            retranslation_plan.get("summary")
            if isinstance(retranslation_plan.get("summary"), dict)
            else {}
        )
        report = _glossary_review_report_for_record(rec)
        summary = _glossary_review_summary(report)
        app_registry.update(job_id, message="已更新术语审核项")
        database.log_audit(
            action="job_glossary_review_update",
            ip=client_ip(request),
            user_id=p.user_id,
            username=p.username,
            job_id=job_id,
            detail={
                "review_id": review_id,
                "decision": payload.get("decision"),
                "candidate_zh_changed": bool(updated_item.get("edited_candidate_zh")),
                "pending_count": summary.get("pending_count"),
                "confirmed_count": summary.get("confirmed_count"),
                "rejected_count": summary.get("rejected_count"),
                "glossary_retranslation_plan_status": plan_summary.get("status"),
                "glossary_retranslation_plan_stale_chunk_count": plan_summary.get("stale_chunk_count"),
            },
        )
        updated = app_registry.get(job_id) or rec
        d = _job_dict(updated)
        d["glossary_review_summary"] = summary
        d["glossary_review"] = report
        d["glossary_retranslation_plan_summary"] = plan_summary
        return d

    @api.get("/jobs/{job_id}/table-merged-cell-review")
    def get_table_merged_cell_review(job_id: str, p: Principal = Depends(bearer_principal)) -> dict:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        out_dir = rec.work_dir / "output"
        table_reconstruction = _read_json_artifact(
            out_dir / "table_reconstruction.json",
            missing_message="表格重建报告尚未生成",
            invalid_message="表格重建报告无法解析",
        )
        return _read_or_create_table_merged_cell_review(out_dir, table_reconstruction)

    @api.get("/jobs/{job_id}/table-merged-cell-review/{review_id}/preview.png")
    def preview_table_merged_cell_review(
        job_id: str,
        review_id: str,
        p: Principal = Depends(bearer_principal),
    ) -> Response:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "job not found")
        out_dir = rec.work_dir / "output"
        table_reconstruction = _read_json_artifact(
            out_dir / "table_reconstruction.json",
            missing_message="table reconstruction report is not available",
            invalid_message="table reconstruction report cannot be parsed",
        )
        report = _read_or_create_table_merged_cell_review(out_dir, table_reconstruction)
        try:
            review = _find_table_merged_cell_review_item(report, review_id)
        except KeyError as exc:
            raise HTTPException(404, f"table merged cell review item not found: {review_id}") from exc
        return Response(
            content=_render_table_merged_cell_review_preview_for_record(rec, review),
            media_type="image/png",
            headers={
                "Cache-Control": "private, max-age=60",
                "Content-Disposition": 'inline; filename="table-merged-cell-review-preview.png"',
            },
        )

    @api.get("/jobs/{job_id}/table-structure-publish")
    def get_table_structure_publish(job_id: str, p: Principal = Depends(bearer_principal)) -> dict:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        out_dir = rec.work_dir / "output"
        table_reconstruction = _read_json_artifact(
            out_dir / "table_reconstruction.json",
            missing_message="表格重建报告尚未生成",
            invalid_message="表格重建报告无法解析",
        )
        table_merged_cell_review = _read_or_create_table_merged_cell_review(out_dir, table_reconstruction)
        return _read_or_create_table_structure_publish(
            out_dir,
            table_reconstruction,
            table_merged_cell_review,
        )

    @api.post("/jobs/{job_id}/table-merged-cell-review/batch")
    def update_table_merged_cell_review_batch(
        job_id: str,
        request: Request,
        payload: dict[str, Any] = Body(...),
        p: Principal = Depends(bearer_principal),
    ) -> dict:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        if rec.status != "done":
            raise HTTPException(409, "任务尚未完成，暂不能批量审核表格合并候选")
        out_dir = rec.work_dir / "output"
        table_reconstruction = _read_json_artifact(
            out_dir / "table_reconstruction.json",
            missing_message="表格重建报告尚未生成",
            invalid_message="表格重建报告无法解析",
        )
        _read_or_create_table_merged_cell_review(out_dir, table_reconstruction)
        review_ids = payload.get("review_ids")
        try:
            report = write_table_merged_cell_review_batch_decision(
                out_dir / "table_merged_cell_review.json",
                out_dir / "table_merged_cell_review.md",
                review_ids,
                decision=payload.get("decision"),
                reviewer=p.username,
                comment=payload.get("comment") or "",
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, f"表格合并候选审核项不存在：{exc}") from exc
        except FileNotFoundError as exc:
            raise HTTPException(404, "表格合并候选确认清单尚未生成") from exc
        updated_review_ids: list[str] = []
        seen_review_ids: set[str] = set()
        if isinstance(review_ids, list):
            for item in review_ids:
                item_id = str(item or "").strip()
                if item_id and item_id not in seen_review_ids:
                    seen_review_ids.add(item_id)
                    updated_review_ids.append(item_id)
        write_table_structure_publish(
            table_reconstruction,
            report,
            out_dir / "table_structure_publish.json",
            out_dir / "table_structure_publish.md",
            confirm=False,
            published_reconstruction_path=out_dir / "table_reconstruction_confirmed.json",
        )
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        updated_count = len(updated_review_ids)
        app_registry.update(job_id, message=f"已批量更新 {updated_count} 个表格合并候选审核项")
        database.log_audit(
            action="job_table_merged_cell_review_batch_update",
            ip=client_ip(request),
            user_id=p.user_id,
            username=p.username,
            job_id=job_id,
            detail={
                "review_ids": updated_review_ids,
                "updated_count": updated_count,
                "decision": payload.get("decision"),
                "review_required_count": summary.get("review_required_count"),
                "human_reviewed_count": summary.get("human_reviewed_count"),
            },
        )
        updated = app_registry.get(job_id) or rec
        d = _job_dict(updated)
        d["table_merged_cell_review_summary"] = summary
        return d

    @api.post("/jobs/{job_id}/table-merged-cell-review/{review_id}")
    def update_table_merged_cell_review(
        job_id: str,
        review_id: str,
        request: Request,
        payload: dict[str, Any] = Body(...),
        p: Principal = Depends(bearer_principal),
    ) -> dict:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        if rec.status != "done":
            raise HTTPException(409, "任务尚未完成，暂不能审核表格合并候选")
        out_dir = rec.work_dir / "output"
        table_reconstruction = _read_json_artifact(
            out_dir / "table_reconstruction.json",
            missing_message="表格重建报告尚未生成",
            invalid_message="表格重建报告无法解析",
        )
        _read_or_create_table_merged_cell_review(out_dir, table_reconstruction)
        try:
            report = write_table_merged_cell_review_decision(
                out_dir / "table_merged_cell_review.json",
                out_dir / "table_merged_cell_review.md",
                review_id,
                decision=payload.get("decision"),
                reviewer=p.username,
                comment=payload.get("comment") or "",
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, f"表格合并候选审核项不存在：{review_id}") from exc
        except FileNotFoundError as exc:
            raise HTTPException(404, "表格合并候选确认清单尚未生成") from exc
        write_table_structure_publish(
            table_reconstruction,
            report,
            out_dir / "table_structure_publish.json",
            out_dir / "table_structure_publish.md",
            confirm=False,
            published_reconstruction_path=out_dir / "table_reconstruction_confirmed.json",
        )
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        app_registry.update(job_id, message="已更新表格合并候选审核")
        database.log_audit(
            action="job_table_merged_cell_review_update",
            ip=client_ip(request),
            user_id=p.user_id,
            username=p.username,
            job_id=job_id,
            detail={
                "review_id": review_id,
                "decision": payload.get("decision"),
                "review_required_count": summary.get("review_required_count"),
                "human_reviewed_count": summary.get("human_reviewed_count"),
            },
        )
        updated = app_registry.get(job_id) or rec
        d = _job_dict(updated)
        d["table_merged_cell_review_summary"] = summary
        return d

    @api.post("/jobs/{job_id}/table-structure-publish/confirm")
    def confirm_table_structure_publish(
        job_id: str,
        request: Request,
        p: Principal = Depends(bearer_principal),
    ) -> dict:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        report = _confirm_table_structure_publish_for_record(rec)
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        app_registry.update(job_id, message="已确认并生成表格结构副本")
        database.log_audit(
            action="job_table_structure_publish_confirm",
            ip=client_ip(request),
            user_id=p.user_id,
            username=p.username,
            job_id=job_id,
            detail={
                "publish_status": summary.get("publish_status"),
                "blocking_review_count": summary.get("blocking_review_count"),
                "applied_confirmed_count": summary.get("applied_confirmed_count"),
                "published_reconstruction_path": summary.get("published_reconstruction_path"),
            },
        )
        updated = app_registry.get(job_id) or rec
        d = _job_dict(updated)
        d["table_structure_publish_summary"] = summary
        return d

    @api.get("/jobs/{job_id}/repair-patch-review")
    def get_repair_patch_review(job_id: str, p: Principal = Depends(bearer_principal)) -> dict:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        out_dir = rec.work_dir / "output"
        repair_merge = _read_json_artifact(
            out_dir / "repair_merge.json",
            missing_message="局部修复合并报告尚未生成",
            invalid_message="局部修复合并报告无法解析",
        )
        return _read_or_create_repair_patch_review(out_dir, repair_merge)

    @api.post("/jobs/{job_id}/repair-patch-review/batch")
    def update_repair_patch_review_batch(
        job_id: str,
        request: Request,
        payload: dict[str, Any] = Body(...),
        p: Principal = Depends(bearer_principal),
    ) -> dict:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        if rec.status != "done":
            raise HTTPException(409, "任务尚未完成，暂不能批量审核局部修复补丁")
        out_dir = rec.work_dir / "output"
        repair_merge = _read_json_artifact(
            out_dir / "repair_merge.json",
            missing_message="局部修复合并报告尚未生成",
            invalid_message="局部修复合并报告无法解析",
        )
        _read_or_create_repair_patch_review(out_dir, repair_merge)
        review_ids = payload.get("review_ids")
        try:
            report = write_repair_patch_review_batch_decision(
                out_dir / "repair_patch_review.json",
                out_dir / "repair_patch_review.md",
                review_ids,
                decision=payload.get("decision"),
                reviewer=p.username,
                comment=payload.get("comment") or "",
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, f"补丁审核项不存在：{exc}") from exc
        except FileNotFoundError as exc:
            raise HTTPException(404, "局部修复补丁审核报告尚未生成") from exc
        updated_review_ids: list[str] = []
        seen_review_ids: set[str] = set()
        if isinstance(review_ids, list):
            for item in review_ids:
                item_id = str(item or "").strip()
                if item_id and item_id not in seen_review_ids:
                    seen_review_ids.add(item_id)
                    updated_review_ids.append(item_id)
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        updated_count = len(updated_review_ids)
        app_registry.update(job_id, message=f"已批量更新 {updated_count} 个局部修复补丁审核项")
        database.log_audit(
            action="job_repair_patch_review_batch_update",
            ip=client_ip(request),
            user_id=p.user_id,
            username=p.username,
            job_id=job_id,
            detail={
                "review_ids": updated_review_ids,
                "updated_count": updated_count,
                "decision": payload.get("decision"),
                "publish_blocking_count": summary.get("publish_blocking_count"),
                "human_reviewed_count": summary.get("human_reviewed_count"),
            },
        )
        updated = app_registry.get(job_id) or rec
        d = _job_dict(updated)
        d["repair_patch_review_summary"] = summary
        return d

    @api.post("/jobs/{job_id}/repair-patch-review/{review_id}")
    def update_repair_patch_review(
        job_id: str,
        review_id: str,
        request: Request,
        payload: dict[str, Any] = Body(...),
        p: Principal = Depends(bearer_principal),
    ) -> dict:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        if rec.status != "done":
            raise HTTPException(409, "任务尚未完成，暂不能审核局部修复补丁")
        out_dir = rec.work_dir / "output"
        repair_merge = _read_json_artifact(
            out_dir / "repair_merge.json",
            missing_message="局部修复合并报告尚未生成",
            invalid_message="局部修复合并报告无法解析",
        )
        _read_or_create_repair_patch_review(out_dir, repair_merge)
        try:
            report = write_repair_patch_review_decision(
                out_dir / "repair_patch_review.json",
                out_dir / "repair_patch_review.md",
                review_id,
                decision=payload.get("decision"),
                reviewer=p.username,
                comment=payload.get("comment") or "",
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, f"补丁审核项不存在：{review_id}") from exc
        except FileNotFoundError as exc:
            raise HTTPException(404, "局部修复补丁审核报告尚未生成") from exc
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        app_registry.update(job_id, message="已更新局部修复补丁审核")
        database.log_audit(
            action="job_repair_patch_review_update",
            ip=client_ip(request),
            user_id=p.user_id,
            username=p.username,
            job_id=job_id,
            detail={
                "review_id": review_id,
                "decision": payload.get("decision"),
                "publish_blocking_count": summary.get("publish_blocking_count"),
                "human_reviewed_count": summary.get("human_reviewed_count"),
            },
        )
        updated = app_registry.get(job_id) or rec
        d = _job_dict(updated)
        d["repair_patch_review_summary"] = summary
        return d

    @api.get("/jobs/{job_id}/download/published-full.md")
    def download_published_full(job_id: str, p: Principal = Depends(bearer_principal)) -> FileResponse:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        path = rec.work_dir / "output" / "published_full.md"
        if not path.is_file() or path.stat().st_size == 0:
            raise HTTPException(404, "人工确认修复发布稿尚未生成")
        ascii_fallback = "published_full.md"
        disp_name = f"{Path(rec.original_filename or 'translated').stem}_published_full.md"
        cd = f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{quote(disp_name)}'
        return FileResponse(
            path,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": cd},
        )

    @api.get("/jobs/{job_id}/download/rollback-full.md")
    def download_rollback_full(job_id: str, p: Principal = Depends(bearer_principal)) -> FileResponse:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        path = rec.work_dir / "output" / "rollback_full.md"
        if not path.is_file() or path.stat().st_size == 0:
            raise HTTPException(404, "局部修复回滚演练稿尚未生成")
        ascii_fallback = "rollback_full.md"
        disp_name = f"{Path(rec.original_filename or 'translated').stem}_rollback_full.md"
        cd = f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{quote(disp_name)}'
        return FileResponse(
            path,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": cd},
        )

    @api.get("/jobs/{job_id}/download/formal-full.md")
    def download_formal_full(job_id: str, p: Principal = Depends(bearer_principal)) -> FileResponse:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        path = rec.work_dir / "output" / "formal_full.md"
        if not path.is_file() or path.stat().st_size == 0:
            raise HTTPException(404, "正式译文尚未生成")
        ascii_fallback = "formal_full.md"
        disp_name = f"{Path(rec.original_filename or 'translated').stem}_formal_full.md"
        cd = f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{quote(disp_name)}'
        return FileResponse(
            path,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": cd},
        )

    @api.get("/jobs/{job_id}/download/formal-before-repair-full.md")
    def download_formal_before_repair_full(
        job_id: str,
        p: Principal = Depends(bearer_principal),
    ) -> FileResponse:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        path = rec.work_dir / "output" / "formal_full.before_repair.md"
        if not path.is_file() or path.stat().st_size == 0:
            raise HTTPException(404, "正式译文修复前备份尚未生成")
        ascii_fallback = "formal_full.before_repair.md"
        disp_name = f"{Path(rec.original_filename or 'translated').stem}_formal_before_repair.md"
        cd = f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{quote(disp_name)}'
        return FileResponse(
            path,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": cd},
        )

    @api.post("/jobs/{job_id}/repair-publish/confirm")
    def confirm_repair_publish(
        job_id: str,
        request: Request,
        p: Principal = Depends(bearer_principal),
    ) -> dict:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        report = _confirm_repair_publish_for_record(rec)
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        app_registry.update(job_id, message="已确认并生成局部修复发布稿")
        database.log_audit(
            action="job_repair_publish_confirm",
            ip=client_ip(request),
            user_id=p.user_id,
            username=p.username,
            job_id=job_id,
            detail={
                "publish_status": summary.get("publish_status"),
                "open_merge_issue_count": summary.get("open_merge_issue_count"),
                "published_full_path": summary.get("published_full_path"),
                "rollback_available": summary.get("rollback_available"),
            },
        )
        updated = app_registry.get(job_id) or rec
        d = _job_dict(updated)
        d["repair_publish_summary"] = summary
        return d

    @api.post("/jobs/{job_id}/repair-rollback/confirm")
    def confirm_repair_rollback(
        job_id: str,
        request: Request,
        p: Principal = Depends(bearer_principal),
    ) -> dict:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        report = _confirm_repair_rollback_for_record(rec)
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        app_registry.update(job_id, message="已生成局部修复回滚演练稿")
        database.log_audit(
            action="job_repair_rollback_confirm",
            ip=client_ip(request),
            user_id=p.user_id,
            username=p.username,
            job_id=job_id,
            detail={
                "rollback_status": summary.get("rollback_status"),
                "rollback_full_path": summary.get("rollback_full_path"),
                "rollback_matches_original": summary.get("rollback_matches_original"),
            },
        )
        updated = app_registry.get(job_id) or rec
        d = _job_dict(updated)
        d["repair_rollback_summary"] = summary
        return d

    @api.post("/jobs/{job_id}/repair-formal-replace/confirm")
    def confirm_repair_formal_replace(
        job_id: str,
        request: Request,
        p: Principal = Depends(bearer_principal),
    ) -> dict:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        report = _confirm_repair_formal_replace_for_record(rec)
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        app_registry.update(job_id, message="已生成正式修复译文")
        database.log_audit(
            action="job_repair_formal_replace_confirm",
            ip=client_ip(request),
            user_id=p.user_id,
            username=p.username,
            job_id=job_id,
            detail={
                "replace_status": summary.get("replace_status"),
                "formal_full_path": summary.get("formal_full_path"),
                "backup_full_path": summary.get("backup_full_path"),
                "formal_matches_published": summary.get("formal_matches_published"),
            },
        )
        updated = app_registry.get(job_id) or rec
        d = _job_dict(updated)
        d["repair_formal_replace_summary"] = summary
        return d

    @api.post("/jobs/{job_id}/repair-formal-rollback/confirm")
    def confirm_repair_formal_rollback(
        job_id: str,
        request: Request,
        p: Principal = Depends(bearer_principal),
    ) -> dict:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        report = _confirm_repair_formal_rollback_for_record(rec)
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        app_registry.update(job_id, message="已把正式译文回滚到修复前备份")
        database.log_audit(
            action="job_repair_formal_rollback_confirm",
            ip=client_ip(request),
            user_id=p.user_id,
            username=p.username,
            job_id=job_id,
            detail={
                "rollback_status": summary.get("rollback_status"),
                "formal_full_path": summary.get("formal_full_path"),
                "backup_full_path": summary.get("backup_full_path"),
                "formal_matches_backup": summary.get("formal_matches_backup"),
            },
        )
        updated = app_registry.get(job_id) or rec
        d = _job_dict(updated)
        d["repair_formal_rollback_summary"] = summary
        return d

    @api.get("/jobs/{job_id}/download/bundle.zip")
    def download_zip(job_id: str, p: Principal = Depends(bearer_principal)) -> Response:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        if rec.status not in ("done", "cancelled"):
            raise HTTPException(409, "任务未完成或未终止，暂不可打包下载")
        md = rec.work_dir / "output" / "translated_full.md"
        if not md.is_file() or md.stat().st_size == 0:
            raise HTTPException(404, "尚无译文可打包")
        data, zip_disp = zip_job_outputs(
            rec.work_dir,
            original_filename=rec.original_filename,
            complete=rec.status == "done",
        )
        ascii_fb = "bundle.zip"
        cd = f'attachment; filename="{ascii_fb}"; filename*=UTF-8\'\'{quote(zip_disp)}'
        return Response(content=data, media_type="application/zip", headers={"Content-Disposition": cd})

    admin = APIRouter(prefix="/admin", tags=["admin"])

    @admin.get("/settings")
    def admin_get_settings(_: Principal = Depends(require_admin)) -> dict:
        snap = settings_service.admin_settings_snapshot()
        return snap

    @admin.get("/security/preflight")
    def admin_security_preflight(_: Principal = Depends(require_admin)) -> dict:
        return build_security_preflight(
            require_data_dir(),
            app_registry.data_root,
        )

    @admin.put("/settings")
    def admin_put_settings(body: dict = Body(...), p: Principal = Depends(require_admin)) -> dict:
        try:
            settings_service.apply_admin_settings(body)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        database.log_audit(
            action="admin_settings_update",
            ip=None,
            user_id=p.user_id,
            username=p.username,
            detail={"keys": list(body.keys())},
        )
        return {"ok": True, "settings": settings_service.admin_settings_snapshot()}

    @admin.get("/audit")
    def admin_audit(
        _: Principal = Depends(require_admin),
        limit: int = 200,
        offset: int = 0,
    ) -> dict:
        return {"events": database.list_audit(limit=limit, offset=offset)}

    @admin.get("/users")
    def admin_users(_: Principal = Depends(require_admin)) -> dict:
        return {"users": database.list_users()}

    @admin.get("/jobs")
    def admin_jobs(_: Principal = Depends(require_admin), limit: int = 500) -> dict:
        return {"jobs": app_registry.merge_status_into_rows(database.list_all_jobs(limit=limit))}

    @admin.get("/jobs/hydration-report")
    def admin_jobs_hydration_report(_: Principal = Depends(require_admin)) -> dict:
        return {
            "hydration": app_registry.hydration_report(),
            "drift": app_registry.storage_drift(set(database.list_all_job_ids())),
        }

    @admin.post("/jobs/{job_id}/requeue")
    def admin_requeue_recovered_job(job_id: str, p: Principal = Depends(require_admin)) -> dict:
        rec = app_registry.get(job_id)
        if not rec:
            raise HTTPException(404, "任务不存在")
        report = app_registry.requeue_recovered_jobs(
            policy="safe",
            max_jobs=1,
            cfg=settings_service.effective_app_config(),
            starter=start_job_thread,
            audit=True,
            job_ids={job_id},
            audit_user_id=p.user_id,
            audit_username=p.username,
        )
        if job_id not in report.get("auto_resume_started_job_ids", []):
            raise HTTPException(409, detail=report)
        return {"ok": True, "recovery": report}

    @admin.post("/jobs/reconcile")
    def admin_jobs_reconcile(
        body: dict | None = Body(None),
        p: Principal = Depends(require_admin),
    ) -> dict:
        apply_cleanup = bool((body or {}).get("apply"))
        indexed = set(database.list_all_job_ids())
        drift = app_registry.storage_drift(indexed)
        deleted_db_rows: list[str] = []
        deleted_work_dirs: list[str] = []
        skipped_active: list[str] = []

        if apply_cleanup:
            active = set(drift.get("active_job_ids") or [])
            for job_id in drift.get("missing_work_dir_job_ids") or []:
                if job_id in active:
                    skipped_active.append(job_id)
                    continue
                database.delete_job_meta_row(job_id)
                app_registry.remove_job(job_id)
                deleted_db_rows.append(job_id)
            for job_id in drift.get("unindexed_work_dir_job_ids") or []:
                if job_id in active:
                    skipped_active.append(job_id)
                    continue
                if app_registry.remove_job(job_id):
                    deleted_work_dirs.append(job_id)
            indexed = set(database.list_all_job_ids())
            drift = app_registry.storage_drift(indexed)
            database.log_audit(
                action="admin_jobs_reconcile",
                ip=None,
                user_id=p.user_id,
                username=p.username,
                detail={
                    "apply": True,
                    "deleted_db_rows": deleted_db_rows,
                    "deleted_work_dirs": deleted_work_dirs,
                    "skipped_active": skipped_active,
                },
            )

        return {
            "apply": apply_cleanup,
            "drift": drift,
            "deleted_db_rows": deleted_db_rows,
            "deleted_work_dirs": deleted_work_dirs,
            "skipped_active": skipped_active,
        }

    @admin.get("/jobs/{job_id}/artifact", response_model=None)
    def admin_artifact(
        job_id: str,
        kind: str,
        _: Principal = Depends(require_admin),
    ) -> Response | FileResponse:
        rec = app_registry.get(job_id)
        if not rec:
            raise HTTPException(404, "任务不存在")
        root = rec.work_dir.resolve()
        if kind == "input":
            p = root / "input.pdf"
            if not p.is_file() or p.stat().st_size == 0:
                raise HTTPException(404, "无上传文件")
            return FileResponse(p, filename="input.pdf", media_type="application/pdf")
        if kind == "output_md":
            p = root / "output" / "translated_full.md"
            if not p.is_file() or p.stat().st_size == 0:
                raise HTTPException(404, "尚无译文")
            return FileResponse(p, filename="translated_full.md", media_type="text/markdown; charset=utf-8")
        if kind == "output_pdf":
            p = root / "output" / "translated_full.pdf"
            if not p.is_file() or p.stat().st_size == 0:
                raise HTTPException(404, "PDF 译文尚未生成")
            return FileResponse(p, filename="translated_full.pdf", media_type="application/pdf")
        if kind == "repair_publish":
            p = root / "output" / "repair_publish.md"
            if not p.is_file() or p.stat().st_size == 0:
                raise HTTPException(404, "局部修复发布确认报告尚未生成")
            return FileResponse(p, filename="repair_publish.md", media_type="text/markdown; charset=utf-8")
        if kind == "repair_effectiveness":
            p = root / "output" / "repair_effectiveness.md"
            if not p.is_file() or p.stat().st_size == 0:
                raise HTTPException(404, "局部修复效果对比报告尚未生成")
            return FileResponse(p, filename="repair_effectiveness.md", media_type="text/markdown; charset=utf-8")
        if kind == "repair_rollback":
            p = root / "output" / "repair_rollback.md"
            if not p.is_file() or p.stat().st_size == 0:
                raise HTTPException(404, "局部修复回滚演练报告尚未生成")
            return FileResponse(p, filename="repair_rollback.md", media_type="text/markdown; charset=utf-8")
        if kind == "repair_formal_replace":
            p = root / "output" / "repair_formal_replace.md"
            if not p.is_file() or p.stat().st_size == 0:
                raise HTTPException(404, "局部修复正式替换报告尚未生成")
            return FileResponse(p, filename="repair_formal_replace.md", media_type="text/markdown; charset=utf-8")
        if kind == "repair_formal_rollback":
            p = root / "output" / "repair_formal_rollback.md"
            if not p.is_file() or p.stat().st_size == 0:
                raise HTTPException(404, "局部修复正式回滚报告尚未生成")
            return FileResponse(p, filename="repair_formal_rollback.md", media_type="text/markdown; charset=utf-8")
        if kind == "repair_patch_review":
            p = root / "output" / "repair_patch_review.md"
            if not p.is_file() or p.stat().st_size == 0:
                raise HTTPException(404, "局部修复补丁审核清单尚未生成")
            return FileResponse(p, filename="repair_patch_review.md", media_type="text/markdown; charset=utf-8")
        if kind == "table_merged_cell_review":
            p = root / "output" / "table_merged_cell_review.md"
            if not p.is_file() or p.stat().st_size == 0:
                raise HTTPException(404, "表格合并候选确认清单尚未生成")
            return FileResponse(
                p,
                filename="table_merged_cell_review.md",
                media_type="text/markdown; charset=utf-8",
            )
        if kind == "table_structure_publish":
            p = root / "output" / "table_structure_publish.md"
            if not p.is_file() or p.stat().st_size == 0:
                raise HTTPException(404, "表格结构确认发布报告尚未生成")
            return FileResponse(
                p,
                filename="table_structure_publish.md",
                media_type="text/markdown; charset=utf-8",
            )
        if kind == "table_reconstruction_confirmed":
            p = root / "output" / "table_reconstruction_confirmed.json"
            if not p.is_file() or p.stat().st_size == 0:
                raise HTTPException(404, "确认后的表格结构副本尚未生成")
            return FileResponse(
                p,
                filename="table_reconstruction_confirmed.json",
                media_type="application/json; charset=utf-8",
            )
        if kind == "glossary_retranslation_plan_md":
            p = root / "output" / "glossary_retranslation_plan.md"
            if not p.is_file() or p.stat().st_size == 0:
                raise HTTPException(404, "术语确认重译计划尚未生成")
            return FileResponse(
                p,
                filename="glossary_retranslation_plan.md",
                media_type="text/markdown; charset=utf-8",
            )
        if kind == "glossary_retranslation_plan_json":
            p = root / "output" / "glossary_retranslation_plan.json"
            if not p.is_file() or p.stat().st_size == 0:
                raise HTTPException(404, "术语确认重译计划尚未生成")
            return FileResponse(
                p,
                filename="glossary_retranslation_plan.json",
                media_type="application/json; charset=utf-8",
            )
        if kind == "glossary_retranslation_result_md":
            p = root / "output" / "glossary_retranslation_result.md"
            if not p.is_file() or p.stat().st_size == 0:
                raise HTTPException(404, "术语重译执行报告尚未生成")
            return FileResponse(
                p,
                filename="glossary_retranslation_result.md",
                media_type="text/markdown; charset=utf-8",
            )
        if kind == "glossary_retranslation_result_json":
            p = root / "output" / "glossary_retranslation_result.json"
            if not p.is_file() or p.stat().st_size == 0:
                raise HTTPException(404, "术语重译执行报告尚未生成")
            return FileResponse(
                p,
                filename="glossary_retranslation_result.json",
                media_type="application/json; charset=utf-8",
            )
        if kind == "glossary_retranslated_full":
            p = root / "output" / "glossary_retranslated_full.md"
            if not p.is_file() or p.stat().st_size == 0:
                raise HTTPException(404, "术语候选重译全文尚未生成")
            return FileResponse(
                p,
                filename="glossary_retranslated_full.md",
                media_type="text/markdown; charset=utf-8",
            )
        if kind == "repair_published_full":
            p = root / "output" / "published_full.md"
            if not p.is_file() or p.stat().st_size == 0:
                raise HTTPException(404, "人工确认修复发布稿尚未生成")
            return FileResponse(p, filename="published_full.md", media_type="text/markdown; charset=utf-8")
        if kind == "repair_rollback_full":
            p = root / "output" / "rollback_full.md"
            if not p.is_file() or p.stat().st_size == 0:
                raise HTTPException(404, "局部修复回滚演练稿尚未生成")
            return FileResponse(p, filename="rollback_full.md", media_type="text/markdown; charset=utf-8")
        if kind == "repair_formal_full":
            p = root / "output" / "formal_full.md"
            if not p.is_file() or p.stat().st_size == 0:
                raise HTTPException(404, "正式译文尚未生成")
            return FileResponse(p, filename="formal_full.md", media_type="text/markdown; charset=utf-8")
        if kind == "repair_formal_backup_full":
            p = root / "output" / "formal_full.before_repair.md"
            if not p.is_file() or p.stat().st_size == 0:
                raise HTTPException(404, "正式译文修复前备份尚未生成")
            return FileResponse(
                p,
                filename="formal_full.before_repair.md",
                media_type="text/markdown; charset=utf-8",
            )
        if kind == "repair_formal_active_before_rollback_full":
            p = root / "output" / "formal_full.repair_applied.md"
            if not p.is_file() or p.stat().st_size == 0:
                raise HTTPException(404, "正式回滚前修复稿尚未生成")
            return FileResponse(
                p,
                filename="formal_full.repair_applied.md",
                media_type="text/markdown; charset=utf-8",
            )
        if kind == "bundle_zip":
            if rec.status not in ("done", "cancelled"):
                raise HTTPException(409, "任务未完成或未终止")
            md = root / "output" / "translated_full.md"
            if not md.is_file() or md.stat().st_size == 0:
                raise HTTPException(404, "尚无译文可打包")
            data, zip_disp = zip_job_outputs(
                rec.work_dir,
                original_filename=rec.original_filename,
                complete=rec.status == "done",
            )
            ascii_fb = "bundle.zip"
            cd = f'attachment; filename="{ascii_fb}"; filename*=UTF-8\'\'{quote(zip_disp)}'
            return Response(content=data, media_type="application/zip", headers={"Content-Disposition": cd})
        raise HTTPException(
            400,
            "kind 必须是 input / output_md / output_pdf / repair_publish / repair_effectiveness / repair_rollback / repair_formal_replace / repair_formal_rollback / repair_patch_review / table_merged_cell_review / table_structure_publish / table_reconstruction_confirmed / glossary_retranslation_plan_md / glossary_retranslation_plan_json / glossary_retranslation_result_md / glossary_retranslation_result_json / glossary_retranslated_full / repair_published_full / repair_rollback_full / repair_formal_full / repair_formal_backup_full / repair_formal_active_before_rollback_full / bundle_zip",
        )

    api.include_router(admin)
    return api
