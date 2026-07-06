from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "vlm-retranslation-plan-v1"


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _read_json_dict(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        return dict(default or {})
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else dict(default or {})


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
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


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _task_list(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    raw = payload.get("tasks") if isinstance(payload, dict) else []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _result_list(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    raw = payload.get("results") if isinstance(payload, dict) else []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _promotion_list(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    raw = payload.get("promotions") if isinstance(payload, dict) else []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _task_id(item: dict[str, Any]) -> str:
    return str(item.get("task_id") or "").strip()


def _writeback_block_id(task: dict[str, Any]) -> str:
    writeback = task.get("writeback") if isinstance(task.get("writeback"), dict) else {}
    return str(writeback.get("block_id") or "").strip()


def _page_no(*items: dict[str, Any]) -> int:
    for item in items:
        value = _as_int(item.get("page_no"))
        if value > 0:
            return value
    for item in items:
        writeback = item.get("writeback") if isinstance(item.get("writeback"), dict) else {}
        value = _as_int(writeback.get("page_no"))
        if value > 0:
            return value
    return 0


def _block_id(*items: dict[str, Any]) -> str:
    for item in items:
        value = str(item.get("block_id") or "").strip()
        if value:
            return value
    for item in items:
        value = _writeback_block_id(item)
        if value:
            return value
    return ""


def _target_structure_type(*items: dict[str, Any]) -> str:
    for item in items:
        value = str(item.get("target_structure_type") or "").strip()
        if value:
            return value
    return ""


def _chunk_pages(entry: dict[str, Any]) -> set[int]:
    raw = entry.get("pages_1based")
    if not isinstance(raw, list):
        return set()
    pages = [_as_int(item) for item in raw]
    pages = [item for item in pages if item > 0]
    if not pages:
        return set()
    if len(pages) >= 2:
        start = min(pages[0], pages[-1])
        end = max(pages[0], pages[-1])
        return set(range(start, end + 1))
    return {pages[0]}


def _entry_block_ids(entry: dict[str, Any]) -> list[str]:
    return _string_list(entry.get("block_ids"))


def _entry_block_types(entry: dict[str, Any]) -> list[str]:
    raw = entry.get("block_types")
    if not isinstance(raw, dict):
        return []
    return [
        str(key)
        for key, value in raw.items()
        if str(key).strip() and _as_int(value) > 0
    ]


def _translation_path(output_dir: Path, chunk_id: str) -> str:
    path = output_dir / "chunks" / f"{chunk_id}.md"
    return f"output/chunks/{chunk_id}.md" if path.is_file() else ""


def _mapping_reasons(result: dict[str, Any], task: dict[str, Any], promotion: dict[str, Any]) -> list[str]:
    reasons = ["vlm_result_applied"]
    if promotion.get("structured_table_promoted"):
        reasons.append("structured_table_promoted")
    if promotion.get("structured_formula_promoted"):
        reasons.append("structured_formula_promoted")
    if promotion.get("canonical_structure_targets"):
        reasons.append("canonical_structure_updated")
    if str(promotion.get("promotion_target") or "") == "document_ir.page.blocks.synthetic":
        reasons.append("synthetic_block_created")
    layout_scope = str(task.get("layout_scope") or "")
    if layout_scope:
        reasons.append(f"layout_scope:{layout_scope}")
    result_engine = str(result.get("engine") or "")
    if result_engine:
        reasons.append(f"engine:{result_engine}")
    return reasons


def _match_chunks(
    chunks: list[dict[str, Any]],
    *,
    block_id: str,
    page_no: int,
) -> tuple[list[dict[str, Any]], str]:
    if block_id:
        matched = [entry for entry in chunks if block_id in _entry_block_ids(entry)]
        if matched:
            return matched, "block_id"
    if page_no > 0:
        matched = [entry for entry in chunks if page_no in _chunk_pages(entry)]
        if matched:
            return matched, "page_range"
    return [], "unmapped"


def build_vlm_retranslation_plan(
    chunks_manifest: list[dict[str, Any]],
    vlm_results: dict[str, Any] | None,
    ocr_tasks: dict[str, Any] | None,
    ocr_candidate_promotion: dict[str, Any] | None,
    *,
    output_dir: Path | None = None,
    vlm_apply_report: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Map applied VLM/OCR corrections to already translated chunks.

    The plan is intentionally non-destructive: it tells the UI or a future
    executor which chunks should be retranslated after VLM results changed the
    promoted DocumentIR, but it never overwrites translated chunks.
    """
    output_dir = output_dir or Path(".")
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    results = _result_list(vlm_results)
    tasks = {_task_id(item): item for item in _task_list(ocr_tasks) if _task_id(item)}
    promotions = {_task_id(item): item for item in _promotion_list(ocr_candidate_promotion) if _task_id(item)}
    chunk_records: dict[str, dict[str, Any]] = {}
    affected_tasks: list[dict[str, Any]] = []
    unmapped_tasks: list[dict[str, Any]] = []
    mapping_counts: Counter[str] = Counter()
    structure_counts: Counter[str] = Counter()

    for result in results:
        task_id = _task_id(result)
        task = tasks.get(task_id, {})
        promotion = promotions.get(task_id, {})
        page_no = _page_no(promotion, task, result)
        block_id = _block_id(promotion, task, result)
        target_structure_type = _target_structure_type(result, promotion, task)
        if target_structure_type:
            structure_counts[target_structure_type] += 1
        matched_chunks, mapping_method = _match_chunks(
            chunks_manifest,
            block_id=block_id,
            page_no=page_no,
        )
        mapping_counts[mapping_method] += 1
        chunk_ids = [str(entry.get("chunk_id") or "").strip() for entry in matched_chunks]
        chunk_ids = [item for item in chunk_ids if item]
        reasons = _mapping_reasons(result, task, promotion)
        record = {
            "task_id": task_id,
            "page_no": page_no,
            "block_id": block_id,
            "target_structure_type": target_structure_type,
            "promotion_target": str(promotion.get("promotion_target") or ""),
            "mapping_method": mapping_method,
            "affected_chunk_ids": chunk_ids,
            "recommended_action": (
                "retranslate_affected_chunks" if chunk_ids else "manual_review_unmapped"
            ),
            "reasons": reasons,
            "confidence": result.get("confidence"),
            "engine": str(result.get("engine") or ""),
            "text_char_count": len(str(result.get("text") or "")),
            "canonical_structure_targets": _string_list(promotion.get("canonical_structure_targets")),
        }
        affected_tasks.append(record)
        if not chunk_ids:
            unmapped_tasks.append(record)

        for entry in matched_chunks:
            chunk_id = str(entry.get("chunk_id") or "").strip()
            if not chunk_id:
                continue
            chunk = chunk_records.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "pages_1based": sorted(_chunk_pages(entry)),
                    "strategy": str(entry.get("strategy") or ""),
                    "source_text_path": str(entry.get("source_text_path") or ""),
                    "translation_path": _translation_path(output_dir, chunk_id),
                    "block_ids": _entry_block_ids(entry),
                    "structure_types": _entry_block_types(entry),
                    "affected_task_ids": [],
                    "affected_block_ids": [],
                    "affected_pages_1based": [],
                    "affected_structure_types": [],
                    "reasons": [],
                    "recommended_action": "retranslate_chunk",
                },
            )
            if task_id and task_id not in chunk["affected_task_ids"]:
                chunk["affected_task_ids"].append(task_id)
            if block_id and block_id not in chunk["affected_block_ids"]:
                chunk["affected_block_ids"].append(block_id)
            if page_no > 0 and page_no not in chunk["affected_pages_1based"]:
                chunk["affected_pages_1based"].append(page_no)
            if target_structure_type and target_structure_type not in chunk["affected_structure_types"]:
                chunk["affected_structure_types"].append(target_structure_type)
            for reason in reasons:
                if reason not in chunk["reasons"]:
                    chunk["reasons"].append(reason)

    chunks = sorted(chunk_records.values(), key=lambda item: item["chunk_id"])
    for chunk in chunks:
        chunk["affected_pages_1based"] = sorted(chunk["affected_pages_1based"])
        chunk["affected_task_ids"] = sorted(chunk["affected_task_ids"])
        chunk["affected_block_ids"] = sorted(chunk["affected_block_ids"])
        chunk["affected_structure_types"] = sorted(chunk["affected_structure_types"])
        chunk["reasons"] = sorted(chunk["reasons"])

    status = "ready"
    if not chunks_manifest:
        status = "missing_chunks_manifest"
    elif not results:
        status = "no_vlm_results"
    elif chunks:
        status = "needs_retranslation"
    elif unmapped_tasks:
        status = "needs_manual_mapping"

    apply_summary = (
        vlm_apply_report.get("summary")
        if isinstance(vlm_apply_report, dict) and isinstance(vlm_apply_report.get("summary"), dict)
        else {}
    )
    structured_table_count = sum(1 for item in promotions.values() if item.get("structured_table_promoted"))
    structured_formula_count = sum(1 for item in promotions.values() if item.get("structured_formula_promoted"))
    canonical_structure_count = sum(1 for item in promotions.values() if item.get("canonical_structure_targets"))
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "source": "vlm_results_apply",
        "summary": {
            "status": status,
            "manifest_chunk_count": len(chunks_manifest),
            "vlm_result_count": len(results),
            "affected_task_count": len(affected_tasks),
            "affected_chunk_count": len(chunks),
            "retranslate_chunk_count": len(chunks),
            "unmapped_task_count": len(unmapped_tasks),
            "promoted_task_count": len(promotions),
            "canonical_structure_promotion_count": (
                _as_int(apply_summary.get("canonical_structure_promotion_count"))
                or canonical_structure_count
            ),
            "structured_table_promotion_count": (
                _as_int(apply_summary.get("structured_table_promotion_count"))
                or structured_table_count
            ),
            "structured_formula_promotion_count": (
                _as_int(apply_summary.get("structured_formula_promotion_count"))
                or structured_formula_count
            ),
            "mapping_method_counts": dict(mapping_counts),
            "target_structure_type_counts": dict(structure_counts),
            "recommended_action": (
                "retranslate_affected_chunks"
                if chunks
                else "manual_review_unmapped"
                if unmapped_tasks
                else "no_retranslation_needed"
            ),
        },
        "source_artifacts": {
            "chunks_manifest": "output/chunks_manifest.json",
            "vlm_results": "output/vlm_results.json",
            "vlm_apply": "output/vlm_apply.json",
            "ocr_tasks": "output/ocr_tasks.json",
            "ocr_candidate_promotion": "output/ocr_candidate_promotion.json",
            "document_ir_promoted": "output/document_ir_promoted.json",
        },
        "affected_tasks": affected_tasks,
        "chunks": chunks,
        "unmapped_tasks": unmapped_tasks,
    }


def _md_cell(value: Any) -> str:
    text = str(value if value is not None else "").replace("\n", " ").strip()
    return text.replace("|", "\\|") or "-"


def vlm_retranslation_plan_to_markdown(plan: dict[str, Any]) -> str:
    summary = plan.get("summary") if isinstance(plan.get("summary"), dict) else {}
    lines = [
        "# VLM Retranslation Plan",
        "",
        "## Summary",
        "",
        f"- Status: `{summary.get('status') or '-'}`",
        f"- VLM results: {summary.get('vlm_result_count', 0)}",
        f"- Affected tasks: {summary.get('affected_task_count', 0)}",
        f"- Affected chunks: {summary.get('affected_chunk_count', 0)}",
        f"- Unmapped tasks: {summary.get('unmapped_task_count', 0)}",
        f"- Recommended action: `{summary.get('recommended_action') or '-'}`",
        "",
        "## Affected Chunks",
        "",
        "| chunk | pages | tasks | blocks | structure | reasons | action |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for chunk in plan.get("chunks") or []:
        if not isinstance(chunk, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(chunk.get("chunk_id")),
                    _md_cell(", ".join(str(item) for item in chunk.get("affected_pages_1based") or [])),
                    _md_cell(", ".join(chunk.get("affected_task_ids") or [])),
                    _md_cell(", ".join(chunk.get("affected_block_ids") or [])),
                    _md_cell(", ".join(chunk.get("affected_structure_types") or [])),
                    _md_cell(", ".join(chunk.get("reasons") or [])),
                    _md_cell(chunk.get("recommended_action")),
                ]
            )
            + " |"
        )
    if not plan.get("chunks"):
        lines.append("| - | - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Affected Tasks",
            "",
            "| task | page | block | structure | mapping | chunks | action |",
            "| --- | ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for task in plan.get("affected_tasks") or []:
        if not isinstance(task, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(task.get("task_id")),
                    _md_cell(task.get("page_no")),
                    _md_cell(task.get("block_id")),
                    _md_cell(task.get("target_structure_type")),
                    _md_cell(task.get("mapping_method")),
                    _md_cell(", ".join(task.get("affected_chunk_ids") or [])),
                    _md_cell(task.get("recommended_action")),
                ]
            )
            + " |"
        )
    if not plan.get("affected_tasks"):
        lines.append("| - | - | - | - | - | - | - |")

    if plan.get("unmapped_tasks"):
        lines.extend(["", "## Unmapped Tasks", ""])
        for task in plan.get("unmapped_tasks") or []:
            if isinstance(task, dict):
                lines.append(
                    f"- `{task.get('task_id')}` page {task.get('page_no') or '-'} "
                    f"block `{task.get('block_id') or '-'}` needs manual mapping."
                )

    lines.extend(
        [
            "",
            "## Policy",
            "",
            "- This report only plans candidate retranslation. It does not overwrite translated chunks or formal output.",
            "- Block-id matches are preferred; page-range matches are used when VLM/OCR creates a new synthetic block.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_vlm_retranslation_plan(
    output_dir: Path,
    *,
    json_path: Path | None = None,
    markdown_path: Path | None = None,
    vlm_apply_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    json_path = json_path or output_dir / "vlm_retranslation_plan.json"
    markdown_path = markdown_path or output_dir / "vlm_retranslation_plan.md"
    plan = build_vlm_retranslation_plan(
        _read_json_list(output_dir / "chunks_manifest.json"),
        _read_json_dict(output_dir / "vlm_results.json", {}),
        _read_json_dict(output_dir / "ocr_tasks.json", {}),
        _read_json_dict(output_dir / "ocr_candidate_promotion.json", {}),
        output_dir=output_dir,
        vlm_apply_report=vlm_apply_report
        if isinstance(vlm_apply_report, dict)
        else _read_json_dict(output_dir / "vlm_apply.json", {}),
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(vlm_retranslation_plan_to_markdown(plan), encoding="utf-8")
    return plan
