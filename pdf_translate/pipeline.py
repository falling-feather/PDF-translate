from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Literal

import fitz

from pdf_translate.chunkers.structure import build_structure_chunks, write_structure_manifest
from pdf_translate.chunking import TextChunk, build_text_chunks
from pdf_translate.config import AppConfig
from pdf_translate.costing import backend_model_name, load_cost_profile, write_cost_estimate
from pdf_translate.exporters.bilingual_html import write_bilingual_html
from pdf_translate.exporters.translated_pdf import write_translated_pdf
from pdf_translate.extractors.document_ir import document_ir_from_json_dict, extract_document_ir
from pdf_translate.memory_store import MemoryStore
from pdf_translate.pdf_structure import SplitManifest, split_main_and_references
from pdf_translate.pipeline_cancel import JobCancelled, is_cancel_requested
from pdf_translate.pipeline_merge import merge_chunks_markdown
from pdf_translate.qa.chunk_boundary import write_chunk_boundary_qa, write_chunk_strategy_comparison
from pdf_translate.qa.metrics import write_experiment_metrics
from pdf_translate.qa.ocr_candidates import write_ocr_candidate_qa
from pdf_translate.qa.repair import (
    write_repair_plan,
    write_repair_requests,
    write_repair_results,
    write_repair_merge,
    write_repair_publish,
    write_repair_validation,
)
from pdf_translate.qa.structure import write_structure_qa
from pdf_translate.qa.table_reconstruction import (
    build_table_translation_hints,
    write_structure_hints_manifest,
    write_table_reconstruction_report,
)
from pdf_translate.qa.translation import write_translation_qa
from pdf_translate.rich_content import extract_page_rich_meta
from pdf_translate.run_metrics import RunMetricsRecorder, elapsed_ms_since
from pdf_translate.continuation_extract import translation_tail_for_next_chunk
from pdf_translate.deferral_markers import (
    parse_model_output_with_deferral,
    strip_markers_from_plain_text,
)
from pdf_translate.survey import run_chunk_survey, survey_result_to_jsonable
from pdf_translate.text_sanitize import collapse_toc_dot_leaders
from pdf_translate.translators.base import TranslationRequest
from pdf_translate.translators.factory import build_translator
from pdf_translate.translators.http_retry import capture_http_retry_events
from pdf_translate.translators.openai_compatible import SYSTEM_PROMPT_VERSION, prompt_fingerprint
from pdf_translate.translators.registry import get_backend_spec
from pdf_translate.vision.ocr_executor import execute_ocr_tasks
from pdf_translate.vision.ocr_promotion import write_ocr_candidate_promotion
from pdf_translate.vision.ocr_tasks import write_ocr_task_manifest
from pdf_translate.vision.ocr_writeback import (
    load_ocr_results,
    write_ocr_results_payload,
    write_ocr_writeback,
)
from pdf_translate.vision.routing import write_vision_route


def init_workdir(work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    MemoryStore(work_dir / "memory").ensure_files()


def run_split(
    input_pdf: Path,
    work_dir: Path,
    *,
    ref_tail_ratio: float = 0.15,
    use_tail_if_no_heading: bool = False,
) -> SplitManifest:
    split_dir = work_dir / "split"
    return split_main_and_references(
        input_pdf,
        split_dir,
        ref_tail_ratio=ref_tail_ratio,
        use_tail_if_no_heading=use_tail_if_no_heading,
    )


def load_manifest(work_dir: Path) -> SplitManifest:
    p = work_dir / "split" / "manifest.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    return SplitManifest(
        source_pdf=raw["source_pdf"],
        total_pages=raw["total_pages"],
        reference_start_page_0based=raw.get("reference_start_page_0based"),
        main_pages_0based=raw["main_pages_0based"],
        reference_pages_0based=raw["reference_pages_0based"],
        main_pdf=raw["main_pdf"],
        references_pdf=raw.get("references_pdf"),
    )


def _page_rows_for_main(main_pdf: Path) -> list[tuple[int, str, int, int]]:
    doc = fitz.open(main_pdf)
    meta = extract_page_rich_meta(main_pdf)
    try:
        rows: list[tuple[int, str, int, int]] = []
        for i in range(len(doc)):
            text = doc[i].get_text("text")
            m = meta[i] if i < len(meta) else None
            lc = m.link_count if m else 0
            ic = m.image_count if m else 0
            rows.append((i, text, lc, ic))
        return rows
    finally:
        doc.close()


def _translator_supports_deferral(translator: object) -> bool:
    """DeepL/echo/混合管线等无法可靠执行「标识符+顺延英文」协议。"""
    name = getattr(translator, "name", "")
    try:
        return get_backend_spec(name).supports_deferral
    except ValueError:
        return False


def _ocr_results_source(out_dir: Path, explicit_path: Path | None, *, reuse_default: bool = True) -> Path | None:
    if explicit_path is not None:
        return explicit_path
    if not reuse_default:
        return None
    default_path = out_dir / "ocr_results.json"
    return default_path if default_path.is_file() else None


def _write_survey_and_merge_glossary(
    work_dir: Path,
    cfg: AppConfig,
    mem: MemoryStore,
    ch: TextChunk,
    text: str,
) -> None:
    """译前巡视：写 output/survey/<chunk_id>.json，并将 draft_terms 合并入 glossary。"""
    if not cfg.survey_enabled:
        return
    p0 = ch.pages_0based[0] + 1
    p1 = ch.pages_0based[-1] + 1
    sur = run_chunk_survey(
        cfg,
        chunk_text=text,
        chunk_id=ch.chunk_id,
        pages_1based=(p0, p1),
        image_count=ch.image_count,
        link_count=ch.link_count,
    )
    out_dir = work_dir / "output" / "survey"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = survey_result_to_jsonable(sur)
    payload["chunk_id"] = ch.chunk_id
    payload["pages_1based"] = [p0, p1]
    (out_dir / f"{ch.chunk_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not sur.skipped and sur.draft_terms:
        mem.merge_glossary_terms_from_survey(sur.draft_terms, first_page_1based=p0)


def _chunk_body_and_meta(ch: TextChunk, zh: str, translator_name: str) -> tuple[str, dict]:
    p0 = ch.pages_0based[0] + 1
    p1 = ch.pages_0based[-1] + 1
    meta = {
        "chunk_id": ch.chunk_id,
        "pages_1based": [p0, p1],
        "link_count": ch.link_count,
        "image_count": ch.image_count,
        "translator": translator_name,
        "prompt_version": SYSTEM_PROMPT_VERSION,
        "prompt_fingerprint": prompt_fingerprint(),
    }
    body = f"---\n{json.dumps(meta, ensure_ascii=False, indent=2)}\n---\n\n{zh}\n"
    return body, meta


def _parallel_translate_one(
    work_dir: Path,
    cfg: AppConfig,
    backend: str,
    style_text: str,
    ch: TextChunk,
    table_reconstruction: dict | None = None,
    structure_hints_by_chunk: dict[str, str] | None = None,
) -> tuple[TextChunk, str, str, dict]:
    if is_cancel_requested(work_dir):
        raise JobCancelled()
    translator = build_translator(backend, cfg)
    text = collapse_toc_dot_leaders(ch.text)
    p0 = ch.pages_0based[0] + 1
    p1 = ch.pages_0based[-1] + 1
    mem = MemoryStore(work_dir / "memory")
    _write_survey_and_merge_glossary(work_dir, cfg, mem, ch, text)
    gloss = mem.glossary_snippet_for_pages(p0, p1)
    structure_hints = (structure_hints_by_chunk or {}).get(ch.chunk_id)
    if structure_hints is None:
        structure_hints = build_table_translation_hints(ch, table_reconstruction)
    req = TranslationRequest(
        source_text=text,
        glossary_excerpt=gloss,
        prior_summaries="",
        style_notes=style_text,
        structure_hints=structure_hints,
    )
    translate_started = perf_counter()
    with capture_http_retry_events() as http_retry_events:
        zh = translator.translate(req)
    translate_elapsed_ms = elapsed_ms_since(translate_started)
    tname = getattr(translator, "name", type(translator).__name__)
    context_char_count = len(gloss) + len(style_text) + len(structure_hints)
    return (
        ch,
        zh,
        tname,
        {
            "elapsed_ms": translate_elapsed_ms,
            "source_char_count": len(text),
            "context_char_count": context_char_count,
            "request_char_count": len(text) + context_char_count,
            "translated_char_count": len(zh),
            "raw_translated_char_count": len(zh),
            "deferred_char_count": 0,
            "http_retry_events": list(http_retry_events),
        },
    )


def run_translate(
    work_dir: Path,
    cfg: AppConfig,
    *,
    backend: str | None = None,
    pages_per_chunk: int = 3,
    overlap_pages: int = 1,
    resume: bool = True,
    max_chunks: int | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    translate_mode: Literal["serial", "parallel"] = "serial",
    parallel_workers: int = 4,
    survey_override: bool | None = None,
    chunk_strategy: Literal["page", "structure"] = "page",
    execute_repair_requests: bool = False,
    max_repair_requests: int | None = None,
    publish_repairs: bool = False,
    ocr_results_path: Path | None = None,
    execute_ocr: bool = False,
    ocr_engine: str = "tesseract_cli",
    ocr_language: str = "eng",
    ocr_timeout_seconds: int = 30,
    ocr_command: str | None = None,
) -> Path:
    """survey_override：None 使用 cfg.survey_enabled；True/False 强制开关译前巡视（精品翻译传 True）。"""
    cfg = replace(cfg, survey_enabled=survey_override) if survey_override is not None else cfg
    work_dir = work_dir.resolve()
    mem = MemoryStore(work_dir / "memory")
    mem.ensure_files()

    manifest = load_manifest(work_dir)
    main_pdf = Path(manifest.main_pdf)
    if not main_pdf.is_file():
        raise FileNotFoundError(main_pdf)

    out_dir = work_dir / "output"
    chunk_dir = out_dir / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    be = backend or cfg.default_translator
    nw = max(1, int(parallel_workers))
    log_path = out_dir / "run_log.jsonl"
    run_metrics = RunMetricsRecorder(log_path)
    run_metrics.record(
        "run_start",
        "run",
        backend=be,
        pipeline_variant=chunk_strategy,
        translate_mode=translate_mode,
        parallel_workers=nw,
    )

    with run_metrics.stage("document_ir"):
        doc_ir = extract_document_ir(main_pdf)
        doc_ir.write_json(out_dir / "document_ir.json")
    with run_metrics.stage("structure_qa"):
        structure_qa = write_structure_qa(doc_ir, out_dir / "structure_qa.json")
    with run_metrics.stage("table_reconstruction"):
        table_reconstruction = write_table_reconstruction_report(
            doc_ir,
            structure_qa,
            out_dir / "table_reconstruction.json",
        )
    with run_metrics.stage("vision_route"):
        vision_route = write_vision_route(doc_ir, out_dir / "vision_route.json")
    with run_metrics.stage("ocr_tasks"):
        ocr_tasks = write_ocr_task_manifest(doc_ir, vision_route, out_dir / "ocr_tasks.json")
    with run_metrics.stage("ocr_results"):
        resolved_ocr_results_path = _ocr_results_source(out_dir, ocr_results_path, reuse_default=not execute_ocr)
        if resolved_ocr_results_path:
            raw_ocr_results = load_ocr_results(resolved_ocr_results_path)
        elif execute_ocr:
            raw_ocr_results = execute_ocr_tasks(
                ocr_tasks,
                work_dir,
                engine=ocr_engine,
                language=ocr_language,
                timeout_seconds=ocr_timeout_seconds,
                command=ocr_command,
            )
        else:
            raw_ocr_results = None
        ocr_results = write_ocr_results_payload(
            ocr_tasks,
            out_dir / "ocr_results.json",
            raw_ocr_results,
            source_path=resolved_ocr_results_path,
        )
    with run_metrics.stage("ocr_writeback"):
        ocr_writeback = write_ocr_writeback(
            doc_ir,
            ocr_tasks,
            out_dir / "ocr_writeback.json",
            out_dir / "document_ir_ocr.json",
            ocr_results,
        )
    with run_metrics.stage("ocr_candidate_qa"):
        document_ir_ocr = json.loads((out_dir / "document_ir_ocr.json").read_text(encoding="utf-8"))
        ocr_candidate_qa = write_ocr_candidate_qa(
            document_ir_ocr,
            ocr_writeback,
            out_dir / "ocr_candidate_qa.json",
            out_dir / "ocr_candidate_qa.md",
        )
    with run_metrics.stage("ocr_candidate_promotion"):
        ocr_candidate_promotion = write_ocr_candidate_promotion(
            document_ir_ocr,
            ocr_candidate_qa,
            out_dir / "ocr_candidate_promotion.json",
            out_dir / "ocr_candidate_promotion.md",
            out_dir / "document_ir_promoted.json",
        )
        promoted_document_ir = document_ir_from_json_dict(
            json.loads((out_dir / "document_ir_promoted.json").read_text(encoding="utf-8"))
        )
    with run_metrics.stage("structure_chunking"):
        structure_chunks = build_structure_chunks(
            promoted_document_ir,
            max_pages_per_chunk=pages_per_chunk,
        )
        write_structure_manifest(structure_chunks, out_dir / "structure_chunks_manifest.json")
    with run_metrics.stage("page_chunking"):
        page_rows = _page_rows_for_main(main_pdf)
        page_chunks = build_text_chunks(
            page_rows,
            pages_per_chunk=pages_per_chunk,
            overlap_pages=overlap_pages,
        )

    if chunk_strategy == "structure":
        chunks = structure_chunks
    elif chunk_strategy == "page":
        chunks = page_chunks
    else:
        raise ValueError("chunk_strategy must be 'page' or 'structure'")

    with run_metrics.stage("chunk_manifest"):
        chunk_manifest = [
            {
                "chunk_id": c.chunk_id,
                "pages_1based": [c.pages_0based[0] + 1, c.pages_0based[-1] + 1],
                "link_count": c.link_count,
                "image_count": c.image_count,
                "strategy": chunk_strategy,
                "block_ids": getattr(c, "block_ids", []),
                "block_types": getattr(c, "block_types", {}),
                "warnings": getattr(c, "warnings", []),
                "boundary_fragment_ids": getattr(c, "boundary_fragment_ids", []),
                "structural_relation_ids": getattr(c, "structural_relation_ids", []),
                "approx_tokens": getattr(c, "approx_tokens", 0),
                "budget": {
                    "target_chars": getattr(c, "budget_target_chars", 0),
                    "max_chars": getattr(c, "budget_max_chars", 0),
                    "split_reason": getattr(c, "split_reason", ""),
                    "overflow_chars": getattr(c, "budget_overflow_chars", 0),
                    "pressure": getattr(c, "budget_pressure", ""),
                },
            }
            for c in chunks
        ]
        (out_dir / "chunks_manifest.json").write_text(
            json.dumps(chunk_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    with run_metrics.stage("chunk_boundary_qa"):
        chunk_boundary_qa = write_chunk_boundary_qa(
            chunks,
            structure_qa,
            out_dir / "chunk_boundary_qa.json",
            pipeline_variant=chunk_strategy,
        )
    with run_metrics.stage("chunk_strategy_comparison"):
        chunk_strategy_comparison = write_chunk_strategy_comparison(
            {
                "page": page_chunks,
                "structure": structure_chunks,
            },
            structure_qa,
            out_dir / "chunk_strategy_comparison.json",
            active_strategy=chunk_strategy,
        )
    with run_metrics.stage("structure_hints_manifest"):
        structure_hints_manifest = write_structure_hints_manifest(
            chunks,
            table_reconstruction,
            out_dir / "structure_hints_manifest.json",
        )
    structure_hints_by_chunk = {
        str(entry.get("chunk_id")): str(entry.get("hint_text") or "")
        for entry in structure_hints_manifest.get("chunks", [])
        if isinstance(entry, dict) and str(entry.get("chunk_id") or "")
    }

    state_path = out_dir / "state.json"
    state: dict = {"completed": [], "prompt_version": SYSTEM_PROMPT_VERSION, "prompt_hash": prompt_fingerprint()}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))

    done: set[str] = set(state.get("completed") or [])

    if progress_callback:
        progress_callback({"event": "translate_start", "chunk_total": len(chunks)})

    style_text = mem.style_path.read_text(encoding="utf-8").strip()
    merged_path = out_dir / "translated_full.md"
    with run_metrics.stage("initial_merge"):
        merge_chunks_markdown(chunk_dir, merged_path, chunks)

    def _maybe_clear_carry_when_fully_done() -> None:
        st = json.loads(state_path.read_text(encoding="utf-8"))
        d = set(st.get("completed") or [])
        if translate_mode == "serial" and chunks and len(d) == len(chunks):
            mem.save_deferred_carry("")

    def _write_translation_qa_report() -> None:
        with run_metrics.stage("translation_qa"):
            qa_report = write_translation_qa(
                chunks,
                chunk_dir,
                out_dir / "qa_report.json",
                out_dir / "qa_report.md",
                glossary=mem.load_glossary(),
                pending_review=mem.load_pending_review(),
                document_ir=doc_ir,
                table_reconstruction=table_reconstruction,
            )
        with run_metrics.stage("repair_plan"):
            repair_plan = write_repair_plan(
                qa_report,
                out_dir / "repair_plan.json",
                out_dir / "repair_plan.md",
            )
        with run_metrics.stage("repair_requests"):
            repair_requests = write_repair_requests(
                repair_plan,
                chunks,
                chunk_dir,
                out_dir / "repair_requests.json",
                out_dir / "repair_requests.md",
            )
        repair_translator = build_translator(be, cfg) if execute_repair_requests else None
        with run_metrics.stage("repair_results", execute=execute_repair_requests):
            repair_results = write_repair_results(
                repair_requests,
                out_dir / "repair_results.json",
                out_dir / "repair_results.md",
                translator=repair_translator,
                execute=execute_repair_requests,
                max_requests=max_repair_requests,
            )
        with run_metrics.stage("repair_validation"):
            repair_validation = write_repair_validation(
                repair_requests,
                repair_results,
                out_dir / "repair_validation.json",
                out_dir / "repair_validation.md",
            )
        with run_metrics.stage("repair_merge"):
            repair_merge = write_repair_merge(
                repair_requests,
                repair_results,
                repair_validation,
                chunks,
                chunk_dir,
                out_dir / "repair_merge.json",
                out_dir / "repair_merge.md",
                repaired_chunk_dir=out_dir / "repaired_chunks",
                repaired_full_path=out_dir / "repaired_full.md",
            )
        with run_metrics.stage("repair_merge_qa"):
            repair_merge_qa = write_translation_qa(
                chunks,
                out_dir / "repaired_chunks",
                out_dir / "repair_merge_qa.json",
                out_dir / "repair_merge_qa.md",
                glossary=mem.load_glossary(),
                pending_review=mem.load_pending_review(),
                document_ir=doc_ir,
                table_reconstruction=table_reconstruction,
            )
        with run_metrics.stage("repair_publish", confirmed=publish_repairs):
            repair_publish = write_repair_publish(
                repair_merge,
                out_dir / "repair_publish.json",
                out_dir / "repair_publish.md",
                confirm=publish_repairs,
                source_full_path=out_dir / "repaired_full.md",
                published_full_path=out_dir / "published_full.md",
                original_full_path=out_dir / "translated_full.md",
            )
        with run_metrics.stage("bilingual_html"):
            write_bilingual_html(
                chunks,
                chunk_dir,
                out_dir / "bilingual.html",
                qa_report=qa_report,
                repair_plan=repair_plan,
                title=f"{main_pdf.stem} 双语对照译文",
            )
        with run_metrics.stage("translated_pdf"):
            translated_pdf_report = write_translated_pdf(
                chunks,
                chunk_dir,
                out_dir / "translated_full.pdf",
                qa_report=qa_report,
                repair_plan=repair_plan,
                structure_qa=structure_qa,
                table_reconstruction=table_reconstruction,
                title=f"{main_pdf.stem} 结构化译文",
                source_pdf=main_pdf,
                report_path=out_dir / "translated_pdf_report.json",
            )
        run_metrics_summary = run_metrics.write_summary(
            out_dir / "run_metrics.json",
            doc_id=doc_ir.doc_id,
            pipeline_variant=chunk_strategy,
            backend=be,
            translate_mode=translate_mode,
            parallel_workers=nw,
            page_count=len(doc_ir.pages),
            chunk_count=len(chunks),
            completed_chunk_count=len(done),
        )
        cost_estimate = write_cost_estimate(
            out_dir / "cost_estimate.json",
            run_metrics_summary,
            load_cost_profile(cfg),
            backend=be,
            model=backend_model_name(be, cfg),
        )
        write_experiment_metrics(
            structure_qa,
            vision_route,
            qa_report,
            repair_plan,
            out_dir / "experiment_metrics.json",
            doc_id=doc_ir.doc_id,
            pipeline_variant=chunk_strategy,
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
            repair_merge_qa=repair_merge_qa,
            repair_publish=repair_publish,
            translated_pdf_report=translated_pdf_report,
            run_metrics=run_metrics_summary,
            cost_estimate=cost_estimate,
        )

    if translate_mode == "parallel":
        mem.save_deferred_carry("")
        pending: list[tuple[int, TextChunk]] = []
        n_sched = 0
        for idx, ch in enumerate(chunks):
            if max_chunks is not None and n_sched >= max_chunks:
                break
            if resume and ch.chunk_id in done:
                run_metrics.record_chunk_skipped(
                    chunk_id=ch.chunk_id,
                    pages_1based=[ch.pages_0based[0] + 1, ch.pages_0based[-1] + 1],
                    reason="resume_completed",
                    chunk_index=idx + 1,
                    chunk_total=len(chunks),
                    mode=translate_mode,
                )
                continue
            pending.append((idx, ch))
            n_sched += 1

        pos = 0
        while pos < len(pending):
            if is_cancel_requested(work_dir):
                merge_chunks_markdown(chunk_dir, merged_path, chunks)
                raise JobCancelled()
            batch = pending[pos : pos + nw]
            pos += len(batch)
            if progress_callback:
                for _idx, ch in batch:
                    ci = next(i for i, c in enumerate(chunks, start=1) if c.chunk_id == ch.chunk_id)
                    progress_callback(
                        {
                            "event": "translate_chunk_start",
                            "chunk_index": ci,
                            "chunk_total": len(chunks),
                            "chunk_id": ch.chunk_id,
                            "approx_chars": len(collapse_toc_dot_leaders(ch.text)),
                        }
                    )
            with ThreadPoolExecutor(max_workers=len(batch)) as ex:
                futs = [
                    ex.submit(
                        _parallel_translate_one,
                        work_dir,
                        cfg,
                        be,
                        style_text,
                        ch,
                        table_reconstruction,
                        structure_hints_by_chunk,
                    )
                    for _idx, ch in batch
                ]
                got: list[tuple[TextChunk, str, str, dict]] = []
                for fu in as_completed(futs):
                    if is_cancel_requested(work_dir):
                        merge_chunks_markdown(chunk_dir, merged_path, chunks)
                        raise JobCancelled()
                    got.append(fu.result())
            batch_idx = {ch.chunk_id: i for i, (_i, ch) in enumerate(batch)}
            got.sort(key=lambda t: batch_idx[t[0].chunk_id])
            for ch, zh, tname, translation_metrics in got:
                body, meta = _chunk_body_and_meta(ch, zh, tname)
                (chunk_dir / f"{ch.chunk_id}.md").write_text(body, encoding="utf-8")
                summary = zh.strip().replace("\n", " ")[:400]
                mem.append_chunk_summary(
                    ch.chunk_id,
                    (meta["pages_1based"][0], meta["pages_1based"][1]),
                    summary,
                    tail_zh=translation_tail_for_next_chunk(zh),
                )
                done.add(ch.chunk_id)
                state["completed"] = sorted(done)
                state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
                chunk_index = next(i for i, c in enumerate(chunks, start=1) if c.chunk_id == ch.chunk_id)
                run_metrics.record_chunk_translation(
                    chunk_id=ch.chunk_id,
                    pages_1based=meta["pages_1based"],
                    translator=meta["translator"],
                    elapsed_ms=translation_metrics["elapsed_ms"],
                    source_char_count=translation_metrics["source_char_count"],
                    context_char_count=translation_metrics["context_char_count"],
                    request_char_count=translation_metrics["request_char_count"],
                    translated_char_count=translation_metrics["translated_char_count"],
                    raw_translated_char_count=translation_metrics["raw_translated_char_count"],
                    deferred_char_count=translation_metrics["deferred_char_count"],
                    http_retry_events=translation_metrics.get("http_retry_events"),
                    chunk_index=chunk_index,
                    chunk_total=len(chunks),
                    prompt_version=SYSTEM_PROMPT_VERSION,
                    prompt_fingerprint=prompt_fingerprint(),
                    mode=translate_mode,
                )
                with run_metrics.stage("incremental_merge", chunk_id=ch.chunk_id):
                    merge_chunks_markdown(chunk_dir, merged_path, chunks)
                if progress_callback:
                    progress_callback(
                        {
                            "event": "translate_chunk_done",
                            "chunk_index": chunk_index,
                            "chunk_total": len(chunks),
                            "chunk_id": ch.chunk_id,
                        }
                    )
        with run_metrics.stage("final_merge"):
            merge_chunks_markdown(chunk_dir, merged_path, chunks)
        _maybe_clear_carry_when_fully_done()
        _write_translation_qa_report()
        return merged_path

    # —— 串联：按顺序调用模型，注入前文摘要；每块写盘后全量重建合并稿 ——
    translator = build_translator(be, cfg)
    tname = getattr(translator, "name", type(translator).__name__)
    defer_protocol = _translator_supports_deferral(translator)
    if not defer_protocol:
        mem.save_deferred_carry("")
    n_done = 0
    for idx, ch in enumerate(chunks):
        if max_chunks is not None and n_done >= max_chunks:
            break
        if is_cancel_requested(work_dir):
            merge_chunks_markdown(chunk_dir, merged_path, chunks)
            raise JobCancelled()
        if resume and ch.chunk_id in done:
            run_metrics.record_chunk_skipped(
                chunk_id=ch.chunk_id,
                pages_1based=[ch.pages_0based[0] + 1, ch.pages_0based[-1] + 1],
                reason="resume_completed",
                chunk_index=idx + 1,
                chunk_total=len(chunks),
                mode=translate_mode,
            )
            if progress_callback:
                progress_callback(
                    {
                        "event": "translate_chunk_skipped",
                        "chunk_index": idx + 1,
                        "chunk_total": len(chunks),
                        "chunk_id": ch.chunk_id,
                    }
                )
            continue

        p0 = ch.pages_0based[0] + 1
        p1 = ch.pages_0based[-1] + 1
        text = collapse_toc_dot_leaders(ch.text)
        _write_survey_and_merge_glossary(work_dir, cfg, mem, ch, text)
        gloss = mem.glossary_snippet_for_pages(p0, p1)
        priors = mem.load_recent_summaries(max_chunks=3)

        prior_tail = mem.load_prior_tail_zh()
        carry = mem.load_deferred_carry() if defer_protocol else ""
        cont_hint = ""
        if overlap_pages > 0 and idx > 0:
            cont_hint = (
                "本块与上一块在原文上存在页级重叠；请勿输出与上一块译文等价的重复句，"
                "用代词或简略承接即可，重点译出本块新增的论述。"
            )

        is_doc_last = idx == len(chunks) - 1
        use_defer = defer_protocol and not is_doc_last
        structure_hints = structure_hints_by_chunk.get(ch.chunk_id)
        if structure_hints is None:
            structure_hints = build_table_translation_hints(ch, table_reconstruction)

        req = TranslationRequest(
            source_text=text,
            glossary_excerpt=gloss,
            prior_summaries=priors,
            style_notes=style_text,
            structure_hints=structure_hints,
            prior_tail_zh=prior_tail,
            continuation_hint=cont_hint,
            prior_untranslated_continuation=carry,
            defer_source_tail_protocol=use_defer,
        )
        context_char_count = (
            len(gloss)
            + len(priors)
            + len(style_text)
            + len(structure_hints)
            + len(prior_tail)
            + len(cont_hint)
            + len(carry)
        )
        approx_n = len(text) + len(carry)
        if progress_callback:
            progress_callback(
                {
                    "event": "translate_chunk_start",
                    "chunk_index": idx + 1,
                    "chunk_total": len(chunks),
                    "chunk_id": ch.chunk_id,
                    "approx_chars": approx_n,
                }
            )
        translate_started = perf_counter()
        with capture_http_retry_events() as http_retry_events:
            raw_zh = translator.translate(req)
        translate_elapsed_ms = elapsed_ms_since(translate_started)
        published, deferred_en = parse_model_output_with_deferral(
            raw_zh,
            use_deferral=use_defer,
        )

        body, meta = _chunk_body_and_meta(ch, published, tname)
        (chunk_dir / f"{ch.chunk_id}.md").write_text(body, encoding="utf-8")
        with run_metrics.stage("incremental_merge", chunk_id=ch.chunk_id):
            merge_chunks_markdown(chunk_dir, merged_path, chunks)

        plain_for_mem = strip_markers_from_plain_text(published)
        summary = plain_for_mem.replace("\n", " ")[:400]
        tail_for_next = translation_tail_for_next_chunk(plain_for_mem)
        mem.append_chunk_summary(ch.chunk_id, (p0, p1), summary, tail_zh=tail_for_next)

        if defer_protocol:
            if is_doc_last:
                mem.save_deferred_carry("")
            else:
                mem.save_deferred_carry(deferred_en)

        done.add(ch.chunk_id)
        state["completed"] = sorted(done)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

        run_metrics.record_chunk_translation(
            chunk_id=ch.chunk_id,
            pages_1based=[p0, p1],
            translator=meta["translator"],
            elapsed_ms=translate_elapsed_ms,
            source_char_count=len(text),
            context_char_count=context_char_count,
            request_char_count=len(text) + context_char_count,
            translated_char_count=len(published),
            raw_translated_char_count=len(raw_zh),
            deferred_char_count=len(deferred_en),
            http_retry_events=list(http_retry_events),
            chunk_index=idx + 1,
            chunk_total=len(chunks),
            prompt_version=SYSTEM_PROMPT_VERSION,
            prompt_fingerprint=prompt_fingerprint(),
            mode=translate_mode,
        )

        if progress_callback:
            progress_callback(
                {
                    "event": "translate_chunk_done",
                    "chunk_index": idx + 1,
                    "chunk_total": len(chunks),
                    "chunk_id": ch.chunk_id,
                }
            )

        n_done += 1

    with run_metrics.stage("final_merge"):
        merge_chunks_markdown(chunk_dir, merged_path, chunks)
    _maybe_clear_carry_when_fully_done()
    _write_translation_qa_report()
    return merged_path


def export_links(work_dir: Path) -> Path:
    manifest = load_manifest(work_dir)
    main_pdf = Path(manifest.main_pdf)
    meta = extract_page_rich_meta(main_pdf)
    csv_path = work_dir / "output" / "links_index.csv"
    from pdf_translate.rich_content import export_links_csv

    export_links_csv(meta, csv_path)
    return csv_path
