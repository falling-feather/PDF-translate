from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import fitz

from pdf_translate.chunking import TextChunk, build_text_chunks
from pdf_translate.config import AppConfig
from pdf_translate.memory_store import MemoryStore
from pdf_translate.pdf_structure import SplitManifest, split_main_and_references
from pdf_translate.pipeline_cancel import JobCancelled, is_cancel_requested
from pdf_translate.pipeline_merge import merge_chunks_markdown
from pdf_translate.rich_content import extract_page_rich_meta
from pdf_translate.continuation_extract import translation_tail_for_next_chunk
from pdf_translate.deferral_markers import (
    parse_model_output_with_deferral,
    strip_markers_from_plain_text,
)
from pdf_translate.text_sanitize import collapse_toc_dot_leaders
from pdf_translate.translators.base import TranslationRequest
from pdf_translate.translators.factory import build_translator
from pdf_translate.translators.openai_compatible import SYSTEM_PROMPT_VERSION, prompt_fingerprint


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
    if name in ("deepl", "echo"):
        return False
    if name == "hybrid":
        return False
    return True


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
) -> tuple[TextChunk, str, str]:
    if is_cancel_requested(work_dir):
        raise JobCancelled()
    translator = build_translator(backend, cfg)
    text = collapse_toc_dot_leaders(ch.text)
    p0 = ch.pages_0based[0] + 1
    p1 = ch.pages_0based[-1] + 1
    mem = MemoryStore(work_dir / "memory")
    gloss = mem.glossary_snippet_for_pages(p0, p1)
    req = TranslationRequest(
        source_text=text,
        glossary_excerpt=gloss,
        prior_summaries="",
        style_notes=style_text,
    )
    zh = translator.translate(req)
    tname = getattr(translator, "name", type(translator).__name__)
    return ch, zh, tname


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
) -> Path:
    work_dir = work_dir.resolve()
    mem = MemoryStore(work_dir / "memory")
    mem.ensure_files()

    manifest = load_manifest(work_dir)
    main_pdf = Path(manifest.main_pdf)
    if not main_pdf.is_file():
        raise FileNotFoundError(main_pdf)

    rows = _page_rows_for_main(main_pdf)
    chunks = build_text_chunks(
        rows,
        pages_per_chunk=pages_per_chunk,
        overlap_pages=overlap_pages,
    )

    out_dir = work_dir / "output"
    chunk_dir = out_dir / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunk_manifest = [
        {
            "chunk_id": c.chunk_id,
            "pages_1based": [c.pages_0based[0] + 1, c.pages_0based[-1] + 1],
            "link_count": c.link_count,
            "image_count": c.image_count,
        }
        for c in chunks
    ]
    (out_dir / "chunks_manifest.json").write_text(
        json.dumps(chunk_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    state_path = out_dir / "state.json"
    state: dict = {"completed": [], "prompt_version": SYSTEM_PROMPT_VERSION, "prompt_hash": prompt_fingerprint()}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))

    done: set[str] = set(state.get("completed") or [])

    if progress_callback:
        progress_callback({"event": "translate_start", "chunk_total": len(chunks)})

    be = backend or cfg.default_translator
    style_text = mem.style_path.read_text(encoding="utf-8").strip()
    log_path = out_dir / "run_log.jsonl"
    merged_path = out_dir / "translated_full.md"
    merge_chunks_markdown(chunk_dir, merged_path, chunks)

    nw = max(1, int(parallel_workers))

    def _maybe_clear_carry_when_fully_done() -> None:
        st = json.loads(state_path.read_text(encoding="utf-8"))
        d = set(st.get("completed") or [])
        if translate_mode == "serial" and chunks and len(d) == len(chunks):
            mem.save_deferred_carry("")

    if translate_mode == "parallel":
        mem.save_deferred_carry("")
        pending: list[tuple[int, TextChunk]] = []
        n_sched = 0
        for idx, ch in enumerate(chunks):
            if max_chunks is not None and n_sched >= max_chunks:
                break
            if resume and ch.chunk_id in done:
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
                    ex.submit(_parallel_translate_one, work_dir, cfg, be, style_text, ch)
                    for _idx, ch in batch
                ]
                got: list[tuple[TextChunk, str, str]] = []
                for fu in as_completed(futs):
                    if is_cancel_requested(work_dir):
                        merge_chunks_markdown(chunk_dir, merged_path, chunks)
                        raise JobCancelled()
                    got.append(fu.result())
            batch_idx = {ch.chunk_id: i for i, (_i, ch) in enumerate(batch)}
            got.sort(key=lambda t: batch_idx[t[0].chunk_id])
            for ch, zh, tname in got:
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
                log_line = {
                    "chunk_id": ch.chunk_id,
                    "pages_1based": meta["pages_1based"],
                    "translator": meta["translator"],
                    "prompt_version": SYSTEM_PROMPT_VERSION,
                }
                with log_path.open("a", encoding="utf-8") as lf:
                    lf.write(json.dumps(log_line, ensure_ascii=False) + "\n")
                merge_chunks_markdown(chunk_dir, merged_path, chunks)
                chunk_index = next(i for i, c in enumerate(chunks, start=1) if c.chunk_id == ch.chunk_id)
                if progress_callback:
                    progress_callback(
                        {
                            "event": "translate_chunk_done",
                            "chunk_index": chunk_index,
                            "chunk_total": len(chunks),
                            "chunk_id": ch.chunk_id,
                        }
                    )
        merge_chunks_markdown(chunk_dir, merged_path, chunks)
        _maybe_clear_carry_when_fully_done()
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
        gloss = mem.glossary_snippet_for_pages(p0, p1)
        priors = mem.load_recent_summaries(max_chunks=3)
        text = collapse_toc_dot_leaders(ch.text)

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

        req = TranslationRequest(
            source_text=text,
            glossary_excerpt=gloss,
            prior_summaries=priors,
            style_notes=style_text,
            prior_tail_zh=prior_tail,
            continuation_hint=cont_hint,
            prior_untranslated_continuation=carry,
            defer_source_tail_protocol=use_defer,
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
        raw_zh = translator.translate(req)
        published, deferred_en = parse_model_output_with_deferral(
            raw_zh,
            use_deferral=use_defer,
        )

        body, meta = _chunk_body_and_meta(ch, published, tname)
        (chunk_dir / f"{ch.chunk_id}.md").write_text(body, encoding="utf-8")
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

        log_line = {
            "chunk_id": ch.chunk_id,
            "pages_1based": [p0, p1],
            "translator": meta["translator"],
            "prompt_version": SYSTEM_PROMPT_VERSION,
        }
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(json.dumps(log_line, ensure_ascii=False) + "\n")

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

    merge_chunks_markdown(chunk_dir, merged_path, chunks)
    _maybe_clear_carry_when_fully_done()
    return merged_path


def export_links(work_dir: Path) -> Path:
    manifest = load_manifest(work_dir)
    main_pdf = Path(manifest.main_pdf)
    meta = extract_page_rich_meta(main_pdf)
    csv_path = work_dir / "output" / "links_index.csv"
    from pdf_translate.rich_content import export_links_csv

    export_links_csv(meta, csv_path)
    return csv_path
