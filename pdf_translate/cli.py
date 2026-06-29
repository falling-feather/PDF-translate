from __future__ import annotations

from pathlib import Path

import typer

from pdf_translate.config import AppConfig
from pdf_translate import pipeline
from pdf_translate.experiments import load_sample_metadata, parse_variant_specs, run_batch_experiment

app = typer.Typer(help="PDF 英文学术文献：拆分参考文献、按块翻译、记忆目录（见 README.md memory/ 说明）")


@app.command("init")
def cmd_init(
    work_dir: Path = typer.Argument(..., help="项目工作目录，将创建 memory/ 等"),
) -> None:
    pipeline.init_workdir(work_dir)
    typer.echo(f"已初始化工作目录: {work_dir.resolve()}")


@app.command("split")
def cmd_split(
    pdf: Path = typer.Argument(..., exists=True, help="原始 PDF"),
    work_dir: Path = typer.Argument(..., help="工作目录（需先 init）"),
    tail_fallback: bool = typer.Option(
        False,
        "--tail-fallback",
        help="未检测到 References 标题时，将最后约 15%% 页作为参考文献 PDF",
    ),
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    mem = work_dir / "memory"
    if not mem.is_dir():
        typer.echo("memory/ 不存在，正在创建…")
        pipeline.init_workdir(work_dir)
    m = pipeline.run_split(pdf, work_dir, use_tail_if_no_heading=tail_fallback)
    typer.echo(f"正文 PDF: {m.main_pdf}")
    if m.references_pdf:
        typer.echo(f"参考文献 PDF: {m.references_pdf}")
    else:
        typer.echo("未拆分出参考文献（全文视作正文）")


@app.command("translate")
def cmd_translate(
    work_dir: Path = typer.Argument(..., help="已完成 split 的工作目录"),
    backend: str = typer.Option(
        None,
        "--backend",
        "-b",
        help="echo | openai | ollama | deepl | hybrid；默认环境变量 PDF_TRANSLATE_BACKEND",
    ),
    pages_per_chunk: int = typer.Option(3, "--pages", min=1, max=3, help="每块页数 1–3"),
    overlap: int = typer.Option(1, "--overlap", min=0, help="块间重叠页数"),
    no_resume: bool = typer.Option(False, "--no-resume", help="忽略断点，重译所有块"),
    max_chunks: int = typer.Option(None, "--max-chunks", help="仅处理前 N 块（调试）"),
    execute_repairs: bool = typer.Option(
        False,
        "--execute-repairs",
        help="执行局部修复请求并写出 repair_results（默认只生成请求，不改写译文）",
    ),
    max_repair_requests: int = typer.Option(
        None,
        "--max-repair-requests",
        min=1,
        help="最多执行 N 条局部修复请求；仅 --execute-repairs 生效",
    ),
    chunk_strategy: str = typer.Option(
        "page",
        "--chunk-strategy",
        help="page | structure；structure 使用 DocumentIR 结构块分段（实验性）",
    ),
    ocr_results: Path | None = typer.Option(
        None,
        "--ocr-results",
        exists=True,
        dir_okay=False,
        help="OCR results JSON (ocr-results-v1) for output/document_ir_ocr.json.",
    ),
    execute_ocr: bool = typer.Option(
        False,
        "--execute-ocr",
        help="Run local OCR tasks when --ocr-results is not provided.",
    ),
    ocr_engine: str = typer.Option("tesseract_cli", "--ocr-engine", help="Local OCR engine id."),
    ocr_language: str = typer.Option("eng", "--ocr-language", help="OCR language code passed to the engine."),
    ocr_timeout_seconds: int = typer.Option(30, "--ocr-timeout", min=1, help="Seconds per OCR task."),
    ocr_command: str | None = typer.Option(None, "--ocr-command", help="Optional OCR command path."),
) -> None:
    cfg = AppConfig.from_env()
    out = pipeline.run_translate(
        work_dir,
        cfg,
        backend=backend,
        pages_per_chunk=pages_per_chunk,
        overlap_pages=overlap,
        resume=not no_resume,
        max_chunks=max_chunks,
        chunk_strategy=chunk_strategy,  # type: ignore[arg-type]
        execute_repair_requests=execute_repairs,
        max_repair_requests=max_repair_requests,
        ocr_results_path=ocr_results,
        execute_ocr=execute_ocr,
        ocr_engine=ocr_engine,
        ocr_language=ocr_language,
        ocr_timeout_seconds=ocr_timeout_seconds,
        ocr_command=ocr_command,
    )
    typer.echo(f"已写入合并稿: {out}")


@app.command("links")
def cmd_links(
    work_dir: Path = typer.Argument(..., help="已完成 split 的工作目录"),
) -> None:
    p = pipeline.export_links(work_dir)
    typer.echo(f"链接索引: {p}")


@app.command("run")
def cmd_run(
    pdf: Path = typer.Argument(..., exists=True),
    work_dir: Path = typer.Argument(..., help="工作目录"),
    backend: str = typer.Option(None, "--backend", "-b"),
    tail_fallback: bool = typer.Option(False, "--tail-fallback"),
    pages_per_chunk: int = typer.Option(3, "--pages", min=1, max=3),
    overlap: int = typer.Option(1, "--overlap", min=0),
    max_chunks: int = typer.Option(None, "--max-chunks"),
    execute_repairs: bool = typer.Option(
        False,
        "--execute-repairs",
        help="执行局部修复请求并写出 repair_results（默认只生成请求，不改写译文）",
    ),
    max_repair_requests: int = typer.Option(
        None,
        "--max-repair-requests",
        min=1,
        help="最多执行 N 条局部修复请求；仅 --execute-repairs 生效",
    ),
    chunk_strategy: str = typer.Option(
        "page",
        "--chunk-strategy",
        help="page | structure；structure 使用 DocumentIR 结构块分段（实验性）",
    ),
    ocr_results: Path | None = typer.Option(
        None,
        "--ocr-results",
        exists=True,
        dir_okay=False,
        help="OCR results JSON (ocr-results-v1) for output/document_ir_ocr.json.",
    ),
    execute_ocr: bool = typer.Option(
        False,
        "--execute-ocr",
        help="Run local OCR tasks when --ocr-results is not provided.",
    ),
    ocr_engine: str = typer.Option("tesseract_cli", "--ocr-engine", help="Local OCR engine id."),
    ocr_language: str = typer.Option("eng", "--ocr-language", help="OCR language code passed to the engine."),
    ocr_timeout_seconds: int = typer.Option(30, "--ocr-timeout", min=1, help="Seconds per OCR task."),
    ocr_command: str | None = typer.Option(None, "--ocr-command", help="Optional OCR command path."),
) -> None:
    """init → split → translate 一键执行。"""
    pipeline.init_workdir(work_dir)
    pipeline.run_split(pdf, work_dir, use_tail_if_no_heading=tail_fallback)
    cfg = AppConfig.from_env()
    out = pipeline.run_translate(
        work_dir,
        cfg,
        backend=backend,
        pages_per_chunk=pages_per_chunk,
        overlap_pages=overlap,
        resume=True,
        max_chunks=max_chunks,
        chunk_strategy=chunk_strategy,  # type: ignore[arg-type]
        execute_repair_requests=execute_repairs,
        max_repair_requests=max_repair_requests,
        ocr_results_path=ocr_results,
        execute_ocr=execute_ocr,
        ocr_engine=ocr_engine,
        ocr_language=ocr_language,
        ocr_timeout_seconds=ocr_timeout_seconds,
        ocr_command=ocr_command,
    )
    typer.echo(f"完成: {out}")


@app.command("experiment")
def cmd_experiment(
    pdfs: list[Path] = typer.Argument(..., exists=True, dir_okay=False, help="用于批量实验的一篇或多篇 PDF"),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="批量实验输出目录"),
    backend: str = typer.Option("echo", "--backend", "-b", help="实验后端；专利指标预跑建议先用 echo"),
    variants: str = typer.Option(
        "page,structure",
        "--variants",
        help="逗号分隔策略：page, structure, structure+ocr, structure+repair",
    ),
    sample_manifest: Path | None = typer.Option(
        None,
        "--sample-manifest",
        exists=True,
        dir_okay=False,
        help="可选样本元数据 JSON/CSV/TSV，字段可含 source_pdf/sample_id/pdf_type/tags/notes",
    ),
    tail_fallback: bool = typer.Option(False, "--tail-fallback", help="未检测到参考文献标题时启用尾部兜底"),
    pages_per_chunk: int = typer.Option(3, "--pages", min=1, max=3, help="每块页数 1-3"),
    overlap: int = typer.Option(1, "--overlap", min=0, help="固定页分块的重叠页数"),
    max_chunks: int | None = typer.Option(None, "--max-chunks", min=1, help="每次运行最多翻译 N 块（调试/预跑）"),
    translate_mode: str = typer.Option("serial", "--translate-mode", help="serial | parallel"),
    parallel_workers: int = typer.Option(4, "--parallel-workers", min=1, help="并行模式 worker 数"),
    resume: bool = typer.Option(False, "--resume", help="复用已有完成块；默认每次重跑当前策略"),
    stop_on_error: bool = typer.Option(False, "--stop-on-error", help="任一运行失败时立即停止"),
) -> None:
    if translate_mode not in ("serial", "parallel"):
        raise typer.BadParameter("translate-mode must be serial or parallel")
    cfg = AppConfig.from_env()
    report = run_batch_experiment(
        pdfs,
        output_dir,
        cfg,
        variants=parse_variant_specs(variants),
        backend=backend,
        pages_per_chunk=pages_per_chunk,
        overlap_pages=overlap,
        max_chunks=max_chunks,
        tail_fallback=tail_fallback,
        translate_mode=translate_mode,  # type: ignore[arg-type]
        parallel_workers=parallel_workers,
        resume=resume,
        stop_on_error=stop_on_error,
        sample_metadata=load_sample_metadata(sample_manifest) if sample_manifest else None,
    )
    typer.echo(f"批量实验汇总: {(output_dir / 'batch_experiment_summary.json').resolve()}")
    typer.echo(f"Markdown 报告: {(output_dir / 'batch_experiment_summary.md').resolve()}")
    typer.echo(f"人工评分表: {(output_dir / 'batch_experiment_review.csv').resolve()}")
    typer.echo(f"成功/总数: {report['succeeded_count']}/{report['run_count']}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
