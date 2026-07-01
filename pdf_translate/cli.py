from __future__ import annotations

import json
import os
from pathlib import Path

import typer

from pdf_translate.config import AppConfig
from pdf_translate import pipeline
from pdf_translate.experiments import (
    load_sample_metadata,
    parse_variant_specs,
    run_batch_experiment,
    write_batch_experiment_evidence,
    write_sample_manifest,
)
from pdf_translate.server.jobs import JobRegistry
from pdf_translate.server.security_preflight import build_security_preflight
from pdf_translate.translators.registry import backend_choice_text

app = typer.Typer(help="PDF 英文学术文献：拆分参考文献、按块翻译、记忆目录（见 README.md memory/ 说明）")


def _default_data_base() -> Path:
    return Path(os.getenv("PDF_TRANSLATE_DATA", Path.cwd() / "data")).resolve()


def _default_web_data_root() -> Path:
    data_base = _default_data_base()
    return Path(os.getenv("PDF_TRANSLATE_WEB_DATA", data_base / "web_jobs")).resolve()


def _expand_pdf_sources(sources: list[Path], *, recursive: bool = False) -> list[Path]:
    pdfs: list[Path] = []
    seen: set[str] = set()
    for source in sources:
        paths = source.rglob("*.pdf") if source.is_dir() and recursive else source.glob("*.pdf") if source.is_dir() else [source]
        for path in paths:
            if path.is_file() and path.suffix.lower() == ".pdf":
                key = str(path.resolve()).lower()
                if key not in seen:
                    pdfs.append(path)
                    seen.add(key)
    return sorted(pdfs, key=lambda item: str(item).lower())


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
        help=f"{backend_choice_text()}；默认环境变量 PDF_TRANSLATE_BACKEND",
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


@app.command("web-status")
def cmd_web_status(
    job_id: str | None = typer.Option(None, "--job-id", "-j", help="只查看单个 Web 任务"),
    data_root: Path | None = typer.Option(
        None,
        "--data-root",
        help="Web 任务目录；默认读取 PDF_TRANSLATE_WEB_DATA 或 data/web_jobs",
    ),
    limit: int = typer.Option(20, "--limit", min=1, help="未指定 job-id 时最多输出任务数"),
) -> None:
    root = (data_root or _default_web_data_root()).resolve()
    registry = JobRegistry(root)
    registry.hydrate_from_disk()
    if job_id:
        rec = registry.get(job_id)
        if not rec:
            typer.echo(f"未找到 Web 任务: {job_id}", err=True)
            raise typer.Exit(1)
        payload = {
            "data_root": str(root),
            "hydration": registry.hydration_report(),
            "job": registry.diagnostic_summary_for_record(rec),
        }
    else:
        records = registry.list_records()
        payload = {
            "data_root": str(root),
            "hydration": registry.hydration_report(),
            "job_count": len(records),
            "jobs": [
                registry.diagnostic_summary_for_record(rec)
                for rec in records[:limit]
            ],
        }
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("security-check")
def cmd_security_check(
    data_dir: Path | None = typer.Option(
        None,
        "--data-dir",
        help="数据目录；默认读取 PDF_TRANSLATE_DATA 或 data",
    ),
    data_root: Path | None = typer.Option(
        None,
        "--data-root",
        help="Web 任务目录；默认读取 PDF_TRANSLATE_WEB_DATA 或 data/web_jobs",
    ),
) -> None:
    base = (data_dir or _default_data_base()).resolve()
    root = (data_root or Path(os.getenv("PDF_TRANSLATE_WEB_DATA", base / "web_jobs"))).resolve()
    payload = build_security_preflight(base, root)
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("run")
def cmd_run(
    pdf: Path = typer.Argument(..., exists=True),
    work_dir: Path = typer.Argument(..., help="工作目录"),
    backend: str = typer.Option(None, "--backend", "-b", help=f"{backend_choice_text()}；默认环境变量 PDF_TRANSLATE_BACKEND"),
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
    backend: str = typer.Option("echo", "--backend", "-b", help=f"实验后端：{backend_choice_text()}；专利指标预跑建议先用 echo"),
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


@app.command("experiment-samples")
def cmd_experiment_samples(
    sources: list[Path] = typer.Argument(
        ...,
        exists=True,
        help="PDF 文件或包含 PDF 的目录",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="写出的样本清单 CSV，可直接传给 experiment --sample-manifest",
    ),
    report: Path | None = typer.Option(
        None,
        "--report",
        help="可选 JSON 分析报告路径；缺省使用 CSV 同名 .json",
    ),
    markdown_report: Path | None = typer.Option(
        None,
        "--markdown-report",
        help="可选 Markdown 覆盖度报告路径；缺省使用 CSV 同名 .md",
    ),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="目录输入时递归扫描 PDF"),
    max_pages: int = typer.Option(20, "--max-pages", min=1, help="每篇 PDF 最多检查的页数"),
) -> None:
    pdfs = _expand_pdf_sources(sources, recursive=recursive)
    if not pdfs:
        raise typer.BadParameter("no PDF files found")
    report_path = report or output.with_suffix(".json")
    markdown_path = markdown_report or output.with_suffix(".md")
    manifest = write_sample_manifest(
        pdfs,
        output,
        report_path=report_path,
        markdown_path=markdown_path,
        max_pages=max_pages,
    )
    typer.echo(f"样本清单 CSV: {output.resolve()}")
    typer.echo(f"样本分析 JSON: {report_path.resolve()}")
    typer.echo(f"覆盖度 Markdown: {markdown_path.resolve()}")
    typer.echo(f"样本数: {manifest['sample_count']}")
    coverage = (manifest.get("summary") or {}).get("coverage") or {}
    if coverage:
        typer.echo(
            f"申请前覆盖建议: {coverage.get('met_requirement_count', 0)}/"
            f"{coverage.get('requirement_count', 0)} 项已达建议数量"
        )
        missing = coverage.get("missing_counts") or {}
        if missing:
            typer.echo(
                "仍需补样本: "
                + ", ".join(f"{category}+{count}" for category, count in sorted(missing.items()))
            )


@app.command("experiment-evidence")
def cmd_experiment_evidence(
    summary_json: Path = typer.Option(
        ...,
        "--summary-json",
        exists=True,
        dir_okay=False,
        help="批量实验生成的 batch_experiment_summary.json",
    ),
    review_csv: Path | None = typer.Option(
        None,
        "--review-csv",
        exists=True,
        dir_okay=False,
        help="填写后的 batch_experiment_review.csv；缺省使用 summary_json 同目录文件",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="证据摘要输出目录；缺省写回 summary_json 同目录",
    ),
    require_selected: bool = typer.Option(
        False,
        "--require-selected",
        help="没有任何 include_in_patent_evidence 入选行时返回错误",
    ),
) -> None:
    review_path = review_csv or summary_json.parent / "batch_experiment_review.csv"
    if not review_path.is_file():
        raise typer.BadParameter(f"review csv not found: {review_path}")
    target_dir = output_dir or summary_json.parent
    evidence = write_batch_experiment_evidence(summary_json, review_path, target_dir)
    if require_selected and not evidence.get("included_count"):
        raise typer.BadParameter("no rows were selected for patent evidence")
    typer.echo(f"专利证据 JSON: {(target_dir / 'batch_experiment_evidence.json').resolve()}")
    typer.echo(f"专利证据 Markdown: {(target_dir / 'batch_experiment_evidence.md').resolve()}")
    typer.echo(f"纳入证据: {evidence.get('included_count', 0)}/{evidence.get('review_row_count', 0)}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
