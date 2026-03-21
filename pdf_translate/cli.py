from __future__ import annotations

from pathlib import Path

import typer

from pdf_translate.config import AppConfig
from pdf_translate import pipeline

app = typer.Typer(help="PDF 英文学术文献：拆分参考文献、按块翻译、记忆目录（见 PROJECT_DESIGN.md）")


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
    )
    typer.echo(f"完成: {out}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
