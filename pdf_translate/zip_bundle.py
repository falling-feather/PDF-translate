"""打包下载：压缩包内使用中文路径别名，便于阅读归档。"""

from __future__ import annotations

from pathlib import Path


def map_bundle_arcname(rel_posix: str) -> str:
    """工作目录相对路径（posix）→ zip 内显示路径。"""
    rel = rel_posix.replace("\\", "/")

    output_map = {
        "output/translated_full.md": "译文/完整译文.md",
        "output/bilingual.html": "译文/双语对照.html",
        "output/chunks_manifest.json": "设置/分段清单.json",
        "output/document_ir.json": "设置/文档结构IR.json",
        "output/structure_chunks_manifest.json": "设置/结构分段清单.json",
        "output/structure_qa.json": "质量/结构QA.json",
        "output/table_reconstruction.json": "质量/表格重建证据.json",
        "output/chunk_boundary_qa.json": "质量/分段边界QA.json",
        "output/chunk_strategy_comparison.json": "质量/分段策略对比.json",
        "output/vision_route.json": "质量/图像OCR路由.json",
        "output/qa_report.json": "质量/翻译QA.json",
        "output/qa_report.md": "质量/翻译QA.md",
        "output/repair_plan.json": "质量/局部修复计划.json",
        "output/repair_plan.md": "质量/局部修复计划.md",
        "output/repair_requests.json": "质量/局部修复请求.json",
        "output/repair_requests.md": "质量/局部修复请求.md",
        "output/experiment_metrics.json": "质量/实验指标.json",
        "output/links_index.csv": "关键词/链接索引.csv",
        "output/state.json": "设置/翻译状态.json",
        "output/run_log.jsonl": "设置/运行日志.jsonl",
    }
    if rel in output_map:
        return output_map[rel]

    split_map = {
        "split/manifest.json": "设置/拆分清单.json",
        "split/main.pdf": "原文/正文.pdf",
        "split/references.pdf": "原文/参考文献.pdf",
    }
    if rel in split_map:
        return split_map[rel]

    if rel.startswith("memory/"):
        name = rel.split("/")[-1]
        mem_map = {
            "glossary.json": "记忆/术语表.json",
            "entities.json": "记忆/实体列表.json",
            "chunk_summaries.json": "记忆/分块摘要.json",
            "style_notes.yaml": "记忆/风格说明.yaml",
            "pending_review.json": "记忆/待复核项.json",
            "running_summary.md": "记忆/叙事摘要.md",
            "deferred_source_carry.txt": "记忆/顺延英文缓存.txt",
        }
        if name in mem_map:
            return mem_map[name]
        return f"记忆/其他/{name}"

    return rel


def iter_bundle_files(root: Path) -> list[Path]:
    """与历史逻辑一致：固定清单 + memory 下全部文件。"""
    root = root.resolve()
    candidates = [
        root / "output" / "translated_full.md",
        root / "output" / "bilingual.html",
        root / "output" / "chunks_manifest.json",
        root / "output" / "document_ir.json",
        root / "output" / "structure_chunks_manifest.json",
        root / "output" / "structure_qa.json",
        root / "output" / "table_reconstruction.json",
        root / "output" / "chunk_boundary_qa.json",
        root / "output" / "chunk_strategy_comparison.json",
        root / "output" / "vision_route.json",
        root / "output" / "qa_report.json",
        root / "output" / "qa_report.md",
        root / "output" / "repair_plan.json",
        root / "output" / "repair_plan.md",
        root / "output" / "repair_requests.json",
        root / "output" / "repair_requests.md",
        root / "output" / "experiment_metrics.json",
        root / "output" / "links_index.csv",
        root / "output" / "state.json",
        root / "output" / "run_log.jsonl",
        root / "split" / "manifest.json",
        root / "split" / "main.pdf",
        root / "split" / "references.pdf",
    ]
    out: list[Path] = []
    for f in candidates:
        if f.is_file():
            out.append(f)
    mem = root / "memory"
    if mem.is_dir():
        for p in mem.rglob("*"):
            if p.is_file():
                out.append(p)
    return out
