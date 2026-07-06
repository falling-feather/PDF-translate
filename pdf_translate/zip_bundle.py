"""打包下载：压缩包内使用中文路径别名，便于阅读归档。"""

from __future__ import annotations

from pathlib import Path


def map_bundle_arcname(rel_posix: str) -> str:
    """工作目录相对路径（posix）→ zip 内显示路径。"""
    rel = rel_posix.replace("\\", "/")

    output_map = {
        "output/translated_full.md": "译文/完整译文.md",
        "output/translated_full.pdf": "译文/结构化译文.pdf",
        "output/translated_pdf_report.json": "质量/PDF译文生成报告.json",
        "output/repaired_full.md": "译文/局部修复合并译文.md",
        "output/published_full.md": "译文/人工确认修复发布稿.md",
        "output/bilingual.html": "译文/双语对照.html",
        "output/chunks_manifest.json": "设置/分段清单.json",
        "output/glossary_retranslation_plan.json": "质量/术语确认重译计划.json",
        "output/glossary_retranslation_plan.md": "质量/术语确认重译计划.md",
        "output/glossary_retranslation_result.json": "质量/术语重译执行报告.json",
        "output/glossary_retranslation_result.md": "质量/术语重译执行报告.md",
        "output/glossary_retranslated_full.md": "译文/术语候选重译全文.md",
        "output/glossary_retranslation_publish.json": "质量/术语重译发布确认.json",
        "output/glossary_retranslation_publish.md": "质量/术语重译发布确认.md",
        "output/glossary_retranslation_published_full.md": "译文/术语重译发布稿.md",
        "output/glossary_retranslation_rollback.json": "质量/术语重译回滚演练.json",
        "output/glossary_retranslation_rollback.md": "质量/术语重译回滚演练.md",
        "output/glossary_retranslation_rollback_full.md": "译文/术语重译回滚演练稿.md",
        "output/document_ir.json": "设置/文档结构IR.json",
        "output/structure_chunks_manifest.json": "设置/结构分段清单.json",
        "output/structure_hints_manifest.json": "设置/结构提示清单.json",
        "output/structure_qa.json": "质量/结构QA.json",
        "output/table_reconstruction.json": "质量/表格重建证据.json",
        "output/table_merged_cell_review.json": "质量/表格合并候选人工确认.json",
        "output/table_merged_cell_review.md": "质量/表格合并候选人工确认.md",
        "output/table_structure_publish.json": "质量/表格结构确认发布.json",
        "output/table_structure_publish.md": "质量/表格结构确认发布.md",
        "output/table_reconstruction_confirmed.json": "质量/表格重建确认副本.json",
        "output/chunk_boundary_qa.json": "质量/分段边界QA.json",
        "output/chunk_strategy_comparison.json": "质量/分段策略对比.json",
        "output/vision_route.json": "质量/图像OCR路由.json",
        "output/ocr_tasks.json": "质量/OCR调度任务.json",
        "output/ocr_results.json": "质量/OCR识别结果.json",
        "output/ocr_writeback.json": "质量/OCR结果回写.json",
        "output/ocr_candidate_qa.json": "质量/OCR候选文本QA.json",
        "output/ocr_candidate_qa.md": "质量/OCR候选文本QA.md",
        "output/ocr_candidate_promotion.json": "质量/OCR候选文本晋级.json",
        "output/ocr_candidate_promotion.md": "质量/OCR候选文本晋级.md",
        "output/document_ir_ocr.json": "设置/OCR增强文档结构IR.json",
        "output/document_ir_promoted.json": "设置/OCR晋级文档结构IR.json",
        "output/qa_report.json": "质量/翻译QA.json",
        "output/qa_report.md": "质量/翻译QA.md",
        "output/repair_plan.json": "质量/局部修复计划.json",
        "output/repair_plan.md": "质量/局部修复计划.md",
        "output/repair_requests.json": "质量/局部修复请求.json",
        "output/repair_requests.md": "质量/局部修复请求.md",
        "output/repair_results.json": "质量/局部修复结果.json",
        "output/repair_results.md": "质量/局部修复结果.md",
        "output/repair_validation.json": "质量/局部修复验证.json",
        "output/repair_validation.md": "质量/局部修复验证.md",
        "output/repair_merge.json": "质量/局部修复合并.json",
        "output/repair_merge.md": "质量/局部修复合并.md",
        "output/repair_patch_review.json": "质量/局部修复补丁审核.json",
        "output/repair_patch_review.md": "质量/局部修复补丁审核.md",
        "output/repair_publish.json": "质量/局部修复发布确认.json",
        "output/repair_publish.md": "质量/局部修复发布确认.md",
        "output/repair_rollback.json": "质量/局部修复回滚演练.json",
        "output/repair_rollback.md": "质量/局部修复回滚演练.md",
        "output/rollback_full.md": "译文/局部修复回滚演练稿.md",
        "output/repair_formal_replace.json": "质量/局部修复正式替换.json",
        "output/repair_formal_replace.md": "质量/局部修复正式替换.md",
        "output/repair_formal_rollback.json": "质量/局部修复正式回滚.json",
        "output/repair_formal_rollback.md": "质量/局部修复正式回滚.md",
        "output/formal_full.md": "译文/正式译文.md",
        "output/formal_full.before_repair.md": "译文/正式译文修复前备份.md",
        "output/formal_full.repair_applied.md": "译文/正式译文回滚前修复稿.md",
        "output/repair_merge_qa.json": "质量/局部修复后QA.json",
        "output/repair_merge_qa.md": "质量/局部修复后QA.md",
        "output/repair_effectiveness.json": "质量/局部修复效果对比.json",
        "output/repair_effectiveness.md": "质量/局部修复效果对比.md",
        "output/experiment_metrics.json": "质量/实验指标.json",
        "output/run_metrics.json": "质量/运行性能指标.json",
        "output/cost_estimate.json": "质量/成本估算.json",
        "output/links_index.csv": "关键词/链接索引.csv",
        "output/state.json": "设置/翻译状态.json",
        "output/run_log.jsonl": "设置/运行日志.jsonl",
    }
    if rel == "output/vlm_tasks.json":
        return "质量/VLM视觉复核任务.json"
    if rel in output_map:
        return output_map[rel]
    if rel.startswith("output/repairs/"):
        name = rel.split("/")[-1]
        return f"质量/局部修复片段/{name}"
    if rel.startswith("output/repaired_chunks/"):
        name = rel.split("/")[-1]
        return f"译文/局部修复分块/{name}"
    if rel.startswith("output/glossary_retranslated_chunks/"):
        name = rel.split("/")[-1]
        return f"译文/术语候选重译分块/{name}"
    if rel.startswith("output/source_chunks/"):
        name = rel.split("/")[-1]
        return f"设置/源文分块/{name}"
    if rel.startswith("output/vision_pages/"):
        name = rel.split("/")[-1]
        return f"质量/图像OCR页面预览/{name}"
    if rel.startswith("output/vision_crops/"):
        parts = rel.split("/")
        name = parts[-1]
        page_dir = parts[-2] if len(parts) >= 3 else "页面"
        return f"质量/图像OCR区域裁剪/{page_dir}/{name}"

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


def _is_within_root(root: Path, candidate: Path) -> bool:
    """Return True when candidate resolves inside root."""
    try:
        root_resolved = root.resolve()
        candidate_resolved = candidate.resolve()
    except OSError:
        return False
    try:
        candidate_resolved.relative_to(root_resolved)
    except ValueError:
        return False
    return True


def _iter_dir_files(directory: Path, root: Path, out: list[Path]) -> None:
    if directory.is_symlink():
        return
    try:
        entries = list(directory.iterdir())
    except (FileNotFoundError, OSError):
        return
    for path in entries:
        if path.is_symlink():
            continue
        if path.is_file():
            if _is_within_root(root, path):
                out.append(path)
            continue
        if path.is_dir():
            _iter_dir_files(path, root, out)


def iter_bundle_files(root: Path) -> list[Path]:
    """与历史逻辑一致：固定清单 + memory 下全部文件。"""
    root = root.resolve()
    out: list[Path] = []

    def _add_candidate(path: Path) -> None:
        if path.is_symlink():
            return
        if path.is_file() and _is_within_root(root, path):
            out.append(path)

    candidates = [
        root / "output" / "translated_full.md",
        root / "output" / "translated_full.pdf",
        root / "output" / "translated_pdf_report.json",
        root / "output" / "repaired_full.md",
        root / "output" / "published_full.md",
        root / "output" / "bilingual.html",
        root / "output" / "chunks_manifest.json",
        root / "output" / "glossary_retranslation_plan.json",
        root / "output" / "glossary_retranslation_plan.md",
        root / "output" / "glossary_retranslation_result.json",
        root / "output" / "glossary_retranslation_result.md",
        root / "output" / "glossary_retranslated_full.md",
        root / "output" / "glossary_retranslation_publish.json",
        root / "output" / "glossary_retranslation_publish.md",
        root / "output" / "glossary_retranslation_published_full.md",
        root / "output" / "glossary_retranslation_rollback.json",
        root / "output" / "glossary_retranslation_rollback.md",
        root / "output" / "glossary_retranslation_rollback_full.md",
        root / "output" / "document_ir.json",
        root / "output" / "structure_chunks_manifest.json",
        root / "output" / "structure_hints_manifest.json",
        root / "output" / "structure_qa.json",
        root / "output" / "table_reconstruction.json",
        root / "output" / "table_merged_cell_review.json",
        root / "output" / "table_merged_cell_review.md",
        root / "output" / "table_structure_publish.json",
        root / "output" / "table_structure_publish.md",
        root / "output" / "table_reconstruction_confirmed.json",
        root / "output" / "chunk_boundary_qa.json",
        root / "output" / "chunk_strategy_comparison.json",
        root / "output" / "vision_route.json",
        root / "output" / "ocr_tasks.json",
        root / "output" / "ocr_results.json",
        root / "output" / "ocr_writeback.json",
        root / "output" / "ocr_candidate_qa.json",
        root / "output" / "ocr_candidate_qa.md",
        root / "output" / "vlm_tasks.json",
        root / "output" / "ocr_candidate_promotion.json",
        root / "output" / "ocr_candidate_promotion.md",
        root / "output" / "document_ir_ocr.json",
        root / "output" / "document_ir_promoted.json",
        root / "output" / "qa_report.json",
        root / "output" / "qa_report.md",
        root / "output" / "repair_plan.json",
        root / "output" / "repair_plan.md",
        root / "output" / "repair_requests.json",
        root / "output" / "repair_requests.md",
        root / "output" / "repair_results.json",
        root / "output" / "repair_results.md",
        root / "output" / "repair_validation.json",
        root / "output" / "repair_validation.md",
        root / "output" / "repair_merge.json",
        root / "output" / "repair_merge.md",
        root / "output" / "repair_patch_review.json",
        root / "output" / "repair_patch_review.md",
        root / "output" / "repair_publish.json",
        root / "output" / "repair_publish.md",
        root / "output" / "repair_rollback.json",
        root / "output" / "repair_rollback.md",
        root / "output" / "rollback_full.md",
        root / "output" / "repair_formal_replace.json",
        root / "output" / "repair_formal_replace.md",
        root / "output" / "repair_formal_rollback.json",
        root / "output" / "repair_formal_rollback.md",
        root / "output" / "formal_full.md",
        root / "output" / "formal_full.before_repair.md",
        root / "output" / "formal_full.repair_applied.md",
        root / "output" / "repair_merge_qa.json",
        root / "output" / "repair_merge_qa.md",
        root / "output" / "repair_effectiveness.json",
        root / "output" / "repair_effectiveness.md",
        root / "output" / "experiment_metrics.json",
        root / "output" / "run_metrics.json",
        root / "output" / "cost_estimate.json",
        root / "output" / "links_index.csv",
        root / "output" / "state.json",
        root / "output" / "run_log.jsonl",
        root / "split" / "manifest.json",
        root / "split" / "main.pdf",
        root / "split" / "references.pdf",
    ]
    for f in candidates:
        _add_candidate(f)
    mem = root / "memory"
    if mem.is_dir():
        _iter_dir_files(mem, root, out)
    repairs = root / "output" / "repairs"
    if repairs.is_dir():
        _iter_dir_files(repairs, root, out)
    repaired_chunks = root / "output" / "repaired_chunks"
    if repaired_chunks.is_dir():
        _iter_dir_files(repaired_chunks, root, out)
    glossary_retranslated_chunks = root / "output" / "glossary_retranslated_chunks"
    if glossary_retranslated_chunks.is_dir():
        _iter_dir_files(glossary_retranslated_chunks, root, out)
    source_chunks = root / "output" / "source_chunks"
    if source_chunks.is_dir():
        _iter_dir_files(source_chunks, root, out)
    vision_pages = root / "output" / "vision_pages"
    if vision_pages.is_dir():
        _iter_dir_files(vision_pages, root, out)
    vision_crops = root / "output" / "vision_crops"
    if vision_crops.is_dir():
        _iter_dir_files(vision_crops, root, out)
    return out
