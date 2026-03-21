"""下载译文的建议文件名（原 PDF 名 + 状态说明）。"""

from __future__ import annotations

import json
import re
from pathlib import Path


def safe_stem(original_filename: str | None) -> str:
    stem = Path(original_filename or "document").stem
    stem = re.sub(r'[<>:"/\\|?*\n\r\t]', "_", stem).strip() or "document"
    return stem[:200]


def _load_state_manifest(work_dir: Path) -> tuple[dict, list] | tuple[None, None]:
    state_path = work_dir / "output" / "state.json"
    man_path = work_dir / "output" / "chunks_manifest.json"
    if not state_path.is_file() or not man_path.is_file():
        return None, None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, None
    return state, manifest


def completed_chunk_segment_span(work_dir: Path) -> tuple[int, int] | None:
    """已译块在正文顺序中的段号范围（1-based，与 manifest 顺序一致）。"""
    state, manifest = _load_state_manifest(work_dir)
    if not state or not manifest:
        return None
    done = set(state.get("completed") or [])
    if not done:
        return None
    indices: list[int] = []
    for i, item in enumerate(manifest):
        if item.get("chunk_id") in done:
            indices.append(i + 1)
    if not indices:
        return None
    return min(indices), max(indices)


def suggest_md_download_name(
    *,
    original_filename: str | None,
    work_dir: Path,
    complete: bool,
) -> str:
    stem = safe_stem(original_filename)
    if complete:
        return f"{stem}（翻译版）.md"
    seg = completed_chunk_segment_span(work_dir)
    if seg:
        a, b = seg
        if a == b:
            return f"{stem}（第{a}段翻译）.md"
        return f"{stem}（第{a}-{b}段翻译）.md"
    return f"{stem}（部分翻译）.md"


def suggest_zip_bundle_name(
    *,
    original_filename: str | None,
    work_dir: Path,
    complete: bool,
) -> str:
    """与译文 .md 命名规则一致，扩展名为 .zip。"""
    base = suggest_md_download_name(
        original_filename=original_filename,
        work_dir=work_dir,
        complete=complete,
    )
    if base.lower().endswith(".md"):
        return base[:-3] + ".zip"
    return f"{safe_stem(original_filename)}（翻译版）.zip"
