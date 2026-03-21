"""合并译文 Markdown：全量重建与增量追加。"""

from __future__ import annotations

from pathlib import Path

from pdf_translate.chunking import TextChunk
from pdf_translate.deferral_markers import (
    finalize_merged_translation_markdown,
    strip_yaml_front_matter,
)


def merge_chunks_markdown(
    chunk_dir: Path,
    target: Path,
    chunks: list[TextChunk],
    *,
    strip_front_matter: bool = True,
    finalize_markers: bool = True,
) -> None:
    parts: list[str] = []
    for ch in chunks:
        fp = chunk_dir / f"{ch.chunk_id}.md"
        if fp.is_file():
            raw = fp.read_text(encoding="utf-8")
            body = strip_yaml_front_matter(raw) if strip_front_matter else raw
            parts.append(body.strip())
    target.parent.mkdir(parents=True, exist_ok=True)
    joined = "\n\n".join(parts)
    if finalize_markers:
        joined = finalize_merged_translation_markdown(joined)
    target.write_text(joined, encoding="utf-8")


def append_chunk_to_merged(merged_path: Path, chunk_body: str) -> None:
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with merged_path.open("a", encoding="utf-8") as f:
        if f.tell() > 0:
            f.write("\n\n")
        f.write(chunk_body)
