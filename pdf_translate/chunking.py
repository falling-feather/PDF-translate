from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextChunk:
    """1–3 页为单元的正文块；带重叠页以保证衔接。"""

    chunk_id: str
    pages_0based: list[int]
    text: str
    link_count: int
    image_count: int


def build_text_chunks(
    pages: list[tuple[int, str, int, int]],
    *,
    pages_per_chunk: int = 3,
    overlap_pages: int = 1,
) -> list[TextChunk]:
    """
    pages: (page_index_0based, text, link_count, image_count) in document order.
    overlap_pages: 相邻块共享的页数（设计文档中的「重叠」，MVP 用整页）。
    """
    if pages_per_chunk < 1 or pages_per_chunk > 3:
        raise ValueError("pages_per_chunk must be 1..3 per design")
    if overlap_pages < 0 or overlap_pages >= pages_per_chunk:
        raise ValueError("overlap_pages must be in [0, pages_per_chunk)")

    if not pages:
        return []

    step = pages_per_chunk - overlap_pages
    chunks: list[TextChunk] = []
    start_idx = 0
    n = len(pages)
    serial = 0

    while start_idx < n:
        end_idx = min(start_idx + pages_per_chunk, n)
        slice_ = pages[start_idx:end_idx]
        pnums = [p[0] for p in slice_]
        text = "\n\n".join(p[1] for p in slice_)
        links = sum(p[2] for p in slice_)
        images = sum(p[3] for p in slice_)
        cid = f"c{serial:04d}"
        chunks.append(
            TextChunk(
                chunk_id=cid,
                pages_0based=pnums,
                text=text,
                link_count=links,
                image_count=images,
            )
        )
        serial += 1
        if end_idx >= n:
            break
        start_idx += step

    return chunks
