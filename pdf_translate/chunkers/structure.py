from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pdf_translate.chunking import TextChunk
from pdf_translate.extractors.document_ir import BlockIR, DocumentIR
from pdf_translate.run_metrics import estimate_token_count
from pdf_translate.structure_boundaries import detect_page_boundary_fragments


@dataclass
class StructureChunk(TextChunk):
    """TextChunk compatible chunk that preserves source IR block provenance."""

    block_ids: list[str] = field(default_factory=list)
    block_types: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    boundary_fragment_ids: list[str] = field(default_factory=list)
    structural_relation_ids: list[str] = field(default_factory=list)
    approx_tokens: int = 0
    budget_target_chars: int = 0
    budget_max_chars: int = 0
    split_reason: str = "unknown"
    budget_overflow_chars: int = 0
    budget_pressure: str = "unknown"

    def to_manifest_entry(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "pages_1based": [self.pages_0based[0] + 1, self.pages_0based[-1] + 1] if self.pages_0based else [],
            "block_ids": self.block_ids,
            "block_types": self.block_types,
            "link_count": self.link_count,
            "image_count": self.image_count,
            "approx_chars": len(self.text),
            "approx_tokens": self.approx_tokens,
            "warnings": self.warnings,
            "boundary_fragment_ids": self.boundary_fragment_ids,
            "structural_relation_ids": self.structural_relation_ids,
            "budget": {
                "target_chars": self.budget_target_chars,
                "max_chars": self.budget_max_chars,
                "split_reason": self.split_reason,
                "overflow_chars": self.budget_overflow_chars,
                "pressure": self.budget_pressure,
            },
        }


def _block_for_translation(block: BlockIR) -> bool:
    if block.type == "header_footer":
        return False
    if block.type == "image":
        promotions = block.meta.get("ocr_promotions") if isinstance(block.meta, dict) else None
        return bool(block.text.strip() and isinstance(promotions, list) and promotions)
    return bool(block.text.strip())


def _markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    column_count = max(len(r) for r in rows)
    normalized = [r + [""] * (column_count - len(r)) for r in rows]
    if column_count < 2:
        return "\n".join(" ".join(r).strip() for r in normalized)
    header = normalized[0]
    body = normalized[1:] or [[""] * column_count]
    lines = [
        "| " + " | ".join(cell.strip() for cell in header) + " |",
        "| " + " | ".join("---" for _ in range(column_count)) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(cell.strip() for cell in row) + " |")
    return "\n".join(lines)


def _format_block(block: BlockIR) -> str:
    label = {
        "heading": "标题",
        "paragraph": "正文",
        "table": "表格",
        "caption": "图表注",
        "footnote": "脚注",
        "formula": "公式",
        "reference": "参考文献",
    }.get(block.type, block.type)
    if block.type == "table":
        table = block.meta.get("table") if isinstance(block.meta, dict) else None
        rows = table.get("rows") if isinstance(table, dict) else None
        if isinstance(rows, list) and rows:
            md = _markdown_table([[str(cell) for cell in row] for row in rows if isinstance(row, list)])
            if md:
                return f"[第 {block.page_no} 页｜{label}｜{block.block_id}]\n{md}"
    return f"[第 {block.page_no} 页｜{label}｜{block.block_id}]\n{block.text.strip()}"


def _make_chunk(
    chunk_index: int,
    blocks: list[BlockIR],
    link_count: int,
    image_count: int,
    warnings: list[str],
    boundary_fragment_ids: list[str],
    structural_relation_ids: list[str],
    target_chars: int,
    max_chars: int,
    split_reason: str,
) -> StructureChunk:
    pages = sorted({b.page_no - 1 for b in blocks})
    type_counts = Counter(b.type for b in blocks)
    text = "\n\n".join(_format_block(b) for b in blocks)
    approx_chars = len(text)
    return StructureChunk(
        chunk_id=f"c{chunk_index:04d}",
        pages_0based=pages,
        text=text,
        link_count=link_count,
        image_count=image_count,
        block_ids=[b.block_id for b in blocks],
        block_types=dict(type_counts),
        warnings=warnings,
        boundary_fragment_ids=boundary_fragment_ids,
        structural_relation_ids=structural_relation_ids,
        approx_tokens=estimate_token_count(approx_chars),
        budget_target_chars=target_chars,
        budget_max_chars=max_chars,
        split_reason=split_reason,
        budget_overflow_chars=max(0, approx_chars - max_chars),
        budget_pressure=_budget_pressure(approx_chars, target_chars, max_chars),
    )


def _boundary_fragment_map(doc_ir: DocumentIR) -> dict[tuple[int, int], dict]:
    out: dict[tuple[int, int], dict] = {}
    for fragment in detect_page_boundary_fragments(doc_ir):
        pages = fragment.get("pages_1based")
        if isinstance(pages, list) and len(pages) == 2:
            out[(int(pages[0]), int(pages[1]))] = fragment
    return out


def _protected_boundary_fragment(
    current: list[BlockIR],
    next_block: BlockIR,
    boundary_fragments: dict[tuple[int, int], dict],
) -> dict | None:
    if not current:
        return None
    previous_page = max(block.page_no for block in current)
    if next_block.page_no != previous_page + 1:
        return None
    return boundary_fragments.get((previous_page, next_block.page_no))


def _relation_label(child: BlockIR) -> str:
    raw = child.meta.get("parent_relation") if isinstance(child.meta, dict) else ""
    return str(raw or "structural_relation")


def _structural_relation_id(previous: BlockIR, current: BlockIR) -> str:
    if previous.parent_id == current.block_id:
        return f"{previous.block_id}->{current.block_id}:{_relation_label(previous)}"
    if current.parent_id == previous.block_id:
        return f"{previous.block_id}->{current.block_id}:{_relation_label(current)}"
    if previous.parent_id and previous.parent_id == current.parent_id:
        return f"{previous.parent_id}->{previous.block_id}+{current.block_id}:shared_parent"
    return ""


def _opens_structural_relation(
    block: BlockIR,
    parent_ids: set[str],
    child_parent_ids: set[str],
    current_ids: set[str],
) -> bool:
    opens_as_child = bool(block.parent_id and block.parent_id in parent_ids and block.parent_id not in current_ids)
    opens_as_parent = block.block_id in child_parent_ids
    return opens_as_child or opens_as_parent


def _budget_pressure(approx_chars: int, target_chars: int, max_chars: int) -> str:
    if approx_chars > max_chars:
        return "over_max"
    if approx_chars > target_chars:
        return "over_target"
    return "within_target"


def build_structure_chunks(
    doc_ir: DocumentIR,
    *,
    target_chars: int = 9000,
    max_chars: int = 14000,
    max_pages_per_chunk: int = 3,
) -> list[StructureChunk]:
    """Build chunks from structure blocks, keeping tables/captions/formulas atomic."""
    if max_pages_per_chunk < 1:
        raise ValueError("max_pages_per_chunk must be >= 1")
    if target_chars < 1000:
        raise ValueError("target_chars must be >= 1000")
    if max_chars < target_chars:
        raise ValueError("max_chars must be >= target_chars")

    chunks: list[StructureChunk] = []
    current: list[BlockIR] = []
    current_links = 0
    current_images = 0
    current_warnings: list[str] = []
    current_page_nos: set[int] = set()
    current_boundary_fragment_ids: list[str] = []
    current_structural_relation_ids: list[str] = []
    boundary_fragments = _boundary_fragment_map(doc_ir)
    protected_page_limit = max_pages_per_chunk + 1
    translatable_blocks = [
        block
        for page in doc_ir.pages
        for block in page.blocks
        if _block_for_translation(block)
    ]
    parent_ids = {block.block_id for block in translatable_blocks}
    child_parent_ids = {
        str(block.parent_id)
        for block in translatable_blocks
        if block.parent_id and str(block.parent_id) in parent_ids
    }

    def flush(split_reason: str = "end_of_document") -> None:
        nonlocal current, current_links, current_images, current_warnings, current_page_nos
        nonlocal current_boundary_fragment_ids, current_structural_relation_ids
        if not current:
            return
        chunks.append(
            _make_chunk(
                len(chunks),
                current,
                current_links,
                current_images,
                sorted(set(current_warnings)),
                sorted(set(current_boundary_fragment_ids)),
                sorted(set(current_structural_relation_ids)),
                target_chars,
                max_chars,
                split_reason,
            )
        )
        current = []
        current_links = 0
        current_images = 0
        current_warnings = []
        current_page_nos = set()
        current_boundary_fragment_ids = []
        current_structural_relation_ids = []

    for page in doc_ir.pages:
        page_blocks = [b for b in page.blocks if _block_for_translation(b)]
        for block in page_blocks:
            candidate_len = sum(len(b.text) for b in current) + len(block.text)
            pages_if_added = {b.page_no for b in current}
            pages_if_added.add(block.page_no)
            protected_fragment = _protected_boundary_fragment(current, block, boundary_fragments)
            current_ids = {b.block_id for b in current}
            structural_relation_id = _structural_relation_id(current[-1], block) if current else ""
            relation_opener = _opens_structural_relation(block, parent_ids, child_parent_ids, current_ids)
            page_limit_exceeded = len(pages_if_added) > max_pages_per_chunk
            protected_page_limit_exceeded = len(pages_if_added) > protected_page_limit
            max_chars_exceeded = candidate_len > max_chars
            target_chars_exceeded = candidate_len > target_chars
            split_reason = ""
            if relation_opener and not structural_relation_id and target_chars_exceeded:
                split_reason = "before_structural_relation"
            elif max_chars_exceeded and not structural_relation_id:
                split_reason = "max_chars"
            elif target_chars_exceeded and len(pages_if_added) > 1 and not protected_fragment and not structural_relation_id:
                split_reason = "target_chars"
            elif page_limit_exceeded and not structural_relation_id and (not protected_fragment or protected_page_limit_exceeded):
                split_reason = "page_limit"
            elif structural_relation_id and protected_page_limit_exceeded:
                split_reason = "protected_relation_page_limit"
            should_flush = bool(current) and (
                bool(split_reason)
            )
            if should_flush:
                flush(split_reason)
                protected_fragment = None
                structural_relation_id = ""
            elif protected_fragment:
                boundary_id = str(protected_fragment.get("boundary_id") or "")
                if boundary_id:
                    current_boundary_fragment_ids.append(boundary_id)
                    current_warnings.append(f"protected_page_boundary:{boundary_id}")
            if structural_relation_id:
                current_structural_relation_ids.append(structural_relation_id)
                current_warnings.append(f"protected_structural_relation:{structural_relation_id}")
                if max_chars_exceeded:
                    current_warnings.append(f"budget_overflow_for_structural_relation:{structural_relation_id}")
                elif target_chars_exceeded:
                    current_warnings.append(f"budget_pressure_for_structural_relation:{structural_relation_id}")
            current.append(block)
            if page.page_no not in current_page_nos:
                current_page_nos.add(page.page_no)
                current_links += page.link_count
                current_images += page.image_count
                current_warnings.extend(page.warnings)
    flush()
    return chunks


def write_structure_manifest(chunks: list[StructureChunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([c.to_manifest_entry() for c in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
