from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pdf_translate.extractors.document_ir import BlockIR, DocumentIR, PageIR


_TRANSLATABLE_BLOCK_TYPES = {
    "paragraph",
    "heading",
    "table",
    "caption",
    "footnote",
    "formula",
    "reference",
}
_CONTINUABLE_BLOCK_TYPES = {"paragraph", "caption", "footnote", "formula"}
_TRAILING_WRAPPERS = "\"'”’）)]}」』"
_TERMINAL_PUNCTUATION = ".!?。！？"
_SOFT_ENDING_PUNCTUATION = ",，;；:：-–—"
_CONTINUATION_START_RE = re.compile(
    r"^(and|or|but|which|that|where|while|when|with|without|between|from|to|of|in|for|as|by|than|therefore|however)\b",
    re.I,
)


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _content_blocks(page: PageIR) -> list[BlockIR]:
    return [
        block
        for block in page.blocks
        if block.type in _TRANSLATABLE_BLOCK_TYPES and block.text.strip()
    ]


def _tail_snippet(text: str, limit: int = 160) -> str:
    return _compact(text)[-limit:]


def _head_snippet(text: str, limit: int = 160) -> str:
    return _compact(text)[:limit]


def _ends_without_terminal_punctuation(text: str) -> bool:
    compact = _compact(text).rstrip(_TRAILING_WRAPPERS)
    if not compact:
        return False
    last = compact[-1]
    if last in _TERMINAL_PUNCTUATION:
        return False
    if last in _SOFT_ENDING_PUNCTUATION:
        return True
    tokens = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", compact)
    return len(tokens) >= 4 and last.isalnum()


def _starts_like_continuation(text: str) -> bool:
    compact = _compact(text)
    if not compact:
        return False
    if re.match(r"^[,;:)\]\}，；：）]", compact):
        return True
    first = compact[0]
    return ("a" <= first <= "z") or bool(_CONTINUATION_START_RE.match(compact))


def _page_boundary_fragment(prev_page: PageIR, next_page: PageIR) -> dict[str, Any] | None:
    prev_blocks = _content_blocks(prev_page)
    next_blocks = _content_blocks(next_page)
    if not prev_blocks or not next_blocks:
        return None

    prev_block = prev_blocks[-1]
    next_block = next_blocks[0]
    prev_unfinished = _ends_without_terminal_punctuation(prev_block.text)
    next_continues = _starts_like_continuation(next_block.text)
    same_continuable_type = (
        prev_block.type == next_block.type
        and prev_block.type in _CONTINUABLE_BLOCK_TYPES
    )
    possible_table_continuation = prev_block.type == "table" and next_block.type == "table"

    reasons: list[str] = []
    if prev_unfinished:
        reasons.append("previous_page_ends_without_terminal_punctuation")
    if next_continues:
        reasons.append("next_page_starts_like_continuation")
    if same_continuable_type:
        reasons.append("same_continuable_block_type_across_boundary")
    if possible_table_continuation:
        reasons.append("possible_table_continuation")

    is_fragment = possible_table_continuation or (
        prev_unfinished and (next_continues or same_continuable_type)
    )
    if not is_fragment:
        return None

    severity = "high" if possible_table_continuation or (prev_unfinished and next_continues) else "medium"
    if possible_table_continuation:
        suggestion = "keep_pages_in_same_structure_chunk_and_reconstruct_continued_table"
    else:
        suggestion = "keep_pages_in_same_structure_chunk_or_apply_deferred_tail"

    return {
        "boundary_id": f"p{prev_page.page_no}-p{next_page.page_no}",
        "pages_1based": [prev_page.page_no, next_page.page_no],
        "severity": severity,
        "reasons": reasons,
        "previous_block_id": prev_block.block_id,
        "next_block_id": next_block.block_id,
        "previous_block_type": prev_block.type,
        "next_block_type": next_block.type,
        "previous_tail": _tail_snippet(prev_block.text),
        "next_head": _head_snippet(next_block.text),
        "suggested_handling": suggestion,
    }


def detect_page_boundary_fragments(doc_ir: DocumentIR) -> list[dict[str, Any]]:
    """Detect adjacent-page fragments caused by page cuts before translation."""
    fragments: list[dict[str, Any]] = []
    for prev_page, next_page in zip(doc_ir.pages, doc_ir.pages[1:]):
        fragment = _page_boundary_fragment(prev_page, next_page)
        if fragment:
            fragments.append(fragment)
    return fragments


def build_structure_qa(doc_ir: DocumentIR) -> dict[str, Any]:
    """Summarize local structure invariants for later translation QA and experiments."""
    block_counts: Counter[str] = Counter()
    page_warnings: list[dict[str, Any]] = []
    table_blocks: list[dict[str, Any]] = []
    page_boundary_fragments = detect_page_boundary_fragments(doc_ir)

    for page in doc_ir.pages:
        if page.warnings:
            page_warnings.append({"page_no": page.page_no, "warnings": page.warnings})
        for block in page.blocks:
            block_counts[block.type] += 1
            if block.type != "table":
                continue
            table = block.meta.get("table") if isinstance(block.meta, dict) else None
            table = table if isinstance(table, dict) else {}
            table_blocks.append(
                {
                    "block_id": block.block_id,
                    "page_no": block.page_no,
                    "bbox": list(block.bbox),
                    "row_count": int(table.get("row_count") or 0),
                    "column_count": int(table.get("column_count") or 0),
                    "header": table.get("header") or [],
                    "numeric_tokens": table.get("numeric_tokens") or [],
                    "warnings": table.get("warnings") or [],
                    "confidence": table.get("confidence") or "low",
                }
            )

    boundary_count = len(page_boundary_fragments)
    possible_boundary_count = max(0, len(doc_ir.pages) - 1)
    return {
        "schema_version": "structure-qa-v1",
        "doc_id": doc_ir.doc_id,
        "summary": {
            "page_count": len(doc_ir.pages),
            "block_counts": dict(block_counts),
            "table_count": len(table_blocks),
            "warning_page_count": len(page_warnings),
            "page_boundary_fragment_count": boundary_count,
            "page_boundary_fragment_rate": round(boundary_count / possible_boundary_count, 4)
            if possible_boundary_count
            else 0.0,
        },
        "tables": table_blocks,
        "page_boundary_fragments": page_boundary_fragments,
        "page_warnings": page_warnings,
    }


def write_structure_qa(doc_ir: DocumentIR, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_structure_qa(doc_ir), ensure_ascii=False, indent=2), encoding="utf-8")
