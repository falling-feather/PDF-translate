from __future__ import annotations

import re
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


def _merged_preview(previous: str, following: str, limit: int = 260) -> str:
    merged = _compact(previous).rstrip() + " " + _compact(following).lstrip()
    return merged[:limit]


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


def _continuation_kind(prev_block: BlockIR, next_block: BlockIR, possible_table_continuation: bool) -> str:
    if possible_table_continuation:
        return "table_continuation"
    if prev_block.type == next_block.type and prev_block.type in _CONTINUABLE_BLOCK_TYPES:
        return f"{prev_block.type}_continuation"
    return "text_continuation"


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
        stitch_action = "preserve_table_segments_together"
        joiner = "preserve_table_rows"
    else:
        suggestion = "keep_pages_in_same_structure_chunk_or_apply_deferred_tail"
        stitch_action = "translate_as_continuous_cross_page_text"
        joiner = "space"
    continuation_kind = _continuation_kind(prev_block, next_block, possible_table_continuation)

    return {
        "boundary_id": f"p{prev_page.page_no}-p{next_page.page_no}",
        "pages_1based": [prev_page.page_no, next_page.page_no],
        "severity": severity,
        "continuation_kind": continuation_kind,
        "stitch_action": stitch_action,
        "stitch_confidence": severity,
        "joiner": joiner,
        "reasons": reasons,
        "previous_block_id": prev_block.block_id,
        "next_block_id": next_block.block_id,
        "previous_block_type": prev_block.type,
        "next_block_type": next_block.type,
        "previous_tail": _tail_snippet(prev_block.text),
        "next_head": _head_snippet(next_block.text),
        "merged_preview": _merged_preview(prev_block.text, next_block.text),
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
