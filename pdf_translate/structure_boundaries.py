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
_HYPHENATED_END_RE = re.compile(r"(?P<prefix>[A-Za-z0-9]{2,})[-\u2010-\u2015]\s*$")
_WORD_FRAGMENT_START_RE = re.compile(r"^(?P<fragment>[A-Za-z][A-Za-z0-9-]*)")
_ACADEMIC_ABBREVIATION_END_RE = re.compile(
    r"(?:et\s+al\.|e\.g\.|i\.e\.|cf\.|vs\.|eqs?\.|figs?\.|tabs?\.|secs?\.|refs?\.|nos?\.|dr\.|prof\.)\s*$",
    re.I,
)
_ABBREVIATION_CONTINUATION_START_RE = re.compile(r"^(?:[\(\[\{]\s*)?(?:[a-z0-9]|[,;:，；：])")
_CONTINUATION_START_RE = re.compile(
    r"^(and|or|but|which|that|where|while|when|with|without|between|from|to|of|in|for|as|by|than|therefore|however)\b",
    re.I,
)
_ABBREVIATION_CONTINUABLE_BLOCK_TYPES = {"paragraph", "caption", "footnote", "formula", "reference"}
_FORMULA_OPERATOR_END_RE = re.compile(
    r"(?:[=+\-−–—*/×÷<>≤≥≈~&|^_(\[{,;:]|\\(?:sum|prod|int|frac|sqrt|left|right|cdot|times|leq|geq|approx)\b)\s*$"
)
_FORMULA_CONTINUATION_START_RE = re.compile(
    r"^(?:[=+\-−–—*/×÷<>≤≥≈~&|^_)\]\},;:]|\\(?:sum|prod|int|frac|sqrt|left|right|cdot|times|leq|geq|approx|alpha|beta|gamma|delta|theta|lambda|mu|sigma|omega)\b|[a-zA-Z]\s*[=+\-−–—*/×÷<>≤≥≈])"
)
_PRESERVE_HYPHEN_PREFIXES = {
    "anti",
    "co",
    "cross",
    "domain",
    "high",
    "inter",
    "intra",
    "low",
    "long",
    "meta",
    "multi",
    "non",
    "post",
    "pre",
    "semi",
    "self",
    "short",
    "state",
    "sub",
    "super",
    "well",
}


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


def _merged_preview(previous: str, following: str, limit: int = 260, *, joiner: str = "space") -> str:
    previous_compact = _compact(previous).rstrip()
    following_compact = _compact(following).lstrip()
    if joiner == "hyphen_elision":
        previous_compact = re.sub(r"[-\u2010-\u2015]\s*$", "", previous_compact)
        merged = previous_compact + following_compact
    elif joiner == "hyphen_preserve":
        merged = previous_compact + following_compact
    elif joiner == "newline":
        merged = previous_compact + "\n" + following_compact
    else:
        merged = previous_compact + " " + following_compact
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


def _ends_with_hyphenated_word(text: str) -> bool:
    return bool(_HYPHENATED_END_RE.search(_compact(text)))


def _hyphenated_word_parts(previous: str, following: str) -> tuple[str, str] | None:
    previous_match = _HYPHENATED_END_RE.search(_compact(previous))
    following_match = _WORD_FRAGMENT_START_RE.match(_compact(following))
    if not previous_match or not following_match:
        return None
    return previous_match.group("prefix"), following_match.group("fragment")


def _starts_with_word_fragment(text: str) -> bool:
    return bool(_WORD_FRAGMENT_START_RE.match(_compact(text)))


def _is_acronym_like(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    return bool(letters) and sum(1 for char in letters if char.isupper()) >= 2


def _should_preserve_hyphen(prefix: str, fragment: str) -> bool:
    normalized_prefix = prefix.strip().lower()
    if _is_acronym_like(prefix):
        return True
    if any(char.isdigit() for char in prefix):
        return True
    if normalized_prefix in _PRESERVE_HYPHEN_PREFIXES:
        return True
    if "-" in fragment:
        return True
    return False


def _is_titlecase_word(text: str) -> bool:
    stripped = re.sub(r"[^A-Za-z0-9]", "", text)
    if not stripped:
        return False
    if stripped.isupper():
        return True
    return stripped[0].isupper() and stripped[1:].islower()


def _looks_like_short_title_break(previous: str, following: str) -> bool:
    prev_compact = _compact(previous)
    next_compact = _compact(following)
    prev_words = re.findall(r"[A-Za-z0-9]+", re.sub(r"[-\u2010-\u2015]\s*$", "", prev_compact))
    next_words = re.findall(r"[A-Za-z0-9]+", next_compact)
    if not prev_words or not next_words:
        return False
    if len(prev_words) > 2 or len(next_words) > 4:
        return False
    return all(_is_titlecase_word(word) for word in prev_words + next_words)


def _hyphenated_joiner(previous: str, following: str) -> str | None:
    parts = _hyphenated_word_parts(previous, following)
    if not parts:
        return None
    prefix, fragment = parts
    if _looks_like_short_title_break(previous, following):
        return None
    if _should_preserve_hyphen(prefix, fragment):
        return "hyphen_preserve"
    return "hyphen_elision"


def _ends_with_academic_abbreviation(text: str) -> bool:
    return bool(_ACADEMIC_ABBREVIATION_END_RE.search(_compact(text)))


def _starts_like_abbreviation_continuation(text: str) -> bool:
    compact = _compact(text)
    if not compact:
        return False
    return _starts_like_continuation(compact) or bool(_ABBREVIATION_CONTINUATION_START_RE.match(compact))


def _starts_like_continuation(text: str) -> bool:
    compact = _compact(text)
    if not compact:
        return False
    if re.match(r"^[,;:)\]\}，；：）]", compact):
        return True
    first = compact[0]
    return ("a" <= first <= "z") or bool(_CONTINUATION_START_RE.match(compact))


def _ends_with_formula_operator(text: str) -> bool:
    return bool(_FORMULA_OPERATOR_END_RE.search(_compact(text)))


def _starts_like_formula_continuation(text: str) -> bool:
    compact = _compact(text)
    if not compact:
        return False
    return bool(_FORMULA_CONTINUATION_START_RE.match(compact)) or _starts_like_continuation(compact)


def _continuation_kind(
    prev_block: BlockIR,
    next_block: BlockIR,
    possible_table_continuation: bool,
    hyphenated_word_break: bool = False,
    academic_abbreviation_continuation: bool = False,
    formula_continuation: bool = False,
) -> str:
    if hyphenated_word_break:
        return "hyphenated_word_continuation"
    if academic_abbreviation_continuation:
        return "academic_abbreviation_continuation"
    if formula_continuation:
        return "formula_continuation"
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
    hyphenated_joiner = _hyphenated_joiner(prev_block.text, next_block.text)
    hyphenated_word_break = (
        prev_block.type == next_block.type
        and prev_block.type in _CONTINUABLE_BLOCK_TYPES
        and bool(hyphenated_joiner)
        and _starts_with_word_fragment(next_block.text)
    )
    academic_abbreviation_continuation = (
        prev_block.type == next_block.type
        and prev_block.type in _ABBREVIATION_CONTINUABLE_BLOCK_TYPES
        and _ends_with_academic_abbreviation(prev_block.text)
        and _starts_like_abbreviation_continuation(next_block.text)
    )
    title_line_break_false_positive = (
        prev_block.type == next_block.type == "paragraph"
        and _looks_like_short_title_break(prev_block.text, next_block.text)
    )
    same_continuable_type = (
        prev_block.type == next_block.type
        and prev_block.type in _CONTINUABLE_BLOCK_TYPES
    )
    possible_table_continuation = prev_block.type == "table" and next_block.type == "table"
    previous_formula_ends_with_operator = (
        prev_block.type == next_block.type == "formula"
        and _ends_with_formula_operator(prev_block.text)
    )
    next_formula_starts_with_math_continuation = (
        prev_block.type == next_block.type == "formula"
        and _starts_like_formula_continuation(next_block.text)
    )
    formula_continuation = (
        prev_block.type == next_block.type == "formula"
        and (
            previous_formula_ends_with_operator
            or next_formula_starts_with_math_continuation
            or prev_unfinished
        )
    )

    reasons: list[str] = []
    if prev_unfinished:
        reasons.append("previous_page_ends_without_terminal_punctuation")
    if next_continues:
        reasons.append("next_page_starts_like_continuation")
    if hyphenated_word_break:
        reasons.append("hyphenated_word_break_across_page")
        if hyphenated_joiner == "hyphen_preserve":
            reasons.append("preserve_hyphenated_compound")
        elif hyphenated_joiner == "hyphen_elision":
            reasons.append("elide_soft_line_break_hyphen")
    if academic_abbreviation_continuation:
        reasons.append("academic_abbreviation_at_page_end")
        reasons.append("next_page_starts_like_abbreviation_continuation")
    if formula_continuation:
        reasons.append("formula_continuation_across_page")
        if previous_formula_ends_with_operator:
            reasons.append("previous_formula_ends_with_operator")
        if next_formula_starts_with_math_continuation:
            reasons.append("next_formula_starts_with_math_continuation")
    if same_continuable_type:
        reasons.append("same_continuable_block_type_across_boundary")
    if possible_table_continuation:
        reasons.append("possible_table_continuation")

    generic_text_continuation = (
        prev_unfinished
        and (next_continues or same_continuable_type)
        and not title_line_break_false_positive
    )
    is_fragment = (
        possible_table_continuation
        or hyphenated_word_break
        or academic_abbreviation_continuation
        or formula_continuation
        or generic_text_continuation
    )
    if not is_fragment:
        return None

    severity = (
        "high"
        if possible_table_continuation
        or hyphenated_word_break
        or academic_abbreviation_continuation
        or formula_continuation
        or (prev_unfinished and next_continues)
        else "medium"
    )
    if possible_table_continuation:
        suggestion = "keep_pages_in_same_structure_chunk_and_reconstruct_continued_table"
        stitch_action = "preserve_table_segments_together"
        joiner = "preserve_table_rows"
    elif hyphenated_word_break:
        if hyphenated_joiner == "hyphen_preserve":
            suggestion = "preserve_hyphenated_compound_and_keep_pages_in_same_structure_chunk"
            stitch_action = "preserve_hyphenated_compound_across_page_boundary"
            joiner = "hyphen_preserve"
        else:
            suggestion = "join_hyphenated_word_and_keep_pages_in_same_structure_chunk"
            stitch_action = "join_hyphenated_word_across_page_boundary"
            joiner = "hyphen_elision"
    elif academic_abbreviation_continuation:
        suggestion = "keep_academic_abbreviation_sentence_in_same_structure_chunk"
        stitch_action = "preserve_academic_abbreviation_context_across_page"
        joiner = "space"
    elif formula_continuation:
        suggestion = "keep_formula_derivation_in_same_structure_chunk"
        stitch_action = "preserve_formula_derivation_across_page_boundary"
        joiner = "newline"
    else:
        suggestion = "keep_pages_in_same_structure_chunk_or_apply_deferred_tail"
        stitch_action = "translate_as_continuous_cross_page_text"
        joiner = "space"
    continuation_kind = _continuation_kind(
        prev_block,
        next_block,
        possible_table_continuation,
        hyphenated_word_break,
        academic_abbreviation_continuation,
        formula_continuation,
    )

    return {
        "boundary_id": f"p{prev_page.page_no}-p{next_page.page_no}",
        "pages_1based": [prev_page.page_no, next_page.page_no],
        "severity": severity,
        "continuation_kind": continuation_kind,
        "stitch_action": stitch_action,
        "stitch_confidence": severity,
        "joiner": joiner,
        "hyphenation_decision": joiner if joiner in {"hyphen_elision", "hyphen_preserve"} else None,
        "reasons": reasons,
        "previous_block_id": prev_block.block_id,
        "next_block_id": next_block.block_id,
        "previous_block_type": prev_block.type,
        "next_block_type": next_block.type,
        "previous_tail": _tail_snippet(prev_block.text),
        "next_head": _head_snippet(next_block.text),
        "merged_preview": _merged_preview(prev_block.text, next_block.text, joiner=joiner),
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
