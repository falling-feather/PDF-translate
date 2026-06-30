from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pdf_translate.chunking import TextChunk
from pdf_translate.extractors.document_ir import BlockIR, DocumentIR

SCHEMA_VERSION = "table-reconstruction-v1"

_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?")
_UNIT_RE = re.compile(
    r"(?i)(?:\b\d+(?:[.,]\d+)?\s*)"
    r"(%|(?:ms|s|sec|seconds?|m|cm|mm|nm|um|µm|kg|g|mg|kb|mb|gb|hz|khz|mhz|ghz)\b)"
)
_SIGNIFICANCE_RE = re.compile(r"(\*{1,3}|†|‡|§|p\s*[<=>]\s*0?\.\d+)", re.I)
_FOOTNOTE_MARKER_TOKEN = r"\*{1,3}|[A-Za-z]|\d{1,3}|[\u2020\u2021\u00a7\u00b6#]"
_FOOTNOTE_PREFIX_RE = re.compile(
    rf"^\s*(?:note\s*[:：]?\s*)?[\[\(]?({_FOOTNOTE_MARKER_TOKEN})[\]\)]?(?=$|[\s\.\):：、])",
    re.I,
)
_FOOTNOTE_EXPLICIT_MARKER_RE = re.compile(
    rf"(?:\[({_FOOTNOTE_MARKER_TOKEN})\]|\(({_FOOTNOTE_MARKER_TOKEN})\)|\^({_FOOTNOTE_MARKER_TOKEN}))(?!\w)",
    re.I,
)
_FOOTNOTE_SYMBOL_MARKER_RE = re.compile(r"(\*{1,3}|[\u2020\u2021\u00a7\u00b6#])")


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _count_values(items: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        counts[text] = counts.get(text, 0) + 1
    return dict(sorted(counts.items()))


def _sum_count_dicts(items: list[dict[str, int]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        for key, value in item.items():
            text = str(key).strip()
            if not text:
                continue
            counts[text] = counts.get(text, 0) + int(value)
    return dict(sorted(counts.items()))


def _count_candidate_types(candidates: list[dict[str, Any]]) -> dict[str, int]:
    return _count_values([str(candidate.get("span_type") or "unknown") for candidate in candidates])


def _count_candidate_reasons(candidates: list[dict[str, Any]]) -> dict[str, int]:
    return _count_values([str(candidate.get("reason") or "unknown") for candidate in candidates])


def _chain_reason_category(reason: str) -> str:
    text = str(reason or "").strip()
    if text.startswith("header_mismatch_segment_"):
        return "header_mismatch"
    if text.startswith("missing_header_for_segment_"):
        return "missing_header"
    if text == "ragged_table_rows_in_chain":
        return "ragged_table_rows"
    if text == "low_confidence_table_structure_in_chain":
        return "low_confidence_table_structure"
    return text or "unknown"


def _clip(text: str, limit: int = 120) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def _as_rows(raw_rows: Any) -> list[list[str]]:
    rows: list[list[str]] = []
    if not isinstance(raw_rows, list):
        return rows
    for row in raw_rows:
        if not isinstance(row, list):
            continue
        rows.append([str(cell).strip() for cell in row])
    return rows


def _normalise_rows(rows: list[list[str]], column_count: int) -> list[list[str]]:
    if column_count <= 0:
        column_count = max((len(row) for row in rows), default=0)
    return [row + [""] * max(0, column_count - len(row)) for row in rows]


def _row_signature(row: list[str]) -> tuple[str, ...]:
    return tuple(re.sub(r"\s+", " ", str(cell or "")).strip().casefold() for cell in row)


def _linked_children(doc_ir: DocumentIR) -> dict[str, dict[str, list[dict[str, Any]]]]:
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for page in doc_ir.pages:
        for block in page.blocks:
            if block.type not in {"caption", "footnote"} or not block.parent_id:
                continue
            meta = block.meta if isinstance(block.meta, dict) else {}
            relation = str(meta.get("parent_relation") or "")
            if relation not in {"caption_for_table", "footnote_for_table"}:
                continue
            bucket = out.setdefault(block.parent_id, {"captions": [], "footnotes": []})
            target = "captions" if block.type == "caption" else "footnotes"
            bucket[target].append(
                {
                    "block_id": block.block_id,
                    "page_no": block.page_no,
                    "relation": relation,
                    "text": block.text.strip(),
                }
            )
    return out


def _structure_table_index(structure_qa: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(structure_qa, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for table in structure_qa.get("tables") or []:
        if not isinstance(table, dict):
            continue
        block_id = str(table.get("block_id") or "")
        if block_id:
            out[block_id] = table
    return out


def _cell_tokens(text: str) -> dict[str, list[str]]:
    numbers = _unique(_NUMBER_RE.findall(text))
    units = _unique(_UNIT_RE.findall(text))
    significance = _unique(match.group(0) for match in _SIGNIFICANCE_RE.finditer(text))
    return {
        "numbers": numbers,
        "units": units,
        "significance": significance,
        "locked_tokens": _unique(numbers + units + significance),
    }


def _cell_role(row_index: int, column_index: int, header: list[str]) -> str:
    if row_index == 0 and header:
        return "header"
    if column_index == 0:
        return "row_header"
    return "data"


def _normalise_marker(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).casefold()


def _marker_is_numeric(marker: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}", str(marker or "")))


def _marker_is_alpha(marker: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]", str(marker or "")))


def _footnote_markers(text: str) -> list[str]:
    markers: list[str] = []
    prefix = _FOOTNOTE_PREFIX_RE.search(text or "")
    if prefix:
        markers.append(prefix.group(1))
    for match in _FOOTNOTE_EXPLICIT_MARKER_RE.finditer(text or ""):
        marker = next((group for group in match.groups() if group), "")
        if marker:
            markers.append(marker)
    markers.extend(match.group(1) for match in _FOOTNOTE_SYMBOL_MARKER_RE.finditer(text or ""))
    markers.extend(match.group(0) for match in _SIGNIFICANCE_RE.finditer(text or ""))
    return _unique(markers)


def _contains_explicit_cell_marker(text: str, marker: str) -> bool:
    escaped = re.escape(marker)
    explicit = re.compile(rf"(?:\[{escaped}\]|\({escaped}\)|\^{escaped})(?!\w)", re.I)
    return bool(explicit.search(text or ""))


def _cell_marker_hits(cell: dict[str, Any], markers: list[str]) -> list[str]:
    text = str(cell.get("text") or "")
    text_norm = _normalise_marker(text)
    searchable_tokens = [
        str(item)
        for key in ("significance", "locked_tokens")
        for item in cell.get(key) or []
        if str(item)
    ]
    token_norms = {_normalise_marker(item) for item in searchable_tokens}
    hits: list[str] = []
    for marker in markers:
        marker_text = str(marker or "").strip()
        if not marker_text:
            continue
        marker_norm = _normalise_marker(marker_text)
        if marker_norm in token_norms and not _marker_is_numeric(marker_text):
            hits.append(marker_text)
            continue
        if marker_text.startswith("*") or marker_text in {"\u2020", "\u2021", "\u00a7", "\u00b6", "#"}:
            if marker_text in text:
                hits.append(marker_text)
            continue
        if marker_norm.startswith("p") and marker_norm in text_norm:
            hits.append(marker_text)
            continue
        if (_marker_is_numeric(marker_text) or _marker_is_alpha(marker_text)) and _contains_explicit_cell_marker(
            text,
            marker_text,
        ):
            hits.append(marker_text)
    return _unique(hits)


def _binding_cell(cell: dict[str, Any], matched_markers: list[str]) -> dict[str, Any]:
    return {
        "row_index": int(cell.get("row_index") or 0),
        "column_index": int(cell.get("column_index") or 0),
        "role": str(cell.get("role") or "data"),
        "column_header": str(cell.get("column_header") or ""),
        "row_header": str(cell.get("row_header") or ""),
        "text": str(cell.get("text") or ""),
        "matched_markers": matched_markers,
    }


def _footnote_bindings(
    footnotes: list[dict[str, Any]],
    cells: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for footnote in footnotes:
        if not isinstance(footnote, dict):
            continue
        text = str(footnote.get("text") or "")
        markers = _footnote_markers(text)
        matched_cells: list[dict[str, Any]] = []
        for cell in cells:
            if not isinstance(cell, dict) or cell.get("empty"):
                continue
            hits = _cell_marker_hits(cell, markers)
            if hits:
                matched_cells.append(_binding_cell(cell, hits))
        warnings: list[str] = []
        if matched_cells:
            status = "bound_to_cells"
        elif markers:
            status = "unbound"
            warnings.append("cell_marker_not_found")
        else:
            status = "table_level_only"
            warnings.append("no_cell_marker_detected")
        bindings.append(
            {
                "footnote_block_id": str(footnote.get("block_id") or ""),
                "page_no": int(footnote.get("page_no") or 0),
                "text": _clip(text),
                "markers": markers,
                "status": status,
                "matched_cell_count": len(matched_cells),
                "matched_row_indices": sorted({cell["row_index"] for cell in matched_cells}),
                "matched_column_indices": sorted({cell["column_index"] for cell in matched_cells}),
                "matched_cells": matched_cells,
                "warnings": warnings,
            }
        )
    return bindings


def _merged_candidate(
    *,
    span_type: str,
    row_index: int,
    column_index: int,
    row_span: int,
    column_span: int,
    text: str,
    reason: str,
    confidence: str,
    covered_cells: list[dict[str, int]],
) -> dict[str, Any]:
    return {
        "span_type": span_type,
        "row_index": row_index,
        "column_index": column_index,
        "row_span": row_span,
        "column_span": column_span,
        "text": _clip(text),
        "reason": reason,
        "confidence": confidence,
        "covered_cells": covered_cells,
    }


def _merged_cell_candidates(rows: list[list[str]], column_count: int) -> list[dict[str, Any]]:
    if column_count < 2 or not rows:
        return []
    normalised = _normalise_rows(rows, column_count)
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def add(candidate: dict[str, Any]) -> None:
        covered = tuple((cell["row_index"], cell["column_index"]) for cell in candidate.get("covered_cells") or [])
        key = (
            candidate.get("span_type"),
            candidate.get("row_index"),
            candidate.get("column_index"),
            candidate.get("row_span"),
            candidate.get("column_span"),
            candidate.get("reason"),
            covered,
        )
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)

    for row_index, row in enumerate(normalised):
        raw_row = rows[row_index] if row_index < len(rows) else row
        padded_columns = set(range(max(0, len(raw_row)), column_count))
        nonempty_columns = [index for index, value in enumerate(row) if str(value).strip()]
        if len(raw_row) < column_count and nonempty_columns:
            if len(nonempty_columns) == 1:
                anchor_column = nonempty_columns[0]
                confidence = "medium" if row_index == 0 and not _NUMBER_RE.search(row[anchor_column]) else "low"
                add(
                    _merged_candidate(
                        span_type="colspan",
                        row_index=row_index,
                        column_index=anchor_column,
                        row_span=1,
                        column_span=max(1, column_count - anchor_column),
                        text=row[anchor_column],
                        reason="single_cell_ragged_row",
                        confidence=confidence,
                        covered_cells=[
                            {"row_index": row_index, "column_index": column_index}
                            for column_index in range(anchor_column + 1, column_count)
                        ],
                    )
                )
            else:
                anchor_column = max(nonempty_columns)
                if anchor_column < column_count - 1:
                    add(
                        _merged_candidate(
                            span_type="colspan",
                            row_index=row_index,
                            column_index=anchor_column,
                            row_span=1,
                            column_span=column_count - anchor_column,
                            text=row[anchor_column],
                            reason="ragged_row_trailing_span",
                            confidence="low",
                            covered_cells=[
                                {"row_index": row_index, "column_index": column_index}
                                for column_index in range(anchor_column + 1, column_count)
                            ],
                        )
                    )

        for column_index, text in enumerate(row):
            if column_index in padded_columns:
                continue
            if str(text).strip():
                continue
            if row_index > 0 and str(normalised[row_index - 1][column_index]).strip():
                confidence = "medium" if column_index == 0 else "low"
                add(
                    _merged_candidate(
                        span_type="rowspan",
                        row_index=row_index - 1,
                        column_index=column_index,
                        row_span=2,
                        column_span=1,
                        text=normalised[row_index - 1][column_index],
                        reason="empty_cell_below_nonempty_anchor",
                        confidence=confidence,
                        covered_cells=[{"row_index": row_index, "column_index": column_index}],
                    )
                )
            if column_index > 0 and str(row[column_index - 1]).strip():
                add(
                    _merged_candidate(
                        span_type="colspan",
                        row_index=row_index,
                        column_index=column_index - 1,
                        row_span=1,
                        column_span=2,
                        text=row[column_index - 1],
                        reason="empty_cell_right_of_nonempty_anchor",
                        confidence="low",
                        covered_cells=[{"row_index": row_index, "column_index": column_index}],
                    )
                )

    return candidates


def _table_cells(rows: list[list[str]], header: list[str]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        row_header = row[0] if row and row_index > 0 else ""
        for column_index, text in enumerate(row):
            tokens = _cell_tokens(text)
            column_header = header[column_index] if column_index < len(header) else ""
            cells.append(
                {
                    "row_index": row_index,
                    "column_index": column_index,
                    "role": _cell_role(row_index, column_index, header),
                    "column_header": column_header,
                    "row_header": row_header if column_index > 0 else "",
                    "text": text,
                    "empty": not bool(text.strip()),
                    "numbers": tokens["numbers"],
                    "units": tokens["units"],
                    "significance": tokens["significance"],
                    "locked_tokens": tokens["locked_tokens"],
                }
            )
    return cells


def _table_warnings(
    table_meta: dict[str, Any],
    *,
    rows: list[list[str]],
    row_count: int,
    column_count: int,
    header: list[str],
) -> list[str]:
    warnings = [str(item) for item in table_meta.get("warnings") or [] if str(item)]
    if not rows:
        warnings.append("missing_table_rows")
    if row_count < 2 or column_count < 2:
        warnings.append("not_reconstructable_shape")
    if not header and column_count >= 2:
        warnings.append("missing_table_header")
    if len({len(row) for row in rows}) > 1:
        warnings.append("ragged_table_rows")
    if str(table_meta.get("confidence") or "low").lower() == "low":
        warnings.append("low_confidence_table_structure")
    return _unique(warnings)


def _table_entry(
    block: BlockIR,
    *,
    linked_children: dict[str, dict[str, list[dict[str, Any]]]],
    structure_tables: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    meta = block.meta if isinstance(block.meta, dict) else {}
    table_meta = meta.get("table") if isinstance(meta.get("table"), dict) else {}
    raw_rows = _as_rows(table_meta.get("rows"))
    row_count = int(table_meta.get("row_count") or len(raw_rows) or 0)
    column_count = int(table_meta.get("column_count") or max((len(row) for row in raw_rows), default=0))
    rows = _normalise_rows(raw_rows, column_count)
    header = [str(cell).strip() for cell in table_meta.get("header") or [] if str(cell).strip()]
    if not header and rows and any(cell.strip() for cell in rows[0]) and not any(_NUMBER_RE.search(cell) for cell in rows[0]):
        header = [cell for cell in rows[0]]
    cells = _table_cells(rows, header)
    ragged_row_indices = [
        row_index
        for row_index, row in enumerate(raw_rows)
        if column_count > 0 and len(row) < column_count
    ]
    empty_cell_count = sum(1 for cell in cells if cell.get("empty"))
    merged_cell_candidates = _merged_cell_candidates(raw_rows, column_count)
    warnings = _table_warnings(
        table_meta,
        rows=raw_rows,
        row_count=row_count,
        column_count=column_count,
        header=header,
    )
    structure_table = structure_tables.get(block.block_id, {})
    children = linked_children.get(block.block_id, {"captions": [], "footnotes": []})
    footnote_bindings = _footnote_bindings(children.get("footnotes", []), cells)
    reconstructable = bool(rows and row_count >= 2 and column_count >= 2)
    return {
        "table_id": block.block_id,
        "block_id": block.block_id,
        "page_no": block.page_no,
        "bbox": list(block.bbox),
        "reconstructable": reconstructable,
        "confidence": table_meta.get("confidence") or "low",
        "row_count": row_count,
        "column_count": column_count,
        "header": header,
        "rows": rows,
        "ragged_row_indices": ragged_row_indices,
        "ragged_row_count": len(ragged_row_indices),
        "empty_cell_count": empty_cell_count,
        "merged_cell_candidate_count": len(merged_cell_candidates),
        "merged_cell_candidate_type_counts": _count_candidate_types(merged_cell_candidates),
        "merged_cell_candidate_reason_counts": _count_candidate_reasons(merged_cell_candidates),
        "merged_cell_candidates": merged_cell_candidates,
        "caption_blocks": children.get("captions", []),
        "footnote_blocks": children.get("footnotes", []),
        "footnote_bindings": footnote_bindings,
        "continued_from_block_id": structure_table.get("continued_from_block_id"),
        "continued_to_block_id": structure_table.get("continued_to_block_id"),
        "numeric_tokens": _unique([token for cell in cells for token in cell["numbers"]]),
        "unit_tokens": _unique([token for cell in cells for token in cell["units"]]),
        "significance_tokens": _unique([token for cell in cells for token in cell["significance"]]),
        "warnings": warnings,
        "cells": cells,
    }


def _continuation_chains(
    structure_qa: dict[str, Any] | None,
    tables_by_id: dict[str, dict[str, Any]],
) -> list[list[str]]:
    if not isinstance(structure_qa, dict):
        return []
    next_by_previous: dict[str, str] = {}
    previous_ids: set[str] = set()
    next_ids: set[str] = set()
    for item in structure_qa.get("table_continuations") or []:
        if not isinstance(item, dict):
            continue
        previous_id = str(item.get("previous_table_block_id") or "")
        next_id = str(item.get("next_table_block_id") or "")
        if previous_id not in tables_by_id or next_id not in tables_by_id:
            continue
        next_by_previous[previous_id] = next_id
        previous_ids.add(previous_id)
        next_ids.add(next_id)

    starts = list(previous_ids - next_ids) or list(previous_ids)
    starts.sort(key=lambda table_id: (int(tables_by_id[table_id].get("page_no") or 0), table_id))
    chains: list[list[str]] = []
    globally_seen: set[str] = set()
    for start in starts:
        if start in globally_seen:
            continue
        chain = [start]
        locally_seen = {start}
        current = start
        while current in next_by_previous:
            nxt = next_by_previous[current]
            if nxt in locally_seen:
                break
            chain.append(nxt)
            locally_seen.add(nxt)
            current = nxt
        if len(chain) >= 2:
            chains.append(chain)
            globally_seen.update(chain)
    return chains


def _continuation_ids_for_chain(chain: list[str], structure_qa: dict[str, Any] | None) -> list[str]:
    if not isinstance(structure_qa, dict):
        return []
    ids: list[str] = []
    pairs = {(chain[index], chain[index + 1]) for index in range(len(chain) - 1)}
    for item in structure_qa.get("table_continuations") or []:
        if not isinstance(item, dict):
            continue
        pair = (
            str(item.get("previous_table_block_id") or ""),
            str(item.get("next_table_block_id") or ""),
        )
        if pair in pairs:
            ids.append(str(item.get("continuation_id") or f"{pair[0]}->{pair[1]}"))
    return _unique(ids)


def _header_similarity(left: list[str], right: list[str]) -> float:
    left_tokens = {item for item in _row_signature(left) if item}
    right_tokens = {item for item in _row_signature(right) if item}
    if not left_tokens or not right_tokens:
        return 0.0
    return round(len(left_tokens & right_tokens) / len(left_tokens | right_tokens), 4)


def _chain_compatibility(source_tables: list[dict[str, Any]], header: list[str]) -> dict[str, Any]:
    column_counts = [int(table.get("column_count") or 0) for table in source_tables]
    row_counts = [int(table.get("row_count") or 0) for table in source_tables]
    header_similarities: list[float] = []
    warnings: list[str] = []
    reject_reasons: list[str] = []

    nonzero_columns = [count for count in column_counts if count > 0]
    if len(nonzero_columns) != len(column_counts):
        reject_reasons.append("missing_column_count")
    elif nonzero_columns and max(nonzero_columns) - min(nonzero_columns) >= 2:
        reject_reasons.append("column_count_drift")

    if any(count < 2 for count in row_counts) or any(count < 2 for count in column_counts):
        reject_reasons.append("not_reconstructable_segment")

    for table_index, table in enumerate(source_tables[1:], start=1):
        table_header = [str(cell).strip() for cell in table.get("header") or [] if str(cell).strip()]
        if not header or not table_header:
            warnings.append(f"missing_header_for_segment_{table_index}")
            continue
        similarity = _header_similarity(header, table_header)
        header_similarities.append(similarity)
        if similarity < 0.5:
            reject_reasons.append(f"header_mismatch_segment_{table_index}")

    for table in source_tables:
        table_warnings = [str(item) for item in table.get("warnings") or [] if str(item)]
        if "ragged_table_rows" in table_warnings:
            warnings.append("ragged_table_rows_in_chain")
        if "low_confidence_table_structure" in table_warnings:
            warnings.append("low_confidence_table_structure_in_chain")

    warnings = _unique(warnings)
    reject_reasons = _unique(reject_reasons)
    if reject_reasons:
        confidence = "low"
        merge_status = "rejected"
    elif warnings or any(score < 0.8 for score in header_similarities):
        confidence = "medium"
        merge_status = "merged"
    else:
        confidence = "high"
        merge_status = "merged"

    return {
        "merge_status": merge_status,
        "chain_confidence": confidence,
        "column_counts": column_counts,
        "row_counts": row_counts,
        "header_similarities": header_similarities,
        "warnings": warnings,
        "reject_reasons": reject_reasons,
    }


def _continued_table_group(
    chain: list[str],
    *,
    tables_by_id: dict[str, dict[str, Any]],
    structure_qa: dict[str, Any] | None,
) -> dict[str, Any]:
    source_tables = [tables_by_id[table_id] for table_id in chain]
    column_count = max((int(table.get("column_count") or 0) for table in source_tables), default=0)
    header: list[str] = []
    for table in source_tables:
        table_header = [str(cell).strip() for cell in table.get("header") or [] if str(cell).strip()]
        if table_header:
            header = table_header
            break
    compatibility = _chain_compatibility(source_tables, header)

    merged_rows: list[list[str]] = []
    skipped_repeated_header_count = 0
    if compatibility["merge_status"] == "merged":
        for table_index, table in enumerate(source_tables):
            rows = _as_rows(table.get("rows"))
            if not rows:
                continue
            if table_index > 0 and header and _row_signature(rows[0][: len(header)]) == _row_signature(header):
                rows = rows[1:]
                skipped_repeated_header_count += 1
            merged_rows.extend(rows)

    column_count = max(column_count, max((len(row) for row in merged_rows), default=0), len(header))
    merged_rows = _normalise_rows(merged_rows, column_count)
    cells = _table_cells(merged_rows, header)
    warnings = _unique(
        ["continued_table_group"]
        + [
            str(warning)
            for table in source_tables
            for warning in table.get("warnings") or []
            if str(warning)
        ]
        + [str(warning) for warning in compatibility.get("warnings") or [] if str(warning)]
        + [str(reason) for reason in compatibility.get("reject_reasons") or [] if str(reason)]
    )
    reconstructable = bool(
        compatibility["merge_status"] == "merged"
        and merged_rows
        and len(merged_rows) >= 2
        and column_count >= 2
    )
    table_ids = [str(table.get("table_id") or table.get("block_id") or "") for table in source_tables]
    pages = sorted({int(table.get("page_no") or 0) for table in source_tables if int(table.get("page_no") or 0)})
    continuation_ids = _continuation_ids_for_chain(chain, structure_qa)
    first_segment_row_count = int(source_tables[0].get("row_count") or 0) if source_tables else 0
    merged_row_gain = max(0, len(merged_rows) - first_segment_row_count) if reconstructable else 0
    footnote_blocks = [
        footnote
        for table in source_tables
        for footnote in table.get("footnote_blocks") or []
        if isinstance(footnote, dict)
    ]
    merged_cell_candidates = [
        {
            **candidate,
            "source_table_id": str(table.get("table_id") or table.get("block_id") or ""),
            "source_page_no": int(table.get("page_no") or 0),
        }
        for table in source_tables
        for candidate in table.get("merged_cell_candidates") or []
        if isinstance(candidate, dict)
    ]
    return {
        "group_id": "continued:" + "->".join(table_ids),
        "continuation_ids": continuation_ids,
        "table_ids": table_ids,
        "pages_1based": pages,
        "source_tables": [
            {
                "table_id": str(table.get("table_id") or table.get("block_id") or ""),
                "page_no": int(table.get("page_no") or 0),
                "row_count": int(table.get("row_count") or 0),
                "column_count": int(table.get("column_count") or 0),
            }
            for table in source_tables
        ],
        "merge_status": compatibility["merge_status"],
        "chain_confidence": compatibility["chain_confidence"],
        "compatibility": {
            "column_counts": compatibility["column_counts"],
            "row_counts": compatibility["row_counts"],
            "header_similarities": compatibility["header_similarities"],
            "warnings": compatibility["warnings"],
            "reject_reasons": compatibility["reject_reasons"],
        },
        "reconstructable": reconstructable,
        "segment_count": len(source_tables),
        "merged_row_count": len(merged_rows),
        "merged_column_count": column_count,
        "merged_row_gain": merged_row_gain,
        "skipped_repeated_header_count": skipped_repeated_header_count,
        "ragged_row_count": sum(int(table.get("ragged_row_count") or 0) for table in source_tables),
        "empty_cell_count": sum(int(table.get("empty_cell_count") or 0) for table in source_tables),
        "merged_cell_candidate_count": len(merged_cell_candidates),
        "merged_cell_candidate_type_counts": _count_candidate_types(merged_cell_candidates),
        "merged_cell_candidate_reason_counts": _count_candidate_reasons(merged_cell_candidates),
        "merged_cell_candidates": merged_cell_candidates,
        "header": header,
        "rows": merged_rows,
        "numeric_tokens": _unique([token for cell in cells for token in cell["numbers"]]),
        "unit_tokens": _unique([token for cell in cells for token in cell["units"]]),
        "significance_tokens": _unique([token for cell in cells for token in cell["significance"]]),
        "caption_blocks": [
            caption
            for table in source_tables
            for caption in table.get("caption_blocks") or []
            if isinstance(caption, dict)
        ],
        "footnote_blocks": footnote_blocks,
        "footnote_bindings": _footnote_bindings(footnote_blocks, cells),
        "warnings": warnings,
        "cells": cells,
    }


def _continued_table_groups(
    tables: list[dict[str, Any]],
    structure_qa: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    tables_by_id = {str(table.get("table_id") or table.get("block_id") or ""): table for table in tables}
    return [
        _continued_table_group(chain, tables_by_id=tables_by_id, structure_qa=structure_qa)
        for chain in _continuation_chains(structure_qa, tables_by_id)
    ]


def build_table_reconstruction_report(
    doc_ir: DocumentIR,
    structure_qa: dict[str, Any] | None = None,
) -> dict[str, Any]:
    linked = _linked_children(doc_ir)
    structure_tables = _structure_table_index(structure_qa)
    tables: list[dict[str, Any]] = []
    for page in doc_ir.pages:
        for block in page.blocks:
            if block.type == "table":
                tables.append(
                    _table_entry(
                        block,
                        linked_children=linked,
                        structure_tables=structure_tables,
                    )
                )

    table_count = len(tables)
    continued_table_groups = _continued_table_groups(tables, structure_qa)
    reconstructable_table_count = sum(1 for table in tables if table.get("reconstructable"))
    low_confidence_table_count = sum(1 for table in tables if table.get("confidence") == "low")
    cell_count = sum(len(table.get("cells") or []) for table in tables)
    nonempty_cell_count = sum(
        1
        for table in tables
        for cell in table.get("cells") or []
        if isinstance(cell, dict) and not cell.get("empty")
    )
    numeric_cell_count = sum(
        1
        for table in tables
        for cell in table.get("cells") or []
        if isinstance(cell, dict) and cell.get("numbers")
    )
    table_empty_cell_count = cell_count - nonempty_cell_count
    merged_cell_candidates = [
        candidate
        for table in tables
        for candidate in table.get("merged_cell_candidates") or []
        if isinstance(candidate, dict)
    ]
    table_merged_cell_candidate_count = len(merged_cell_candidates)
    table_merged_cell_candidate_type_counts = _count_candidate_types(merged_cell_candidates)
    table_merged_cell_candidate_reason_counts = _count_candidate_reasons(merged_cell_candidates)
    table_ragged_row_count = sum(int(table.get("ragged_row_count") or 0) for table in tables)
    table_ragged_table_count = sum(
        1
        for table in tables
        if int(table.get("ragged_row_count") or 0) > 0
        or "ragged_table_rows" in {str(item) for item in table.get("warnings") or []}
    )
    continuation_table_count = sum(
        1
        for table in tables
        if table.get("continued_from_block_id") or table.get("continued_to_block_id")
    )
    continuation_group_count = len((structure_qa or {}).get("table_continuations") or []) if isinstance(structure_qa, dict) else 0
    continued_table_group_count = len(continued_table_groups)
    continued_table_segment_count = sum(int(group.get("segment_count") or 0) for group in continued_table_groups)
    continued_table_merged_cell_candidate_count = sum(
        int(group.get("merged_cell_candidate_count") or 0) for group in continued_table_groups
    )
    continued_table_reconstructable_group_count = sum(
        1 for group in continued_table_groups if group.get("reconstructable")
    )
    continued_table_merged_row_count = sum(
        int(group.get("merged_row_count") or 0)
        for group in continued_table_groups
        if group.get("merge_status") == "merged"
    )
    table_chain_candidate_count = continued_table_group_count
    table_chain_merged_count = sum(
        1 for group in continued_table_groups if group.get("merge_status") == "merged"
    )
    table_chain_reject_count = sum(
        1 for group in continued_table_groups if group.get("merge_status") == "rejected"
    )
    table_chain_row_gain = sum(int(group.get("merged_row_gain") or 0) for group in continued_table_groups)
    table_chain_warning_count = sum(
        len(group.get("compatibility", {}).get("warnings") or [])
        + len(group.get("compatibility", {}).get("reject_reasons") or [])
        for group in continued_table_groups
        if isinstance(group.get("compatibility"), dict)
    )
    table_chain_reject_reasons = [
        str(reason)
        for group in continued_table_groups
        if isinstance(group.get("compatibility"), dict)
        for reason in group.get("compatibility", {}).get("reject_reasons") or []
        if str(reason).strip()
    ]
    table_chain_warning_reasons = [
        str(reason)
        for group in continued_table_groups
        if isinstance(group.get("compatibility"), dict)
        for reason in group.get("compatibility", {}).get("warnings") or []
        if str(reason).strip()
    ]
    table_chain_reject_reason_counts = _count_values(table_chain_reject_reasons)
    table_chain_reject_reason_category_counts = _count_values(
        [_chain_reason_category(reason) for reason in table_chain_reject_reasons]
    )
    table_chain_warning_reason_counts = _count_values(table_chain_warning_reasons)
    table_chain_warning_reason_category_counts = _count_values(
        [_chain_reason_category(reason) for reason in table_chain_warning_reasons]
    )
    caption_linked_table_count = sum(1 for table in tables if table.get("caption_blocks"))
    footnote_linked_table_count = sum(1 for table in tables if table.get("footnote_blocks"))
    footnote_bindings = [
        binding
        for table in tables
        for binding in table.get("footnote_bindings") or []
        if isinstance(binding, dict)
    ]
    table_footnote_binding_count = len(footnote_bindings)
    table_footnote_cell_binding_count = sum(
        1 for binding in footnote_bindings if binding.get("status") == "bound_to_cells"
    )
    table_footnote_bound_cell_count = sum(
        int(binding.get("matched_cell_count") or 0) for binding in footnote_bindings
    )
    table_footnote_unbound_count = sum(1 for binding in footnote_bindings if binding.get("status") == "unbound")
    table_footnote_table_level_count = sum(
        1 for binding in footnote_bindings if binding.get("status") == "table_level_only"
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "doc_id": doc_ir.doc_id,
        "summary": {
            "table_count": table_count,
            "reconstructable_table_count": reconstructable_table_count,
            "low_confidence_table_count": low_confidence_table_count,
            "cell_count": cell_count,
            "nonempty_cell_count": nonempty_cell_count,
            "empty_cell_count": table_empty_cell_count,
            "numeric_cell_count": numeric_cell_count,
            "numeric_token_count": sum(len(table.get("numeric_tokens") or []) for table in tables),
            "unit_token_count": sum(len(table.get("unit_tokens") or []) for table in tables),
            "significance_token_count": sum(len(table.get("significance_tokens") or []) for table in tables),
            "ragged_table_count": table_ragged_table_count,
            "ragged_row_count": table_ragged_row_count,
            "merged_cell_candidate_count": table_merged_cell_candidate_count,
            "merged_cell_candidate_type_counts": table_merged_cell_candidate_type_counts,
            "merged_cell_candidate_reason_counts": table_merged_cell_candidate_reason_counts,
            "caption_linked_table_count": caption_linked_table_count,
            "footnote_linked_table_count": footnote_linked_table_count,
            "table_footnote_binding_count": table_footnote_binding_count,
            "table_footnote_cell_binding_count": table_footnote_cell_binding_count,
            "table_footnote_bound_cell_count": table_footnote_bound_cell_count,
            "table_footnote_unbound_count": table_footnote_unbound_count,
            "table_footnote_table_level_count": table_footnote_table_level_count,
            "continuation_table_count": continuation_table_count,
            "continuation_group_count": continuation_group_count,
            "continued_table_group_count": continued_table_group_count,
            "continued_table_segment_count": continued_table_segment_count,
            "continued_table_merged_cell_candidate_count": continued_table_merged_cell_candidate_count,
            "continued_table_reconstructable_group_count": continued_table_reconstructable_group_count,
            "continued_table_merged_row_count": continued_table_merged_row_count,
            "table_chain_candidate_count": table_chain_candidate_count,
            "table_chain_merged_count": table_chain_merged_count,
            "table_chain_reject_count": table_chain_reject_count,
            "table_chain_row_gain": table_chain_row_gain,
            "table_chain_warning_count": table_chain_warning_count,
            "table_chain_reject_reason_count": len(table_chain_reject_reasons),
            "table_chain_warning_reason_count": len(table_chain_warning_reasons),
            "table_chain_reject_reason_counts": table_chain_reject_reason_counts,
            "table_chain_reject_reason_category_counts": table_chain_reject_reason_category_counts,
            "table_chain_warning_reason_counts": table_chain_warning_reason_counts,
            "table_chain_warning_reason_category_counts": table_chain_warning_reason_category_counts,
            "table_reconstruction_ready_rate": round(reconstructable_table_count / table_count, 4)
            if table_count
            else 0.0,
        },
        "tables": tables,
        "continued_table_groups": continued_table_groups,
    }


def write_table_reconstruction_report(
    doc_ir: DocumentIR,
    structure_qa: dict[str, Any] | None,
    path: Path,
) -> dict[str, Any]:
    report = build_table_reconstruction_report(doc_ir, structure_qa)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _chunk_block_ids(chunk: TextChunk) -> set[str]:
    return {str(block_id) for block_id in getattr(chunk, "block_ids", []) if str(block_id)}


def _table_matches_chunk(table: dict[str, Any], chunk: TextChunk, block_ids: set[str]) -> bool:
    table_id = str(table.get("block_id") or table.get("table_id") or "")
    if block_ids:
        return table_id in block_ids
    pages = {page + 1 for page in chunk.pages_0based}
    return int(table.get("page_no") or 0) in pages


def _table_locked_tokens(table: dict[str, Any], limit: int = 32) -> list[str]:
    tokens: list[str] = []
    for key in ("numeric_tokens", "unit_tokens", "significance_tokens"):
        tokens.extend(str(item) for item in table.get(key) or [] if str(item))
    for cell in table.get("cells") or []:
        if not isinstance(cell, dict):
            continue
        tokens.extend(str(item) for item in cell.get("locked_tokens") or [] if str(item))
    return _unique(tokens)[:limit]


def _cell_hint(cell: dict[str, Any]) -> str:
    row = int(cell.get("row_index") or 0)
    col = int(cell.get("column_index") or 0)
    role = str(cell.get("role") or "data")
    text = _clip(str(cell.get("text") or ""), 80)
    column_header = _clip(str(cell.get("column_header") or ""), 50)
    row_header = _clip(str(cell.get("row_header") or ""), 50)
    locked = _unique([str(item) for item in cell.get("locked_tokens") or [] if str(item)])
    parts = [f"r{row}c{col}", role]
    if column_header:
        parts.append(f"列={column_header}")
    if row_header:
        parts.append(f"行={row_header}")
    parts.append(f"值={text or '<空>'}")
    if locked:
        parts.append("锁定=" + ", ".join(locked[:8]))
    return "；".join(parts)


def _footnote_binding_hint(binding: dict[str, Any]) -> str:
    markers = [str(item) for item in binding.get("markers") or [] if str(item)]
    cells = [
        f"r{int(cell.get('row_index') or 0)}c{int(cell.get('column_index') or 0)}"
        for cell in binding.get("matched_cells") or []
        if isinstance(cell, dict)
    ]
    block_id = str(binding.get("footnote_block_id") or "")
    marker_text = ",".join(markers[:6]) if markers else "no-marker"
    cell_text = ",".join(cells[:12]) if cells else str(binding.get("status") or "unbound")
    return f"{block_id}:{marker_text}->{cell_text}"


def _merged_cell_candidate_hint(candidate: dict[str, Any]) -> str:
    row = int(candidate.get("row_index") or 0)
    col = int(candidate.get("column_index") or 0)
    span_type = str(candidate.get("span_type") or "unknown")
    row_span = int(candidate.get("row_span") or 1)
    column_span = int(candidate.get("column_span") or 1)
    span_label = {"colspan": "跨列", "rowspan": "跨行"}.get(span_type, span_type)
    covered = [
        f"r{int(cell.get('row_index') or 0)}c{int(cell.get('column_index') or 0)}"
        for cell in candidate.get("covered_cells") or []
        if isinstance(cell, dict)
    ]
    source_table_id = str(candidate.get("source_table_id") or "").strip()
    reason = str(candidate.get("reason") or "").strip()
    confidence = str(candidate.get("confidence") or "").strip()
    text = _clip(str(candidate.get("text") or ""), 60)
    cell_ref = f"r{row}c{col}"
    if source_table_id:
        cell_ref = f"{source_table_id}:{cell_ref}"
    parts = [cell_ref, f"疑似{span_label}候选({span_type} {row_span}x{column_span})"]
    if covered:
        parts.append("覆盖候选空位=" + ",".join(covered[:8]))
    if reason:
        parts.append("原因=" + reason)
    if confidence:
        parts.append("置信=" + confidence)
    if text:
        parts.append("锚文本=" + text)
    return "；".join(parts)


def build_table_translation_hints(
    chunk: TextChunk,
    table_reconstruction: dict[str, Any] | None,
    *,
    max_tables: int = 3,
    max_cells_per_table: int = 18,
    max_merged_candidates_per_table: int = 3,
) -> str:
    """Build compact table-preservation instructions for one translation chunk."""
    if not isinstance(table_reconstruction, dict):
        return ""
    tables = [table for table in table_reconstruction.get("tables") or [] if isinstance(table, dict)]
    if not tables:
        return ""
    block_ids = _chunk_block_ids(chunk)
    selected = [table for table in tables if _table_matches_chunk(table, chunk, block_ids)]
    if not selected:
        return ""

    lines = [
        "以下表格结构来自本地 DocumentIR。翻译时请保留相同行列数，输出 Markdown 表格；不要把表格线性化为普通段落；锁定 token 必须原样保留。",
    ]
    selected_table_ids = {str(table.get("table_id") or table.get("block_id") or "") for table in selected}
    for group in table_reconstruction.get("continued_table_groups") or []:
        if not isinstance(group, dict):
            continue
        group_table_ids = [str(table_id) for table_id in group.get("table_ids") or [] if str(table_id)]
        if not selected_table_ids.intersection(group_table_ids):
            continue
        merge_status = str(group.get("merge_status") or "")
        if merge_status == "merged":
            lines.append(
                "- 续表合并组 "
                + str(group.get("group_id") or "")
                + f"：覆盖 {' -> '.join(group_table_ids)}，合并后 {int(group.get('merged_row_count') or 0)} 行 x {int(group.get('merged_column_count') or 0)} 列。"
            )
        else:
            compatibility = group.get("compatibility") if isinstance(group.get("compatibility"), dict) else {}
            reasons = [
                str(item)
                for item in (compatibility.get("reject_reasons") or compatibility.get("warnings") or [])
                if str(item)
            ]
            lines.append(
                "- 续表候选 "
                + str(group.get("group_id") or "")
                + f"：覆盖 {' -> '.join(group_table_ids)}，当前未安全合并；请分别保留原表格形状。"
            )
            if reasons:
                lines.append("  未合并原因：" + ", ".join(reasons[:6]))
        header = [str(item).strip() for item in group.get("header") or [] if str(item).strip()]
        if merge_status == "merged" and header:
            lines.append("  合并表头：" + " | ".join(_clip(item, 40) for item in header[:12]))
        group_candidates = [
            candidate
            for candidate in group.get("merged_cell_candidates") or []
            if isinstance(candidate, dict)
        ]
        if group_candidates:
            lines.append(
                f"  续表组内疑似合并单元格候选 {len(group_candidates)} 个（未确认，仅作结构保护提示，不作为已确认合并结构处理）。"
            )
            lines.append(
                "  候选示例："
                + " / ".join(
                    _merged_cell_candidate_hint(candidate)
                    for candidate in group_candidates[:max(1, min(4, max_merged_candidates_per_table))]
                )
            )
    for table in selected[:max_tables]:
        table_id = str(table.get("table_id") or table.get("block_id") or "unknown")
        row_count = int(table.get("row_count") or 0)
        column_count = int(table.get("column_count") or 0)
        page_no = int(table.get("page_no") or 0)
        header = [str(item).strip() for item in table.get("header") or [] if str(item).strip()]
        locked_tokens = _table_locked_tokens(table)
        lines.append(f"- 表格 {table_id}（第 {page_no} 页）：{row_count} 行 x {column_count} 列。")
        if header:
            lines.append("  表头：" + " | ".join(_clip(item, 40) for item in header[:12]))
        if locked_tokens:
            lines.append("  锁定 token：" + ", ".join(locked_tokens[:32]))
        captions = table.get("caption_blocks") or []
        if captions:
            caption_texts = [_clip(item.get("text"), 100) for item in captions[:2] if isinstance(item, dict)]
            if caption_texts:
                lines.append("  表注/标题：" + " / ".join(caption_texts))
        footnotes = table.get("footnote_blocks") or []
        if footnotes:
            footnote_texts = [_clip(item.get("text"), 100) for item in footnotes[:2] if isinstance(item, dict)]
            if footnote_texts:
                lines.append("  表格脚注：" + " / ".join(footnote_texts))
        bindings = [
            binding
            for binding in table.get("footnote_bindings") or []
            if isinstance(binding, dict)
        ]
        if bindings:
            lines.append(
                "  footnote-cell bindings: "
                + " / ".join(_footnote_binding_hint(binding) for binding in bindings[:6])
            )
        merged_candidates = [
            candidate
            for candidate in table.get("merged_cell_candidates") or []
            if isinstance(candidate, dict)
        ]
        if merged_candidates:
            lines.append("  疑似合并单元格候选（未确认，仅作结构保护提示，不作为已确认合并结构处理）：")
            for candidate in merged_candidates[:max_merged_candidates_per_table]:
                lines.append("    - " + _merged_cell_candidate_hint(candidate))
            if len(merged_candidates) > max_merged_candidates_per_table:
                lines.append(f"    - 其余 {len(merged_candidates) - max_merged_candidates_per_table} 个候选略。")
        continuation = [
            str(table.get("continued_from_block_id") or "").strip(),
            str(table.get("continued_to_block_id") or "").strip(),
        ]
        continuation = [item for item in continuation if item]
        if continuation:
            lines.append("  续表关系：" + " / ".join(continuation))
        cells = [
            cell
            for cell in table.get("cells") or []
            if isinstance(cell, dict)
            and (
                cell.get("locked_tokens")
                or cell.get("role") in {"header", "row_header"}
            )
        ]
        if cells:
            lines.append("  单元格上下文：")
            for cell in cells[:max_cells_per_table]:
                lines.append("    - " + _cell_hint(cell))
    if len(selected) > max_tables:
        lines.append(f"- 其余 {len(selected) - max_tables} 个表格仅按原文中的 Markdown 表格形状保持。")
    return "\n".join(lines)


def build_structure_hints_manifest(
    chunks: list[TextChunk],
    table_reconstruction: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build an audit manifest for per-chunk structure hints sent to translators."""
    if not isinstance(table_reconstruction, dict):
        table_reconstruction = {}
    tables = [table for table in table_reconstruction.get("tables") or [] if isinstance(table, dict)]
    groups = [
        group
        for group in table_reconstruction.get("continued_table_groups") or []
        if isinstance(group, dict)
    ]
    chunk_entries: list[dict[str, Any]] = []
    for chunk in chunks:
        block_ids = _chunk_block_ids(chunk)
        selected = [table for table in tables if _table_matches_chunk(table, chunk, block_ids)]
        selected_table_ids = [
            str(table.get("table_id") or table.get("block_id") or "")
            for table in selected
            if str(table.get("table_id") or table.get("block_id") or "")
        ]
        selected_table_id_set = set(selected_table_ids)
        selected_groups = [
            group
            for group in groups
            if selected_table_id_set.intersection(
                {str(table_id) for table_id in group.get("table_ids") or [] if str(table_id)}
            )
        ]
        hints = build_table_translation_hints(chunk, table_reconstruction)
        merged_candidates = [
            candidate
            for table in selected
            for candidate in table.get("merged_cell_candidates") or []
            if isinstance(candidate, dict)
        ]
        footnote_binding_count = sum(
            len([binding for binding in table.get("footnote_bindings") or [] if isinstance(binding, dict)])
            for table in selected
        )
        locked_token_count = sum(len(_table_locked_tokens(table)) for table in selected)
        chunk_entries.append(
            {
                "chunk_id": chunk.chunk_id,
                "pages_1based": [chunk.pages_0based[0] + 1, chunk.pages_0based[-1] + 1],
                "block_ids": sorted(block_ids),
                "has_structure_hints": bool(hints.strip()),
                "hint_char_count": len(hints),
                "hint_line_count": len([line for line in hints.splitlines() if line.strip()]),
                "table_ids": selected_table_ids,
                "table_count": len(selected_table_ids),
                "continued_table_group_ids": [
                    str(group.get("group_id") or "") for group in selected_groups if str(group.get("group_id") or "")
                ],
                "continued_table_group_count": len(selected_groups),
                "merged_cell_candidate_count": len(merged_candidates),
                "merged_cell_candidate_type_counts": _count_candidate_types(merged_candidates),
                "merged_cell_candidate_reason_counts": _count_candidate_reasons(merged_candidates),
                "footnote_binding_count": footnote_binding_count,
                "locked_token_count": locked_token_count,
                "hint_text": hints,
            }
        )
    hinted_chunks = [entry for entry in chunk_entries if entry["has_structure_hints"]]
    hint_char_counts = [int(entry["hint_char_count"]) for entry in chunk_entries]
    structure_hint_char_count = sum(hint_char_counts)
    return {
        "schema_version": "structure-hints-manifest-v1",
        "doc_id": str(table_reconstruction.get("doc_id") or ""),
        "summary": {
            "chunk_count": len(chunk_entries),
            "structure_hint_chunk_count": len(hinted_chunks),
            "structure_hint_empty_chunk_count": len(chunk_entries) - len(hinted_chunks),
            "structure_hint_char_count": structure_hint_char_count,
            "structure_hint_avg_char_count": round(
                structure_hint_char_count / len(chunk_entries),
                4,
            )
            if chunk_entries
            else 0.0,
            "structure_hint_max_char_count": max(hint_char_counts) if hint_char_counts else 0,
            "structure_hint_table_count": sum(int(entry["table_count"]) for entry in chunk_entries),
            "structure_hint_continued_group_count": sum(
                int(entry["continued_table_group_count"]) for entry in chunk_entries
            ),
            "structure_hint_merged_cell_candidate_count": sum(
                int(entry["merged_cell_candidate_count"]) for entry in chunk_entries
            ),
            "structure_hint_merged_cell_candidate_type_counts": _sum_count_dicts(
                [
                    entry["merged_cell_candidate_type_counts"]
                    for entry in chunk_entries
                    if isinstance(entry["merged_cell_candidate_type_counts"], dict)
                ]
            ),
            "structure_hint_merged_cell_candidate_reason_counts": _sum_count_dicts(
                [
                    entry["merged_cell_candidate_reason_counts"]
                    for entry in chunk_entries
                    if isinstance(entry["merged_cell_candidate_reason_counts"], dict)
                ]
            ),
            "structure_hint_footnote_binding_count": sum(
                int(entry["footnote_binding_count"]) for entry in chunk_entries
            ),
            "structure_hint_locked_token_count": sum(int(entry["locked_token_count"]) for entry in chunk_entries),
        },
        "chunks": chunk_entries,
    }


def write_structure_hints_manifest(
    chunks: list[TextChunk],
    table_reconstruction: dict[str, Any] | None,
    path: Path,
) -> dict[str, Any]:
    manifest = build_structure_hints_manifest(chunks, table_reconstruction)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
