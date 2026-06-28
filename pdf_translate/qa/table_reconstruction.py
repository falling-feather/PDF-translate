from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pdf_translate.extractors.document_ir import BlockIR, DocumentIR

SCHEMA_VERSION = "table-reconstruction-v1"

_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?")
_UNIT_RE = re.compile(
    r"(?i)(?:\b\d+(?:[.,]\d+)?\s*)"
    r"(%|(?:ms|s|sec|seconds?|m|cm|mm|nm|um|µm|kg|g|mg|kb|mb|gb|hz|khz|mhz|ghz)\b)"
)
_SIGNIFICANCE_RE = re.compile(r"(\*{1,3}|†|‡|§|p\s*[<=>]\s*0?\.\d+)", re.I)


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
    rows = _as_rows(table_meta.get("rows"))
    row_count = int(table_meta.get("row_count") or len(rows) or 0)
    column_count = int(table_meta.get("column_count") or max((len(row) for row in rows), default=0))
    rows = _normalise_rows(rows, column_count)
    header = [str(cell).strip() for cell in table_meta.get("header") or [] if str(cell).strip()]
    if not header and rows and any(cell.strip() for cell in rows[0]) and not any(_NUMBER_RE.search(cell) for cell in rows[0]):
        header = [cell for cell in rows[0]]
    cells = _table_cells(rows, header)
    warnings = _table_warnings(
        table_meta,
        rows=rows,
        row_count=row_count,
        column_count=column_count,
        header=header,
    )
    structure_table = structure_tables.get(block.block_id, {})
    children = linked_children.get(block.block_id, {"captions": [], "footnotes": []})
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
        "caption_blocks": children.get("captions", []),
        "footnote_blocks": children.get("footnotes", []),
        "continued_from_block_id": structure_table.get("continued_from_block_id"),
        "continued_to_block_id": structure_table.get("continued_to_block_id"),
        "numeric_tokens": _unique([token for cell in cells for token in cell["numbers"]]),
        "unit_tokens": _unique([token for cell in cells for token in cell["units"]]),
        "significance_tokens": _unique([token for cell in cells for token in cell["significance"]]),
        "warnings": warnings,
        "cells": cells,
    }


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
    continuation_table_count = sum(
        1
        for table in tables
        if table.get("continued_from_block_id") or table.get("continued_to_block_id")
    )
    continuation_group_count = len((structure_qa or {}).get("table_continuations") or []) if isinstance(structure_qa, dict) else 0
    caption_linked_table_count = sum(1 for table in tables if table.get("caption_blocks"))
    footnote_linked_table_count = sum(1 for table in tables if table.get("footnote_blocks"))

    return {
        "schema_version": SCHEMA_VERSION,
        "doc_id": doc_ir.doc_id,
        "summary": {
            "table_count": table_count,
            "reconstructable_table_count": reconstructable_table_count,
            "low_confidence_table_count": low_confidence_table_count,
            "cell_count": cell_count,
            "nonempty_cell_count": nonempty_cell_count,
            "numeric_cell_count": numeric_cell_count,
            "numeric_token_count": sum(len(table.get("numeric_tokens") or []) for table in tables),
            "unit_token_count": sum(len(table.get("unit_tokens") or []) for table in tables),
            "significance_token_count": sum(len(table.get("significance_tokens") or []) for table in tables),
            "caption_linked_table_count": caption_linked_table_count,
            "footnote_linked_table_count": footnote_linked_table_count,
            "continuation_table_count": continuation_table_count,
            "continuation_group_count": continuation_group_count,
            "table_reconstruction_ready_rate": round(reconstructable_table_count / table_count, 4)
            if table_count
            else 0.0,
        },
        "tables": tables,
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
