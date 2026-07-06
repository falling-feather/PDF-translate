from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pdf_translate.chunking import TextChunk
from pdf_translate.deferral_markers import strip_yaml_front_matter
from pdf_translate.extractors.document_ir import DocumentIR, extract_entity_candidates

SCHEMA_VERSION = "translation-qa-v1"

_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
_BRACKET_CITATION_RE = re.compile(r"\[[0-9,\-\s;]+\]")
_AUTHOR_YEAR_RE = re.compile(r"\([A-Z][A-Za-z\-]+(?:\s+et\s+al\.)?,\s*\d{4}[a-z]?\)")
_TABLE_FIGURE_RE = re.compile(
    r"\b(?P<label>Table|Fig(?:ure)?\.?)\s*(?P<num>\d+[A-Za-z]?)\b",
    re.I,
)
_MATH_SYMBOL_RE = re.compile(
    "(\u2264|\u2265|\u2260|\u00b1|\u2248|=|\u2211|\u222b|\u221a|"
    "\u2192|\u03b1|\u03b2|\u03b3|\u03b4|\u03bb|\u03bc|\u03c3)"
)
_EQUATION_LABEL_RE = re.compile(r"(?<![\w])\(\s*\d{1,3}[A-Za-z]?\s*\)")
_VARIABLE_TOKEN_RE = re.compile(
    r"\b[A-Za-z](?:_\{?[A-Za-z0-9,+\-]+\}?|\^\{?[A-Za-z0-9,+\-]+\}?)"
    r"(?:_\{?[A-Za-z0-9,+\-]+\}?|\^\{?[A-Za-z0-9,+\-]+\}?)*\b"
)
_METRIC_FORMULA_TOKEN_RE = re.compile(
    r"\b(?:F1(?:-score)?|p-value|ROC-AUC|AUC|BLEU|ROUGE(?:-[L12])?)\b",
    re.I,
)
_STAT_FORMULA_RE = re.compile(r"\bp\s*(?:<|<=|=|>|>=|\u2264|\u2265)\s*0?\.\d+\b", re.I)
_FOOTNOTE_ANCHOR_RE = re.compile(r"^\s*(?P<marker>\d+|[*†‡§])[\).、\s]*")
_ENTITY_MEDIUM_SEVERITY_TYPES = {"model_or_dataset", "acronym"}
_STRUCTURE_RELATION_TYPES = {
    "caption_for_table",
    "caption_for_figure",
    "footnote_for_table",
    "footnote_for_block",
}


def _unique_in_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _source_text_for_qa(text: str) -> str:
    # Structure chunks add local provenance labels; these should not become QA invariants.
    lines = [line for line in text.splitlines() if not line.startswith("[第 ")]
    return "\n".join(lines)


def _numbers(text: str) -> list[str]:
    return _unique_in_order(_NUMBER_RE.findall(text))


def _references(text: str) -> list[str]:
    return _unique_in_order(_BRACKET_CITATION_RE.findall(text) + _AUTHOR_YEAR_RE.findall(text))


def _math_symbols(text: str) -> list[str]:
    return _unique_in_order(_MATH_SYMBOL_RE.findall(text))


def _formula_invariants(text: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, token: str) -> None:
        value = re.sub(r"\s+", " ", token.strip())
        if kind == "equation_label":
            value = re.sub(r"\s+", "", value)
        if not value:
            return
        key = (kind, value.casefold() if kind in {"metric", "statistic"} else value)
        if key in seen:
            return
        seen.add(key)
        out.append({"kind": kind, "token": value})

    for token in _math_symbols(text):
        add("symbol", token)
    for match in _EQUATION_LABEL_RE.finditer(text):
        add("equation_label", match.group(0))
    for match in _VARIABLE_TOKEN_RE.finditer(text):
        add("variable", match.group(0))
    for match in _METRIC_FORMULA_TOKEN_RE.finditer(text):
        add("metric", match.group(0))
    for match in _STAT_FORMULA_RE.finditer(text):
        add("statistic", match.group(0))
    return out


def _target_contains_formula_invariant(target: str, invariant: dict[str, str]) -> bool:
    token = str(invariant.get("token") or "")
    if not token:
        return True
    if token in target:
        return True
    compact_target = re.sub(r"\s+", "", target)
    compact_token = re.sub(r"\s+", "", token)
    if compact_token and compact_token in compact_target:
        return True
    kind = str(invariant.get("kind") or "")
    if kind == "equation_label":
        fullwidth = compact_token.replace("(", "（").replace(")", "）")
        return fullwidth in compact_target
    if kind == "symbol":
        fullwidth_symbols = {
            "=": "＝",
            "\u2264": "≤",
            "\u2265": "≥",
            "\u2260": "≠",
            "\u00b1": "±",
            "\u2248": "≈",
        }
        return fullwidth_symbols.get(token, token) in target
    if kind in {"metric", "statistic"}:
        return compact_token.casefold() in compact_target.casefold()
    return False


def _table_figure_tokens(text: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for match in _TABLE_FIGURE_RE.finditer(text):
        raw_label = match.group("label")
        num = match.group("num")
        kind = "table" if raw_label.lower().startswith("table") else "figure"
        key = (kind, num)
        if key in seen:
            continue
        seen.add(key)
        out.append({"kind": kind, "num": num, "token": match.group(0)})
    return out


def _has_table_figure_token(target: str, token: dict[str, str]) -> bool:
    if re.search(re.escape(token["token"]), target, flags=re.I):
        return True
    num = re.escape(token["num"])
    if token["kind"] == "table":
        return bool(
            re.search(rf"\bTable\s*{num}\b", target, flags=re.I)
            or re.search(rf"表\s*{num}(?![0-9A-Za-z])", target)
        )
    return bool(
        re.search(rf"\bFig(?:ure)?\.?\s*{num}\b", target, flags=re.I)
        or re.search(rf"图\s*{num}(?![0-9A-Za-z])", target)
    )


def _markdown_separator_row(row: list[str]) -> bool:
    return bool(row) and all(cell.replace("-", "").replace(":", "").strip() == "" for cell in row)


def _markdown_tables(text: str) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    current_rows: list[list[str]] = []

    def flush() -> None:
        nonlocal current_rows
        if not current_rows:
            return
        data_rows = [row for row in current_rows if not _markdown_separator_row(row)]
        if data_rows:
            column_count = max(len(row) for row in data_rows)
            rows = [row + [""] * max(0, column_count - len(row)) for row in data_rows]
            tables.append(
                {
                    "rows": rows,
                    "row_count": len(rows),
                    "column_count": column_count,
                }
            )
        current_rows = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            current_rows.append(cells)
            continue
        flush()
    flush()
    return tables


def _markdown_table_shapes(text: str) -> list[dict[str, int]]:
    shapes: list[dict[str, int]] = []
    for table in _markdown_tables(text):
        shapes.append(
            {
                "row_count": int(table.get("row_count") or 0),
                "column_count": int(table.get("column_count") or 0),
            }
        )
    return shapes


def _document_table_invariants(doc_ir: DocumentIR | None, chunk: TextChunk) -> list[dict[str, Any]]:
    if doc_ir is None:
        return []
    block_ids = set(str(block_id) for block_id in getattr(chunk, "block_ids", []) if str(block_id))
    pages = {page + 1 for page in chunk.pages_0based}
    out: list[dict[str, Any]] = []
    for page in doc_ir.pages:
        if not block_ids and page.page_no not in pages:
            continue
        for block in page.blocks:
            if block.type != "table":
                continue
            if block_ids and block.block_id not in block_ids:
                continue
            table = block.meta.get("table") if isinstance(block.meta, dict) else None
            table = table if isinstance(table, dict) else {}
            row_count = int(table.get("row_count") or 0)
            column_count = int(table.get("column_count") or 0)
            if row_count < 2 or column_count < 2:
                continue
            out.append(
                {
                    "block_id": block.block_id,
                    "page_no": block.page_no,
                    "row_count": row_count,
                    "column_count": column_count,
                    "header": table.get("header") or [],
                    "numeric_tokens": table.get("numeric_tokens") or [],
                    "warnings": table.get("warnings") or [],
                    "confidence": table.get("confidence") or "low",
                }
            )
    return out


def _document_table_shape_errors(
    table_invariants: list[dict[str, Any]],
    target_tables: list[dict[str, int]],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if not table_invariants:
        return errors
    for idx, table in enumerate(table_invariants):
        expected = {
            "row_count": int(table.get("row_count") or 0),
            "column_count": int(table.get("column_count") or 0),
        }
        target = target_tables[idx] if idx < len(target_tables) else None
        if target == expected:
            continue
        errors.append(
            {
                "table_index": idx,
                "block_id": table.get("block_id"),
                "page_no": table.get("page_no"),
                "source": expected,
                "target": target,
                "reason": "missing_markdown_table" if target is None else "document_ir_table_shape_mismatch",
                "header": table.get("header") or [],
                "numeric_tokens": table.get("numeric_tokens") or [],
                "confidence": table.get("confidence") or "low",
            }
        )
    return errors


def _table_reconstruction_tables(
    table_reconstruction: dict[str, Any] | None,
    chunk: TextChunk,
) -> list[dict[str, Any]]:
    if not isinstance(table_reconstruction, dict):
        return []
    block_ids = set(str(block_id) for block_id in getattr(chunk, "block_ids", []) if str(block_id))
    pages = {page + 1 for page in chunk.pages_0based}
    out: list[dict[str, Any]] = []
    for table in table_reconstruction.get("tables") or []:
        if not isinstance(table, dict):
            continue
        table_id = str(table.get("block_id") or table.get("table_id") or "")
        page_no = int(table.get("page_no") or 0)
        if block_ids:
            if table_id not in block_ids:
                continue
        elif page_no not in pages:
            continue
        row_count = int(table.get("row_count") or 0)
        column_count = int(table.get("column_count") or 0)
        if row_count < 2 or column_count < 2:
            continue
        out.append(table)
    return out


def _table_locked_token_count(tables: list[dict[str, Any]]) -> int:
    count = 0
    for table in tables:
        for cell in table.get("cells") or []:
            if not isinstance(cell, dict):
                continue
            count += len([token for token in cell.get("locked_tokens") or [] if str(token)])
    return count


def _target_cell_text(target_table: dict[str, Any], row_index: int, column_index: int) -> str | None:
    rows = target_table.get("rows") or []
    if row_index < 0 or column_index < 0:
        return None
    if row_index >= len(rows):
        return None
    row = rows[row_index]
    if not isinstance(row, list) or column_index >= len(row):
        return None
    return str(row[column_index])


def _safe_nonnegative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _patch_cell_coord(cell: Any) -> tuple[int, int] | None:
    if not isinstance(cell, dict):
        return None
    row_index = _safe_nonnegative_int(cell.get("row_index"))
    column_index = _safe_nonnegative_int(cell.get("column_index"))
    if row_index is None or column_index is None:
        return None
    return row_index, column_index


def _matched_structure_patches(
    patches: list[dict[str, Any]],
    row_index: int,
    column_index: int,
) -> list[dict[str, Any]]:
    coord = (row_index, column_index)
    matched: list[dict[str, Any]] = []
    for patch in patches:
        anchor_coord = _patch_cell_coord(patch.get("anchor_cell"))
        covered_coords = [
            covered_coord
            for covered_coord in (_patch_cell_coord(cell) for cell in patch.get("covered_cells") or [])
            if covered_coord is not None
        ]
        if coord != anchor_coord and coord not in covered_coords:
            continue
        span = patch.get("span") if isinstance(patch.get("span"), dict) else {}
        bbox_evidence = patch.get("bbox_evidence") if isinstance(patch.get("bbox_evidence"), dict) else {}
        matched.append(
            {
                "patch_id": str(patch.get("patch_id") or ""),
                "source_review_id": str(patch.get("source_review_id") or ""),
                "operation": str(patch.get("operation") or patch.get("patch_type") or ""),
                "cell_role": "anchor" if coord == anchor_coord else "covered",
                "anchor_cell": patch.get("anchor_cell") if isinstance(patch.get("anchor_cell"), dict) else {},
                "span": span,
                "covered_cells": [
                    cell for cell in patch.get("covered_cells") or [] if isinstance(cell, dict)
                ][:20],
                "bbox_evidence_status": str(
                    patch.get("bbox_evidence_status")
                    or bbox_evidence.get("status")
                    or ""
                ),
            }
        )
    return matched


def _table_cell_token_errors(
    source_tables: list[dict[str, Any]],
    target_tables: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for table_index, table in enumerate(source_tables):
        target_table = target_tables[table_index] if table_index < len(target_tables) else None
        if not target_table:
            continue
        table_id = str(table.get("table_id") or table.get("block_id") or "")
        source_shape = {
            "row_count": int(table.get("row_count") or 0),
            "column_count": int(table.get("column_count") or 0),
        }
        merged_cell_candidates = [
            candidate
            for candidate in table.get("merged_cell_candidates") or []
            if isinstance(candidate, dict)
        ]
        table_structure_patches = [
            patch
            for patch in table.get("structure_patches") or []
            if isinstance(patch, dict)
        ]
        for cell in table.get("cells") or []:
            if not isinstance(cell, dict):
                continue
            locked_tokens = [str(token).strip() for token in cell.get("locked_tokens") or [] if str(token).strip()]
            if not locked_tokens:
                continue
            row_index = int(cell.get("row_index") or 0)
            column_index = int(cell.get("column_index") or 0)
            target_text = _target_cell_text(target_table, row_index, column_index)
            missing_tokens = (
                locked_tokens
                if target_text is None
                else [token for token in locked_tokens if token not in target_text]
            )
            if not missing_tokens:
                continue
            matched_structure_patches = _matched_structure_patches(
                table_structure_patches,
                row_index,
                column_index,
            )
            errors.append(
                {
                    "table_index": table_index,
                    "table_id": table_id,
                    "block_id": table.get("block_id") or table_id,
                    "page_no": table.get("page_no"),
                    "source_table_shape": source_shape,
                    "merged_cell_candidates": merged_cell_candidates[:20],
                    "merged_cell_candidate_count": len(merged_cell_candidates),
                    "table_structure_patches": table_structure_patches[:20],
                    "table_structure_patch_count": len(table_structure_patches),
                    "matched_structure_patches": matched_structure_patches[:20],
                    "matched_structure_patch_count": len(matched_structure_patches),
                    "row_index": row_index,
                    "column_index": column_index,
                    "role": cell.get("role") or "data",
                    "column_header": cell.get("column_header") or "",
                    "row_header": cell.get("row_header") or "",
                    "source_cell_text": cell.get("text") or "",
                    "target_cell_text": target_text,
                    "missing_tokens": missing_tokens[:20],
                    "reason": "missing_target_cell" if target_text is None else "missing_locked_tokens",
                }
            )
    return errors


def _document_blocks(doc_ir: DocumentIR | None) -> list[Any]:
    if doc_ir is None:
        return []
    return [block for page in doc_ir.pages for block in page.blocks]


def _footnote_anchor_markers(text: str) -> list[str]:
    match = _FOOTNOTE_ANCHOR_RE.match(str(text or ""))
    if not match:
        return []
    marker = str(match.group("marker") or "").strip()
    return [marker] if marker else []


def _has_footnote_marker(target: str, marker: str) -> bool:
    marker = str(marker or "").strip()
    if not marker:
        return True
    if marker in target:
        return True
    escaped = re.escape(marker)
    if marker.isdigit():
        return bool(re.search(rf"(^|[\s\[\(（]){escaped}(?=$|[\).、\]\）\s])", target))
    return False


def _target_contains_structure_anchor(target: str, anchor: dict[str, Any]) -> bool:
    kind = str(anchor.get("kind") or "")
    token = str(anchor.get("token") or "").strip()
    if not token:
        return True
    if kind in {"table", "figure"}:
        return _has_table_figure_token(target, {"kind": kind, "num": str(anchor.get("num") or ""), "token": token})
    if kind == "footnote_marker":
        return _has_footnote_marker(target, token)
    return bool(re.search(re.escape(token), target, flags=re.I))


def _document_structure_relation_checks(
    doc_ir: DocumentIR | None,
    chunk: TextChunk,
) -> list[dict[str, Any]]:
    blocks = _document_blocks(doc_ir)
    if not blocks:
        return []
    block_by_id = {str(block.block_id): block for block in blocks if str(getattr(block, "block_id", ""))}
    block_ids = {str(block_id) for block_id in getattr(chunk, "block_ids", []) if str(block_id)}
    pages = {page + 1 for page in chunk.pages_0based}
    checks: list[dict[str, Any]] = []
    for block in blocks:
        relation = str(getattr(block, "meta", {}).get("parent_relation") or "")
        parent_id = str(getattr(block, "parent_id", "") or "")
        if relation not in _STRUCTURE_RELATION_TYPES or not parent_id:
            continue
        parent = block_by_id.get(parent_id)
        if block_ids:
            if str(block.block_id) not in block_ids and parent_id not in block_ids:
                continue
        elif block.page_no not in pages and (parent is None or parent.page_no not in pages):
            continue

        anchors: list[dict[str, Any]] = []
        if block.type == "caption":
            anchors.extend(_table_figure_tokens(block.text))
        elif block.type == "footnote":
            anchors.extend(
                {"kind": "footnote_marker", "token": marker}
                for marker in _footnote_anchor_markers(block.text)
            )
        if not anchors:
            continue
        meta = getattr(block, "meta", {}) if isinstance(getattr(block, "meta", {}), dict) else {}
        checks.append(
            {
                "relation": relation,
                "child_block_id": str(block.block_id),
                "parent_block_id": parent_id,
                "child_type": str(block.type),
                "page_no": int(block.page_no or 0),
                "parent_page_no": int(meta.get("parent_page_no") or getattr(parent, "page_no", 0) or 0),
                "cross_page": bool(meta.get("cross_page_parent")),
                "anchors": anchors[:12],
                "source_text": str(block.text or "")[:200],
            }
        )
    return checks


def _structure_relation_mismatches(
    checks: list[dict[str, Any]],
    target_text: str,
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for check in checks:
        missing_anchors = [
            anchor
            for anchor in check.get("anchors") or []
            if isinstance(anchor, dict) and not _target_contains_structure_anchor(target_text, anchor)
        ]
        if not missing_anchors:
            continue
        item = dict(check)
        item["missing_anchors"] = missing_anchors[:12]
        mismatches.append(item)
    return mismatches


def _table_footnote_binding_checks(source_tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for table_index, table in enumerate(source_tables):
        table_id = str(table.get("table_id") or table.get("block_id") or "")
        for binding in table.get("footnote_bindings") or []:
            if not isinstance(binding, dict):
                continue
            if str(binding.get("status") or "") != "bound_to_cells":
                continue
            markers = [str(marker).strip() for marker in binding.get("markers") or [] if str(marker).strip()]
            matched_cells = [cell for cell in binding.get("matched_cells") or [] if isinstance(cell, dict)]
            if not markers or not matched_cells:
                continue
            checks.append(
                {
                    "table_index": table_index,
                    "table_id": table_id,
                    "block_id": str(table.get("block_id") or table_id),
                    "page_no": table.get("page_no"),
                    "footnote_block_id": str(binding.get("footnote_block_id") or ""),
                    "markers": markers[:12],
                    "source_text": str(binding.get("text") or "")[:200],
                    "matched_cells": matched_cells[:30],
                }
            )
    return checks


def _table_footnote_binding_mismatches(
    checks: list[dict[str, Any]],
    target_tables: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for check in checks:
        table_index = int(check.get("table_index") or 0)
        target_table = target_tables[table_index] if table_index < len(target_tables) else None
        missing_cells: list[dict[str, Any]] = []
        for cell in check.get("matched_cells") or []:
            if not isinstance(cell, dict):
                continue
            row_index = _safe_nonnegative_int(cell.get("row_index"))
            column_index = _safe_nonnegative_int(cell.get("column_index"))
            if row_index is None or column_index is None:
                continue
            target_cell_text = _target_cell_text(target_table or {}, row_index, column_index) if target_table else None
            markers = [
                str(marker).strip()
                for marker in cell.get("matched_markers") or check.get("markers") or []
                if str(marker).strip()
            ]
            missing_markers = [
                marker
                for marker in markers
                if target_cell_text is None or not _has_footnote_marker(target_cell_text, marker)
            ]
            if not missing_markers:
                continue
            missing_cells.append(
                {
                    "row_index": row_index,
                    "column_index": column_index,
                    "role": cell.get("role") or "data",
                    "column_header": cell.get("column_header") or "",
                    "row_header": cell.get("row_header") or "",
                    "source_cell_text": cell.get("text") or "",
                    "target_cell_text": target_cell_text,
                    "missing_markers": missing_markers[:12],
                }
            )
        if not missing_cells:
            continue
        item = dict(check)
        item["missing_cells"] = missing_cells[:30]
        mismatches.append(item)
    return mismatches


def _english_residual_ratio(text: str) -> float:
    body = re.sub(r"`[^`]*`", "", text)
    letters = len(re.findall(r"[A-Za-z]", body))
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", body))
    visible = letters + zh_chars
    if visible == 0:
        return 0.0
    return round(letters / visible, 4)


def _duplicate_paragraphs(text: str) -> list[str]:
    paras = [
        re.sub(r"\s+", " ", p).strip()
        for p in re.split(r"\n\s*\n", text)
        if len(re.sub(r"\s+", " ", p).strip()) >= 60
    ]
    counts = Counter(paras)
    return [para[:160] for para, count in counts.items() if count > 1]


def _chunk_translation_text(chunk_dir: Path, chunk_id: str) -> str | None:
    path = chunk_dir / f"{chunk_id}.md"
    if not path.is_file():
        return None
    return strip_yaml_front_matter(path.read_text(encoding="utf-8")).strip()


def _glossary_terms(glossary: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not glossary:
        return []
    out: list[dict[str, Any]] = []
    for term in glossary.get("terms") or []:
        if not isinstance(term, dict):
            continue
        en = str(term.get("en") or "").strip()
        zh = str(term.get("zh") or "").strip()
        if not en or not zh:
            continue
        out.append(
            {
                "en": en,
                "zh": zh,
                "first_page": term.get("first_page"),
                "source": term.get("source"),
                "status": term.get("status"),
            }
        )
    return out


def _glossary_conflicts(
    glossary: dict[str, Any] | None,
    pending_review: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    by_en: dict[str, dict[str, Any]] = {}
    for term in (glossary or {}).get("terms") or []:
        if not isinstance(term, dict):
            continue
        en = str(term.get("en") or "").strip()
        zh = str(term.get("zh") or "").strip()
        if not en or not zh:
            continue
        if str(term.get("status") or "").strip().lower() == "rejected":
            continue
        key = en.lower()
        entry = by_en.setdefault(
            key,
            {
                "en": en,
                "translations": [],
                "sources": [],
                "first_pages": [],
            },
        )
        if zh not in entry["translations"]:
            entry["translations"].append(zh)
        source = term.get("source")
        if source and source not in entry["sources"]:
            entry["sources"].append(source)
        first_page = term.get("first_page")
        if first_page is not None and first_page not in entry["first_pages"]:
            entry["first_pages"].append(first_page)
    for entry in by_en.values():
        if len(entry["translations"]) > 1:
            conflicts.append(entry)

    seen_pending: set[tuple[str, tuple[str, ...]]] = set()
    for item in (pending_review or {}).get("items") or []:
        if not isinstance(item, dict) or item.get("type") != "glossary_conflict":
            continue
        if str(item.get("status") or "pending").lower() not in {"pending", "open", ""}:
            continue
        en = str(item.get("en") or "").strip()
        existing = [str(v).strip() for v in item.get("existing_zh") or [] if str(v).strip()]
        candidate = str(item.get("candidate_zh") or "").strip()
        translations = existing + ([candidate] if candidate and candidate not in existing else [])
        if not en or len(translations) < 2:
            continue
        key = (en.lower(), tuple(sorted(translations)))
        if key in seen_pending:
            continue
        seen_pending.add(key)
        conflicts.append(
            {
                "en": en,
                "translations": translations,
                "first_pages": [item.get("first_page")] if item.get("first_page") is not None else [],
                "sources": [item.get("source")] if item.get("source") else [],
                "status": "pending_review",
            }
        )
    return conflicts


def _missing_glossary_terms(
    source: str,
    target: str,
    glossary_terms: list[dict[str, Any]],
    conflict_en: set[str],
) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for term in glossary_terms:
        en = str(term.get("en") or "").strip()
        zh = str(term.get("zh") or "").strip()
        if not en or not zh:
            continue
        if str(term.get("status") or "").strip().lower() == "rejected":
            continue
        if en.lower() in conflict_en:
            continue
        if not re.search(re.escape(en), source, flags=re.I):
            continue
        if zh in target:
            continue
        missing.append(
            {
                "en": en,
                "expected_zh": zh,
                "first_page": term.get("first_page"),
                "source": term.get("source"),
            }
        )
    return missing


def _missing_entity_tokens(source: str, target: str) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entity in extract_entity_candidates(source):
        entity_text = str(entity.get("text") or "").strip()
        entity_type = str(entity.get("type") or "unknown").strip() or "unknown"
        if not entity_text:
            continue
        key = (entity_text.casefold(), entity_type)
        if key in seen:
            continue
        seen.add(key)
        if re.search(re.escape(entity_text), target, flags=re.I):
            continue
        missing.append(
            {
                "text": entity_text,
                "type": entity_type,
                "confidence": str(entity.get("confidence") or "unknown"),
                "source": str(entity.get("source") or "unknown"),
            }
        )
    return missing


def _chunk_report(
    chunk: TextChunk,
    target_text: str | None,
    glossary_terms: list[dict[str, Any]],
    glossary_conflicts: list[dict[str, Any]],
    table_invariants: list[dict[str, Any]] | None = None,
    table_reconstruction_tables: list[dict[str, Any]] | None = None,
    structure_relation_checks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    source = _source_text_for_qa(chunk.text)
    pages_1based = [p + 1 for p in chunk.pages_0based]
    if target_text is None:
        return {
            "chunk_id": chunk.chunk_id,
            "pages_1based": pages_1based,
            "status": "missing_translation",
            "issues": [
                {
                    "type": "missing_translation",
                    "severity": "high",
                    "detail": "缺少该块译文文件。",
                }
            ],
            "metrics": {},
        }

    source_numbers = _numbers(source)
    missing_numbers = [token for token in source_numbers if token not in target_text]
    source_refs = _references(source)
    missing_refs = [token for token in source_refs if token not in target_text]
    table_figure_tokens = _table_figure_tokens(source)
    missing_table_figures = [
        token["token"] for token in table_figure_tokens if not _has_table_figure_token(target_text, token)
    ]
    source_symbols = _math_symbols(source)
    missing_symbols = [token for token in source_symbols if token not in target_text]
    source_formula_invariants = _formula_invariants(source)
    missing_formula_invariants = [
        item
        for item in source_formula_invariants
        if not _target_contains_formula_invariant(target_text, item)
    ]
    source_equation_label_count = sum(
        1 for item in source_formula_invariants if item.get("kind") == "equation_label"
    )
    missing_equation_label_count = sum(
        1 for item in missing_formula_invariants if item.get("kind") == "equation_label"
    )
    source_tables = _markdown_table_shapes(source)
    target_markdown_tables = _markdown_tables(target_text)
    target_tables = [
        {"row_count": int(table.get("row_count") or 0), "column_count": int(table.get("column_count") or 0)}
        for table in target_markdown_tables
    ]
    table_shape_errors: list[dict[str, Any]] = []
    for idx, source_shape in enumerate(source_tables):
        target_shape = target_tables[idx] if idx < len(target_tables) else None
        if target_shape != source_shape:
            table_shape_errors.append(
                {
                    "table_index": idx,
                    "source": source_shape,
                    "target": target_shape,
                }
            )
    if not source_tables:
        table_shape_errors.extend(_document_table_shape_errors(table_invariants or [], target_tables))
    table_cell_token_errors = _table_cell_token_errors(
        table_reconstruction_tables or [],
        target_markdown_tables,
    )
    missing_table_locked_token_count = sum(
        len(error.get("missing_tokens") or []) for error in table_cell_token_errors
    )
    structure_relation_checks = structure_relation_checks or []
    structure_relation_mismatches = _structure_relation_mismatches(
        structure_relation_checks,
        target_text,
    )
    table_footnote_binding_checks = _table_footnote_binding_checks(table_reconstruction_tables or [])
    table_footnote_binding_mismatches = _table_footnote_binding_mismatches(
        table_footnote_binding_checks,
        target_markdown_tables,
    )

    duplicates = _duplicate_paragraphs(target_text)
    english_ratio = _english_residual_ratio(target_text)
    source_conflicts = [
        conflict
        for conflict in glossary_conflicts
        if re.search(re.escape(str(conflict.get("en") or "")), source, flags=re.I)
    ]
    conflict_en = {str(conflict.get("en") or "").lower() for conflict in source_conflicts}
    missing_glossary = _missing_glossary_terms(source, target_text, glossary_terms, conflict_en)
    missing_entities = _missing_entity_tokens(source, target_text)

    issues: list[dict[str, Any]] = []
    if missing_numbers:
        issues.append({"type": "missing_numbers", "severity": "high", "tokens": missing_numbers[:80]})
    if missing_refs:
        issues.append({"type": "missing_references", "severity": "high", "tokens": missing_refs[:80]})
    if missing_table_figures:
        issues.append(
            {
                "type": "missing_table_figure_tokens",
                "severity": "medium",
                "tokens": missing_table_figures[:80],
            }
        )
    if missing_symbols:
        issues.append({"type": "missing_math_symbols", "severity": "medium", "tokens": missing_symbols[:80]})
    if missing_formula_invariants:
        high_risk_kinds = {"equation_label", "variable", "statistic"}
        severity = (
            "high"
            if any(item.get("kind") in high_risk_kinds for item in missing_formula_invariants)
            else "medium"
        )
        issues.append(
            {
                "type": "formula_mismatch",
                "severity": severity,
                "formulas": missing_formula_invariants[:80],
            }
        )
    if table_shape_errors:
        issues.append({"type": "table_shape_mismatch", "severity": "high", "tables": table_shape_errors})
    if table_cell_token_errors:
        issues.append(
            {
                "type": "table_cell_token_mismatch",
                "severity": "high",
                "cells": table_cell_token_errors[:80],
            }
        )
    if table_footnote_binding_mismatches:
        issues.append(
            {
                "type": "table_footnote_binding_mismatch",
                "severity": "high",
                "bindings": table_footnote_binding_mismatches[:80],
            }
        )
    if structure_relation_mismatches:
        issues.append(
            {
                "type": "caption_or_footnote_relation_mismatch",
                "severity": "medium",
                "relations": structure_relation_mismatches[:80],
            }
        )
    if duplicates:
        issues.append({"type": "duplicate_paragraphs", "severity": "medium", "samples": duplicates[:5]})
    if english_ratio >= 0.45:
        issues.append({"type": "high_english_residual", "severity": "low", "ratio": english_ratio})
    if missing_glossary:
        issues.append(
            {
                "type": "missing_glossary_terms",
                "severity": "medium",
                "terms": missing_glossary[:80],
            }
        )
    if missing_entities:
        severity = (
            "medium"
            if any(entity.get("type") in _ENTITY_MEDIUM_SEVERITY_TYPES for entity in missing_entities)
            else "low"
        )
        issues.append(
            {
                "type": "missing_entity_tokens",
                "severity": severity,
                "entities": missing_entities[:80],
            }
        )
    if source_conflicts:
        issues.append(
            {
                "type": "glossary_translation_conflict",
                "severity": "medium",
                "conflicts": source_conflicts[:40],
            }
        )

    return {
        "chunk_id": chunk.chunk_id,
        "pages_1based": pages_1based,
        "status": "ok" if not issues else "issues",
        "issues": issues,
        "metrics": {
            "source_number_count": len(source_numbers),
            "missing_number_count": len(missing_numbers),
            "source_reference_count": len(source_refs),
            "missing_reference_count": len(missing_refs),
            "source_formula_token_count": len(source_formula_invariants),
            "missing_formula_token_count": len(missing_formula_invariants),
            "source_equation_label_count": source_equation_label_count,
            "missing_equation_label_count": missing_equation_label_count,
            "source_table_count": len(source_tables) or len(table_invariants or []),
            "source_table_ids": [
                str(table.get("block_id"))
                for table in (table_invariants or [])
                if str(table.get("block_id") or "")
            ]
            or [
                str(table.get("block_id") or table.get("table_id"))
                for table in (table_reconstruction_tables or [])
                if str(table.get("block_id") or table.get("table_id") or "")
            ],
            "table_shape_error_count": len(table_shape_errors),
            "source_table_locked_token_count": _table_locked_token_count(table_reconstruction_tables or []),
            "table_cell_token_error_count": len(table_cell_token_errors),
            "missing_table_locked_token_count": missing_table_locked_token_count,
            "structure_relation_check_count": len(structure_relation_checks),
            "structure_relation_mismatch_count": len(structure_relation_mismatches),
            "structure_relation_missing_anchor_count": sum(
                len(item.get("missing_anchors") or []) for item in structure_relation_mismatches
            ),
            "table_footnote_binding_check_count": len(table_footnote_binding_checks),
            "table_footnote_binding_mismatch_count": len(table_footnote_binding_mismatches),
            "table_footnote_binding_missing_cell_count": sum(
                len(item.get("missing_cells") or []) for item in table_footnote_binding_mismatches
            ),
            "english_residual_ratio": english_ratio,
            "duplicate_paragraph_count": len(duplicates),
            "missing_glossary_term_count": len(missing_glossary),
            "glossary_conflict_count": len(source_conflicts),
            "source_entity_candidate_count": len(extract_entity_candidates(source)),
            "missing_entity_token_count": len(missing_entities),
        },
    }


def build_translation_qa(
    chunks: list[TextChunk],
    chunk_dir: Path,
    *,
    glossary: dict[str, Any] | None = None,
    pending_review: dict[str, Any] | None = None,
    document_ir: DocumentIR | None = None,
    table_reconstruction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    terms = _glossary_terms(glossary)
    conflicts = _glossary_conflicts(glossary, pending_review)
    reports = [
        _chunk_report(
            chunk,
            _chunk_translation_text(chunk_dir, chunk.chunk_id),
            terms,
            conflicts,
            table_invariants=_document_table_invariants(document_ir, chunk),
            table_reconstruction_tables=_table_reconstruction_tables(table_reconstruction, chunk),
            structure_relation_checks=_document_structure_relation_checks(document_ir, chunk),
        )
        for chunk in chunks
    ]
    issue_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    translated_count = 0
    entity_candidate_count = 0
    missing_entity_count = 0
    source_table_count_without_ids = 0
    source_table_ids: set[str] = set()
    table_shape_error_count = 0
    source_table_locked_token_count = 0
    table_cell_token_error_count = 0
    missing_table_locked_token_count = 0
    structure_relation_check_count = 0
    structure_relation_mismatch_count = 0
    structure_relation_missing_anchor_count = 0
    table_footnote_binding_check_count = 0
    table_footnote_binding_mismatch_count = 0
    table_footnote_binding_missing_cell_count = 0
    source_formula_token_count = 0
    missing_formula_token_count = 0
    source_equation_label_count = 0
    missing_equation_label_count = 0
    for report in reports:
        if report["status"] != "missing_translation":
            translated_count += 1
        metrics = report.get("metrics") or {}
        entity_candidate_count += int(metrics.get("source_entity_candidate_count") or 0)
        missing_entity_count += int(metrics.get("missing_entity_token_count") or 0)
        source_formula_token_count += int(metrics.get("source_formula_token_count") or 0)
        missing_formula_token_count += int(metrics.get("missing_formula_token_count") or 0)
        source_equation_label_count += int(metrics.get("source_equation_label_count") or 0)
        missing_equation_label_count += int(metrics.get("missing_equation_label_count") or 0)
        table_ids = [str(item) for item in metrics.get("source_table_ids") or [] if str(item)]
        if table_ids:
            source_table_ids.update(table_ids)
        else:
            source_table_count_without_ids += int(metrics.get("source_table_count") or 0)
        table_shape_error_count += int(metrics.get("table_shape_error_count") or 0)
        source_table_locked_token_count += int(metrics.get("source_table_locked_token_count") or 0)
        table_cell_token_error_count += int(metrics.get("table_cell_token_error_count") or 0)
        missing_table_locked_token_count += int(metrics.get("missing_table_locked_token_count") or 0)
        structure_relation_check_count += int(metrics.get("structure_relation_check_count") or 0)
        structure_relation_mismatch_count += int(metrics.get("structure_relation_mismatch_count") or 0)
        structure_relation_missing_anchor_count += int(metrics.get("structure_relation_missing_anchor_count") or 0)
        table_footnote_binding_check_count += int(metrics.get("table_footnote_binding_check_count") or 0)
        table_footnote_binding_mismatch_count += int(
            metrics.get("table_footnote_binding_mismatch_count") or 0
        )
        table_footnote_binding_missing_cell_count += int(
            metrics.get("table_footnote_binding_missing_cell_count") or 0
        )
        for issue in report["issues"]:
            issue_counts[issue["type"]] += 1
            severity_counts[issue["severity"]] += 1

    max_english_ratio = max(
        (r.get("metrics", {}).get("english_residual_ratio", 0.0) for r in reports),
        default=0.0,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "summary": {
            "chunk_count": len(chunks),
            "translated_chunk_count": translated_count,
            "glossary_term_count": len(terms),
            "glossary_conflict_count": len(conflicts),
            "entity_candidate_count": entity_candidate_count,
            "missing_entity_token_count": missing_entity_count,
            "source_formula_token_count": source_formula_token_count,
            "missing_formula_token_count": missing_formula_token_count,
            "source_equation_label_count": source_equation_label_count,
            "missing_equation_label_count": missing_equation_label_count,
            "source_table_count": len(source_table_ids) + source_table_count_without_ids,
            "table_shape_error_count": table_shape_error_count,
            "source_table_locked_token_count": source_table_locked_token_count,
            "table_cell_token_error_count": table_cell_token_error_count,
            "missing_table_locked_token_count": missing_table_locked_token_count,
            "structure_relation_check_count": structure_relation_check_count,
            "structure_relation_mismatch_count": structure_relation_mismatch_count,
            "structure_relation_missing_anchor_count": structure_relation_missing_anchor_count,
            "table_footnote_binding_check_count": table_footnote_binding_check_count,
            "table_footnote_binding_mismatch_count": table_footnote_binding_mismatch_count,
            "table_footnote_binding_missing_cell_count": table_footnote_binding_missing_cell_count,
            "issue_count": sum(issue_counts.values()),
            "issue_counts": dict(issue_counts),
            "severity_counts": dict(severity_counts),
            "max_english_residual_ratio": max_english_ratio,
        },
        "glossary_conflicts": conflicts,
        "chunks": reports,
    }


def translation_qa_to_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# 翻译 QA 报告",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| 块总数 | {summary.get('chunk_count', 0)} |",
        f"| 已有译文块 | {summary.get('translated_chunk_count', 0)} |",
        f"| 术语库条目 | {summary.get('glossary_term_count', 0)} |",
        f"| 术语冲突 | {summary.get('glossary_conflict_count', 0)} |",
        f"| 实体候选 | {summary.get('entity_candidate_count', 0)} |",
        f"| 缺失实体 | {summary.get('missing_entity_token_count', 0)} |",
        f"| 公式不变量 | {summary.get('source_formula_token_count', 0)} |",
        f"| 缺失公式不变量 | {summary.get('missing_formula_token_count', 0)} |",
        f"| 公式编号 | {summary.get('source_equation_label_count', 0)} |",
        f"| 缺失公式编号 | {summary.get('missing_equation_label_count', 0)} |",
        f"| 源表格 | {summary.get('source_table_count', 0)} |",
        f"| 表格形状异常 | {summary.get('table_shape_error_count', 0)} |",
        f"| 表格单元格 token 异常 | {summary.get('table_cell_token_error_count', 0)} |",
        f"| 缺失表格锁定 token | {summary.get('missing_table_locked_token_count', 0)} |",
        f"| 结构关系检查 | {summary.get('structure_relation_check_count', 0)} |",
        f"| 结构关系异常 | {summary.get('structure_relation_mismatch_count', 0)} |",
        f"| 表格脚注绑定检查 | {summary.get('table_footnote_binding_check_count', 0)} |",
        f"| 表格脚注绑定异常 | {summary.get('table_footnote_binding_mismatch_count', 0)} |",
        f"| 问题总数 | {summary.get('issue_count', 0)} |",
        f"| 最高英文残留比例 | {summary.get('max_english_residual_ratio', 0)} |",
        "",
        "## 问题分布",
        "",
    ]
    issue_counts = summary.get("issue_counts") or {}
    if issue_counts:
        lines.extend(["| 类型 | 数量 |", "| --- | --- |"])
        for issue_type, count in sorted(issue_counts.items()):
            lines.append(f"| `{issue_type}` | {count} |")
    else:
        lines.append("未发现规则 QA 问题。")

    lines.extend(["", "## 块级明细", ""])
    for chunk in report.get("chunks", []):
        if not chunk.get("issues"):
            continue
        pages = chunk.get("pages_1based") or []
        page_text = f"{pages[0]}-{pages[-1]}" if pages else "-"
        lines.append(f"### {chunk.get('chunk_id')}（页 {page_text}）")
        for issue in chunk.get("issues", []):
            issue_type = issue.get("type")
            severity = issue.get("severity")
            detail = issue.get("detail")
            if detail:
                lines.append(f"- `{severity}` `{issue_type}`：{detail}")
            elif "tokens" in issue:
                tokens = ", ".join(str(token) for token in issue.get("tokens", [])[:20])
                lines.append(f"- `{severity}` `{issue_type}`：{tokens}")
            elif "terms" in issue:
                terms = ", ".join(
                    f"{term.get('en')} -> {term.get('expected_zh')}"
                    for term in issue.get("terms", [])[:20]
                    if isinstance(term, dict)
                )
                lines.append(f"- `{severity}` `{issue_type}`：{terms}")
            elif "entities" in issue:
                entities = ", ".join(
                    f"{entity.get('text')} ({entity.get('type')})"
                    for entity in issue.get("entities", [])[:20]
                    if isinstance(entity, dict)
                )
                lines.append(f"- `{severity}` `{issue_type}`：{entities}")
            elif "formulas" in issue:
                formulas = ", ".join(
                    f"{item.get('kind')}:{item.get('token')}"
                    for item in issue.get("formulas", [])[:20]
                    if isinstance(item, dict)
                )
                lines.append(f"- `{severity}` `{issue_type}`：{formulas}")
            elif "conflicts" in issue:
                conflicts = ", ".join(
                    f"{conflict.get('en')} -> {' / '.join(str(v) for v in conflict.get('translations', []))}"
                    for conflict in issue.get("conflicts", [])[:20]
                    if isinstance(conflict, dict)
                )
                lines.append(f"- `{severity}` `{issue_type}`：{conflicts}")
            elif "ratio" in issue:
                lines.append(f"- `{severity}` `{issue_type}`：{issue.get('ratio')}")
            elif "tables" in issue:
                lines.append(f"- `{severity}` `{issue_type}`：{json.dumps(issue.get('tables'), ensure_ascii=False)}")
            elif "cells" in issue:
                lines.append(f"- `{severity}` `{issue_type}`：{json.dumps(issue.get('cells'), ensure_ascii=False)}")
            elif "bindings" in issue:
                lines.append(
                    f"- `{severity}` `{issue_type}`：{json.dumps(issue.get('bindings'), ensure_ascii=False)}"
                )
            elif "relations" in issue:
                lines.append(
                    f"- `{severity}` `{issue_type}`：{json.dumps(issue.get('relations'), ensure_ascii=False)}"
                )
            elif "samples" in issue:
                lines.append(f"- `{severity}` `{issue_type}`：{json.dumps(issue.get('samples'), ensure_ascii=False)}")
            else:
                lines.append(f"- `{severity}` `{issue_type}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_translation_qa(
    chunks: list[TextChunk],
    chunk_dir: Path,
    json_path: Path,
    markdown_path: Path,
    *,
    glossary: dict[str, Any] | None = None,
    pending_review: dict[str, Any] | None = None,
    document_ir: DocumentIR | None = None,
    table_reconstruction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = build_translation_qa(
        chunks,
        chunk_dir,
        glossary=glossary,
        pending_review=pending_review,
        document_ir=document_ir,
        table_reconstruction=table_reconstruction,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(translation_qa_to_markdown(report), encoding="utf-8")
    return report
