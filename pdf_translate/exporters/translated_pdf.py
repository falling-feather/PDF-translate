from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fitz

from pdf_translate.chunking import TextChunk
from pdf_translate.deferral_markers import (
    finalize_merged_translation_markdown,
    strip_yaml_front_matter,
)

SCHEMA_VERSION = "translated-pdf-report-v1"
PAGE_WIDTH = 595.0
PAGE_HEIGHT = 842.0
MARGIN_X = 48.0
MARGIN_TOP = 54.0
MARGIN_BOTTOM = 46.0
FONT_CJK = "china-s"
FONT_LATIN = "helv"
STRUCTURE_QA_SUMMARY_FIELDS = [
    "caption_count",
    "caption_linked_count",
    "caption_orphan_count",
    "footnote_count",
    "footnote_linked_count",
    "footnote_orphan_count",
    "table_footnote_count",
    "cross_page_relationship_count",
    "caption_cross_page_linked_count",
    "caption_cross_page_orphan_count",
    "footnote_cross_page_linked_count",
    "footnote_cross_page_orphan_count",
    "cross_page_parent_gap_max",
]
TABLE_RECONSTRUCTION_SUMMARY_FIELDS = [
    "table_footnote_binding_count",
    "table_footnote_cell_binding_count",
    "table_footnote_bound_cell_count",
    "table_footnote_unbound_count",
    "table_footnote_table_level_count",
    "effective_merged_cell_candidate_count",
    "confirmed_merged_cell_candidate_count",
    "rejected_merged_cell_candidate_count",
    "needs_revision_merged_cell_candidate_count",
    "pending_merged_cell_candidate_count",
    "tables_with_confirmed_merged_cells",
    "table_structure_patch_count",
    "table_structure_patch_applied_count",
    "table_structure_patch_table_count",
    "table_structure_patch_cell_count",
    "table_structure_patch_covered_cell_count",
]


def _chunk_translation_text(chunk_dir: Path, chunk_id: str) -> str:
    path = chunk_dir / f"{chunk_id}.md"
    if not path.is_file():
        return ""
    body = strip_yaml_front_matter(path.read_text(encoding="utf-8")).strip()
    return finalize_merged_translation_markdown(body).strip()


def _index_by_chunk(items: list[dict[str, Any]], key: str = "chunk_id") -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        chunk_id = item.get(key)
        if isinstance(chunk_id, str) and chunk_id:
            out.setdefault(chunk_id, []).append(item)
    return out


def _qa_issues_by_chunk(qa_report: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    if not qa_report:
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for chunk in qa_report.get("chunks") or []:
        chunk_id = chunk.get("chunk_id")
        issues = chunk.get("issues") or []
        if isinstance(chunk_id, str) and issues:
            out[chunk_id] = list(issues)
    return out


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def _is_separator_row(row: list[str]) -> bool:
    if not row:
        return False
    return all(cell.replace("-", "").replace(":", "").strip() == "" for cell in row)


def _parse_table(lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if _is_separator_row(cells):
            continue
        rows.append(cells)
    if not rows:
        return []
    column_count = max(len(row) for row in rows)
    return [row + [""] * (column_count - len(row)) for row in rows]


def _markdown_blocks(text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    paragraph: list[str] = []
    table: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            blocks.append({"type": "paragraph", "text": "\n".join(paragraph).strip()})
            paragraph = []

    def flush_table() -> None:
        nonlocal table
        if table:
            rows = _parse_table(table)
            if rows:
                blocks.append({"type": "table", "rows": rows})
            table = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if _is_table_line(stripped):
            flush_paragraph()
            table.append(stripped)
            continue
        flush_table()
        if not stripped:
            flush_paragraph()
            continue
        if stripped.startswith("#"):
            flush_paragraph()
            hashes = len(stripped) - len(stripped.lstrip("#"))
            blocks.append(
                {
                    "type": "heading",
                    "level": max(1, min(hashes, 4)),
                    "text": stripped[hashes:].strip() or stripped,
                }
            )
            continue
        paragraph.append(stripped)

    flush_table()
    flush_paragraph()
    return blocks


def _clean_text(text: str) -> str:
    return " ".join(str(text or "").replace("\r", "\n").replace("\t", " ").split())


def _short_item(item: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if value:
            return _clean_text(str(value))[:96]
    return "unknown"


def _summary(report: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    summary = report.get("summary")
    return summary if isinstance(summary, dict) else {}


def _table_reconstruction_source(table_reconstruction: dict[str, Any] | None) -> str:
    if not isinstance(table_reconstruction, dict):
        return "missing"
    summary = table_reconstruction.get("summary") if isinstance(table_reconstruction.get("summary"), dict) else {}
    source = str(summary.get("table_structure_source") or "").strip()
    if source:
        return source
    if str(table_reconstruction.get("confirmation_schema_version") or ""):
        return "confirmed"
    return "source"


def _int_summary_fields(summary: dict[str, Any], fields: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for field in fields:
        value = summary.get(field, 0)
        if isinstance(value, bool):
            out[field] = int(value)
        elif isinstance(value, (int, float)):
            out[field] = int(value)
        elif isinstance(value, str):
            try:
                out[field] = int(float(value))
            except ValueError:
                out[field] = 0
        else:
            out[field] = 0
    return out


def _chunk_block_ids(chunk: TextChunk) -> set[str]:
    return {str(block_id) for block_id in getattr(chunk, "block_ids", []) if str(block_id)}


def _chunk_pages_1based(chunk: TextChunk) -> list[int]:
    return [page + 1 for page in chunk.pages_0based]


def _table_matches_chunk(table: dict[str, Any], chunk: TextChunk, block_ids: set[str]) -> bool:
    table_id = str(table.get("block_id") or table.get("table_id") or "")
    if block_ids:
        return table_id in block_ids
    pages = set(_chunk_pages_1based(chunk))
    try:
        page_no = int(table.get("page_no") or 0)
    except (TypeError, ValueError):
        page_no = 0
    return page_no in pages


def _tables_for_chunk(chunk: TextChunk, table_reconstruction: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(table_reconstruction, dict):
        return []
    tables = [table for table in table_reconstruction.get("tables") or [] if isinstance(table, dict)]
    block_ids = _chunk_block_ids(chunk)
    return [table for table in tables if _table_matches_chunk(table, chunk, block_ids)]


def _renderable_tables_for_chunk(
    chunk: TextChunk,
    table_reconstruction: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if _table_reconstruction_source(table_reconstruction) != "confirmed":
        return []
    return _tables_for_chunk(chunk, table_reconstruction)


def _safe_positive_int(value: Any, default: int = 1) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return default
    return max(default, out)


def _safe_zero_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def _table_patch_maps(
    rows: list[list[str]],
    patches: list[dict[str, Any]],
) -> tuple[dict[tuple[int, int], dict[str, Any]], set[tuple[int, int]]]:
    anchors: dict[tuple[int, int], dict[str, Any]] = {}
    covered: set[tuple[int, int]] = set()
    row_count = len(rows)
    if not row_count:
        return anchors, covered
    column_count = max(1, max(len(row) for row in rows))
    for patch in patches:
        if not isinstance(patch, dict) or patch.get("applied") is False:
            continue
        anchor = patch.get("anchor_cell") if isinstance(patch.get("anchor_cell"), dict) else {}
        row_index = _safe_zero_int(anchor.get("row_index"))
        column_index = _safe_zero_int(anchor.get("column_index"))
        if row_index is None or column_index is None:
            continue
        if row_index >= row_count or column_index >= column_count:
            continue
        span = patch.get("span") if isinstance(patch.get("span"), dict) else {}
        row_span = min(_safe_positive_int(span.get("row_span")), row_count - row_index)
        column_span = min(_safe_positive_int(span.get("column_span")), column_count - column_index)
        if row_span <= 1 and column_span <= 1:
            continue
        if (row_index, column_index) in anchors:
            continue
        anchors[(row_index, column_index)] = {
            "row_span": row_span,
            "column_span": column_span,
            "patch_id": str(patch.get("patch_id") or ""),
        }
        for row_offset in range(row_span):
            current_row = row_index + row_offset
            if current_row >= row_count:
                continue
            for col_offset in range(column_span):
                current_col = column_index + col_offset
                if current_col >= column_count or (current_row, current_col) == (row_index, column_index):
                    continue
                covered.add((current_row, current_col))
        for cell in patch.get("covered_cells") or []:
            if not isinstance(cell, dict):
                continue
            covered_row = _safe_zero_int(cell.get("row_index"))
            covered_col = _safe_zero_int(cell.get("column_index"))
            if covered_row is None or covered_col is None:
                continue
            if covered_row < row_count and covered_col < column_count:
                covered.add((covered_row, covered_col))
    return anchors, covered


def _table_structure_context(chunk: TextChunk, table_reconstruction: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(table_reconstruction, dict):
        table_reconstruction = {}
    groups = [group for group in table_reconstruction.get("continued_table_groups") or [] if isinstance(group, dict)]
    selected = _tables_for_chunk(chunk, table_reconstruction)
    table_ids = [
        str(table.get("table_id") or table.get("block_id") or "")
        for table in selected
        if str(table.get("table_id") or table.get("block_id") or "")
    ]
    table_id_set = set(table_ids)
    selected_groups = [
        group
        for group in groups
        if table_id_set.intersection({str(table_id) for table_id in group.get("table_ids") or [] if str(table_id)})
    ]
    footnote_bindings = [
        binding
        for table in selected
        for binding in table.get("footnote_bindings") or []
        if isinstance(binding, dict)
    ]
    structural_relation_ids = [
        str(relation_id)
        for relation_id in getattr(chunk, "structural_relation_ids", [])
        if str(relation_id)
    ]
    return {
        "source_table_ids": table_ids,
        "source_table_count": len(table_ids),
        "source_caption_count": sum(len(table.get("caption_blocks") or []) for table in selected),
        "source_footnote_count": sum(len(table.get("footnote_blocks") or []) for table in selected),
        "source_footnote_cell_binding_count": sum(
            1 for binding in footnote_bindings if str(binding.get("status") or "") == "bound_to_cells"
        ),
        "source_footnote_bound_cell_count": sum(
            int(binding.get("matched_cell_count") or 0) for binding in footnote_bindings
        ),
        "merged_cell_candidate_reference_count": sum(
            len(table.get("merged_cell_candidates") or []) for table in selected
        ),
        "confirmed_merged_cell_candidate_reference_count": sum(
            len(table.get("confirmed_merged_cell_candidates") or []) for table in selected
        ),
        "table_structure_patch_reference_count": sum(
            len(table.get("structure_patches") or []) for table in selected
        ),
        "table_structure_patch_covered_cell_reference_count": sum(
            len(patch.get("covered_cells") or [])
            for table in selected
            for patch in table.get("structure_patches") or []
            if isinstance(patch, dict)
        ),
        "continued_table_group_ids": [
            str(group.get("group_id") or "") for group in selected_groups if str(group.get("group_id") or "")
        ],
        "continued_table_group_count": len(selected_groups),
        "structural_relation_ids": structural_relation_ids,
        "structural_relation_count": len(structural_relation_ids),
    }


def _structure_context_lines(chunk_report: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    table_ids = [str(item) for item in chunk_report.get("source_table_ids") or [] if str(item)]
    if table_ids:
        lines.append(
            "Source tables: "
            + ", ".join(table_ids[:6])
            + (f" (+{len(table_ids) - 6})" if len(table_ids) > 6 else "")
        )
    caption_count = int(chunk_report.get("source_caption_count") or 0)
    footnote_count = int(chunk_report.get("source_footnote_count") or 0)
    if caption_count or footnote_count:
        lines.append(f"Caption/footnote links: captions={caption_count}, footnotes={footnote_count}")
    cell_binding_count = int(chunk_report.get("source_footnote_cell_binding_count") or 0)
    bound_cell_count = int(chunk_report.get("source_footnote_bound_cell_count") or 0)
    if cell_binding_count or bound_cell_count:
        lines.append(f"Footnote cell bindings: bindings={cell_binding_count}, cells={bound_cell_count}")
    merged_candidate_count = int(chunk_report.get("merged_cell_candidate_reference_count") or 0)
    if merged_candidate_count:
        lines.append(f"Merged-cell candidates: {merged_candidate_count}")
    confirmed_candidate_count = int(chunk_report.get("confirmed_merged_cell_candidate_reference_count") or 0)
    if confirmed_candidate_count:
        lines.append(f"Confirmed merged-cell candidates: {confirmed_candidate_count}")
    patch_count = int(chunk_report.get("table_structure_patch_reference_count") or 0)
    if patch_count:
        covered_count = int(chunk_report.get("table_structure_patch_covered_cell_reference_count") or 0)
        lines.append(f"Table structure patches: {patch_count}, covered cells={covered_count}")
    group_ids = [str(item) for item in chunk_report.get("continued_table_group_ids") or [] if str(item)]
    if group_ids:
        lines.append(
            "Continued table groups: "
            + ", ".join(group_ids[:6])
            + (f" (+{len(group_ids) - 6})" if len(group_ids) > 6 else "")
        )
    relation_ids = [str(item) for item in chunk_report.get("structural_relation_ids") or [] if str(item)]
    if relation_ids:
        lines.append(
            "Protected structure relations: "
            + ", ".join(relation_ids[:4])
            + (f" (+{len(relation_ids) - 4})" if len(relation_ids) > 4 else "")
        )
    return lines


class _PdfFlow:
    def __init__(self, title: str) -> None:
        self.doc = fitz.open()
        self.title = title
        self.font = fitz.Font(fontname=FONT_CJK)
        self.page: fitz.Page | None = None
        self.y = MARGIN_TOP
        self.new_page()

    @property
    def content_right(self) -> float:
        return PAGE_WIDTH - MARGIN_X

    @property
    def content_bottom(self) -> float:
        return PAGE_HEIGHT - MARGIN_BOTTOM

    def new_page(self) -> None:
        self.page = self.doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        self.y = MARGIN_TOP

    def ensure_space(self, height: float) -> None:
        if self.page is None or self.y + height > self.content_bottom:
            self.new_page()

    def text_width(self, text: str, fontsize: float) -> float:
        return self.font.text_length(text, fontsize=fontsize)

    def wrap_text(self, text: str, fontsize: float, width: float) -> list[str]:
        out: list[str] = []
        for raw in str(text or "").splitlines() or [""]:
            line = ""
            for char in raw:
                char = " " if char == "\t" else char
                candidate = line + char
                if line and self.text_width(candidate, fontsize) > width:
                    out.append(line.rstrip() or " ")
                    line = "" if char.isspace() else char
                else:
                    line = candidate
            out.append(line.rstrip() or " ")
        return out

    def add_text(
        self,
        text: str,
        *,
        fontsize: float = 10.5,
        color: tuple[float, float, float] = (0.12, 0.16, 0.22),
        before: float = 0,
        after: float = 6,
        left: float = MARGIN_X,
        width: float | None = None,
        leading: float | None = None,
    ) -> None:
        if self.page is None:
            self.new_page()
        if before:
            self.y += before
        width = width if width is not None else self.content_right - left
        leading = leading if leading is not None else fontsize * 1.45
        for line in self.wrap_text(text, fontsize, width):
            self.ensure_space(leading + after)
            assert self.page is not None
            self.page.insert_text(
                (left, self.y),
                line,
                fontname=FONT_CJK,
                fontsize=fontsize,
                color=color,
            )
            self.y += leading
        self.y += after

    def add_rule(self) -> None:
        self.ensure_space(10)
        assert self.page is not None
        self.page.draw_line(
            (MARGIN_X, self.y),
            (self.content_right, self.y),
            color=(0.78, 0.82, 0.88),
            width=0.6,
        )
        self.y += 10

    def add_table(self, rows: list[list[str]], table_context: dict[str, Any] | None = None) -> int:
        if not rows:
            return 0
        column_count = max(1, max(len(row) for row in rows))
        normalized_rows = [row + [""] * (column_count - len(row)) for row in rows]
        patches = [
            patch
            for patch in (table_context or {}).get("structure_patches") or []
            if isinstance(patch, dict)
        ]
        anchors, covered = _table_patch_maps(normalized_rows, patches)
        available = self.content_right - MARGIN_X
        col_width = available / column_count
        fontsize = 8.5 if column_count > 4 else 9.2
        leading = fontsize * 1.28
        pad_x = 4.0
        pad_y = 5.0
        max_row_lines = max(2, int((self.content_bottom - MARGIN_TOP - 12) / leading))
        wrapped_cells: dict[tuple[int, int], list[str]] = {}
        row_heights: list[float] = []
        for row_index, row in enumerate(normalized_rows):
            row_line_counts: list[int] = []
            for column_index, cell in enumerate(row):
                if (row_index, column_index) in covered:
                    continue
                span = anchors.get((row_index, column_index), {})
                column_span = min(int(span.get("column_span") or 1), column_count - column_index)
                wrap_width = max(20.0, col_width * column_span - pad_x * 2)
                lines = self.wrap_text(cell, fontsize, wrap_width)
                if len(lines) > max_row_lines:
                    lines = lines[: max_row_lines - 1] + ["..."]
                wrapped_cells[(row_index, column_index)] = lines
                row_line_counts.append(max(1, len(lines)))
            row_heights.append((max(row_line_counts) if row_line_counts else 1) * leading + pad_y * 2)
        self.y += 2
        for row_index, row_height in enumerate(row_heights):
            span_end = row_index + 1
            for (anchor_row, _anchor_col), span in anchors.items():
                if anchor_row == row_index:
                    span_end = max(span_end, row_index + int(span.get("row_span") or 1))
            span_end = min(span_end, len(row_heights))
            remaining_height = sum(row_heights[row_index:span_end])
            self.ensure_space(remaining_height + 8)
            assert self.page is not None
            x = MARGIN_X
            fill = (0.93, 0.96, 0.97) if row_index == 0 else None
            column_index = 0
            while column_index < column_count:
                if (row_index, column_index) in covered:
                    x += col_width
                    column_index += 1
                    continue
                span = anchors.get((row_index, column_index), {})
                column_span = min(int(span.get("column_span") or 1), column_count - column_index)
                row_span = min(int(span.get("row_span") or 1), len(row_heights) - row_index)
                cell_width = col_width * column_span
                cell_height = sum(row_heights[row_index : row_index + row_span])
                cell_fill = (0.88, 0.97, 0.94) if span else fill
                rect = fitz.Rect(x, self.y, x + cell_width, self.y + cell_height)
                self.page.draw_rect(rect, color=(0.70, 0.75, 0.82), fill=cell_fill, width=0.5)
                ty = self.y + pad_y + fontsize
                for line in wrapped_cells.get((row_index, column_index), []):
                    if ty > self.y + cell_height - pad_y:
                        break
                    self.page.insert_text(
                        (x + pad_x, ty),
                        line,
                        fontname=FONT_CJK,
                        fontsize=fontsize,
                        color=(0.13, 0.18, 0.25),
                    )
                    ty += leading
                x += cell_width
                column_index += column_span
            self.y += row_height
        self.y += 8
        return len(anchors)

    def add_box(self, title: str, lines: list[str]) -> None:
        if not lines:
            return
        fontsize = 9.2
        leading = fontsize * 1.35
        wrapped: list[str] = []
        for line in lines:
            wrapped.extend(self.wrap_text(line, fontsize, self.content_right - MARGIN_X - 18))
        height = (len(wrapped) + 1) * leading + 18
        self.ensure_space(height)
        assert self.page is not None
        rect = fitz.Rect(MARGIN_X, self.y, self.content_right, self.y + height)
        self.page.draw_rect(rect, color=(0.80, 0.70, 0.45), fill=(1.0, 0.98, 0.90), width=0.6)
        self.page.insert_text(
            (MARGIN_X + 9, self.y + 16),
            title,
            fontname=FONT_CJK,
            fontsize=9.8,
            color=(0.55, 0.30, 0.08),
        )
        ty = self.y + 16 + leading
        for line in wrapped:
            self.page.insert_text(
                (MARGIN_X + 9, ty),
                line,
                fontname=FONT_CJK,
                fontsize=fontsize,
                color=(0.32, 0.24, 0.12),
            )
            ty += leading
        self.y += height + 10

    def finalize(self, path: Path) -> int:
        for idx, page in enumerate(self.doc, start=1):
            page.insert_text(
                (MARGIN_X, PAGE_HEIGHT - 24),
                f"{self.title} / {idx}",
                fontname=FONT_LATIN,
                fontsize=8,
                color=(0.55, 0.60, 0.66),
            )
        self.doc.set_metadata({"title": self.title, "creator": "pdf_translate"})
        path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(path)
        page_count = len(self.doc)
        self.doc.close()
        return page_count


def build_translated_pdf_report(
    chunks: list[TextChunk],
    chunk_dir: Path,
    *,
    qa_report: dict[str, Any] | None = None,
    repair_plan: dict[str, Any] | None = None,
    structure_qa: dict[str, Any] | None = None,
    table_reconstruction: dict[str, Any] | None = None,
    title: str = "结构化译文",
    source_pdf: Path | str | None = None,
) -> dict[str, Any]:
    issues_by_chunk = _qa_issues_by_chunk(qa_report)
    repairs_by_chunk = _index_by_chunk((repair_plan or {}).get("items") or [])
    chunk_reports: list[dict[str, Any]] = []
    warnings: list[str] = []
    total_tables = 0
    total_chars = 0
    structure_summary_fields = _int_summary_fields(_summary(structure_qa), STRUCTURE_QA_SUMMARY_FIELDS)
    table_summary_fields = _int_summary_fields(
        _summary(table_reconstruction),
        TABLE_RECONSTRUCTION_SUMMARY_FIELDS,
    )
    structure_context_chunk_count = 0
    source_table_reference_count = 0
    source_caption_reference_count = 0
    source_footnote_reference_count = 0
    source_footnote_cell_binding_count = 0
    source_footnote_bound_cell_count = 0
    merged_cell_candidate_reference_count = 0
    confirmed_merged_cell_candidate_reference_count = 0
    table_structure_patch_reference_count = 0
    table_structure_patch_covered_cell_reference_count = 0
    continued_table_group_reference_count = 0
    structural_relation_reference_count = 0

    for chunk in chunks:
        translation = _chunk_translation_text(chunk_dir, chunk.chunk_id)
        blocks = _markdown_blocks(translation)
        table_count = sum(1 for block in blocks if block.get("type") == "table")
        structure_context = _table_structure_context(chunk, table_reconstruction)
        has_structure_context = any(
            int(structure_context.get(key) or 0) > 0
            for key in (
                "source_table_count",
                "source_caption_count",
                "source_footnote_count",
                "source_footnote_cell_binding_count",
                "merged_cell_candidate_reference_count",
                "confirmed_merged_cell_candidate_reference_count",
                "table_structure_patch_reference_count",
                "continued_table_group_count",
                "structural_relation_count",
            )
        )
        total_tables += table_count
        total_chars += len(translation)
        structure_context_chunk_count += int(has_structure_context)
        source_table_reference_count += int(structure_context["source_table_count"])
        source_caption_reference_count += int(structure_context["source_caption_count"])
        source_footnote_reference_count += int(structure_context["source_footnote_count"])
        source_footnote_cell_binding_count += int(structure_context["source_footnote_cell_binding_count"])
        source_footnote_bound_cell_count += int(structure_context["source_footnote_bound_cell_count"])
        merged_cell_candidate_reference_count += int(structure_context["merged_cell_candidate_reference_count"])
        confirmed_merged_cell_candidate_reference_count += int(
            structure_context["confirmed_merged_cell_candidate_reference_count"]
        )
        table_structure_patch_reference_count += int(
            structure_context["table_structure_patch_reference_count"]
        )
        table_structure_patch_covered_cell_reference_count += int(
            structure_context["table_structure_patch_covered_cell_reference_count"]
        )
        continued_table_group_reference_count += int(structure_context["continued_table_group_count"])
        structural_relation_reference_count += int(structure_context["structural_relation_count"])
        if not translation:
            warnings.append(f"missing_translation:{chunk.chunk_id}")
        chunk_reports.append(
            {
                "chunk_id": chunk.chunk_id,
                "pages_1based": _chunk_pages_1based(chunk),
                "translated_char_count": len(translation),
                "table_count": table_count,
                "qa_issue_count": len(issues_by_chunk.get(chunk.chunk_id, [])),
                "repair_item_count": len(repairs_by_chunk.get(chunk.chunk_id, [])),
                "table_structure_patch_rendered_count": 0,
                "has_structure_context": has_structure_context,
                **structure_context,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "title": title,
        "source_pdf": str(source_pdf) if source_pdf is not None else "",
        "table_reconstruction_source": _table_reconstruction_source(table_reconstruction),
        "table_reconstruction_confirmation_schema_version": str(
            (table_reconstruction or {}).get("confirmation_schema_version") or ""
        )
        if isinstance(table_reconstruction, dict)
        else "",
        "summary": {
            "generated": False,
            "chunk_count": len(chunks),
            "translated_chunk_count": sum(1 for item in chunk_reports if item["translated_char_count"] > 0),
            "translated_char_count": total_chars,
            "table_count": total_tables,
            "qa_issue_count": sum(len(v) for v in issues_by_chunk.values()),
            "repair_item_count": sum(len(v) for v in repairs_by_chunk.values()),
            **structure_summary_fields,
            **table_summary_fields,
            "structure_context_chunk_count": structure_context_chunk_count,
            "source_table_reference_count": source_table_reference_count,
            "source_caption_reference_count": source_caption_reference_count,
            "source_footnote_reference_count": source_footnote_reference_count,
            "source_footnote_cell_binding_count": source_footnote_cell_binding_count,
            "source_footnote_bound_cell_count": source_footnote_bound_cell_count,
            "merged_cell_candidate_reference_count": merged_cell_candidate_reference_count,
            "confirmed_merged_cell_candidate_reference_count": (
                confirmed_merged_cell_candidate_reference_count
            ),
            "table_structure_patch_reference_count": table_structure_patch_reference_count,
            "table_structure_patch_covered_cell_reference_count": (
                table_structure_patch_covered_cell_reference_count
            ),
            "table_structure_patch_rendered_count": 0,
            "continued_table_group_reference_count": continued_table_group_reference_count,
            "structural_relation_reference_count": structural_relation_reference_count,
            "page_count": 0,
            "font": FONT_CJK,
            "warning_count": len(warnings),
        },
        "warnings": warnings,
        "chunks": chunk_reports,
    }


def write_translated_pdf(
    chunks: list[TextChunk],
    chunk_dir: Path,
    path: Path,
    *,
    qa_report: dict[str, Any] | None = None,
    repair_plan: dict[str, Any] | None = None,
    structure_qa: dict[str, Any] | None = None,
    table_reconstruction: dict[str, Any] | None = None,
    title: str = "结构化译文",
    source_pdf: Path | str | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    report = build_translated_pdf_report(
        chunks,
        chunk_dir,
        qa_report=qa_report,
        repair_plan=repair_plan,
        structure_qa=structure_qa,
        table_reconstruction=table_reconstruction,
        title=title,
        source_pdf=source_pdf,
    )
    issues_by_chunk = _qa_issues_by_chunk(qa_report)
    repairs_by_chunk = _index_by_chunk((repair_plan or {}).get("items") or [])
    chunk_reports_by_id = {
        str(item.get("chunk_id")): item
        for item in report.get("chunks", [])
        if isinstance(item, dict) and str(item.get("chunk_id") or "")
    }

    flow = _PdfFlow(title)
    flow.add_text(title, fontsize=20, color=(0.06, 0.32, 0.30), after=10)
    source_text = f"源文件: {source_pdf}" if source_pdf else "源文件: -"
    summary = report["summary"]
    flow.add_text(
        (
            f"{source_text}\n"
            f"翻译块: {summary['chunk_count']} | 已有译文块: {summary['translated_chunk_count']} | "
            f"PDF 表格: {summary['table_count']} | QA 标注: {summary['qa_issue_count']} | "
            f"修复建议: {summary['repair_item_count']}"
        ),
        fontsize=10.5,
        color=(0.30, 0.35, 0.42),
        after=12,
    )
    flow.add_rule()

    for chunk in chunks:
        pages = [p + 1 for p in chunk.pages_0based]
        page_text = f"{pages[0]}-{pages[-1]}" if pages else "-"
        translation = _chunk_translation_text(chunk_dir, chunk.chunk_id)
        issues = issues_by_chunk.get(chunk.chunk_id, [])
        repairs = repairs_by_chunk.get(chunk.chunk_id, [])
        table_contexts = _renderable_tables_for_chunk(chunk, table_reconstruction)
        table_context_index = 0
        chunk_report = chunk_reports_by_id.get(chunk.chunk_id, {})
        rendered_patch_count = 0

        flow.add_text(
            f"{chunk.chunk_id} | 页码 {page_text}",
            fontsize=14.5,
            color=(0.08, 0.38, 0.35),
            before=6,
            after=6,
        )
        note_lines = []
        if issues:
            note_lines.append("QA: " + "; ".join(_short_item(issue, "type") for issue in issues[:6]))
        if repairs:
            note_lines.append("修复: " + "; ".join(_short_item(item, "action", "reason") for item in repairs[:6]))
        flow.add_box("质量提示", note_lines)
        flow.add_box("Structure context", _structure_context_lines(chunk_report))

        blocks = _markdown_blocks(translation)
        if not blocks:
            flow.add_text("未生成译文内容。", fontsize=10.5, color=(0.60, 0.22, 0.18), after=12)
            continue
        for block in blocks:
            btype = block.get("type")
            if btype == "heading":
                level = int(block.get("level") or 2)
                size = max(11.5, 15.0 - level)
                flow.add_text(
                    str(block.get("text") or ""),
                    fontsize=size,
                    color=(0.10, 0.28, 0.45),
                    before=4,
                    after=5,
                )
            elif btype == "table":
                table_context = None
                if table_context_index < len(table_contexts):
                    table_context = table_contexts[table_context_index]
                rendered_patch_count += flow.add_table(block.get("rows") or [], table_context=table_context)
                table_context_index += 1
            else:
                flow.add_text(str(block.get("text") or ""), fontsize=10.5, after=7)
        chunk_report["table_structure_patch_rendered_count"] = rendered_patch_count
        report["summary"]["table_structure_patch_rendered_count"] += rendered_patch_count

    page_count = flow.finalize(path)
    report["summary"]["generated"] = True
    report["summary"]["page_count"] = page_count
    report["pdf_path"] = str(path)
    if report_path is None:
        report_path = path.with_name("translated_pdf_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
