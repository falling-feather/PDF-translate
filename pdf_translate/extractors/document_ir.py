from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import fitz

BlockType = Literal[
    "paragraph",
    "heading",
    "table",
    "caption",
    "footnote",
    "formula",
    "reference",
    "header_footer",
    "image",
]

SCHEMA_VERSION = "document-ir-v1"

_CAPTION_RE = re.compile(r"^\s*(fig(?:ure)?|table|图|表)\s*[\dIVX一二三四五六七八九十]+[\.:：、\s]", re.I)
_REFERENCE_RE = re.compile(r"^\s*(references|bibliography|参考文献|参考资料)\s*$", re.I)
_HEADING_RE = re.compile(r"^\s*((\d+(\.\d+)*)|[IVX]+)\s+[\w\u4e00-\u9fff].{0,90}$", re.I)
_FOOTNOTE_RE = re.compile(r"^\s*(\d+|[*†‡§])[\).、\s]")
_MATH_RE = re.compile(r"(=|≈|≤|≥|±|∑|∫|√|α|β|γ|λ|μ|σ|\\frac|\\sum|\\alpha|\\beta)")
_LOCKED_TOKEN_RE = re.compile(
    r"(\[[0-9,\-\s]+\]|\([A-Z][A-Za-z]+,\s*\d{4}\)|Table\s+\d+|Fig(?:ure)?\.?\s+\d+|"
    r"\b\d+(?:\.\d+)?%?\b|[A-Za-z]\d\b|[A-Z]{2,}(?:-[A-Z0-9]+)*)"
)


@dataclass
class BlockIR:
    block_id: str
    page_no: int
    type: BlockType
    text: str
    bbox: tuple[float, float, float, float]
    order: int
    parent_id: str | None = None
    locked_tokens: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "page_no": self.page_no,
            "type": self.type,
            "text": self.text,
            "bbox": list(self.bbox),
            "order": self.order,
            "parent_id": self.parent_id,
            "locked_tokens": self.locked_tokens,
            "meta": self.meta,
        }


@dataclass
class PageIR:
    page_no: int
    width: float
    height: float
    text: str
    blocks: list[BlockIR]
    link_count: int = 0
    image_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "page_no": self.page_no,
            "width": self.width,
            "height": self.height,
            "text": self.text,
            "link_count": self.link_count,
            "image_count": self.image_count,
            "warnings": self.warnings,
            "blocks": [b.to_json_dict() for b in self.blocks],
        }


@dataclass
class DocumentIR:
    doc_id: str
    source_pdf: str
    pages: list[PageIR]
    schema_version: str = SCHEMA_VERSION

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "doc_id": self.doc_id,
            "source_pdf": self.source_pdf,
            "pages": [p.to_json_dict() for p in self.pages],
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def _span_text(line: dict[str, Any]) -> str:
    return "".join(str(span.get("text") or "") for span in line.get("spans") or []).strip()


def _line_x_positions(line: dict[str, Any]) -> list[float]:
    out: list[float] = []
    for span in line.get("spans") or []:
        bbox = span.get("bbox")
        if isinstance(bbox, (list, tuple)) and bbox:
            out.append(float(bbox[0]))
    return out


def _span_sizes(lines: list[dict[str, Any]]) -> list[float]:
    sizes: list[float] = []
    for line in lines:
        for span in line.get("spans") or []:
            size = span.get("size")
            if isinstance(size, (int, float)):
                sizes.append(float(size))
    return sizes


def _looks_like_table(line_texts: list[str], line_xs: list[list[float]]) -> bool:
    if not line_texts:
        return False
    multi_span_lines = sum(1 for xs in line_xs if len(xs) >= 3 and (max(xs) - min(xs) > 120))
    numeric_dense = 0
    for text in line_texts:
        tokens = re.findall(r"\b\d+(?:\.\d+)?%?\b", text)
        if len(tokens) >= 3:
            numeric_dense += 1
        if "\t" in text or re.search(r"\S\s{3,}\S", text):
            numeric_dense += 1
    return multi_span_lines >= 2 or numeric_dense >= 2


def _locked_tokens(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in _LOCKED_TOKEN_RE.finditer(text):
        token = m.group(0).strip()
        if token and token not in seen:
            seen.add(token)
            out.append(token)
    return out[:80]


def classify_text_block(
    text: str,
    *,
    bbox: tuple[float, float, float, float],
    page_height: float,
    line_texts: list[str],
    line_xs: list[list[float]],
    span_sizes: list[float],
) -> BlockType:
    stripped = text.strip()
    if not stripped:
        return "paragraph"

    y0, y1 = bbox[1], bbox[3]
    avg_size = sum(span_sizes) / len(span_sizes) if span_sizes else 11.0
    one_line = len([ln for ln in line_texts if ln.strip()]) <= 1

    if (y1 < page_height * 0.08 or y0 > page_height * 0.94) and len(stripped) < 140:
        return "header_footer"
    if _REFERENCE_RE.match(stripped):
        return "reference"
    if _CAPTION_RE.match(stripped):
        return "caption"
    if _FOOTNOTE_RE.match(stripped) and (y0 > page_height * 0.70 or avg_size <= 9.5):
        return "footnote"
    if y0 > page_height * 0.78 and avg_size <= 8.8 and len(stripped) < 500:
        return "footnote"
    if _looks_like_table(line_texts, line_xs):
        return "table"
    if _MATH_RE.search(stripped) and len(stripped) < 400:
        return "formula"
    if (_HEADING_RE.match(stripped) or (one_line and len(stripped) <= 90 and avg_size >= 12.5)) and not stripped.endswith("."):
        return "heading"
    return "paragraph"


def _image_area_ratio(raw_blocks: list[dict[str, Any]], page_area: float) -> float:
    if page_area <= 0:
        return 0.0
    area = 0.0
    for block in raw_blocks:
        if block.get("type") != 1:
            continue
        bbox = block.get("bbox") or (0, 0, 0, 0)
        try:
            x0, y0, x1, y1 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            continue
        area += max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return min(1.0, area / page_area)


def extract_document_ir(pdf_path: Path, *, doc_id: str | None = None) -> DocumentIR:
    """Extract a lightweight structure IR from a PDF using local PyMuPDF data only."""
    pdf_path = pdf_path.resolve()
    doc = fitz.open(pdf_path)
    try:
        pages: list[PageIR] = []
        for page_index, page in enumerate(doc):
            page_no = page_index + 1
            page_dict = page.get_text("dict")
            raw_blocks = list(page_dict.get("blocks") or [])
            rect = page.rect
            blocks: list[BlockIR] = []
            text_parts: list[str] = []
            type_counts: Counter[str] = Counter()

            for order, raw in enumerate(raw_blocks):
                bbox_raw = raw.get("bbox") or (0, 0, 0, 0)
                try:
                    bbox = tuple(float(v) for v in bbox_raw)  # type: ignore[assignment]
                except (TypeError, ValueError):
                    bbox = (0.0, 0.0, 0.0, 0.0)
                block_id = f"p{page_no}-b{order:04d}"

                if raw.get("type") == 1:
                    block = BlockIR(
                        block_id=block_id,
                        page_no=page_no,
                        type="image",
                        text="",
                        bbox=bbox,
                        order=order,
                        meta={
                            "width": raw.get("width"),
                            "height": raw.get("height"),
                            "ext": raw.get("ext"),
                        },
                    )
                    blocks.append(block)
                    type_counts[block.type] += 1
                    continue

                lines = list(raw.get("lines") or [])
                line_texts = [_span_text(line) for line in lines]
                line_texts = [ln for ln in line_texts if ln]
                if not line_texts:
                    continue
                text = "\n".join(line_texts).strip()
                line_xs = [_line_x_positions(line) for line in lines]
                sizes = _span_sizes(lines)
                block_type = classify_text_block(
                    text,
                    bbox=bbox,
                    page_height=float(rect.height),
                    line_texts=line_texts,
                    line_xs=line_xs,
                    span_sizes=sizes,
                )
                block = BlockIR(
                    block_id=block_id,
                    page_no=page_no,
                    type=block_type,
                    text=text,
                    bbox=bbox,
                    order=order,
                    locked_tokens=_locked_tokens(text),
                    meta={"avg_font_size": round(sum(sizes) / len(sizes), 2) if sizes else None},
                )
                blocks.append(block)
                type_counts[block.type] += 1
                if block.type != "header_footer":
                    text_parts.append(text)

            page_text = "\n\n".join(text_parts)
            image_count = len(page.get_images() or [])
            link_count = len(page.get_links() or [])
            image_ratio = _image_area_ratio(raw_blocks, float(rect.width * rect.height))
            warnings: list[str] = []
            if len(page_text.strip()) < 120 and (image_count > 0 or image_ratio > 0.35):
                warnings.append("low_text_image_heavy_page")
            if type_counts.get("table", 0):
                warnings.append("table_like_content")
            if type_counts.get("caption", 0) and type_counts.get("image", 0):
                warnings.append("image_caption_page")

            pages.append(
                PageIR(
                    page_no=page_no,
                    width=float(rect.width),
                    height=float(rect.height),
                    text=page_text,
                    blocks=blocks,
                    link_count=link_count,
                    image_count=image_count,
                    warnings=warnings,
                )
            )

        return DocumentIR(
            doc_id=doc_id or pdf_path.stem,
            source_pdf=str(pdf_path),
            pages=pages,
        )
    finally:
        doc.close()

