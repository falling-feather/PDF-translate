from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import fitz

from pdf_translate.extractors.document_ir import DocumentIR, PageIR

SCHEMA_VERSION = "vision-route-v1"
PREVIEW_DIR_NAME = "vision_pages"
PREVIEW_MAX_WIDTH = 960


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _page_area(page: PageIR) -> float:
    return max(1.0, float(page.width) * float(page.height))


def _image_area_ratio(page: PageIR) -> float:
    raw = page.meta.get("image_area_ratio") if isinstance(page.meta, dict) else None
    if isinstance(raw, (int, float)):
        return float(raw)
    image_area = sum(_bbox_area(block.bbox) for block in page.blocks if block.type == "image")
    return min(1.0, image_area / _page_area(page))


def _text_area_ratio(page: PageIR) -> float:
    raw = page.meta.get("text_area_ratio") if isinstance(page.meta, dict) else None
    if isinstance(raw, (int, float)):
        return float(raw)
    text_types = {
        "paragraph",
        "heading",
        "table",
        "caption",
        "footnote",
        "formula",
        "reference",
    }
    text_area = sum(_bbox_area(block.bbox) for block in page.blocks if block.type in text_types)
    return min(1.0, text_area / _page_area(page))


def _risk_reasons(
    page: PageIR,
    *,
    text_chars: int,
    text_area_ratio: float,
    image_area_ratio: float,
    block_counts: Counter[str],
) -> tuple[list[str], float]:
    reasons: list[str] = []
    score = 0.0

    if text_chars < 40:
        reasons.append("very_low_text")
        score += 0.35
    elif text_chars < 120:
        reasons.append("low_text")
        score += 0.2

    if text_area_ratio < 0.03 and text_chars < 300:
        reasons.append("low_text_area")
        score += 0.2

    if image_area_ratio > 0.45:
        reasons.append("image_area_heavy")
        score += 0.3
    elif image_area_ratio > 0.25:
        reasons.append("image_area_present")
        score += 0.18

    if page.image_count >= 2 or block_counts.get("image", 0) >= 2:
        reasons.append("multiple_images")
        score += 0.12

    if block_counts.get("caption", 0) and (page.image_count or block_counts.get("image", 0)):
        reasons.append("image_caption_context")
        score += 0.15

    if block_counts.get("table", 0) and (page.image_count or image_area_ratio > 0.2):
        reasons.append("possible_image_table")
        score += 0.12

    if block_counts.get("formula", 0) and text_chars < 300:
        reasons.append("formula_dense_low_text")
        score += 0.1

    for warning in page.warnings:
        if warning in {
            "low_text_image_heavy_page",
            "low_text_area_page",
            "image_area_heavy_page",
            "image_caption_page",
        } and warning not in reasons:
            reasons.append(warning)
            score += 0.08

    return reasons, min(1.0, round(score, 3))


def _route_action(
    page: PageIR,
    *,
    text_chars: int,
    image_area_ratio: float,
    risk_score: float,
    reasons: list[str],
) -> tuple[str, str]:
    has_image = page.image_count > 0 or image_area_ratio > 0.05
    if text_chars < 30 and not has_image:
        return "skip_blank", "文本层和图像区域都很少，优先视为空白页或封面边缘页。"
    if has_image and ("very_low_text" in reasons or "low_text_image_heavy_page" in reasons):
        return "local_ocr", "文本层不足且存在图像区域，建议先走本地 OCR 或版面解析。"
    if has_image and risk_score >= 0.65:
        return "local_ocr", "图像/低文本风险较高，先用本地 OCR 补齐文本层。"
    if has_image and ("image_caption_context" in reasons or "possible_image_table" in reasons):
        return "vlm_review", "存在图像、图注或疑似图片型表格，OCR 后仍异常时再交给 VLM。"
    return "text_only", "文本层足够，按普通结构分段和翻译处理。"


def build_vision_route(doc_ir: DocumentIR) -> dict[str, Any]:
    """Build a local page-level OCR/VLM routing manifest from DocumentIR facts."""
    pages: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()

    for page in doc_ir.pages:
        block_counts = Counter(block.type for block in page.blocks)
        text_chars = (
            int(page.meta.get("text_char_count") or len(page.text.strip()))
            if isinstance(page.meta, dict)
            else len(page.text.strip())
        )
        image_area_ratio = _image_area_ratio(page)
        text_area_ratio = _text_area_ratio(page)
        reasons, risk_score = _risk_reasons(
            page,
            text_chars=text_chars,
            text_area_ratio=text_area_ratio,
            image_area_ratio=image_area_ratio,
            block_counts=block_counts,
        )
        action, next_step = _route_action(
            page,
            text_chars=text_chars,
            image_area_ratio=image_area_ratio,
            risk_score=risk_score,
            reasons=reasons,
        )
        risk_level = "high" if risk_score >= 0.65 else "medium" if risk_score >= 0.35 else "low"
        action_counts[action] += 1
        risk_counts[risk_level] += 1
        pages.append(
            {
                "page_no": page.page_no,
                "action": action,
                "risk_level": risk_level,
                "risk_score": risk_score,
                "reasons": reasons,
                "next_step": next_step,
                "metrics": {
                    "text_chars": text_chars,
                    "text_area_ratio": round(text_area_ratio, 4),
                    "image_count": page.image_count,
                    "image_area_ratio": round(image_area_ratio, 4),
                    "block_counts": dict(block_counts),
                    "page_warnings": page.warnings,
                },
                "evidence": {
                    "page_preview_status": "not_rendered",
                    "page_preview_path": "",
                    "page_preview_width": 0,
                    "page_preview_height": 0,
                    "page_preview_scale": 0,
                },
            }
        )

    routed_count = sum(count for action, count in action_counts.items() if action != "text_only")
    return {
        "schema_version": SCHEMA_VERSION,
        "doc_id": doc_ir.doc_id,
        "summary": {
            "page_count": len(doc_ir.pages),
            "routed_page_count": routed_count,
            "high_risk_page_count": risk_counts.get("high", 0),
            "preview_page_count": 0,
            "action_counts": dict(action_counts),
            "risk_counts": dict(risk_counts),
        },
        "pages": pages,
    }


def _render_scale(page: fitz.Page) -> float:
    width = max(1.0, float(page.rect.width))
    return min(2.0, max(1.0, PREVIEW_MAX_WIDTH / width))


def _attach_page_previews(route: dict[str, Any], doc_ir: DocumentIR, output_dir: Path) -> None:
    pages = route.get("pages")
    if not isinstance(pages, list):
        return

    source_pdf = Path(doc_ir.source_pdf)
    preview_dir = output_dir / PREVIEW_DIR_NAME
    rendered_count = 0
    missing_count = 0

    if not source_pdf.is_file():
        for page in pages:
            if not isinstance(page, dict):
                continue
            evidence = page.setdefault("evidence", {})
            if page.get("action") == "text_only":
                evidence["page_preview_status"] = "not_needed"
            else:
                evidence["page_preview_status"] = "source_missing"
                missing_count += 1
        summary = route.get("summary")
        if isinstance(summary, dict):
            summary["preview_page_count"] = rendered_count
            summary["preview_missing_count"] = missing_count
            summary["preview_dir"] = PREVIEW_DIR_NAME
        return

    try:
        doc = fitz.open(source_pdf)
    except Exception:
        for page in pages:
            if not isinstance(page, dict):
                continue
            evidence = page.setdefault("evidence", {})
            if page.get("action") == "text_only":
                evidence["page_preview_status"] = "not_needed"
                continue
            evidence["page_preview_status"] = "source_open_failed"
            missing_count += 1
        summary = route.get("summary")
        if isinstance(summary, dict):
            summary["preview_page_count"] = rendered_count
            summary["preview_missing_count"] = missing_count
            summary["preview_dir"] = PREVIEW_DIR_NAME
        return

    try:
        for page in pages:
            if not isinstance(page, dict):
                continue
            evidence = page.setdefault("evidence", {})
            if page.get("action") == "text_only":
                evidence["page_preview_status"] = "not_needed"
                continue
            page_no = int(page.get("page_no") or 0)
            if page_no < 1 or page_no > len(doc):
                evidence["page_preview_status"] = "page_missing"
                missing_count += 1
                continue

            preview_dir.mkdir(parents=True, exist_ok=True)
            pdf_page = doc[page_no - 1]
            scale = _render_scale(pdf_page)
            pix = pdf_page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            preview_path = preview_dir / f"page-{page_no:04d}.png"
            pix.save(preview_path)
            evidence.update(
                {
                    "page_preview_status": "rendered",
                    "page_preview_path": preview_path.relative_to(output_dir).as_posix(),
                    "page_preview_width": pix.width,
                    "page_preview_height": pix.height,
                    "page_preview_scale": round(scale, 3),
                }
            )
            rendered_count += 1
    finally:
        doc.close()

    summary = route.get("summary")
    if isinstance(summary, dict):
        summary["preview_page_count"] = rendered_count
        summary["preview_missing_count"] = missing_count
        summary["preview_dir"] = PREVIEW_DIR_NAME


def write_vision_route(doc_ir: DocumentIR, path: Path) -> dict[str, Any]:
    route = build_vision_route(doc_ir)
    path.parent.mkdir(parents=True, exist_ok=True)
    _attach_page_previews(route, doc_ir, path.parent)
    path.write_text(
        json.dumps(route, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return route
