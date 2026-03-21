from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass
class PageRichMeta:
    page_0based: int
    link_count: int
    image_count: int
    links_sample: list[str]


def _link_urls(page: fitz.Page) -> list[str]:
    urls: list[str] = []
    for link in page.get_links() or []:
        uri = link.get("uri")
        if uri:
            urls.append(uri)
    return urls


def extract_page_rich_meta(pdf_path: Path) -> list[PageRichMeta]:
    """每页超链接与图片数量，供块元数据与设计文档中的 HTML/清单导出（阶段 C 基础）。"""
    doc = fitz.open(pdf_path)
    try:
        out: list[PageRichMeta] = []
        for i in range(len(doc)):
            page = doc[i]
            urls = _link_urls(page)
            imgs = page.get_images() or []
            out.append(
                PageRichMeta(
                    page_0based=i,
                    link_count=len(urls),
                    image_count=len(imgs),
                    links_sample=urls[:20],
                )
            )
        return out
    finally:
        doc.close()


def export_links_csv(meta: list[PageRichMeta], csv_path: Path) -> None:
    import csv

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["page_1based", "link_count", "sample_urls"])
        for m in meta:
            w.writerow([m.page_0based + 1, m.link_count, " | ".join(m.links_sample)])
