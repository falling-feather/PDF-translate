from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # pymupdf


REF_LINE_RE = re.compile(
    r"^(References|REFERENCES|Bibliography|BIBLIOGRAPHY|参考文献)\s*$",
    re.MULTILINE,
)


@dataclass
class SplitManifest:
    source_pdf: str
    total_pages: int
    reference_start_page_0based: int | None
    main_pages_0based: list[int]
    reference_pages_0based: list[int]
    main_pdf: str
    references_pdf: str | None

    def to_json_dict(self) -> dict:
        return {
            "source_pdf": self.source_pdf,
            "total_pages": self.total_pages,
            "reference_start_page_0based": self.reference_start_page_0based,
            "main_pages_0based": self.main_pages_0based,
            "reference_pages_0based": self.reference_pages_0based,
            "main_pdf": self.main_pdf,
            "references_pdf": self.references_pdf,
        }


def _page_starts_with_reference_heading(page: fitz.Page) -> bool:
    text = page.get_text("text")
    head = "\n".join(text.splitlines()[:40])
    return bool(REF_LINE_RE.search(head))


def find_reference_start_page(doc: fitz.Document) -> int | None:
    """Return 0-based index of the page where the reference section likely starts."""
    n = len(doc)
    if n == 0:
        return None
    candidates: list[int] = []
    for i in range(n):
        if _page_starts_with_reference_heading(doc[i]):
            candidates.append(i)
    if not candidates:
        return None
    mid = n // 2
    late = [c for c in candidates if c >= mid]
    return max(late) if late else max(candidates)


def split_main_and_references(
    input_pdf: Path,
    out_dir: Path,
    *,
    ref_tail_ratio: float = 0.15,
    use_tail_if_no_heading: bool = False,
) -> SplitManifest:
    """
    Export main body and references (low-priority) into separate PDFs (see README pipeline section).
    If no heading match and use_tail_if_no_heading is False, all pages go to main.
    If use_tail_if_no_heading is True, last ref_tail_ratio pages become references.pdf.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(input_pdf)
    try:
        total = len(doc)
        ref_start = find_reference_start_page(doc)
        if ref_start is None and use_tail_if_no_heading and total > 2:
            k = max(1, int(total * ref_tail_ratio))
            ref_start = total - k

        if ref_start is None:
            main_pages = list(range(total))
            ref_pages: list[int] = []
        else:
            main_pages = list(range(ref_start))
            ref_pages = list(range(ref_start, total))

        if not main_pages:
            # 不应出现：参考文献从第 1 页开始则回退为全文正文
            main_pages = list(range(total))
            ref_pages = []
            ref_start = None

        main_path = out_dir / "main.pdf"
        ref_path = out_dir / "references.pdf"

        main_doc = fitz.open()
        for p in main_pages:
            main_doc.insert_pdf(doc, from_page=p, to_page=p)
        main_doc.save(main_path)
        main_doc.close()

        ref_pdf_str: str | None = None
        if ref_pages:
            ref_doc = fitz.open()
            for p in ref_pages:
                ref_doc.insert_pdf(doc, from_page=p, to_page=p)
            ref_doc.save(ref_path)
            ref_doc.close()
            ref_pdf_str = str(ref_path.resolve())

        manifest = SplitManifest(
            source_pdf=str(input_pdf.resolve()),
            total_pages=total,
            reference_start_page_0based=ref_start,
            main_pages_0based=main_pages,
            reference_pages_0based=ref_pages,
            main_pdf=str(main_path.resolve()),
            references_pdf=ref_pdf_str,
        )
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest.to_json_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest
    finally:
        doc.close()
