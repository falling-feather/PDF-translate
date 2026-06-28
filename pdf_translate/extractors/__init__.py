"""PDF extraction adapters that produce structured intermediate data."""

from pdf_translate.extractors.document_ir import (
    BlockIR,
    DocumentIR,
    PageIR,
    classify_text_block,
    extract_document_ir,
)

__all__ = [
    "BlockIR",
    "DocumentIR",
    "PageIR",
    "classify_text_block",
    "extract_document_ir",
]

