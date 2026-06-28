from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from pdf_translate.extractors.document_ir import DocumentIR
from pdf_translate.structure_boundaries import detect_page_boundary_fragments


def build_structure_qa(doc_ir: DocumentIR) -> dict[str, Any]:
    """Summarize local structure invariants for later translation QA and experiments."""
    block_counts: Counter[str] = Counter()
    page_warnings: list[dict[str, Any]] = []
    table_blocks: list[dict[str, Any]] = []
    page_boundary_fragments = detect_page_boundary_fragments(doc_ir)

    for page in doc_ir.pages:
        if page.warnings:
            page_warnings.append({"page_no": page.page_no, "warnings": page.warnings})
        for block in page.blocks:
            block_counts[block.type] += 1
            if block.type != "table":
                continue
            table = block.meta.get("table") if isinstance(block.meta, dict) else None
            table = table if isinstance(table, dict) else {}
            table_blocks.append(
                {
                    "block_id": block.block_id,
                    "page_no": block.page_no,
                    "bbox": list(block.bbox),
                    "row_count": int(table.get("row_count") or 0),
                    "column_count": int(table.get("column_count") or 0),
                    "header": table.get("header") or [],
                    "numeric_tokens": table.get("numeric_tokens") or [],
                    "warnings": table.get("warnings") or [],
                    "confidence": table.get("confidence") or "low",
                }
            )

    boundary_count = len(page_boundary_fragments)
    possible_boundary_count = max(0, len(doc_ir.pages) - 1)
    return {
        "schema_version": "structure-qa-v1",
        "doc_id": doc_ir.doc_id,
        "summary": {
            "page_count": len(doc_ir.pages),
            "block_counts": dict(block_counts),
            "table_count": len(table_blocks),
            "warning_page_count": len(page_warnings),
            "page_boundary_fragment_count": boundary_count,
            "page_boundary_fragment_rate": round(boundary_count / possible_boundary_count, 4)
            if possible_boundary_count
            else 0.0,
        },
        "tables": table_blocks,
        "page_boundary_fragments": page_boundary_fragments,
        "page_warnings": page_warnings,
    }


def write_structure_qa(doc_ir: DocumentIR, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_structure_qa(doc_ir), ensure_ascii=False, indent=2), encoding="utf-8")
