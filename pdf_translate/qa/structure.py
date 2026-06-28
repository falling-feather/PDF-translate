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
    relationships: list[dict[str, Any]] = []
    page_boundary_fragments = detect_page_boundary_fragments(doc_ir)
    table_continuations = _table_continuations(page_boundary_fragments)
    continued_from_by_block = {
        item["next_table_block_id"]: item["previous_table_block_id"] for item in table_continuations
    }
    continued_to_by_block = {
        item["previous_table_block_id"]: item["next_table_block_id"] for item in table_continuations
    }

    for page in doc_ir.pages:
        if page.warnings:
            page_warnings.append({"page_no": page.page_no, "warnings": page.warnings})
        for block in page.blocks:
            block_counts[block.type] += 1
            if block.type in {"caption", "footnote"}:
                meta = block.meta if isinstance(block.meta, dict) else {}
                relationships.append(
                    {
                        "block_id": block.block_id,
                        "page_no": block.page_no,
                        "type": block.type,
                        "parent_id": block.parent_id,
                        "parent_relation": meta.get("parent_relation"),
                        "warning": meta.get("parent_warning"),
                        "caption_kind": meta.get("caption_kind") if block.type == "caption" else None,
                        "text_preview": block.text.strip().replace("\n", " ")[:160],
                    }
                )
            if block.type != "table":
                continue
            table = block.meta.get("table") if isinstance(block.meta, dict) else None
            table = table if isinstance(table, dict) else {}
            table_blocks.append(
                {
                    "block_id": block.block_id,
                    "page_no": block.page_no,
                    "bbox": list(block.bbox),
                    "continued_from_block_id": continued_from_by_block.get(block.block_id),
                    "continued_to_block_id": continued_to_by_block.get(block.block_id),
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
    caption_count = sum(1 for item in relationships if item["type"] == "caption")
    footnote_count = sum(1 for item in relationships if item["type"] == "footnote")
    linked_relationship_count = sum(1 for item in relationships if item.get("parent_id"))
    caption_orphan_count = sum(1 for item in relationships if item["type"] == "caption" and not item.get("parent_id"))
    footnote_orphan_count = sum(1 for item in relationships if item["type"] == "footnote" and not item.get("parent_id"))
    table_footnote_count = sum(1 for item in relationships if item.get("parent_relation") == "footnote_for_table")
    return {
        "schema_version": "structure-qa-v1",
        "doc_id": doc_ir.doc_id,
        "summary": {
            "page_count": len(doc_ir.pages),
            "block_counts": dict(block_counts),
            "table_count": len(table_blocks),
            "caption_count": caption_count,
            "caption_linked_count": caption_count - caption_orphan_count,
            "caption_orphan_count": caption_orphan_count,
            "footnote_count": footnote_count,
            "footnote_linked_count": footnote_count - footnote_orphan_count,
            "footnote_orphan_count": footnote_orphan_count,
            "table_footnote_count": table_footnote_count,
            "table_continuation_count": len(table_continuations),
            "relationship_count": linked_relationship_count,
            "relationship_warning_count": len(relationships) - linked_relationship_count,
            "warning_page_count": len(page_warnings),
            "page_boundary_fragment_count": boundary_count,
            "page_boundary_fragment_rate": round(boundary_count / possible_boundary_count, 4)
            if possible_boundary_count
            else 0.0,
        },
        "tables": table_blocks,
        "table_continuations": table_continuations,
        "relationships": relationships,
        "page_boundary_fragments": page_boundary_fragments,
        "page_warnings": page_warnings,
    }


def write_structure_qa(doc_ir: DocumentIR, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_structure_qa(doc_ir), ensure_ascii=False, indent=2), encoding="utf-8")


def _table_continuations(page_boundary_fragments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    continuations: list[dict[str, Any]] = []
    for fragment in page_boundary_fragments:
        reasons = fragment.get("reasons")
        if not isinstance(reasons, list) or "possible_table_continuation" not in reasons:
            continue
        previous_block_id = str(fragment.get("previous_block_id") or "")
        next_block_id = str(fragment.get("next_block_id") or "")
        if not previous_block_id or not next_block_id:
            continue
        continuations.append(
            {
                "continuation_id": str(fragment.get("boundary_id") or f"{previous_block_id}->{next_block_id}"),
                "pages_1based": fragment.get("pages_1based") or [],
                "previous_table_block_id": previous_block_id,
                "next_table_block_id": next_block_id,
                "severity": fragment.get("severity") or "medium",
                "previous_tail": fragment.get("previous_tail") or "",
                "next_head": fragment.get("next_head") or "",
                "suggested_handling": "merge_table_segments_before_translation_or_reconstruct_after_translation",
            }
        )
    return continuations
