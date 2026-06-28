from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pdf_translate.chunking import TextChunk

SCHEMA_VERSION = "chunk-boundary-qa-v1"


def _chunk_pages_1based(chunk: TextChunk) -> set[int]:
    return {int(page) + 1 for page in chunk.pages_0based}


def _fragment_id(fragment: dict[str, Any]) -> str:
    boundary_id = str(fragment.get("boundary_id") or "").strip()
    if boundary_id:
        return boundary_id
    pages = fragment.get("pages_1based")
    if isinstance(pages, list) and len(pages) == 2:
        return f"p{pages[0]}-p{pages[1]}"
    return "unknown"


def build_chunk_boundary_qa(
    chunks: list[TextChunk],
    structure_qa: dict[str, Any] | None,
    *,
    pipeline_variant: str | None = None,
) -> dict[str, Any]:
    """Report how the active chunk strategy handled risky page boundaries."""
    fragments = []
    if isinstance(structure_qa, dict) and isinstance(structure_qa.get("page_boundary_fragments"), list):
        fragments = [
            item for item in structure_qa.get("page_boundary_fragments", [])
            if isinstance(item, dict)
        ]

    chunk_rows = [
        {
            "chunk_id": chunk.chunk_id,
            "pages": _chunk_pages_1based(chunk),
            "boundary_fragment_ids": {
                str(boundary_id)
                for boundary_id in getattr(chunk, "boundary_fragment_ids", [])
                if str(boundary_id)
            },
        }
        for chunk in chunks
    ]

    boundaries: list[dict[str, Any]] = []
    split_count = 0
    co_located_count = 0
    protected_count = 0
    high_risk_count = 0
    high_risk_split_count = 0

    for fragment in fragments:
        pages = fragment.get("pages_1based")
        if not isinstance(pages, list) or len(pages) != 2:
            continue
        try:
            page_pair = [int(pages[0]), int(pages[1])]
        except (TypeError, ValueError):
            continue
        boundary_id = _fragment_id(fragment)
        severity = str(fragment.get("severity") or "unknown")
        if severity == "high":
            high_risk_count += 1

        co_located_chunks = [
            row["chunk_id"]
            for row in chunk_rows
            if page_pair[0] in row["pages"] and page_pair[1] in row["pages"]
        ]
        protected_chunks = [
            row["chunk_id"]
            for row in chunk_rows
            if boundary_id in row["boundary_fragment_ids"]
        ]
        previous_page_chunks = [
            row["chunk_id"]
            for row in chunk_rows
            if page_pair[0] in row["pages"]
        ]
        next_page_chunks = [
            row["chunk_id"]
            for row in chunk_rows
            if page_pair[1] in row["pages"]
        ]

        if protected_chunks:
            status = "protected"
            protected_count += 1
            co_located_count += 1
        elif co_located_chunks:
            status = "co_located"
            co_located_count += 1
        else:
            status = "split"
            split_count += 1
            if severity == "high":
                high_risk_split_count += 1

        boundaries.append(
            {
                "boundary_id": boundary_id,
                "pages_1based": page_pair,
                "severity": severity,
                "status": status,
                "co_located_chunk_ids": co_located_chunks,
                "protected_by_chunk_ids": protected_chunks,
                "previous_page_chunk_ids": previous_page_chunks,
                "next_page_chunk_ids": next_page_chunks,
                "previous_block_id": fragment.get("previous_block_id"),
                "next_block_id": fragment.get("next_block_id"),
                "reasons": fragment.get("reasons") or [],
                "suggested_handling": fragment.get("suggested_handling"),
            }
        )

    boundary_count = len(boundaries)
    return {
        "schema_version": SCHEMA_VERSION,
        "doc_id": (structure_qa or {}).get("doc_id") if isinstance(structure_qa, dict) else None,
        "pipeline_variant": pipeline_variant or "unknown",
        "summary": {
            "chunk_count": len(chunks),
            "boundary_fragment_count": boundary_count,
            "co_located_boundary_count": co_located_count,
            "split_boundary_count": split_count,
            "protected_boundary_count": protected_count,
            "high_risk_boundary_count": high_risk_count,
            "high_risk_split_count": high_risk_split_count,
            "split_boundary_rate": round(split_count / boundary_count, 4) if boundary_count else 0.0,
            "protected_boundary_rate": round(protected_count / boundary_count, 4) if boundary_count else 0.0,
            "high_risk_split_rate": round(high_risk_split_count / high_risk_count, 4) if high_risk_count else 0.0,
        },
        "boundaries": boundaries,
    }


def write_chunk_boundary_qa(
    chunks: list[TextChunk],
    structure_qa: dict[str, Any] | None,
    path: Path,
    *,
    pipeline_variant: str | None = None,
) -> dict[str, Any]:
    report = build_chunk_boundary_qa(
        chunks,
        structure_qa,
        pipeline_variant=pipeline_variant,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
