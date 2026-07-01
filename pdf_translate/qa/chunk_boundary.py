from __future__ import annotations

import json
from collections import Counter
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


def _is_table_continuation(fragment: dict[str, Any]) -> bool:
    if str(fragment.get("continuation_kind") or "") == "table_continuation":
        return True
    reasons = fragment.get("reasons")
    return isinstance(reasons, list) and "possible_table_continuation" in reasons


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
            "approx_chars": len(chunk.text),
            "approx_tokens": int(getattr(chunk, "approx_tokens", 0) or 0),
            "split_reason": str(getattr(chunk, "split_reason", "") or ""),
            "budget_pressure": str(getattr(chunk, "budget_pressure", "") or ""),
            "budget_overflow_chars": int(getattr(chunk, "budget_overflow_chars", 0) or 0),
            "structural_relation_ids": [
                str(relation_id)
                for relation_id in getattr(chunk, "structural_relation_ids", [])
                if str(relation_id)
            ],
            "boundary_fragment_ids": {
                str(boundary_id)
                for boundary_id in getattr(chunk, "boundary_fragment_ids", [])
                if str(boundary_id)
            },
        }
        for chunk in chunks
    ]
    budget_split_reason_counts = Counter(
        row["split_reason"] for row in chunk_rows if row["split_reason"]
    )
    budget_pressure_counts = Counter(
        row["budget_pressure"] for row in chunk_rows if row["budget_pressure"]
    )
    budget_overflow_chunk_count = sum(1 for row in chunk_rows if row["budget_overflow_chars"] > 0)
    budget_overflow_char_total = sum(row["budget_overflow_chars"] for row in chunk_rows)
    structural_relation_protected_count = sum(len(row["structural_relation_ids"]) for row in chunk_rows)

    boundaries: list[dict[str, Any]] = []
    split_count = 0
    co_located_count = 0
    protected_count = 0
    high_risk_count = 0
    high_risk_split_count = 0
    table_continuation_boundary_count = 0
    table_continuation_protected_count = 0
    table_continuation_split_count = 0
    table_continuation_co_located_count = 0

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
        is_table_continuation = _is_table_continuation(fragment)
        if is_table_continuation:
            table_continuation_boundary_count += 1

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
            if is_table_continuation:
                table_continuation_protected_count += 1
                table_continuation_co_located_count += 1
        elif co_located_chunks:
            status = "co_located"
            co_located_count += 1
            if is_table_continuation:
                table_continuation_co_located_count += 1
        else:
            status = "split"
            split_count += 1
            if severity == "high":
                high_risk_split_count += 1
            if is_table_continuation:
                table_continuation_split_count += 1

        boundaries.append(
            {
                "boundary_id": boundary_id,
                "pages_1based": page_pair,
                "severity": severity,
                "continuation_kind": fragment.get("continuation_kind"),
                "stitch_action": fragment.get("stitch_action"),
                "is_table_continuation": is_table_continuation,
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
            "table_continuation_boundary_count": table_continuation_boundary_count,
            "table_continuation_protected_count": table_continuation_protected_count,
            "table_continuation_split_count": table_continuation_split_count,
            "table_continuation_co_located_count": table_continuation_co_located_count,
            "split_boundary_rate": round(split_count / boundary_count, 4) if boundary_count else 0.0,
            "protected_boundary_rate": round(protected_count / boundary_count, 4) if boundary_count else 0.0,
            "high_risk_split_rate": round(high_risk_split_count / high_risk_count, 4) if high_risk_count else 0.0,
            "table_continuation_split_rate": round(
                table_continuation_split_count / table_continuation_boundary_count,
                4,
            )
            if table_continuation_boundary_count
            else 0.0,
            "table_continuation_protected_rate": round(
                table_continuation_protected_count / table_continuation_boundary_count,
                4,
            )
            if table_continuation_boundary_count
            else 0.0,
            "budget_split_reason_counts": dict(sorted(budget_split_reason_counts.items())),
            "budget_pressure_counts": dict(sorted(budget_pressure_counts.items())),
            "budget_overflow_chunk_count": budget_overflow_chunk_count,
            "budget_overflow_char_total": budget_overflow_char_total,
            "structural_relation_protected_count": structural_relation_protected_count,
        },
        "chunks": [
            {
                "chunk_id": row["chunk_id"],
                "pages_1based": sorted(row["pages"]),
                "approx_chars": row["approx_chars"],
                "approx_tokens": row["approx_tokens"],
                "split_reason": row["split_reason"],
                "budget_pressure": row["budget_pressure"],
                "budget_overflow_chars": row["budget_overflow_chars"],
                "structural_relation_ids": row["structural_relation_ids"],
                "boundary_fragment_ids": sorted(row["boundary_fragment_ids"]),
            }
            for row in chunk_rows
        ],
        "boundaries": boundaries,
    }


def _rate(numerator: int, denominator: int) -> float:
    if not denominator:
        return 0.0
    return round(numerator / denominator, 4)


def build_chunk_strategy_comparison(
    strategies: dict[str, list[TextChunk]],
    structure_qa: dict[str, Any] | None,
    *,
    active_strategy: str | None = None,
    baseline_strategy: str = "page",
) -> dict[str, Any]:
    reports = {
        name: build_chunk_boundary_qa(chunks, structure_qa, pipeline_variant=name)
        for name, chunks in strategies.items()
    }
    strategy_summaries: dict[str, dict[str, Any]] = {
        name: report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
        for name, report in reports.items()
    }

    baseline_summary = strategy_summaries.get(baseline_strategy, {})
    baseline_split = int(baseline_summary.get("split_boundary_count") or 0)
    baseline_table_continuation_split = int(baseline_summary.get("table_continuation_split_count") or 0)
    best_strategy = None
    best_split_rate: float | None = None
    best_split_count: int | None = None
    for name, summary in strategy_summaries.items():
        split_rate = float(summary.get("split_boundary_rate") or 0.0)
        split_count = int(summary.get("split_boundary_count") or 0)
        if (
            best_strategy is None
            or split_rate < float(best_split_rate)
            or (split_rate == best_split_rate and split_count < int(best_split_count or 0))
        ):
            best_strategy = name
            best_split_rate = split_rate
            best_split_count = split_count

    boundary_rows: dict[str, dict[str, Any]] = {}
    for strategy_name, report in reports.items():
        for boundary in report.get("boundaries", []) or []:
            if not isinstance(boundary, dict):
                continue
            boundary_id = str(boundary.get("boundary_id") or "")
            if not boundary_id:
                continue
            row = boundary_rows.setdefault(
                boundary_id,
                {
                    "boundary_id": boundary_id,
                    "pages_1based": boundary.get("pages_1based") or [],
                    "severity": boundary.get("severity") or "unknown",
                    "reasons": boundary.get("reasons") or [],
                    "status_by_strategy": {},
                },
            )
            row["status_by_strategy"][strategy_name] = boundary.get("status") or "unknown"

    strategy_entries = []
    for name, summary in strategy_summaries.items():
        split_count = int(summary.get("split_boundary_count") or 0)
        split_delta = baseline_split - split_count if name != baseline_strategy else 0
        table_continuation_split_count = int(summary.get("table_continuation_split_count") or 0)
        table_continuation_split_delta = (
            baseline_table_continuation_split - table_continuation_split_count
            if name != baseline_strategy
            else 0
        )
        strategy_entries.append(
            {
                "strategy": name,
                "is_active": name == active_strategy,
                "is_baseline": name == baseline_strategy,
                "chunk_count": int(summary.get("chunk_count") or 0),
                "boundary_fragment_count": int(summary.get("boundary_fragment_count") or 0),
                "split_boundary_count": split_count,
                "protected_boundary_count": int(summary.get("protected_boundary_count") or 0),
                "co_located_boundary_count": int(summary.get("co_located_boundary_count") or 0),
                "high_risk_split_count": int(summary.get("high_risk_split_count") or 0),
                "table_continuation_boundary_count": int(summary.get("table_continuation_boundary_count") or 0),
                "table_continuation_protected_count": int(summary.get("table_continuation_protected_count") or 0),
                "table_continuation_split_count": table_continuation_split_count,
                "table_continuation_co_located_count": int(
                    summary.get("table_continuation_co_located_count") or 0
                ),
                "budget_overflow_chunk_count": int(summary.get("budget_overflow_chunk_count") or 0),
                "structural_relation_protected_count": int(
                    summary.get("structural_relation_protected_count") or 0
                ),
                "split_boundary_rate": float(summary.get("split_boundary_rate") or 0.0),
                "protected_boundary_rate": float(summary.get("protected_boundary_rate") or 0.0),
                "table_continuation_split_rate": float(summary.get("table_continuation_split_rate") or 0.0),
                "table_continuation_protected_rate": float(
                    summary.get("table_continuation_protected_rate") or 0.0
                ),
                "split_reduction_vs_baseline": split_delta,
                "split_reduction_rate_vs_baseline": _rate(split_delta, baseline_split),
                "table_continuation_split_reduction_vs_baseline": table_continuation_split_delta,
                "table_continuation_split_reduction_rate_vs_baseline": _rate(
                    table_continuation_split_delta,
                    baseline_table_continuation_split,
                ),
            }
        )
    strategy_entries.sort(key=lambda item: str(item["strategy"]))

    active_summary = strategy_summaries.get(active_strategy or "", {})
    active_split = int(active_summary.get("split_boundary_count") or 0)
    active_table_continuation_split = int(active_summary.get("table_continuation_split_count") or 0)
    return {
        "schema_version": "chunk-strategy-comparison-v1",
        "doc_id": (structure_qa or {}).get("doc_id") if isinstance(structure_qa, dict) else None,
        "active_strategy": active_strategy or "unknown",
        "baseline_strategy": baseline_strategy,
        "summary": {
            "strategy_count": len(strategies),
            "boundary_fragment_count": max(
                (int(summary.get("boundary_fragment_count") or 0) for summary in strategy_summaries.values()),
                default=0,
            ),
            "best_strategy_by_split_rate": best_strategy,
            "baseline_split_boundary_count": baseline_split,
            "active_split_boundary_count": active_split,
            "active_split_reduction_vs_baseline": baseline_split - active_split,
            "active_split_reduction_rate_vs_baseline": _rate(baseline_split - active_split, baseline_split),
            "baseline_table_continuation_split_count": baseline_table_continuation_split,
            "active_table_continuation_split_count": active_table_continuation_split,
            "active_table_continuation_split_reduction_vs_baseline": (
                baseline_table_continuation_split - active_table_continuation_split
            ),
            "active_table_continuation_split_reduction_rate_vs_baseline": _rate(
                baseline_table_continuation_split - active_table_continuation_split,
                baseline_table_continuation_split,
            ),
        },
        "strategies": strategy_entries,
        "boundaries": sorted(boundary_rows.values(), key=lambda item: str(item["boundary_id"])),
    }


def write_chunk_strategy_comparison(
    strategies: dict[str, list[TextChunk]],
    structure_qa: dict[str, Any] | None,
    path: Path,
    *,
    active_strategy: str | None = None,
    baseline_strategy: str = "page",
) -> dict[str, Any]:
    report = build_chunk_strategy_comparison(
        strategies,
        structure_qa,
        active_strategy=active_strategy,
        baseline_strategy=baseline_strategy,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


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
