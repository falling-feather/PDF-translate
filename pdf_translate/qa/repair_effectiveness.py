from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "repair-effectiveness-v1"

METRIC_KEYS = [
    "issue_count",
    "table_shape_error_count",
    "table_cell_token_error_count",
    "missing_table_locked_token_count",
    "missing_formula_token_count",
    "missing_equation_label_count",
    "missing_entity_token_count",
    "structure_relation_mismatch_count",
    "table_footnote_binding_mismatch_count",
]


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(str(value or "0").strip() or "0")
    except ValueError:
        return 0


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _summary(report: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(report, dict) and isinstance(report.get("summary"), dict):
        return report["summary"]
    return {}


def _counter_from_mapping(value: Any) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not isinstance(value, dict):
        return counter
    for key, count in value.items():
        counter[str(key)] += _as_int(count)
    return counter


def _chunk_issue_types(chunk: dict[str, Any]) -> list[str]:
    issue_types: list[str] = []
    for issue in chunk.get("issues") or []:
        if isinstance(issue, dict):
            issue_type = str(issue.get("type") or "").strip()
            if issue_type:
                issue_types.append(issue_type)
    return issue_types


def _chunks_by_id(report: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    chunks: dict[str, dict[str, Any]] = {}
    if not isinstance(report, dict):
        return chunks
    for index, chunk in enumerate(report.get("chunks") or []):
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunk_id") or f"chunk-{index:04d}")
        chunks[chunk_id] = chunk
    return chunks


def _comparison_status(before: int, after: int) -> str:
    if before > 0 and after == 0:
        return "resolved"
    if before == 0 and after > 0:
        return "regressed"
    if after < before:
        return "improved"
    if after > before:
        return "worsened"
    if after > 0:
        return "persisted"
    return "clean"


def _overall_status(
    before_issue_count: int,
    after_issue_count: int,
    issue_delta: int,
    new_issue_count: int,
    regressed_chunk_count: int,
) -> str:
    if new_issue_count > 0 or regressed_chunk_count > 0:
        return "improved_with_regressions" if issue_delta > 0 else "regressed"
    if issue_delta > 0:
        return "improved"
    if issue_delta == 0 and before_issue_count > 0:
        return "unchanged"
    if after_issue_count == 0:
        return "clean"
    return "needs_review"


def build_repair_effectiveness(
    before_qa: dict[str, Any],
    after_qa: dict[str, Any],
    *,
    repair_merge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare translation QA before and after local repair merge."""
    before_summary = _summary(before_qa)
    after_summary = _summary(after_qa)
    before_issue_count = _as_int(before_summary.get("issue_count"))
    after_issue_count = _as_int(after_summary.get("issue_count"))
    issue_delta = before_issue_count - after_issue_count

    metric_comparisons: dict[str, dict[str, Any]] = {}
    for key in METRIC_KEYS:
        before_value = _as_int(before_summary.get(key))
        after_value = _as_int(after_summary.get(key))
        delta = before_value - after_value
        metric_comparisons[key] = {
            "before": before_value,
            "after": after_value,
            "delta": delta,
            "reduction_rate": _rate(delta, before_value),
            "status": _comparison_status(before_value, after_value),
        }

    before_issue_counts = _counter_from_mapping(before_summary.get("issue_counts"))
    after_issue_counts = _counter_from_mapping(after_summary.get("issue_counts"))
    issue_type_comparisons: dict[str, dict[str, Any]] = {}
    for issue_type in sorted(set(before_issue_counts) | set(after_issue_counts)):
        before_value = before_issue_counts.get(issue_type, 0)
        after_value = after_issue_counts.get(issue_type, 0)
        delta = before_value - after_value
        issue_type_comparisons[issue_type] = {
            "before": before_value,
            "after": after_value,
            "delta": delta,
            "reduction_rate": _rate(delta, before_value),
            "status": _comparison_status(before_value, after_value),
        }

    before_severity_counts = _counter_from_mapping(before_summary.get("severity_counts"))
    after_severity_counts = _counter_from_mapping(after_summary.get("severity_counts"))
    severity_comparisons: dict[str, dict[str, Any]] = {}
    for severity in sorted(set(before_severity_counts) | set(after_severity_counts)):
        before_value = before_severity_counts.get(severity, 0)
        after_value = after_severity_counts.get(severity, 0)
        delta = before_value - after_value
        severity_comparisons[severity] = {
            "before": before_value,
            "after": after_value,
            "delta": delta,
            "reduction_rate": _rate(delta, before_value),
            "status": _comparison_status(before_value, after_value),
        }

    resolved_issue_count = sum(
        max(0, item["before"] - item["after"])
        for item in issue_type_comparisons.values()
    )
    persisted_issue_count = sum(
        min(item["before"], item["after"])
        for item in issue_type_comparisons.values()
        if item["before"] > 0
    )
    new_issue_count = sum(
        max(0, item["after"] - item["before"])
        for item in issue_type_comparisons.values()
    )

    before_chunks = _chunks_by_id(before_qa)
    after_chunks = _chunks_by_id(after_qa)
    chunk_comparisons: list[dict[str, Any]] = []
    improved_chunk_count = 0
    regressed_chunk_count = 0
    unchanged_problem_chunk_count = 0
    clean_chunk_count_after = 0

    for chunk_id in sorted(set(before_chunks) | set(after_chunks)):
        before_chunk = before_chunks.get(chunk_id, {})
        after_chunk = after_chunks.get(chunk_id, {})
        before_counter = Counter(_chunk_issue_types(before_chunk))
        after_counter = Counter(_chunk_issue_types(after_chunk))
        before_count = sum(before_counter.values())
        after_count = sum(after_counter.values())
        delta = before_count - after_count
        resolved_types = sorted(
            issue_type
            for issue_type in set(before_counter)
            if after_counter.get(issue_type, 0) < before_counter.get(issue_type, 0)
        )
        persisted_types = sorted(
            issue_type
            for issue_type in set(before_counter)
            if after_counter.get(issue_type, 0) > 0
        )
        new_types = sorted(
            issue_type
            for issue_type in set(after_counter)
            if after_counter.get(issue_type, 0) > before_counter.get(issue_type, 0)
        )

        status = _comparison_status(before_count, after_count)
        if status in {"improved", "resolved"}:
            improved_chunk_count += 1
        elif status in {"regressed", "worsened"}:
            regressed_chunk_count += 1
        elif status == "persisted":
            unchanged_problem_chunk_count += 1
        if after_count == 0:
            clean_chunk_count_after += 1

        pages = before_chunk.get("pages_1based") or after_chunk.get("pages_1based") or []
        chunk_comparisons.append(
            {
                "chunk_id": chunk_id,
                "pages_1based": pages,
                "before_issue_count": before_count,
                "after_issue_count": after_count,
                "delta": delta,
                "resolved_types": resolved_types,
                "persisted_types": persisted_types,
                "new_types": new_types,
                "status": status,
            }
        )

    merge_summary = _summary(repair_merge)
    status = _overall_status(
        before_issue_count,
        after_issue_count,
        issue_delta,
        new_issue_count,
        regressed_chunk_count,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "summary": {
            "before_qa_schema_version": before_qa.get("schema_version"),
            "after_qa_schema_version": after_qa.get("schema_version"),
            "repair_merge_schema_version": (
                repair_merge.get("schema_version") if isinstance(repair_merge, dict) else None
            ),
            "chunk_count": len(chunk_comparisons),
            "before_issue_count": before_issue_count,
            "after_issue_count": after_issue_count,
            "issue_delta": issue_delta,
            "issue_reduction_rate": _rate(issue_delta, before_issue_count),
            "resolved_issue_count": resolved_issue_count,
            "persisted_issue_count": persisted_issue_count,
            "new_issue_count": new_issue_count,
            "improved_chunk_count": improved_chunk_count,
            "regressed_chunk_count": regressed_chunk_count,
            "unchanged_problem_chunk_count": unchanged_problem_chunk_count,
            "clean_chunk_count_after": clean_chunk_count_after,
            "repair_merge_applied_count": _as_int(merge_summary.get("applied_count")),
            "repair_merge_manual_required_count": _as_int(
                merge_summary.get("manual_merge_required_count")
            ),
            "status": status,
        },
        "metric_comparisons": metric_comparisons,
        "issue_type_comparisons": issue_type_comparisons,
        "severity_comparisons": severity_comparisons,
        "chunk_comparisons": chunk_comparisons,
        "evidence_files": {
            "before_qa": "output/qa_report.json",
            "after_qa": "output/repair_merge_qa.json",
            "repair_merge": "output/repair_merge.json",
        },
    }


def repair_effectiveness_to_markdown(report: dict[str, Any]) -> str:
    summary = _summary(report)
    lines = [
        "# 局部修复效果对比",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| 状态 | {summary.get('status') or '-'} |",
        f"| 修复前问题数 | {summary.get('before_issue_count', 0)} |",
        f"| 修复后问题数 | {summary.get('after_issue_count', 0)} |",
        f"| 问题减少数 | {summary.get('issue_delta', 0)} |",
        f"| 问题减少率 | {summary.get('issue_reduction_rate', 0)} |",
        f"| 已解决问题实例 | {summary.get('resolved_issue_count', 0)} |",
        f"| 持续存在问题实例 | {summary.get('persisted_issue_count', 0)} |",
        f"| 新增问题实例 | {summary.get('new_issue_count', 0)} |",
        f"| 改善 chunk | {summary.get('improved_chunk_count', 0)} |",
        f"| 回归 chunk | {summary.get('regressed_chunk_count', 0)} |",
        f"| 修复后无问题 chunk | {summary.get('clean_chunk_count_after', 0)} |",
        "",
    ]

    issue_type_comparisons = (
        report.get("issue_type_comparisons")
        if isinstance(report.get("issue_type_comparisons"), dict)
        else {}
    )
    if issue_type_comparisons:
        lines.extend(
            [
                "## 问题类型变化",
                "",
                "| 类型 | 修复前 | 修复后 | 减少 | 状态 |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for issue_type, item in sorted(issue_type_comparisons.items()):
            if not isinstance(item, dict):
                continue
            lines.append(
                f"| `{issue_type}` | {item.get('before', 0)} | {item.get('after', 0)} | "
                f"{item.get('delta', 0)} | {item.get('status') or '-'} |"
            )
        lines.append("")

    chunk_comparisons = [
        item for item in report.get("chunk_comparisons") or [] if isinstance(item, dict)
    ]
    changed_chunks = [item for item in chunk_comparisons if item.get("status") != "clean"]
    if changed_chunks:
        lines.extend(
            [
                "## Chunk 对比",
                "",
                "| Chunk | 页码 | 修复前 | 修复后 | 减少 | 状态 |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for item in changed_chunks[:80]:
            pages = item.get("pages_1based") or []
            page_text = (
                f"{pages[0]}-{pages[-1]}"
                if len(pages) > 1
                else (str(pages[0]) if pages else "-")
            )
            lines.append(
                f"| `{item.get('chunk_id')}` | {page_text} | "
                f"{item.get('before_issue_count', 0)} | "
                f"{item.get('after_issue_count', 0)} | "
                f"{item.get('delta', 0)} | {item.get('status') or '-'} |"
            )
        if len(changed_chunks) > 80:
            lines.append(
                f"| ... | ... | ... | ... | ... | 另有 {len(changed_chunks) - 80} 个 chunk |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_repair_effectiveness(
    before_qa: dict[str, Any],
    after_qa: dict[str, Any],
    json_path: Path,
    markdown_path: Path,
    *,
    repair_merge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = build_repair_effectiveness(before_qa, after_qa, repair_merge=repair_merge)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(repair_effectiveness_to_markdown(report), encoding="utf-8")
    return report
