from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "repair-plan-v1"

_ISSUE_RULES = {
    "missing_translation": {
        "action": "translate_missing_chunk",
        "scope": "chunk",
        "executor": "translation_backend",
        "reason": "该块缺少译文，需要补译整个块。",
    },
    "missing_numbers": {
        "action": "rewrite_with_locked_tokens",
        "scope": "chunk",
        "executor": "translation_backend",
        "reason": "源文数字未全部保留，重译时必须锁定缺失数字。",
    },
    "missing_references": {
        "action": "rewrite_with_locked_tokens",
        "scope": "chunk",
        "executor": "translation_backend",
        "reason": "源文引用未全部保留，重译时必须锁定缺失引用。",
    },
    "missing_table_figure_tokens": {
        "action": "rewrite_with_locked_tokens",
        "scope": "chunk",
        "executor": "translation_backend",
        "reason": "图表编号未全部保留，重译时必须锁定图表编号。",
    },
    "missing_math_symbols": {
        "action": "rewrite_formula_context",
        "scope": "paragraph",
        "executor": "translation_backend",
        "reason": "公式或变量符号缺失，优先修复公式邻近段落。",
    },
    "missing_glossary_terms": {
        "action": "rewrite_with_glossary_terms",
        "scope": "chunk",
        "executor": "translation_backend",
        "reason": "术语库中的期望译名未出现在译文中，重译时必须注入并锁定术语。",
    },
    "missing_entity_tokens": {
        "action": "rewrite_with_entity_tokens",
        "scope": "chunk",
        "executor": "translation_backend",
        "reason": "源文实体候选未在译文中保留，重译时应锁定模型、数据集、机构或缩写等实体 token。",
    },
    "glossary_translation_conflict": {
        "action": "review_glossary_conflict",
        "scope": "glossary",
        "executor": "human_review",
        "reason": "同一英文术语存在多个候选译名，需要先人工确认术语再重译相关块。",
    },
    "table_shape_mismatch": {
        "action": "repair_table_shape",
        "scope": "table",
        "executor": "translation_backend",
        "reason": "译文表格形状与源表格不一致，需要按源表格维度重构。",
    },
    "duplicate_paragraphs": {
        "action": "deduplicate_overlap",
        "scope": "paragraph",
        "executor": "local_rule",
        "reason": "检测到重复段落，优先判断是否由重叠页或合并造成。",
    },
    "high_english_residual": {
        "action": "review_english_residual",
        "scope": "chunk",
        "executor": "human_or_translation_backend",
        "reason": "英文残留比例较高，需要判断是术语保留还是漏译。",
    },
}


def _priority(severity: str, issue_type: str) -> str:
    if issue_type in {"missing_translation", "table_shape_mismatch"}:
        return "P0"
    if severity == "high":
        return "P0"
    if severity == "medium":
        return "P1"
    return "P2"


def _issue_evidence(issue: dict[str, Any]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for key in ("tokens", "terms", "entities", "conflicts", "tables", "samples", "ratio", "detail"):
        if key in issue:
            evidence[key] = issue[key]
    return evidence


def build_repair_plan(qa_report: dict[str, Any]) -> dict[str, Any]:
    """Convert translation QA issues into executable repair candidates."""
    items: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    priority_counts: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()

    for chunk in qa_report.get("chunks", []):
        chunk_id = chunk.get("chunk_id")
        pages = chunk.get("pages_1based") or []
        for issue in chunk.get("issues", []):
            issue_type = str(issue.get("type") or "unknown")
            severity = str(issue.get("severity") or "low")
            rule = _ISSUE_RULES.get(
                issue_type,
                {
                    "action": "review_chunk",
                    "scope": "chunk",
                    "executor": "human_review",
                    "reason": "未知 QA 问题，需要人工复核。",
                },
            )
            priority = _priority(severity, issue_type)
            action = str(rule["action"])
            scope = str(rule["scope"])
            action_counts[action] += 1
            priority_counts[priority] += 1
            scope_counts[scope] += 1
            items.append(
                {
                    "repair_id": f"r{len(items):04d}",
                    "chunk_id": chunk_id,
                    "pages_1based": pages,
                    "priority": priority,
                    "issue_type": issue_type,
                    "severity": severity,
                    "action": action,
                    "scope": scope,
                    "executor": rule["executor"],
                    "reason": rule["reason"],
                    "evidence": _issue_evidence(issue),
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "summary": {
            "qa_schema_version": qa_report.get("schema_version"),
            "chunk_count": qa_report.get("summary", {}).get("chunk_count", 0),
            "repair_item_count": len(items),
            "action_counts": dict(action_counts),
            "priority_counts": dict(priority_counts),
            "scope_counts": dict(scope_counts),
        },
        "items": items,
    }


def repair_plan_to_markdown(plan: dict[str, Any]) -> str:
    summary = plan.get("summary", {})
    lines = [
        "# 局部修复计划",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| 源 QA 版本 | {summary.get('qa_schema_version') or '-'} |",
        f"| 块总数 | {summary.get('chunk_count', 0)} |",
        f"| 修复项总数 | {summary.get('repair_item_count', 0)} |",
        "",
    ]

    items = plan.get("items") or []
    if not items:
        lines.append("当前 QA 未生成局部修复项。")
        return "\n".join(lines).rstrip() + "\n"

    lines.extend(["## 修复项", ""])
    for item in items:
        pages = item.get("pages_1based") or []
        page_text = f"{pages[0]}-{pages[-1]}" if pages else "-"
        lines.append(
            f"### {item.get('repair_id')}｜{item.get('priority')}｜{item.get('chunk_id')}｜页 {page_text}"
        )
        lines.append(f"- 问题：`{item.get('issue_type')}` / `{item.get('severity')}`")
        lines.append(f"- 动作：`{item.get('action')}`，范围 `{item.get('scope')}`，执行器 `{item.get('executor')}`")
        lines.append(f"- 原因：{item.get('reason')}")
        evidence = item.get("evidence") or {}
        if evidence:
            lines.append(f"- 证据：{json.dumps(evidence, ensure_ascii=False)}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_repair_plan(qa_report: dict[str, Any], json_path: Path, markdown_path: Path) -> dict[str, Any]:
    plan = build_repair_plan(qa_report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(repair_plan_to_markdown(plan), encoding="utf-8")
    return plan
