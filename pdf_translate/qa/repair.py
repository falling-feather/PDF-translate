from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from pdf_translate.chunking import TextChunk
from pdf_translate.deferral_markers import strip_yaml_front_matter
from pdf_translate.translators.base import TranslationRequest, Translator

SCHEMA_VERSION = "repair-plan-v1"
REQUEST_SCHEMA_VERSION = "repair-requests-v1"
RESULT_SCHEMA_VERSION = "repair-results-v1"
VALIDATION_SCHEMA_VERSION = "repair-validation-v1"

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
    "table_cell_token_mismatch": {
        "action": "repair_table_cell_tokens",
        "scope": "table_cell",
        "executor": "translation_backend",
        "reason": "译文表格单元格缺失源表格锁定 token，需要按单元格上下文局部重译或重构。",
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
    if issue_type in {"missing_translation", "table_shape_mismatch", "table_cell_token_mismatch"}:
        return "P0"
    if severity == "high":
        return "P0"
    if severity == "medium":
        return "P1"
    return "P2"


def _issue_evidence(issue: dict[str, Any]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for key in ("tokens", "terms", "entities", "conflicts", "tables", "cells", "samples", "ratio", "detail"):
        if key in issue:
            evidence[key] = issue[key]
    return evidence


def _clip_text(text: str, limit: int = 1200) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def _chunk_translation_text(chunk_dir: Path, chunk_id: str | None) -> str:
    if not chunk_id:
        return ""
    path = chunk_dir / f"{chunk_id}.md"
    if not path.is_file():
        return ""
    return strip_yaml_front_matter(path.read_text(encoding="utf-8")).strip()


def _locked_tokens_from_evidence(evidence: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    tokens.extend(str(token) for token in evidence.get("tokens") or [] if str(token))
    for term in evidence.get("terms") or []:
        if not isinstance(term, dict):
            continue
        tokens.extend(str(term.get(key) or "") for key in ("en", "expected_zh") if str(term.get(key) or ""))
    for entity in evidence.get("entities") or []:
        if isinstance(entity, dict) and str(entity.get("text") or ""):
            tokens.append(str(entity["text"]))
    for table in evidence.get("tables") or []:
        if isinstance(table, dict):
            tokens.extend(str(token) for token in table.get("numeric_tokens") or [] if str(token))
    for cell in evidence.get("cells") or []:
        if isinstance(cell, dict):
            tokens.extend(str(token) for token in cell.get("missing_tokens") or [] if str(token))
    return list(dict.fromkeys(token.strip() for token in tokens if token.strip()))


def _markdown_separator_row(row: list[str]) -> bool:
    return bool(row) and all(cell.replace("-", "").replace(":", "").strip() == "" for cell in row)


def _markdown_table_shapes(text: str) -> list[dict[str, int]]:
    tables: list[dict[str, int]] = []
    current_rows: list[list[str]] = []

    def flush() -> None:
        nonlocal current_rows
        if not current_rows:
            return
        data_rows = [row for row in current_rows if not _markdown_separator_row(row)]
        if len(data_rows) >= 2:
            column_count = max(len(row) for row in data_rows)
            tables.append({"row_count": len(data_rows), "column_count": column_count})
        current_rows = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and "|" in stripped[1:]:
            current_rows.append([cell.strip() for cell in stripped.strip("|").split("|")])
        else:
            flush()
    flush()
    return tables


def _expected_table_shapes_from_evidence(evidence: dict[str, Any]) -> list[dict[str, int]]:
    shapes: list[dict[str, int]] = []
    for table in evidence.get("tables") or []:
        if not isinstance(table, dict):
            continue
        source = table.get("source")
        if not isinstance(source, dict):
            continue
        row_count = int(source.get("row_count") or 0)
        column_count = int(source.get("column_count") or 0)
        if row_count >= 2 and column_count >= 2:
            shapes.append({"row_count": row_count, "column_count": column_count})
    return shapes


def _repair_result_text(result: dict[str, Any]) -> str:
    path_value = str(result.get("result_path") or "")
    if path_value:
        try:
            path = Path(path_value)
            if path.is_file():
                return path.read_text(encoding="utf-8")
        except OSError:
            pass
    return str(result.get("result_excerpt") or "")


def _repair_instruction(item: dict[str, Any], locked_tokens: list[str]) -> str:
    action = str(item.get("action") or "")
    if action == "repair_table_cell_tokens":
        return "按表格单元格证据修复译文表格；保持原表格行列与 Markdown 形状，缺失的锁定 token 必须回到对应单元格。"
    if action == "repair_table_shape":
        return "按源表格维度重构译文表格；保持表头、行头、数字和单位，不要把表格线性化为普通段落。"
    if action == "rewrite_with_glossary_terms":
        return "重译当前块并严格使用术语表中的期望中文译名；不要输出解释。"
    if action == "rewrite_with_entity_tokens":
        return "重译当前块并保留模型、数据集、机构、缩写等实体 token；不要输出解释。"
    if action == "rewrite_with_locked_tokens":
        return "重译当前块并原样保留所有锁定 token；不要输出解释。"
    if action == "rewrite_formula_context":
        return "修复公式或变量邻近段落；保留公式符号、编号和变量名。"
    if action == "translate_missing_chunk":
        return "补译缺失块；保持学术论文语气和结构。"
    if action == "deduplicate_overlap":
        return "复核并移除由重叠页或合并造成的重复译文，保留完整语义。"
    if action == "review_glossary_conflict":
        return "先人工确认术语译名，再重译相关块。"
    if action == "review_english_residual":
        return "判断英文残留是术语保留还是漏译；必要时重译相关句子。"
    if locked_tokens:
        return "修复当前 QA 问题，并确保锁定 token 原样保留。"
    return "复核并修复当前 QA 问题。"


def _backend_payload(
    item: dict[str, Any],
    source_text: str,
    current_translation: str,
    locked_tokens: list[str],
) -> dict[str, str]:
    evidence = item.get("evidence") or {}
    instruction = _repair_instruction(item, locked_tokens)
    parts = [
        "【修复目标】",
        instruction,
        "",
        "【问题类型】",
        str(item.get("issue_type") or "unknown"),
        "",
        "【源文范围】",
        _clip_text(source_text),
    ]
    if current_translation:
        parts.extend(["", "【当前译文】", _clip_text(current_translation)])
    if locked_tokens:
        parts.extend(["", "【必须保留的锁定 token】", ", ".join(locked_tokens[:80])])
    if evidence:
        parts.extend(["", "【QA 证据】", json.dumps(evidence, ensure_ascii=False)])
    parts.extend(["", "【输出要求】", "只输出修复后的中文译文或 Markdown 表格，不要解释原因，不要添加额外标题。"])
    return {
        "system_message": "你是学术论文翻译局部修复执行器，任务是最小范围修复译文错误并保持结构不变量。",
        "user_message": "\n".join(parts),
        "expected_output": "repaired_translation_fragment",
    }


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


def build_repair_requests(
    repair_plan: dict[str, Any],
    chunks: list[TextChunk],
    chunk_dir: Path,
) -> dict[str, Any]:
    """Turn repair plan items into backend/human executable request envelopes."""
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    requests: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    priority_counts: Counter[str] = Counter()
    executor_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()

    for item in repair_plan.get("items") or []:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id") or "")
        chunk = chunks_by_id.get(chunk_id)
        source_text = chunk.text if chunk else ""
        current_translation = _chunk_translation_text(chunk_dir, chunk_id)
        evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
        locked_tokens = _locked_tokens_from_evidence(evidence)
        executor = str(item.get("executor") or "human_review")
        action = str(item.get("action") or "review_chunk")
        priority = str(item.get("priority") or "P2")
        status = "ready_for_translation_backend" if "translation_backend" in executor else "needs_manual_review"

        action_counts[action] += 1
        priority_counts[priority] += 1
        executor_counts[executor] += 1
        status_counts[status] += 1
        requests.append(
            {
                "request_id": f"rq{len(requests):04d}",
                "repair_id": item.get("repair_id"),
                "chunk_id": chunk_id,
                "pages_1based": item.get("pages_1based") or [],
                "priority": priority,
                "issue_type": item.get("issue_type"),
                "action": action,
                "scope": item.get("scope"),
                "executor": executor,
                "status": status,
                "instruction": _repair_instruction(item, locked_tokens),
                "locked_tokens": locked_tokens,
                "source_excerpt": _clip_text(source_text),
                "current_translation_excerpt": _clip_text(current_translation),
                "evidence": evidence,
                "backend_payload": _backend_payload(item, source_text, current_translation, locked_tokens),
            }
        )

    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "summary": {
            "repair_plan_schema_version": repair_plan.get("schema_version"),
            "repair_item_count": repair_plan.get("summary", {}).get("repair_item_count", 0),
            "repair_request_count": len(requests),
            "ready_for_translation_backend_count": status_counts.get("ready_for_translation_backend", 0),
            "manual_review_request_count": status_counts.get("needs_manual_review", 0),
            "action_counts": dict(action_counts),
            "priority_counts": dict(priority_counts),
            "executor_counts": dict(executor_counts),
            "status_counts": dict(status_counts),
        },
        "requests": requests,
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


def repair_requests_to_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# 局部修复请求",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| 源修复项 | {summary.get('repair_item_count', 0)} |",
        f"| 修复请求 | {summary.get('repair_request_count', 0)} |",
        f"| 可交给翻译后端 | {summary.get('ready_for_translation_backend_count', 0)} |",
        f"| 需人工复核 | {summary.get('manual_review_request_count', 0)} |",
        "",
    ]
    requests = report.get("requests") or []
    if not requests:
        lines.append("当前没有生成局部修复请求。")
        return "\n".join(lines).rstrip() + "\n"

    lines.extend(["## 请求明细", ""])
    for request in requests:
        pages = request.get("pages_1based") or []
        page_text = f"{pages[0]}-{pages[-1]}" if pages else "-"
        locked = ", ".join(str(token) for token in request.get("locked_tokens", [])[:30])
        lines.append(
            f"### {request.get('request_id')}｜{request.get('priority')}｜{request.get('chunk_id')}｜页 {page_text}"
        )
        lines.append(f"- 状态：`{request.get('status')}`，执行器 `{request.get('executor')}`")
        lines.append(f"- 动作：`{request.get('action')}`，范围 `{request.get('scope')}`")
        lines.append(f"- 指令：{request.get('instruction')}")
        if locked:
            lines.append(f"- 锁定 token：{locked}")
        evidence = request.get("evidence") or {}
        if evidence:
            lines.append(f"- 证据：{json.dumps(evidence, ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _execute_one_repair_request(
    request: dict[str, Any],
    translator: Translator,
) -> str:
    payload = request.get("backend_payload") if isinstance(request.get("backend_payload"), dict) else {}
    source_text = str(payload.get("user_message") or "")
    style_notes = str(payload.get("system_message") or "局部修复执行器。")
    req = TranslationRequest(
        source_text=source_text,
        glossary_excerpt="",
        prior_summaries="",
        style_notes=style_notes,
    )
    return translator.translate(req).strip()


def build_repair_results(
    repair_requests: dict[str, Any],
    *,
    translator: Translator | None = None,
    execute: bool = False,
    max_requests: int | None = None,
    repairs_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute or account for repair requests without mutating chunk translations."""
    results: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    executed_count = 0

    for raw_request in repair_requests.get("requests") or []:
        if not isinstance(raw_request, dict):
            continue
        request_id = str(raw_request.get("request_id") or f"rq{len(results):04d}")
        action = str(raw_request.get("action") or "unknown")
        request_status = str(raw_request.get("status") or "")
        base = {
            "request_id": request_id,
            "repair_id": raw_request.get("repair_id"),
            "chunk_id": raw_request.get("chunk_id"),
            "pages_1based": raw_request.get("pages_1based") or [],
            "priority": raw_request.get("priority"),
            "issue_type": raw_request.get("issue_type"),
            "action": action,
            "scope": raw_request.get("scope"),
            "executor": raw_request.get("executor"),
        }
        action_counts[action] += 1

        if request_status != "ready_for_translation_backend":
            status = "skipped_not_ready"
            status_counts[status] += 1
            results.append({**base, "status": status, "reason": "该请求不是翻译后端可直接执行项。"})
            continue
        if not execute:
            status = "skipped_execution_disabled"
            status_counts[status] += 1
            results.append({**base, "status": status, "reason": "局部修复执行未开启。"})
            continue
        if max_requests is not None and executed_count >= max_requests:
            status = "skipped_limit"
            status_counts[status] += 1
            results.append({**base, "status": status, "reason": "达到本次局部修复执行数量上限。"})
            continue
        if translator is None:
            status = "failed"
            status_counts[status] += 1
            results.append({**base, "status": status, "error": "未提供翻译后端。"})
            continue

        try:
            repaired_text = _execute_one_repair_request(raw_request, translator)
            executed_count += 1
            result_path = ""
            if repairs_dir is not None:
                repairs_dir.mkdir(parents=True, exist_ok=True)
                out_path = repairs_dir / f"{request_id}.md"
                out_path.write_text(repaired_text + "\n", encoding="utf-8")
                result_path = out_path.as_posix()
            status = "succeeded"
            status_counts[status] += 1
            results.append(
                {
                    **base,
                    "status": status,
                    "result_path": result_path,
                    "result_excerpt": _clip_text(repaired_text, 800),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive boundary for external backends
            status = "failed"
            status_counts[status] += 1
            results.append({**base, "status": status, "error": str(exc)})

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "summary": {
            "repair_requests_schema_version": repair_requests.get("schema_version"),
            "repair_request_count": repair_requests.get("summary", {}).get("repair_request_count", 0),
            "execution_enabled": bool(execute),
            "execution_backend": getattr(translator, "name", None) if translator else None,
            "executed_request_count": executed_count,
            "succeeded_count": status_counts.get("succeeded", 0),
            "failed_count": status_counts.get("failed", 0),
            "skipped_count": sum(
                count
                for status, count in status_counts.items()
                if status.startswith("skipped_")
            ),
            "status_counts": dict(status_counts),
            "action_counts": dict(action_counts),
        },
        "results": results,
    }


def repair_results_to_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# 局部修复执行结果",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| 请求总数 | {summary.get('repair_request_count', 0)} |",
        f"| 执行开关 | {summary.get('execution_enabled', False)} |",
        f"| 执行后端 | {summary.get('execution_backend') or '-'} |",
        f"| 已执行请求 | {summary.get('executed_request_count', 0)} |",
        f"| 成功 | {summary.get('succeeded_count', 0)} |",
        f"| 失败 | {summary.get('failed_count', 0)} |",
        f"| 跳过 | {summary.get('skipped_count', 0)} |",
        "",
    ]
    results = report.get("results") or []
    if not results:
        lines.append("当前没有局部修复执行结果。")
        return "\n".join(lines).rstrip() + "\n"
    lines.extend(["## 结果明细", ""])
    for result in results:
        pages = result.get("pages_1based") or []
        page_text = f"{pages[0]}-{pages[-1]}" if pages else "-"
        lines.append(
            f"### {result.get('request_id')}｜{result.get('status')}｜{result.get('chunk_id')}｜页 {page_text}"
        )
        lines.append(f"- 动作：`{result.get('action')}`，范围 `{result.get('scope')}`")
        if result.get("reason"):
            lines.append(f"- 原因：{result.get('reason')}")
        if result.get("error"):
            lines.append(f"- 错误：{result.get('error')}")
        if result.get("result_path"):
            lines.append(f"- 结果文件：`{result.get('result_path')}`")
        if result.get("result_excerpt"):
            lines.append(f"- 结果预览：{result.get('result_excerpt')}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_repair_validation(
    repair_requests: dict[str, Any],
    repair_results: dict[str, Any],
) -> dict[str, Any]:
    """Validate candidate repair fragments against local invariants."""
    results_by_id = {
        str(result.get("request_id") or ""): result
        for result in repair_results.get("results") or []
        if isinstance(result, dict)
    }
    validations: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    checked_locked_token_count = 0
    missing_locked_token_count = 0
    table_shape_check_count = 0
    table_shape_passed_count = 0

    for raw_request in repair_requests.get("requests") or []:
        if not isinstance(raw_request, dict):
            continue
        request_id = str(raw_request.get("request_id") or f"rq{len(validations):04d}")
        action = str(raw_request.get("action") or "unknown")
        result = results_by_id.get(request_id)
        locked_tokens = [str(token) for token in raw_request.get("locked_tokens") or [] if str(token)]
        evidence = raw_request.get("evidence") if isinstance(raw_request.get("evidence"), dict) else {}
        expected_shapes = _expected_table_shapes_from_evidence(evidence)
        base = {
            "request_id": request_id,
            "repair_id": raw_request.get("repair_id"),
            "chunk_id": raw_request.get("chunk_id"),
            "pages_1based": raw_request.get("pages_1based") or [],
            "priority": raw_request.get("priority"),
            "issue_type": raw_request.get("issue_type"),
            "action": action,
            "scope": raw_request.get("scope"),
            "executor": raw_request.get("executor"),
            "locked_token_count": len(locked_tokens),
            "expected_table_shape_count": len(expected_shapes),
        }
        action_counts[action] += 1

        if not result:
            status = "skipped_missing_result"
            status_counts[status] += 1
            validations.append({**base, "status": status, "reason": "未找到对应的修复执行结果。"})
            continue
        if result.get("status") != "succeeded":
            status = "skipped_not_succeeded"
            status_counts[status] += 1
            validations.append(
                {
                    **base,
                    "status": status,
                    "result_status": result.get("status"),
                    "reason": "候选修复片段未成功生成，暂不做不变量验证。",
                }
            )
            continue

        repaired_text = _repair_result_text(result)
        if not repaired_text.strip():
            status = "failed"
            status_counts[status] += 1
            validations.append({**base, "status": status, "reason": "候选修复片段为空。"})
            continue

        missing_tokens = [token for token in locked_tokens if token not in repaired_text]
        checked_locked_token_count += len(locked_tokens)
        missing_locked_token_count += len(missing_tokens)

        target_shapes = _markdown_table_shapes(repaired_text)
        table_shape_errors: list[dict[str, Any]] = []
        for index, expected in enumerate(expected_shapes):
            table_shape_check_count += 1
            actual = target_shapes[index] if index < len(target_shapes) else None
            if actual == expected:
                table_shape_passed_count += 1
            else:
                table_shape_errors.append({"index": index, "expected": expected, "actual": actual})

        if missing_tokens or table_shape_errors:
            status = "failed"
        elif locked_tokens or expected_shapes:
            status = "passed"
        else:
            status = "unchecked"
        status_counts[status] += 1
        validations.append(
            {
                **base,
                "status": status,
                "missing_locked_tokens": missing_tokens,
                "table_shape_errors": table_shape_errors,
                "result_path": result.get("result_path") or "",
                "result_excerpt": _clip_text(repaired_text, 500),
            }
        )

    validated_count = status_counts.get("passed", 0) + status_counts.get("failed", 0) + status_counts.get("unchecked", 0)
    checked_and_passed = checked_locked_token_count - missing_locked_token_count
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "summary": {
            "repair_requests_schema_version": repair_requests.get("schema_version"),
            "repair_results_schema_version": repair_results.get("schema_version"),
            "repair_request_count": repair_requests.get("summary", {}).get("repair_request_count", 0),
            "validated_result_count": validated_count,
            "passed_count": status_counts.get("passed", 0),
            "failed_count": status_counts.get("failed", 0),
            "unchecked_count": status_counts.get("unchecked", 0),
            "skipped_count": sum(
                count
                for status, count in status_counts.items()
                if status.startswith("skipped_")
            ),
            "checked_locked_token_count": checked_locked_token_count,
            "missing_locked_token_count": missing_locked_token_count,
            "locked_token_pass_rate": round(checked_and_passed / checked_locked_token_count, 4)
            if checked_locked_token_count
            else 0.0,
            "table_shape_check_count": table_shape_check_count,
            "table_shape_passed_count": table_shape_passed_count,
            "table_shape_pass_rate": round(table_shape_passed_count / table_shape_check_count, 4)
            if table_shape_check_count
            else 0.0,
            "status_counts": dict(status_counts),
            "action_counts": dict(action_counts),
        },
        "validations": validations,
    }


def repair_validation_to_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# 局部修复验证",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| 修复请求 | {summary.get('repair_request_count', 0)} |",
        f"| 已验证候选 | {summary.get('validated_result_count', 0)} |",
        f"| 通过 | {summary.get('passed_count', 0)} |",
        f"| 失败 | {summary.get('failed_count', 0)} |",
        f"| 未配置本地检查 | {summary.get('unchecked_count', 0)} |",
        f"| 跳过 | {summary.get('skipped_count', 0)} |",
        f"| 锁定 token 通过率 | {summary.get('locked_token_pass_rate', 0)} |",
        f"| 表格形状通过率 | {summary.get('table_shape_pass_rate', 0)} |",
        "",
    ]
    validations = report.get("validations") or []
    if not validations:
        lines.append("当前没有局部修复验证记录。")
        return "\n".join(lines).rstrip() + "\n"

    lines.extend(["## 验证明细", ""])
    for item in validations:
        pages = item.get("pages_1based") or []
        page_text = f"{pages[0]}-{pages[-1]}" if pages else "-"
        lines.append(
            f"### {item.get('request_id')} · {item.get('status')} · {item.get('chunk_id')} · 页 {page_text}"
        )
        lines.append(f"- 动作：`{item.get('action')}`，范围 `{item.get('scope')}`")
        if item.get("reason"):
            lines.append(f"- 原因：{item.get('reason')}")
        missing = item.get("missing_locked_tokens") or []
        if missing:
            lines.append(f"- 缺失锁定 token：{', '.join(str(token) for token in missing[:30])}")
        shape_errors = item.get("table_shape_errors") or []
        if shape_errors:
            lines.append(f"- 表格形状异常：{json.dumps(shape_errors, ensure_ascii=False)}")
        if item.get("result_path"):
            lines.append(f"- 结果文件：`{item.get('result_path')}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_repair_plan(qa_report: dict[str, Any], json_path: Path, markdown_path: Path) -> dict[str, Any]:
    plan = build_repair_plan(qa_report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(repair_plan_to_markdown(plan), encoding="utf-8")
    return plan


def write_repair_requests(
    repair_plan: dict[str, Any],
    chunks: list[TextChunk],
    chunk_dir: Path,
    json_path: Path,
    markdown_path: Path,
) -> dict[str, Any]:
    report = build_repair_requests(repair_plan, chunks, chunk_dir)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(repair_requests_to_markdown(report), encoding="utf-8")
    return report


def write_repair_results(
    repair_requests: dict[str, Any],
    json_path: Path,
    markdown_path: Path,
    *,
    translator: Translator | None = None,
    execute: bool = False,
    max_requests: int | None = None,
) -> dict[str, Any]:
    report = build_repair_results(
        repair_requests,
        translator=translator,
        execute=execute,
        max_requests=max_requests,
        repairs_dir=json_path.parent / "repairs",
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(repair_results_to_markdown(report), encoding="utf-8")
    return report


def write_repair_validation(
    repair_requests: dict[str, Any],
    repair_results: dict[str, Any],
    json_path: Path,
    markdown_path: Path,
) -> dict[str, Any]:
    report = build_repair_validation(repair_requests, repair_results)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(repair_validation_to_markdown(report), encoding="utf-8")
    return report
