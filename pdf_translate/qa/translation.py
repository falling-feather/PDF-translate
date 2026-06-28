from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pdf_translate.chunking import TextChunk
from pdf_translate.deferral_markers import strip_yaml_front_matter

SCHEMA_VERSION = "translation-qa-v1"

_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
_BRACKET_CITATION_RE = re.compile(r"\[[0-9,\-\s;]+\]")
_AUTHOR_YEAR_RE = re.compile(r"\([A-Z][A-Za-z\-]+(?:\s+et\s+al\.)?,\s*\d{4}[a-z]?\)")
_TABLE_FIGURE_RE = re.compile(
    r"\b(?P<label>Table|Fig(?:ure)?\.?)\s*(?P<num>\d+[A-Za-z]?)\b",
    re.I,
)
_MATH_SYMBOL_RE = re.compile(r"(≤|≥|±|≈|=|∑|∫|√|α|β|γ|λ|μ|σ)")


def _unique_in_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _source_text_for_qa(text: str) -> str:
    # Structure chunks add local provenance labels; these should not become QA invariants.
    lines = [line for line in text.splitlines() if not line.startswith("[第 ")]
    return "\n".join(lines)


def _numbers(text: str) -> list[str]:
    return _unique_in_order(_NUMBER_RE.findall(text))


def _references(text: str) -> list[str]:
    return _unique_in_order(_BRACKET_CITATION_RE.findall(text) + _AUTHOR_YEAR_RE.findall(text))


def _math_symbols(text: str) -> list[str]:
    return _unique_in_order(_MATH_SYMBOL_RE.findall(text))


def _table_figure_tokens(text: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for match in _TABLE_FIGURE_RE.finditer(text):
        raw_label = match.group("label")
        num = match.group("num")
        kind = "table" if raw_label.lower().startswith("table") else "figure"
        key = (kind, num)
        if key in seen:
            continue
        seen.add(key)
        out.append({"kind": kind, "num": num, "token": match.group(0)})
    return out


def _has_table_figure_token(target: str, token: dict[str, str]) -> bool:
    if token["token"] in target:
        return True
    num = re.escape(token["num"])
    if token["kind"] == "table":
        return bool(re.search(rf"表\s*{num}\b", target))
    return bool(re.search(rf"图\s*{num}\b", target))


def _markdown_table_shapes(text: str) -> list[dict[str, int]]:
    shapes: list[dict[str, int]] = []
    current_rows: list[list[str]] = []

    def flush() -> None:
        nonlocal current_rows
        if not current_rows:
            return
        data_rows = [
            row
            for row in current_rows
            if not row
            or not all(cell.replace("-", "").replace(":", "").strip() == "" for cell in row)
        ]
        if data_rows:
            shapes.append(
                {
                    "row_count": len(data_rows),
                    "column_count": max(len(row) for row in data_rows),
                }
            )
        current_rows = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            current_rows.append(cells)
            continue
        flush()
    flush()
    return shapes


def _english_residual_ratio(text: str) -> float:
    body = re.sub(r"`[^`]*`", "", text)
    letters = len(re.findall(r"[A-Za-z]", body))
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", body))
    visible = letters + zh_chars
    if visible == 0:
        return 0.0
    return round(letters / visible, 4)


def _duplicate_paragraphs(text: str) -> list[str]:
    paras = [
        re.sub(r"\s+", " ", p).strip()
        for p in re.split(r"\n\s*\n", text)
        if len(re.sub(r"\s+", " ", p).strip()) >= 60
    ]
    counts = Counter(paras)
    return [para[:160] for para, count in counts.items() if count > 1]


def _chunk_translation_text(chunk_dir: Path, chunk_id: str) -> str | None:
    path = chunk_dir / f"{chunk_id}.md"
    if not path.is_file():
        return None
    return strip_yaml_front_matter(path.read_text(encoding="utf-8")).strip()


def _glossary_terms(glossary: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not glossary:
        return []
    out: list[dict[str, Any]] = []
    for term in glossary.get("terms") or []:
        if not isinstance(term, dict):
            continue
        en = str(term.get("en") or "").strip()
        zh = str(term.get("zh") or "").strip()
        if not en or not zh:
            continue
        out.append(
            {
                "en": en,
                "zh": zh,
                "first_page": term.get("first_page"),
                "source": term.get("source"),
                "status": term.get("status"),
            }
        )
    return out


def _glossary_conflicts(
    glossary: dict[str, Any] | None,
    pending_review: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    by_en: dict[str, dict[str, Any]] = {}
    for term in (glossary or {}).get("terms") or []:
        if not isinstance(term, dict):
            continue
        en = str(term.get("en") or "").strip()
        zh = str(term.get("zh") or "").strip()
        if not en or not zh:
            continue
        if str(term.get("status") or "").strip().lower() == "rejected":
            continue
        key = en.lower()
        entry = by_en.setdefault(
            key,
            {
                "en": en,
                "translations": [],
                "sources": [],
                "first_pages": [],
            },
        )
        if zh not in entry["translations"]:
            entry["translations"].append(zh)
        source = term.get("source")
        if source and source not in entry["sources"]:
            entry["sources"].append(source)
        first_page = term.get("first_page")
        if first_page is not None and first_page not in entry["first_pages"]:
            entry["first_pages"].append(first_page)
    for entry in by_en.values():
        if len(entry["translations"]) > 1:
            conflicts.append(entry)

    seen_pending: set[tuple[str, tuple[str, ...]]] = set()
    for item in (pending_review or {}).get("items") or []:
        if not isinstance(item, dict) or item.get("type") != "glossary_conflict":
            continue
        if str(item.get("status") or "pending").lower() not in {"pending", "open", ""}:
            continue
        en = str(item.get("en") or "").strip()
        existing = [str(v).strip() for v in item.get("existing_zh") or [] if str(v).strip()]
        candidate = str(item.get("candidate_zh") or "").strip()
        translations = existing + ([candidate] if candidate and candidate not in existing else [])
        if not en or len(translations) < 2:
            continue
        key = (en.lower(), tuple(sorted(translations)))
        if key in seen_pending:
            continue
        seen_pending.add(key)
        conflicts.append(
            {
                "en": en,
                "translations": translations,
                "first_pages": [item.get("first_page")] if item.get("first_page") is not None else [],
                "sources": [item.get("source")] if item.get("source") else [],
                "status": "pending_review",
            }
        )
    return conflicts


def _missing_glossary_terms(
    source: str,
    target: str,
    glossary_terms: list[dict[str, Any]],
    conflict_en: set[str],
) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for term in glossary_terms:
        en = str(term.get("en") or "").strip()
        zh = str(term.get("zh") or "").strip()
        if not en or not zh:
            continue
        if str(term.get("status") or "").strip().lower() == "rejected":
            continue
        if en.lower() in conflict_en:
            continue
        if not re.search(re.escape(en), source, flags=re.I):
            continue
        if zh in target:
            continue
        missing.append(
            {
                "en": en,
                "expected_zh": zh,
                "first_page": term.get("first_page"),
                "source": term.get("source"),
            }
        )
    return missing


def _chunk_report(
    chunk: TextChunk,
    target_text: str | None,
    glossary_terms: list[dict[str, Any]],
    glossary_conflicts: list[dict[str, Any]],
) -> dict[str, Any]:
    source = _source_text_for_qa(chunk.text)
    pages_1based = [p + 1 for p in chunk.pages_0based]
    if target_text is None:
        return {
            "chunk_id": chunk.chunk_id,
            "pages_1based": pages_1based,
            "status": "missing_translation",
            "issues": [
                {
                    "type": "missing_translation",
                    "severity": "high",
                    "detail": "缺少该块译文文件。",
                }
            ],
            "metrics": {},
        }

    source_numbers = _numbers(source)
    missing_numbers = [token for token in source_numbers if token not in target_text]
    source_refs = _references(source)
    missing_refs = [token for token in source_refs if token not in target_text]
    table_figure_tokens = _table_figure_tokens(source)
    missing_table_figures = [
        token["token"] for token in table_figure_tokens if not _has_table_figure_token(target_text, token)
    ]
    source_symbols = _math_symbols(source)
    missing_symbols = [token for token in source_symbols if token not in target_text]
    source_tables = _markdown_table_shapes(source)
    target_tables = _markdown_table_shapes(target_text)
    table_shape_errors: list[dict[str, Any]] = []
    for idx, source_shape in enumerate(source_tables):
        target_shape = target_tables[idx] if idx < len(target_tables) else None
        if target_shape != source_shape:
            table_shape_errors.append(
                {
                    "table_index": idx,
                    "source": source_shape,
                    "target": target_shape,
                }
            )

    duplicates = _duplicate_paragraphs(target_text)
    english_ratio = _english_residual_ratio(target_text)
    source_conflicts = [
        conflict
        for conflict in glossary_conflicts
        if re.search(re.escape(str(conflict.get("en") or "")), source, flags=re.I)
    ]
    conflict_en = {str(conflict.get("en") or "").lower() for conflict in source_conflicts}
    missing_glossary = _missing_glossary_terms(source, target_text, glossary_terms, conflict_en)

    issues: list[dict[str, Any]] = []
    if missing_numbers:
        issues.append({"type": "missing_numbers", "severity": "high", "tokens": missing_numbers[:80]})
    if missing_refs:
        issues.append({"type": "missing_references", "severity": "high", "tokens": missing_refs[:80]})
    if missing_table_figures:
        issues.append(
            {
                "type": "missing_table_figure_tokens",
                "severity": "medium",
                "tokens": missing_table_figures[:80],
            }
        )
    if missing_symbols:
        issues.append({"type": "missing_math_symbols", "severity": "medium", "tokens": missing_symbols[:80]})
    if table_shape_errors:
        issues.append({"type": "table_shape_mismatch", "severity": "high", "tables": table_shape_errors})
    if duplicates:
        issues.append({"type": "duplicate_paragraphs", "severity": "medium", "samples": duplicates[:5]})
    if english_ratio >= 0.45:
        issues.append({"type": "high_english_residual", "severity": "low", "ratio": english_ratio})
    if missing_glossary:
        issues.append(
            {
                "type": "missing_glossary_terms",
                "severity": "medium",
                "terms": missing_glossary[:80],
            }
        )
    if source_conflicts:
        issues.append(
            {
                "type": "glossary_translation_conflict",
                "severity": "medium",
                "conflicts": source_conflicts[:40],
            }
        )

    return {
        "chunk_id": chunk.chunk_id,
        "pages_1based": pages_1based,
        "status": "ok" if not issues else "issues",
        "issues": issues,
        "metrics": {
            "source_number_count": len(source_numbers),
            "missing_number_count": len(missing_numbers),
            "source_reference_count": len(source_refs),
            "missing_reference_count": len(missing_refs),
            "table_shape_error_count": len(table_shape_errors),
            "english_residual_ratio": english_ratio,
            "duplicate_paragraph_count": len(duplicates),
            "missing_glossary_term_count": len(missing_glossary),
            "glossary_conflict_count": len(source_conflicts),
        },
    }


def build_translation_qa(
    chunks: list[TextChunk],
    chunk_dir: Path,
    *,
    glossary: dict[str, Any] | None = None,
    pending_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    terms = _glossary_terms(glossary)
    conflicts = _glossary_conflicts(glossary, pending_review)
    reports = [
        _chunk_report(chunk, _chunk_translation_text(chunk_dir, chunk.chunk_id), terms, conflicts)
        for chunk in chunks
    ]
    issue_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    translated_count = 0
    for report in reports:
        if report["status"] != "missing_translation":
            translated_count += 1
        for issue in report["issues"]:
            issue_counts[issue["type"]] += 1
            severity_counts[issue["severity"]] += 1

    max_english_ratio = max(
        (r.get("metrics", {}).get("english_residual_ratio", 0.0) for r in reports),
        default=0.0,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "summary": {
            "chunk_count": len(chunks),
            "translated_chunk_count": translated_count,
            "glossary_term_count": len(terms),
            "glossary_conflict_count": len(conflicts),
            "issue_count": sum(issue_counts.values()),
            "issue_counts": dict(issue_counts),
            "severity_counts": dict(severity_counts),
            "max_english_residual_ratio": max_english_ratio,
        },
        "glossary_conflicts": conflicts,
        "chunks": reports,
    }


def translation_qa_to_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# 翻译 QA 报告",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| 块总数 | {summary.get('chunk_count', 0)} |",
        f"| 已有译文块 | {summary.get('translated_chunk_count', 0)} |",
        f"| 术语库条目 | {summary.get('glossary_term_count', 0)} |",
        f"| 术语冲突 | {summary.get('glossary_conflict_count', 0)} |",
        f"| 问题总数 | {summary.get('issue_count', 0)} |",
        f"| 最高英文残留比例 | {summary.get('max_english_residual_ratio', 0)} |",
        "",
        "## 问题分布",
        "",
    ]
    issue_counts = summary.get("issue_counts") or {}
    if issue_counts:
        lines.extend(["| 类型 | 数量 |", "| --- | --- |"])
        for issue_type, count in sorted(issue_counts.items()):
            lines.append(f"| `{issue_type}` | {count} |")
    else:
        lines.append("未发现规则 QA 问题。")

    lines.extend(["", "## 块级明细", ""])
    for chunk in report.get("chunks", []):
        if not chunk.get("issues"):
            continue
        pages = chunk.get("pages_1based") or []
        page_text = f"{pages[0]}-{pages[-1]}" if pages else "-"
        lines.append(f"### {chunk.get('chunk_id')}（页 {page_text}）")
        for issue in chunk.get("issues", []):
            issue_type = issue.get("type")
            severity = issue.get("severity")
            detail = issue.get("detail")
            if detail:
                lines.append(f"- `{severity}` `{issue_type}`：{detail}")
            elif "tokens" in issue:
                tokens = ", ".join(str(token) for token in issue.get("tokens", [])[:20])
                lines.append(f"- `{severity}` `{issue_type}`：{tokens}")
            elif "terms" in issue:
                terms = ", ".join(
                    f"{term.get('en')} -> {term.get('expected_zh')}"
                    for term in issue.get("terms", [])[:20]
                    if isinstance(term, dict)
                )
                lines.append(f"- `{severity}` `{issue_type}`：{terms}")
            elif "conflicts" in issue:
                conflicts = ", ".join(
                    f"{conflict.get('en')} -> {' / '.join(str(v) for v in conflict.get('translations', []))}"
                    for conflict in issue.get("conflicts", [])[:20]
                    if isinstance(conflict, dict)
                )
                lines.append(f"- `{severity}` `{issue_type}`：{conflicts}")
            elif "ratio" in issue:
                lines.append(f"- `{severity}` `{issue_type}`：{issue.get('ratio')}")
            elif "tables" in issue:
                lines.append(f"- `{severity}` `{issue_type}`：{json.dumps(issue.get('tables'), ensure_ascii=False)}")
            elif "samples" in issue:
                lines.append(f"- `{severity}` `{issue_type}`：{json.dumps(issue.get('samples'), ensure_ascii=False)}")
            else:
                lines.append(f"- `{severity}` `{issue_type}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_translation_qa(
    chunks: list[TextChunk],
    chunk_dir: Path,
    json_path: Path,
    markdown_path: Path,
    *,
    glossary: dict[str, Any] | None = None,
    pending_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = build_translation_qa(
        chunks,
        chunk_dir,
        glossary=glossary,
        pending_review=pending_review,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(translation_qa_to_markdown(report), encoding="utf-8")
    return report
