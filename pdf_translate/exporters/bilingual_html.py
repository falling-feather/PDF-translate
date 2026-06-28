from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from pdf_translate.chunking import TextChunk
from pdf_translate.deferral_markers import (
    finalize_merged_translation_markdown,
    strip_yaml_front_matter,
)


def _chunk_translation_text(chunk_dir: Path, chunk_id: str) -> str:
    path = chunk_dir / f"{chunk_id}.md"
    if not path.is_file():
        return ""
    body = strip_yaml_front_matter(path.read_text(encoding="utf-8")).strip()
    return finalize_merged_translation_markdown(body).strip()


def _table_block(lines: list[str]) -> str | None:
    if not lines or not all(line.strip().startswith("|") and line.strip().endswith("|") for line in lines):
        return None
    rows = [[escape(cell.strip()) for cell in line.strip().strip("|").split("|")] for line in lines]
    data_rows = [
        row
        for row in rows
        if not row or not all(cell.replace("-", "").replace(":", "").strip() == "" for cell in row)
    ]
    if not data_rows:
        return None
    head = data_rows[0]
    body = data_rows[1:]
    parts = ["<table>", "<thead><tr>"]
    parts.extend(f"<th>{cell}</th>" for cell in head)
    parts.extend(["</tr></thead>", "<tbody>"])
    for row in body:
        parts.append("<tr>")
        parts.extend(f"<td>{cell}</td>" for cell in row)
        parts.append("</tr>")
    parts.extend(["</tbody>", "</table>"])
    return "".join(parts)


def _render_markdownish(text: str) -> str:
    blocks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        table = _table_block(current)
        if table:
            blocks.append(table)
        else:
            body = "<br>".join(escape(line) for line in current)
            blocks.append(f"<p>{body}</p>")
        current = []

    for line in text.splitlines():
        if not line.strip():
            flush()
            continue
        current.append(line)
    flush()
    return "\n".join(blocks) if blocks else "<p class=\"muted\">无内容</p>"


def _index_by_chunk(items: list[dict[str, Any]], key: str = "chunk_id") -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        chunk_id = item.get(key)
        if isinstance(chunk_id, str) and chunk_id:
            out.setdefault(chunk_id, []).append(item)
    return out


def _qa_issues_by_chunk(qa_report: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    if not qa_report:
        return {}
    chunks = qa_report.get("chunks") or []
    out: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        issues = chunk.get("issues") or []
        if isinstance(chunk_id, str) and issues:
            out[chunk_id] = list(issues)
    return out


def _badges(items: list[dict[str, Any]], *, label_key: str, class_prefix: str) -> str:
    if not items:
        return "<span class=\"badge ok\">OK</span>"
    badges = []
    for item in items[:8]:
        label = escape(str(item.get(label_key) or "unknown"))
        severity = escape(str(item.get("severity") or item.get("priority") or "info").lower())
        badges.append(f"<span class=\"badge {class_prefix}-{severity}\">{label}</span>")
    if len(items) > 8:
        badges.append(f"<span class=\"badge more\">+{len(items) - 8}</span>")
    return "".join(badges)


def _issue_list(issues: list[dict[str, Any]]) -> str:
    if not issues:
        return "<p class=\"muted\">未发现规则 QA 问题。</p>"
    parts = ["<ul>"]
    for issue in issues:
        issue_type = escape(str(issue.get("type") or "unknown"))
        severity = escape(str(issue.get("severity") or "unknown"))
        evidence = []
        for key in ("tokens", "tables", "samples", "ratio", "detail"):
            if key in issue:
                evidence.append(f"{escape(key)}={escape(str(issue[key]))}")
        evidence_text = "；".join(evidence)
        parts.append(f"<li><strong>{issue_type}</strong> <em>{severity}</em>")
        if evidence_text:
            parts.append(f"<br><code>{evidence_text}</code>")
        parts.append("</li>")
    parts.append("</ul>")
    return "".join(parts)


def _repair_list(items: list[dict[str, Any]]) -> str:
    if not items:
        return "<p class=\"muted\">未生成局部修复项。</p>"
    parts = ["<ul>"]
    for item in items:
        repair_id = escape(str(item.get("repair_id") or "-"))
        priority = escape(str(item.get("priority") or "-"))
        action = escape(str(item.get("action") or "-"))
        scope = escape(str(item.get("scope") or "-"))
        reason = escape(str(item.get("reason") or ""))
        parts.append(
            f"<li><strong>{repair_id}</strong> <em>{priority}</em> "
            f"<code>{action}</code> / <code>{scope}</code><br>{reason}</li>"
        )
    parts.append("</ul>")
    return "".join(parts)


def build_bilingual_html(
    chunks: list[TextChunk],
    chunk_dir: Path,
    *,
    qa_report: dict[str, Any] | None = None,
    repair_plan: dict[str, Any] | None = None,
    title: str = "双语对照译文",
) -> str:
    issues_by_chunk = _qa_issues_by_chunk(qa_report)
    repairs_by_chunk = _index_by_chunk((repair_plan or {}).get("items") or [])
    issue_count = sum(len(v) for v in issues_by_chunk.values())
    repair_count = sum(len(v) for v in repairs_by_chunk.values())
    safe_title = escape(title)

    sections: list[str] = []
    for chunk in chunks:
        pages = [p + 1 for p in chunk.pages_0based]
        page_text = f"{pages[0]}-{pages[-1]}" if pages else "-"
        translation = _chunk_translation_text(chunk_dir, chunk.chunk_id)
        issues = issues_by_chunk.get(chunk.chunk_id, [])
        repairs = repairs_by_chunk.get(chunk.chunk_id, [])
        issue_badges = _badges(issues, label_key="type", class_prefix="issue")
        repair_badges = _badges(repairs, label_key="action", class_prefix="repair")
        sections.append(
            f"""
<section class="chunk" id="{escape(chunk.chunk_id)}">
  <header class="chunk-head">
    <div>
      <h2>{escape(chunk.chunk_id)}</h2>
      <p class="muted">页码 {escape(page_text)} · 原文 {len(chunk.text)} 字符 · 译文 {len(translation)} 字符</p>
    </div>
    <div class="badge-row">{issue_badges}{repair_badges}</div>
  </header>
  <div class="columns">
    <article>
      <h3>原文</h3>
      {_render_markdownish(chunk.text)}
    </article>
    <article>
      <h3>译文</h3>
      {_render_markdownish(translation)}
    </article>
  </div>
  <details>
    <summary>QA 与修复建议</summary>
    <div class="details-grid">
      <div><h4>QA 问题</h4>{_issue_list(issues)}</div>
      <div><h4>修复计划</h4>{_repair_list(repairs)}</div>
    </div>
  </details>
</section>"""
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #697386;
      --line: #d9dee7;
      --accent: #0f766e;
      --warn: #b45309;
      --bad: #b91c1c;
      --soft: #eef6f5;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
      line-height: 1.65;
    }}
    header.hero {{
      padding: 28px 32px 20px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    h1, h2, h3, h4 {{ margin: 0; line-height: 1.25; }}
    .summary {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 12px; }}
    .summary span, .badge {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 10px;
      background: var(--soft);
      font-size: 12px;
      display: inline-flex;
      align-items: center;
      margin: 2px;
    }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 24px; }}
    .chunk {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-bottom: 18px;
      overflow: hidden;
    }}
    .chunk-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
    }}
    .badge-row {{ text-align: right; min-width: 220px; }}
    .badge.ok {{ color: var(--accent); }}
    .issue-high, .repair-p0 {{ color: var(--bad); border-color: #fecaca; background: #fff1f2; }}
    .issue-medium, .repair-p1 {{ color: var(--warn); border-color: #fed7aa; background: #fff7ed; }}
    .issue-low, .repair-p2 {{ color: #475569; background: #f8fafc; }}
    .columns {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }}
    article {{ padding: 18px; min-width: 0; }}
    article + article {{ border-left: 1px solid var(--line); }}
    article h3 {{ margin-bottom: 10px; color: var(--accent); }}
    p {{ margin: 0 0 12px; }}
    .muted {{ color: var(--muted); }}
    table {{ width: 100%; border-collapse: collapse; margin: 10px 0 14px; font-size: 14px; }}
    th, td {{ border: 1px solid var(--line); padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f3f6f8; text-align: left; }}
    details {{ border-top: 1px solid var(--line); padding: 12px 18px 16px; }}
    summary {{ cursor: pointer; color: var(--accent); font-weight: 600; }}
    .details-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-top: 12px; }}
    code {{ white-space: pre-wrap; word-break: break-word; }}
    @media (max-width: 900px) {{
      header.hero {{ position: static; padding: 20px; }}
      main {{ padding: 12px; }}
      .chunk-head, .columns, .details-grid {{ display: block; }}
      article + article {{ border-left: 0; border-top: 1px solid var(--line); }}
      .badge-row {{ text-align: left; margin-top: 10px; }}
    }}
  </style>
</head>
<body>
  <header class="hero">
    <h1>{safe_title}</h1>
    <div class="summary">
      <span>翻译块 {len(chunks)}</span>
      <span>QA 问题 {issue_count}</span>
      <span>修复项 {repair_count}</span>
    </div>
  </header>
  <main>
    {''.join(sections)}
  </main>
</body>
</html>
"""


def write_bilingual_html(
    chunks: list[TextChunk],
    chunk_dir: Path,
    path: Path,
    *,
    qa_report: dict[str, Any] | None = None,
    repair_plan: dict[str, Any] | None = None,
    title: str = "双语对照译文",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_bilingual_html(
            chunks,
            chunk_dir,
            qa_report=qa_report,
            repair_plan=repair_plan,
            title=title,
        ),
        encoding="utf-8",
    )
