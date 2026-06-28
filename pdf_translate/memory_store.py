from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import yaml

_glossary_write_lock = threading.Lock()

DEFAULT_GLOSSARY = {"terms": []}
DEFAULT_ENTITIES = {"entities": []}
DEFAULT_CHUNK_SUMMARIES: dict[str, Any] = {"chunks": []}
DEFAULT_PENDING = {"items": []}
DEFAULT_STYLE = {
    "tone": "学术、中性",
    "preserve_formulas": True,
    "notes": "",
}


def _normalize_term_key(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _load_json_or_default(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return json.loads(json.dumps(default, ensure_ascii=False))
    return json.loads(path.read_text(encoding="utf-8"))


def _pending_dedupe_key(item: dict[str, Any]) -> str:
    if item.get("dedupe_key"):
        return str(item["dedupe_key"])
    kind = str(item.get("type") or "pending")
    en = _normalize_term_key(str(item.get("en") or ""))
    zh = str(item.get("candidate_zh") or item.get("zh") or "").strip()
    return f"{kind}:{en}:{zh}"


class MemoryStore:
    """memory/ 目录：glossary、entities、chunk_summaries、style_notes、pending_review。"""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.glossary_path = self.root / "glossary.json"
        self.entities_path = self.root / "entities.json"
        self.chunk_summaries_path = self.root / "chunk_summaries.json"
        self.style_path = self.root / "style_notes.yaml"
        self.pending_path = self.root / "pending_review.json"
        self.running_summary_path = self.root / "running_summary.md"

    def deferred_carry_path(self) -> Path:
        """上一块预留的、尚未译完的英文尾巴（串联顺延）。"""
        return self.root / "deferred_source_carry.txt"

    def load_deferred_carry(self) -> str:
        p = self.deferred_carry_path()
        if not p.is_file():
            return ""
        return p.read_text(encoding="utf-8").strip()

    def save_deferred_carry(self, text: str) -> None:
        p = self.deferred_carry_path()
        self.root.mkdir(parents=True, exist_ok=True)
        t = text.strip()
        if t:
            p.write_text(t, encoding="utf-8")
        elif p.is_file():
            p.unlink()

    def ensure_files(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.glossary_path.exists():
            self.glossary_path.write_text(
                json.dumps(DEFAULT_GLOSSARY, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        if not self.entities_path.exists():
            self.entities_path.write_text(
                json.dumps(DEFAULT_ENTITIES, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        if not self.chunk_summaries_path.exists():
            self.chunk_summaries_path.write_text(
                json.dumps(DEFAULT_CHUNK_SUMMARIES, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        if not self.pending_path.exists():
            self.pending_path.write_text(
                json.dumps(DEFAULT_PENDING, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        if not self.style_path.exists():
            self.style_path.write_text(
                yaml.safe_dump(DEFAULT_STYLE, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
        if not self.running_summary_path.exists():
            self.running_summary_path.write_text(
                "# 叙事线索摘要（可由程序追加，也可手工编辑）\n\n", encoding="utf-8"
            )

    def load_glossary(self) -> dict[str, Any]:
        return json.loads(self.glossary_path.read_text(encoding="utf-8"))

    def load_pending_review(self) -> dict[str, Any]:
        return _load_json_or_default(self.pending_path, DEFAULT_PENDING)

    def _append_pending_items_locked(self, items: list[dict[str, Any]]) -> int:
        if not items:
            return 0
        data = self.load_pending_review()
        existing = list(data.get("items") or [])
        keys = {_pending_dedupe_key(item) for item in existing if isinstance(item, dict)}
        added = 0
        for item in items:
            key = _pending_dedupe_key(item)
            if key in keys:
                continue
            item = dict(item)
            item["dedupe_key"] = key
            existing.append(item)
            keys.add(key)
            added += 1
        if added:
            data["items"] = existing
            self.pending_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return added

    def merge_glossary_terms_from_survey(
        self,
        terms: list[dict[str, str]],
        *,
        first_page_1based: int,
        source: str = "survey",
    ) -> int:
        """将巡视产出的 en/zh 术语合并入 glossary；冲突写入 pending_review。"""
        if not terms:
            return 0
        with _glossary_write_lock:
            data = self.load_glossary()
            existing: list[dict[str, Any]] = list(data.get("terms") or [])
            seen_en = {_normalize_term_key(str(t.get("en", ""))) for t in existing if t.get("en")}
            added = 0
            pending: list[dict[str, Any]] = []
            for t in terms:
                en = str(t.get("en", "")).strip()
                zh = str(t.get("zh", "")).strip()
                if not en or not zh:
                    continue
                key = _normalize_term_key(en)
                if key in seen_en:
                    same_en = [
                        item
                        for item in existing
                        if _normalize_term_key(str(item.get("en") or "")) == key
                    ]
                    existing_zh = sorted(
                        {
                            str(item.get("zh") or "").strip()
                            for item in same_en
                            if str(item.get("zh") or "").strip()
                        }
                    )
                    if existing_zh and zh not in existing_zh:
                        pending.append(
                            {
                                "type": "glossary_conflict",
                                "status": "pending",
                                "en": en,
                                "existing_zh": existing_zh,
                                "candidate_zh": zh,
                                "first_page": int(first_page_1based),
                                "source": source,
                                "reason": "同一英文术语出现不同中文译名，需要人工确认。",
                            }
                        )
                    continue
                same_zh_terms = [
                    item
                    for item in existing
                    if str(item.get("zh") or "").strip() == zh
                    and _normalize_term_key(str(item.get("en") or "")) != key
                ]
                if same_zh_terms:
                    pending.append(
                        {
                            "type": "shared_translation_review",
                            "status": "pending",
                            "en": en,
                            "candidate_zh": zh,
                            "existing_en": [str(item.get("en") or "").strip() for item in same_zh_terms],
                            "first_page": int(first_page_1based),
                            "source": source,
                            "reason": "多个英文术语共享同一中文译名，需要确认是否为同义术语。",
                        }
                    )
                seen_en.add(key)
                existing.append(
                    {
                        "en": en,
                        "zh": zh,
                        "first_page": int(first_page_1based),
                        "source": source,
                        "status": "candidate",
                    }
                )
                added += 1
            data["terms"] = existing
            self.glossary_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            self._append_pending_items_locked(pending)
            return added

    def load_style_notes(self) -> dict[str, Any]:
        return yaml.safe_load(self.style_path.read_text(encoding="utf-8")) or {}

    def glossary_snippet_for_pages(
        self,
        start_page_1based: int,
        end_page_1based: int,
        *,
        max_terms: int = 40,
    ) -> str:
        """按 first_page 落在块内的术语注入；若无 first_page 则计入全局直至上限。"""
        data = self.load_glossary()
        terms = data.get("terms") or []
        picked: list[dict[str, Any]] = []
        in_range: list[dict[str, Any]] = []
        no_page: list[dict[str, Any]] = []
        for t in terms:
            if str(t.get("status") or "").strip().lower() == "rejected":
                continue
            fp = t.get("first_page")
            if fp is None:
                no_page.append(t)
            elif start_page_1based <= int(fp) <= end_page_1based:
                in_range.append(t)
        picked.extend(in_range)
        rest = max_terms - len(picked)
        if rest > 0:
            picked.extend(no_page[:rest])
        if not picked:
            return ""
        lines = []
        for t in picked:
            en = t.get("en", "")
            zh = t.get("zh", "")
            if en and zh:
                lines.append(f"- {en} → {zh}")
        return "\n".join(lines)

    def load_recent_summaries(self, max_chunks: int = 3) -> str:
        data = json.loads(self.chunk_summaries_path.read_text(encoding="utf-8"))
        chunks = data.get("chunks") or []
        tail = chunks[-max_chunks:] if max_chunks else chunks
        parts = []
        for c in tail:
            parts.append(f"[{c.get('chunk_id')}] {c.get('summary_zh', '')}")
        return "\n".join(parts)

    def load_prior_tail_zh(self) -> str:
        """上一块已写入的译文段尾（串联衔接用）。"""
        data = json.loads(self.chunk_summaries_path.read_text(encoding="utf-8"))
        chunks = data.get("chunks") or []
        if not chunks:
            return ""
        return str(chunks[-1].get("tail_zh") or "").strip()

    def append_chunk_summary(
        self,
        chunk_id: str,
        page_range_1based: tuple[int, int],
        summary_zh: str,
        *,
        tail_zh: str = "",
    ) -> None:
        data = json.loads(self.chunk_summaries_path.read_text(encoding="utf-8"))
        chunks = data.setdefault("chunks", [])
        chunks.append(
            {
                "chunk_id": chunk_id,
                "page_start_1based": page_range_1based[0],
                "page_end_1based": page_range_1based[1],
                "summary_zh": summary_zh,
                "tail_zh": (tail_zh or "").strip(),
            }
        )
        self.chunk_summaries_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        line = f"\n## {chunk_id} (pp.{page_range_1based[0]}–{page_range_1based[1]})\n\n{summary_zh}\n"
        with self.running_summary_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def add_pending_items(self, items: list[dict[str, Any]]) -> None:
        if not items:
            return
        with _glossary_write_lock:
            self._append_pending_items_locked(items)
