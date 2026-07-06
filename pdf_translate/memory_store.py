from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

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


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _pending_dedupe_key(item: dict[str, Any]) -> str:
    if item.get("dedupe_key"):
        return str(item["dedupe_key"])
    kind = str(item.get("type") or "pending")
    en = _normalize_term_key(str(item.get("en") or ""))
    zh = str(item.get("candidate_zh") or item.get("zh") or "").strip()
    return f"{kind}:{en}:{zh}"


def _normalized_review_decision(decision: str) -> Literal["confirm_candidate", "reject_candidate"]:
    text = decision.strip().lower()
    if text in {"confirm", "confirmed", "confirm_candidate", "accept", "approve"}:
        return "confirm_candidate"
    if text in {"reject", "rejected", "reject_candidate", "deny"}:
        return "reject_candidate"
    raise ValueError("decision must be confirm_candidate or reject_candidate")


def _review_timestamp(value: str | None) -> str:
    text = str(value or "").strip()
    return text or datetime.now(timezone.utc).isoformat()


def _normalize_confidence(value: float | int | str | None) -> float | None:
    if value is None:
        return None
    normalized = value.strip() if isinstance(value, str) else value
    if normalized == "":
        return None
    number = float(normalized)
    if number < 0 or number > 1:
        raise ValueError("confidence must be between 0 and 1")
    return round(number, 4)


def _normalize_candidate_zh_override(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("candidate_zh must not be empty when provided")
    if len(normalized) > 120:
        raise ValueError("candidate_zh must be at most 120 characters")
    return normalized


_SCOPE_SPLIT_RE = re.compile(r"[,，;；、|/\n]+")
_SCOPE_WORD_RE = re.compile(r"[^0-9a-z\u4e00-\u9fff]+")
_BLOCK_ID_RE = re.compile(r"^p\d+-b\d+$", re.I)
_SECTION_SCOPE_PREFIXES = {"section", "sec", "chapter", "heading", "章节", "小节", "标题"}
_STRUCTURE_SCOPE_PREFIXES = {"structure", "type", "block_type", "layout", "结构", "类型", "版式"}
_BLOCK_SCOPE_PREFIXES = {"block", "block_id", "id", "块"}
_STRUCTURE_ALIASES: dict[str, set[str]] = {
    "paragraph": {"paragraph", "body", "text", "正文", "段落"},
    "heading": {"heading", "title", "section", "标题", "章节"},
    "table": {"table", "tabular", "表格", "表"},
    "caption": {"caption", "figure caption", "table caption", "图注", "表注", "图表注"},
    "footnote": {"footnote", "note", "annotation", "脚注", "注释"},
    "formula": {"formula", "equation", "math", "公式", "方程"},
    "reference": {"reference", "references", "bibliography", "参考文献"},
    "image": {"image", "figure", "picture", "图像", "图片", "图"},
}
_STRUCTURE_ALIAS_LOOKUP = {
    alias: canonical
    for canonical, aliases in _STRUCTURE_ALIASES.items()
    for alias in aliases | {canonical}
}


def _scope_token(value: Any) -> str:
    return _SCOPE_WORD_RE.sub(" ", str(value or "").strip().casefold()).strip()


def _scope_raw_token(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def _scope_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_values = _SCOPE_SPLIT_RE.split(value)
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, dict)):
        raw_values = []
        for item in value:
            raw_values.extend(_scope_values(item))
    else:
        raw_values = [str(value)]
    out: list[str] = []
    seen: set[str] = set()
    for item in raw_values:
        token = _scope_raw_token(item)
        if token and token not in seen:
            out.append(token)
            seen.add(token)
    return out


def _scope_prefixed_value(token: str) -> tuple[str, str]:
    for separator in (":", "：", "="):
        if separator in token:
            prefix, value = token.split(separator, 1)
            return _scope_token(prefix), _scope_raw_token(value)
    return "", _scope_raw_token(token)


def _normalize_structure_type(value: Any) -> str:
    token = _scope_token(value)
    return _STRUCTURE_ALIAS_LOOKUP.get(token, token)


def _scope_text_matches(needle: str, contexts: set[str]) -> bool:
    if not needle or not contexts:
        return False
    needle_parts = needle.split()
    for context in contexts:
        if needle == context:
            return True
        context_parts = context.split()
        if needle_parts and all(part in context_parts for part in needle_parts):
            return True
        if len(needle) >= 4 and needle in context:
            return True
    return False


def _term_scope_matches(
    term: dict[str, Any],
    *,
    section_scope: Any = None,
    structure_types: Any = None,
    block_ids: Any = None,
) -> bool:
    tokens = _scope_values(term.get("section_scope"))
    if not tokens:
        return True

    section_contexts = {_scope_token(item) for item in _scope_values(section_scope)}
    type_contexts = {_normalize_structure_type(item) for item in _scope_values(structure_types)}
    block_contexts = {str(item).strip().casefold() for item in _scope_values(block_ids)}
    if not section_contexts and not type_contexts and not block_contexts:
        return True

    for raw_token in tokens:
        prefix, value = _scope_prefixed_value(raw_token)
        if not value:
            continue
        if prefix in _BLOCK_SCOPE_PREFIXES:
            if not block_contexts or value in block_contexts:
                return True
            continue
        if prefix in _STRUCTURE_SCOPE_PREFIXES:
            if not type_contexts or _normalize_structure_type(value) in type_contexts:
                return True
            continue
        if prefix in _SECTION_SCOPE_PREFIXES:
            if not section_contexts or _scope_text_matches(_scope_token(value), section_contexts):
                return True
            continue

        if _BLOCK_ID_RE.match(value):
            if not block_contexts or value in block_contexts:
                return True
            continue
        normalized_type = _normalize_structure_type(value)
        if normalized_type in _STRUCTURE_ALIASES:
            if not type_contexts or normalized_type in type_contexts:
                return True
            continue
        if _scope_text_matches(_scope_token(value), section_contexts):
            return True
        if not section_contexts:
            return True

    return False


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
        return _load_json_or_default(self.glossary_path, DEFAULT_GLOSSARY)

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
            _write_json_atomic(self.pending_path, data)
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
            _write_json_atomic(self.glossary_path, data)
            self._append_pending_items_locked(pending)
            return added

    def build_glossary_review_report(self) -> dict[str, Any]:
        """Build a web-facing glossary review report from memory files."""
        glossary_data = self.load_glossary()
        pending_data = self.load_pending_review()
        raw_terms = glossary_data.get("terms") or []
        raw_items = pending_data.get("items") or []

        terms: list[dict[str, Any]] = []
        for term in raw_terms:
            if not isinstance(term, dict):
                continue
            normalized = dict(term)
            normalized["en"] = str(term.get("en") or "").strip()
            normalized["zh"] = str(term.get("zh") or "").strip()
            normalized["status"] = str(term.get("status") or "candidate").strip() or "candidate"
            terms.append(normalized)

        pending_reviews: list[dict[str, Any]] = []
        type_counts: dict[str, int] = {}
        status_counts: dict[str, int] = {}
        reviewable_count = 0
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            normalized = dict(item)
            review_id = _pending_dedupe_key(normalized)
            item_type = str(normalized.get("type") or "pending").strip() or "pending"
            status = str(normalized.get("status") or "pending").strip() or "pending"
            normalized["review_id"] = review_id
            normalized["dedupe_key"] = review_id
            normalized["type"] = item_type
            normalized["status"] = status
            normalized["action_supported"] = item_type == "glossary_conflict" and status == "pending"
            if normalized["action_supported"]:
                reviewable_count += 1
            type_counts[item_type] = type_counts.get(item_type, 0) + 1
            status_counts[status] = status_counts.get(status, 0) + 1
            pending_reviews.append(normalized)

        summary = {
            "term_count": len(terms),
            "active_term_count": sum(1 for term in terms if term.get("status") != "rejected"),
            "pending_count": status_counts.get("pending", 0),
            "reviewable_count": reviewable_count,
            "glossary_conflict_count": type_counts.get("glossary_conflict", 0),
            "pending_glossary_conflict_count": sum(
                1
                for item in pending_reviews
                if item.get("type") == "glossary_conflict" and item.get("status") == "pending"
            ),
            "shared_translation_review_count": type_counts.get("shared_translation_review", 0),
            "confirmed_count": status_counts.get("confirmed", 0),
            "rejected_count": status_counts.get("rejected", 0),
        }
        return {
            "schema_version": "glossary-review-v1",
            "summary": summary,
            "terms": terms,
            "pending_reviews": pending_reviews,
        }

    def apply_glossary_review_decision(
        self,
        pending_key: str,
        decision: str,
        *,
        reviewer: str = "",
        reviewed_at: str | None = None,
        comment: str = "",
        confidence: float | int | str | None = None,
        section_scope: str = "",
        candidate_zh: Any = None,
    ) -> dict[str, Any]:
        """Apply a human decision for a pending glossary review item.

        The first supported review loop is glossary conflicts: confirming the
        candidate replaces the active translation for the same English term;
        rejecting the candidate closes the review without injecting it.
        """
        normalized_decision = _normalized_review_decision(decision)
        normalized_confidence = _normalize_confidence(confidence)
        with _glossary_write_lock:
            pending_data = self.load_pending_review()
            items = list(pending_data.get("items") or [])
            match_index: int | None = None
            for index, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                key = str(item.get("dedupe_key") or _pending_dedupe_key(item))
                if key == pending_key:
                    item["dedupe_key"] = key
                    match_index = index
                    break
            if match_index is None:
                raise ValueError(f"pending glossary review item not found: {pending_key}")

            item = dict(items[match_index])
            if item.get("type") != "glossary_conflict":
                raise ValueError("only glossary_conflict review items are supported")
            if str(item.get("status") or "pending").strip() != "pending":
                raise ValueError("pending glossary review item has already been reviewed")

            en = str(item.get("en") or "").strip()
            original_candidate_zh = str(item.get("candidate_zh") or "").strip()
            edited_candidate_zh = (
                _normalize_candidate_zh_override(candidate_zh)
                if normalized_decision == "confirm_candidate"
                else None
            )
            confirmed_zh = edited_candidate_zh or original_candidate_zh
            if normalized_decision == "confirm_candidate" and (not en or not confirmed_zh):
                raise ValueError("glossary_conflict item must contain en and candidate_zh")

            review_meta = {
                "review_decision": normalized_decision,
                "reviewed_by": str(reviewer or "").strip(),
                "reviewed_at": _review_timestamp(reviewed_at),
                "review_comment": str(comment or "").strip(),
            }
            if normalized_confidence is not None:
                review_meta["confidence"] = normalized_confidence
            normalized_section_scope = str(section_scope or "").strip()
            if normalized_section_scope:
                review_meta["section_scope"] = normalized_section_scope

            if normalized_decision == "confirm_candidate":
                if edited_candidate_zh and edited_candidate_zh != original_candidate_zh:
                    review_meta["original_candidate_zh"] = original_candidate_zh
                    review_meta["edited_candidate_zh"] = edited_candidate_zh
                self._confirm_glossary_candidate_locked(
                    en,
                    confirmed_zh,
                    first_page=item.get("first_page"),
                    source=item.get("source") or "human_review",
                    review_meta=review_meta,
                )
                item["status"] = "confirmed"
                item["confirmed_zh"] = confirmed_zh
                item["candidate_zh"] = confirmed_zh
            else:
                item["status"] = "rejected"

            item.update(review_meta)
            items[match_index] = item
            pending_data["items"] = items
            _write_json_atomic(self.pending_path, pending_data)
            return item

    def _confirm_glossary_candidate_locked(
        self,
        en: str,
        zh: str,
        *,
        first_page: Any = None,
        source: Any = None,
        review_meta: dict[str, Any],
    ) -> None:
        data = self.load_glossary()
        terms = list(data.get("terms") or [])
        key = _normalize_term_key(en)
        updated = False
        next_terms: list[dict[str, Any]] = []
        for term in terms:
            if not isinstance(term, dict):
                next_terms.append(term)
                continue
            if _normalize_term_key(str(term.get("en") or "")) != key:
                next_terms.append(term)
                continue
            if not updated:
                confirmed = dict(term)
                confirmed["zh"] = zh
                confirmed["status"] = "confirmed"
                if confirmed.get("first_page") is None and first_page is not None:
                    confirmed["first_page"] = first_page
                if source:
                    confirmed["source"] = source
                confirmed.update(review_meta)
                next_terms.append(confirmed)
                updated = True
                continue
            superseded = dict(term)
            superseded["status"] = "rejected"
            superseded["rejection_reason"] = "superseded_by_confirmed_glossary_review"
            superseded.update(review_meta)
            next_terms.append(superseded)
        if not updated:
            confirmed = {
                "en": en,
                "zh": zh,
                "first_page": first_page,
                "source": source or "human_review",
                "status": "confirmed",
            }
            confirmed.update(review_meta)
            next_terms.append(confirmed)
        data["terms"] = next_terms
        _write_json_atomic(self.glossary_path, data)

    def load_style_notes(self) -> dict[str, Any]:
        return yaml.safe_load(self.style_path.read_text(encoding="utf-8")) or {}

    def glossary_snippet_for_pages(
        self,
        start_page_1based: int,
        end_page_1based: int,
        *,
        max_terms: int = 40,
        section_scope: Any = None,
        structure_types: Any = None,
        block_ids: Any = None,
    ) -> str:
        """按页码和可选章节/结构上下文注入术语；无上下文时保留全局兼容行为。"""
        data = self.load_glossary()
        terms = data.get("terms") or []
        picked: list[dict[str, Any]] = []
        in_range: list[dict[str, Any]] = []
        no_page: list[dict[str, Any]] = []
        for t in terms:
            if str(t.get("status") or "").strip().lower() == "rejected":
                continue
            if not _term_scope_matches(
                t,
                section_scope=section_scope,
                structure_types=structure_types,
                block_ids=block_ids,
            ):
                continue
            fp = t.get("first_page")
            if fp is None:
                no_page.append(t)
                continue
            try:
                first_page = int(fp)
            except (TypeError, ValueError):
                no_page.append(t)
                continue
            if start_page_1based <= first_page <= end_page_1based:
                in_range.append(t)
        picked.extend(in_range[:max_terms])
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
