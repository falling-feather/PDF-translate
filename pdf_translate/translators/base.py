from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class TranslationRequest:
    source_text: str
    glossary_excerpt: str
    prior_summaries: str
    style_notes: str
    source_lang: str = "English"
    target_lang: str = "Simplified Chinese"
    # 串联翻译：上一块译文的段尾（勿重复输出）；与 continuation_hint 配合处理页级重叠
    prior_tail_zh: str = ""
    continuation_hint: str = ""
    # 段尾顺延：上一块未译完的英文；defer_source_tail_protocol 为 True 时要求模型输出标识符+顺延原文
    prior_untranslated_continuation: str = ""
    defer_source_tail_protocol: bool = False


class Translator(Protocol):
    """可插拔翻译后端（OpenAI / Ollama / DeepL / Hybrid）。"""

    name: str

    def translate(self, req: TranslationRequest) -> str: ...
