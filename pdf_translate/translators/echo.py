from __future__ import annotations

from pdf_translate.translators.base import TranslationRequest


class EchoTranslator:
    """无 API 时用于联调：不调用外网。"""

    name = "echo"

    def translate(self, req: TranslationRequest) -> str:
        head = req.source_text[:200].replace("\n", " ")
        return f"[ECHO 未翻译预览] {head}..."
