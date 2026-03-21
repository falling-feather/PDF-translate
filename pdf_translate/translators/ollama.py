from __future__ import annotations

from pdf_translate.translators.base import TranslationRequest
from pdf_translate.translators.openai_compatible import OpenAICompatibleTranslator


class OllamaTranslator:
    """本地 Ollama OpenAI 兼容端点（默认 /v1）。"""

    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_s: float = 300.0,
    ) -> None:
        self._inner = OpenAICompatibleTranslator(
            api_key="ollama",
            base_url=base_url,
            model=model,
            timeout_s=timeout_s,
        )

    def translate(self, req: TranslationRequest) -> str:
        return self._inner.translate(req)
