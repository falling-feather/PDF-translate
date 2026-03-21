from __future__ import annotations

from pdf_translate.translators.base import TranslationRequest, Translator


class HybridTranslator:
    """MT 初稿 + LLM 术语/指代润色（阶段 B）；框架期可只接一侧。"""

    name = "hybrid"

    def __init__(
        self,
        *,
        machine: Translator,
        polisher: Translator | None,
    ) -> None:
        self.machine = machine
        self.polisher = polisher

    def translate(self, req: TranslationRequest) -> str:
        draft = self.machine.translate(req)
        if self.polisher is None:
            return draft
        polish_req = TranslationRequest(
            source_text=draft,
            glossary_excerpt=req.glossary_excerpt,
            prior_summaries=req.prior_summaries,
            style_notes=req.style_notes or "",
            source_lang=req.target_lang,
            target_lang=req.target_lang,
            prior_tail_zh=req.prior_tail_zh,
            continuation_hint=req.continuation_hint,
            prior_untranslated_continuation="",
            defer_source_tail_protocol=False,
        )
        return self.polisher.translate(polish_req)
