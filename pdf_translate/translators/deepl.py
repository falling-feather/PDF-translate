from __future__ import annotations

from typing import Any

import httpx

from pdf_translate.translators.base import TranslationRequest
from pdf_translate.translators.http_retry import call_with_http_retry


class DeepLTranslator:
    """DeepL HTTP API（机器翻译）；术语可后续与 glossary 对齐。"""

    name = "deepl"

    def __init__(
        self,
        *,
        api_key: str,
        api_url: str,
        timeout_s: float = 60.0,
        target_lang: str = "ZH",
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.timeout_s = max(30.0, timeout_s)
        self.target_lang = target_lang

    def translate(self, req: TranslationRequest) -> str:
        data: dict[str, Any] = {
            "auth_key": self.api_key,
            "text": req.source_text,
            "target_lang": self.target_lang,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        read_s = max(60.0, float(self.timeout_s))
        timeout = httpx.Timeout(connect=30.0, read=read_s, write=60.0, pool=30.0)

        def _do() -> str:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(self.api_url, data=data, headers=headers)
                r.raise_for_status()
                out = r.json()
            texts = out.get("translations") or []
            if not texts:
                raise RuntimeError(f"DeepL unexpected: {out}")
            return "\n".join(t.get("text", "") for t in texts).strip()

        return call_with_http_retry(_do, context="DeepL")
