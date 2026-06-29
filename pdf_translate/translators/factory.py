from __future__ import annotations

from pdf_translate.config import AppConfig
from pdf_translate.error_codes import PdfTranslateError, make_error_info
from pdf_translate.translators.deepl import DeepLTranslator
from pdf_translate.translators.echo import EchoTranslator
from pdf_translate.translators.hybrid import HybridTranslator
from pdf_translate.translators.ollama import OllamaTranslator
from pdf_translate.translators.openai_compatible import OpenAICompatibleTranslator, POLISH_SYSTEM_PROMPT
from pdf_translate.translators.base import Translator


def _config_error(code: str, detail: str, *, source: str) -> PdfTranslateError:
    return PdfTranslateError(make_error_info(code, detail=detail, source=source))


def build_translator(backend: str, cfg: AppConfig) -> Translator:
    b = backend.lower().strip()
    if b in ("echo", "dry", "noop"):
        return EchoTranslator()
    if b == "openai":
        if not cfg.openai_api_key:
            raise _config_error(
                "CONFIG_MISSING_API_KEY",
                "OPENAI_API_KEY is not configured.",
                source="translator:openai",
            )
        return OpenAICompatibleTranslator(
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_base_url,
            model=cfg.openai_model,
            timeout_s=cfg.http_timeout_s,
        )
    if b == "deepseek":
        if not cfg.deepseek_api_key:
            raise _config_error(
                "CONFIG_MISSING_API_KEY",
                (
                    "DEEPSEEK_API_KEY is not configured. Configure it in the admin "
                    "settings or server environment, then restart the service."
                ),
                source="translator:deepseek",
            )
        return OpenAICompatibleTranslator(
            api_key=cfg.deepseek_api_key,
            base_url=cfg.deepseek_base_url.rstrip("/"),
            model=cfg.deepseek_model,
            timeout_s=max(cfg.http_timeout_s, 180.0),
        )
    if b == "ollama":
        return OllamaTranslator(
            base_url=cfg.ollama_base_url,
            model=cfg.ollama_model,
            timeout_s=max(cfg.http_timeout_s, 300.0),
        )
    if b == "deepl":
        if not cfg.deepl_api_key:
            raise _config_error(
                "CONFIG_MISSING_API_KEY",
                "DEEPL_API_KEY is not configured.",
                source="translator:deepl",
            )
        return DeepLTranslator(
            api_key=cfg.deepl_api_key,
            api_url=cfg.deepl_api_url,
            timeout_s=cfg.http_timeout_s,
        )
    if b == "hybrid":
        if not cfg.deepl_api_key:
            raise _config_error(
                "CONFIG_MISSING_API_KEY",
                "hybrid backend requires DEEPL_API_KEY.",
                source="translator:hybrid",
            )
        if not cfg.openai_api_key:
            raise _config_error(
                "CONFIG_MISSING_API_KEY",
                "hybrid backend requires OPENAI_API_KEY for polishing.",
                source="translator:hybrid",
            )
        mt = DeepLTranslator(
            api_key=cfg.deepl_api_key,
            api_url=cfg.deepl_api_url,
            timeout_s=cfg.http_timeout_s,
        )
        polish = OpenAICompatibleTranslator(
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_base_url,
            model=cfg.openai_model,
            timeout_s=cfg.http_timeout_s,
            system_prompt=POLISH_SYSTEM_PROMPT,
        )
        return HybridTranslator(machine=mt, polisher=polish)
    raise _config_error(
        "CONFIG_INVALID_BACKEND",
        f"Unknown backend: {backend}. Choose echo/openai/deepseek/ollama/deepl/hybrid.",
        source="translator:factory",
    )
