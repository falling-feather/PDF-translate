from __future__ import annotations

from pdf_translate.config import AppConfig
from pdf_translate.translators.deepl import DeepLTranslator
from pdf_translate.translators.echo import EchoTranslator
from pdf_translate.translators.hybrid import HybridTranslator
from pdf_translate.translators.ollama import OllamaTranslator
from pdf_translate.translators.openai_compatible import OpenAICompatibleTranslator, POLISH_SYSTEM_PROMPT
from pdf_translate.translators.base import Translator


def build_translator(backend: str, cfg: AppConfig) -> Translator:
    b = backend.lower().strip()
    if b in ("echo", "dry", "noop"):
        return EchoTranslator()
    if b == "openai":
        if not cfg.openai_api_key:
            raise ValueError("OPENAI_API_KEY 未设置，见 SETUP_MANUAL.md")
        return OpenAICompatibleTranslator(
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_base_url,
            model=cfg.openai_model,
            timeout_s=cfg.http_timeout_s,
        )
    if b == "deepseek":
        if not cfg.deepseek_api_key:
            raise ValueError(
                "未配置 DeepSeek：请使用管理员账号登录，在「管理后台 → API 与策略」填写 DeepSeek API Key，"
                "或在服务器环境变量中设置 DEEPSEEK_API_KEY 后重启服务。"
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
            raise ValueError("DEEPL_API_KEY 未设置，见 SETUP_MANUAL.md")
        return DeepLTranslator(
            api_key=cfg.deepl_api_key,
            api_url=cfg.deepl_api_url,
            timeout_s=cfg.http_timeout_s,
        )
    if b == "hybrid":
        if not cfg.deepl_api_key:
            raise ValueError("hybrid 需要 DEEPL_API_KEY，见 SETUP_MANUAL.md")
        if not cfg.openai_api_key:
            raise ValueError("hybrid 需要 OPENAI_API_KEY 作为润色端，见 SETUP_MANUAL.md")
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
    raise ValueError(f"未知后端: {backend}，可选 echo/openai/deepseek/ollama/deepl/hybrid")
