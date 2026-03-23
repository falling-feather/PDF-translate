from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class AppConfig:
    """Runtime config from environment (see SETUP_MANUAL.md)."""

    openai_api_key: str | None
    openai_base_url: str
    openai_model: str
    ollama_base_url: str
    ollama_model: str
    deepl_api_key: str | None
    deepl_api_url: str
    deepseek_api_key: str | None
    deepseek_base_url: str
    deepseek_model: str
    default_translator: str
    http_timeout_s: float

    @classmethod
    def from_env(cls) -> AppConfig:
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            deepl_api_key=os.getenv("DEEPL_API_KEY"),
            deepl_api_url=os.getenv("DEEPL_API_URL", "https://api-free.deepl.com/v2/translate"),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            default_translator=os.getenv("PDF_TRANSLATE_BACKEND", "deepseek"),
            http_timeout_s=float(os.getenv("HTTP_TIMEOUT_S", "240")),
        )


def project_root_from_workdir(work_dir: Path) -> Path:
    return work_dir.resolve()
