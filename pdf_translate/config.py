from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


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
    # 可选：译前巡视（硅基流动等 OpenAI 兼容端点）
    survey_enabled: bool
    siliconflow_api_key: str | None
    siliconflow_base_url: str
    siliconflow_survey_model: str
    siliconflow_vision_model: str
    survey_max_text_chars: int
    # 可选：全文/分章规划收束（后续阶段接入，仅占位）
    planner_enabled: bool
    planner_api_key: str | None
    planner_base_url: str
    planner_model: str

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
            survey_enabled=_env_bool("PDF_TRANSLATE_SURVEY_ENABLED", False),
            siliconflow_api_key=os.getenv("SILICONFLOW_API_KEY"),
            siliconflow_base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.com/v1"),
            siliconflow_survey_model=os.getenv("SILICONFLOW_SURVEY_MODEL", ""),
            siliconflow_vision_model=os.getenv("SILICONFLOW_VISION_MODEL", ""),
            survey_max_text_chars=int(os.getenv("PDF_TRANSLATE_SURVEY_MAX_CHARS", "12000")),
            planner_enabled=_env_bool("PDF_TRANSLATE_PLANNER_ENABLED", False),
            planner_api_key=os.getenv("PDF_TRANSLATE_PLANNER_API_KEY"),
            planner_base_url=os.getenv("PDF_TRANSLATE_PLANNER_BASE_URL", "https://api.siliconflow.com/v1"),
            planner_model=os.getenv("PDF_TRANSLATE_PLANNER_MODEL", ""),
        )


def project_root_from_workdir(work_dir: Path) -> Path:
    return work_dir.resolve()
