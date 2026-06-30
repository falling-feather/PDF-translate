from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BackendSpec:
    """Static capability metadata for a translation backend.

    The registry is value-only: it names setting fields but never stores values
    or secrets.
    """

    id: str
    label: str
    description: str
    kind: str
    aliases: tuple[str, ...] = ()
    default_enabled: bool = False
    supports_custom_api: bool = False
    supports_deferral: bool = True
    api_key_attr: str | None = None
    base_url_attr: str | None = None
    model_attr: str | None = None
    capability_tags: tuple[str, ...] = ()

    @property
    def requires_api_key(self) -> bool:
        return self.api_key_attr is not None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "kind": self.kind,
            "aliases": list(self.aliases),
            "default_enabled": self.default_enabled,
            "supports_custom_api": self.supports_custom_api,
            "supports_deferral": self.supports_deferral,
            "requires_api_key": self.requires_api_key,
            "api_key_setting": self.api_key_attr or "",
            "base_url_setting": self.base_url_attr or "",
            "model_setting": self.model_attr or "",
            "capability_tags": list(self.capability_tags),
        }


BACKEND_SPECS: tuple[BackendSpec, ...] = (
    BackendSpec(
        id="echo",
        label="echo（联调/测试）",
        description="不调用外部 API，用于本地冒烟、批量实验预跑和管线调试。",
        kind="local_test",
        aliases=("dry", "noop"),
        default_enabled=True,
        supports_custom_api=False,
        supports_deferral=False,
        capability_tags=("offline", "test", "zero_cost"),
    ),
    BackendSpec(
        id="deepseek",
        label="DeepSeek",
        description="DeepSeek 官方 OpenAI 兼容接口，适合作为默认文字翻译后端。",
        kind="external_api",
        default_enabled=True,
        supports_custom_api=True,
        api_key_attr="deepseek_api_key",
        base_url_attr="deepseek_base_url",
        model_attr="deepseek_model",
        capability_tags=("openai_compatible", "text_translation"),
    ),
    BackendSpec(
        id="openai",
        label="OpenAI 兼容",
        description="标准 OpenAI 兼容聊天接口，可接 OpenAI 或私有兼容网关。",
        kind="external_api",
        supports_custom_api=True,
        api_key_attr="openai_api_key",
        base_url_attr="openai_base_url",
        model_attr="openai_model",
        capability_tags=("openai_compatible", "text_translation"),
    ),
    BackendSpec(
        id="ollama",
        label="Ollama 本地模型",
        description="本地 Ollama OpenAI 兼容接口，用于低外部依赖或离线试验。",
        kind="local_model",
        supports_custom_api=True,
        base_url_attr="ollama_base_url",
        model_attr="ollama_model",
        capability_tags=("local_model", "openai_compatible", "optional_offline"),
    ),
    BackendSpec(
        id="deepl",
        label="DeepL",
        description="DeepL 机器翻译接口，适合作为机器翻译初稿或对照基线。",
        kind="external_api",
        supports_custom_api=True,
        supports_deferral=False,
        api_key_attr="deepl_api_key",
        base_url_attr="deepl_api_url",
        capability_tags=("machine_translation", "baseline"),
    ),
    BackendSpec(
        id="hybrid",
        label="Hybrid（DeepL + 润色）",
        description="DeepL 初稿加 OpenAI 兼容后端润色，适合机器翻译和 LLM 的对比实验。",
        kind="composite",
        supports_custom_api=False,
        supports_deferral=False,
        model_attr="openai_model",
        capability_tags=("machine_translation", "polish", "experiment"),
    ),
)

_SPECS_BY_ID: dict[str, BackendSpec] = {spec.id: spec for spec in BACKEND_SPECS}
_ALIASES: dict[str, str] = {
    alias: spec.id
    for spec in BACKEND_SPECS
    for alias in (spec.id, *spec.aliases)
}


def backend_ids() -> list[str]:
    return [spec.id for spec in BACKEND_SPECS]


def default_enabled_backend_ids() -> list[str]:
    return [spec.id for spec in BACKEND_SPECS if spec.default_enabled]


def custom_api_backend_ids() -> list[str]:
    return [spec.id for spec in BACKEND_SPECS if spec.supports_custom_api]


def backend_ui_labels() -> dict[str, str]:
    return {spec.id: spec.label for spec in BACKEND_SPECS}


def backend_catalog() -> list[dict[str, Any]]:
    return [spec.to_public_dict() for spec in BACKEND_SPECS]


def normalize_backend_id(raw: str | None) -> str:
    key = str(raw or "").strip().lower()
    if not key:
        raise ValueError("backend is empty")
    if key not in _ALIASES:
        raise ValueError(f"Unknown backend: {raw}")
    return _ALIASES[key]


def get_backend_spec(raw: str | None) -> BackendSpec:
    return _SPECS_BY_ID[normalize_backend_id(raw)]


def backend_choice_text() -> str:
    return "/".join(backend_ids())


def unknown_backend_detail(raw: str | None) -> str:
    return f"Unknown backend: {raw}. Choose {backend_choice_text()}."
