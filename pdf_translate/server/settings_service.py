from __future__ import annotations

from pdf_translate.config import AppConfig
from pdf_translate.server import database

DEFAULT_ENABLED = ["echo", "openai", "deepseek", "ollama", "deepl", "hybrid"]


def _coalesce(kv_val: str | None, base_val: str | None) -> str | None:
    if kv_val is not None and str(kv_val).strip() != "":
        return str(kv_val).strip()
    return base_val


def _coalesce_str(kv_val: str | None, base_val: str) -> str:
    if kv_val is not None and str(kv_val).strip() != "":
        return str(kv_val).strip()
    return base_val


def effective_app_config() -> AppConfig:
    base = AppConfig.from_env()
    return AppConfig(
        openai_api_key=_coalesce(database.kv_get("openai_api_key"), base.openai_api_key),
        openai_base_url=_coalesce_str(database.kv_get("openai_base_url"), base.openai_base_url),
        openai_model=_coalesce_str(database.kv_get("openai_model"), base.openai_model),
        ollama_base_url=_coalesce_str(database.kv_get("ollama_base_url"), base.ollama_base_url),
        ollama_model=_coalesce_str(database.kv_get("ollama_model"), base.ollama_model),
        deepl_api_key=_coalesce(database.kv_get("deepl_api_key"), base.deepl_api_key),
        deepl_api_url=_coalesce_str(database.kv_get("deepl_api_url"), base.deepl_api_url),
        deepseek_api_key=_coalesce(database.kv_get("deepseek_api_key"), base.deepseek_api_key),
        deepseek_base_url=_coalesce_str(database.kv_get("deepseek_base_url"), base.deepseek_base_url),
        deepseek_model=_coalesce_str(database.kv_get("deepseek_model"), base.deepseek_model),
        default_translator=_coalesce_str(database.kv_get("default_backend"), base.default_translator),
        http_timeout_s=_parse_timeout(base),
    )


def _parse_timeout(base: AppConfig) -> float:
    raw = database.kv_get("http_timeout_s")
    s = _coalesce_str(raw, str(int(base.http_timeout_s)))
    try:
        return float(s)
    except ValueError:
        return base.http_timeout_s


def enabled_backends() -> list[str]:
    raw = database.kv_get_json("enabled_backends", None)
    if isinstance(raw, list) and raw:
        return [str(x).lower().strip() for x in raw if str(x).strip()]
    return list(DEFAULT_ENABLED)


def assert_backend_allowed(backend: str | None, default_backend: str) -> str:
    allowed = set(enabled_backends())
    b = (backend or default_backend or "echo").lower().strip()
    if b not in allowed:
        raise ValueError(f"后端「{b}」未在管理端启用")
    return b


def admin_settings_snapshot() -> dict:
    keys = [
        "openai_api_key",
        "openai_base_url",
        "openai_model",
        "ollama_base_url",
        "ollama_model",
        "deepl_api_key",
        "deepl_api_url",
        "deepseek_api_key",
        "deepseek_base_url",
        "deepseek_model",
        "default_backend",
        "http_timeout_s",
        "registration_open",
    ]
    out: dict = {}
    for k in keys:
        v = database.kv_get(k)
        if v is not None:
            out[k] = v
    out["enabled_backends"] = enabled_backends()
    out["registration_open"] = database.registration_open()
    return out


def apply_admin_settings(patch: dict) -> None:
    if "enabled_backends" in patch and patch["enabled_backends"] is not None:
        database.kv_set_json("enabled_backends", patch["enabled_backends"])
    if "registration_open" in patch and patch["registration_open"] is not None:
        v = patch["registration_open"]
        database.kv_set("registration_open", "true" if v else "false")
    str_keys = [
        "openai_api_key",
        "openai_base_url",
        "openai_model",
        "ollama_base_url",
        "ollama_model",
        "deepl_api_key",
        "deepl_api_url",
        "deepseek_api_key",
        "deepseek_base_url",
        "deepseek_model",
        "default_backend",
        "http_timeout_s",
    ]
    # 空字符串不写入：避免管理端只改「启用后端」等选项时，把未在表单中显示的密钥整表覆盖成空
    for k in str_keys:
        if k not in patch:
            continue
        val = patch[k]
        if val is None:
            continue
        s = str(val).strip()
        if s == "":
            continue
        database.kv_set(k, s)
