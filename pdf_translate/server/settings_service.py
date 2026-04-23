from __future__ import annotations

from pdf_translate.config import AppConfig
from pdf_translate.server import database

DEFAULT_ENABLED = ["echo", "deepseek"]


def _coalesce(kv_val: str | None, base_val: str | None) -> str | None:
    if kv_val is not None and str(kv_val).strip() != "":
        return str(kv_val).strip()
    return base_val


def _coalesce_str(kv_val: str | None, base_val: str) -> str:
    if kv_val is not None and str(kv_val).strip() != "":
        return str(kv_val).strip()
    return base_val


def _coalesce_bool(kv_val: str | None, base: bool) -> bool:
    if kv_val is None or str(kv_val).strip() == "":
        return base
    return str(kv_val).strip().lower() in ("1", "true", "yes", "on")


def _parse_survey_max_chars(base: AppConfig) -> int:
    raw = _coalesce_str(database.kv_get("survey_max_text_chars"), str(base.survey_max_text_chars))
    try:
        n = int(raw.strip())
        return max(1000, min(n, 200_000))
    except ValueError:
        return base.survey_max_text_chars


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
        survey_enabled=_coalesce_bool(database.kv_get("survey_enabled"), base.survey_enabled),
        siliconflow_api_key=_coalesce(database.kv_get("siliconflow_api_key"), base.siliconflow_api_key),
        siliconflow_base_url=_coalesce_str(database.kv_get("siliconflow_base_url"), base.siliconflow_base_url),
        siliconflow_survey_model=_coalesce_str(
            database.kv_get("siliconflow_survey_model"), base.siliconflow_survey_model
        ),
        siliconflow_vision_model=_coalesce_str(
            database.kv_get("siliconflow_vision_model"), base.siliconflow_vision_model
        ),
        survey_max_text_chars=_parse_survey_max_chars(base),
        planner_enabled=_coalesce_bool(database.kv_get("planner_enabled"), base.planner_enabled),
        planner_api_key=_coalesce(database.kv_get("planner_api_key"), base.planner_api_key),
        planner_base_url=_coalesce_str(database.kv_get("planner_base_url"), base.planner_base_url),
        planner_model=_coalesce_str(database.kv_get("planner_model"), base.planner_model),
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
    b = (backend or default_backend or "deepseek").lower().strip()
    if b not in allowed:
        raise ValueError(f"后端「{b}」未在管理端启用")
    return b


def admin_settings_snapshot() -> dict:
    keys = [
        "deepseek_api_key",
        "deepseek_base_url",
        "deepseek_model",
        "default_backend",
        "http_timeout_s",
        "survey_enabled",
        "siliconflow_api_key",
        "siliconflow_base_url",
        "siliconflow_survey_model",
        "siliconflow_vision_model",
        "survey_max_text_chars",
        "planner_enabled",
        "planner_api_key",
        "planner_base_url",
        "planner_model",
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
    if "survey_enabled" in patch and patch["survey_enabled"] is not None:
        v = patch["survey_enabled"]
        if isinstance(v, bool):
            database.kv_set("survey_enabled", "true" if v else "false")
        else:
            s = str(v).strip().lower()
            database.kv_set("survey_enabled", "true" if s in ("1", "true", "yes", "on") else "false")
    if "planner_enabled" in patch and patch["planner_enabled"] is not None:
        v = patch["planner_enabled"]
        if isinstance(v, bool):
            database.kv_set("planner_enabled", "true" if v else "false")
        else:
            s = str(v).strip().lower()
            database.kv_set("planner_enabled", "true" if s in ("1", "true", "yes", "on") else "false")
    str_keys = [
        "deepseek_api_key",
        "deepseek_base_url",
        "deepseek_model",
        "default_backend",
        "http_timeout_s",
        "siliconflow_api_key",
        "siliconflow_base_url",
        "siliconflow_survey_model",
        "siliconflow_vision_model",
        "survey_max_text_chars",
        "planner_api_key",
        "planner_base_url",
        "planner_model",
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
