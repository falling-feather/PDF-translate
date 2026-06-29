from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pdf_translate.config import AppConfig

COST_PROFILE_SCHEMA_VERSION = "cost-profile-v1"
COST_ESTIMATE_SCHEMA_VERSION = "cost-estimate-v1"

ZERO_EXTERNAL_COST_BACKENDS = {
    "echo": {
        "currency": "USD",
        "input_per_1m_tokens": 0,
        "output_per_1m_tokens": 0,
        "per_request": 0,
        "note": "echo 后端不调用外部 API。",
    },
    "ollama": {
        "currency": "USD",
        "input_per_1m_tokens": 0,
        "output_per_1m_tokens": 0,
        "per_request": 0,
        "note": "Ollama 默认为本地推理；该估算不包含本机电力、显卡或折旧成本。",
    },
}


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return 0.0
    return 0.0


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return 0
    return 0


def _round_money(value: float) -> float:
    return round(float(value), 8)


def _normalize_entry(raw: Any, default_currency: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"currency": default_currency}
    return {
        "currency": str(raw.get("currency") or default_currency).upper(),
        "input_per_1m_tokens": _as_float(
            raw.get("input_per_1m_tokens", raw.get("input_per_million_tokens"))
        ),
        "output_per_1m_tokens": _as_float(
            raw.get("output_per_1m_tokens", raw.get("output_per_million_tokens"))
        ),
        "input_per_1m_chars": _as_float(
            raw.get("input_per_1m_chars", raw.get("input_per_million_chars"))
        ),
        "output_per_1m_chars": _as_float(
            raw.get("output_per_1m_chars", raw.get("output_per_million_chars"))
        ),
        "per_request": _as_float(raw.get("per_request")),
        "note": str(raw.get("note") or ""),
    }


def normalize_cost_profile(
    raw: Any,
    *,
    source: str,
    default_currency: str = "USD",
) -> dict[str, Any]:
    warnings: list[str] = []
    currency = default_currency.upper()
    raw_backends: dict[str, Any] = {}
    if isinstance(raw, dict):
        currency = str(raw.get("currency") or default_currency).upper()
        backends = raw.get("backends")
        if isinstance(backends, dict):
            raw_backends = backends
        else:
            raw_backends = {
                str(key): value
                for key, value in raw.items()
                if key not in {"schema_version", "currency", "source", "warnings"}
            }
    elif raw is not None:
        warnings.append("成本画像不是 JSON 对象，已忽略用户配置。")

    merged = dict(ZERO_EXTERNAL_COST_BACKENDS)
    merged.update(raw_backends)
    normalized = {
        str(key).strip().lower(): _normalize_entry(value, currency)
        for key, value in merged.items()
        if str(key).strip()
    }
    return {
        "schema_version": COST_PROFILE_SCHEMA_VERSION,
        "source": source,
        "currency": currency,
        "backends": normalized,
        "warnings": warnings,
    }


def empty_cost_profile(default_currency: str = "USD") -> dict[str, Any]:
    return normalize_cost_profile({}, source="built-in", default_currency=default_currency)


def load_cost_profile(cfg: AppConfig) -> dict[str, Any]:
    default_currency = (cfg.cost_default_currency or "USD").upper()
    warnings: list[str] = []
    if cfg.cost_profile_json.strip():
        try:
            raw = json.loads(cfg.cost_profile_json)
            profile = normalize_cost_profile(
                raw,
                source="env:PDF_TRANSLATE_COST_PROFILE_JSON",
                default_currency=default_currency,
            )
        except json.JSONDecodeError as exc:
            profile = empty_cost_profile(default_currency)
            warnings.append(f"PDF_TRANSLATE_COST_PROFILE_JSON 解析失败：{exc.msg}")
        profile["warnings"] = list(profile.get("warnings") or []) + warnings
        return profile

    if cfg.cost_profile_path.strip():
        path = Path(cfg.cost_profile_path).expanduser()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            profile = normalize_cost_profile(
                raw,
                source=f"file:{path}",
                default_currency=default_currency,
            )
        except (OSError, json.JSONDecodeError) as exc:
            profile = empty_cost_profile(default_currency)
            warnings.append(f"成本画像文件读取失败：{exc}")
        profile["warnings"] = list(profile.get("warnings") or []) + warnings
        return profile

    return empty_cost_profile(default_currency)


def backend_model_name(backend: str, cfg: AppConfig) -> str:
    b = backend.lower().strip()
    if b == "deepseek":
        return cfg.deepseek_model
    if b == "openai":
        return cfg.openai_model
    if b == "ollama":
        return cfg.ollama_model
    if b == "deepl":
        return "deepl"
    if b == "hybrid":
        return f"deepl+{cfg.openai_model}"
    return b


def _candidate_profile_keys(backend: str, model: str | None) -> list[str]:
    b = backend.lower().strip()
    m = (model or "").lower().strip()
    keys = []
    if b and m:
        keys.extend([f"{b}:{m}", f"{b}/{m}"])
    if m:
        keys.append(m)
    if b:
        keys.append(b)
    keys.append("default")
    out: list[str] = []
    for key in keys:
        if key and key not in out:
            out.append(key)
    return out


def estimate_cost(
    run_metrics: dict[str, Any] | None,
    cost_profile: dict[str, Any] | None,
    *,
    backend: str,
    model: str | None = None,
) -> dict[str, Any]:
    profile = cost_profile if isinstance(cost_profile, dict) else empty_cost_profile()
    backends = profile.get("backends") if isinstance(profile.get("backends"), dict) else {}
    selected_key = ""
    entry: dict[str, Any] | None = None
    for key in _candidate_profile_keys(backend, model):
        raw = backends.get(key)
        if isinstance(raw, dict):
            selected_key = key
            entry = raw
            break

    summary = run_metrics.get("summary") if isinstance(run_metrics, dict) else {}
    if not isinstance(summary, dict):
        summary = {}
    request_tokens = _as_int(summary.get("estimated_request_token_count"))
    response_tokens = _as_int(summary.get("estimated_translated_token_count"))
    request_chars = _as_int(summary.get("request_char_count"))
    response_chars = _as_int(summary.get("translated_char_count"))
    request_count = _as_int(summary.get("translation_request_count"))
    http_attempt_count = _as_int(summary.get("http_attempt_count"))
    http_retry_count = _as_int(summary.get("http_retry_count"))
    http_failed_attempt_count = _as_int(summary.get("http_failed_attempt_count"))
    http_retryable_error_count = _as_int(summary.get("http_retryable_error_count"))
    billable_request_count = http_attempt_count if http_attempt_count > 0 else request_count
    billable_request_count_source = (
        "http_attempt_count" if http_attempt_count > 0 else "translation_request_count_fallback"
    )
    currency = str((entry or {}).get("currency") or profile.get("currency") or "USD").upper()
    warnings = list(profile.get("warnings") or [])
    if http_retry_count > 0:
        warnings.append(
            "已捕获 HTTP 重试；per_request 按 HTTP 尝试次数估算，失败尝试的 token/字符计费仍按成功译文用量近似。"
        )

    if entry is None:
        warnings.append("未命中后端成本画像，成本仅保留 token/字符用量，不估算金额。")
        return {
            "schema_version": COST_ESTIMATE_SCHEMA_VERSION,
            "configured": False,
            "backend": backend,
            "model": model or "",
            "profile_source": profile.get("source") or "unknown",
            "profile_key": "",
            "currency": currency,
            "usage": {
                "estimated_request_token_count": request_tokens,
                "estimated_response_token_count": response_tokens,
                "request_char_count": request_chars,
                "response_char_count": response_chars,
                "translation_request_count": request_count,
                "http_attempt_count": http_attempt_count,
                "http_retry_count": http_retry_count,
                "http_failed_attempt_count": http_failed_attempt_count,
                "http_retryable_error_count": http_retryable_error_count,
                "billable_request_count": billable_request_count,
                "billable_request_count_source": billable_request_count_source,
            },
            "unit_prices": {},
            "summary": {
                "input_token_cost": 0,
                "output_token_cost": 0,
                "input_char_cost": 0,
                "output_char_cost": 0,
                "request_cost": 0,
                "estimated_total_cost": 0,
            },
            "warnings": warnings,
        }

    input_token_cost = request_tokens / 1_000_000 * _as_float(entry.get("input_per_1m_tokens"))
    output_token_cost = response_tokens / 1_000_000 * _as_float(entry.get("output_per_1m_tokens"))
    input_char_cost = request_chars / 1_000_000 * _as_float(entry.get("input_per_1m_chars"))
    output_char_cost = response_chars / 1_000_000 * _as_float(entry.get("output_per_1m_chars"))
    per_request = _as_float(entry.get("per_request"))
    if (
        billable_request_count_source == "translation_request_count_fallback"
        and request_count > 0
        and per_request > 0
    ):
        warnings.append("未捕获 HTTP 尝试次数，per_request 已退回按成功翻译请求次数估算。")
    request_cost = billable_request_count * per_request
    total = input_token_cost + output_token_cost + input_char_cost + output_char_cost + request_cost
    return {
        "schema_version": COST_ESTIMATE_SCHEMA_VERSION,
        "configured": True,
        "backend": backend,
        "model": model or "",
        "profile_source": profile.get("source") or "unknown",
        "profile_key": selected_key,
        "currency": currency,
        "usage": {
            "estimated_request_token_count": request_tokens,
            "estimated_response_token_count": response_tokens,
            "request_char_count": request_chars,
            "response_char_count": response_chars,
            "translation_request_count": request_count,
            "http_attempt_count": http_attempt_count,
            "http_retry_count": http_retry_count,
            "http_failed_attempt_count": http_failed_attempt_count,
            "http_retryable_error_count": http_retryable_error_count,
            "billable_request_count": billable_request_count,
            "billable_request_count_source": billable_request_count_source,
        },
        "unit_prices": {
            "input_per_1m_tokens": _as_float(entry.get("input_per_1m_tokens")),
            "output_per_1m_tokens": _as_float(entry.get("output_per_1m_tokens")),
            "input_per_1m_chars": _as_float(entry.get("input_per_1m_chars")),
            "output_per_1m_chars": _as_float(entry.get("output_per_1m_chars")),
            "per_request": per_request,
        },
        "summary": {
            "input_token_cost": _round_money(input_token_cost),
            "output_token_cost": _round_money(output_token_cost),
            "input_char_cost": _round_money(input_char_cost),
            "output_char_cost": _round_money(output_char_cost),
            "request_cost": _round_money(request_cost),
            "estimated_total_cost": _round_money(total),
        },
        "warnings": warnings,
    }


def write_cost_estimate(
    path: Path,
    run_metrics: dict[str, Any],
    cost_profile: dict[str, Any],
    *,
    backend: str,
    model: str | None = None,
) -> dict[str, Any]:
    estimate = estimate_cost(run_metrics, cost_profile, backend=backend, model=model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(estimate, ensure_ascii=False, indent=2), encoding="utf-8")
    return estimate
