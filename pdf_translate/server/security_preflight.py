from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SECURITY_PREFLIGHT_SCHEMA_VERSION = "security-preflight-v1"
DEFAULT_BOOTSTRAP_ADMIN_PASSWORD = "mic820323"
DEFAULT_MAX_UPLOAD_MB = 120
MAX_UPLOAD_MB_MIN = 1
MAX_UPLOAD_MB_MAX = 1024

SECRET_SETTING_KEYS = (
    "deepseek_api_key",
    "openai_api_key",
    "deepl_api_key",
    "siliconflow_api_key",
    "planner_api_key",
)


@dataclass(frozen=True)
class UploadLimitConfig:
    max_mb: int
    max_bytes: int
    raw_value: str | None
    uses_default: bool
    invalid_reason: str | None = None


def _env_get(env: Mapping[str, str] | None, name: str) -> str | None:
    source = os.environ if env is None else env
    return source.get(name)


def deployment_mode(env: Mapping[str, str] | None = None) -> str:
    raw = (
        _env_get(env, "PDF_TRANSLATE_ENV")
        or _env_get(env, "PDF_TRANSLATE_DEPLOYMENT_MODE")
        or "development"
    )
    return raw.strip().lower() or "development"


def is_production_mode(env: Mapping[str, str] | None = None) -> bool:
    return deployment_mode(env) in {"prod", "production", "public", "release"}


def cors_origins_from_env(env: Mapping[str, str] | None = None) -> list[str]:
    raw = _env_get(env, "PDF_TRANSLATE_CORS_ORIGINS")
    if raw is None:
        raw = "*"
    return [item.strip() for item in raw.split(",") if item.strip()]


def upload_limit_config(
    env: Mapping[str, str] | None = None,
    *,
    raw_value: str | None = None,
) -> UploadLimitConfig:
    raw = _env_get(env, "PDF_TRANSLATE_MAX_UPLOAD_MB") if raw_value is None else raw_value
    if raw is None or str(raw).strip() == "":
        mb = DEFAULT_MAX_UPLOAD_MB
        return UploadLimitConfig(
            max_mb=mb,
            max_bytes=mb * 1024 * 1024,
            raw_value=None,
            uses_default=True,
        )
    try:
        mb = int(str(raw).strip())
    except ValueError:
        mb = DEFAULT_MAX_UPLOAD_MB
        return UploadLimitConfig(
            max_mb=mb,
            max_bytes=mb * 1024 * 1024,
            raw_value=str(raw),
            uses_default=True,
            invalid_reason="not_an_integer",
        )
    if mb < MAX_UPLOAD_MB_MIN or mb > MAX_UPLOAD_MB_MAX:
        fallback = DEFAULT_MAX_UPLOAD_MB
        return UploadLimitConfig(
            max_mb=fallback,
            max_bytes=fallback * 1024 * 1024,
            raw_value=str(raw),
            uses_default=True,
            invalid_reason=f"outside_{MAX_UPLOAD_MB_MIN}_{MAX_UPLOAD_MB_MAX}_mb",
        )
    return UploadLimitConfig(
        max_mb=mb,
        max_bytes=mb * 1024 * 1024,
        raw_value=str(raw),
        uses_default=False,
    )


def max_upload_bytes(env: Mapping[str, str] | None = None) -> int:
    return upload_limit_config(env).max_bytes


def max_upload_mb(env: Mapping[str, str] | None = None) -> int:
    return upload_limit_config(env).max_mb


def _add_issue(
    issues: list[dict[str, Any]],
    *,
    code: str,
    severity: str,
    message: str,
    next_step: str,
    env_var: str | None = None,
) -> None:
    issue: dict[str, Any] = {
        "code": code,
        "severity": severity,
        "message": message,
        "next_step": next_step,
    }
    if env_var:
        issue["env_var"] = env_var
    issues.append(issue)


def _path_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _stored_secret_keys(db_path: Path) -> list[str]:
    if not db_path.is_file():
        return []
    try:
        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='app_kv'"
            ).fetchone()
            if not row:
                return []
            placeholders = ",".join("?" for _ in SECRET_SETTING_KEYS)
            rows = conn.execute(
                f"SELECT key, value FROM app_kv WHERE key IN ({placeholders})",
                tuple(SECRET_SETTING_KEYS),
            ).fetchall()
    except sqlite3.Error:
        return []
    return sorted(str(key) for key, value in rows if str(value or "").strip())


def build_security_preflight(
    data_base: Path,
    data_root: Path,
    *,
    env: Mapping[str, str] | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    base = data_base.resolve()
    root = data_root.resolve()
    app_db = (db_path or (base / "app.db")).resolve()
    production = is_production_mode(env)
    mode = deployment_mode(env)
    issues: list[dict[str, Any]] = []

    bootstrap_pw = _env_get(env, "PDF_TRANSLATE_BOOTSTRAP_ADMIN_PASSWORD")
    if bootstrap_pw is None or bootstrap_pw == DEFAULT_BOOTSTRAP_ADMIN_PASSWORD:
        _add_issue(
            issues,
            code="DEFAULT_BOOTSTRAP_ADMIN_PASSWORD",
            severity="high" if production else "medium",
            message="The bootstrap admin password is unset or still uses the project default.",
            next_step="Set PDF_TRANSLATE_BOOTSTRAP_ADMIN_PASSWORD before creating the first admin user.",
            env_var="PDF_TRANSLATE_BOOTSTRAP_ADMIN_PASSWORD",
        )

    origins = cors_origins_from_env(env)
    cors_allow_all = "*" in origins
    if cors_allow_all:
        _add_issue(
            issues,
            code="CORS_ALLOW_ALL",
            severity="high" if production else "medium",
            message="CORS allows every origin.",
            next_step="Set PDF_TRANSLATE_CORS_ORIGINS to the exact public frontend origin list.",
            env_var="PDF_TRANSLATE_CORS_ORIGINS",
        )

    if not (_env_get(env, "PDF_TRANSLATE_JWT_SECRET") or "").strip():
        _add_issue(
            issues,
            code="JWT_SECRET_FILE_FALLBACK",
            severity="high" if production else "medium",
            message="JWT signing falls back to a local jwt_secret.txt file.",
            next_step="Set a stable, private PDF_TRANSLATE_JWT_SECRET in production.",
            env_var="PDF_TRANSLATE_JWT_SECRET",
        )

    upload = upload_limit_config(env)
    if upload.invalid_reason:
        _add_issue(
            issues,
            code="UPLOAD_LIMIT_INVALID",
            severity="medium",
            message="PDF_TRANSLATE_MAX_UPLOAD_MB is invalid; the default upload limit is being used.",
            next_step=f"Set PDF_TRANSLATE_MAX_UPLOAD_MB to an integer from {MAX_UPLOAD_MB_MIN} to {MAX_UPLOAD_MB_MAX}.",
            env_var="PDF_TRANSLATE_MAX_UPLOAD_MB",
        )
    elif upload.uses_default:
        _add_issue(
            issues,
            code="UPLOAD_LIMIT_DEFAULT",
            severity="low",
            message="Upload limit uses the default value.",
            next_step="Set PDF_TRANSLATE_MAX_UPLOAD_MB explicitly for the target deployment size.",
            env_var="PDF_TRANSLATE_MAX_UPLOAD_MB",
        )

    if not (_env_get(env, "PDF_TRANSLATE_DATA") or "").strip():
        _add_issue(
            issues,
            code="DATA_DIR_DEFAULT",
            severity="medium" if production else "low",
            message="Data directory uses the process working directory default.",
            next_step="Set PDF_TRANSLATE_DATA to a dedicated persistent data directory.",
            env_var="PDF_TRANSLATE_DATA",
        )

    data_root_within_base = _path_within(root, base)
    if not data_root_within_base:
        _add_issue(
            issues,
            code="WEB_DATA_OUTSIDE_DATA_BASE",
            severity="low",
            message="Web job data root is outside PDF_TRANSLATE_DATA.",
            next_step="Confirm the external PDF_TRANSLATE_WEB_DATA path is dedicated to this service.",
            env_var="PDF_TRANSLATE_WEB_DATA",
        )

    stored_keys = _stored_secret_keys(app_db)
    if stored_keys:
        _add_issue(
            issues,
            code="API_KEYS_STORED_IN_LOCAL_DB",
            severity="medium" if production else "low",
            message="One or more API keys are stored in local SQLite settings.",
            next_step="Limit app.db file permissions and prefer environment variables or a secret manager for production.",
        )

    severity_counts = {level: 0 for level in ("high", "medium", "low")}
    for issue in issues:
        level = str(issue.get("severity") or "low")
        severity_counts[level] = severity_counts.get(level, 0) + 1
    blocking_issue_count = severity_counts.get("high", 0) + severity_counts.get("medium", 0)

    return {
        "schema_version": SECURITY_PREFLIGHT_SCHEMA_VERSION,
        "mode": mode,
        "production_mode": production,
        "ok": blocking_issue_count == 0,
        "issue_count": len(issues),
        "blocking_issue_count": blocking_issue_count,
        "severity_counts": severity_counts,
        "data_base": str(base),
        "data_root": str(root),
        "data_root_within_data_base": data_root_within_base,
        "cors": {
            "origins": origins,
            "allow_all": cors_allow_all,
        },
        "upload": {
            "max_mb": upload.max_mb,
            "max_bytes": upload.max_bytes,
            "raw_value": upload.raw_value,
            "uses_default": upload.uses_default,
            "invalid_reason": upload.invalid_reason,
        },
        "api_keys": {
            "stored_key_count": len(stored_keys),
            "stored_key_names": stored_keys,
        },
        "issues": issues,
    }
