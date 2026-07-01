from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pdf_translate.server.secrets_store import (
    SecretStoreError,
    reveal_secret_value,
    secret_encryption_config,
    secret_value_is_encrypted,
)

SECURITY_PREFLIGHT_SCHEMA_VERSION = "security-preflight-v1"
DEFAULT_BOOTSTRAP_ADMIN_PASSWORD = "mic820323"
DEFAULT_MAX_UPLOAD_MB = 120
MAX_UPLOAD_MB_MIN = 1
MAX_UPLOAD_MB_MAX = 1024
DEFAULT_JWT_TTL_MINUTES = 12 * 60
JWT_TTL_MINUTES_MIN = 1
JWT_TTL_MINUTES_MAX = 7 * 24 * 60

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


@dataclass(frozen=True)
class JwtTtlConfig:
    minutes: int
    seconds: int
    raw_value: str | None
    uses_default: bool
    invalid_reason: str | None = None


class ProductionSecurityError(RuntimeError):
    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        issues = [
            str(issue.get("code") or "UNKNOWN")
            for issue in report.get("issues", [])
            if str(issue.get("severity") or "low") in {"high", "medium"}
        ]
        super().__init__(
            "Production security gate failed: "
            + (", ".join(issues) if issues else "unknown blocking issue")
        )


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


def jwt_ttl_config(
    env: Mapping[str, str] | None = None,
    *,
    raw_value: str | None = None,
) -> JwtTtlConfig:
    raw = _env_get(env, "PDF_TRANSLATE_JWT_TTL_MINUTES") if raw_value is None else raw_value
    if raw is None or str(raw).strip() == "":
        minutes = DEFAULT_JWT_TTL_MINUTES
        return JwtTtlConfig(
            minutes=minutes,
            seconds=minutes * 60,
            raw_value=None,
            uses_default=True,
        )
    try:
        minutes = int(str(raw).strip())
    except ValueError:
        minutes = DEFAULT_JWT_TTL_MINUTES
        return JwtTtlConfig(
            minutes=minutes,
            seconds=minutes * 60,
            raw_value=str(raw),
            uses_default=True,
            invalid_reason="not_an_integer",
        )
    if minutes < JWT_TTL_MINUTES_MIN or minutes > JWT_TTL_MINUTES_MAX:
        fallback = DEFAULT_JWT_TTL_MINUTES
        return JwtTtlConfig(
            minutes=fallback,
            seconds=fallback * 60,
            raw_value=str(raw),
            uses_default=True,
            invalid_reason=f"outside_{JWT_TTL_MINUTES_MIN}_{JWT_TTL_MINUTES_MAX}_minutes",
        )
    return JwtTtlConfig(
        minutes=minutes,
        seconds=minutes * 60,
        raw_value=str(raw),
        uses_default=False,
    )


def max_upload_bytes(env: Mapping[str, str] | None = None) -> int:
    return upload_limit_config(env).max_bytes


def max_upload_mb(env: Mapping[str, str] | None = None) -> int:
    return upload_limit_config(env).max_mb


def jwt_ttl_seconds(env: Mapping[str, str] | None = None) -> int:
    return jwt_ttl_config(env).seconds


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


def _stored_secret_state(db_path: Path, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    empty = {
        "stored_key_names": [],
        "encrypted_key_names": [],
        "plaintext_key_names": [],
        "decryptable_encrypted_key_names": [],
        "undecryptable_encrypted_key_names": [],
    }
    if not db_path.is_file():
        return empty
    try:
        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='app_kv'"
            ).fetchone()
            if not row:
                return empty
            placeholders = ",".join("?" for _ in SECRET_SETTING_KEYS)
            rows = conn.execute(
                f"SELECT key, value FROM app_kv WHERE key IN ({placeholders})",
                tuple(SECRET_SETTING_KEYS),
            ).fetchall()
    except sqlite3.Error:
        return empty
    encrypted: list[str] = []
    plaintext: list[str] = []
    decryptable: list[str] = []
    undecryptable: list[str] = []
    for key, value in rows:
        text = str(value or "").strip()
        if not text:
            continue
        name = str(key)
        if secret_value_is_encrypted(text):
            encrypted.append(name)
            try:
                reveal_secret_value(text, env)
            except SecretStoreError:
                undecryptable.append(name)
            else:
                decryptable.append(name)
        else:
            plaintext.append(name)
    encrypted = sorted(encrypted)
    plaintext = sorted(plaintext)
    decryptable = sorted(decryptable)
    undecryptable = sorted(undecryptable)
    return {
        "stored_key_names": sorted(encrypted + plaintext),
        "encrypted_key_names": encrypted,
        "plaintext_key_names": plaintext,
        "decryptable_encrypted_key_names": decryptable,
        "undecryptable_encrypted_key_names": undecryptable,
    }


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

    jwt_ttl = jwt_ttl_config(env)
    if jwt_ttl.invalid_reason:
        _add_issue(
            issues,
            code="JWT_TTL_INVALID",
            severity="medium",
            message="PDF_TRANSLATE_JWT_TTL_MINUTES is invalid; the default session lifetime is being used.",
            next_step=f"Set PDF_TRANSLATE_JWT_TTL_MINUTES to an integer from {JWT_TTL_MINUTES_MIN} to {JWT_TTL_MINUTES_MAX}.",
            env_var="PDF_TRANSLATE_JWT_TTL_MINUTES",
        )
    elif jwt_ttl.uses_default:
        _add_issue(
            issues,
            code="JWT_TTL_DEFAULT",
            severity="low",
            message="JWT session lifetime uses the default value.",
            next_step="Set PDF_TRANSLATE_JWT_TTL_MINUTES explicitly for the target deployment.",
            env_var="PDF_TRANSLATE_JWT_TTL_MINUTES",
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

    secret_state = _stored_secret_state(app_db, env)
    stored_keys = secret_state["stored_key_names"]
    encrypted_keys = secret_state["encrypted_key_names"]
    plaintext_keys = secret_state["plaintext_key_names"]
    decryptable_encrypted_keys = secret_state["decryptable_encrypted_key_names"]
    undecryptable_encrypted_keys = secret_state["undecryptable_encrypted_key_names"]
    encryption = secret_encryption_config(env)
    if encryption.invalid_reason:
        _add_issue(
            issues,
            code="SECRET_KEY_INVALID",
            severity="high" if production else "medium",
            message="Secret encryption key source is configured but cannot be used.",
            next_step="Fix PDF_TRANSLATE_SECRET_KEY_FILE or provide PDF_TRANSLATE_SECRET_KEY.",
            env_var="PDF_TRANSLATE_SECRET_KEY_FILE",
        )
    if encrypted_keys and not encryption.enabled and not encryption.invalid_reason:
        _add_issue(
            issues,
            code="SECRET_KEY_MISSING_FOR_ENCRYPTED_VALUES",
            severity="high" if production else "medium",
            message="Encrypted API keys exist in local SQLite but no secret encryption key is configured.",
            next_step="Set PDF_TRANSLATE_SECRET_KEY or PDF_TRANSLATE_SECRET_KEY_FILE before starting the service.",
            env_var="PDF_TRANSLATE_SECRET_KEY",
        )
    if encrypted_keys and encryption.enabled and undecryptable_encrypted_keys:
        _add_issue(
            issues,
            code="SECRET_KEY_DECRYPT_CHECK_FAILED",
            severity="high" if production else "medium",
            message="One or more encrypted API keys in local SQLite cannot be decrypted with the configured key.",
            next_step="Restore the original PDF_TRANSLATE_SECRET_KEY or re-save these API keys in the admin settings.",
            env_var="PDF_TRANSLATE_SECRET_KEY",
        )
    if plaintext_keys:
        _add_issue(
            issues,
            code="API_KEYS_STORED_PLAINTEXT_IN_LOCAL_DB",
            severity="high" if production else "medium",
            message="One or more API keys are stored as plaintext in local SQLite settings.",
            next_step="Set PDF_TRANSLATE_SECRET_KEY and re-save these keys in the admin settings, or prefer environment variables/secret manager.",
        )
    elif stored_keys:
        _add_issue(
            issues,
            code="API_KEYS_STORED_ENCRYPTED_IN_LOCAL_DB",
            severity="low",
            message="One or more API keys are stored encrypted in local SQLite settings.",
            next_step="Keep PDF_TRANSLATE_SECRET_KEY stable and restrict app.db file permissions.",
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
        "jwt": {
            "ttl_minutes": jwt_ttl.minutes,
            "ttl_seconds": jwt_ttl.seconds,
            "ttl_raw_value": jwt_ttl.raw_value,
            "ttl_uses_default": jwt_ttl.uses_default,
            "ttl_invalid_reason": jwt_ttl.invalid_reason,
        },
        "api_keys": {
            "stored_key_count": len(stored_keys),
            "stored_key_names": stored_keys,
            "encrypted_key_count": len(encrypted_keys),
            "encrypted_key_names": encrypted_keys,
            "plaintext_key_count": len(plaintext_keys),
            "plaintext_key_names": plaintext_keys,
            "decryptable_encrypted_key_names": decryptable_encrypted_keys,
            "undecryptable_encrypted_key_names": undecryptable_encrypted_keys,
            "needs_reencrypt_key_count": len(plaintext_keys),
            "local_encryption_enabled": encryption.enabled,
            "local_encryption_source": encryption.source,
            "local_encryption_invalid_reason": encryption.invalid_reason,
        },
        "issues": issues,
    }


def assert_production_security_ready(
    data_base: Path,
    data_root: Path,
    *,
    env: Mapping[str, str] | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    report = build_security_preflight(data_base, data_root, env=env, db_path=db_path)
    if not report["production_mode"]:
        return report
    blocking = [
        issue
        for issue in report["issues"]
        if str(issue.get("severity") or "low") in {"high", "medium"}
    ]
    if blocking:
        raise ProductionSecurityError(report)
    return report
