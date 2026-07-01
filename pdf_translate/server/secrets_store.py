from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from cryptography.fernet import Fernet, InvalidToken

SECRET_VALUE_PREFIX = "enc:v1:"


class SecretStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class SecretEncryptionConfig:
    enabled: bool
    source: str | None
    invalid_reason: str | None = None


def _env_get(env: Mapping[str, str] | None, name: str) -> str | None:
    source = os.environ if env is None else env
    return source.get(name)


def _raw_secret_key(env: Mapping[str, str] | None = None) -> tuple[str | None, str | None, str | None]:
    direct = (_env_get(env, "PDF_TRANSLATE_SECRET_KEY") or "").strip()
    if direct:
        return direct, "env", None
    key_file = (_env_get(env, "PDF_TRANSLATE_SECRET_KEY_FILE") or "").strip()
    if not key_file:
        return None, None, None
    try:
        value = Path(key_file).expanduser().read_text(encoding="utf-8").strip()
    except OSError:
        return None, "file", "key_file_unreadable"
    if not value:
        return None, "file", "key_file_empty"
    return value, "file", None


def secret_encryption_config(env: Mapping[str, str] | None = None) -> SecretEncryptionConfig:
    raw, source, invalid_reason = _raw_secret_key(env)
    if invalid_reason:
        return SecretEncryptionConfig(enabled=False, source=source, invalid_reason=invalid_reason)
    return SecretEncryptionConfig(enabled=bool(raw), source=source)


def secret_value_is_encrypted(value: str | None) -> bool:
    return bool(value and value.startswith(SECRET_VALUE_PREFIX))


def _fernet(env: Mapping[str, str] | None = None) -> Fernet | None:
    raw, _, invalid_reason = _raw_secret_key(env)
    if invalid_reason:
        raise SecretStoreError(invalid_reason)
    if not raw:
        return None
    key = base64.urlsafe_b64encode(hashlib.sha256(raw.encode("utf-8")).digest())
    return Fernet(key)


def protect_secret_value(value: str, env: Mapping[str, str] | None = None) -> str:
    fernet = _fernet(env)
    if fernet is None:
        return value
    token = fernet.encrypt(value.encode("utf-8")).decode("ascii")
    return SECRET_VALUE_PREFIX + token


def reveal_secret_value(value: str | None, env: Mapping[str, str] | None = None) -> str | None:
    if value is None:
        return None
    if not secret_value_is_encrypted(value):
        return value
    fernet = _fernet(env)
    if fernet is None:
        raise SecretStoreError("secret_key_missing")
    token = value[len(SECRET_VALUE_PREFIX) :]
    try:
        return fernet.decrypt(token.encode("ascii")).decode("utf-8")
    except (InvalidToken, UnicodeDecodeError) as exc:
        raise SecretStoreError("decrypt_failed") from exc
