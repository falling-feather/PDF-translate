from __future__ import annotations

import os
from dataclasses import dataclass

import jwt
from fastapi import Depends, Header, HTTPException

from pdf_translate.server import database
from pdf_translate.server.runtime_state import require_data_dir


@dataclass
class Principal:
    user_id: int
    username: str
    role: str


def _jwt_secret() -> str:
    dd = require_data_dir()
    env = os.getenv("PDF_TRANSLATE_JWT_SECRET")
    if env and env.strip():
        return env.strip()
    return database.get_jwt_secret_file(dd)


def mint_token(*, user_id: int, username: str, role: str) -> str:
    payload = {"sub": str(user_id), "username": username, "role": role}
    return jwt.encode(payload, _jwt_secret(), algorithm="HS256")


def decode_token(token: str) -> Principal:
    try:
        payload = jwt.decode(token, _jwt_secret(), algorithms=["HS256"])
        return Principal(
            user_id=int(payload["sub"]),
            username=str(payload["username"]),
            role=str(payload["role"]),
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录") from e


def bearer_principal(authorization: str | None = Header(None)) -> Principal:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="需要登录")
    token = authorization.split(" ", 1)[1].strip()
    return decode_token(token)


def require_admin(p: Principal = Depends(bearer_principal)) -> Principal:
    if p.role != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return p