from __future__ import annotations

import os
import unittest
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import HTTPException

from pdf_translate.server.auth_deps import decode_token, mint_token


class AuthDepsTests(unittest.TestCase):
    def _set_env(self, **values: str) -> None:
        old = {key: os.environ.get(key) for key in values}
        for key, value in values.items():
            os.environ[key] = value

        def restore() -> None:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.addCleanup(restore)

    def test_mint_token_includes_expiration_claim(self) -> None:
        secret = "test-secret-" + "x" * 32
        self._set_env(
            PDF_TRANSLATE_JWT_SECRET=secret,
            PDF_TRANSLATE_JWT_TTL_MINUTES="5",
        )

        token = mint_token(user_id=7, username="alice", role="admin")
        payload = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            options={"require": ["exp", "iat"]},
        )

        self.assertEqual(payload["sub"], "7")
        self.assertEqual(payload["username"], "alice")
        self.assertEqual(payload["role"], "admin")
        self.assertGreaterEqual(payload["exp"] - payload["iat"], 299)
        self.assertLessEqual(payload["exp"] - payload["iat"], 301)

    def test_decode_token_accepts_valid_token(self) -> None:
        secret = "test-secret-" + "y" * 32
        self._set_env(PDF_TRANSLATE_JWT_SECRET=secret)
        now = datetime.now(timezone.utc)
        token = jwt.encode(
            {
                "sub": "42",
                "username": "bob",
                "role": "user",
                "exp": now + timedelta(minutes=10),
            },
            secret,
            algorithm="HS256",
        )

        principal = decode_token(token)

        self.assertEqual(principal.user_id, 42)
        self.assertEqual(principal.username, "bob")
        self.assertEqual(principal.role, "user")

    def test_decode_token_rejects_expired_token(self) -> None:
        secret = "test-secret-" + "z" * 32
        self._set_env(PDF_TRANSLATE_JWT_SECRET=secret)
        now = datetime.now(timezone.utc)
        token = jwt.encode(
            {
                "sub": "42",
                "username": "bob",
                "role": "user",
                "exp": now - timedelta(seconds=1),
            },
            secret,
            algorithm="HS256",
        )

        with self.assertRaises(HTTPException) as raised:
            decode_token(token)

        self.assertEqual(raised.exception.status_code, 401)

    def test_decode_token_rejects_token_without_expiration(self) -> None:
        secret = "test-secret-" + "m" * 32
        self._set_env(PDF_TRANSLATE_JWT_SECRET=secret)
        token = jwt.encode(
            {
                "sub": "42",
                "username": "bob",
                "role": "user",
            },
            secret,
            algorithm="HS256",
        )

        with self.assertRaises(HTTPException) as raised:
            decode_token(token)

        self.assertEqual(raised.exception.status_code, 401)


if __name__ == "__main__":
    unittest.main()
