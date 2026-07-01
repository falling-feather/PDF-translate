from __future__ import annotations

import json
import os
import shutil
import unittest
from pathlib import Path

from pdf_translate.server import database, settings_service
from pdf_translate.server.security_preflight import build_security_preflight
from pdf_translate.server.secrets_store import SECRET_VALUE_PREFIX, protect_secret_value, reveal_secret_value


class SecretStoreTests(unittest.TestCase):
    def _case_root(self, name: str) -> Path:
        root = Path.cwd() / "test-output" / "secret-store" / name
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        database.configure(root / "app.db")
        return root

    def _env(self, **values: str | None) -> None:
        keys = {
            "PDF_TRANSLATE_SECRET_KEY",
            "PDF_TRANSLATE_SECRET_KEY_FILE",
            "DEEPSEEK_API_KEY",
            *values.keys(),
        }
        old = {key: os.environ.get(key) for key in keys}
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        def restore() -> None:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.addCleanup(restore)

    def test_secret_store_round_trip_and_legacy_plaintext(self) -> None:
        encrypted = protect_secret_value(
            "sk-placeholder",
            env={"PDF_TRANSLATE_SECRET_KEY": "local-test-secret"},
        )

        self.assertTrue(encrypted.startswith(SECRET_VALUE_PREFIX))
        self.assertNotIn("sk-placeholder", encrypted)
        self.assertEqual(
            reveal_secret_value(encrypted, env={"PDF_TRANSLATE_SECRET_KEY": "local-test-secret"}),
            "sk-placeholder",
        )
        self.assertEqual(
            reveal_secret_value("sk-legacy-placeholder", env={"PDF_TRANSLATE_SECRET_KEY": "local-test-secret"}),
            "sk-legacy-placeholder",
        )

    def test_admin_secret_write_uses_plaintext_when_no_secret_key_is_configured(self) -> None:
        root = self._case_root("plaintext")
        self._env(PDF_TRANSLATE_SECRET_KEY=None, PDF_TRANSLATE_SECRET_KEY_FILE=None)

        settings_service.apply_admin_settings({"deepseek_api_key": "sk-placeholder"})

        self.assertEqual(database.kv_get("deepseek_api_key"), "sk-placeholder")
        self.assertEqual(settings_service.effective_app_config().deepseek_api_key, "sk-placeholder")
        report = build_security_preflight(root, root / "web_jobs", env={})
        self.assertEqual(report["api_keys"]["plaintext_key_names"], ["deepseek_api_key"])
        self.assertEqual(report["api_keys"]["needs_reencrypt_key_count"], 1)
        self.assertIn("API_KEYS_STORED_PLAINTEXT_IN_LOCAL_DB", {i["code"] for i in report["issues"]})

    def test_admin_secret_write_encrypts_when_secret_key_is_configured(self) -> None:
        root = self._case_root("encrypted")
        self._env(PDF_TRANSLATE_SECRET_KEY="local-test-secret")

        settings_service.apply_admin_settings({"deepseek_api_key": "sk-placeholder"})

        raw = database.kv_get("deepseek_api_key") or ""
        rendered_snapshot = json.dumps(settings_service.admin_settings_snapshot(), ensure_ascii=False)
        report = build_security_preflight(
            root,
            root / "web_jobs",
            env={"PDF_TRANSLATE_SECRET_KEY": "local-test-secret"},
        )
        rendered_report = json.dumps(report, ensure_ascii=False)

        self.assertTrue(raw.startswith(SECRET_VALUE_PREFIX))
        self.assertNotIn("sk-placeholder", raw)
        self.assertEqual(settings_service.effective_app_config().deepseek_api_key, "sk-placeholder")
        self.assertNotIn("sk-placeholder", rendered_snapshot)
        self.assertNotIn("sk-placeholder", rendered_report)
        self.assertEqual(report["api_keys"]["encrypted_key_names"], ["deepseek_api_key"])
        self.assertEqual(report["api_keys"]["plaintext_key_names"], [])
        self.assertEqual(report["api_keys"]["decryptable_encrypted_key_names"], ["deepseek_api_key"])
        self.assertEqual(report["api_keys"]["undecryptable_encrypted_key_names"], [])

    def test_legacy_plaintext_secret_is_still_readable_with_secret_key_configured(self) -> None:
        self._case_root("legacy-plaintext")
        self._env(PDF_TRANSLATE_SECRET_KEY="local-test-secret")
        database.kv_set("deepseek_api_key", "sk-legacy-placeholder")

        self.assertEqual(settings_service.effective_app_config().deepseek_api_key, "sk-legacy-placeholder")

    def test_preflight_reports_encrypted_values_without_secret_key(self) -> None:
        root = self._case_root("encrypted-missing-key")
        self._env(PDF_TRANSLATE_SECRET_KEY="local-test-secret")
        settings_service.apply_admin_settings({"deepseek_api_key": "sk-placeholder"})
        self._env(PDF_TRANSLATE_SECRET_KEY=None, PDF_TRANSLATE_SECRET_KEY_FILE=None)

        report = build_security_preflight(root, root / "web_jobs", env={})
        codes = {issue["code"] for issue in report["issues"]}

        self.assertEqual(report["api_keys"]["encrypted_key_names"], ["deepseek_api_key"])
        self.assertEqual(report["api_keys"]["undecryptable_encrypted_key_names"], ["deepseek_api_key"])
        self.assertIn("SECRET_KEY_MISSING_FOR_ENCRYPTED_VALUES", codes)

    def test_preflight_reports_encrypted_values_with_wrong_secret_key(self) -> None:
        root = self._case_root("encrypted-wrong-key")
        self._env(PDF_TRANSLATE_SECRET_KEY="local-test-secret")
        settings_service.apply_admin_settings({"deepseek_api_key": "sk-placeholder"})

        report = build_security_preflight(
            root,
            root / "web_jobs",
            env={"PDF_TRANSLATE_SECRET_KEY": "wrong-local-test-secret"},
        )
        codes = {issue["code"] for issue in report["issues"]}
        rendered_report = json.dumps(report, ensure_ascii=False)

        self.assertEqual(report["api_keys"]["encrypted_key_names"], ["deepseek_api_key"])
        self.assertEqual(report["api_keys"]["decryptable_encrypted_key_names"], [])
        self.assertEqual(report["api_keys"]["undecryptable_encrypted_key_names"], ["deepseek_api_key"])
        self.assertIn("SECRET_KEY_DECRYPT_CHECK_FAILED", codes)
        self.assertNotIn("sk-placeholder", rendered_report)


if __name__ == "__main__":
    unittest.main()
