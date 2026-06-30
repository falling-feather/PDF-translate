from __future__ import annotations

import json
import os
import shutil
import unittest
from pathlib import Path

from typer.testing import CliRunner

from pdf_translate.cli import app
from pdf_translate.server import database, settings_service
from pdf_translate.server.security_preflight import (
    DEFAULT_JWT_TTL_MINUTES,
    DEFAULT_MAX_UPLOAD_MB,
    ProductionSecurityError,
    assert_production_security_ready,
    build_security_preflight,
    jwt_ttl_config,
    upload_limit_config,
)


class SecurityPreflightTests(unittest.TestCase):
    def _case_root(self, name: str) -> Path:
        root = Path.cwd() / "test-output" / "security-preflight" / name
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return root

    def _configure_db(self, root: Path) -> None:
        database.configure(root / "app.db")

    def test_default_environment_reports_deployment_warnings(self) -> None:
        root = self._case_root("default-env")
        report = build_security_preflight(root, root / "web_jobs", env={})
        codes = {issue["code"] for issue in report["issues"]}

        self.assertFalse(report["ok"])
        self.assertIn("DEFAULT_BOOTSTRAP_ADMIN_PASSWORD", codes)
        self.assertIn("CORS_ALLOW_ALL", codes)
        self.assertIn("JWT_SECRET_FILE_FALLBACK", codes)
        self.assertIn("UPLOAD_LIMIT_DEFAULT", codes)
        self.assertEqual(report["upload"]["max_mb"], DEFAULT_MAX_UPLOAD_MB)
        self.assertTrue(report["cors"]["allow_all"])

    def test_hardened_environment_clears_core_warnings(self) -> None:
        root = self._case_root("hardened-env")
        env = {
            "PDF_TRANSLATE_ENV": "production",
            "PDF_TRANSLATE_BOOTSTRAP_ADMIN_PASSWORD": "change-me-before-release-123!",
            "PDF_TRANSLATE_CORS_ORIGINS": "https://paper.example.edu",
            "PDF_TRANSLATE_JWT_SECRET": "x" * 48,
            "PDF_TRANSLATE_JWT_TTL_MINUTES": "30",
            "PDF_TRANSLATE_DATA": str(root),
            "PDF_TRANSLATE_MAX_UPLOAD_MB": "64",
        }
        report = build_security_preflight(root, root / "web_jobs", env=env)
        codes = {issue["code"] for issue in report["issues"]}

        self.assertNotIn("DEFAULT_BOOTSTRAP_ADMIN_PASSWORD", codes)
        self.assertNotIn("CORS_ALLOW_ALL", codes)
        self.assertNotIn("JWT_SECRET_FILE_FALLBACK", codes)
        self.assertNotIn("JWT_TTL_DEFAULT", codes)
        self.assertTrue(report["ok"])
        self.assertEqual(report["jwt"]["ttl_minutes"], 30)
        self.assertNotIn("UPLOAD_LIMIT_DEFAULT", codes)
        self.assertEqual(report["upload"]["max_mb"], 64)
        self.assertFalse(report["cors"]["allow_all"])

    def test_invalid_upload_limit_falls_back_and_reports_issue(self) -> None:
        cfg = upload_limit_config(env={"PDF_TRANSLATE_MAX_UPLOAD_MB": "not-a-number"})
        self.assertEqual(cfg.max_mb, DEFAULT_MAX_UPLOAD_MB)
        self.assertEqual(cfg.invalid_reason, "not_an_integer")

        root = self._case_root("invalid-upload")
        report = build_security_preflight(
            root,
            root / "web_jobs",
            env={"PDF_TRANSLATE_MAX_UPLOAD_MB": "not-a-number"},
        )
        codes = {issue["code"] for issue in report["issues"]}
        self.assertIn("UPLOAD_LIMIT_INVALID", codes)

    def test_invalid_jwt_ttl_falls_back_and_reports_issue(self) -> None:
        cfg = jwt_ttl_config(env={"PDF_TRANSLATE_JWT_TTL_MINUTES": "0"})
        self.assertEqual(cfg.minutes, DEFAULT_JWT_TTL_MINUTES)
        self.assertEqual(cfg.invalid_reason, "outside_1_10080_minutes")

        root = self._case_root("invalid-jwt-ttl")
        report = build_security_preflight(
            root,
            root / "web_jobs",
            env={"PDF_TRANSLATE_JWT_TTL_MINUTES": "not-a-number"},
        )
        codes = {issue["code"] for issue in report["issues"]}
        self.assertIn("JWT_TTL_INVALID", codes)

    def test_production_startup_gate_blocks_insecure_environment(self) -> None:
        root = self._case_root("startup-blocked")

        with self.assertRaises(ProductionSecurityError) as raised:
            assert_production_security_ready(
                root,
                root / "web_jobs",
                env={"PDF_TRANSLATE_ENV": "production"},
            )

        report = raised.exception.report
        codes = {issue["code"] for issue in report["issues"]}
        self.assertTrue(report["production_mode"])
        self.assertIn("DEFAULT_BOOTSTRAP_ADMIN_PASSWORD", codes)
        self.assertIn("CORS_ALLOW_ALL", codes)
        self.assertIn("JWT_SECRET_FILE_FALLBACK", codes)

    def test_production_startup_gate_allows_hardened_environment(self) -> None:
        root = self._case_root("startup-hardened")
        report = assert_production_security_ready(
            root,
            root / "web_jobs",
            env={
                "PDF_TRANSLATE_ENV": "production",
                "PDF_TRANSLATE_BOOTSTRAP_ADMIN_PASSWORD": "change-me-before-release-123!",
                "PDF_TRANSLATE_CORS_ORIGINS": "https://paper.example.edu",
                "PDF_TRANSLATE_JWT_SECRET": "x" * 48,
                "PDF_TRANSLATE_JWT_TTL_MINUTES": "30",
                "PDF_TRANSLATE_DATA": str(root),
                "PDF_TRANSLATE_MAX_UPLOAD_MB": "64",
            },
        )

        self.assertTrue(report["production_mode"])
        self.assertTrue(report["ok"])

    def test_startup_gate_does_not_block_development_environment(self) -> None:
        root = self._case_root("startup-dev")
        report = assert_production_security_ready(root, root / "web_jobs", env={})

        self.assertFalse(report["production_mode"])
        self.assertFalse(report["ok"])

    def test_local_secret_storage_is_reported_without_leaking_values(self) -> None:
        root = self._case_root("stored-secrets")
        self._configure_db(root)
        database.kv_set("deepseek_api_key", "sk-test-placeholder")

        report = build_security_preflight(root, root / "web_jobs", env={}, db_path=root / "app.db")
        rendered = json.dumps(report, ensure_ascii=False)

        self.assertEqual(report["api_keys"]["stored_key_count"], 1)
        self.assertEqual(report["api_keys"]["stored_key_names"], ["deepseek_api_key"])
        self.assertIn("API_KEYS_STORED_IN_LOCAL_DB", {issue["code"] for issue in report["issues"]})
        self.assertNotIn("sk-test-placeholder", rendered)

    def test_admin_settings_snapshot_does_not_return_secret_values(self) -> None:
        root = self._case_root("settings-snapshot")
        self._configure_db(root)
        database.kv_set("deepseek_api_key", "sk-test-placeholder")
        database.kv_set("siliconflow_api_key", "sf-test-placeholder")

        snapshot = settings_service.admin_settings_snapshot()
        rendered = json.dumps(snapshot, ensure_ascii=False)

        self.assertNotIn("deepseek_api_key", snapshot)
        self.assertNotIn("siliconflow_api_key", snapshot)
        self.assertTrue(snapshot["secret_fields"]["deepseek_api_key"])
        self.assertTrue(snapshot["secret_fields"]["siliconflow_api_key"])
        self.assertNotIn("sk-test-placeholder", rendered)
        self.assertNotIn("sf-test-placeholder", rendered)

    def test_cli_security_check_uses_same_report_shape(self) -> None:
        root = self._case_root("cli")
        old_env = {
            key: os.environ.get(key)
            for key in (
                "PDF_TRANSLATE_ENV",
                "PDF_TRANSLATE_DEPLOYMENT_MODE",
                "PDF_TRANSLATE_BOOTSTRAP_ADMIN_PASSWORD",
                "PDF_TRANSLATE_CORS_ORIGINS",
                "PDF_TRANSLATE_JWT_SECRET",
                "PDF_TRANSLATE_MAX_UPLOAD_MB",
                "PDF_TRANSLATE_DATA",
                "PDF_TRANSLATE_WEB_DATA",
            )
        }
        try:
            for key in old_env:
                os.environ.pop(key, None)
            result = CliRunner().invoke(
                app,
                ["security-check", "--data-dir", str(root), "--data-root", str(root / "web_jobs")],
            )
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual(result.exit_code, 0, result.output)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["schema_version"], "security-preflight-v1")
        self.assertIn("issues", payload)
        self.assertEqual(payload["upload"]["max_mb"], DEFAULT_MAX_UPLOAD_MB)


if __name__ == "__main__":
    unittest.main()
