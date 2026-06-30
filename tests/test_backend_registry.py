from __future__ import annotations

import unittest
from unittest.mock import patch

from pdf_translate.config import AppConfig
from pdf_translate.costing import backend_model_name
from pdf_translate.error_codes import PdfTranslateError
from pdf_translate.server import settings_service
from pdf_translate.server.routes_web import _build_runtime_cfg_for_custom_api
from pdf_translate.translators.factory import build_translator
from pdf_translate.translators.registry import (
    backend_catalog,
    backend_ids,
    custom_api_backend_ids,
    default_enabled_backend_ids,
    get_backend_spec,
    normalize_backend_id,
)


def _cfg(**overrides) -> AppConfig:
    values = {
        "openai_api_key": None,
        "openai_base_url": "https://api.openai.com/v1",
        "openai_model": "gpt-test",
        "ollama_base_url": "http://127.0.0.1:11434/v1",
        "ollama_model": "llama-test",
        "deepl_api_key": None,
        "deepl_api_url": "https://api-free.deepl.com/v2/translate",
        "deepseek_api_key": None,
        "deepseek_base_url": "https://api.deepseek.com/v1",
        "deepseek_model": "deepseek-chat",
        "default_translator": "deepseek",
        "http_timeout_s": 120.0,
        "survey_enabled": False,
        "siliconflow_api_key": None,
        "siliconflow_base_url": "https://api.siliconflow.com/v1",
        "siliconflow_survey_model": "",
        "siliconflow_vision_model": "",
        "survey_max_text_chars": 12000,
        "planner_enabled": False,
        "planner_api_key": None,
        "planner_base_url": "https://api.siliconflow.com/v1",
        "planner_model": "",
        "cost_profile_json": "",
        "cost_profile_path": "",
        "cost_default_currency": "USD",
    }
    values.update(overrides)
    return AppConfig(**values)


class BackendRegistryTests(unittest.TestCase):
    def test_registry_exposes_canonical_backend_catalog(self) -> None:
        self.assertEqual(normalize_backend_id("dry"), "echo")
        self.assertEqual(normalize_backend_id("NOOP"), "echo")
        self.assertIn("deepseek", backend_ids())
        self.assertEqual(default_enabled_backend_ids(), ["echo", "deepseek"])
        self.assertIn("openai", custom_api_backend_ids())

        catalog = backend_catalog()
        deepseek = next(item for item in catalog if item["id"] == "deepseek")
        self.assertEqual(deepseek["model_setting"], "deepseek_model")
        self.assertTrue(deepseek["requires_api_key"])
        self.assertFalse(get_backend_spec("echo").supports_deferral)

    def test_factory_and_costing_use_registry_normalization(self) -> None:
        translator = build_translator("dry", _cfg())
        self.assertEqual(translator.name, "echo")
        self.assertEqual(backend_model_name("deepseek", _cfg(deepseek_model="deepseek-reasoner")), "deepseek-reasoner")

        with self.assertRaises(PdfTranslateError) as caught:
            build_translator("missing-backend", _cfg())

        info = caught.exception.error_info
        self.assertEqual(info.code, "CONFIG_INVALID_BACKEND")
        for backend_id in backend_ids():
            self.assertIn(backend_id, info.detail)

    def test_settings_service_sanitizes_enabled_backends_from_registry(self) -> None:
        self.assertEqual(
            settings_service.sanitize_backend_ids(["dry", "deepseek", "echo"]),
            ["echo", "deepseek"],
        )

        with patch.object(settings_service.database, "kv_get_json", return_value=["dry", "bad", "deepseek"]):
            self.assertEqual(settings_service.enabled_backends(), ["echo", "deepseek"])

    def test_custom_api_config_uses_backend_spec_fields(self) -> None:
        cfg = _cfg()
        openai_cfg = _build_runtime_cfg_for_custom_api(
            cfg=cfg,
            backend="openai",
            api_key="sk-test",
            api_base_url="https://example.test/v1/",
            api_model="gpt-custom",
        )
        self.assertEqual(openai_cfg.openai_api_key, "sk-test")
        self.assertEqual(openai_cfg.openai_base_url, "https://example.test/v1")
        self.assertEqual(openai_cfg.openai_model, "gpt-custom")

        ollama_cfg = _build_runtime_cfg_for_custom_api(
            cfg=cfg,
            backend="ollama",
            api_key="",
            api_base_url="http://127.0.0.1:11434/v1/",
            api_model="qwen-local",
        )
        self.assertEqual(ollama_cfg.ollama_base_url, "http://127.0.0.1:11434/v1")
        self.assertEqual(ollama_cfg.ollama_model, "qwen-local")


if __name__ == "__main__":
    unittest.main()
