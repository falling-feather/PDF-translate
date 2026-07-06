from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

import fitz
from typer.testing import CliRunner
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from pdf_translate.cli import app
from pdf_translate.config import AppConfig
from pdf_translate.pipeline_cancel import cancel_flag_path
from pdf_translate.server import database
from pdf_translate.server.auth_deps import Principal, bearer_principal
from pdf_translate.server.routes_web import (
    _confirm_repair_formal_replace_for_record,
    _confirm_repair_formal_rollback_for_record,
    _confirm_repair_publish_for_record,
    _confirm_repair_rollback_for_record,
    _confirm_table_structure_publish_for_record,
    _render_table_merged_cell_review_preview_for_record,
    register_web_routes,
)
from pdf_translate.server.jobs import JOB_STATUS_SCHEMA_VERSION, JobRegistry


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
        "default_translator": "echo",
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


class JobStatusSnapshotTests(unittest.TestCase):
    def _case_root(self, name: str) -> Path:
        root = Path.cwd() / "test-output" / "job-status" / name
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return root

    def test_status_snapshot_schema_hydrates_after_restart(self) -> None:
        root = self._case_root("snapshot-hydrate")
        registry = JobRegistry(root)
        rec = registry.create_job(
            owner_user_id=7,
            owner_username="alice",
            original_filename="paper.pdf",
            translate_mode="parallel",
            parallel_max_workers=3,
        )
        registry.update(
            rec.job_id,
            status="running",
            phase="translate",
            message="translating",
            chunk_total=3,
            chunk_index=1,
            chunk_id="c0001",
            main_pages=5,
            reference_pages=2,
        )

        status_path = root / rec.job_id / "web_status.json"
        raw = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertEqual(raw["schema_version"], JOB_STATUS_SCHEMA_VERSION)
        self.assertEqual(raw["status"], "running")
        self.assertEqual(raw["translate_mode"], "parallel")
        self.assertFalse(status_path.with_name("web_status.json.tmp").exists())

        restored = JobRegistry(root)
        restored.hydrate_from_disk()
        restored_rec = restored.get(rec.job_id)

        self.assertIsNotNone(restored_rec)
        assert restored_rec is not None
        self.assertEqual(restored_rec.status, "running")
        self.assertEqual(restored_rec.phase, "translate")
        self.assertEqual(restored_rec.chunk_total, 3)
        self.assertEqual(restored_rec.owner_user_id, 7)
        self.assertEqual(restored_rec.original_filename, "paper.pdf")
        self.assertEqual(restored_rec.work_dir, (root / rec.job_id).resolve())

    def test_hydrate_uses_status_file_directory_as_work_dir_boundary(self) -> None:
        root = self._case_root("hydrate-workdir-boundary")
        registry = JobRegistry(root)
        rec = registry.create_job(original_filename="paper.pdf")
        status_path = root / rec.job_id / "web_status.json"
        raw = json.loads(status_path.read_text(encoding="utf-8"))
        raw["work_dir"] = str(root.parent / "outside")
        status_path.write_text(json.dumps(raw), encoding="utf-8")

        restored = JobRegistry(root)
        restored.hydrate_from_disk()
        restored_rec = restored.get(rec.job_id)

        self.assertIsNotNone(restored_rec)
        assert restored_rec is not None
        self.assertEqual(restored_rec.work_dir, (root / rec.job_id).resolve())

    def test_hydrate_report_records_mismatched_snapshot_job_id(self) -> None:
        root = self._case_root("hydrate-job-id-mismatch")
        registry = JobRegistry(root)
        rec = registry.create_job(original_filename="paper.pdf")
        status_path = root / rec.job_id / "web_status.json"
        raw = json.loads(status_path.read_text(encoding="utf-8"))
        raw["job_id"] = "wrong-job-id"
        status_path.write_text(json.dumps(raw), encoding="utf-8")

        restored = JobRegistry(root)
        restored.hydrate_from_disk()
        report = restored.hydration_report()
        restored_rec = restored.get(rec.job_id)

        self.assertEqual(report["restored_count"], 1)
        self.assertEqual(report["job_id_mismatch_count"], 1)
        self.assertIn(rec.job_id, report["restored_job_ids"])
        self.assertIsNotNone(restored_rec)
        self.assertIsNone(restored.get("wrong-job-id"))
        assert restored_rec is not None
        self.assertTrue(restored_rec.recovered_from_disk)
        self.assertEqual(restored_rec.work_dir, (root / rec.job_id).resolve())

    def test_pipeline_state_diagnostic_reports_resume_boundary(self) -> None:
        root = self._case_root("pipeline-state-diagnostic")
        registry = JobRegistry(root)
        rec = registry.create_job(original_filename="paper.pdf")
        (rec.work_dir / "input.pdf").write_bytes(b"%PDF-1.4 test")
        out = rec.work_dir / "output"
        chunks = out / "chunks"
        chunks.mkdir(parents=True)
        (out / "chunks_manifest.json").write_text(
            json.dumps(
                [
                    {"chunk_id": "c0001"},
                    {"chunk_id": "c0002"},
                ]
            ),
            encoding="utf-8",
        )
        (out / "state.json").write_text(
            json.dumps({"completed": ["c0001"]}),
            encoding="utf-8",
        )
        (chunks / "c0001.md").write_text("translated", encoding="utf-8")
        registry.update(rec.job_id, status="running", phase="translate", chunk_total=2)

        restored = JobRegistry(root)
        restored.hydrate_from_disk()
        restored_rec = restored.get(rec.job_id)

        self.assertIsNotNone(restored_rec)
        assert restored_rec is not None
        summary = restored.diagnostic_summary_for_record(restored_rec)
        self.assertEqual(summary["pipeline_state_status"], "partial")
        self.assertEqual(summary["pipeline_completed_chunk_count"], 1)
        self.assertEqual(summary["pipeline_chunk_total"], 2)
        self.assertEqual(summary["pipeline_pending_chunk_ids"], ["c0002"])
        self.assertTrue(summary["pipeline_resume_ready"])
        self.assertEqual(summary["job_recovery_status"], "needs_manual_resume_or_cancel")
        self.assertIn("recovered_active_without_worker", summary["job_diagnostic_warnings"])

    def test_hydrate_report_tracks_active_recovered_status_counts(self) -> None:
        root = self._case_root("hydrate-status-counts")
        registry = JobRegistry(root)
        running = registry.create_job(original_filename="running.pdf")
        done = registry.create_job(original_filename="done.pdf")
        registry.update(running.job_id, status="running", phase="translate")
        registry.update(done.job_id, status="done", phase="done")

        restored = JobRegistry(root)
        restored.hydrate_from_disk()
        report = restored.hydration_report()

        self.assertEqual(report["restored_count"], 2)
        self.assertEqual(report["recovered_status_counts"]["running"], 1)
        self.assertEqual(report["recovered_status_counts"]["done"], 1)
        self.assertIn(running.job_id, report["active_recovered_job_ids"])
        self.assertNotIn(done.job_id, report["active_recovered_job_ids"])

    def test_log_job_finished_records_cancelled_terminal_audit(self) -> None:
        root = self._case_root("cancelled-terminal-audit")
        database.configure(root / "app.db")
        work = root / "job123"
        output = work / "output"
        output.mkdir(parents=True)
        (work / "input.pdf").write_bytes(b"%PDF-1.4 test")
        (output / "translated_full.md").write_text("partial", encoding="utf-8")

        database.log_job_finished(
            job_id="job123",
            user_id=7,
            username="alice",
            work_dir=work,
            ok=False,
            err="cancelled",
            status="cancelled",
            phase="cancelled",
            duration_seconds=3.5,
            run_started_at="2026-07-06T01:00:00Z",
            status_updated_at="2026-07-06T01:00:03Z",
            original_filename="paper.pdf",
            translate_mode="parallel",
            parallel_max_workers=2,
            error_code="TASK_CANCELLED",
            error_category="task",
            error_retryable=False,
            error_next_step="Task stopped.",
            error_source="server:run_pipeline",
        )

        events = database.list_audit(limit=10)
        self.assertEqual(events[0]["action"], "job_cancelled")
        self.assertEqual(events[0]["job_id"], "job123")
        detail = events[0]["detail"]
        self.assertEqual(detail["terminal_status"], "cancelled")
        self.assertEqual(detail["duration_seconds"], 3.5)
        self.assertEqual(detail["original_filename"], "paper.pdf")
        self.assertEqual(detail["translate_mode"], "parallel")
        self.assertEqual(detail["parallel_max_workers"], 2)
        self.assertEqual(detail["error_code"], "TASK_CANCELLED")
        self.assertFalse(detail["error_retryable"])
        self.assertEqual(detail["work_dir"], str(work.resolve()))
        self.assertFalse(detail["bundle_zip_ready"])

    def test_log_job_hydration_report_persists_recovery_scan(self) -> None:
        root = self._case_root("hydration-report-audit")
        database.configure(root / "app.db")
        jobs_root = root / "jobs"
        registry = JobRegistry(jobs_root)
        rec = registry.create_job(original_filename="paper.pdf")
        registry.update(rec.job_id, status="running", phase="translate")

        restored = JobRegistry(jobs_root)
        restored.hydrate_from_disk()
        database.log_job_hydration_report(restored.hydration_report())

        events = database.list_audit(limit=10)
        self.assertEqual(events[0]["action"], "job_hydration_report")
        detail = events[0]["detail"]
        self.assertEqual(detail["restored_count"], 1)
        self.assertEqual(detail["recovered_status_counts"]["running"], 1)
        self.assertIn(rec.job_id, detail["active_recovered_job_ids"])

    def test_cancel_job_writes_request_audit(self) -> None:
        root = self._case_root("cancel-request-audit")
        database.configure(root / "app.db")
        registry = JobRegistry(root / "jobs")
        rec = registry.create_job(
            owner_user_id=7,
            owner_username="alice",
            original_filename="paper.pdf",
        )
        registry.update(
            rec.job_id,
            status="running",
            phase="translate",
            message="translating",
            chunk_index=1,
            chunk_total=3,
            chunk_id="c0001",
        )

        api = FastAPI()
        api.include_router(register_web_routes(registry))
        api.dependency_overrides[bearer_principal] = lambda: Principal(
            user_id=7,
            username="alice",
            role="user",
        )
        self.addCleanup(api.dependency_overrides.clear)

        response = TestClient(api).post(f"/api/jobs/{rec.job_id}/cancel")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(cancel_flag_path(rec.work_dir).is_file())
        events = database.list_audit(limit=10)
        self.assertEqual(events[0]["action"], "job_cancel_requested")
        detail = events[0]["detail"]
        self.assertEqual(detail["previous_status"], "running")
        self.assertEqual(detail["previous_phase"], "translate")
        self.assertEqual(detail["previous_message"], "translating")
        self.assertEqual(detail["chunk_index"], 1)
        self.assertEqual(detail["chunk_total"], 3)
        self.assertEqual(detail["chunk_id"], "c0001")
        self.assertEqual(detail["requested_by_user_id"], 7)
        self.assertEqual(detail["requested_by_username"], "alice")

    def test_run_pipeline_records_started_and_error_audit(self) -> None:
        root = self._case_root("started-error-audit")
        database.configure(root / "app.db")
        registry = JobRegistry(root / "jobs")
        rec = registry.create_job(
            owner_user_id=7,
            owner_username="alice",
            original_filename="paper.pdf",
            translate_mode="parallel",
            parallel_max_workers=2,
        )

        registry.run_pipeline(
            rec.job_id,
            tail_fallback=False,
            pages_per_chunk=1,
            overlap_pages=0,
            backend="echo",
            max_chunks=None,
            cfg=_cfg(),
        )

        events = database.list_audit(limit=10)
        started = next(event for event in events if event["action"] == "job_started")
        failed = next(event for event in events if event["action"] == "job_error")
        self.assertEqual(started["job_id"], rec.job_id)
        self.assertEqual(started["detail"]["previous_status"], "queued")
        self.assertEqual(started["detail"]["status"], "running")
        self.assertEqual(started["detail"]["phase"], "init")
        self.assertEqual(started["detail"]["translate_mode"], "parallel")
        self.assertEqual(started["detail"]["parallel_max_workers"], 2)
        self.assertEqual(failed["detail"]["terminal_status"], "error")
        self.assertTrue(failed["detail"]["error_code"])
        self.assertEqual(failed["detail"]["original_filename"], "paper.pdf")

    def test_merge_status_into_rows_preserves_database_created_at(self) -> None:
        root = self._case_root("merge-preserves-db-created-at")
        registry = JobRegistry(root)
        rec = registry.create_job(original_filename="paper.pdf")
        (rec.work_dir / "input.pdf").write_bytes(b"%PDF-1.4 test")
        out = rec.work_dir / "output"
        out.mkdir()
        (out / "translated_full.md").write_text("translated", encoding="utf-8")
        (out / "translated_full.pdf").write_bytes(b"%PDF-1.4 translated")
        registry.update(rec.job_id, status="done", phase="done", duration_seconds=12.5)

        rows = [
            {
                "job_id": rec.job_id,
                "user_id": 7,
                "username": "alice",
                "original_filename": "paper.pdf",
                "created_at": "db-created-at",
            }
        ]
        merged = registry.merge_status_into_rows(rows)

        self.assertEqual(merged[0]["created_at"], "db-created-at")
        self.assertTrue(merged[0]["status_available"])
        self.assertEqual(merged[0]["status_schema_version"], JOB_STATUS_SCHEMA_VERSION)
        self.assertEqual(merged[0]["status"], "done")
        self.assertEqual(merged[0]["phase"], "done")
        self.assertEqual(merged[0]["runtime_created_at"], rec.created_at)
        self.assertEqual(merged[0]["duration_seconds"], 12.5)
        self.assertTrue(merged[0]["artifact_consistent"])
        self.assertEqual(merged[0]["artifact_consistency_status"], "ready")
        self.assertTrue(merged[0]["input_pdf_ready"])
        self.assertTrue(merged[0]["partial_output_ready"])
        self.assertTrue(merged[0]["translated_pdf_ready"])
        self.assertGreater(merged[0]["translated_pdf_bytes"], 0)
        self.assertTrue(merged[0]["bundle_zip_ready"])

    def test_artifact_summary_reports_repair_publish_status(self) -> None:
        root = self._case_root("repair-publish-artifacts")
        registry = JobRegistry(root)
        rec = registry.create_job(original_filename="paper.pdf")
        (rec.work_dir / "input.pdf").write_bytes(b"%PDF-1.4 test")
        out = rec.work_dir / "output"
        out.mkdir()
        (out / "translated_full.md").write_text("translated", encoding="utf-8")
        (out / "translated_full.pdf").write_bytes(b"%PDF-1.4 translated")
        (out / "repair_patch_review.md").write_text("# 补丁审核", encoding="utf-8")
        (out / "repair_patch_review.json").write_text(
            json.dumps(
                {
                    "schema_version": "repair-patch-review-v1",
                    "summary": {
                        "patch_count": 3,
                        "review_required_count": 1,
                        "publish_blocking_count": 1,
                    },
                }
            ),
            encoding="utf-8",
        )
        (out / "repair_effectiveness.md").write_text("# 修复效果", encoding="utf-8")
        (out / "repair_effectiveness.json").write_text(
            json.dumps(
                {
                    "schema_version": "repair-effectiveness-v1",
                    "summary": {
                        "status": "improved_with_regressions",
                        "before_issue_count": 6,
                        "after_issue_count": 3,
                        "issue_delta": 3,
                        "issue_reduction_rate": 0.5,
                        "resolved_issue_count": 4,
                        "persisted_issue_count": 2,
                        "new_issue_count": 1,
                        "improved_chunk_count": 2,
                        "regressed_chunk_count": 1,
                    },
                }
            ),
            encoding="utf-8",
        )
        (out / "table_merged_cell_review.md").write_text("# 表格确认", encoding="utf-8")
        (out / "table_merged_cell_review.json").write_text(
            json.dumps(
                {
                    "schema_version": "table-merged-cell-review-v1",
                    "summary": {
                        "candidate_review_count": 4,
                        "review_required_count": 2,
                        "pending_review_count": 1,
                        "visual_supported_count": 1,
                        "human_reviewed_count": 2,
                        "human_confirmed_count": 1,
                        "rejected_count": 1,
                        "needs_revision_count": 1,
                    },
                }
            ),
            encoding="utf-8",
        )
        (out / "repair_publish.md").write_text("# 发布确认", encoding="utf-8")
        (out / "published_full.md").write_text("published translation", encoding="utf-8")
        (out / "repair_rollback.md").write_text("# 回滚演练", encoding="utf-8")
        (out / "rollback_full.md").write_text("translated", encoding="utf-8")
        (out / "repair_formal_replace.md").write_text("# formal replace", encoding="utf-8")
        (out / "repair_formal_rollback.md").write_text("# formal rollback", encoding="utf-8")
        (out / "formal_full.md").write_text("formal original", encoding="utf-8")
        (out / "formal_full.before_repair.md").write_text("formal before", encoding="utf-8")
        (out / "formal_full.repair_applied.md").write_text("formal repaired", encoding="utf-8")
        (out / "repair_publish.json").write_text(
            json.dumps(
                {
                    "schema_version": "repair-publish-v1",
                    "summary": {
                        "confirmed": True,
                        "published": True,
                        "publish_status": "published_with_warnings",
                        "open_merge_issue_count": 2,
                        "rollback_available": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        (out / "repair_rollback.json").write_text(
            json.dumps(
                {
                    "schema_version": "repair-rollback-v1",
                    "summary": {
                        "rollback_available": True,
                        "confirmed": True,
                        "rollback_applied": True,
                        "rollback_status": "rolled_back",
                        "rollback_matches_original": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        (out / "repair_formal_replace.json").write_text(
            json.dumps(
                {
                    "schema_version": "repair-formal-replace-v1",
                    "summary": {
                        "replace_available": True,
                        "confirmed": True,
                        "replaced": True,
                        "replace_status": "replaced",
                        "formal_matches_published": True,
                        "rollback_available": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        (out / "repair_formal_rollback.json").write_text(
            json.dumps(
                {
                    "schema_version": "repair-formal-rollback-v1",
                    "summary": {
                        "rollback_available": True,
                        "confirmed": True,
                        "rollback_applied": True,
                        "rollback_status": "rolled_back",
                        "formal_matches_backup": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        (out / "table_structure_publish.md").write_text("# 表格发布", encoding="utf-8")
        (out / "table_reconstruction_confirmed.json").write_text(
            json.dumps({"schema_version": "table-reconstruction-v1"}),
            encoding="utf-8",
        )
        (out / "table_structure_publish.json").write_text(
            json.dumps(
                {
                    "schema_version": "table-structure-publish-v1",
                    "summary": {
                        "confirmed": True,
                        "published": True,
                        "publish_status": "published",
                        "blocking_review_count": 0,
                        "applied_confirmed_count": 1,
                        "rollback_available": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        registry.update(rec.job_id, status="done", phase="done")

        merged = registry.merge_status_into_rows([{"job_id": rec.job_id}])

        self.assertTrue(merged[0]["repair_publish_report_ready"])
        self.assertGreater(merged[0]["repair_publish_report_bytes"], 0)
        self.assertTrue(merged[0]["repair_rollback_report_ready"])
        self.assertGreater(merged[0]["repair_rollback_report_bytes"], 0)
        self.assertTrue(merged[0]["repair_patch_review_ready"])
        self.assertGreater(merged[0]["repair_patch_review_bytes"], 0)
        self.assertEqual(merged[0]["repair_patch_review_count"], 3)
        self.assertEqual(merged[0]["repair_patch_review_required_count"], 1)
        self.assertEqual(merged[0]["repair_patch_review_blocking_count"], 1)
        self.assertTrue(merged[0]["repair_effectiveness_report_ready"])
        self.assertGreater(merged[0]["repair_effectiveness_report_bytes"], 0)
        self.assertEqual(merged[0]["repair_effectiveness_status"], "improved_with_regressions")
        self.assertEqual(merged[0]["repair_effectiveness_before_issue_count"], 6)
        self.assertEqual(merged[0]["repair_effectiveness_after_issue_count"], 3)
        self.assertEqual(merged[0]["repair_effectiveness_issue_delta"], 3)
        self.assertEqual(merged[0]["repair_effectiveness_issue_reduction_rate"], 0.5)
        self.assertEqual(merged[0]["repair_effectiveness_resolved_issue_count"], 4)
        self.assertEqual(merged[0]["repair_effectiveness_persisted_issue_count"], 2)
        self.assertEqual(merged[0]["repair_effectiveness_new_issue_count"], 1)
        self.assertEqual(merged[0]["repair_effectiveness_improved_chunk_count"], 2)
        self.assertEqual(merged[0]["repair_effectiveness_regressed_chunk_count"], 1)
        self.assertTrue(merged[0]["table_merged_cell_review_ready"])
        self.assertGreater(merged[0]["table_merged_cell_review_bytes"], 0)
        self.assertEqual(merged[0]["table_merged_cell_review_count"], 4)
        self.assertEqual(merged[0]["table_merged_cell_review_required_count"], 2)
        self.assertEqual(merged[0]["table_merged_cell_review_pending_count"], 1)
        self.assertEqual(merged[0]["table_merged_cell_review_visual_supported_count"], 1)
        self.assertEqual(merged[0]["table_merged_cell_review_human_reviewed_count"], 2)
        self.assertEqual(merged[0]["table_merged_cell_review_human_confirmed_count"], 1)
        self.assertEqual(merged[0]["table_merged_cell_review_rejected_count"], 1)
        self.assertEqual(merged[0]["table_merged_cell_review_needs_revision_count"], 1)
        self.assertTrue(merged[0]["table_structure_publish_ready"])
        self.assertGreater(merged[0]["table_structure_publish_bytes"], 0)
        self.assertTrue(merged[0]["table_structure_publish_confirmed"])
        self.assertTrue(merged[0]["table_structure_publish_published"])
        self.assertEqual(merged[0]["table_structure_publish_status"], "published")
        self.assertEqual(merged[0]["table_structure_publish_blocking_count"], 0)
        self.assertEqual(merged[0]["table_structure_publish_applied_count"], 1)
        self.assertTrue(merged[0]["table_structure_publish_rollback_available"])
        self.assertTrue(merged[0]["table_reconstruction_confirmed_ready"])
        self.assertGreater(merged[0]["table_reconstruction_confirmed_bytes"], 0)
        self.assertTrue(merged[0]["repair_publish_confirmed"])
        self.assertTrue(merged[0]["repair_publish_published"])
        self.assertEqual(merged[0]["repair_publish_status"], "published_with_warnings")
        self.assertEqual(merged[0]["repair_publish_open_issue_count"], 2)
        self.assertTrue(merged[0]["repair_publish_rollback_available"])
        self.assertTrue(merged[0]["repair_rollback_available"])
        self.assertTrue(merged[0]["repair_rollback_confirmed"])
        self.assertTrue(merged[0]["repair_rollback_applied"])
        self.assertEqual(merged[0]["repair_rollback_status"], "rolled_back")
        self.assertTrue(merged[0]["repair_rollback_matches_original"])
        self.assertTrue(merged[0]["repair_published_full_ready"])
        self.assertGreater(merged[0]["repair_published_full_bytes"], 0)
        self.assertTrue(merged[0]["repair_rollback_full_ready"])
        self.assertGreater(merged[0]["repair_rollback_full_bytes"], 0)
        self.assertTrue(merged[0]["repair_formal_replace_report_ready"])
        self.assertGreater(merged[0]["repair_formal_replace_report_bytes"], 0)
        self.assertTrue(merged[0]["repair_formal_replace_available"])
        self.assertTrue(merged[0]["repair_formal_replace_confirmed"])
        self.assertTrue(merged[0]["repair_formal_replace_replaced"])
        self.assertEqual(merged[0]["repair_formal_replace_status"], "replaced")
        self.assertTrue(merged[0]["repair_formal_replace_matches_published"])
        self.assertTrue(merged[0]["repair_formal_replace_rollback_available"])
        self.assertTrue(merged[0]["repair_formal_rollback_report_ready"])
        self.assertGreater(merged[0]["repair_formal_rollback_report_bytes"], 0)
        self.assertTrue(merged[0]["repair_formal_rollback_available"])
        self.assertTrue(merged[0]["repair_formal_rollback_confirmed"])
        self.assertTrue(merged[0]["repair_formal_rollback_applied"])
        self.assertEqual(merged[0]["repair_formal_rollback_status"], "rolled_back")
        self.assertTrue(merged[0]["repair_formal_rollback_matches_backup"])
        self.assertTrue(merged[0]["repair_formal_full_ready"])
        self.assertGreater(merged[0]["repair_formal_full_bytes"], 0)
        self.assertTrue(merged[0]["repair_formal_backup_full_ready"])
        self.assertGreater(merged[0]["repair_formal_backup_full_bytes"], 0)
        self.assertTrue(merged[0]["repair_formal_active_before_rollback_full_ready"])
        self.assertGreater(merged[0]["repair_formal_active_before_rollback_full_bytes"], 0)
        self.assertIn("repair_publish_open_issues", merged[0]["artifact_warnings"])
        self.assertIn("repair_patch_review_blocking_items", merged[0]["artifact_warnings"])
        self.assertIn("table_merged_cell_review_required_items", merged[0]["artifact_warnings"])

    def test_confirm_repair_publish_for_completed_job_writes_publish_copy(self) -> None:
        root = self._case_root("repair-publish-confirm")
        registry = JobRegistry(root)
        rec = registry.create_job(original_filename="paper.pdf")
        (rec.work_dir / "input.pdf").write_bytes(b"%PDF-1.4 test")
        out = rec.work_dir / "output"
        out.mkdir()
        (out / "translated_full.md").write_text("original translation", encoding="utf-8")
        (out / "repaired_full.md").write_text("repaired translation", encoding="utf-8")
        (out / "repair_merge.json").write_text(
            json.dumps(
                {
                    "schema_version": "repair-merge-v1",
                    "summary": {
                        "applied_count": 1,
                        "patched_chunk_count": 1,
                        "manual_merge_required_count": 0,
                        "conflict_count": 0,
                        "skipped_count": 0,
                        "repaired_full_path": (out / "repaired_full.md").as_posix(),
                    },
                    "patches": [],
                }
            ),
            encoding="utf-8",
        )
        registry.update(rec.job_id, status="done", phase="done")

        report = _confirm_repair_publish_for_record(rec)

        self.assertTrue((out / "repair_patch_review.json").is_file())
        self.assertTrue((out / "repair_patch_review.md").is_file())
        self.assertTrue((out / "published_full.md").is_file())
        self.assertEqual((out / "published_full.md").read_text(encoding="utf-8"), "repaired translation")
        self.assertTrue(report["summary"]["confirmed"])
        self.assertTrue(report["summary"]["published"])
        self.assertEqual(report["summary"]["publish_status"], "published")
        merged = registry.merge_status_into_rows([{"job_id": rec.job_id}])
        self.assertTrue(merged[0]["repair_publish_published"])
        self.assertTrue(merged[0]["repair_published_full_ready"])

    def test_confirm_repair_rollback_for_completed_job_writes_rollback_copy(self) -> None:
        root = self._case_root("repair-rollback-confirm")
        registry = JobRegistry(root)
        rec = registry.create_job(original_filename="paper.pdf")
        (rec.work_dir / "input.pdf").write_bytes(b"%PDF-1.4 test")
        out = rec.work_dir / "output"
        out.mkdir()
        (out / "translated_full.md").write_text("original translation", encoding="utf-8")
        (out / "published_full.md").write_text("published translation", encoding="utf-8")
        (out / "repair_publish.json").write_text(
            json.dumps(
                {
                    "schema_version": "repair-publish-v1",
                    "summary": {
                        "confirmed": True,
                        "published": True,
                        "publish_status": "published",
                        "original_full_path": (out / "translated_full.md").as_posix(),
                        "published_full_path": (out / "published_full.md").as_posix(),
                    },
                }
            ),
            encoding="utf-8",
        )
        registry.update(rec.job_id, status="done", phase="done")

        report = _confirm_repair_rollback_for_record(rec)

        self.assertTrue((out / "repair_rollback.json").is_file())
        self.assertTrue((out / "repair_rollback.md").is_file())
        self.assertTrue((out / "rollback_full.md").is_file())
        self.assertEqual((out / "rollback_full.md").read_text(encoding="utf-8"), "original translation")
        self.assertEqual((out / "published_full.md").read_text(encoding="utf-8"), "published translation")
        self.assertTrue(report["summary"]["confirmed"])
        self.assertTrue(report["summary"]["rollback_applied"])
        self.assertTrue(report["summary"]["rollback_matches_original"])
        self.assertEqual(report["summary"]["rollback_status"], "rolled_back")
        merged = registry.merge_status_into_rows([{"job_id": rec.job_id}])
        self.assertTrue(merged[0]["repair_rollback_applied"])
        self.assertTrue(merged[0]["repair_rollback_full_ready"])

    def test_confirm_repair_formal_replace_for_completed_job_writes_formal_copy(self) -> None:
        root = self._case_root("repair-formal-replace-confirm")
        registry = JobRegistry(root)
        rec = registry.create_job(original_filename="paper.pdf")
        (rec.work_dir / "input.pdf").write_bytes(b"%PDF-1.4 test")
        out = rec.work_dir / "output"
        out.mkdir()
        (out / "translated_full.md").write_text("original translation", encoding="utf-8")
        (out / "published_full.md").write_text("published translation", encoding="utf-8")
        (out / "repair_publish.json").write_text(
            json.dumps(
                {
                    "schema_version": "repair-publish-v1",
                    "summary": {
                        "confirmed": True,
                        "published": True,
                        "publish_status": "published",
                        "original_full_path": (out / "translated_full.md").as_posix(),
                        "published_full_path": (out / "published_full.md").as_posix(),
                    },
                }
            ),
            encoding="utf-8",
        )
        registry.update(rec.job_id, status="done", phase="done")

        report = _confirm_repair_formal_replace_for_record(rec)

        self.assertTrue((out / "repair_formal_replace.json").is_file())
        self.assertTrue((out / "repair_formal_replace.md").is_file())
        self.assertEqual((out / "formal_full.md").read_text(encoding="utf-8"), "published translation")
        self.assertEqual((out / "formal_full.before_repair.md").read_text(encoding="utf-8"), "original translation")
        self.assertEqual((out / "translated_full.md").read_text(encoding="utf-8"), "original translation")
        self.assertEqual((out / "published_full.md").read_text(encoding="utf-8"), "published translation")
        self.assertTrue(report["summary"]["confirmed"])
        self.assertTrue(report["summary"]["replaced"])
        self.assertEqual(report["summary"]["replace_status"], "replaced")
        self.assertTrue(report["summary"]["formal_matches_published"])
        merged = registry.merge_status_into_rows([{"job_id": rec.job_id}])
        self.assertTrue(merged[0]["repair_formal_replace_replaced"])
        self.assertTrue(merged[0]["repair_formal_full_ready"])
        self.assertTrue(merged[0]["repair_formal_backup_full_ready"])

    def test_confirm_repair_formal_rollback_for_completed_job_restores_formal_copy(self) -> None:
        root = self._case_root("repair-formal-rollback-confirm")
        registry = JobRegistry(root)
        rec = registry.create_job(original_filename="paper.pdf")
        (rec.work_dir / "input.pdf").write_bytes(b"%PDF-1.4 test")
        out = rec.work_dir / "output"
        out.mkdir()
        (out / "translated_full.md").write_text("original translation", encoding="utf-8")
        (out / "published_full.md").write_text("published translation", encoding="utf-8")
        (out / "formal_full.md").write_text("published translation", encoding="utf-8")
        (out / "formal_full.before_repair.md").write_text("original translation", encoding="utf-8")
        (out / "repair_formal_replace.json").write_text(
            json.dumps(
                {
                    "schema_version": "repair-formal-replace-v1",
                    "summary": {
                        "confirmed": True,
                        "replaced": True,
                        "replace_status": "replaced",
                        "formal_full_path": (out / "formal_full.md").as_posix(),
                        "backup_full_path": (out / "formal_full.before_repair.md").as_posix(),
                        "published_full_path": (out / "published_full.md").as_posix(),
                    },
                }
            ),
            encoding="utf-8",
        )
        registry.update(rec.job_id, status="done", phase="done")

        report = _confirm_repair_formal_rollback_for_record(rec)

        self.assertTrue((out / "repair_formal_rollback.json").is_file())
        self.assertTrue((out / "repair_formal_rollback.md").is_file())
        self.assertEqual((out / "formal_full.md").read_text(encoding="utf-8"), "original translation")
        self.assertEqual((out / "formal_full.repair_applied.md").read_text(encoding="utf-8"), "published translation")
        self.assertEqual((out / "published_full.md").read_text(encoding="utf-8"), "published translation")
        self.assertEqual((out / "translated_full.md").read_text(encoding="utf-8"), "original translation")
        self.assertTrue(report["summary"]["confirmed"])
        self.assertTrue(report["summary"]["rollback_applied"])
        self.assertEqual(report["summary"]["rollback_status"], "rolled_back")
        self.assertTrue(report["summary"]["formal_matches_backup"])
        merged = registry.merge_status_into_rows([{"job_id": rec.job_id}])
        self.assertTrue(merged[0]["repair_formal_rollback_applied"])
        self.assertTrue(merged[0]["repair_formal_active_before_rollback_full_ready"])

    def test_confirm_repair_publish_respects_existing_patch_review_gate(self) -> None:
        root = self._case_root("repair-publish-patch-review-gate")
        registry = JobRegistry(root)
        rec = registry.create_job(original_filename="paper.pdf")
        (rec.work_dir / "input.pdf").write_bytes(b"%PDF-1.4 test")
        out = rec.work_dir / "output"
        out.mkdir()
        (out / "translated_full.md").write_text("original translation", encoding="utf-8")
        (out / "repaired_full.md").write_text("repaired translation", encoding="utf-8")
        (out / "repair_merge.json").write_text(
            json.dumps(
                {
                    "schema_version": "repair-merge-v1",
                    "summary": {
                        "applied_count": 1,
                        "patched_chunk_count": 1,
                        "manual_merge_required_count": 0,
                        "conflict_count": 0,
                        "skipped_count": 0,
                        "repaired_full_path": (out / "repaired_full.md").as_posix(),
                    },
                    "patches": [
                        {
                            "request_id": "rq0000",
                            "repair_id": "rp0000",
                            "chunk_id": "c0000",
                            "status": "applied",
                            "strategy": "replace_chunk",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (out / "repair_patch_review.json").write_text(
            json.dumps(
                {
                    "schema_version": "repair-patch-review-v1",
                    "summary": {
                        "repair_merge_schema_version": "repair-merge-v1",
                        "patch_count": 1,
                        "auto_merge_safe_count": 1,
                        "effective_safe_count": 0,
                        "review_required_count": 1,
                        "publish_blocking_count": 1,
                        "human_reviewed_count": 1,
                        "human_rejected_count": 1,
                        "human_decision_counts": {"reject": 1},
                        "effective_decision_counts": {"reject_candidate": 1},
                    },
                    "patch_reviews": [
                        {
                            "review_id": "pr0000",
                            "chunk_id": "c0000",
                            "merge_status": "applied",
                            "merge_strategy": "replace_chunk",
                            "default_decision": "approve_candidate",
                            "effective_decision": "reject_candidate",
                            "human_decision": "reject",
                            "publish_blocking": True,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        registry.update(rec.job_id, status="done", phase="done")

        with self.assertRaises(HTTPException) as ctx:
            _confirm_repair_publish_for_record(rec)

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertFalse((out / "published_full.md").exists())
        publish_report = json.loads((out / "repair_publish.json").read_text(encoding="utf-8"))
        self.assertEqual(publish_report["summary"]["publish_status"], "blocked_patch_review")
        self.assertEqual(publish_report["summary"]["patch_review_blocking_count"], 1)
        review_report = json.loads((out / "repair_patch_review.json").read_text(encoding="utf-8"))
        self.assertEqual(review_report["patch_reviews"][0]["human_decision"], "reject")

    def test_confirm_table_structure_publish_for_completed_job_writes_confirmed_copy(self) -> None:
        root = self._case_root("table-structure-publish-confirm")
        registry = JobRegistry(root)
        rec = registry.create_job(original_filename="paper.pdf")
        (rec.work_dir / "input.pdf").write_bytes(b"%PDF-1.4 test")
        out = rec.work_dir / "output"
        out.mkdir()
        (out / "table_reconstruction.json").write_text(
            json.dumps(
                {
                    "schema_version": "table-reconstruction-v1",
                    "doc_id": "table-doc",
                    "summary": {"table_count": 1, "merged_cell_candidate_count": 1},
                    "tables": [
                        {
                            "table_id": "p1-b0000",
                            "block_id": "p1-b0000",
                            "merged_cell_candidates": [
                                {
                                    "span_type": "colspan",
                                    "row_index": 0,
                                    "column_index": 0,
                                    "row_span": 1,
                                    "column_span": 2,
                                    "reason": "visual_span_supported",
                                    "text": "Dataset metrics",
                                    "candidate_status": "visually_supported",
                                    "covered_cells": [
                                        {"row_index": 0, "column_index": 0},
                                        {"row_index": 0, "column_index": 1},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (out / "table_merged_cell_review.json").write_text(
            json.dumps(
                {
                    "schema_version": "table-merged-cell-review-v1",
                    "summary": {
                        "candidate_review_count": 1,
                        "review_required_count": 0,
                        "human_reviewed_count": 1,
                        "human_confirmed_count": 1,
                        "rejected_count": 0,
                        "needs_revision_count": 0,
                    },
                    "candidate_reviews": [
                        {
                            "review_id": "tmc-0001-p1-b0000-r0c0",
                            "table_id": "p1-b0000",
                            "block_id": "p1-b0000",
                            "span_type": "colspan",
                            "row_index": 0,
                            "column_index": 0,
                            "row_span": 1,
                            "column_span": 2,
                            "reason": "visual_span_supported",
                            "covered_cells": [
                                {"row_index": 0, "column_index": 0},
                                {"row_index": 0, "column_index": 1},
                            ],
                            "human_decision": "confirm",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        chunk_dir = out / "chunks"
        chunk_dir.mkdir()
        (chunk_dir / "c0000.md").write_text(
            "| Header | Value |\n| --- | --- |\n| Dataset metrics | 91.2 |\n",
            encoding="utf-8",
        )
        (out / "chunks_manifest.json").write_text(
            json.dumps(
                [
                    {
                        "chunk_id": "c0000",
                        "pages_1based": [1],
                        "link_count": 0,
                        "image_count": 0,
                        "block_ids": ["p1-b0000"],
                        "structural_relation_ids": [],
                        "budget": {},
                    }
                ]
            ),
            encoding="utf-8",
        )
        registry.update(rec.job_id, status="done", phase="done")

        report = _confirm_table_structure_publish_for_record(rec)

        self.assertTrue(report["summary"]["published"])
        self.assertEqual(report["summary"]["publish_status"], "published")
        self.assertEqual(report["summary"]["applied_confirmed_count"], 1)
        self.assertEqual(report["summary"]["structure_patch_count"], 1)
        self.assertEqual(report["summary"]["structure_patch_applied_count"], 1)
        self.assertEqual(report["summary"]["structure_patch_covered_cell_count"], 1)
        self.assertTrue((out / "table_structure_publish.json").is_file())
        self.assertTrue((out / "table_structure_publish.md").is_file())
        self.assertTrue((out / "table_reconstruction_confirmed.json").is_file())
        confirmed = json.loads((out / "table_reconstruction_confirmed.json").read_text(encoding="utf-8"))
        self.assertEqual(confirmed["summary"]["confirmed_merged_cell_candidate_count"], 1)
        self.assertEqual(confirmed["summary"]["table_structure_patch_count"], 1)
        self.assertEqual(confirmed["summary"]["table_structure_patch_covered_cell_count"], 1)
        self.assertEqual(confirmed["tables"][0]["confirmed_merged_cell_candidate_count"], 1)
        self.assertEqual(confirmed["tables"][0]["structure_patches"][0]["source_review_id"], "tmc-0001-p1-b0000-r0c0")
        self.assertEqual(report["summary"]["translated_pdf_refresh_status"], "refreshed")
        self.assertEqual(report["summary"]["translated_pdf_table_reconstruction_source"], "confirmed")
        self.assertEqual(report["summary"]["translated_pdf_confirmed_candidate_reference_count"], 1)
        self.assertTrue((out / "translated_full.pdf").is_file())
        translated_pdf_report = json.loads((out / "translated_pdf_report.json").read_text(encoding="utf-8"))
        self.assertEqual(translated_pdf_report["table_reconstruction_source"], "confirmed")
        self.assertEqual(
            translated_pdf_report["summary"]["confirmed_merged_cell_candidate_reference_count"],
            1,
        )
        self.assertEqual(translated_pdf_report["summary"]["table_structure_patch_reference_count"], 1)
        merged = registry.merge_status_into_rows([{"job_id": rec.job_id}])
        self.assertTrue(merged[0]["table_structure_publish_published"])
        self.assertEqual(merged[0]["table_structure_patch_count"], 1)
        self.assertEqual(merged[0]["table_structure_patch_applied_count"], 1)
        self.assertEqual(merged[0]["table_structure_patch_covered_cell_count"], 1)
        self.assertTrue(merged[0]["table_reconstruction_confirmed_ready"])

    def test_confirm_table_structure_publish_blocks_and_removes_stale_copy(self) -> None:
        root = self._case_root("table-structure-publish-review-gate")
        registry = JobRegistry(root)
        rec = registry.create_job(original_filename="paper.pdf")
        (rec.work_dir / "input.pdf").write_bytes(b"%PDF-1.4 test")
        out = rec.work_dir / "output"
        out.mkdir()
        (out / "table_reconstruction_confirmed.json").write_text("stale", encoding="utf-8")
        (out / "table_reconstruction.json").write_text(
            json.dumps(
                {
                    "schema_version": "table-reconstruction-v1",
                    "doc_id": "table-doc",
                    "summary": {"table_count": 1, "merged_cell_candidate_count": 1},
                    "tables": [
                        {
                            "table_id": "p1-b0000",
                            "block_id": "p1-b0000",
                            "merged_cell_candidates": [
                                {
                                    "span_type": "colspan",
                                    "row_index": 0,
                                    "column_index": 0,
                                    "row_span": 1,
                                    "column_span": 2,
                                    "reason": "visual_span_supported",
                                    "candidate_status": "visually_supported",
                                }
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (out / "table_merged_cell_review.json").write_text(
            json.dumps(
                {
                    "schema_version": "table-merged-cell-review-v1",
                    "summary": {
                        "candidate_review_count": 1,
                        "review_required_count": 1,
                        "human_reviewed_count": 1,
                        "human_confirmed_count": 0,
                        "rejected_count": 0,
                        "needs_revision_count": 1,
                    },
                    "candidate_reviews": [
                        {
                            "review_id": "tmc-0001-p1-b0000-r0c0",
                            "table_id": "p1-b0000",
                            "block_id": "p1-b0000",
                            "span_type": "colspan",
                            "row_index": 0,
                            "column_index": 0,
                            "row_span": 1,
                            "column_span": 2,
                            "reason": "visual_span_supported",
                            "human_decision": "needs_revision",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        registry.update(rec.job_id, status="done", phase="done")

        with self.assertRaises(HTTPException) as ctx:
            _confirm_table_structure_publish_for_record(rec)

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertFalse((out / "table_reconstruction_confirmed.json").exists())
        publish_report = json.loads((out / "table_structure_publish.json").read_text(encoding="utf-8"))
        self.assertTrue(publish_report["summary"]["confirmed"])
        self.assertFalse(publish_report["summary"]["published"])
        self.assertEqual(publish_report["summary"]["publish_status"], "blocked_review_required")
        self.assertEqual(publish_report["summary"]["blocking_review_count"], 1)

    def test_table_merged_cell_review_preview_renders_png(self) -> None:
        root = self._case_root("table-review-preview")
        registry = JobRegistry(root)
        rec = registry.create_job(original_filename="paper.pdf")
        doc = fitz.open()
        page = doc.new_page(width=200, height=160)
        page.insert_text((24, 36), "merged table candidate")
        doc.save(rec.work_dir / "input.pdf")
        doc.close()

        png = _render_table_merged_cell_review_preview_for_record(
            rec,
            {
                "page_no": 1,
                "bbox_evidence": {
                    "span_bbox": [20, 20, 130, 70],
                    "evidence_bbox": [24, 24, 80, 58],
                },
            },
        )

        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertGreater(len(png), 1000)

    def test_table_merged_cell_review_preview_rejects_invalid_page_no(self) -> None:
        root = self._case_root("table-review-preview-invalid-page")
        registry = JobRegistry(root)
        rec = registry.create_job(original_filename="paper.pdf")
        doc = fitz.open()
        doc.new_page(width=200, height=160)
        doc.save(rec.work_dir / "input.pdf")
        doc.close()

        with self.assertRaises(HTTPException) as ctx:
            _render_table_merged_cell_review_preview_for_record(rec, {"page_no": 0})

        self.assertEqual(ctx.exception.status_code, 400)

    def test_artifact_summary_marks_done_without_translation_inconsistent(self) -> None:
        root = self._case_root("artifact-inconsistent")
        registry = JobRegistry(root)
        rec = registry.create_job(original_filename="paper.pdf")
        (rec.work_dir / "input.pdf").write_bytes(b"%PDF-1.4 test")
        registry.update(rec.job_id, status="done", phase="done")

        merged = registry.merge_status_into_rows([{"job_id": rec.job_id}])

        self.assertFalse(merged[0]["artifact_consistent"])
        self.assertEqual(merged[0]["artifact_consistency_status"], "inconsistent")
        self.assertIn("translated_md_missing_for_done", merged[0]["artifact_warnings"])
        self.assertTrue(merged[0]["input_pdf_ready"])
        self.assertFalse(merged[0]["partial_output_ready"])
        self.assertFalse(merged[0]["translated_pdf_ready"])
        self.assertEqual(merged[0]["translated_pdf_bytes"], 0)
        self.assertFalse(merged[0]["repair_publish_report_ready"])
        self.assertFalse(merged[0]["repair_patch_review_ready"])
        self.assertFalse(merged[0]["repair_effectiveness_report_ready"])
        self.assertEqual(merged[0]["repair_effectiveness_report_bytes"], 0)
        self.assertFalse(merged[0]["table_merged_cell_review_ready"])
        self.assertFalse(merged[0]["table_structure_publish_ready"])
        self.assertFalse(merged[0]["table_reconstruction_confirmed_ready"])
        self.assertFalse(merged[0]["repair_published_full_ready"])
        self.assertFalse(merged[0]["bundle_zip_ready"])

    def test_storage_drift_reports_missing_and_unindexed_work_dirs(self) -> None:
        root = self._case_root("storage-drift")
        registry = JobRegistry(root)
        rec = registry.create_job(original_filename="paper.pdf")
        orphan = root / "orphan-job"
        orphan.mkdir()
        (orphan / "web_status.json").write_text("{}", encoding="utf-8")

        drift = registry.storage_drift({rec.job_id, "db-only-job"})

        self.assertEqual(drift["indexed_job_count"], 2)
        self.assertIn("db-only-job", drift["missing_work_dir_job_ids"])
        self.assertIn("orphan-job", drift["unindexed_work_dir_job_ids"])
        self.assertIn(rec.job_id, drift["active_job_ids"])

    def test_remove_job_ignores_path_traversal_job_id(self) -> None:
        root = self._case_root("remove-job-boundary")
        registry = JobRegistry(root)
        outside = root.parent / "outside-keep"
        outside.mkdir(exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(outside, ignore_errors=True))

        removed = registry.remove_job("../outside-keep")

        self.assertFalse(removed)
        self.assertTrue(outside.is_dir())

    def test_merge_status_into_rows_marks_missing_runtime_status(self) -> None:
        root = self._case_root("merge-missing-status")
        registry = JobRegistry(root)
        merged = registry.merge_status_into_rows(
            [{"job_id": "missing-job", "created_at": "db-created-at"}]
        )

        self.assertEqual(merged[0]["created_at"], "db-created-at")
        self.assertFalse(merged[0]["status_available"])
        self.assertEqual(merged[0]["artifact_consistency_status"], "missing_status")
        self.assertIn("status_snapshot_missing", merged[0]["artifact_warnings"])
        self.assertFalse(merged[0]["translated_pdf_ready"])
        self.assertEqual(merged[0]["translated_pdf_bytes"], 0)
        self.assertFalse(merged[0]["repair_publish_report_ready"])
        self.assertEqual(merged[0]["repair_publish_open_issue_count"], 0)
        self.assertFalse(merged[0]["repair_effectiveness_report_ready"])
        self.assertEqual(merged[0]["repair_effectiveness_report_bytes"], 0)
        self.assertEqual(merged[0]["repair_effectiveness_status"], "")
        self.assertEqual(merged[0]["repair_effectiveness_before_issue_count"], 0)
        self.assertEqual(merged[0]["repair_effectiveness_after_issue_count"], 0)
        self.assertEqual(merged[0]["repair_effectiveness_issue_delta"], 0)
        self.assertEqual(merged[0]["repair_effectiveness_issue_reduction_rate"], 0.0)
        self.assertFalse(merged[0]["table_merged_cell_review_ready"])
        self.assertFalse(merged[0]["table_structure_publish_ready"])
        self.assertFalse(merged[0]["table_reconstruction_confirmed_ready"])
        self.assertFalse(merged[0]["repair_published_full_ready"])
        self.assertNotIn("status", merged[0])

    def test_cli_web_status_reads_same_diagnostic_summary(self) -> None:
        root = self._case_root("cli-web-status")
        registry = JobRegistry(root)
        rec = registry.create_job(original_filename="paper.pdf")
        (rec.work_dir / "input.pdf").write_bytes(b"%PDF-1.4 test")
        out = rec.work_dir / "output"
        chunks = out / "chunks"
        chunks.mkdir(parents=True)
        (out / "chunks_manifest.json").write_text(
            json.dumps([{"chunk_id": "c0001"}]),
            encoding="utf-8",
        )
        (out / "state.json").write_text(
            json.dumps({"completed": ["c0001"]}),
            encoding="utf-8",
        )
        (chunks / "c0001.md").write_text("translated", encoding="utf-8")
        (out / "translated_full.md").write_text("translated", encoding="utf-8")
        registry.update(rec.job_id, status="done", phase="done")

        result = CliRunner().invoke(
            app,
            ["web-status", "--data-root", str(root), "--job-id", rec.job_id],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["hydration"]["restored_count"], 1)
        self.assertEqual(payload["job"]["job_id"], rec.job_id)
        self.assertEqual(payload["job"]["status"], "done")
        self.assertEqual(payload["job"]["pipeline_state_status"], "complete")
        self.assertEqual(payload["job"]["pipeline_completion_ratio"], 1.0)
        self.assertEqual(payload["job"]["artifact_consistency_status"], "ready")


if __name__ == "__main__":
    unittest.main()
