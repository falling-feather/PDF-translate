from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from typer.testing import CliRunner

from pdf_translate.cli import app
from pdf_translate.server.routes_web import _confirm_repair_publish_for_record
from pdf_translate.server.jobs import JOB_STATUS_SCHEMA_VERSION, JobRegistry


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
        (out / "repair_publish.md").write_text("# 发布确认", encoding="utf-8")
        (out / "published_full.md").write_text("published translation", encoding="utf-8")
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
        registry.update(rec.job_id, status="done", phase="done")

        merged = registry.merge_status_into_rows([{"job_id": rec.job_id}])

        self.assertTrue(merged[0]["repair_publish_report_ready"])
        self.assertGreater(merged[0]["repair_publish_report_bytes"], 0)
        self.assertTrue(merged[0]["repair_patch_review_ready"])
        self.assertGreater(merged[0]["repair_patch_review_bytes"], 0)
        self.assertEqual(merged[0]["repair_patch_review_count"], 3)
        self.assertEqual(merged[0]["repair_patch_review_required_count"], 1)
        self.assertEqual(merged[0]["repair_patch_review_blocking_count"], 1)
        self.assertTrue(merged[0]["repair_publish_confirmed"])
        self.assertTrue(merged[0]["repair_publish_published"])
        self.assertEqual(merged[0]["repair_publish_status"], "published_with_warnings")
        self.assertEqual(merged[0]["repair_publish_open_issue_count"], 2)
        self.assertTrue(merged[0]["repair_publish_rollback_available"])
        self.assertTrue(merged[0]["repair_published_full_ready"])
        self.assertGreater(merged[0]["repair_published_full_bytes"], 0)
        self.assertIn("repair_publish_open_issues", merged[0]["artifact_warnings"])
        self.assertIn("repair_patch_review_blocking_items", merged[0]["artifact_warnings"])

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
