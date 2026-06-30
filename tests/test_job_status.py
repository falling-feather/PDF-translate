from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

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

    def test_merge_status_into_rows_preserves_database_created_at(self) -> None:
        root = self._case_root("merge-preserves-db-created-at")
        registry = JobRegistry(root)
        rec = registry.create_job(original_filename="paper.pdf")
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

    def test_merge_status_into_rows_marks_missing_runtime_status(self) -> None:
        root = self._case_root("merge-missing-status")
        registry = JobRegistry(root)
        merged = registry.merge_status_into_rows(
            [{"job_id": "missing-job", "created_at": "db-created-at"}]
        )

        self.assertEqual(merged[0]["created_at"], "db-created-at")
        self.assertFalse(merged[0]["status_available"])
        self.assertNotIn("status", merged[0])


if __name__ == "__main__":
    unittest.main()
