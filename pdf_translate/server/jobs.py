from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from pdf_translate.config import AppConfig
from pdf_translate.error_codes import error_info_from_exception, make_error_info
from pdf_translate.pipeline import export_links, init_workdir, run_split, run_translate
from pdf_translate.pipeline_cancel import JobCancelled, is_cancel_requested

JOB_STATUS_SCHEMA_VERSION = "web-job-status-v1"
JOB_HYDRATION_REPORT_SCHEMA_VERSION = "web-job-hydration-report-v1"
VALID_JOB_STATUSES = {"queued", "running", "done", "error", "cancelled"}
VALID_AUTO_RESUME_POLICIES = {"off", "safe", "all"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def job_auto_resume_policy_from_env() -> str:
    raw = (
        os.getenv("PDF_TRANSLATE_JOB_AUTO_RESUME")
        or os.getenv("PDF_TRANSLATE_RECOVERY_POLICY")
        or "off"
    ).strip().lower()
    aliases = {
        "0": "off",
        "false": "off",
        "no": "off",
        "manual": "off",
        "1": "safe",
        "true": "safe",
        "yes": "safe",
        "requeue": "safe",
    }
    raw = aliases.get(raw, raw)
    return raw if raw in VALID_AUTO_RESUME_POLICIES else "off"


def job_auto_resume_max_from_env(default: int = 2) -> int:
    raw = os.getenv("PDF_TRANSLATE_JOB_AUTO_RESUME_MAX", "").strip()
    if not raw:
        return default
    try:
        return max(0, min(int(raw), 32))
    except ValueError:
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class JobPublic:
    job_id: str
    status: Literal["queued", "running", "done", "error", "cancelled"]
    phase: str
    message: str
    chunk_total: int | None
    chunk_index: int | None
    chunk_id: str | None
    error: str | None
    error_code: str | None
    error_category: str | None
    error_retryable: bool | None
    error_next_step: str | None
    error_source: str | None
    error_http_status: int | None
    created_at: str
    updated_at: str
    main_pages: int | None = None
    reference_pages: int | None = None
    translate_mode: str | None = None
    parallel_max_workers: int | None = None
    backend: str | None = None
    tail_fallback: bool = False
    pages_per_chunk: int = 3
    overlap_pages: int = 1
    max_chunks: int | None = None
    use_custom_api: bool = False
    duration_seconds: float | None = None
    run_started_at: str | None = None
    recovered_from_disk: bool = False


@dataclass
class JobRecord:
    job_id: str
    work_dir: Path
    status: Literal["queued", "running", "done", "error", "cancelled"] = "queued"
    phase: str = "queued"
    message: str = ""
    chunk_total: int | None = None
    chunk_index: int | None = None
    chunk_id: str | None = None
    error: str | None = None
    error_code: str | None = None
    error_category: str | None = None
    error_retryable: bool | None = None
    error_next_step: str | None = None
    error_source: str | None = None
    error_http_status: int | None = None
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    main_pages: int | None = None
    reference_pages: int | None = None
    owner_user_id: int | None = None
    owner_username: str | None = None
    original_filename: str | None = None
    translate_mode: str = "serial"
    parallel_max_workers: int | None = None
    backend: str | None = None
    tail_fallback: bool = False
    pages_per_chunk: int = 3
    overlap_pages: int = 1
    max_chunks: int | None = None
    use_custom_api: bool = False
    duration_seconds: float | None = None
    run_started_at: str | None = None
    recovered_from_disk: bool = False

    def touch(self) -> None:
        self.updated_at = _utc_now_iso()

    def to_public(self) -> JobPublic:
        return JobPublic(
            job_id=self.job_id,
            status=self.status,
            phase=self.phase,
            message=self.message,
            chunk_total=self.chunk_total,
            chunk_index=self.chunk_index,
            chunk_id=self.chunk_id,
            error=self.error,
            error_code=self.error_code,
            error_category=self.error_category,
            error_retryable=self.error_retryable,
            error_next_step=self.error_next_step,
            error_source=self.error_source,
            error_http_status=self.error_http_status,
            created_at=self.created_at,
            updated_at=self.updated_at,
            main_pages=self.main_pages,
            reference_pages=self.reference_pages,
            translate_mode=self.translate_mode,
            parallel_max_workers=self.parallel_max_workers,
            backend=self.backend,
            tail_fallback=self.tail_fallback,
            pages_per_chunk=self.pages_per_chunk,
            overlap_pages=self.overlap_pages,
            max_chunks=self.max_chunks,
            use_custom_api=self.use_custom_api,
            duration_seconds=self.duration_seconds,
            run_started_at=self.run_started_at,
            recovered_from_disk=self.recovered_from_disk,
        )

    def to_status_dict(self) -> dict[str, Any]:
        """Persistent status snapshot used by Web APIs after restart."""
        return {
            "schema_version": JOB_STATUS_SCHEMA_VERSION,
            "job_id": self.job_id,
            "work_dir": str(self.work_dir),
            "status": self.status,
            "phase": self.phase,
            "message": self.message,
            "chunk_total": self.chunk_total,
            "chunk_index": self.chunk_index,
            "chunk_id": self.chunk_id,
            "error": self.error,
            "error_code": self.error_code,
            "error_category": self.error_category,
            "error_retryable": self.error_retryable,
            "error_next_step": self.error_next_step,
            "error_source": self.error_source,
            "error_http_status": self.error_http_status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "main_pages": self.main_pages,
            "reference_pages": self.reference_pages,
            "owner_user_id": self.owner_user_id,
            "owner_username": self.owner_username,
            "original_filename": self.original_filename,
            "translate_mode": self.translate_mode,
            "parallel_max_workers": self.parallel_max_workers,
            "backend": self.backend,
            "tail_fallback": self.tail_fallback,
            "pages_per_chunk": self.pages_per_chunk,
            "overlap_pages": self.overlap_pages,
            "max_chunks": self.max_chunks,
            "use_custom_api": self.use_custom_api,
            "duration_seconds": self.duration_seconds,
            "run_started_at": self.run_started_at,
        }

    @classmethod
    def from_status_dict(cls, raw: dict[str, Any], fallback_work_dir: Path) -> JobRecord:
        status = str(raw.get("status") or "queued")
        if status not in VALID_JOB_STATUSES:
            status = "queued"
        return cls(
            job_id=str(raw.get("job_id") or fallback_work_dir.name),
            work_dir=fallback_work_dir.resolve(),
            status=status,  # type: ignore[arg-type]
            phase=str(raw.get("phase") or "queued"),
            message=str(raw.get("message") or ""),
            chunk_total=raw.get("chunk_total"),
            chunk_index=raw.get("chunk_index"),
            chunk_id=raw.get("chunk_id"),
            error=raw.get("error"),
            error_code=raw.get("error_code"),
            error_category=raw.get("error_category"),
            error_retryable=raw.get("error_retryable"),
            error_next_step=raw.get("error_next_step"),
            error_source=raw.get("error_source"),
            error_http_status=raw.get("error_http_status"),
            created_at=raw.get("created_at") or _utc_now_iso(),
            updated_at=raw.get("updated_at") or _utc_now_iso(),
            main_pages=raw.get("main_pages"),
            reference_pages=raw.get("reference_pages"),
            owner_user_id=raw.get("owner_user_id"),
            owner_username=raw.get("owner_username"),
            original_filename=raw.get("original_filename"),
            translate_mode=raw.get("translate_mode") or "serial",
            parallel_max_workers=_coerce_optional_int(raw.get("parallel_max_workers")),
            backend=raw.get("backend"),
            tail_fallback=_coerce_bool(raw.get("tail_fallback")),
            pages_per_chunk=_coerce_int(raw.get("pages_per_chunk"), 3),
            overlap_pages=_coerce_int(raw.get("overlap_pages"), 1),
            max_chunks=_coerce_optional_int(raw.get("max_chunks")),
            use_custom_api=_coerce_bool(raw.get("use_custom_api")),
            duration_seconds=raw.get("duration_seconds"),
            run_started_at=raw.get("run_started_at"),
            recovered_from_disk=True,
        )


class JobRegistry:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root.resolve()
        self.data_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._last_hydration_report: dict[str, Any] = self._empty_hydration_report()

    def _empty_hydration_report(self) -> dict[str, Any]:
        return {
            "schema_version": JOB_HYDRATION_REPORT_SCHEMA_VERSION,
            "data_root": str(self.data_root),
            "scanned_dir_count": 0,
            "restored_count": 0,
            "missing_status_count": 0,
            "invalid_json_count": 0,
            "job_id_mismatch_count": 0,
            "restored_job_ids": [],
            "active_recovered_job_ids": [],
            "recovered_status_counts": {},
            "auto_resume_policy": "off",
            "auto_resume_enabled": False,
            "auto_resume_max_jobs": 0,
            "auto_resume_attempted_job_ids": [],
            "auto_resume_started_job_ids": [],
            "auto_resume_skipped": [],
            "auto_resume_error_count": 0,
            "warnings": [],
        }

    def _job_dir(self, job_id: str) -> Path | None:
        raw = str(job_id or "").strip()
        if not raw or Path(raw).name != raw:
            return None
        path = (self.data_root / raw).resolve()
        try:
            path.relative_to(self.data_root)
        except ValueError:
            return None
        return path

    def _status_path(self, job_id: str) -> Path:
        work = self._job_dir(job_id)
        if work is None:
            raise ValueError("Invalid job_id path segment")
        return work / "web_status.json"

    def _persist(self, rec: JobRecord) -> None:
        p = self._status_path(rec.job_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(rec.to_status_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, p)

    def create_job(
        self,
        *,
        owner_user_id: int | None = None,
        owner_username: str | None = None,
        original_filename: str | None = None,
        translate_mode: str = "serial",
        parallel_max_workers: int | None = None,
        backend: str | None = None,
        tail_fallback: bool = False,
        pages_per_chunk: int = 3,
        overlap_pages: int = 1,
        max_chunks: int | None = None,
        use_custom_api: bool = False,
    ) -> JobRecord:
        job_id = uuid.uuid4().hex[:12]
        work = self.data_root / job_id
        work.mkdir(parents=True, exist_ok=False)
        rec = JobRecord(
            job_id=job_id,
            work_dir=work,
            owner_user_id=owner_user_id,
            owner_username=owner_username,
            original_filename=original_filename,
            translate_mode=translate_mode,
            parallel_max_workers=parallel_max_workers,
            backend=backend,
            tail_fallback=tail_fallback,
            pages_per_chunk=pages_per_chunk,
            overlap_pages=overlap_pages,
            max_chunks=max_chunks,
            use_custom_api=use_custom_api,
        )
        with self._lock:
            self._jobs[job_id] = rec
        self._persist(rec)
        return rec

    def hydrate_from_disk(self) -> None:
        """服务重启后从 web_status.json 恢复内存中的任务状态。"""
        report = self._empty_hydration_report()
        if not self.data_root.is_dir():
            report["warnings"].append("data_root_missing")
            self._last_hydration_report = report
            return
        with self._lock:
            for sub in self.data_root.iterdir():
                if not sub.is_dir():
                    continue
                report["scanned_dir_count"] += 1
                st = sub / "web_status.json"
                if not st.is_file():
                    report["missing_status_count"] += 1
                    continue
                try:
                    raw = json.loads(st.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    report["invalid_json_count"] += 1
                    continue
                raw_job_id = str(raw.get("job_id") or "").strip()
                if raw_job_id and raw_job_id != sub.name:
                    report["job_id_mismatch_count"] += 1
                    raw = dict(raw)
                    raw["job_id"] = sub.name
                rec = JobRecord.from_status_dict(raw, fallback_work_dir=sub)
                self._jobs[rec.job_id] = rec
                report["restored_count"] += 1
                report["restored_job_ids"].append(rec.job_id)
                report["recovered_status_counts"][rec.status] = (
                    int(report["recovered_status_counts"].get(rec.status) or 0) + 1
                )
                if rec.status in ("queued", "running"):
                    report["active_recovered_job_ids"].append(rec.job_id)
            self._last_hydration_report = report

    def hydration_report(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._last_hydration_report, ensure_ascii=False))

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_records(self) -> list[JobRecord]:
        with self._lock:
            return sorted(
                self._jobs.values(),
                key=lambda rec: rec.updated_at or rec.created_at,
                reverse=True,
            )

    def remove_job(self, job_id: str) -> bool:
        import shutil

        work = self._job_dir(job_id)
        if work is None:
            return False
        with self._lock:
            self._jobs.pop(job_id, None)
        if work.is_dir():
            shutil.rmtree(work, ignore_errors=True)
            return True
        return False

    def active_job_ids(self) -> set[str]:
        with self._lock:
            return {jid for jid, rec in self._jobs.items() if rec.status in ("queued", "running")}

    def storage_drift(self, indexed_job_ids: set[str]) -> dict[str, Any]:
        if self.data_root.is_dir():
            dir_job_ids = {sub.name for sub in self.data_root.iterdir() if sub.is_dir()}
        else:
            dir_job_ids = set()
        active = self.active_job_ids()
        indexed = {str(jid) for jid in indexed_job_ids if str(jid)}
        missing_work_dir = sorted(indexed - dir_job_ids)
        unindexed_work_dir = sorted(dir_job_ids - indexed)
        return {
            "indexed_job_count": len(indexed),
            "work_dir_count": len(dir_job_ids),
            "missing_work_dir_count": len(missing_work_dir),
            "unindexed_work_dir_count": len(unindexed_work_dir),
            "active_job_count": len(active),
            "missing_work_dir_job_ids": missing_work_dir,
            "unindexed_work_dir_job_ids": unindexed_work_dir,
            "active_job_ids": sorted(active),
        }

    def update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return
            for k, v in kwargs.items():
                if hasattr(rec, k):
                    setattr(rec, k, v)
            rec.touch()
            self._persist(rec)

    @staticmethod
    def _file_size(path: Path) -> int:
        try:
            if not path.is_file():
                return 0
            return path.stat().st_size
        except OSError:
            return 0

    @staticmethod
    def _repair_publish_summary(path: Path) -> tuple[dict[str, Any], str | None]:
        if not path.is_file():
            return {}, None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}, "repair_publish_report_invalid"
        summary = raw.get("summary")
        if not isinstance(summary, dict):
            return {}, "repair_publish_summary_missing"
        return summary, None

    @staticmethod
    def _repair_rollback_summary(path: Path) -> tuple[dict[str, Any], str | None]:
        if not path.is_file():
            return {}, None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}, "repair_rollback_report_invalid"
        summary = raw.get("summary")
        if not isinstance(summary, dict):
            return {}, "repair_rollback_summary_missing"
        return summary, None

    @staticmethod
    def _repair_formal_replace_summary(path: Path) -> tuple[dict[str, Any], str | None]:
        if not path.is_file():
            return {}, None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}, "repair_formal_replace_report_invalid"
        summary = raw.get("summary")
        if not isinstance(summary, dict):
            return {}, "repair_formal_replace_summary_missing"
        return summary, None

    @staticmethod
    def _repair_formal_rollback_summary(path: Path) -> tuple[dict[str, Any], str | None]:
        if not path.is_file():
            return {}, None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}, "repair_formal_rollback_report_invalid"
        summary = raw.get("summary")
        if not isinstance(summary, dict):
            return {}, "repair_formal_rollback_summary_missing"
        return summary, None

    @staticmethod
    def _repair_patch_review_summary(path: Path) -> tuple[dict[str, Any], str | None]:
        if not path.is_file():
            return {}, None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}, "repair_patch_review_invalid"
        summary = raw.get("summary")
        if not isinstance(summary, dict):
            return {}, "repair_patch_review_summary_missing"
        return summary, None

    @staticmethod
    def _repair_effectiveness_summary(path: Path) -> tuple[dict[str, Any], str | None]:
        if not path.is_file():
            return {}, None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}, "repair_effectiveness_report_invalid"
        summary = raw.get("summary")
        if not isinstance(summary, dict):
            return {}, "repair_effectiveness_summary_missing"
        return summary, None

    @staticmethod
    def _table_merged_cell_review_summary(path: Path) -> tuple[dict[str, Any], str | None]:
        if not path.is_file():
            return {}, None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}, "table_merged_cell_review_invalid"
        summary = raw.get("summary")
        if not isinstance(summary, dict):
            return {}, "table_merged_cell_review_summary_missing"
        return summary, None

    @staticmethod
    def _table_structure_publish_summary(path: Path) -> tuple[dict[str, Any], str | None]:
        if not path.is_file():
            return {}, None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}, "table_structure_publish_invalid"
        summary = raw.get("summary")
        if not isinstance(summary, dict):
            return {}, "table_structure_publish_summary_missing"
        return summary, None

    @staticmethod
    def _as_int(value: Any) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        try:
            return int(str(value or "0").strip() or "0")
        except ValueError:
            return 0

    @staticmethod
    def _as_float(value: Any) -> float:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value or "0").strip() or "0")
        except ValueError:
            return 0.0

    def artifact_fields_for_record(self, rec: JobRecord) -> dict[str, Any]:
        input_pdf = rec.work_dir / "input.pdf"
        output_dir = rec.work_dir / "output"
        translated_md = output_dir / "translated_full.md"
        translated_pdf = output_dir / "translated_full.pdf"
        bilingual_html = output_dir / "bilingual.html"
        repair_publish_json = output_dir / "repair_publish.json"
        repair_publish_md = output_dir / "repair_publish.md"
        repair_rollback_json = output_dir / "repair_rollback.json"
        repair_rollback_md = output_dir / "repair_rollback.md"
        repair_patch_review_json = output_dir / "repair_patch_review.json"
        repair_patch_review_md = output_dir / "repair_patch_review.md"
        repair_effectiveness_json = output_dir / "repair_effectiveness.json"
        repair_effectiveness_md = output_dir / "repair_effectiveness.md"
        repair_published_full = output_dir / "published_full.md"
        repair_rollback_full = output_dir / "rollback_full.md"
        repair_formal_replace_json = output_dir / "repair_formal_replace.json"
        repair_formal_replace_md = output_dir / "repair_formal_replace.md"
        repair_formal_rollback_json = output_dir / "repair_formal_rollback.json"
        repair_formal_rollback_md = output_dir / "repair_formal_rollback.md"
        repair_formal_full = output_dir / "formal_full.md"
        repair_formal_backup_full = output_dir / "formal_full.before_repair.md"
        repair_formal_active_before_rollback_full = output_dir / "formal_full.repair_applied.md"
        table_reconstruction_json = output_dir / "table_reconstruction.json"
        table_merged_cell_review_json = output_dir / "table_merged_cell_review.json"
        table_merged_cell_review_md = output_dir / "table_merged_cell_review.md"
        table_structure_publish_json = output_dir / "table_structure_publish.json"
        table_structure_publish_md = output_dir / "table_structure_publish.md"
        table_reconstruction_confirmed_json = output_dir / "table_reconstruction_confirmed.json"
        input_bytes = self._file_size(input_pdf)
        translated_bytes = self._file_size(translated_md)
        pdf_bytes = self._file_size(translated_pdf)
        html_bytes = self._file_size(bilingual_html)
        repair_publish_json_bytes = self._file_size(repair_publish_json)
        repair_publish_md_bytes = self._file_size(repair_publish_md)
        repair_rollback_json_bytes = self._file_size(repair_rollback_json)
        repair_rollback_md_bytes = self._file_size(repair_rollback_md)
        repair_patch_review_json_bytes = self._file_size(repair_patch_review_json)
        repair_patch_review_md_bytes = self._file_size(repair_patch_review_md)
        repair_effectiveness_json_bytes = self._file_size(repair_effectiveness_json)
        repair_effectiveness_md_bytes = self._file_size(repair_effectiveness_md)
        repair_published_full_bytes = self._file_size(repair_published_full)
        repair_rollback_full_bytes = self._file_size(repair_rollback_full)
        repair_formal_replace_json_bytes = self._file_size(repair_formal_replace_json)
        repair_formal_replace_md_bytes = self._file_size(repair_formal_replace_md)
        repair_formal_rollback_json_bytes = self._file_size(repair_formal_rollback_json)
        repair_formal_rollback_md_bytes = self._file_size(repair_formal_rollback_md)
        repair_formal_full_bytes = self._file_size(repair_formal_full)
        repair_formal_backup_full_bytes = self._file_size(repair_formal_backup_full)
        repair_formal_active_before_rollback_full_bytes = self._file_size(
            repair_formal_active_before_rollback_full
        )
        table_merged_cell_review_json_bytes = self._file_size(table_merged_cell_review_json)
        table_merged_cell_review_md_bytes = self._file_size(table_merged_cell_review_md)
        table_structure_publish_json_bytes = self._file_size(table_structure_publish_json)
        table_structure_publish_md_bytes = self._file_size(table_structure_publish_md)
        table_reconstruction_confirmed_bytes = self._file_size(table_reconstruction_confirmed_json)
        repair_summary, repair_warning = self._repair_publish_summary(repair_publish_json)
        repair_rollback_summary, repair_rollback_warning = self._repair_rollback_summary(repair_rollback_json)
        repair_formal_replace_summary, repair_formal_replace_warning = self._repair_formal_replace_summary(
            repair_formal_replace_json
        )
        repair_formal_rollback_summary, repair_formal_rollback_warning = self._repair_formal_rollback_summary(
            repair_formal_rollback_json
        )
        patch_review_summary, patch_review_warning = self._repair_patch_review_summary(repair_patch_review_json)
        repair_effectiveness_summary, repair_effectiveness_warning = self._repair_effectiveness_summary(
            repair_effectiveness_json
        )
        table_review_summary, table_review_warning = self._table_merged_cell_review_summary(
            table_merged_cell_review_json
        )
        table_publish_summary, table_publish_warning = self._table_structure_publish_summary(
            table_structure_publish_json
        )

        warnings: list[str] = []
        if not rec.work_dir.is_dir():
            warnings.append("work_dir_missing")
        if input_bytes <= 0:
            warnings.append("input_pdf_missing")
        if rec.status == "done" and translated_bytes <= 0:
            warnings.append("translated_md_missing_for_done")
        if rec.status == "done" and translated_bytes > 0 and pdf_bytes <= 0:
            warnings.append("translated_pdf_missing_for_done")
        if rec.status == "cancelled" and translated_bytes <= 0:
            warnings.append("translated_md_missing_for_cancelled")
        if rec.status == "done" and translated_bytes > 0 and repair_publish_json_bytes <= 0:
            warnings.append("repair_publish_report_missing_for_done")
        if repair_warning:
            warnings.append(repair_warning)
        if repair_rollback_warning:
            warnings.append(repair_rollback_warning)
        if repair_formal_replace_warning:
            warnings.append(repair_formal_replace_warning)
        if repair_formal_rollback_warning:
            warnings.append(repair_formal_rollback_warning)
        if patch_review_warning:
            warnings.append(patch_review_warning)
        if repair_effectiveness_warning:
            warnings.append(repair_effectiveness_warning)
        if table_review_warning:
            warnings.append(table_review_warning)
        if table_publish_warning:
            warnings.append(table_publish_warning)
        if (
            rec.status == "done"
            and table_reconstruction_json.is_file()
            and table_merged_cell_review_json_bytes <= 0
            and table_merged_cell_review_md_bytes <= 0
        ):
            warnings.append("table_merged_cell_review_missing_for_done")

        repair_publish_confirmed = bool(repair_summary.get("confirmed"))
        repair_publish_published = bool(repair_summary.get("published"))
        repair_publish_status = str(repair_summary.get("publish_status") or "")
        repair_publish_open_issue_count = self._as_int(repair_summary.get("open_merge_issue_count"))
        repair_publish_rollback_available = bool(repair_summary.get("rollback_available"))
        repair_rollback_available = bool(repair_rollback_summary.get("rollback_available"))
        repair_rollback_confirmed = bool(repair_rollback_summary.get("confirmed"))
        repair_rollback_applied = bool(repair_rollback_summary.get("rollback_applied"))
        repair_rollback_status = str(repair_rollback_summary.get("rollback_status") or "")
        repair_rollback_matches_original = bool(repair_rollback_summary.get("rollback_matches_original"))
        repair_formal_replace_available = bool(repair_formal_replace_summary.get("replace_available"))
        repair_formal_replace_confirmed = bool(repair_formal_replace_summary.get("confirmed"))
        repair_formal_replace_replaced = bool(repair_formal_replace_summary.get("replaced"))
        repair_formal_replace_status = str(repair_formal_replace_summary.get("replace_status") or "")
        repair_formal_replace_matches_published = bool(
            repair_formal_replace_summary.get("formal_matches_published")
        )
        repair_formal_replace_rollback_available = bool(
            repair_formal_replace_summary.get("rollback_available")
        )
        repair_formal_rollback_available = bool(repair_formal_rollback_summary.get("rollback_available"))
        repair_formal_rollback_confirmed = bool(repair_formal_rollback_summary.get("confirmed"))
        repair_formal_rollback_applied = bool(repair_formal_rollback_summary.get("rollback_applied"))
        repair_formal_rollback_status = str(repair_formal_rollback_summary.get("rollback_status") or "")
        repair_formal_rollback_matches_backup = bool(
            repair_formal_rollback_summary.get("formal_matches_backup")
        )
        repair_patch_review_count = self._as_int(patch_review_summary.get("patch_count"))
        repair_patch_review_required_count = self._as_int(patch_review_summary.get("review_required_count"))
        repair_patch_review_blocking_count = self._as_int(patch_review_summary.get("publish_blocking_count"))
        repair_patch_review_human_reviewed_count = self._as_int(patch_review_summary.get("human_reviewed_count"))
        repair_patch_review_effective_safe_count = self._as_int(patch_review_summary.get("effective_safe_count"))
        repair_effectiveness_status = str(repair_effectiveness_summary.get("status") or "")
        repair_effectiveness_before_issue_count = self._as_int(
            repair_effectiveness_summary.get("before_issue_count")
        )
        repair_effectiveness_after_issue_count = self._as_int(
            repair_effectiveness_summary.get("after_issue_count")
        )
        repair_effectiveness_issue_delta = self._as_int(repair_effectiveness_summary.get("issue_delta"))
        repair_effectiveness_issue_reduction_rate = self._as_float(
            repair_effectiveness_summary.get("issue_reduction_rate")
        )
        repair_effectiveness_resolved_issue_count = self._as_int(
            repair_effectiveness_summary.get("resolved_issue_count")
        )
        repair_effectiveness_persisted_issue_count = self._as_int(
            repair_effectiveness_summary.get("persisted_issue_count")
        )
        repair_effectiveness_new_issue_count = self._as_int(
            repair_effectiveness_summary.get("new_issue_count")
        )
        repair_effectiveness_improved_chunk_count = self._as_int(
            repair_effectiveness_summary.get("improved_chunk_count")
        )
        repair_effectiveness_regressed_chunk_count = self._as_int(
            repair_effectiveness_summary.get("regressed_chunk_count")
        )
        table_merged_cell_review_count = self._as_int(table_review_summary.get("candidate_review_count"))
        table_merged_cell_review_required_count = self._as_int(table_review_summary.get("review_required_count"))
        table_merged_cell_review_pending_count = self._as_int(table_review_summary.get("pending_review_count"))
        table_merged_cell_review_visual_supported_count = self._as_int(
            table_review_summary.get("visual_supported_count")
        )
        table_merged_cell_review_human_reviewed_count = self._as_int(
            table_review_summary.get("human_reviewed_count")
        )
        table_merged_cell_review_human_confirmed_count = self._as_int(
            table_review_summary.get("human_confirmed_count")
        )
        table_merged_cell_review_rejected_count = self._as_int(table_review_summary.get("rejected_count"))
        table_merged_cell_review_needs_revision_count = self._as_int(
            table_review_summary.get("needs_revision_count")
        )
        table_structure_publish_confirmed = bool(table_publish_summary.get("confirmed"))
        table_structure_publish_published = bool(table_publish_summary.get("published"))
        table_structure_publish_status = str(table_publish_summary.get("publish_status") or "")
        table_structure_publish_blocking_count = self._as_int(table_publish_summary.get("blocking_review_count"))
        table_structure_publish_applied_count = self._as_int(table_publish_summary.get("applied_confirmed_count"))
        table_structure_patch_count = self._as_int(table_publish_summary.get("structure_patch_count"))
        table_structure_patch_applied_count = self._as_int(
            table_publish_summary.get("structure_patch_applied_count")
        )
        table_structure_patch_covered_cell_count = self._as_int(
            table_publish_summary.get("structure_patch_covered_cell_count")
        )
        table_structure_publish_rollback_available = bool(table_publish_summary.get("rollback_available"))
        if repair_publish_open_issue_count > 0:
            warnings.append("repair_publish_open_issues")
        if repair_patch_review_blocking_count > 0:
            warnings.append("repair_patch_review_blocking_items")
        if table_merged_cell_review_required_count > 0:
            warnings.append("table_merged_cell_review_required_items")
        if table_structure_publish_blocking_count > 0:
            warnings.append("table_structure_publish_blocking_items")
        if table_structure_publish_confirmed and not table_structure_publish_published:
            warnings.append("table_structure_publish_requested_not_published")
        if table_structure_publish_published and table_reconstruction_confirmed_bytes <= 0:
            warnings.append("table_reconstruction_confirmed_missing")
        if repair_publish_confirmed and not repair_publish_published:
            warnings.append("repair_publish_requested_not_published")
        if repair_publish_published and repair_published_full_bytes <= 0:
            warnings.append("repair_published_full_missing")
        if repair_rollback_confirmed and not repair_rollback_applied:
            warnings.append("repair_rollback_requested_not_applied")
        if repair_rollback_applied and repair_rollback_full_bytes <= 0:
            warnings.append("repair_rollback_full_missing")
        if repair_formal_replace_confirmed and not repair_formal_replace_replaced:
            warnings.append("repair_formal_replace_requested_not_replaced")
        if repair_formal_replace_replaced and repair_formal_full_bytes <= 0:
            warnings.append("repair_formal_full_missing")
        if repair_formal_replace_rollback_available and repair_formal_backup_full_bytes <= 0:
            warnings.append("repair_formal_backup_full_missing")
        if repair_formal_rollback_confirmed and not repair_formal_rollback_applied:
            warnings.append("repair_formal_rollback_requested_not_applied")
        if repair_formal_rollback_applied and repair_formal_full_bytes <= 0:
            warnings.append("repair_formal_full_missing_after_rollback")

        severe = {
            "work_dir_missing",
            "input_pdf_missing",
            "translated_md_missing_for_done",
        }
        artifact_consistent = not any(item in severe for item in warnings)

        if not artifact_consistent:
            consistency_status = "inconsistent"
        elif rec.status == "done":
            consistency_status = "ready"
        elif rec.status == "cancelled":
            consistency_status = "partial" if translated_bytes > 0 else "no_output"
        elif rec.status == "error":
            consistency_status = "partial" if translated_bytes > 0 else "no_output"
        else:
            consistency_status = "pending"

        return {
            "artifact_consistent": artifact_consistent,
            "artifact_consistency_status": consistency_status,
            "artifact_warnings": warnings,
            "input_pdf_ready": input_bytes > 0,
            "input_pdf_bytes": input_bytes,
            "output_dir_ready": output_dir.is_dir(),
            "partial_output_ready": translated_bytes > 0,
            "partial_output_bytes": translated_bytes,
            "translated_pdf_ready": pdf_bytes > 0,
            "translated_pdf_bytes": pdf_bytes,
            "bilingual_html_ready": html_bytes > 0,
            "bilingual_html_bytes": html_bytes,
            "repair_publish_report_ready": repair_publish_json_bytes > 0 or repair_publish_md_bytes > 0,
            "repair_publish_report_bytes": max(repair_publish_json_bytes, repair_publish_md_bytes),
            "repair_rollback_report_ready": repair_rollback_json_bytes > 0 or repair_rollback_md_bytes > 0,
            "repair_rollback_report_bytes": max(repair_rollback_json_bytes, repair_rollback_md_bytes),
            "repair_patch_review_ready": repair_patch_review_json_bytes > 0 or repair_patch_review_md_bytes > 0,
            "repair_patch_review_bytes": max(repair_patch_review_json_bytes, repair_patch_review_md_bytes),
            "repair_patch_review_count": repair_patch_review_count,
            "repair_patch_review_required_count": repair_patch_review_required_count,
            "repair_patch_review_blocking_count": repair_patch_review_blocking_count,
            "repair_patch_review_human_reviewed_count": repair_patch_review_human_reviewed_count,
            "repair_patch_review_effective_safe_count": repair_patch_review_effective_safe_count,
            "repair_effectiveness_report_ready": (
                repair_effectiveness_json_bytes > 0 or repair_effectiveness_md_bytes > 0
            ),
            "repair_effectiveness_report_bytes": max(
                repair_effectiveness_json_bytes,
                repair_effectiveness_md_bytes,
            ),
            "repair_effectiveness_status": repair_effectiveness_status,
            "repair_effectiveness_before_issue_count": repair_effectiveness_before_issue_count,
            "repair_effectiveness_after_issue_count": repair_effectiveness_after_issue_count,
            "repair_effectiveness_issue_delta": repair_effectiveness_issue_delta,
            "repair_effectiveness_issue_reduction_rate": repair_effectiveness_issue_reduction_rate,
            "repair_effectiveness_resolved_issue_count": repair_effectiveness_resolved_issue_count,
            "repair_effectiveness_persisted_issue_count": repair_effectiveness_persisted_issue_count,
            "repair_effectiveness_new_issue_count": repair_effectiveness_new_issue_count,
            "repair_effectiveness_improved_chunk_count": repair_effectiveness_improved_chunk_count,
            "repair_effectiveness_regressed_chunk_count": repair_effectiveness_regressed_chunk_count,
            "table_merged_cell_review_ready": (
                table_merged_cell_review_json_bytes > 0 or table_merged_cell_review_md_bytes > 0
            ),
            "table_merged_cell_review_bytes": max(
                table_merged_cell_review_json_bytes,
                table_merged_cell_review_md_bytes,
            ),
            "table_merged_cell_review_count": table_merged_cell_review_count,
            "table_merged_cell_review_required_count": table_merged_cell_review_required_count,
            "table_merged_cell_review_pending_count": table_merged_cell_review_pending_count,
            "table_merged_cell_review_visual_supported_count": table_merged_cell_review_visual_supported_count,
            "table_merged_cell_review_human_reviewed_count": table_merged_cell_review_human_reviewed_count,
            "table_merged_cell_review_human_confirmed_count": table_merged_cell_review_human_confirmed_count,
            "table_merged_cell_review_rejected_count": table_merged_cell_review_rejected_count,
            "table_merged_cell_review_needs_revision_count": table_merged_cell_review_needs_revision_count,
            "table_structure_publish_ready": (
                table_structure_publish_json_bytes > 0 or table_structure_publish_md_bytes > 0
            ),
            "table_structure_publish_bytes": max(
                table_structure_publish_json_bytes,
                table_structure_publish_md_bytes,
            ),
            "table_structure_publish_confirmed": table_structure_publish_confirmed,
            "table_structure_publish_published": table_structure_publish_published,
            "table_structure_publish_status": table_structure_publish_status,
            "table_structure_publish_blocking_count": table_structure_publish_blocking_count,
            "table_structure_publish_applied_count": table_structure_publish_applied_count,
            "table_structure_patch_count": table_structure_patch_count,
            "table_structure_patch_applied_count": table_structure_patch_applied_count,
            "table_structure_patch_covered_cell_count": table_structure_patch_covered_cell_count,
            "table_structure_publish_rollback_available": table_structure_publish_rollback_available,
            "table_reconstruction_confirmed_ready": table_reconstruction_confirmed_bytes > 0,
            "table_reconstruction_confirmed_bytes": table_reconstruction_confirmed_bytes,
            "repair_publish_confirmed": repair_publish_confirmed,
            "repair_publish_published": repair_publish_published,
            "repair_publish_status": repair_publish_status,
            "repair_publish_open_issue_count": repair_publish_open_issue_count,
            "repair_publish_rollback_available": repair_publish_rollback_available,
            "repair_rollback_available": repair_rollback_available,
            "repair_rollback_confirmed": repair_rollback_confirmed,
            "repair_rollback_applied": repair_rollback_applied,
            "repair_rollback_status": repair_rollback_status,
            "repair_rollback_matches_original": repair_rollback_matches_original,
            "repair_published_full_ready": repair_published_full_bytes > 0,
            "repair_published_full_bytes": repair_published_full_bytes,
            "repair_rollback_full_ready": repair_rollback_full_bytes > 0,
            "repair_rollback_full_bytes": repair_rollback_full_bytes,
            "repair_formal_replace_report_ready": (
                repair_formal_replace_json_bytes > 0 or repair_formal_replace_md_bytes > 0
            ),
            "repair_formal_replace_report_bytes": max(
                repair_formal_replace_json_bytes,
                repair_formal_replace_md_bytes,
            ),
            "repair_formal_replace_available": repair_formal_replace_available,
            "repair_formal_replace_confirmed": repair_formal_replace_confirmed,
            "repair_formal_replace_replaced": repair_formal_replace_replaced,
            "repair_formal_replace_status": repair_formal_replace_status,
            "repair_formal_replace_matches_published": repair_formal_replace_matches_published,
            "repair_formal_replace_rollback_available": repair_formal_replace_rollback_available,
            "repair_formal_rollback_report_ready": (
                repair_formal_rollback_json_bytes > 0 or repair_formal_rollback_md_bytes > 0
            ),
            "repair_formal_rollback_report_bytes": max(
                repair_formal_rollback_json_bytes,
                repair_formal_rollback_md_bytes,
            ),
            "repair_formal_rollback_available": repair_formal_rollback_available,
            "repair_formal_rollback_confirmed": repair_formal_rollback_confirmed,
            "repair_formal_rollback_applied": repair_formal_rollback_applied,
            "repair_formal_rollback_status": repair_formal_rollback_status,
            "repair_formal_rollback_matches_backup": repair_formal_rollback_matches_backup,
            "repair_formal_full_ready": repair_formal_full_bytes > 0,
            "repair_formal_full_bytes": repair_formal_full_bytes,
            "repair_formal_backup_full_ready": repair_formal_backup_full_bytes > 0,
            "repair_formal_backup_full_bytes": repair_formal_backup_full_bytes,
            "repair_formal_active_before_rollback_full_ready": (
                repair_formal_active_before_rollback_full_bytes > 0
            ),
            "repair_formal_active_before_rollback_full_bytes": repair_formal_active_before_rollback_full_bytes,
            "bundle_zip_ready": rec.status in ("done", "cancelled") and translated_bytes > 0,
        }

    @staticmethod
    def _load_json_file(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    def pipeline_state_fields_for_record(self, rec: JobRecord) -> dict[str, Any]:
        out_dir = rec.work_dir / "output"
        state_path = out_dir / "state.json"
        manifest_path = out_dir / "chunks_manifest.json"
        chunk_dir = out_dir / "chunks"
        warnings: list[str] = []
        completed: list[str] = []
        manifest_chunk_ids: list[str] = []
        state_available = False
        state_status = "missing"

        if state_path.is_file():
            try:
                raw_state = self._load_json_file(state_path)
                completed = [
                    str(item)
                    for item in (raw_state.get("completed") or [])
                    if str(item or "").strip()
                ]
                state_available = True
            except (json.JSONDecodeError, OSError, AttributeError):
                warnings.append("pipeline_state_invalid")
                state_status = "invalid"
        elif out_dir.exists():
            warnings.append("pipeline_state_missing")

        if manifest_path.is_file():
            try:
                raw_manifest = self._load_json_file(manifest_path)
                if isinstance(raw_manifest, list):
                    manifest_chunk_ids = [
                        str(item.get("chunk_id"))
                        for item in raw_manifest
                        if isinstance(item, dict) and str(item.get("chunk_id") or "").strip()
                    ]
            except (json.JSONDecodeError, OSError):
                warnings.append("chunks_manifest_invalid")

        completed_set = set(completed)
        total = len(manifest_chunk_ids) if manifest_chunk_ids else None
        completed_count = len(completed_set)
        pending_chunk_ids = [
            chunk_id for chunk_id in manifest_chunk_ids if chunk_id not in completed_set
        ]
        missing_chunk_files = [
            chunk_id
            for chunk_id in completed
            if not (chunk_dir / f"{chunk_id}.md").is_file()
        ]
        if missing_chunk_files:
            warnings.append("completed_chunk_file_missing")

        if state_available:
            if total is not None:
                if total > 0 and not pending_chunk_ids and completed_count >= total:
                    state_status = "complete"
                elif completed_count > 0:
                    state_status = "partial"
                else:
                    state_status = "empty"
            elif completed_count > 0:
                state_status = "partial_unknown_total"
            else:
                state_status = "empty_unknown_total"

        if rec.status == "done" and state_status not in {"complete", "partial_unknown_total"}:
            warnings.append("runtime_done_pipeline_incomplete")
        if rec.status in {"queued", "running"} and rec.recovered_from_disk:
            warnings.append("recovered_active_without_worker")
        if rec.status in {"queued", "running"} and state_status == "complete":
            warnings.append("runtime_active_pipeline_complete")

        ratio: float | None = None
        if total and total > 0:
            ratio = round(min(completed_count, total) / total, 4)

        resume_ready = completed_count > 0 and state_status not in {"complete", "invalid"}
        if "recovered_active_without_worker" in warnings:
            recovery_status = "needs_manual_resume_or_cancel"
        elif resume_ready:
            recovery_status = "resume_available"
        elif state_status == "complete":
            recovery_status = "complete"
        elif state_status in {"missing", "invalid"}:
            recovery_status = "not_ready"
        else:
            recovery_status = "not_started"

        return {
            "pipeline_state_available": state_available,
            "pipeline_state_status": state_status,
            "pipeline_state_path": str(state_path),
            "pipeline_chunks_manifest_available": bool(manifest_chunk_ids),
            "pipeline_completed_chunk_count": completed_count,
            "pipeline_chunk_total": total,
            "pipeline_pending_chunk_count": len(pending_chunk_ids) if total is not None else None,
            "pipeline_completion_ratio": ratio,
            "pipeline_completed_chunk_ids": sorted(completed_set),
            "pipeline_pending_chunk_ids": pending_chunk_ids,
            "pipeline_missing_chunk_files": missing_chunk_files,
            "pipeline_resume_ready": resume_ready,
            "job_recovered_from_disk": rec.recovered_from_disk,
            "job_recovery_status": recovery_status,
            "job_diagnostic_warnings": warnings,
        }

    @staticmethod
    def _runtime_options_for_record(rec: JobRecord) -> dict[str, Any]:
        return {
            "backend": rec.backend,
            "tail_fallback": rec.tail_fallback,
            "pages_per_chunk": rec.pages_per_chunk,
            "overlap_pages": rec.overlap_pages,
            "max_chunks": rec.max_chunks,
            "translate_mode": rec.translate_mode,
            "parallel_max_workers": rec.parallel_max_workers,
            "use_custom_api": rec.use_custom_api,
        }

    def recovery_fields_for_record(
        self,
        rec: JobRecord,
        pipeline_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        pipeline = pipeline_fields or self.pipeline_state_fields_for_record(rec)
        blockers: list[str] = []
        input_ready = (rec.work_dir / "input.pdf").is_file()
        cancel_requested = is_cancel_requested(rec.work_dir)

        if not rec.recovered_from_disk:
            blockers.append("not_recovered_from_disk")
        if rec.status not in ("queued", "running"):
            blockers.append("not_active_status")
        if not input_ready:
            blockers.append("input_pdf_missing")
        if cancel_requested:
            blockers.append("cancel_requested")
        if rec.use_custom_api:
            blockers.append("custom_api_not_resumable")
        if not str(rec.backend or "").strip():
            blockers.append("backend_missing")
        if rec.pages_per_chunk < 1 or rec.pages_per_chunk > 3:
            blockers.append("pages_per_chunk_invalid")
        if rec.overlap_pages < 0 or rec.overlap_pages >= rec.pages_per_chunk:
            blockers.append("overlap_pages_invalid")

        active_recovered = rec.recovered_from_disk and rec.status in ("queued", "running")
        requeue_candidate = active_recovered and not blockers
        if requeue_candidate:
            action = "requeueable"
        elif active_recovered:
            action = "manual_required"
        else:
            action = "not_applicable"

        return {
            "job_requeue_candidate": requeue_candidate,
            "job_requeue_action": action,
            "job_requeue_blockers": blockers,
            "job_requeue_input_ready": input_ready,
            "job_requeue_cancel_requested": cancel_requested,
            "job_requeue_runtime_options": self._runtime_options_for_record(rec),
            "job_requeue_pipeline_state_status": pipeline.get("pipeline_state_status"),
            "job_requeue_pipeline_resume_ready": pipeline.get("pipeline_resume_ready"),
        }

    def requeue_recovered_jobs(
        self,
        *,
        policy: str,
        max_jobs: int,
        cfg: AppConfig,
        starter: Callable[..., None] | None = None,
        audit: bool = True,
        job_ids: set[str] | None = None,
        audit_user_id: int | None = None,
        audit_username: str | None = "system",
    ) -> dict[str, Any]:
        normalized_policy = (policy or "off").strip().lower()
        if normalized_policy not in VALID_AUTO_RESUME_POLICIES:
            normalized_policy = "off"
        max_jobs = max(0, min(int(max_jobs), 32))
        report: dict[str, Any] = {
            "auto_resume_policy": normalized_policy,
            "auto_resume_enabled": normalized_policy != "off",
            "auto_resume_max_jobs": max_jobs,
            "auto_resume_attempted_job_ids": [],
            "auto_resume_started_job_ids": [],
            "auto_resume_skipped": [],
            "auto_resume_error_count": 0,
        }

        with self._lock:
            records = sorted(
                self._jobs.values(),
                key=lambda rec: rec.updated_at or rec.created_at,
            )

        wanted = {str(item) for item in job_ids} if job_ids is not None else None
        starter_fn = starter or start_job_thread

        def add_skip(rec: JobRecord, reason: str, fields: dict[str, Any] | None = None) -> None:
            report["auto_resume_skipped"].append(
                {
                    "job_id": rec.job_id,
                    "reason": reason,
                    "status": rec.status,
                    "phase": rec.phase,
                    "blockers": list((fields or {}).get("job_requeue_blockers") or []),
                    "pipeline_state_status": (fields or {}).get("job_requeue_pipeline_state_status"),
                    "pipeline_resume_ready": (fields or {}).get("job_requeue_pipeline_resume_ready"),
                    "runtime_options": self._runtime_options_for_record(rec),
                }
            )

        def log_requeued(rec: JobRecord, fields: dict[str, Any]) -> None:
            if not audit:
                return
            try:
                from pdf_translate.server import database as srv_db

                srv_db.log_audit(
                    action="job_recovery_requeued",
                    ip=None,
                    user_id=audit_user_id,
                    username=audit_username,
                    job_id=rec.job_id,
                    detail={
                        "previous_status": rec.status,
                        "previous_phase": rec.phase,
                        "pipeline_state_status": fields.get("job_requeue_pipeline_state_status"),
                        "pipeline_resume_ready": fields.get("job_requeue_pipeline_resume_ready"),
                        "runtime_options": self._runtime_options_for_record(rec),
                        "work_dir": str(rec.work_dir.resolve()),
                        "policy": normalized_policy,
                    },
                )
            except Exception:
                pass

        for rec in records:
            if wanted is not None and rec.job_id not in wanted:
                continue
            if not (rec.recovered_from_disk and rec.status in ("queued", "running")):
                continue
            fields = self.recovery_fields_for_record(rec)
            if normalized_policy == "off":
                add_skip(rec, "policy_off", fields)
                continue
            if len(report["auto_resume_started_job_ids"]) >= max_jobs:
                add_skip(rec, "max_jobs_reached", fields)
                continue
            if not bool(fields.get("job_requeue_candidate")):
                blockers = list(fields.get("job_requeue_blockers") or [])
                add_skip(rec, blockers[0] if blockers else "not_requeueable", fields)
                continue

            report["auto_resume_attempted_job_ids"].append(rec.job_id)
            try:
                log_requeued(rec, fields)
                self.update(
                    rec.job_id,
                    status="queued",
                    phase="queued",
                    message="服务重启后重新入队，准备断点续跑…",
                    recovered_from_disk=False,
                    run_started_at=None,
                    duration_seconds=None,
                    error=None,
                    error_code=None,
                    error_category=None,
                    error_retryable=None,
                    error_next_step=None,
                    error_source=None,
                    error_http_status=None,
                )
                starter_fn(
                    self,
                    rec.job_id,
                    tail_fallback=rec.tail_fallback,
                    pages_per_chunk=rec.pages_per_chunk,
                    overlap_pages=rec.overlap_pages,
                    backend=rec.backend,
                    max_chunks=rec.max_chunks,
                    cfg=cfg,
                )
                report["auto_resume_started_job_ids"].append(rec.job_id)
            except Exception as exc:
                report["auto_resume_error_count"] += 1
                info = make_error_info(
                    "PIPELINE_ERROR",
                    detail=f"Failed to start recovered job thread: {exc}",
                    source="server:requeue_recovered_jobs",
                    exception=exc,
                )
                self.update(
                    rec.job_id,
                    status="error",
                    phase="error",
                    message="恢复重新入队启动失败",
                    recovered_from_disk=False,
                    error=str(exc),
                    error_code=info.code,
                    error_category=info.category,
                    error_retryable=info.retryable,
                    error_next_step=info.next_step,
                    error_source=info.source,
                    error_http_status=info.http_status,
                )
                add_skip(rec, f"start_failed:{type(exc).__name__}", fields)

        with self._lock:
            merged = dict(self._last_hydration_report)
            merged.update(report)
            self._last_hydration_report = merged
        return report

    def diagnostic_summary_for_record(self, rec: JobRecord) -> dict[str, Any]:
        data = asdict(rec.to_public())
        data.update(
            {
                "owner_user_id": rec.owner_user_id,
                "owner_username": rec.owner_username,
                "original_filename": rec.original_filename,
            }
        )
        data.update(self.artifact_fields_for_record(rec))
        pipeline_fields = self.pipeline_state_fields_for_record(rec)
        data.update(pipeline_fields)
        data.update(self.recovery_fields_for_record(rec, pipeline_fields))
        return data

    def status_fields_for_job(self, job_id: str) -> dict[str, Any]:
        rec = self.get(job_id)
        if not rec:
            return {
                "status_available": False,
                "artifact_consistent": False,
                "artifact_consistency_status": "missing_status",
                "artifact_warnings": ["status_snapshot_missing"],
                "input_pdf_ready": False,
                "input_pdf_bytes": 0,
                "output_dir_ready": False,
                "partial_output_ready": False,
                "partial_output_bytes": 0,
                "translated_pdf_ready": False,
                "translated_pdf_bytes": 0,
                "bilingual_html_ready": False,
                "bilingual_html_bytes": 0,
                "repair_publish_report_ready": False,
                "repair_publish_report_bytes": 0,
                "repair_rollback_report_ready": False,
                "repair_rollback_report_bytes": 0,
                "repair_patch_review_ready": False,
                "repair_patch_review_bytes": 0,
                "repair_patch_review_count": 0,
                "repair_patch_review_required_count": 0,
                "repair_patch_review_blocking_count": 0,
                "repair_patch_review_human_reviewed_count": 0,
                "repair_patch_review_effective_safe_count": 0,
                "repair_effectiveness_report_ready": False,
                "repair_effectiveness_report_bytes": 0,
                "repair_effectiveness_status": "",
                "repair_effectiveness_before_issue_count": 0,
                "repair_effectiveness_after_issue_count": 0,
                "repair_effectiveness_issue_delta": 0,
                "repair_effectiveness_issue_reduction_rate": 0.0,
                "repair_effectiveness_resolved_issue_count": 0,
                "repair_effectiveness_persisted_issue_count": 0,
                "repair_effectiveness_new_issue_count": 0,
                "repair_effectiveness_improved_chunk_count": 0,
                "repair_effectiveness_regressed_chunk_count": 0,
                "table_merged_cell_review_ready": False,
                "table_merged_cell_review_bytes": 0,
                "table_merged_cell_review_count": 0,
                "table_merged_cell_review_required_count": 0,
                "table_merged_cell_review_pending_count": 0,
                "table_merged_cell_review_visual_supported_count": 0,
                "table_merged_cell_review_human_reviewed_count": 0,
                "table_merged_cell_review_human_confirmed_count": 0,
                "table_merged_cell_review_rejected_count": 0,
                "table_merged_cell_review_needs_revision_count": 0,
                "table_structure_publish_ready": False,
                "table_structure_publish_bytes": 0,
                "table_structure_publish_confirmed": False,
                "table_structure_publish_published": False,
                "table_structure_publish_status": "",
                "table_structure_publish_blocking_count": 0,
                "table_structure_publish_applied_count": 0,
                "table_structure_patch_count": 0,
                "table_structure_patch_applied_count": 0,
                "table_structure_patch_covered_cell_count": 0,
                "table_structure_publish_rollback_available": False,
                "table_reconstruction_confirmed_ready": False,
                "table_reconstruction_confirmed_bytes": 0,
                "repair_publish_confirmed": False,
                "repair_publish_published": False,
                "repair_publish_status": "",
                "repair_publish_open_issue_count": 0,
                "repair_publish_rollback_available": False,
                "repair_rollback_available": False,
                "repair_rollback_confirmed": False,
                "repair_rollback_applied": False,
                "repair_rollback_status": "",
                "repair_rollback_matches_original": False,
                "repair_published_full_ready": False,
                "repair_published_full_bytes": 0,
                "repair_rollback_full_ready": False,
                "repair_rollback_full_bytes": 0,
                "repair_formal_replace_report_ready": False,
                "repair_formal_replace_report_bytes": 0,
                "repair_formal_replace_available": False,
                "repair_formal_replace_confirmed": False,
                "repair_formal_replace_replaced": False,
                "repair_formal_replace_status": "",
                "repair_formal_replace_matches_published": False,
                "repair_formal_replace_rollback_available": False,
                "repair_formal_rollback_report_ready": False,
                "repair_formal_rollback_report_bytes": 0,
                "repair_formal_rollback_available": False,
                "repair_formal_rollback_confirmed": False,
                "repair_formal_rollback_applied": False,
                "repair_formal_rollback_status": "",
                "repair_formal_rollback_matches_backup": False,
                "repair_formal_full_ready": False,
                "repair_formal_full_bytes": 0,
                "repair_formal_backup_full_ready": False,
                "repair_formal_backup_full_bytes": 0,
                "repair_formal_active_before_rollback_full_ready": False,
                "repair_formal_active_before_rollback_full_bytes": 0,
                "bundle_zip_ready": False,
            }
        pub = rec.to_public()
        fields = {
            "status_available": True,
            "status_schema_version": JOB_STATUS_SCHEMA_VERSION,
            "status": pub.status,
            "phase": pub.phase,
            "message": pub.message,
            "chunk_total": pub.chunk_total,
            "chunk_index": pub.chunk_index,
            "chunk_id": pub.chunk_id,
            "error": pub.error,
            "error_code": pub.error_code,
            "error_category": pub.error_category,
            "error_retryable": pub.error_retryable,
            "error_next_step": pub.error_next_step,
            "error_source": pub.error_source,
            "error_http_status": pub.error_http_status,
            "runtime_created_at": pub.created_at,
            "runtime_updated_at": pub.updated_at,
            "main_pages": pub.main_pages,
            "reference_pages": pub.reference_pages,
            "translate_mode": pub.translate_mode,
            "parallel_max_workers": pub.parallel_max_workers,
            "duration_seconds": pub.duration_seconds,
            "run_started_at": pub.run_started_at,
        }
        fields.update(self.artifact_fields_for_record(rec))
        fields.update(self.pipeline_state_fields_for_record(rec))
        return fields

    def merge_status_into_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        for row in rows:
            job_id = str(row.get("job_id") or "")
            out = dict(row)
            out.update(self.status_fields_for_job(job_id))
            merged.append(out)
        return merged

    def run_pipeline(
        self,
        job_id: str,
        *,
        tail_fallback: bool,
        pages_per_chunk: int,
        overlap_pages: int,
        backend: str | None,
        max_chunks: int | None,
        cfg: AppConfig,
    ) -> None:
        rec = self.get(job_id)
        if not rec:
            return

        t0 = time.perf_counter()
        tm_raw = (rec.translate_mode or "serial").strip().lower()
        if tm_raw == "premium":
            translate_mode = "serial"
            survey_override = True
        elif tm_raw == "parallel":
            translate_mode = "parallel"
            survey_override = False
        else:
            translate_mode = "serial"
            survey_override = False
        pw = rec.parallel_max_workers
        if pw is None:
            try:
                pw = int(os.getenv("PDF_TRANSLATE_PARALLEL_WORKERS", "4"))
            except ValueError:
                pw = 4
        pw = max(1, pw)

        def progress(ev: dict) -> None:
            evt = ev.get("event")
            if evt == "translate_start":
                total = int(ev.get("chunk_total") or 0)
                self.update(
                    job_id,
                    phase="translate",
                    chunk_total=total,
                    message=f"准备翻译，共 {total} 个块；首块请求模型可能需 1～5 分钟，进度条在整块完成后才前进。",
                )
            elif evt == "translate_chunk_start":
                ci = int(ev.get("chunk_index") or 0)
                ct = int(ev.get("chunk_total") or 0)
                chid = str(ev.get("chunk_id") or "")
                nchar = int(ev.get("approx_chars") or 0)
                self.update(
                    job_id,
                    chunk_total=ct,
                    chunk_id=chid,
                    message=f"正在请求模型：第 {ci}/{ct} 块 ({chid})，本块约 {nchar} 字符，请稍候…",
                )
            elif evt == "translate_chunk_skipped":
                self.update(
                    job_id,
                    chunk_index=int(ev.get("chunk_index") or 0),
                    chunk_total=int(ev.get("chunk_total") or 0),
                    chunk_id=str(ev.get("chunk_id") or ""),
                    message=f"跳过已完成块 {ev.get('chunk_id')}",
                )
            elif evt == "translate_chunk_done":
                self.update(
                    job_id,
                    chunk_index=int(ev.get("chunk_index") or 0),
                    chunk_total=int(ev.get("chunk_total") or 0),
                    chunk_id=str(ev.get("chunk_id") or ""),
                    message=f"已完成 {ev.get('chunk_index')}/{ev.get('chunk_total')} ({ev.get('chunk_id')})",
                )

        try:
            previous_status = rec.status
            previous_phase = rec.phase
            run_started_at = _utc_now_iso()
            self.update(
                job_id,
                status="running",
                phase="init",
                message="初始化工作目录…",
                run_started_at=run_started_at,
                duration_seconds=None,
                error=None,
                error_code=None,
                error_category=None,
                error_retryable=None,
                error_next_step=None,
                error_source=None,
                error_http_status=None,
            )
            try:
                from pdf_translate.server import database as srv_db

                rec_started = self.get(job_id)
                if rec_started:
                    srv_db.log_audit(
                        action="job_started",
                        ip=None,
                        user_id=rec_started.owner_user_id,
                        username=rec_started.owner_username,
                        job_id=job_id,
                        detail={
                            "previous_status": previous_status,
                            "previous_phase": previous_phase,
                            "status": "running",
                            "phase": "init",
                            "run_started_at": run_started_at,
                            "backend": backend,
                            "translate_mode": translate_mode,
                            "parallel_max_workers": pw,
                            "pages_per_chunk": pages_per_chunk,
                            "overlap_pages": overlap_pages,
                            "max_chunks": max_chunks,
                            "resume": True,
                            "recovered_from_disk": rec_started.recovered_from_disk,
                            "work_dir": str(rec_started.work_dir.resolve()),
                        },
                    )
            except Exception:
                pass
            init_workdir(rec.work_dir)

            inp = rec.work_dir / "input.pdf"
            if not inp.is_file():
                raise FileNotFoundError("input.pdf 缺失")

            self.update(job_id, phase="split", message="正在拆分正文与参考文献…")
            manifest = run_split(
                inp,
                rec.work_dir,
                use_tail_if_no_heading=tail_fallback,
            )
            self.update(
                job_id,
                main_pages=len(manifest.main_pages_0based),
                reference_pages=len(manifest.reference_pages_0based),
                message="拆分完成，开始翻译…",
            )

            if is_cancel_requested(rec.work_dir):
                raise JobCancelled()

            run_translate(
                rec.work_dir,
                cfg,
                backend=backend,
                pages_per_chunk=pages_per_chunk,
                overlap_pages=overlap_pages,
                resume=True,
                max_chunks=max_chunks,
                progress_callback=progress,
                translate_mode="parallel" if translate_mode == "parallel" else "serial",
                parallel_workers=pw,
                survey_override=survey_override,
            )

            self.update(job_id, phase="links", message="导出链接索引…")
            export_links(rec.work_dir)

            elapsed = round(time.perf_counter() - t0, 2)
            self.update(
                job_id,
                status="done",
                phase="done",
                message=f"全部完成，总用时 {elapsed} 秒",
                duration_seconds=elapsed,
                error=None,
                error_code=None,
                error_category=None,
                error_retryable=None,
                error_next_step=None,
                error_source=None,
                error_http_status=None,
            )
            try:
                from pdf_translate.server import database as srv_db

                rec2 = self.get(job_id)
                if rec2:
                    srv_db.log_job_finished(
                        job_id=job_id,
                        user_id=rec2.owner_user_id,
                        username=rec2.owner_username,
                        work_dir=rec2.work_dir,
                        ok=True,
                        status=rec2.status,
                        phase=rec2.phase,
                        duration_seconds=rec2.duration_seconds,
                        run_started_at=rec2.run_started_at,
                        status_updated_at=rec2.updated_at,
                        original_filename=rec2.original_filename,
                        translate_mode=rec2.translate_mode,
                        parallel_max_workers=pw,
                    )
            except Exception:
                pass
        except JobCancelled:
            elapsed = round(time.perf_counter() - t0, 2)
            info = make_error_info(
                "TASK_CANCELLED",
                detail="Task cancelled by user request.",
                source="server:run_pipeline",
            )
            self.update(
                job_id,
                status="cancelled",
                phase="cancelled",
                message=f"已按请求终止，已保留已译部分；总用时 {elapsed} 秒",
                duration_seconds=elapsed,
                error=None,
                error_code=info.code,
                error_category=info.category,
                error_retryable=info.retryable,
                error_next_step=info.next_step,
                error_source=info.source,
                error_http_status=info.http_status,
            )
            try:
                export_links(rec.work_dir)
            except Exception:
                pass
            try:
                from pdf_translate.server import database as srv_db

                rec2 = self.get(job_id)
                if rec2:
                    srv_db.log_job_finished(
                        job_id=job_id,
                        user_id=rec2.owner_user_id,
                        username=rec2.owner_username,
                        work_dir=rec2.work_dir,
                        ok=False,
                        err="cancelled",
                        status=rec2.status,
                        phase=rec2.phase,
                        duration_seconds=rec2.duration_seconds,
                        run_started_at=rec2.run_started_at,
                        status_updated_at=rec2.updated_at,
                        original_filename=rec2.original_filename,
                        translate_mode=rec2.translate_mode,
                        parallel_max_workers=pw,
                        error_code=rec2.error_code,
                        error_category=rec2.error_category,
                        error_retryable=rec2.error_retryable,
                        error_next_step=rec2.error_next_step,
                        error_source=rec2.error_source,
                        error_http_status=rec2.error_http_status,
                    )
            except Exception:
                pass
        except Exception as e:
            elapsed = round(time.perf_counter() - t0, 2)
            info = error_info_from_exception(e, source="server:run_pipeline")
            self.update(
                job_id,
                status="error",
                phase="error",
                error=str(e),
                message="失败",
                duration_seconds=elapsed,
                error_code=info.code,
                error_category=info.category,
                error_retryable=info.retryable,
                error_next_step=info.next_step,
                error_source=info.source,
                error_http_status=info.http_status,
            )
            try:
                from pdf_translate.server import database as srv_db

                rec2 = self.get(job_id)
                if rec2:
                    srv_db.log_job_finished(
                        job_id=job_id,
                        user_id=rec2.owner_user_id,
                        username=rec2.owner_username,
                        work_dir=rec2.work_dir,
                        ok=False,
                        err=str(e),
                        status=rec2.status,
                        phase=rec2.phase,
                        duration_seconds=rec2.duration_seconds,
                        run_started_at=rec2.run_started_at,
                        status_updated_at=rec2.updated_at,
                        original_filename=rec2.original_filename,
                        translate_mode=rec2.translate_mode,
                        parallel_max_workers=pw,
                        error_code=rec2.error_code,
                        error_category=rec2.error_category,
                        error_retryable=rec2.error_retryable,
                        error_next_step=rec2.error_next_step,
                        error_source=rec2.error_source,
                        error_http_status=rec2.error_http_status,
                    )
            except Exception:
                pass


def start_job_thread(
    registry: JobRegistry,
    job_id: str,
    *,
    tail_fallback: bool,
    pages_per_chunk: int,
    overlap_pages: int,
    backend: str | None,
    max_chunks: int | None,
    cfg: AppConfig,
) -> None:
    t = threading.Thread(
        target=registry.run_pipeline,
        kwargs={
            "job_id": job_id,
            "tail_fallback": tail_fallback,
            "pages_per_chunk": pages_per_chunk,
            "overlap_pages": overlap_pages,
            "backend": backend,
            "max_chunks": max_chunks,
            "cfg": cfg,
        },
        daemon=True,
    )
    t.start()


def zip_job_outputs(
    work_dir: Path,
    *,
    original_filename: str | None = None,
    complete: bool = True,
) -> tuple[bytes, str]:
    import io
    import zipfile

    from pdf_translate.export_filename import suggest_zip_bundle_name
    from pdf_translate.zip_bundle import iter_bundle_files, map_bundle_arcname

    root = work_dir.resolve()
    zip_name = suggest_zip_bundle_name(
        original_filename=original_filename,
        work_dir=work_dir,
        complete=complete,
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in iter_bundle_files(root):
            rel = f.relative_to(root).as_posix()
            arc = map_bundle_arcname(rel.replace("\\", "/"))
            zf.write(f, arcname=arc)
    return buf.getvalue(), zip_name
