from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pdf_translate.config import AppConfig
from pdf_translate.pipeline import export_links, init_workdir, run_split, run_translate
from pdf_translate.pipeline_cancel import JobCancelled, is_cancel_requested


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    created_at: str
    updated_at: str
    main_pages: int | None = None
    reference_pages: int | None = None
    translate_mode: str | None = None
    parallel_max_workers: int | None = None
    duration_seconds: float | None = None
    run_started_at: str | None = None


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
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    main_pages: int | None = None
    reference_pages: int | None = None
    owner_user_id: int | None = None
    owner_username: str | None = None
    original_filename: str | None = None
    translate_mode: str = "serial"
    parallel_max_workers: int | None = None
    duration_seconds: float | None = None
    run_started_at: str | None = None

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
            created_at=self.created_at,
            updated_at=self.updated_at,
            main_pages=self.main_pages,
            reference_pages=self.reference_pages,
            translate_mode=self.translate_mode,
            parallel_max_workers=self.parallel_max_workers,
            duration_seconds=self.duration_seconds,
            run_started_at=self.run_started_at,
        )


class JobRegistry:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root.resolve()
        self.data_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}

    def _status_path(self, job_id: str) -> Path:
        return self.data_root / job_id / "web_status.json"

    def _persist(self, rec: JobRecord) -> None:
        p = self._status_path(rec.job_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        d = {
            "job_id": rec.job_id,
            "work_dir": str(rec.work_dir),
            "status": rec.status,
            "phase": rec.phase,
            "message": rec.message,
            "chunk_total": rec.chunk_total,
            "chunk_index": rec.chunk_index,
            "chunk_id": rec.chunk_id,
            "error": rec.error,
            "created_at": rec.created_at,
            "updated_at": rec.updated_at,
            "main_pages": rec.main_pages,
            "reference_pages": rec.reference_pages,
            "owner_user_id": rec.owner_user_id,
            "owner_username": rec.owner_username,
            "original_filename": rec.original_filename,
            "translate_mode": rec.translate_mode,
            "parallel_max_workers": rec.parallel_max_workers,
            "duration_seconds": rec.duration_seconds,
            "run_started_at": rec.run_started_at,
        }
        p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

    def create_job(
        self,
        *,
        owner_user_id: int | None = None,
        owner_username: str | None = None,
        original_filename: str | None = None,
        translate_mode: str = "serial",
        parallel_max_workers: int | None = None,
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
        )
        with self._lock:
            self._jobs[job_id] = rec
        self._persist(rec)
        return rec

    def hydrate_from_disk(self) -> None:
        """服务重启后从 web_status.json 恢复内存中的任务状态。"""
        if not self.data_root.is_dir():
            return
        with self._lock:
            for sub in self.data_root.iterdir():
                if not sub.is_dir():
                    continue
                st = sub / "web_status.json"
                if not st.is_file():
                    continue
                try:
                    raw = json.loads(st.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                jid = raw.get("job_id") or sub.name
                rec = JobRecord(
                    job_id=jid,
                    work_dir=sub,
                    status=raw.get("status", "queued"),
                    phase=raw.get("phase", "queued"),
                    message=raw.get("message", ""),
                    chunk_total=raw.get("chunk_total"),
                    chunk_index=raw.get("chunk_index"),
                    chunk_id=raw.get("chunk_id"),
                    error=raw.get("error"),
                    created_at=raw.get("created_at") or _utc_now_iso(),
                    updated_at=raw.get("updated_at") or _utc_now_iso(),
                    main_pages=raw.get("main_pages"),
                    reference_pages=raw.get("reference_pages"),
                    owner_user_id=raw.get("owner_user_id"),
                    owner_username=raw.get("owner_username"),
                    original_filename=raw.get("original_filename"),
                    translate_mode=raw.get("translate_mode") or "serial",
                    parallel_max_workers=raw.get("parallel_max_workers"),
                    duration_seconds=raw.get("duration_seconds"),
                    run_started_at=raw.get("run_started_at"),
                )
                self._jobs[jid] = rec

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def remove_job(self, job_id: str) -> None:
        import shutil

        work = self.data_root / job_id
        with self._lock:
            self._jobs.pop(job_id, None)
        if work.is_dir():
            shutil.rmtree(work, ignore_errors=True)

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
        translate_mode = (rec.translate_mode or "serial").strip().lower()
        parallel_max_workers = rec.parallel_max_workers

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
            self.update(
                job_id,
                status="running",
                phase="init",
                message="初始化工作目录…",
                run_started_at=_utc_now_iso(),
                duration_seconds=None,
            )
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

            pw = parallel_max_workers
            if pw is None:
                try:
                    pw = int(os.getenv("PDF_TRANSLATE_PARALLEL_WORKERS", "4"))
                except ValueError:
                    pw = 4
            pw = max(1, pw)

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
                    )
            except Exception:
                pass
        except JobCancelled:
            elapsed = round(time.perf_counter() - t0, 2)
            self.update(
                job_id,
                status="cancelled",
                phase="cancelled",
                message=f"已按请求终止，已保留已译部分；总用时 {elapsed} 秒",
                duration_seconds=elapsed,
                error=None,
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
                    )
            except Exception:
                pass
        except Exception as e:
            elapsed = round(time.perf_counter() - t0, 2)
            self.update(
                job_id,
                status="error",
                phase="error",
                error=str(e),
                message="失败",
                duration_seconds=elapsed,
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
