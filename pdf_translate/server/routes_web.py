from __future__ import annotations

import os
from urllib.parse import quote

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response

from pdf_translate.export_filename import suggest_md_download_name, suggest_zip_bundle_name
from pdf_translate.pipeline_cancel import cancel_flag_path

from pdf_translate.server.auth_deps import Principal, bearer_principal, mint_token, require_admin
from pdf_translate.server import database
from pdf_translate.server.jobs import JobRecord, JobRegistry, start_job_thread, zip_job_outputs
from pdf_translate.server import settings_service
from pdf_translate.translators.factory import build_translator

ALL_BACKENDS = ["echo", "openai", "deepseek", "ollama", "deepl", "hybrid"]


def client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for") or request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    if request.client:
        return request.client.host or ""
    return ""


def register_web_routes(app_registry: JobRegistry) -> APIRouter:
    api = APIRouter(prefix="/api")

    @api.get("/health")
    def health() -> dict:
        return {"ok": True}

    auth = APIRouter(prefix="/auth", tags=["auth"])

    @auth.post("/login")
    def login(request: Request, username: str = Form(...), password: str = Form(...)) -> dict:
        ip = client_ip(request)
        u = database.verify_user(username, password)
        if not u:
            database.log_audit(
                action="login_fail",
                ip=ip,
                user_id=None,
                username=username.strip(),
                detail={"reason": "bad_credentials"},
            )
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        database.log_audit(
            action="login_success",
            ip=ip,
            user_id=u["id"],
            username=u["username"],
            detail={"role": u["role"]},
        )
        token = mint_token(user_id=u["id"], username=u["username"], role=u["role"])
        return {"access_token": token, "token_type": "bearer", "role": u["role"], "username": u["username"]}

    @auth.post("/register")
    def register(request: Request, username: str = Form(...), password: str = Form(...)) -> dict:
        if not database.registration_open():
            raise HTTPException(400, "管理员已关闭自助注册")
        reserved = os.getenv("PDF_TRANSLATE_ADMIN_USERNAME", "falling-feather").strip()
        if username.strip() == reserved:
            raise HTTPException(400, "该用户名为系统保留")
        if len(password) < 6:
            raise HTTPException(400, "密码至少 6 位")
        ip = client_ip(request)
        try:
            uid = database.create_user(username=username, password=password, role="user")
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        database.log_audit(
            action="register",
            ip=ip,
            user_id=uid,
            username=username.strip(),
            detail={},
        )
        u = database.get_user_by_id(uid)
        assert u
        token = mint_token(user_id=u["id"], username=u["username"], role=u["role"])
        return {"access_token": token, "token_type": "bearer", "role": u["role"], "username": u["username"]}

    @auth.get("/me")
    def me(p: Principal = Depends(bearer_principal)) -> dict:
        return {"user_id": p.user_id, "username": p.username, "role": p.role}

    api.include_router(auth)

    def _can_access_job(p: Principal, rec: JobRecord | None) -> bool:
        if not rec:
            return False
        if p.role == "admin":
            return True
        return rec.owner_user_id is not None and rec.owner_user_id == p.user_id

    @api.get("/user/backends")
    def user_backends(p: Principal = Depends(bearer_principal)) -> dict:
        _ = p
        eb = settings_service.enabled_backends()
        cfg = settings_service.effective_app_config()
        return {
            "enabled": [b for b in ALL_BACKENDS if b in eb],
            "default_backend": cfg.default_translator,
        }

    @api.get("/user/jobs")
    def my_jobs(p: Principal = Depends(bearer_principal), limit: int = 100) -> dict:
        rows = database.list_jobs_for_user(p.user_id, limit=limit)
        return {"jobs": rows}

    @api.post("/jobs")
    async def create_job(
        request: Request,
        p: Principal = Depends(bearer_principal),
        file: UploadFile = File(...),
        tail_fallback: bool = Form(False),
        pages_per_chunk: int = Form(3),
        overlap_pages: int = Form(1),
        backend: str | None = Form(None),
        max_chunks: str | None = Form(None),
        translate_mode: str = Form("serial"),
        parallel_max_workers: str | None = Form(None),
    ) -> dict:
        if p.role not in ("user", "admin"):
            raise HTTPException(403, "无权提交翻译任务")
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(400, "请上传 .pdf 文件")
        if pages_per_chunk < 1 or pages_per_chunk > 3:
            raise HTTPException(400, "pages_per_chunk 必须在 1–3 之间")
        if overlap_pages < 0 or overlap_pages >= pages_per_chunk:
            raise HTTPException(400, "overlap_pages 无效")

        cfg = settings_service.effective_app_config()
        try:
            be = settings_service.assert_backend_allowed(backend, cfg.default_translator)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

        try:
            build_translator(be, cfg)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

        max_n: int | None = None
        if max_chunks not in (None, ""):
            try:
                max_n = int(max_chunks)
            except ValueError:
                raise HTTPException(400, "max_chunks 必须为整数") from None

        tm = (translate_mode or "serial").strip().lower()
        if tm not in ("serial", "parallel"):
            raise HTTPException(400, "translate_mode 须为 serial 或 parallel")
        pwm: int | None = None
        if parallel_max_workers not in (None, ""):
            try:
                pwm = int(parallel_max_workers)
            except ValueError:
                raise HTTPException(400, "parallel_max_workers 须为整数") from None
            if pwm < 1 or pwm > 32:
                raise HTTPException(400, "parallel_max_workers 应在 1–32 之间")

        rec = app_registry.create_job(
            owner_user_id=p.user_id,
            owner_username=p.username,
            original_filename=file.filename or "upload.pdf",
            translate_mode=tm,
            parallel_max_workers=pwm,
        )
        dest = rec.work_dir / "input.pdf"
        content = await file.read()
        if len(content) > 120 * 1024 * 1024:
            raise HTTPException(400, "文件超过 120MB 上限")
        dest.write_bytes(content)

        database.insert_job_meta(rec.job_id, p.user_id, p.username, file.filename or "upload.pdf")
        database.log_audit(
            action="job_submit",
            ip=client_ip(request),
            user_id=p.user_id,
            username=p.username,
            job_id=rec.job_id,
            detail={"filename": file.filename, "backend": be},
        )

        start_job_thread(
            app_registry,
            rec.job_id,
            tail_fallback=tail_fallback,
            pages_per_chunk=pages_per_chunk,
            overlap_pages=overlap_pages,
            backend=be,
            max_chunks=max_n,
            cfg=cfg,
        )
        return {"job_id": rec.job_id}

    def _job_dict(rec: JobRecord) -> dict:
        pub = rec.to_public()
        partial = rec.work_dir / "output" / "translated_full.md"
        partial_ok = partial.is_file() and partial.stat().st_size > 0
        complete = rec.status == "done"
        suggested_name = suggest_md_download_name(
            original_filename=rec.original_filename,
            work_dir=rec.work_dir,
            complete=complete,
        )
        suggested_zip = suggest_zip_bundle_name(
            original_filename=rec.original_filename,
            work_dir=rec.work_dir,
            complete=complete,
        )
        d = {
            "job_id": pub.job_id,
            "status": pub.status,
            "phase": pub.phase,
            "message": pub.message,
            "chunk_total": pub.chunk_total,
            "chunk_index": pub.chunk_index,
            "chunk_id": pub.chunk_id,
            "error": pub.error,
            "created_at": pub.created_at,
            "updated_at": pub.updated_at,
            "main_pages": pub.main_pages,
            "reference_pages": pub.reference_pages,
            "owner_user_id": rec.owner_user_id,
            "owner_username": rec.owner_username,
            "original_filename": rec.original_filename,
            "partial_output_ready": partial_ok,
            "partial_output_bytes": partial.stat().st_size if partial.is_file() else 0,
            "translate_mode": rec.translate_mode,
            "parallel_max_workers": rec.parallel_max_workers,
            "duration_seconds": rec.duration_seconds,
            "run_started_at": rec.run_started_at,
            "suggested_download_filename": suggested_name,
            "suggested_zip_filename": suggested_zip,
        }
        return d

    @api.get("/jobs/{job_id}")
    def get_job(job_id: str, p: Principal = Depends(bearer_principal)) -> dict:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        return _job_dict(rec)

    @api.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, p: Principal = Depends(bearer_principal)) -> dict:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        if rec.status not in ("queued", "running"):
            raise HTTPException(400, "当前状态不可终止")
        cancel_flag_path(rec.work_dir).write_text("1", encoding="utf-8")
        app_registry.update(job_id, message="已收到终止请求，将在当前块结束后停止…")
        return {"ok": True}

    @api.get("/jobs/{job_id}/download/full.md")
    def download_full(job_id: str, p: Principal = Depends(bearer_principal)) -> FileResponse:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        path = rec.work_dir / "output" / "translated_full.md"
        if not path.is_file() or path.stat().st_size == 0:
            raise HTTPException(404, "译文尚未生成或仍为空")
        disp_name = suggest_md_download_name(
            original_filename=rec.original_filename,
            work_dir=rec.work_dir,
            complete=rec.status == "done",
        )
        ascii_fallback = "translated.md"
        cd = (
            f'attachment; filename="{ascii_fallback}"; '
            f"filename*=UTF-8''{quote(disp_name)}"
        )
        return FileResponse(
            path,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": cd},
        )

    @api.get("/jobs/{job_id}/download/bundle.zip")
    def download_zip(job_id: str, p: Principal = Depends(bearer_principal)) -> Response:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        if rec.status not in ("done", "cancelled"):
            raise HTTPException(409, "任务未完成或未终止，暂不可打包下载")
        md = rec.work_dir / "output" / "translated_full.md"
        if not md.is_file() or md.stat().st_size == 0:
            raise HTTPException(404, "尚无译文可打包")
        data, zip_disp = zip_job_outputs(
            rec.work_dir,
            original_filename=rec.original_filename,
            complete=rec.status == "done",
        )
        ascii_fb = "bundle.zip"
        cd = f'attachment; filename="{ascii_fb}"; filename*=UTF-8\'\'{quote(zip_disp)}'
        return Response(content=data, media_type="application/zip", headers={"Content-Disposition": cd})

    admin = APIRouter(prefix="/admin", tags=["admin"])

    @admin.get("/settings")
    def admin_get_settings(_: Principal = Depends(require_admin)) -> dict:
        snap = settings_service.admin_settings_snapshot()
        return snap

    @admin.put("/settings")
    def admin_put_settings(body: dict = Body(...), p: Principal = Depends(require_admin)) -> dict:
        settings_service.apply_admin_settings(body)
        database.log_audit(
            action="admin_settings_update",
            ip=None,
            user_id=p.user_id,
            username=p.username,
            detail={"keys": list(body.keys())},
        )
        return {"ok": True, "settings": settings_service.admin_settings_snapshot()}

    @admin.get("/audit")
    def admin_audit(
        _: Principal = Depends(require_admin),
        limit: int = 200,
        offset: int = 0,
    ) -> dict:
        return {"events": database.list_audit(limit=limit, offset=offset)}

    @admin.get("/users")
    def admin_users(_: Principal = Depends(require_admin)) -> dict:
        return {"users": database.list_users()}

    @admin.get("/jobs")
    def admin_jobs(_: Principal = Depends(require_admin), limit: int = 500) -> dict:
        return {"jobs": database.list_all_jobs(limit=limit)}

    @admin.get("/jobs/{job_id}/artifact", response_model=None)
    def admin_artifact(
        job_id: str,
        kind: str,
        _: Principal = Depends(require_admin),
    ) -> Response | FileResponse:
        rec = app_registry.get(job_id)
        if not rec:
            raise HTTPException(404, "任务不存在")
        root = rec.work_dir.resolve()
        if kind == "input":
            p = root / "input.pdf"
            if not p.is_file():
                raise HTTPException(404, "无上传文件")
            return FileResponse(p, filename="input.pdf", media_type="application/pdf")
        if kind == "output_md":
            p = root / "output" / "translated_full.md"
            if not p.is_file():
                raise HTTPException(404, "尚无译文")
            return FileResponse(p, filename="translated_full.md", media_type="text/markdown; charset=utf-8")
        if kind == "bundle_zip":
            if rec.status not in ("done", "cancelled"):
                raise HTTPException(409, "任务未完成或未终止")
            data, zip_disp = zip_job_outputs(
                rec.work_dir,
                original_filename=rec.original_filename,
                complete=rec.status == "done",
            )
            ascii_fb = "bundle.zip"
            cd = f'attachment; filename="{ascii_fb}"; filename*=UTF-8\'\'{quote(zip_disp)}'
            return Response(content=data, media_type="application/zip", headers={"Content-Disposition": cd})
        raise HTTPException(400, "kind 必须是 input / output_md / bundle_zip")

    api.include_router(admin)
    return api
