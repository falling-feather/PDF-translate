from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime, timezone
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

BACKEND_UI_LABELS: dict[str, str] = {
    "echo": "echo（联调/测试）",
    "openai": "OpenAI 兼容（含官方 OpenAI 等）",
    "deepseek": "DeepSeek",
    "ollama": "Ollama（本地）",
    "deepl": "DeepL",
    "hybrid": "hybrid（初稿 + 润色）",
}


def client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for") or request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    if request.client:
        return request.client.host or ""
    return ""


def _build_runtime_cfg_for_custom_api(
    *,
    cfg,
    backend: str,
    api_key: str | None,
    api_base_url: str | None,
    api_model: str | None,
):
    out = replace(cfg)
    b = backend.lower().strip()
    key = (api_key or "").strip()
    base = (api_base_url or "").strip()
    model = (api_model or "").strip()

    if b == "openai":
        if not key:
            raise ValueError("openai 需要填写 API Key")
        out.openai_api_key = key
        if base:
            out.openai_base_url = base
        if model:
            out.openai_model = model
        return out
    if b == "deepseek":
        if not key:
            raise ValueError("deepseek 需要填写 API Key")
        out.deepseek_api_key = key
        if base:
            out.deepseek_base_url = base
        if model:
            out.deepseek_model = model
        return out
    if b == "deepl":
        if not key:
            raise ValueError("deepl 需要填写 API Key")
        out.deepl_api_key = key
        if base:
            out.deepl_api_url = base
        return out
    if b == "ollama":
        if base:
            out.ollama_base_url = base
        if model:
            out.ollama_model = model
        return out
    raise ValueError("API翻译目前仅支持 openai / deepseek / ollama / deepl")


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
        enabled_ordered = [b for b in ALL_BACKENDS if b in eb]
        return {
            "enabled": enabled_ordered,
            "default_backend": cfg.default_translator,
            "labels": {k: BACKEND_UI_LABELS.get(k, k) for k in enabled_ordered},
        }

    @api.get("/user/jobs")
    def my_jobs(p: Principal = Depends(bearer_principal), limit: int = 100) -> dict:
        rows = database.list_jobs_for_user(p.user_id, limit=limit)
        favorites = database.list_favorite_jobs_for_user(p.user_id, limit=limit)
        return {
            "jobs": rows,
            "favorites": favorites,
            "favorite_max": database.MAX_JOB_FAVORITES_PER_USER,
        }

    @api.post("/user/jobs/cleanup-stale")
    def cleanup_stale_jobs(
        p: Principal = Depends(bearer_principal),
        hours: int = 24,
    ) -> dict:
        if hours < 1 or hours > 168:
            raise HTTPException(400, "hours 应在 1–168 之间")
        stale = database.list_stale_job_ids_for_user(p.user_id, hours=hours)
        deleted: list[str] = []
        for jid in stale:
            rec = app_registry.get(jid)
            if rec and rec.status in ("queued", "running"):
                continue
            database.delete_job_meta_row(jid)
            app_registry.remove_job(jid)
            deleted.append(jid)
        return {"deleted": deleted, "hours": hours}

    # 收藏接口单独路径，避免与 GET /user/jobs/favorites 等产生「POST 命中仅 GET 路由 → 405」的歧义
    @api.post("/user/favorites/{job_id}")
    def favorite_job(job_id: str, p: Principal = Depends(bearer_principal)) -> dict:
        try:
            database.add_job_favorite(p.user_id, job_id)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return {"ok": True}

    @api.delete("/user/favorites/{job_id}")
    def unfavorite_job(job_id: str, p: Principal = Depends(bearer_principal)) -> dict:
        try:
            database.remove_job_favorite(p.user_id, job_id)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        ts = datetime.now(timezone.utc).isoformat()
        rec = app_registry.get(job_id)
        if rec:
            app_registry.update(job_id, created_at=ts)
        return {"ok": True}

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
        use_custom_api: bool = Form(False),
        custom_backend: str | None = Form(None),
        custom_api_key: str | None = Form(None),
        custom_api_base_url: str | None = Form(None),
        custom_api_model: str | None = Form(None),
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
        runtime_cfg = cfg
        if use_custom_api:
            be = (custom_backend or "").strip().lower()
            if not be:
                raise HTTPException(400, "已启用 API翻译，请选择 API 后端")
            if be not in ALL_BACKENDS:
                raise HTTPException(400, f"不支持的 API 后端：{be}")
            try:
                runtime_cfg = _build_runtime_cfg_for_custom_api(
                    cfg=cfg,
                    backend=be,
                    api_key=custom_api_key,
                    api_base_url=custom_api_base_url,
                    api_model=custom_api_model,
                )
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
        else:
            try:
                be = settings_service.assert_backend_allowed(backend, cfg.default_translator)
            except ValueError as e:
                raise HTTPException(400, str(e)) from e

        try:
            build_translator(be, runtime_cfg)
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
            detail={"filename": file.filename, "backend": be, "use_custom_api": bool(use_custom_api)},
        )

        start_job_thread(
            app_registry,
            rec.job_id,
            tail_fallback=tail_fallback,
            pages_per_chunk=pages_per_chunk,
            overlap_pages=overlap_pages,
            backend=be,
            max_chunks=max_n,
            cfg=runtime_cfg,
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

    @api.get("/jobs/{job_id}/download/input.pdf")
    def download_input_pdf(job_id: str, p: Principal = Depends(bearer_principal)) -> FileResponse:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        path = rec.work_dir / "input.pdf"
        if not path.is_file() or path.stat().st_size == 0:
            raise HTTPException(404, "未找到上传原文件")
        disp_name = rec.original_filename or "input.pdf"
        ascii_fallback = "input.pdf"
        cd = (
            f'attachment; filename="{ascii_fallback}"; '
            f"filename*=UTF-8''{quote(disp_name)}"
        )
        return FileResponse(
            path,
            media_type="application/pdf",
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
