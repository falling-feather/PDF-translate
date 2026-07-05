from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response

from pdf_translate.error_codes import PdfTranslateError, error_info_from_exception, make_error_info
from pdf_translate.export_filename import suggest_md_download_name, suggest_zip_bundle_name
from pdf_translate.pipeline_cancel import cancel_flag_path

from pdf_translate.server.auth_deps import Principal, bearer_principal, mint_token, require_admin
from pdf_translate.server import database
from pdf_translate.server.jobs import JobRecord, JobRegistry, start_job_thread, zip_job_outputs
from pdf_translate.server.runtime_state import require_data_dir
from pdf_translate.server.security_preflight import build_security_preflight, max_upload_mb
from pdf_translate.server import settings_service
from pdf_translate.translators.factory import build_translator
from pdf_translate.translators.registry import (
    backend_catalog,
    backend_ids,
    backend_ui_labels,
    custom_api_backend_ids,
    get_backend_spec,
    normalize_backend_id,
)

PDF_UPLOAD_MAGIC = b"%PDF-"
UPLOAD_READ_CHUNK_BYTES = 1024 * 1024


def _is_deepseek_model_name(model_name: str | None) -> bool:
    return "deepseek" in str(model_name or "").strip().lower()


def client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for") or request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    if request.client:
        return request.client.host or ""
    return ""


async def _save_pdf_upload_streaming(
    file: UploadFile,
    dest: Path,
    *,
    max_bytes: int,
    max_mb: int,
) -> int:
    header = await file.read(len(PDF_UPLOAD_MAGIC))
    if header != PDF_UPLOAD_MAGIC:
        raise HTTPException(400, "上传文件不是有效 PDF")
    total = len(header)
    if total > max_bytes:
        raise HTTPException(400, f"文件超过 {max_mb}MB 上限")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dest.open("wb") as out:
            out.write(header)
            while True:
                chunk = await file.read(UPLOAD_READ_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(400, f"文件超过 {max_mb}MB 上限")
                out.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    return total


def _build_runtime_cfg_for_custom_api(
    *,
    cfg,
    backend: str,
    api_key: str | None,
    api_base_url: str | None,
    api_model: str | None,
):
    out = replace(cfg)
    b = normalize_backend_id(backend)
    key = (api_key or "").strip()
    base = (api_base_url or "").strip()
    model = (api_model or "").strip()
    spec = get_backend_spec(b)

    if not spec.supports_custom_api:
        raise PdfTranslateError(
            make_error_info(
                "CONFIG_INVALID_BACKEND",
                detail=f"{b} does not support per-job custom API settings.",
                source="api:create_job",
            )
        )

    if b in ("deepseek", "openai"):
        if not key:
            raise PdfTranslateError(
                make_error_info(
                    "CONFIG_MISSING_API_KEY",
                    detail=f"{b} custom API requires API Key.",
                    source="api:create_job",
                )
            )
        setattr(out, spec.api_key_attr or "", key)
        if base:
            setattr(out, spec.base_url_attr or "", base.rstrip("/"))
        if model:
            setattr(out, spec.model_attr or "", model)
        return out
    if b == "deepl":
        if not key:
            raise PdfTranslateError(
                make_error_info(
                    "CONFIG_MISSING_API_KEY",
                    detail="deepl custom API requires API Key.",
                    source="api:create_job",
                )
            )
        out.deepl_api_key = key
        if base:
            out.deepl_api_url = base
        return out
    if b == "ollama":
        if base:
            out.ollama_base_url = base.rstrip("/")
        if model:
            out.ollama_model = model
        return out
    raise PdfTranslateError(
        make_error_info(
            "CONFIG_INVALID_BACKEND",
            detail=f"Custom API translation does not support {backend}.",
            source="api:create_job",
        )
    )


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
        enabled_ordered = [b for b in backend_ids() if b in eb]
        labels = backend_ui_labels()
        return {
            "enabled": enabled_ordered,
            "default_backend": cfg.default_translator,
            "labels": {k: labels.get(k, k) for k in enabled_ordered},
            "catalog": backend_catalog(),
            "custom_api_backends": custom_api_backend_ids(),
        }

    @api.get("/user/jobs")
    def my_jobs(p: Principal = Depends(bearer_principal), limit: int = 100) -> dict:
        rows = app_registry.merge_status_into_rows(database.list_jobs_for_user(p.user_id, limit=limit))
        favorites = app_registry.merge_status_into_rows(database.list_favorite_jobs_for_user(p.user_id, limit=limit))
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
            try:
                be = normalize_backend_id(be)
            except ValueError as e:
                raise HTTPException(400, f"不支持的 API 后端：{be}") from e
            if be not in custom_api_backend_ids():
                raise HTTPException(400, f"不支持的 API 后端：{be}")
            try:
                runtime_cfg = _build_runtime_cfg_for_custom_api(
                    cfg=cfg,
                    backend=be,
                    api_key=custom_api_key,
                    api_base_url=custom_api_base_url,
                    api_model=custom_api_model,
                )
            except PdfTranslateError as e:
                raise HTTPException(400, detail=e.error_info.to_dict()) from e
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
        else:
            try:
                be = settings_service.assert_backend_allowed(backend, cfg.default_translator)
            except ValueError as e:
                info = error_info_from_exception(e, source="api:create_job")
                raise HTTPException(400, detail=info.to_dict()) from e

        try:
            build_translator(be, runtime_cfg)
        except PdfTranslateError as e:
            raise HTTPException(400, detail=e.error_info.to_dict()) from e
        except ValueError as e:
            info = error_info_from_exception(e, source="api:create_job")
            raise HTTPException(400, detail=info.to_dict()) from e

        max_n: int | None = None
        if max_chunks not in (None, ""):
            try:
                max_n = int(max_chunks)
            except ValueError:
                raise HTTPException(400, "max_chunks 必须为整数") from None

        tm = (translate_mode or "serial").strip().lower()
        if tm not in ("serial", "parallel", "premium"):
            raise HTTPException(400, "translate_mode 须为 serial、parallel 或 premium（精品翻译）")
        if tm == "premium":
            cfg_sf = settings_service.effective_app_config()
            if not (cfg_sf.siliconflow_api_key or "").strip():
                raise HTTPException(400, "精品翻译需要管理员在后台配置硅基流动 API Key")
            if not (cfg_sf.siliconflow_survey_model or "").strip():
                raise HTTPException(400, "精品翻译需要管理员在后台配置硅基流动「巡视模型」")
            if _is_deepseek_model_name(cfg_sf.siliconflow_survey_model):
                raise HTTPException(400, "精品翻译的硅基巡视模型不能填写 DeepSeek，请改为 Qwen/Kimi 等模型；DeepSeek 请走 DeepSeek API。")
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
        max_mb = max_upload_mb()
        try:
            uploaded_bytes = await _save_pdf_upload_streaming(
                file,
                dest,
                max_bytes=max_mb * 1024 * 1024,
                max_mb=max_mb,
            )
        except HTTPException:
            app_registry.remove_job(rec.job_id)
            raise

        database.insert_job_meta(rec.job_id, p.user_id, p.username, file.filename or "upload.pdf")
        database.log_audit(
            action="job_submit",
            ip=client_ip(request),
            user_id=p.user_id,
            username=p.username,
            job_id=rec.job_id,
            detail={
                "filename": file.filename,
                "backend": be,
                "use_custom_api": bool(use_custom_api),
                "upload_bytes": uploaded_bytes,
                "upload_limit_mb": max_mb,
            },
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
        d = app_registry.diagnostic_summary_for_record(rec)
        d["suggested_download_filename"] = suggested_name
        d["suggested_zip_filename"] = suggested_zip
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

    @api.get("/jobs/{job_id}/download/translated.pdf")
    def download_translated_pdf(job_id: str, p: Principal = Depends(bearer_principal)) -> FileResponse:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        path = rec.work_dir / "output" / "translated_full.pdf"
        if not path.is_file() or path.stat().st_size == 0:
            raise HTTPException(404, "PDF 译文尚未生成")
        ascii_fallback = "translated.pdf"
        disp_name = f"{Path(rec.original_filename or 'translated').stem}_translated.pdf"
        cd = f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{quote(disp_name)}'
        return FileResponse(
            path,
            media_type="application/pdf",
            headers={"Content-Disposition": cd},
        )

    @api.get("/jobs/{job_id}/download/repair-publish.md")
    def download_repair_publish(job_id: str, p: Principal = Depends(bearer_principal)) -> FileResponse:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        path = rec.work_dir / "output" / "repair_publish.md"
        if not path.is_file() or path.stat().st_size == 0:
            raise HTTPException(404, "局部修复发布确认报告尚未生成")
        ascii_fallback = "repair_publish.md"
        disp_name = f"{Path(rec.original_filename or 'translated').stem}_repair_publish.md"
        cd = f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{quote(disp_name)}'
        return FileResponse(
            path,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": cd},
        )

    @api.get("/jobs/{job_id}/download/published-full.md")
    def download_published_full(job_id: str, p: Principal = Depends(bearer_principal)) -> FileResponse:
        rec = app_registry.get(job_id)
        if not rec or not _can_access_job(p, rec):
            raise HTTPException(404, "任务不存在或无权访问")
        path = rec.work_dir / "output" / "published_full.md"
        if not path.is_file() or path.stat().st_size == 0:
            raise HTTPException(404, "人工确认修复发布稿尚未生成")
        ascii_fallback = "published_full.md"
        disp_name = f"{Path(rec.original_filename or 'translated').stem}_published_full.md"
        cd = f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{quote(disp_name)}'
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

    @admin.get("/security/preflight")
    def admin_security_preflight(_: Principal = Depends(require_admin)) -> dict:
        return build_security_preflight(
            require_data_dir(),
            app_registry.data_root,
        )

    @admin.put("/settings")
    def admin_put_settings(body: dict = Body(...), p: Principal = Depends(require_admin)) -> dict:
        try:
            settings_service.apply_admin_settings(body)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
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
        return {"jobs": app_registry.merge_status_into_rows(database.list_all_jobs(limit=limit))}

    @admin.get("/jobs/hydration-report")
    def admin_jobs_hydration_report(_: Principal = Depends(require_admin)) -> dict:
        return {
            "hydration": app_registry.hydration_report(),
            "drift": app_registry.storage_drift(set(database.list_all_job_ids())),
        }

    @admin.post("/jobs/reconcile")
    def admin_jobs_reconcile(
        body: dict | None = Body(None),
        p: Principal = Depends(require_admin),
    ) -> dict:
        apply_cleanup = bool((body or {}).get("apply"))
        indexed = set(database.list_all_job_ids())
        drift = app_registry.storage_drift(indexed)
        deleted_db_rows: list[str] = []
        deleted_work_dirs: list[str] = []
        skipped_active: list[str] = []

        if apply_cleanup:
            active = set(drift.get("active_job_ids") or [])
            for job_id in drift.get("missing_work_dir_job_ids") or []:
                if job_id in active:
                    skipped_active.append(job_id)
                    continue
                database.delete_job_meta_row(job_id)
                app_registry.remove_job(job_id)
                deleted_db_rows.append(job_id)
            for job_id in drift.get("unindexed_work_dir_job_ids") or []:
                if job_id in active:
                    skipped_active.append(job_id)
                    continue
                if app_registry.remove_job(job_id):
                    deleted_work_dirs.append(job_id)
            indexed = set(database.list_all_job_ids())
            drift = app_registry.storage_drift(indexed)
            database.log_audit(
                action="admin_jobs_reconcile",
                ip=None,
                user_id=p.user_id,
                username=p.username,
                detail={
                    "apply": True,
                    "deleted_db_rows": deleted_db_rows,
                    "deleted_work_dirs": deleted_work_dirs,
                    "skipped_active": skipped_active,
                },
            )

        return {
            "apply": apply_cleanup,
            "drift": drift,
            "deleted_db_rows": deleted_db_rows,
            "deleted_work_dirs": deleted_work_dirs,
            "skipped_active": skipped_active,
        }

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
            if not p.is_file() or p.stat().st_size == 0:
                raise HTTPException(404, "无上传文件")
            return FileResponse(p, filename="input.pdf", media_type="application/pdf")
        if kind == "output_md":
            p = root / "output" / "translated_full.md"
            if not p.is_file() or p.stat().st_size == 0:
                raise HTTPException(404, "尚无译文")
            return FileResponse(p, filename="translated_full.md", media_type="text/markdown; charset=utf-8")
        if kind == "output_pdf":
            p = root / "output" / "translated_full.pdf"
            if not p.is_file() or p.stat().st_size == 0:
                raise HTTPException(404, "PDF 译文尚未生成")
            return FileResponse(p, filename="translated_full.pdf", media_type="application/pdf")
        if kind == "repair_publish":
            p = root / "output" / "repair_publish.md"
            if not p.is_file() or p.stat().st_size == 0:
                raise HTTPException(404, "局部修复发布确认报告尚未生成")
            return FileResponse(p, filename="repair_publish.md", media_type="text/markdown; charset=utf-8")
        if kind == "repair_published_full":
            p = root / "output" / "published_full.md"
            if not p.is_file() or p.stat().st_size == 0:
                raise HTTPException(404, "人工确认修复发布稿尚未生成")
            return FileResponse(p, filename="published_full.md", media_type="text/markdown; charset=utf-8")
        if kind == "bundle_zip":
            if rec.status not in ("done", "cancelled"):
                raise HTTPException(409, "任务未完成或未终止")
            md = root / "output" / "translated_full.md"
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
        raise HTTPException(400, "kind 必须是 input / output_md / output_pdf / repair_publish / repair_published_full / bundle_zip")

    api.include_router(admin)
    return api
