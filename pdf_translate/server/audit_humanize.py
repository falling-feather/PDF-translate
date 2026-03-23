from __future__ import annotations

from typing import Any


def summarize_audit_event(
    action: str,
    detail: dict[str, Any],
    username: str | None,
    job_id: str | None,
) -> str:
    u = (username or "").strip() or "某用户"
    fn = (detail.get("filename") or detail.get("name") or "").strip()

    if action == "login_success":
        return f"{u} 登录成功"
    if action == "login_fail":
        return f"{u} 登录失败（凭据错误）"
    if action == "register":
        return f"{u} 注册了新账号"
    if action == "job_submit":
        name = fn or "PDF 文件"
        be = (detail.get("backend") or "").strip()
        custom = bool(detail.get("use_custom_api"))
        extra = f"，后端 {be}" if be else ""
        if custom:
            extra += "（用户自带 API）"
        return f"{u} 上传了翻译任务《{name}》{extra}".rstrip("，")
    if action == "job_done":
        tail = f"（任务 {job_id}）" if job_id else ""
        return f"{u} 的翻译任务已完成{tail}"
    if action == "job_error":
        tail = f"（任务 {job_id}）" if job_id else ""
        return f"{u} 的翻译任务失败{tail}"
    if action == "admin_settings_update":
        keys = detail.get("keys")
        if isinstance(keys, list) and keys:
            return f"管理员 {u} 更新了系统设置（涉及 {len(keys)} 项配置）"
        return f"管理员 {u} 更新了系统设置"

    if job_id:
        return f"{action} · 用户 {u} · 任务 {job_id}"
    return f"{action} · 用户 {u}"
