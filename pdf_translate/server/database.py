from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import bcrypt

_db_lock = threading.Lock()
_db_path: Path | None = None


def configure(db_path: Path) -> None:
    global _db_path
    _db_path = db_path.resolve()
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    init_schema()


def _conn() -> sqlite3.Connection:
    if _db_path is None:
        raise RuntimeError("database not configured")
    c = sqlite3.connect(str(_db_path), check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_schema() -> None:
    with _db_lock, _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              username TEXT UNIQUE NOT NULL,
              password_hash BLOB NOT NULL,
              role TEXT NOT NULL CHECK(role IN ('admin','user')),
              created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jobs_meta (
              job_id TEXT PRIMARY KEY,
              user_id INTEGER NOT NULL,
              username TEXT NOT NULL,
              original_filename TEXT,
              created_at TEXT NOT NULL,
              FOREIGN KEY(user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS audit_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created_at TEXT NOT NULL,
              user_id INTEGER,
              username TEXT,
              ip TEXT,
              action TEXT NOT NULL,
              detail_json TEXT,
              job_id TEXT
            );
            CREATE TABLE IF NOT EXISTS app_kv (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS job_favorites (
              user_id INTEGER NOT NULL,
              job_id TEXT NOT NULL,
              favorited_at TEXT NOT NULL,
              PRIMARY KEY (user_id, job_id),
              FOREIGN KEY(user_id) REFERENCES users(id),
              FOREIGN KEY(job_id) REFERENCES jobs_meta(job_id)
            );
            """
        )
        c.commit()
    bootstrap_admin_if_empty()
    ensure_migrations()


def ensure_migrations() -> None:
    """旧版 app.db 可能缺少 job_favorites 表；单独补齐，避免列表/收藏接口报错。"""
    with _db_lock, _conn() as c:
        row = c.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='job_favorites'",
        ).fetchone()
        if row:
            return
        c.execute(
            """
            CREATE TABLE job_favorites (
              user_id INTEGER NOT NULL,
              job_id TEXT NOT NULL,
              favorited_at TEXT NOT NULL,
              PRIMARY KEY (user_id, job_id),
              FOREIGN KEY(user_id) REFERENCES users(id),
              FOREIGN KEY(job_id) REFERENCES jobs_meta(job_id)
            );
            """
        )
        c.commit()


def bootstrap_admin_if_empty() -> None:
    admin_user = os.getenv("PDF_TRANSLATE_ADMIN_USERNAME", "falling-feather").strip()
    bootstrap_pw = os.getenv("PDF_TRANSLATE_BOOTSTRAP_ADMIN_PASSWORD", "mic820323")
    with _db_lock, _conn() as c:
        n = c.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
        if n > 0:
            return
        h = bcrypt.hashpw(bootstrap_pw.encode("utf-8"), bcrypt.gensalt())
        c.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?,?,?,?)",
            (admin_user, h, "admin", _now_iso()),
        )
        c.commit()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_created_at(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        raw = s.strip().replace("Z", "+00:00")
        # SQLite 中可能出现 "2026-03-21 15:51:57" 等无时区字符串，按 UTC 理解
        if " " in raw and "T" not in raw:
            raw = raw.replace(" ", "T", 1)
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


MAX_JOB_FAVORITES_PER_USER = 3


def create_user(*, username: str, password: str, role: str = "user") -> int:
    h = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    with _db_lock, _conn() as c:
        try:
            cur = c.execute(
                "INSERT INTO users (username, password_hash, role, created_at) VALUES (?,?,?,?)",
                (username.strip(), h, role, _now_iso()),
            )
            c.commit()
            return int(cur.lastrowid)
        except sqlite3.IntegrityError as e:
            raise ValueError("用户名已存在") from e


def verify_user(username: str, password: str) -> dict[str, Any] | None:
    with _db_lock, _conn() as c:
        row = c.execute(
            "SELECT id, username, password_hash, role FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
        if not row:
            return None
        if not bcrypt.checkpw(password.encode("utf-8"), row["password_hash"]):
            return None
        return {"id": int(row["id"]), "username": row["username"], "role": row["role"]}


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    with _db_lock, _conn() as c:
        row = c.execute(
            "SELECT id, username, role FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if not row:
            return None
        return {"id": int(row["id"]), "username": row["username"], "role": row["role"]}


def insert_job_meta(job_id: str, user_id: int, username: str, original_filename: str) -> None:
    with _db_lock, _conn() as c:
        c.execute(
            """INSERT INTO jobs_meta (job_id, user_id, username, original_filename, created_at)
               VALUES (?,?,?,?,?)""",
            (job_id, user_id, username, original_filename, _now_iso()),
        )
        c.commit()


def list_jobs_for_user(user_id: int, limit: int = 100) -> list[dict[str, Any]]:
    with _db_lock, _conn() as c:
        rows = c.execute(
            """SELECT j.job_id, j.user_id, j.username, j.original_filename, j.created_at
               FROM jobs_meta j
               WHERE j.user_id = ?
                 AND j.job_id NOT IN (SELECT job_id FROM job_favorites WHERE user_id = ?)
               ORDER BY j.created_at DESC LIMIT ?""",
            (user_id, user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def count_job_favorites(user_id: int) -> int:
    with _db_lock, _conn() as c:
        row = c.execute(
            "SELECT COUNT(*) AS n FROM job_favorites WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return int(row["n"]) if row else 0


def list_favorite_jobs_for_user(user_id: int, limit: int = 100) -> list[dict[str, Any]]:
    with _db_lock, _conn() as c:
        rows = c.execute(
            """SELECT j.job_id, j.user_id, j.username, j.original_filename, j.created_at, f.favorited_at
               FROM job_favorites f
               JOIN jobs_meta j ON j.job_id = f.job_id AND j.user_id = f.user_id
               WHERE f.user_id = ?
               ORDER BY f.favorited_at DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def assert_job_owned_by_user(job_id: str, user_id: int) -> dict[str, Any]:
    with _db_lock, _conn() as c:
        row = c.execute(
            "SELECT job_id, user_id, username, original_filename, created_at FROM jobs_meta WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if not row or int(row["user_id"]) != user_id:
            raise ValueError("任务不存在或无权操作")
        return dict(row)


def add_job_favorite(user_id: int, job_id: str) -> None:
    assert_job_owned_by_user(job_id, user_id)
    with _db_lock, _conn() as c:
        n = c.execute(
            "SELECT COUNT(*) AS n FROM job_favorites WHERE user_id = ?",
            (user_id,),
        ).fetchone()["n"]
        if int(n) >= MAX_JOB_FAVORITES_PER_USER:
            raise ValueError(f"最多收藏 {MAX_JOB_FAVORITES_PER_USER} 条任务")
        try:
            c.execute(
                "INSERT INTO job_favorites (user_id, job_id, favorited_at) VALUES (?,?,?)",
                (user_id, job_id, _now_iso()),
            )
            c.commit()
        except sqlite3.IntegrityError as e:
            raise ValueError("该任务已在收藏中") from e


def remove_job_favorite(user_id: int, job_id: str) -> None:
    with _db_lock, _conn() as c:
        row = c.execute(
            "SELECT 1 FROM job_favorites WHERE user_id = ? AND job_id = ?",
            (user_id, job_id),
        ).fetchone()
        if not row:
            raise ValueError("未在收藏中找到该任务")
        c.execute("DELETE FROM job_favorites WHERE user_id = ? AND job_id = ?", (user_id, job_id))
        c.execute(
            "UPDATE jobs_meta SET created_at = ? WHERE job_id = ? AND user_id = ?",
            (_now_iso(), job_id, user_id),
        )
        c.commit()


def list_stale_job_ids_for_user(user_id: int, *, hours: int = 24) -> list[str]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    with _db_lock, _conn() as c:
        rows = c.execute(
            "SELECT job_id, created_at FROM jobs_meta WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        fav_rows = c.execute(
            "SELECT job_id FROM job_favorites WHERE user_id = ?",
            (user_id,),
        ).fetchall()
    fav = {r["job_id"] for r in fav_rows}
    out: list[str] = []
    for r in rows:
        jid = r["job_id"]
        if jid in fav:
            continue
        try:
            dt = _parse_created_at(r["created_at"])
            if dt is not None and dt < cutoff:
                out.append(jid)
        except (TypeError, ValueError, OSError):
            continue
    return out


def delete_job_meta_row(job_id: str) -> None:
    with _db_lock, _conn() as c:
        c.execute("DELETE FROM job_favorites WHERE job_id = ?", (job_id,))
        c.execute("DELETE FROM jobs_meta WHERE job_id = ?", (job_id,))
        c.commit()


def list_users(limit: int = 500) -> list[dict[str, Any]]:
    with _db_lock, _conn() as c:
        rows = c.execute(
            "SELECT id, username, role, created_at FROM users ORDER BY id ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def list_all_jobs(limit: int = 500) -> list[dict[str, Any]]:
    with _db_lock, _conn() as c:
        rows = c.execute(
            """SELECT job_id, user_id, username, original_filename, created_at
               FROM jobs_meta ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def log_audit(
    *,
    action: str,
    ip: str | None,
    user_id: int | None,
    username: str | None,
    job_id: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    with _db_lock, _conn() as c:
        c.execute(
            """INSERT INTO audit_events (created_at, user_id, username, ip, action, detail_json, job_id)
               VALUES (?,?,?,?,?,?,?)""",
            (
                _now_iso(),
                user_id,
                username,
                ip,
                action,
                json.dumps(detail or {}, ensure_ascii=False),
                job_id,
            ),
        )
        c.commit()


def log_job_finished(
    *,
    job_id: str,
    user_id: int | None,
    username: str | None,
    work_dir: Path,
    ok: bool,
    err: str | None = None,
) -> None:
    root = work_dir.resolve()
    detail: dict[str, Any] = {
        "ok": ok,
        "input_pdf": str((root / "input.pdf").resolve()) if (root / "input.pdf").is_file() else None,
        "translated_md": str((root / "output" / "translated_full.md").resolve())
        if (root / "output" / "translated_full.md").is_file()
        else None,
        "bundle_zip_ready": ok and (root / "output" / "translated_full.md").is_file(),
    }
    if err:
        detail["error"] = err
    log_audit(
        action="job_done" if ok else "job_error",
        ip=None,
        user_id=user_id,
        username=username,
        job_id=job_id,
        detail=detail,
    )


def list_audit(limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
    from pdf_translate.server.audit_humanize import summarize_audit_event

    with _db_lock, _conn() as c:
        rows = c.execute(
            """SELECT id, created_at, user_id, username, ip, action, detail_json, job_id
               FROM audit_events ORDER BY id DESC LIMIT ? OFFSET ?""",
            (limit, offset),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["detail"] = json.loads(d.pop("detail_json") or "{}")
            except json.JSONDecodeError:
                d["detail"] = {}
            d["summary"] = summarize_audit_event(
                d.get("action") or "",
                d.get("detail") or {},
                d.get("username"),
                d.get("job_id"),
            )
            out.append(d)
        return out


def kv_get(key: str, default: str | None = None) -> str | None:
    with _db_lock, _conn() as c:
        row = c.execute("SELECT value FROM app_kv WHERE key = ?", (key,)).fetchone()
        if not row:
            return default
        return row["value"]


def kv_set(key: str, value: str) -> None:
    with _db_lock, _conn() as c:
        c.execute(
            "INSERT INTO app_kv (key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        c.commit()


def kv_get_json(key: str, default: Any) -> Any:
    raw = kv_get(key)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def kv_set_json(key: str, obj: Any) -> None:
    kv_set(key, json.dumps(obj, ensure_ascii=False))


def registration_open() -> bool:
    v = kv_get("registration_open")
    if v is None:
        return True
    return str(v).lower() in ("1", "true", "yes", "on")


def get_jwt_secret_file(base_dir: Path) -> str:
    p = base_dir / "jwt_secret.txt"
    if p.is_file():
        return p.read_text(encoding="utf-8").strip()
    secret = uuid.uuid4().hex + uuid.uuid4().hex
    p.write_text(secret, encoding="utf-8")
    return secret
