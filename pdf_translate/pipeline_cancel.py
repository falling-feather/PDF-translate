from __future__ import annotations

from pathlib import Path


class JobCancelled(Exception):
    """用户请求终止翻译。"""


def cancel_flag_path(work_dir: Path) -> Path:
    return work_dir.resolve() / ".cancel_requested"


def clear_cancel_flag(work_dir: Path) -> None:
    p = cancel_flag_path(work_dir)
    if p.is_file():
        p.unlink()


def is_cancel_requested(work_dir: Path) -> bool:
    return cancel_flag_path(work_dir).is_file()
