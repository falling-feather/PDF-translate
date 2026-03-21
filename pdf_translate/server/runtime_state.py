from __future__ import annotations

from pathlib import Path

DATA_DIR: Path | None = None


def set_data_dir(p: Path) -> None:
    global DATA_DIR
    DATA_DIR = p.resolve()


def require_data_dir() -> Path:
    if DATA_DIR is None:
        raise RuntimeError("DATA_DIR not set")
    return DATA_DIR
