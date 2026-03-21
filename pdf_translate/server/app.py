from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pdf_translate.server import database
from pdf_translate.server.jobs import JobRegistry
from pdf_translate.server.routes_web import register_web_routes
from pdf_translate.server.runtime_state import set_data_dir

STATIC_DIR = Path(__file__).resolve().parent / "static"

DATA_BASE = Path(os.getenv("PDF_TRANSLATE_DATA", Path.cwd() / "data")).resolve()
DATA_ROOT = Path(os.getenv("PDF_TRANSLATE_WEB_DATA", DATA_BASE / "web_jobs")).resolve()

set_data_dir(DATA_BASE)
database.configure(DATA_BASE / "app.db")

registry = JobRegistry(DATA_ROOT)
registry.hydrate_from_disk()

app = FastAPI(
    title="PDF Translate Web",
    description="团队用：用户翻译 / 管理端配置与审计",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("PDF_TRANSLATE_CORS_ORIGINS", "*").split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(register_web_routes(registry))

if STATIC_DIR.is_dir() and (STATIC_DIR / "index.html").is_file():
    app.mount(
        "/",
        StaticFiles(directory=str(STATIC_DIR), html=True),
        name="frontend",
    )
else:

    @app.get("/")
    def no_frontend() -> dict:
        return {
            "hint": "前端未构建：在 frontend 目录执行 npm install && npm run build",
            "api": "/api/health",
            "docs": "/docs",
        }
