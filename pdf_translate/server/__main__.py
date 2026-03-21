"""python -m pdf_translate.server 或 pdf-translate-web 启动 Web 服务。"""

from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    host = os.getenv("PDF_TRANSLATE_WEB_HOST", "0.0.0.0")
    port = int(os.getenv("PDF_TRANSLATE_WEB_PORT", "901"))
    reload = os.getenv("PDF_TRANSLATE_WEB_RELOAD", "").lower() in ("1", "true", "yes")
    uvicorn.run(
        "pdf_translate.server.app:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()
