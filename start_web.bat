@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM 服务端口（你提到的 0901 按 TCP 端口使用 901）
set PDF_TRANSLATE_WEB_PORT=901
REM 数据目录：数据库 app.db、任务目录 web_jobs 均在此之下
set PDF_TRANSLATE_DATA=%~dp0data
REM 仅在「数据库中还没有任何用户」时用于创建首个管理员；之后改密请用管理端或重建 data\app.db
set PDF_TRANSLATE_BOOTSTRAP_ADMIN_PASSWORD=mic820323

if not exist ".venv\Scripts\activate.bat" (
  echo [错误] 未找到 .venv，请先执行：
  echo   python -m venv .venv
  echo   .\.venv\Scripts\pip install -e .
  pause
  exit /b 1
)

call .venv\Scripts\activate.bat
python -m pip install -e . -q
echo 启动 Web 服务： http://127.0.0.1:%PDF_TRANSLATE_WEB_PORT%
python -m pdf_translate.server
pause
