@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM ========== 试运行常用环境变量（可按需修改）==========
REM 监听端口
set PDF_TRANSLATE_WEB_PORT=901
REM 监听地址：本机试用可改为 127.0.0.1；局域网访问请保持 0.0.0.0 并在防火墙放行端口
set PDF_TRANSLATE_WEB_HOST=0.0.0.0
REM CLI/部分默认后端（管理后台可覆盖；用户端仍受「启用后端」限制）
set PDF_TRANSLATE_BACKEND=deepseek
REM 数据根：其下为 app.db 与 web_jobs\
set PDF_TRANSLATE_DATA=%~dp0data
REM 仅当数据库中还没有任何用户时，用于创建首个管理员账号
set PDF_TRANSLATE_BOOTSTRAP_ADMIN_PASSWORD=mic820323
REM 开发时改后端代码可自动重载（生产请勿开启）
REM set PDF_TRANSLATE_WEB_RELOAD=1

echo.
echo ========================================
echo   PDF Translate Web 启动
echo ========================================
echo   数据目录: %PDF_TRANSLATE_DATA%
echo   访问地址: http://127.0.0.1:%PDF_TRANSLATE_WEB_PORT%
echo   API 文档: http://127.0.0.1:%PDF_TRANSLATE_WEB_PORT%/docs
echo ========================================
echo.

if not exist "pdf_translate\server\static\index.html" (
  echo [提示] 未找到已构建的前端页面。
  echo        请先双击运行 build_frontend.bat，或在 frontend 目录执行: npm install ^&^& npm run build
  echo.
)

if not exist ".venv\Scripts\activate.bat" (
  echo [错误] 未找到 .venv，请先在本目录执行：
  echo   python -m venv .venv
  echo   .\.venv\Scripts\activate.bat
  echo   pip install -e .
  echo.
  pause
  exit /b 1
)

call .venv\Scripts\activate.bat
python -m pip install -e . -q
echo 正在启动服务…（关闭本窗口即停止）
echo.
python -m pdf_translate.server
pause
