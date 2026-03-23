@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist "frontend\package.json" (
  echo [错误] 未找到 frontend\package.json
  pause
  exit /b 1
)

where node >nul 2>&1
if errorlevel 1 (
  echo [错误] 未检测到 Node.js。请安装 LTS 版：https://nodejs.org/
  pause
  exit /b 1
)

cd frontend
call npm install
if errorlevel 1 (
  echo [错误] npm install 失败
  cd ..
  pause
  exit /b 1
)
call npm run build
set ERR=%ERRORLEVEL%
cd ..
if not %ERR%==0 (
  echo [错误] npm run build 失败
  pause
  exit /b 1
)

echo.
echo [完成] 前端已构建到 pdf_translate\server\static\
pause
