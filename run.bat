@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
    echo 未找到项目 Python 环境。请先按照 README.md 准备依赖。
    pause
    exit /b 1
)
if "%~1"=="--check" (
    ".venv\Scripts\python.exe" -B app.py %*
) else (
    ".venv\Scripts\python.exe" -B desktop.py %*
)
set "app_exit=%errorlevel%"
if not "%app_exit%"=="0" pause
exit /b %app_exit%
