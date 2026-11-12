@echo off
setlocal
set "REFINE_ENABLED=false"
cd /d "%~dp0backend" || exit /b 1
"%~dp0.venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8766
