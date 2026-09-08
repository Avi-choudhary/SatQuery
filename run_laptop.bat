@echo off
title SatQuery AI - Laptop Launcher
echo ================================================================
echo   SatQuery AI - Laptop Launcher (SIH 2026 / ISRO PS 26167)
echo ================================================================
echo.
echo Notice: If your desktop PC is acting as the GPU server, ensure you
echo have configured MODEL_SERVICE_URL in backend\.env or .env
echo.

:: Locate unified Python virtual environment
set "PYTHON_EXE=python"
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else if exist "%~dp0Model Training\venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0Model Training\venv\Scripts\python.exe"
) else if exist "%~dp0backend\SatQuery-master\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0backend\SatQuery-master\.venv\Scripts\python.exe"
)

:: 1. Start backend in a separate window
echo Starting SatQuery FastAPI Backend on port 8000...
start "SatQuery Backend (:8000)" cmd /k "cd /d ""%~dp0backend\SatQuery-master\backend"" && ""%PYTHON_EXE%"" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

:: 2. Start frontend
echo Starting SatQuery Frontend (:5173)...
cd /d "%~dp0frontend"
echo.
echo Launching Vite development server...
call npm run dev
pause
