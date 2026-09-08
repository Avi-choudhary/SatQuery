@echo off
title SatQuery AI - Full Stack Local Launcher (PC Mode)
echo ================================================================
echo   SatQuery AI - Full Stack Local Launcher (SIH 2026 / ISRO PS 26167)
echo ================================================================
echo.
echo Running full stack locally on this PC:
echo   - Backend:  FastAPI on port 8000
echo   - AI Model: Qwen3-VL-2B (Local GPU CUDA acceleration)
echo   - Frontend: Vite React on port 5173
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

:: 1. Start FastAPI Backend in a separate window
echo [1/3] Starting SatQuery Backend (:8000)...
start "SatQuery Backend (:8000)" cmd /k "cd /d ""%~dp0backend\SatQuery-master\backend"" && ""%PYTHON_EXE%"" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

:: 2. Auto-open Web Browser after 3 seconds
echo [2/3] Opening browser at http://localhost:5173/console...
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:5173/console"

:: 3. Start Vite Frontend in current window
echo [3/3] Starting SatQuery Frontend (:5173)...
cd /d "%~dp0frontend"
echo.
echo Launching Vite development server...
call npm run dev
pause
