@echo off
title SatQuery Remote GPU Inference Host
echo ================================================================
echo   SatQuery Remote GPU Inference Host (SIH 2026 / ISRO PS 26167)
echo ================================================================
echo Starting GPU model server and public HTTPS tunnel...
cd /d "%~dp0Model Training"
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" run_gpu_host.py
) else (
    python run_gpu_host.py
)
pause
