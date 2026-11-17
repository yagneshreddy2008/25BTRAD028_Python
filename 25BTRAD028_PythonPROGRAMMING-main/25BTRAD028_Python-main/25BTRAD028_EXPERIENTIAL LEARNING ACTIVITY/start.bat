@echo off
title SignBridge AI Server
echo =====================================================
echo   SignBridge AI - ISL to Text Translator
echo =====================================================
echo.

cd /d C:\Users\teega\.gemini\antigravity\scratch\signbridge-ai

REM Check that the model exists
if not exist "model\trained_model.pt" (
    echo [!] Model not found. Running training first...
    echo.
    C:\signenv\Scripts\python.exe model\train_model.py --epochs 40
    echo.
)

echo [*] Starting SignBridge AI backend server...
echo [*] Open your browser to: http://localhost:8000
echo [*] Press Ctrl+C to stop the server.
echo.

C:\signenv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

pause
