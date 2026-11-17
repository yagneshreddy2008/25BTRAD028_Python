@echo off
title SignBridge AI - Train Model
echo =====================================================
echo   SignBridge AI - Model Training Only
echo =====================================================
echo.
cd /d C:\Users\teega\.gemini\antigravity\scratch\signbridge-ai

echo [*] Training LSTM model (may take 5-15 minutes)...
C:\signenv\Scripts\python.exe model\train_model.py --epochs 40 --batch_size 32

echo.
echo [*] Running evaluation and generating confusion matrix...
C:\signenv\Scripts\python.exe model\evaluate_model.py

echo.
echo [*] Training complete! You can now run start.bat
pause
