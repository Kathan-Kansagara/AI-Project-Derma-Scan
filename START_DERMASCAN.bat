@echo off
title DermaScan AI - Starting Server...
color 0A

echo.
echo  ========================================
echo     DermaScan AI - Clinical Portal
echo     EfficientNet-B2 Skin Analyzer
echo  ========================================
echo.
echo  [*] Starting backend server...
echo  [*] Website will open automatically...
echo.

:: Wait 3 seconds then open browser
start "" cmd /c "timeout /t 4 /nobreak >nul && start http://localhost:5000"

:: Start the FastAPI backend
cd /d "e:\AI project\backend"
python main.py

pause
