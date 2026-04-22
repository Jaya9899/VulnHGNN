@echo off
REM ========================================
REM  VulnHGNN - One-Click Launcher
REM  Activates venv and starts Flask server
REM ========================================

cd /d "%~dp0"

echo [*] Activating virtual environment...
call venv\Scripts\activate.bat

echo [*] Starting VulnHGNN server on http://127.0.0.1:5000
echo [*] Press Ctrl+C to stop.
echo.

cd src
python app.py
