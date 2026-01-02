@echo off
REM Installation script for Windows VPS
REM Run this script to install all required dependencies

echo ========================================
echo XAUUSD Trading System - Installation
echo ========================================
echo.

REM Check Python version
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo Installing dependencies from requirements.txt...
echo.

REM Upgrade pip first
python -m pip install --upgrade pip

REM Install dependencies
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Some packages failed to install
    echo.
    echo Note: ta-lib is NOT required for this candlestick-only system
    echo All features are calculated from raw OHLC data
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation completed successfully!
echo ========================================
echo.
echo You can now run:
echo   python main.py --mode fetch
echo.
pause

