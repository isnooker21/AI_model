#!/bin/bash
# Installation script for Mac/Linux
# Run this script to install all required dependencies

echo "========================================"
echo "XAUUSD Trading System - Installation"
echo "========================================"
echo ""

# Check Python version
python3 --version
if [ $? -ne 0 ]; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    exit 1
fi

echo ""
echo "Installing dependencies from requirements.txt..."
echo ""

# Upgrade pip first
python3 -m pip install --upgrade pip

# Install dependencies
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Some packages failed to install"
    echo ""
    echo "If ta-lib installation fails, try:"
    echo "  pip3 install TA-Lib-binary"
    echo ""
    exit 1
fi

echo ""
echo "========================================"
echo "Installation completed successfully!"
echo "========================================"
echo ""
echo "You can now run:"
echo "  python3 main.py --mode fetch"
echo ""

