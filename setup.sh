#!/bin/bash
set -e

echo "[SETUP] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "[SETUP] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[SETUP] Done."