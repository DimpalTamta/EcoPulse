#!/usr/bin/env bash
# Start EcoPulse (macOS / Linux)
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate
python -m streamlit run app/main.py
