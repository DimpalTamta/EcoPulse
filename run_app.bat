@echo off
REM Double-click to start EcoPulse (Windows)
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
python -m streamlit run app\main.py
pause
