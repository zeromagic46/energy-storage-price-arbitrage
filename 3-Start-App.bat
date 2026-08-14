@echo off
REM Start the interactive web app (Streamlit)
cd /d "%~dp0scripts"
python -m streamlit run price_trading_app.py
pause
