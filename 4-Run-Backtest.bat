@echo off
REM Run the command-line backtest (station_analysis.py)
REM Edit XLSX_PATH inside scripts\station_analysis.py first,
REM e.g. XLSX_PATH = "../data/your_price_file.xlsx"
REM Results (CSV/PNG/HTML) are saved into the "output" folder.
cd /d "%~dp0scripts"
python station_analysis.py
pause
