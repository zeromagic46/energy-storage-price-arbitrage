@echo off
REM Install dependencies ONLINE (needs internet)
cd /d "%~dp0"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Install finished. Next: double-click 3-Start-App.bat
pause
