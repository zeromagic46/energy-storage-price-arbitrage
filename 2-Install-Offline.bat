@echo off
REM Install dependencies OFFLINE from the local "packages" folder
REM NOTE: offline wheels were downloaded for Python 3.12 (64-bit Windows).
REM If your Python version differs, use 1-Install-Online.bat instead.
cd /d "%~dp0"
python -m pip install --no-index --find-links=packages -r requirements.txt
echo.
echo Install finished. Next: double-click 3-Start-App.bat
pause
