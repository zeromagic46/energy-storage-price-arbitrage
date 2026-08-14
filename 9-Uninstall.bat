@echo off
REM ============================================================
REM  Uninstall dependencies and delete this toolkit
REM ============================================================
cd /d "%~dp0"
echo This folder: %~dp0
echo.
echo [Step 1/2] Uninstall Python packages
echo   Core packages (only used by this tool) : streamlit plotly pulp
echo   Shared packages (other projects may use): pandas matplotlib openpyxl
echo.
set /p UN1=Uninstall CORE packages (streamlit plotly pulp)? [Y/N]: 
if /i "%UN1%"=="Y" (
    python -m pip uninstall -y streamlit plotly pulp
)
set /p UN2=Also uninstall SHARED packages (pandas matplotlib openpyxl)? [Y/N]: 
if /i "%UN2%"=="Y" (
    python -m pip uninstall -y pandas matplotlib openpyxl
) else (
    echo Skipped shared packages.
)
echo.
echo [Step 2/2] Delete this folder and ALL files in it
echo   WARNING: removes scripts, offline packages, data and output files.
echo.
set /p DEL1=Delete the whole folder? [Y/N]: 
if /i not "%DEL1%"=="Y" (
    echo Folder kept. Uninstall finished.
    pause
    exit /b 0
)
echo Deleting folder in 3 seconds... close this window to cancel.
start "" /min cmd /c "timeout /t 3 /nobreak >nul & rd /s /q "%~dp0""
exit /b 0
