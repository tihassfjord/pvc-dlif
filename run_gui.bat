@echo off
REM Double-click launcher for Windows.
REM Activates the conda environment if it exists, then starts the GUI.

setlocal
cd /d "%~dp0"

where conda >nul 2>nul
if %errorlevel%==0 (
    call conda activate pvc-dlif 2>nul
)

python gui\run_gui.py
if %errorlevel% neq 0 (
    echo.
    echo The GUI exited with an error. Common causes:
    echo   - the pvc-dlif environment is not created:  conda env create -f environment.yml
    echo   - dependencies are missing:                 pip install -e ".[all]"
    echo.
    pause
)
endlocal
