@echo off
rem DustView launcher: runs tools\dustview.py with the project venv (numpy, pillow, numba).
rem Creates the venv on first run if it is missing.
rem Double-click for the live game, or pass options, e.g.
rem   dustview.bat --model ALPHA_Map7_TheSandSeas
rem   dustview.bat --model ALPHA_Map11_Void --progress none --scale 1
setlocal
set "VENV=%~dp0.venv"
if exist "%VENV%\Scripts\python.exe" goto run

echo Creating Python venv in %VENV% ...
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -m venv "%VENV%"
) else (
    python -m venv "%VENV%"
)
if not exist "%VENV%\Scripts\python.exe" (
    echo Could not create the venv. Is Python 3 installed and on PATH?
    pause
    exit /b 1
)
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip
"%VENV%\Scripts\python.exe" -m pip install numpy pillow numba
if errorlevel 1 (
    echo Installing packages failed.
    pause
    exit /b 1
)

:run
"%VENV%\Scripts\python.exe" "%~dp0tools\dustview.py" %*
if errorlevel 1 pause
