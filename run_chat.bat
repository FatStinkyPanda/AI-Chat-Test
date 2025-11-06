@echo off
REM Launcher script that finds and uses Python 3.10.x automatically

echo ============================================================
echo Brain-Inspired AI - Chat Launcher
echo ============================================================
echo.
echo Searching for Python 3.10.x installation...
echo.

REM Try common Python 3.10 locations
set PYTHON_CMD=

REM Try py launcher first (recommended for Windows)
py -3.10 --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Found Python 3.10 via py launcher
    set PYTHON_CMD=py -3.10
    goto :run_chat
)

REM Try standard installation path
if exist "C:\Python310\python.exe" (
    echo Found Python 3.10 at C:\Python310\python.exe
    set PYTHON_CMD=C:\Python310\python.exe
    goto :run_chat
)

REM Try Python310 in Program Files
if exist "C:\Program Files\Python310\python.exe" (
    echo Found Python 3.10 at C:\Program Files\Python310\python.exe
    set PYTHON_CMD="C:\Program Files\Python310\python.exe"
    goto :run_chat
)

REM Try user AppData
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    echo Found Python 3.10 at %LOCALAPPDATA%\Programs\Python\Python310\python.exe
    set PYTHON_CMD="%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    goto :run_chat
)

REM Try searching PATH
where python3.10 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Found python3.10 in PATH
    set PYTHON_CMD=python3.10
    goto :run_chat
)

REM Python 3.10 not found
echo.
echo ============================================================
echo ERROR: Python 3.10.x not found!
echo ============================================================
echo.
echo Please run setup first: run_setup.bat
echo.
pause
exit /b 1

:run_chat
echo.
echo Using: %PYTHON_CMD%
echo.

REM Verify it's actually Python 3.10
%PYTHON_CMD% --version

echo.
echo Starting AI chat...
echo.
%PYTHON_CMD% chat.py

echo.
pause
