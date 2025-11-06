@echo off
REM Helper script to locate Python installations

echo ============================================================
echo Python Installation Finder
echo ============================================================
echo.
echo Searching for Python 3.10.x installations...
echo.

echo Checking py launcher:
py -3.10 --version 2>nul
if %ERRORLEVEL% EQU 0 (
    echo   [FOUND] py -3.10 works!
    py -3.10 -c "import sys; print('   Location:', sys.executable)"
) else (
    echo   [NOT FOUND] py -3.10
)
echo.

echo Checking common installation paths:
echo.

if exist "C:\Python310\python.exe" (
    echo   [FOUND] C:\Python310\python.exe
    C:\Python310\python.exe --version
) else (
    echo   [NOT FOUND] C:\Python310\python.exe
)
echo.

if exist "C:\Program Files\Python310\python.exe" (
    echo   [FOUND] C:\Program Files\Python310\python.exe
    "C:\Program Files\Python310\python.exe" --version
) else (
    echo   [NOT FOUND] C:\Program Files\Python310\python.exe
)
echo.

if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    echo   [FOUND] %LOCALAPPDATA%\Programs\Python\Python310\python.exe
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" --version
) else (
    echo   [NOT FOUND] %LOCALAPPDATA%\Programs\Python\Python310\python.exe
)
echo.

echo Checking PATH:
where python 2>nul
where python3 2>nul
where python3.10 2>nul

echo.
echo ============================================================
echo Current Python (default):
echo ============================================================
python --version 2>nul
python -c "import sys; print('Location:', sys.executable)" 2>nul

echo.
echo ============================================================
echo All Python installations found:
echo ============================================================
where /R C:\ python.exe 2>nul | findstr /I "python310 python3.10"

echo.
echo ============================================================
echo To use Python 3.10 for this project:
echo ============================================================
echo   Option 1: Double-click run_setup.bat
echo   Option 2: py -3.10 setup.py
echo   Option 3: C:\Python310\python.exe setup.py
echo.
pause
