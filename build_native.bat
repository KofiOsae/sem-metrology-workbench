@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" -m pip install -r requirements-native.txt -r requirements-build.txt
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean SEM_Metrology_Native.spec
if errorlevel 1 exit /b 1
echo Build complete: dist\SEM Metrology Workbench.exe
endlocal
