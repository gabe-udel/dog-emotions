@echo off
REM Double-click to open the point-and-click app.
REM pythonw.exe = no console window behind the app.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Could not find .venv\Scripts\pythonw.exe in %CD%
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "src\video_app.py"
