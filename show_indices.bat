@echo off
REM Double-click this file to render landmark-index overlays.
REM No arguments      -> one random dog face.
REM With arguments    -> passed straight through to the Python script, e.g.
REM                      show_indices.bat -r -n 6
REM                      show_indices.bat -r --ear-type pointy

setlocal
REM %~dp0 is this .bat file's own folder, so double-clicking works from anywhere.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo Could not find .venv\Scripts\python.exe
  echo Expected it in: %CD%
  echo The virtual environment may not have been created yet.
  echo.
  pause
  exit /b 1
)

if "%~1"=="" (
  ".venv\Scripts\python.exe" src\show_landmark_indices.py -r
) else (
  ".venv\Scripts\python.exe" src\show_landmark_indices.py %*
)

echo.
echo Done. Images are in outputs\figures\indices\
echo.
pause
