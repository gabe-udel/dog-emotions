@echo off
REM ===================================================================
REM  Double-click to open Pupchat in your browser.
REM
REM  Unlike "Dog Keypoints App.bat" this one KEEPS its console window.
REM  Pupchat is a web server, so that window is the server: closing it
REM  or pressing Ctrl+C stops the page. It also prints progress, which
REM  matters because the first question about a new video takes a few
REM  minutes of CPU before anything appears.
REM ===================================================================

cd /d "%~dp0"
title Pupchat - close this window to stop the server

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo   Could not find .venv\Scripts\python.exe in:
  echo     %CD%
  echo.
  echo   Run setup.bat first to build the Python environment.
  echo.
  pause
  exit /b 1
)

if not exist ".env" (
  echo.
  echo   No .env file found, so there is no API key.
  echo   Copy .env.example to .env and put your key in it:
  echo.
  echo     PUPCHAT_PROVIDER=gemini
  echo     GEMINI_API_KEY=your-key-here
  echo.
  echo   The page will still open, and the measurements will still work,
  echo   but asking a question will fail until a key is set.
  echo.
  pause
)

echo.
echo   Starting Pupchat...
echo   Your browser will open at http://127.0.0.1:7860
echo.
echo   Leave this window open. Press Ctrl+C or close it to stop.
echo.

".venv\Scripts\python.exe" -m pupchat.web

echo.
echo   Pupchat has stopped.
pause
