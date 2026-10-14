@echo off
REM ===================================================================
REM  Double-click to open the Pupchat web app (the Stitch design).
REM
REM  This window IS the server. Closing it or pressing Ctrl+C stops the
REM  app. It also prints what the server is doing, which matters because
REM  the first question about a new clip takes a few minutes of CPU.
REM
REM  Your API key stays in .env on this machine. The web page never
REM  receives it - it only ever talks to this local server.
REM ===================================================================

cd /d "%~dp0"
title Pupchat Web App - close this window to stop the server

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
  echo     PUPCHAT_GEMINI_MODEL=gemini-3.8-flash
  echo     GEMINI_API_KEY=your-key-here
  echo.
  echo   The app will still open and still measure videos, but asking a
  echo   question will fail until a key is set.
  echo.
  pause
)

echo.
echo   Starting Pupchat...
echo   Opening http://127.0.0.1:8000 in your browser.
echo.
echo   Clips must be between 5 and 10 seconds.
echo   Leave this window open. Press Ctrl+C or close it to stop.
echo.

REM The server opens the browser once it is actually listening.
".venv\Scripts\python.exe" -m pupchat.api

echo.
echo   Pupchat has stopped.
pause
