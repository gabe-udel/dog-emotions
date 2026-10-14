@echo off
REM ===================================================================
REM  Double-click to put Pupchat on the internet for a while.
REM
REM  It starts the Pupchat server on this machine, then opens a
REM  Cloudflare "quick tunnel" and prints a public https:// address
REM  anyone can open. Your machine does all the measuring, and your
REM  API key never leaves it - the tunnel only forwards HTTP.
REM
REM  THIS WINDOW IS THE TUNNEL. Close it, or press Ctrl+C, and the
REM  public address stops working immediately. The address is random
REM  and a new one is issued every time, so nobody can come back to an
REM  old link later.
REM
REM  Share mode also turns on, which means:
REM    - uploaded clips are deleted after 24 hours
REM    - each visitor gets 10 videos and 20 questions per day
REM    - the whole demo stops at 300 questions a day, so the most it
REM      can cost you is roughly $1.50
REM
REM  Licence: the models are non-commercial (DogFLW is CC BY-NC 4.0,
REM  SuperAnimal follows the DeepLabCut Model Zoo terms). A free demo
REM  is fine. Charging for it, putting ads on it, or using it to
REM  promote a business is not.
REM ===================================================================

cd /d "%~dp0"
title Pupchat PUBLIC tunnel - close this window to take it offline
setlocal enabledelayedexpansion

set PORT=8000
set PUPCHAT_SHARE=1
REM The server reads this file to learn its own public address, which is
REM random and new on every run. Deleting it first matters: yesterday's log
REM parses perfectly and yields a hostname that is dead.
set PUPCHAT_TUNNEL_LOG=%~dp0outputs\pupchat\tunnel.log
if exist "%PUPCHAT_TUNNEL_LOG%" del /q "%PUPCHAT_TUNNEL_LOG%"

REM ------------------------------------------------- python environment
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

REM ------------------------------------------------- the API key
REM Without a key the page still measures videos but every question
REM fails, which is a poor thing to hand a stranger.
if not exist ".env" (
  echo.
  echo   No .env file, so there is no API key - visitors would be able
  echo   to upload a clip but every question would fail.
  echo.
  echo   Copy .env.example to .env and put your key in it:
  echo     PUPCHAT_PROVIDER=gemini
  echo     PUPCHAT_GEMINI_MODEL=gemini-3.8-flash
  echo     GEMINI_API_KEY=your-key-here
  echo.
  echo   Press Ctrl+C to stop, or any key to share it anyway.
  pause >nul
)

REM ------------------------------------------------- is the port free?
REM The server refuses to start on a busy port, and would otherwise
REM leave the tunnel pointing at whatever else is listening.
netstat -ano | findstr /r /c:"127.0.0.1:%PORT% .*LISTENING" >nul
if not errorlevel 1 (
  echo.
  echo   Something is already listening on port %PORT%.
  echo.
  echo   That is almost always another Pupchat window. Close it first,
  echo   or the tunnel would publish whatever it is instead of this.
  echo.
  pause
  exit /b 1
)

REM ------------------------------------------------- cloudflared
where cloudflared >nul 2>&1
if errorlevel 1 (
  echo.
  echo   cloudflared is not installed. It is Cloudflare's tunnel client,
  echo   and it is what gives you the public address.
  echo.
  echo   Installing it with winget now. Windows may ask permission.
  echo.
  winget install --id Cloudflare.cloudflared --exact --accept-package-agreements --accept-source-agreements
  where cloudflared >nul 2>&1
  if errorlevel 1 (
    echo.
    echo   cloudflared still is not on PATH. Two things to try:
    echo     1. Close this window and run it again - a new PATH needs a
    echo        new shell.
    echo     2. Install it by hand from
    echo        https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/
    echo.
    pause
    exit /b 1
  )
)

REM ------------------------------------------------- start the server
echo.
echo   Starting the Pupchat server on port %PORT% in its own window...
start "Pupchat server (share mode)" cmd /c ""%CD%\.venv\Scripts\python.exe" -m pupchat.api"

REM Wait for it rather than guessing. The first start imports torch,
REM which is not quick.
echo   Waiting for it to come up (this can take 30 seconds)...
set READY=0
for /l %%i in (1,1,60) do (
  if !READY!==0 (
    curl -s -o nul -m 2 "http://127.0.0.1:%PORT%/api/health" && set READY=1
    if !READY!==0 timeout /t 1 /nobreak >nul
  )
)

if !READY!==0 (
  echo.
  echo   The server did not answer on port %PORT%.
  echo   Look at the "Pupchat server" window - it will say why.
  echo.
  pause
  exit /b 1
)

echo   Server is up.
echo.
echo ===================================================================
echo   Opening the public tunnel.
echo.
echo   Look for the https://....trycloudflare.com address below and
echo   send that to whoever is trying it. It may take a few seconds
echo   to appear.
echo.
echo   The first question about a new clip takes about 90 seconds,
echo   because the keypoint model runs on this machine's CPU. Only one
echo   clip is measured at a time; the rest queue.
echo.
echo   On the Pupchat window, press "Open on your phone" for a QR code.
echo.
echo   Closing this window takes the demo offline.
echo ===================================================================
echo.

cloudflared tunnel --url "http://127.0.0.1:%PORT%" --logfile "%PUPCHAT_TUNNEL_LOG%"

REM ------------------------------------------------- tidy up
echo.
echo   Tunnel closed. Stopping the Pupchat server...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }" >nul 2>&1
echo   Pupchat is offline.
echo.
pause
