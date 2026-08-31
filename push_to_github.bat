@echo off
REM ===================================================================
REM  Push this project to github.com/gabe-udel/dog-emotions
REM
REM  Double-click this file. The first time, a browser window opens so
REM  you can sign in to GitHub - that is Git Credential Manager, and it
REM  remembers you afterwards.
REM
REM  Nothing here touches your videos, the DogFLW images, or the model
REM  weights: .gitignore keeps all of those out of the repo.
REM ===================================================================

setlocal
cd /d "%~dp0"
set "PATH=%ProgramFiles%\Git\cmd;%PATH%"

where git >nul 2>&1
if errorlevel 1 (
  echo   ERROR: git is not installed, or not in %ProgramFiles%\Git\cmd
  echo.
  pause
  exit /b 1
)

echo.
echo   Repository : %CD%
git remote get-url origin
echo.
echo   Commits waiting to be pushed:
git log --oneline origin/main..HEAD
echo.

git push origin main
if errorlevel 1 (
  echo.
  echo   Push failed. The usual causes:
  echo     - you are not signed in to a GitHub account with write access
  echo     - the sign-in window was closed or cancelled
  echo     - the remote has commits you do not have yet ^(run: git pull --rebase^)
  echo.
  pause
  exit /b 1
)

echo.
echo   Pushed. https://github.com/gabe-udel/dog-emotions
echo.
pause
