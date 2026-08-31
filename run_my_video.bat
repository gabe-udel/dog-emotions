@echo off
REM ===================================================================
REM  Run the DogFLW-trained model on your own video.
REM
REM  WAY 1: Double-click this file. It will ask for the video - you can
REM         drag the video INTO the black window, which pastes its path,
REM         then press Enter.
REM
REM  WAY 2: In File Explorer, drag a video onto this file's icon.
REM         (Dropping it on a VS Code window does nothing - it has to be
REM         the icon in Explorer.)
REM
REM  WAY 3: From a terminal:
REM           run_my_video.bat "C:\path\to\clip.mp4"
REM           run_my_video.bat "C:\path\to\clip.mp4" 300
REM
REM  Output lands in outputs\ and opens automatically.
REM ===================================================================

setlocal
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
set "MODEL=model_weights\superanimal_quadruped_dogface_final.pt"
set "CONFIG=model_weights\pytorch_config.yaml"

echo.
echo   DogFLW face-keypoint model
echo   folder: %CD%
echo.

if not exist "%PY%" (
  echo   ERROR: could not find %PY%
  echo   The virtual environment is missing from this folder.
  echo.
  pause
  exit /b 1
)
if not exist "%MODEL%" (
  echo   ERROR: could not find the trained model:
  echo   %MODEL%
  echo.
  pause
  exit /b 1
)

REM -- video path: from the command line / drop, else ask for it ------
set "VIDEO=%~1"
if not defined VIDEO (
  echo   Drag your video file into THIS WINDOW and press Enter.
  echo   ^(Or paste/type the full path.^)
  echo.
  set /p "VIDEO=  Video: "
)

REM Dragging into a console wraps the path in quotes - strip them.
if defined VIDEO set "VIDEO=%VIDEO:"=%"

if not defined VIDEO (
  echo.
  echo   No video given - nothing to do.
  echo.
  pause
  exit /b 1
)
if not exist "%VIDEO%" (
  echo.
  echo   File not found:
  echo   %VIDEO%
  echo.
  echo   Check the path is complete and the file still exists.
  echo.
  pause
  exit /b 1
)

REM Filename without extension, works whether it came from %1 or set /p.
for %%F in ("%VIDEO%") do set "STEM=%%~nF"

REM Second argument = how many frames. Default 150 (~6 seconds at 25fps).
set "FRAMES=%~2"
if not defined FRAMES set "FRAMES=150"

set "OUT=outputs\%STEM%_keypoints.mp4"
if not exist outputs mkdir outputs

set OMP_NUM_THREADS=6

echo.
echo   video   : %VIDEO%
echo   frames  : %FRAMES%
echo   output  : %OUT%
echo.
echo   Running both models per frame for the side-by-side comparison.
echo   About 1 second per frame on this machine, so %FRAMES% frames is
echo   a few minutes. Progress prints below - it is not frozen.
echo.

"%PY%" src\run_video.py --video "%VIDEO%" --out "%OUT%" --config "%CONFIG%" --snapshot "%MODEL%" --compare --width 960 --smooth 3 --max-frames %FRAMES%

if errorlevel 1 (
  echo.
  echo   Something went wrong - the message above says what.
  echo.
  pause
  exit /b 1
)

echo.
echo   Done: %OUT%
echo.
start "" "%OUT%"
pause
