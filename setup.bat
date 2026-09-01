@echo off
REM ===================================================================
REM  One-time setup. Double-click this, then wait.
REM
REM  Builds the Python environment the app needs. Takes 5-15 minutes
REM  depending on your connection - it downloads about 1 GB of packages
REM  (PyTorch is most of that).
REM
REM  Safe to run again if it fails partway: it reuses what is there.
REM ===================================================================

setlocal
cd /d "%~dp0"

echo.
echo   Dog Keypoints - environment setup
echo   folder: %CD%
echo.

REM ---- Python 3.11 ---------------------------------------------------
REM 3.11 specifically: DeepLabCut 3.0.1 and its dependency wheels are not
REM all available for 3.13, and 3.12 has not been tested here.
py -3.11 --version >nul 2>&1
if errorlevel 1 (
  echo   ERROR: Python 3.11 not found.
  echo.
  echo   Install it from https://www.python.org/downloads/release/python-3119/
  echo   ^(pick "Windows installer 64-bit", and tick "Add python.exe to PATH"^)
  echo   then run this file again.
  echo.
  pause
  exit /b 1
)
for /f "tokens=*" %%v in ('py -3.11 --version') do echo   found %%v

REM ---- venv ----------------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
  echo   creating .venv ...
  py -3.11 -m venv .venv
  if errorlevel 1 goto :failed
) else (
  echo   .venv already exists - reusing it
)
set "PY=.venv\Scripts\python.exe"

echo   upgrading pip ...
"%PY%" -m pip install --upgrade pip setuptools wheel --quiet
if errorlevel 1 goto :failed

REM ---- torch, CPU build ----------------------------------------------
REM CPU wheels on purpose. PyTorch's ROCm builds are Linux-only, so an AMD
REM iGPU is not usable here; on an NVIDIA machine you may swap this line
REM for the CUDA index URL from pytorch.org.
echo   installing torch ^(CPU build, this is the slow one^) ...
"%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
if errorlevel 1 goto :failed

REM ---- DeepLabCut ----------------------------------------------------
REM --no-deps dodges DeepLabCut's numpy<2 pin, which is a leftover from its
REM TensorFlow engine. The PyTorch engine runs fine against numpy 2.
echo   installing deeplabcut 3.0.1 ...
"%PY%" -m pip install --no-deps deeplabcut==3.0.1 --quiet
if errorlevel 1 goto :failed

echo   installing the rest ...
"%PY%" -m pip install "dlclibrary>=0.0.12" matplotlib einops filterpy networkx pydantic tqdm imageio-ffmpeg scikit-learn scikit-image statsmodels tables pycocotools numba "albumentations<=1.4.3" --quiet
if errorlevel 1 goto :failed

REM timm is needed by DeepLabCut's HRNet backbone but is not declared by it.
echo   installing timm ...
"%PY%" -m pip install timm --quiet
if errorlevel 1 goto :failed

REM ---- verify --------------------------------------------------------
echo.
echo   verifying ...
"%PY%" -c "import deeplabcut, torch, timm, cv2; print('   deeplabcut', deeplabcut.__version__); print('   torch', torch.__version__)"
if errorlevel 1 goto :failed

echo.
if not exist "model_weights\superanimal_quadruped_dogface_final.pt" (
  echo   ================================================================
  echo   Environment is ready, but the trained model is NOT here yet.
  echo.
  echo   The checkpoint is 113 MB, over GitHub's 100 MB file limit, so it
  echo   is not in the repository. Download it from the Releases page:
  echo.
  echo       https://github.com/gabe-udel/dog-emotions/releases
  echo.
  echo   Save superanimal_quadruped_dogface_final.pt into:
  echo       %CD%\model_weights\
  echo.
  echo   Then double-click "Dog Keypoints App.bat".
  echo   ================================================================
) else (
  echo   Setup complete. Double-click "Dog Keypoints App.bat" to start.
)
echo.
pause
exit /b 0

:failed
echo.
echo   Setup failed at the step above. The error text is just before this.
echo   Re-running this file is safe - it keeps whatever already installed.
echo.
pause
exit /b 1
