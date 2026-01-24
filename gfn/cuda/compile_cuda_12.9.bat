@echo off

echo ================================================================
echo [GFN] Custom CUDA Kernel Compilation Pipeline (VS 2022 + CUDA 12.9)
echo ================================================================

REM --- 1. Find Visual Studio 2022 Installation ---
set "VS_PATH="

if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
    set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
)

if "%VS_PATH%"=="" (
    echo [ERROR] Could not find Visual Studio 2022 installation.
    pause
    exit /b 1
)

echo [*] Found MSVC Environment: "%VS_PATH%"
echo [*] Initializing Developer Console...
call "%VS_PATH%"

REM --- 2. Setup CUDA Environment (12.9) ---
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "PATH=%CUDA_PATH%\bin;%PATH%"

echo [*] Using CUDA Path: "%CUDA_PATH%"
nvcc --version

REM --- 3. Compile Kernels ---
echo [*] Cleaning old builds...
rmdir /s /q build
rmdir /s /q gfn_cuda.egg-info
del /q *.pyc
rmdir /s /q __pycache__

echo.
echo [*] Starting Setup Compilation (In-place)...

REM Fix for "It seems that the VC environment is activated..." warning
set DISTUTILS_USE_SDK=1



python  setup.py build_ext --inplace 

if %errorlevel% neq 0 (
    echo [ERROR] Compilation failed.
    echo Ensure you have PyTorch installed for CUDA 12.x:
    echo pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Kernels compiled to local .pyd file!
echo Verified import:

python -c "import gfn_cuda; print('SUCCESS: gfn_cuda module imported directly!')"
pause
