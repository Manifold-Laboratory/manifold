@echo off

echo ================================================================
echo [GFN] Custom CUDA Kernel Compilation Pipeline (VS 2022 + CUDA 11.8)
echo ================================================================

REM --- 1. Find Visual Studio 2022 Installation ---
set "VS_PATH="

if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
    set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    set "VS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)

if "%VS_PATH%"=="" (
    echo [ERROR] Could not find Visual Studio 2022 or 2019 installation.
    echo Please ensure "Desktop development with C++" is installed.
    pause
    exit /b 1
)

echo [*] Found MSVC Environment: "%VS_PATH%"
echo [*] Initializing Developer Console...
call "%VS_PATH%"

REM --- 2. Setup CUDA Environment ---
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
set "PATH=%CUDA_PATH%\bin;%PATH%"

echo [*] Using CUDA Path: "%CUDA_PATH%"
nvcc --version

REM --- 3. Compile Kernels ---
echo [*] Cleaning old builds...
rmdir /s /q build
rmdir /s /q gfn_cuda.egg-info

echo.
echo [*] Starting JIT Compilation...
python -c "import torch; from gfn.cuda.ops import christoffel_fused; print('Compilation API check passed')"

if %errorlevel% neq 0 (
    echo [ERROR] Compilation failed.
    echo Tip: Try running 'pip install ninja' if not installed.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Kernels compiled and loaded successfully!
echo You can now run the benchmarks with native CUDA acceleration.
pause
