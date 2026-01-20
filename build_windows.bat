@echo off
REM Windows Build Script for Manifold (GFN)
REM Handles environment variables for VS Developer Prompt and memory limits.

echo [1/4] Setting Environment Variables...
set DISTUTILS_USE_SDK=1
set MSSdk=1
set MAX_JOBS=1

echo [2/4] Cleaning previous build artifacts...
if exist build (
    echo Deleting build/ ...
    rmdir /s /q build
)
if exist dist (
    echo Deleting dist/ ...
    rmdir /s /q dist
)
if exist gfn.egg-info (
    echo Deleting gfn.egg-info/ ...
    rmdir /s /q gfn.egg-info
)

echo [3/4] Building Wheel (Serialized execution)...
echo This may take a few minutes. Please wait...
python -m build --no-isolation

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Build Failed!
    exit /b %ERRORLEVEL%
)

echo [4/4] Build Successful!
echo You can now push your changes.
