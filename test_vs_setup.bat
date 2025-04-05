@echo off
echo **************************************
echo * TinyAI Visual Studio Setup Tester *
echo **************************************
echo.

REM Check for VS compiler
where cl >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Visual Studio compiler (cl.exe) not found in path.
    echo Run this script from the VS Developer Command Prompt task in VS Code.
    exit /b 1
) else (
    echo [OK] Visual Studio compiler (cl.exe) found.
    cl /? | findstr /C:"Microsoft" | findstr /C:"Compiler"
)

echo.
REM Check for CMake
where cmake >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] CMake not found in path.
    exit /b 1
) else (
    echo [OK] CMake found.
    cmake --version | findstr /C:"version"
)

echo.
REM Check for build directory
if not exist "build" (
    echo [INFO] Build directory does not exist. Creating it...
    mkdir build
    echo [OK] Build directory created.
) else (
    echo [OK] Build directory exists.
)

echo.
echo To build TinyAI:
echo 1. Press Ctrl+Shift+P
echo 2. Type "Tasks: Run Task"
echo 3. Select "Configure CMake"
echo 4. After configuration completes, run "Build Debug"
echo.
echo Or use these commands in the terminal:
echo - cmake -G "Visual Studio 17 2022" -S . -B build
echo - cmake --build build --config Debug
echo.
echo * Setup verification complete. *
echo * See VSCODE_TERMINAL_GUIDE.md for detailed instructions. *
