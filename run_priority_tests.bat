@echo off
REM TinyAI Priority Test Execution Script
REM This script runs tests in the priority order defined in the test plan

echo ===== TinyAI Priority Test Execution =====
echo.

REM Create build directory if it doesn't exist
if not exist build mkdir build
cd build

REM Configure and build the project
echo Configuring and building...
cmake -G "Visual Studio 17 2022" ..
if %ERRORLEVEL% neq 0 (
  echo ERROR: Configuration failed
  exit /b %ERRORLEVEL%
)

cmake --build . --config Debug
if %ERRORLEVEL% neq 0 (
  echo ERROR: Build failed
  exit /b %ERRORLEVEL%
)

echo.
echo ===== Running tests in priority order =====
echo.

REM 1. Core Memory Management Tests
echo ------ Memory Management Tests ------
Debug\tinyai_tests.exe memory
if %ERRORLEVEL% neq 0 (
  echo ERROR: Memory management tests failed
  exit /b %ERRORLEVEL%
)

REM Out-of-Memory Handling Test
echo ------ Out-of-Memory Handling Test ------
Debug\oom_handling_test.exe
if %ERRORLEVEL% neq 0 (
  echo ERROR: Out-of-memory handling test failed
  exit /b %ERRORLEVEL%
)

REM Memory-Mapped Model Test
echo ------ Memory-Mapped Model Test ------
Debug\mmap_loader_test.exe
if %ERRORLEVEL% neq 0 (
  echo ERROR: Memory-mapped model test failed
  exit /b %ERRORLEVEL%
)

REM 2. 4-bit Quantization Tests
echo ------ Quantization Tests ------
Debug\tinyai_tests.exe quantize
if %ERRORLEVEL% neq 0 (
  echo ERROR: Quantization tests failed
  exit /b %ERRORLEVEL%
)

REM Sparse Matrix Tests
echo ------ Sparse Matrix Tests ------
Debug\sparse_matrix_test.exe
if %ERRORLEVEL% neq 0 (
  echo ERROR: Sparse matrix tests failed
  exit /b %ERRORLEVEL%
)

REM 3. SIMD Acceleration Tests
echo ------ SIMD Acceleration Tests ------
Debug\tinyai_tests.exe simd
if %ERRORLEVEL% neq 0 (
  echo ERROR: SIMD acceleration tests failed
  exit /b %ERRORLEVEL%
)

REM SIMD Benchmark Test
echo ------ SIMD Benchmark Test ------
Debug\simd_benchmark_test.exe
if %ERRORLEVEL% neq 0 (
  echo ERROR: SIMD benchmark test failed
  exit /b %ERRORLEVEL%
)

REM Attention Mechanism Test
echo ------ Attention Mechanism Test ------
Debug\attention_test.exe
if %ERRORLEVEL% neq 0 (
  echo ERROR: Attention mechanism test failed
  exit /b %ERRORLEVEL%
)

REM 4. Text Generation Tests
echo ------ Text Generation Tests ------
Debug\tinyai_tests.exe generate
if %ERRORLEVEL% neq 0 (
  echo ERROR: Text generation tests failed
  exit /b %ERRORLEVEL%
)

REM 5. Image Model Tests
echo ------ Image Model Tests ------
Debug\image_test.exe
if %ERRORLEVEL% neq 0 (
  echo ERROR: Image model tests failed
  exit /b %ERRORLEVEL%
)

REM 6. Multimodal Capabilities Tests
echo ------ Multimodal Capabilities Tests ------
Debug\multimodal_test.exe
if %ERRORLEVEL% neq 0 (
  echo ERROR: Multimodal capabilities tests failed
  exit /b %ERRORLEVEL%
)

echo.
echo ===== All priority tests completed successfully =====
echo.

cd ..
exit /b 0 