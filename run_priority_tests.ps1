# TinyAI Priority Test Execution Script (PowerShell)
# This script runs tests in the priority order defined in the test plan

Write-Host "===== TinyAI Priority Test Execution =====" -ForegroundColor Cyan
Write-Host

# Create build directory if it doesn't exist
if (-not (Test-Path -Path "build")) {
    New-Item -ItemType Directory -Path "build" | Out-Null
}
Set-Location -Path "build"

# Configure and build the project
Write-Host "Configuring and building..." -ForegroundColor Yellow
cmake -G "Visual Studio 17 2022" ..
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Configuration failed" -ForegroundColor Red
    Set-Location -Path ".."
    exit $LASTEXITCODE
}

cmake --build . --config Debug
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Build failed" -ForegroundColor Red
    Set-Location -Path ".."
    exit $LASTEXITCODE
}

Write-Host
Write-Host "===== Running tests in priority order =====" -ForegroundColor Cyan
Write-Host

# Function to run a test and check result
function Run-Test {
    param (
        [string]$TestName,
        [string]$Command,
        [string]$Args = $null
    )
    
    Write-Host "------ $TestName ------" -ForegroundColor Yellow
    
    if ($Args) {
        & $Command $Args
    } else {
        & $Command
    }
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: $TestName failed" -ForegroundColor Red
        Set-Location -Path ".."
        exit $LASTEXITCODE
    }
    
    Write-Host
}

# 1. Core Memory Management Tests
Run-Test "Memory Management Tests" ".\Debug\tinyai_tests.exe" "memory"

# Out-of-Memory Handling Test
Run-Test "Out-of-Memory Handling Test" ".\Debug\oom_handling_test.exe"

# Memory-Mapped Model Test
Run-Test "Memory-Mapped Model Test" ".\Debug\mmap_loader_test.exe"

# 2. 4-bit Quantization Tests
Run-Test "Quantization Tests" ".\Debug\tinyai_tests.exe" "quantize"

# Sparse Matrix Tests
Run-Test "Sparse Matrix Tests" ".\Debug\sparse_matrix_test.exe"

# 3. SIMD Acceleration Tests
Run-Test "SIMD Acceleration Tests" ".\Debug\tinyai_tests.exe" "simd"

# SIMD Benchmark Test
Run-Test "SIMD Benchmark Test" ".\Debug\simd_benchmark_test.exe"

# Attention Mechanism Test
Run-Test "Attention Mechanism Test" ".\Debug\attention_test.exe"

# 4. Text Generation Tests
Run-Test "Text Generation Tests" ".\Debug\tinyai_tests.exe" "generate"

# 5. Image Model Tests
Run-Test "Image Model Tests" ".\Debug\image_test.exe"

# 6. Multimodal Capabilities Tests
Run-Test "Multimodal Capabilities Tests" ".\Debug\multimodal_test.exe"

Write-Host
Write-Host "===== All priority tests completed successfully =====" -ForegroundColor Green
Write-Host

Set-Location -Path ".." 