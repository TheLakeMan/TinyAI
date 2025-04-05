# PowerShell script to clean up build artifacts for GitHub publishing

Write-Host "TinyAI Cleanup Script - Preparing for GitHub Publishing" -ForegroundColor Cyan
Write-Host "------------------------------------------------" -ForegroundColor Cyan
Write-Host ""

# Remove build directory
if (Test-Path -Path "build") {
    Write-Host "Removing build directory..." -ForegroundColor Yellow
    Remove-Item -Path "build" -Recurse -Force
    Write-Host "✓ build directory removed" -ForegroundColor Green
} else {
    Write-Host "✓ build directory not found (already clean)" -ForegroundColor Green
}

# Remove VS2022 directory
if (Test-Path -Path "vs2022") {
    Write-Host "Removing vs2022 directory..." -ForegroundColor Yellow
    Remove-Item -Path "vs2022" -Recurse -Force
    Write-Host "✓ vs2022 directory removed" -ForegroundColor Green
} else {
    Write-Host "✓ vs2022 directory not found (already clean)" -ForegroundColor Green
}

# Remove Visual Studio specific files
Write-Host "Removing Visual Studio specific files..." -ForegroundColor Yellow
$vsFiles = Get-ChildItem -Path "." -Recurse -File -Include "*.sdf", "*.opensdf", "*.suo", "*.user", "*.vcxproj.filters", "*.vcproj.*.user", "*.vcxproj.user", "*.aps", "*.pdb", "*.ilk", "*.idb", "*.ipdb", "*.iobj", "*.lastbuildstate", "*.tlog", "*.log", "*.ipch", "*.db", "*.opendb", "*.exp"
if ($vsFiles) {
    $vsFiles | ForEach-Object {
        Remove-Item -Path $_.FullName -Force
        Write-Host "  Deleted: $($_.FullName)" -ForegroundColor DarkYellow
    }
    Write-Host "✓ Visual Studio files removed" -ForegroundColor Green
} else {
    Write-Host "✓ No Visual Studio files found (already clean)" -ForegroundColor Green
}

# Remove CMake generated files at root level
Write-Host "Removing CMake generated files..." -ForegroundColor Yellow
$cmakeFiles = Get-ChildItem -Path "." -File -Include "CMakeCache.txt", "cmake_install.cmake", "cmake_uninstall.cmake", "CPackConfig.cmake", "CPackSourceConfig.cmake", "CTestTestfile.cmake", "install_manifest.txt", "compile_commands.json" -ErrorAction SilentlyContinue
if ($cmakeFiles) {
    $cmakeFiles | ForEach-Object {
        Remove-Item -Path $_.FullName -Force
        Write-Host "  Deleted: $($_.FullName)" -ForegroundColor DarkYellow
    }
    Write-Host "✓ CMake files removed" -ForegroundColor Green
} else {
    Write-Host "✓ No CMake files found at root level (already clean)" -ForegroundColor Green
}

# Remove any CMakeFiles directories
$cmakeDirs = Get-ChildItem -Path "." -Directory -Recurse -Include "CMakeFiles" -ErrorAction SilentlyContinue
if ($cmakeDirs) {
    $cmakeDirs | ForEach-Object {
        Remove-Item -Path $_.FullName -Recurse -Force
        Write-Host "  Deleted directory: $($_.FullName)" -ForegroundColor DarkYellow
    }
    Write-Host "✓ CMakeFiles directories removed" -ForegroundColor Green
} else {
    Write-Host "✓ No CMakeFiles directories found (already clean)" -ForegroundColor Green
}

# Remove Python cache files
Write-Host "Removing Python cache files..." -ForegroundColor Yellow
$pythonCacheFiles = Get-ChildItem -Path "." -Directory -Recurse -Include "__pycache__", "*.egg-info" -ErrorAction SilentlyContinue
if ($pythonCacheFiles) {
    $pythonCacheFiles | ForEach-Object {
        Remove-Item -Path $_.FullName -Recurse -Force
        Write-Host "  Deleted directory: $($_.FullName)" -ForegroundColor DarkYellow
    }
    Write-Host "✓ Python cache files removed" -ForegroundColor Green
} else {
    Write-Host "✓ No Python cache files found (already clean)" -ForegroundColor Green
}

# Check for .vs directory (hidden)
if (Test-Path -Path ".vs" -ErrorAction SilentlyContinue) {
    Write-Host "Removing .vs directory..." -ForegroundColor Yellow
    Remove-Item -Path ".vs" -Recurse -Force
    Write-Host "✓ .vs directory removed" -ForegroundColor Green
} else {
    Write-Host "✓ .vs directory not found (already clean)" -ForegroundColor Green
}

Write-Host ""
Write-Host "Cleanup complete! Your project is now ready for GitHub." -ForegroundColor Cyan
Write-Host "You can now run 'git add .' and commit your clean project." -ForegroundColor Cyan
