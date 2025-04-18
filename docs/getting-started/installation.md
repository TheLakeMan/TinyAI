# Installing TinyAI

This guide walks you through the process of installing TinyAI on different platforms.

## System Requirements

### Minimum Requirements
- CPU: Any x86 processor (Intel/AMD) from 1995 or later
- RAM: 50MB minimum
- Storage: 200MB
- OS: Windows 95+, Linux 2.0+, or macOS 7.6+

### Recommended Requirements
- CPU: x86 processor with SSE2 support
- RAM: 100MB or more
- Storage: 500MB
- OS: Windows XP+, Linux 2.6+, or macOS X+

### Optional Features
- AVX/AVX2 support for improved performance
- Additional RAM for larger models
- Additional storage for model files

## Installation Methods

### 1. Pre-built Binaries

#### Windows
1. Download the latest release from GitHub
2. Extract the ZIP file to your desired location
3. Add the bin directory to your PATH
4. Open Command Prompt and verify installation:
   ```cmd
   tinyai --version
   ```

#### Linux
1. Download the latest .tar.gz release
2. Extract the archive:
   ```bash
   tar xzf tinyai-0.1.0.tar.gz
   ```
3. Install dependencies:
   ```bash
   # Debian/Ubuntu
   sudo apt-get install libc6
   
   # RedHat/CentOS
   sudo yum install glibc
   ```
4. Add to PATH:
   ```bash
   export PATH=$PATH:/path/to/tinyai/bin
   ```
5. Verify installation:
   ```bash
   tinyai --version
   ```

#### macOS
1. Download the latest .dmg file
2. Mount the DMG and drag TinyAI to Applications
3. Open Terminal and verify installation:
   ```bash
   tinyai --version
   ```

### 2. Building from Source

#### Prerequisites
- C compiler (GCC 2.95+, MSVC 6.0+, or Clang)
- CMake 3.10+
- Git (for cloning repository)

#### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/TheLakeMan/tinyai.git
   cd tinyai
   ```

2. Create build directory:
   ```bash
   mkdir build
   cd build
   ```

3. Configure build:
   ```bash
   # Windows (MSVC)
   cmake -G "Visual Studio 17 2022" ..
   
   # Unix-like systems
   cmake ..
   ```

4. Build:
   ```bash
   # Windows
   cmake --build . --config Release
   
   # Unix-like systems
   make
   ```

5. Install (optional):
   ```bash
   # Windows (as Administrator)
   cmake --install .
   
   # Unix-like systems
   sudo make install
   ```

## Verifying Installation

Run the following commands to verify your installation:

```bash
# Check version
tinyai --version

# Run self-test
tinyai --self-test

# Try a simple example
tinyai --run-example hello
```

## Common Issues

### Windows
1. **Missing MSVC Runtime**
   - Download and install the appropriate Visual C++ Redistributable

2. **PATH Issues**
   - Ensure the bin directory is in your system PATH
   - Try restarting your command prompt

### Linux
1. **Library Dependencies**
   - Install required libraries using package manager
   - Check library versions match requirements

2. **Permission Issues**
   - Ensure proper file permissions
   - Use sudo for installation if needed

### macOS
1. **Security Warnings**
   - Right-click the application and select Open
   - Allow execution in Security & Privacy settings

## Next Steps

- Read the [Quick Start Guide](quickstart.md)
- Try the [Basic Usage Guide](basic-usage.md)
- Explore [Examples](../examples/) 