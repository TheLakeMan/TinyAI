# TinyAI: Working with Visual Studio Tools in VS Code Terminal

This guide explains how to build and develop TinyAI using Visual Studio 2022 compiler and tools directly from VS Code's terminal interface.

## Overview

We've configured VS Code to use Visual Studio 2022's compiler and build tools through custom tasks. This approach gives you:

1. The power of Visual Studio's compiler and debugging tools
2. The lightweight, efficient interface of VS Code
3. Direct terminal access for build and run commands

## Required Extensions

For the best experience, install these VS Code extensions:

1. **C/C++** - Microsoft's C/C++ extension for VS Code
2. **CMake Tools** - For improved CMake integration
3. **CMake** - For CMake language support

## Building the Project

You can build TinyAI directly from VS Code:

### Using VS Code Tasks (Easiest Method)

1. Open the Command Palette (`Ctrl+Shift+P`)
2. Type `Tasks: Run Task`
3. Select one of the following tasks:
   - **Configure CMake** - Run this first to create/update the build configuration
   - **Build Debug** - Build with debugging symbols
   - **Build Release** - Build optimized version
   - **Run TinyAI** - Build and run the main executable
   - **Run Tests** - Build and run the test suite
   - **Open VS Developer Command Prompt** - Open a fully configured VS command prompt

### Using Terminal Commands

Alternatively, you can run commands directly in the VS Code terminal:

```bash
# First, open a VS Developer Command Prompt task
# Then run these commands:

# Configure the build
cmake -G "Visual Studio 17 2022" -S . -B build

# Build the debug version
cmake --build build --config Debug

# Run the main executable
build\Debug\tinyai.exe

# Run the tests
build\Debug\tinyai_tests.exe
```

## Debugging

You can debug the project directly in VS Code:

1. Set breakpoints by clicking in the margin next to line numbers
2. Open the Run and Debug sidebar (`Ctrl+Shift+D`)
3. Select either "Debug TinyAI" or "Debug Tests" from the dropdown
4. Press the green Play button or F5

The debugger provides:
- Variable inspection
- Call stack navigation
- Memory viewing
- Conditional breakpoints

## Project Status

The project now builds successfully with all core components implemented. The following areas are ready for development:

### 1. Testing Model Components

Now that the core components are working, focus on testing:

```bash
# Run all tests
build\Debug\tinyai_tests.exe

# Create test data for the tokenizer
echo "This is a test sentence for tokenization." > test_data.txt

# Run the main application with test data
build\Debug\tinyai.exe tokenize test_data.txt --output tokens.txt
```

### 2. Performance Testing

Test the performance of the 4-bit quantized operations:

```bash
# Time the execution
Measure-Command { build\Debug\tinyai.exe generate "Test prompt" --max-tokens 100 }

# Memory usage can be monitored with Windows Task Manager
```

### 3. Development Environment

All necessary components are installed. The project uses:

- Visual Studio 2022 Build Tools (C++ compiler, linker)
- CMake for build configuration
- VS Code for editing and task management

## Recommended Development Workflow

Now that the project builds successfully, follow this workflow for further development:

1. **Open TinyAI in VS Code**:
   ```
   code C:\Users\verme\OneDrive\Desktop\TinyAI
   ```

2. **Pull latest changes** if working in a team

3. **Run the Build Debug task** to build the project

4. **Run tests** to verify existing functionality

5. **Implement new features**:
   - Add test cases first (test-driven development)
   - Implement the feature
   - Test and benchmark the implementation

6. **Document your changes**:
   - Update API documentation
   - Update the relevant .md files
   - Add examples for new functionality

7. **Commit your changes** to source control

## Working on Specific Components

### Testing and Enhancing the Tokenizer

The tokenizer implementation is complete but needs testing:

1. Create test vocabulary files of different sizes
2. Test BPE encoding/decoding functionality
3. Verify special token handling
4. Benchmark performance with different text sizes

```bash
# Example command to test tokenizer
build\Debug\tinyai.exe tokenize input.txt --vocab vocab.txt --output tokens.txt
```

### Testing and Enhancing Text Generation

The text generation implementation is complete but needs testing:

1. Create small test model files
2. Test different sampling methods (greedy, top-k, top-p)
3. Verify 4-bit quantization accuracy
4. Measure generation speed and memory usage

```bash
# Example command to test generation
build\Debug\tinyai.exe generate "Test prompt" --model model.bin --max-tokens 100 --temp 0.7
```

### Creating Example Models

To test the system end-to-end:

1. Create small model files in the required format
2. Add test vocabulary files
3. Create script to convert from standard formats (e.g., ONNX) to TinyAI format

## Keyboard Shortcuts for Efficient Development

| Shortcut       | Action                          |
|----------------|----------------------------------|
| `Ctrl+Shift+B` | Run the default build task      |
| `Ctrl+Shift+P` | Open command palette            |
| `F5`           | Start debugging                 |
| `F9`           | Toggle breakpoint               |
| `F10`          | Step over during debugging      |
| `F11`          | Step into during debugging      |
| `Shift+F11`    | Step out during debugging       |
| `Ctrl+Shift+M` | Show Problems panel             |
| `Ctrl+`        | Open terminal                   |

## Next Development Steps

With the core implementation complete, focus on these tasks:

1. **Testing Infrastructure**:
   - Add more comprehensive tests for all components
   - Create benchmark suite for performance measurement
   - Add continuous integration via GitHub Actions or similar

2. **User Experience**:
   - Enhance command-line interface options
   - Add progress indicators for long operations
   - Create user-friendly error messages
   - Improve documentation with examples

3. **Future Components**:
   - Implement reasoning module for knowledge retrieval
   - Add vision processing capabilities
   - Create sample applications showcasing the framework

The project now has a solid foundation with all core components implemented. Focus on testing, optimization, and enhancing the user experience.
