# Contributing to TinyAI

Thank you for your interest in contributing to TinyAI! This document outlines the process for contributing to the project and helps ensure a smooth collaboration experience.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful, inclusive, and considerate in all interactions.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with the following information:

1. A clear, descriptive title
2. A detailed description of the issue, including steps to reproduce
3. The expected and actual behavior
4. Any relevant logs, error messages, or screenshots
5. System information (OS, compiler version, etc.)

### Suggesting Enhancements

We welcome suggestions for enhancements! Please create an issue with:

1. A clear, descriptive title
2. A detailed description of the proposed enhancement
3. The rationale for the enhancement
4. Any relevant examples or mock-ups

### Pull Requests

1. Fork the repository
2. Create a new branch from the `main` branch
3. Make your changes
4. Run the tests and ensure they pass
5. Submit a pull request to the `main` branch

Please include:
- A clear, descriptive title
- A detailed description of the changes
- Any relevant issue numbers

## Development Setup

### Prerequisites

- C compiler (GCC, Clang, MSVC, etc.)
- CMake 3.10+

### Building the Project

```bash
# Clone the repository
git clone https://github.com/TheLakeMan/tinyai.git
cd tinyai

# Create a build directory
mkdir build
cd build

# Configure and build
cmake ..
cmake --build .
```

### Running Tests

```bash
cd build
ctest
```

## Coding Standards

### C Style Guide

- Use 4 spaces for indentation
- Maximum line length of 80 characters
- Use snake_case for function and variable names
- Use UPPER_CASE for constants and macros
- Add comments for public API functions and complex logic
- Include proper error handling

Example:

```c
/**
 * Brief description of the function
 * 
 * @param param1 Description of param1
 * @param param2 Description of param2
 * @return Description of return value
 */
int tinyai_function_name(int param1, const char *param2) {
    /* Function implementation */
    if (param1 < 0) {
        return -1;  /* Error case */
    }
    
    /* Complex logic with comment */
    int result = calculate_something(param1, param2);
    
    return result;
}
```

### Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

## License

By contributing to TinyAI, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

## Questions?

Feel free to reach out to the maintainers if you have any questions or need help with the contribution process.

Thank you for contributing to TinyAI!
