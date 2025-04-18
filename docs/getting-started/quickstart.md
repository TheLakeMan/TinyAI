# Quick Start Guide

This guide will help you get started with TinyAI quickly. We'll create a simple text generation example.

## Prerequisites

- TinyAI installed ([Installation Guide](installation.md))
- Basic understanding of C programming
- Text editor or IDE

## Your First TinyAI Program

### 1. Create a New Project

Create a new directory for your project:
```bash
mkdir tinyai-hello
cd tinyai-hello
```

### 2. Create the Source File

Create a file named `hello.c` with the following content:

```c
#include <tinyai/core.h>
#include <tinyai/models/text.h>
#include <stdio.h>

int main() {
    // Initialize TinyAI
    tinyai_init();
    
    // Load a small text model (included with TinyAI)
    TinyAIModel* model = tinyai_load_model("models/small_text_model.tmai");
    if (!model) {
        printf("Failed to load model\n");
        return 1;
    }
    
    // Generate text
    const char* prompt = "TinyAI is";
    char output[256];
    
    tinyai_generate_text(model, prompt, output, sizeof(output), 50);
    
    // Print the generated text
    printf("Generated: %s\n", output);
    
    // Clean up
    tinyai_free_model(model);
    tinyai_shutdown();
    
    return 0;
}
```

### 3. Compile the Program

Create a `CMakeLists.txt` file:

```cmake
cmake_minimum_required(VERSION 3.10)
project(tinyai-hello C)

find_package(TinyAI REQUIRED)

add_executable(hello hello.c)
target_link_libraries(hello PRIVATE TinyAI::TinyAI)
```

Build the project:
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

### 4. Run the Program

```bash
./hello
```

You should see output similar to:
```
Generated: TinyAI is a lightweight AI framework designed for minimal hardware requirements...
```

## Basic Concepts

### 1. Initialization

Always initialize TinyAI before use:
```c
tinyai_init();
```

### 2. Loading Models

Load pre-trained models:
```c
TinyAIModel* model = tinyai_load_model("path/to/model.tmai");
```

### 3. Memory Management

TinyAI provides memory-efficient operations:
```c
// Configure memory
TinyAIMemoryConfig mem_config = {
    .initial_pool_size = 1024 * 1024,  // 1MB
    .enable_optimization = true
};
tinyai_configure_memory(&mem_config);
```

### 4. Error Handling

Check return values and handle errors:
```c
if (!model) {
    printf("Error: %s\n", tinyai_get_error());
    return 1;
}
```

### 5. Cleanup

Always clean up resources:
```c
tinyai_free_model(model);
tinyai_shutdown();
```

## Next Steps

1. Try more examples:
   - Text classification
   - Image recognition
   - Memory optimization

2. Learn about:
   - [Memory Management](../guides/memory-management.md)
   - [Performance Optimization](../guides/optimization.md)
   - [Model Types](../api/models.md)

3. Explore the [API Reference](../api/core.md)

## Common Patterns

### 1. Text Generation

```c
// Configure generation parameters
TinyAIGenerationConfig config = {
    .max_length = 100,
    .temperature = 0.7f,
    .top_k = 40,
    .top_p = 0.9f
};

// Generate text
tinyai_generate_text_with_config(model, prompt, output, sizeof(output), &config);
```

### 2. Memory-Efficient Operations

```c
// Enable in-place operations
TinyAITensorConfig tensor_config = {
    .enable_in_place = true,
    .strategy = TINYAI_TENSOR_POOLED
};
TinyAITensor* tensor = tinyai_create_tensor_with_config(&tensor_config);
```

### 3. Performance Monitoring

```c
// Track performance metrics
TinyAIPerformanceConfig perf_config = {
    .track_memory_usage = true,
    .track_execution_time = true
};
TinyAIPerformanceMetrics metrics;
tinyai_get_performance_metrics(&metrics);
```

## Tips and Best Practices

1. **Memory Management**
   - Use memory pools for frequent allocations
   - Enable in-place operations when possible
   - Monitor memory usage with performance tools

2. **Performance**
   - Enable SIMD operations when available
   - Use appropriate batch sizes
   - Monitor performance metrics

3. **Error Handling**
   - Always check return values
   - Use error reporting functions
   - Clean up resources properly

## Troubleshooting

If you encounter issues:

1. Check the [Debugging and Troubleshooting Guide](../guides/debugging.md)
2. Verify your installation
3. Check system requirements
4. Enable debug logging:
   ```c
   tinyai_set_log_level(TINYAI_LOG_DEBUG);
   ```

## Getting Help

- Read the [Documentation](../index.md)
- Check [Examples](../examples/)
- Visit the [GitHub Repository](https://github.com/TheLakeMan/tinyai)
- Open an issue for bugs or questions 