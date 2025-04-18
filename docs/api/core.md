# Core API Reference

## Overview

The Core API provides the fundamental functionality for the TinyAI framework, including initialization, memory management, and basic operations.

## Initialization and Cleanup

### `tinyai_init()`
```c
void tinyai_init(void);
```
Initializes the TinyAI framework. Must be called before any other TinyAI functions.

**Example:**
```c
#include <tinyai/core.h>

int main() {
    tinyai_init();
    // Use TinyAI functionality
    tinyai_shutdown();
    return 0;
}
```

### `tinyai_shutdown()`
```c
void tinyai_shutdown(void);
```
Cleans up resources used by the TinyAI framework. Should be called when TinyAI is no longer needed.

## Memory Management

### `tinyai_configure_memory()`
```c
void tinyai_configure_memory(const TinyAIMemoryConfig* config);
```
Configures memory management settings.

**Parameters:**
- `config`: Pointer to memory configuration structure

**Example:**
```c
TinyAIMemoryConfig mem_config = {
    .initial_pool_size = 1024 * 1024,  // 1MB
    .enable_optimization = true
};
tinyai_configure_memory(&mem_config);
```

### `tinyai_get_memory_usage()`
```c
size_t tinyai_get_memory_usage(void);
```
Returns the current memory usage in bytes.

## Error Handling

### `tinyai_get_error()`
```c
const char* tinyai_get_error(void);
```
Returns the last error message.

**Example:**
```c
TinyAIModel* model = tinyai_load_model("model.tmai");
if (!model) {
    printf("Error: %s\n", tinyai_get_error());
    return 1;
}
```

## Logging

### `tinyai_set_log_level()`
```c
void tinyai_set_log_level(TinyAILogLevel level);
```
Sets the logging level.

**Parameters:**
- `level`: Logging level (TINYAI_LOG_DEBUG, TINYAI_LOG_INFO, TINYAI_LOG_WARN, TINYAI_LOG_ERROR)

**Example:**
```c
tinyai_set_log_level(TINYAI_LOG_DEBUG);
```

## Performance Monitoring

### `tinyai_get_performance_metrics()`
```c
void tinyai_get_performance_metrics(TinyAIPerformanceMetrics* metrics);
```
Retrieves current performance metrics.

**Parameters:**
- `metrics`: Pointer to metrics structure to be filled

**Example:**
```c
TinyAIPerformanceMetrics metrics;
tinyai_get_performance_metrics(&metrics);
printf("Memory usage: %zu bytes\n", metrics.memory_usage);
```

## Data Types

### `TinyAIMemoryConfig`
```c
typedef struct {
    size_t initial_pool_size;
    bool enable_optimization;
    bool track_usage;
} TinyAIMemoryConfig;
```
Memory configuration structure.

### `TinyAIPerformanceMetrics`
```c
typedef struct {
    size_t memory_usage;
    double execution_time;
    size_t cache_hits;
    size_t cache_misses;
} TinyAIPerformanceMetrics;
```
Performance metrics structure.

### `TinyAILogLevel`
```c
typedef enum {
    TINYAI_LOG_DEBUG,
    TINYAI_LOG_INFO,
    TINYAI_LOG_WARN,
    TINYAI_LOG_ERROR
} TinyAILogLevel;
```
Logging level enumeration.

## Best Practices

1. Always initialize TinyAI before use
2. Configure memory settings based on your needs
3. Check error messages when operations fail
4. Set appropriate logging level for debugging
5. Monitor performance metrics for optimization
6. Clean up resources when done

## Common Patterns

### Basic Usage
```c
#include <tinyai/core.h>

int main() {
    // Initialize
    tinyai_init();
    
    // Configure memory
    TinyAIMemoryConfig mem_config = {
        .initial_pool_size = 1024 * 1024,
        .enable_optimization = true
    };
    tinyai_configure_memory(&mem_config);
    
    // Set logging level
    tinyai_set_log_level(TINYAI_LOG_INFO);
    
    // Use TinyAI functionality
    
    // Clean up
    tinyai_shutdown();
    return 0;
}
```

### Error Handling
```c
TinyAIModel* model = tinyai_load_model("model.tmai");
if (!model) {
    printf("Error: %s\n", tinyai_get_error());
    return 1;
}
```

### Performance Monitoring
```c
TinyAIPerformanceMetrics metrics;
tinyai_get_performance_metrics(&metrics);
printf("Memory usage: %zu bytes\n", metrics.memory_usage);
printf("Execution time: %f ms\n", metrics.execution_time);
```

## Related Documentation

- [Memory Management API](memory.md)
- [Performance Tools API](performance.md)
- [Models API](models.md) 