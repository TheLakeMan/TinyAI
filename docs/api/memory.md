# Memory Management API Reference

## Overview

The Memory Management API provides functions for configuring and managing memory usage in TinyAI, including memory pools, allocation strategies, and optimization features.

## Memory Configuration

### `tinyai_configure_memory()`
```c
void tinyai_configure_memory(const TinyAIMemoryConfig* config);
```
Configures the memory management system with specified settings.

**Parameters:**
- `config`: Pointer to memory configuration structure

**Example:**
```c
TinyAIMemoryConfig mem_config = {
    .initial_pool_size = 1024 * 1024,  // 1MB
    .enable_optimization = true,
    .track_usage = true
};
tinyai_configure_memory(&mem_config);
```

### `tinyai_get_memory_config()`
```c
void tinyai_get_memory_config(TinyAIMemoryConfig* config);
```
Retrieves the current memory configuration.

**Parameters:**
- `config`: Pointer to store configuration

## Memory Allocation

### `tinyai_malloc()`
```c
void* tinyai_malloc(size_t size);
```
Allocates memory from the memory pool.

**Parameters:**
- `size`: Size in bytes to allocate

**Returns:**
- Pointer to allocated memory or NULL on failure

**Example:**
```c
void* data = tinyai_malloc(1024);
if (!data) {
    printf("Memory allocation failed\n");
    return 1;
}
```

### `tinyai_free()`
```c
void tinyai_free(void* ptr);
```
Frees previously allocated memory.

**Parameters:**
- `ptr`: Pointer to memory to free

**Example:**
```c
void* data = tinyai_malloc(1024);
// Use memory
tinyai_free(data);
```

## Memory Pool Management

### `tinyai_create_memory_pool()`
```c
TinyAIMemoryPool* tinyai_create_memory_pool(size_t size);
```
Creates a new memory pool.

**Parameters:**
- `size`: Initial size of the pool

**Returns:**
- Pointer to memory pool or NULL on failure

### `tinyai_destroy_memory_pool()`
```c
void tinyai_destroy_memory_pool(TinyAIMemoryPool* pool);
```
Destroys a memory pool and frees all associated memory.

**Parameters:**
- `pool`: Pointer to memory pool

## Memory Usage Tracking

### `tinyai_get_memory_usage()`
```c
size_t tinyai_get_memory_usage(void);
```
Returns the current memory usage in bytes.

**Returns:**
- Current memory usage in bytes

### `tinyai_get_memory_stats()`
```c
void tinyai_get_memory_stats(TinyAIMemoryStats* stats);
```
Retrieves detailed memory statistics.

**Parameters:**
- `stats`: Pointer to store statistics

## Memory Optimization

### `tinyai_optimize_memory()`
```c
void tinyai_optimize_memory(void);
```
Performs memory optimization operations.

### `tinyai_set_memory_strategy()`
```c
void tinyai_set_memory_strategy(TinyAIMemoryStrategy strategy);
```
Sets the memory allocation strategy.

**Parameters:**
- `strategy`: Memory strategy to use

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

### `TinyAIMemoryStats`
```c
typedef struct {
    size_t total_allocated;
    size_t peak_usage;
    size_t current_usage;
    size_t fragmentation;
} TinyAIMemoryStats;
```
Memory statistics structure.

### `TinyAIMemoryStrategy`
```c
typedef enum {
    TINYAI_MEMORY_STRATEGY_DEFAULT,
    TINYAI_MEMORY_STRATEGY_POOLED,
    TINYAI_MEMORY_STRATEGY_STREAMING
} TinyAIMemoryStrategy;
```
Memory allocation strategy enumeration.

## Best Practices

1. Configure memory settings based on application needs
2. Use memory pools for frequent allocations
3. Monitor memory usage during development
4. Enable optimization for production use
5. Track memory usage for debugging
6. Use appropriate allocation strategies

## Common Patterns

### Memory Pool Usage
```c
// Create memory pool
TinyAIMemoryPool* pool = tinyai_create_memory_pool(1024 * 1024);
if (!pool) {
    printf("Failed to create memory pool\n");
    return 1;
}

// Configure memory
TinyAIMemoryConfig mem_config = {
    .initial_pool_size = 1024 * 1024,
    .enable_optimization = true
};
tinyai_configure_memory(&mem_config);

// Allocate memory
void* data = tinyai_malloc(1024);
if (!data) {
    printf("Memory allocation failed\n");
    return 1;
}

// Use memory
// ...

// Free memory
tinyai_free(data);

// Clean up
tinyai_destroy_memory_pool(pool);
```

### Memory Usage Monitoring
```c
// Enable memory tracking
TinyAIMemoryConfig mem_config = {
    .track_usage = true
};
tinyai_configure_memory(&mem_config);

// Get memory statistics
TinyAIMemoryStats stats;
tinyai_get_memory_stats(&stats);
printf("Total allocated: %zu bytes\n", stats.total_allocated);
printf("Peak usage: %zu bytes\n", stats.peak_usage);
printf("Current usage: %zu bytes\n", stats.current_usage);
```

## Related Documentation

- [Core API](core.md)
- [Performance Tools API](performance.md)
- [Models API](models.md) 