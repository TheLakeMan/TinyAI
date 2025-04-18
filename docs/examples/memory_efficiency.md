# Memory Efficiency Examples

This document provides practical examples of using TinyAI's memory-efficient tensor operations and performance impact assessment tools.

## Memory-Efficient Tensor Operations

### Example 1: Creating a Memory-Efficient Tensor

```c
#include "tinyai/tensor.h"
#include "tinyai/memory.h"

void create_memory_efficient_tensor() {
    // Configure tensor memory allocation
    TinyAITensorMemoryConfig config = {
        .strategy = TINYAI_TENSOR_POOLED,
        .pool_size = 1024 * 1024,  // 1MB pool
        .stream_buffer_size = 0,    // Not using streaming
        .enable_in_place = true     // Enable in-place operations
    };

    // Create tensor dimensions
    int dims[] = {32, 32, 3};  // 32x32x3 tensor
    TinyAIDataType dtype = TINYAI_FLOAT32;

    // Create the tensor
    TinyAITensor* tensor = tinyaiCreateTensorWithMemoryConfig(
        &config, 3, dims, dtype);

    // Use the tensor...
    
    // Get memory statistics
    TinyAITensorMemoryStats stats = tinyaiGetTensorMemoryStats(tensor);
    printf("Memory usage: %zu bytes\n", stats.used_memory);
    printf("Pool efficiency: %.2f%%\n", stats.pool_efficiency * 100.0f);

    // Clean up
    tinyaiFreeTensor(tensor);
}
```

### Example 2: In-Place Tensor Operations

```c
void perform_in_place_operations() {
    // Create tensor (as shown in Example 1)
    TinyAITensor* tensor = /* ... */;

    // Perform in-place addition
    float add_value = 1.0f;
    tinyaiTensorInPlaceOp(tensor, TINYAI_OP_ADD, &add_value);

    // Perform in-place multiplication
    float mul_value = 2.0f;
    tinyaiTensorInPlaceOp(tensor, TINYAI_OP_MUL, &mul_value);

    // Note: No additional memory is allocated for these operations
}
```

### Example 3: Streaming Tensor Data

```c
void stream_tensor_data() {
    // Configure for streaming
    TinyAITensorMemoryConfig config = {
        .strategy = TINYAI_TENSOR_STREAMING,
        .stream_buffer_size = 4096,  // 4KB buffer
        .enable_in_place = false
    };

    // Create a large tensor
    int dims[] = {1000, 1000};  // 1M elements
    TinyAITensor* tensor = tinyaiCreateTensorWithMemoryConfig(
        &config, 2, dims, TINYAI_FLOAT32);

    // Stream data in chunks
    float* data_chunk = malloc(1024 * sizeof(float));
    for (size_t offset = 0; offset < 1000000; offset += 1024) {
        // Fill data_chunk with values...
        tinyaiStreamTensorData(tensor, data_chunk, offset, 1024);
    }
    free(data_chunk);
}
```

## Performance Impact Assessment

### Example 1: Basic Performance Tracking

```c
#include "tinyai/performance.h"

void track_performance() {
    // Configure performance tracking
    TinyAIPerformanceConfig config = {
        .track_execution_time = true,
        .track_memory_usage = true,
        .track_cpu_usage = true,
        .track_cache_usage = true,
        .enable_optimization = true
    };

    // Create performance analysis context
    TinyAIPerformanceAnalysis* analysis = tinyaiCreatePerformanceAnalysis(&config);

    // Record baseline metrics
    TinyAIPerformanceMetrics baseline = {
        .execution_time_ms = 100.0,
        .memory_usage_bytes = 1024 * 1024,
        .cpu_usage_percent = 50.0f,
        .cache_misses = 1000,
        .cache_hits = 9000
    };
    tinyaiRecordMetrics(analysis, &baseline);

    // Perform some operations...

    // Record current metrics
    TinyAIPerformanceMetrics current = {
        .execution_time_ms = 80.0,
        .memory_usage_bytes = 512 * 1024,
        .cpu_usage_percent = 40.0f,
        .cache_misses = 800,
        .cache_hits = 9200
    };
    tinyaiRecordMetrics(analysis, &current);

    // Analyze optimization impact
    tinyaiAnalyzeOptimizationImpact(analysis, &baseline, &current);

    // Get and print the impact
    TinyAIOptimizationImpact impact = tinyaiGetOptimizationImpact(analysis);
    printf("Speedup: %.2fx\n", impact.speedup_factor);
    printf("Memory reduction: %.2f%%\n", impact.memory_reduction);
    printf("CPU efficiency: %.2f%%\n", impact.cpu_efficiency);
    printf("Recommendations: %s\n", impact.recommendations);

    // Generate report
    tinyaiGeneratePerformanceReport(analysis, "performance_report.txt");

    // Clean up
    tinyaiFreePerformanceAnalysis(analysis);
}
```

### Example 2: Continuous Performance Monitoring

```c
void monitor_performance() {
    TinyAIPerformanceAnalysis* analysis = /* ... */;
    
    // Take performance samples at regular intervals
    for (int i = 0; i < 10; i++) {
        // Perform operations...
        
        // Take a performance sample
        tinyaiTakePerformanceSample(analysis);
        
        // Get current metrics
        TinyAIPerformanceMetrics metrics = tinyaiGetPerformanceMetrics(analysis);
        printf("Sample %d: %.2f ms, %zu bytes\n", 
               i, metrics.execution_time_ms, metrics.memory_usage_bytes);
        
        // Calculate trend
        float trend = tinyaiGetPerformanceTrend(analysis);
        printf("Performance trend: %.2f%%\n", trend * 100.0f);
    }
}
```

### Example 3: Memory Usage Analysis

```c
void analyze_memory_usage() {
    TinyAIPerformanceAnalysis* analysis = /* ... */;
    
    // Enable memory tracking
    tinyaiEnablePerformanceAnalysis(analysis, true);
    
    // Perform memory-intensive operations...
    
    // Get detailed memory statistics
    TinyAIMemoryStats stats = tinyaiGetMemoryStats();
    printf("Total allocations: %zu\n", stats.total_allocations);
    printf("Total deallocations: %zu\n", stats.total_deallocations);
    printf("Peak memory usage: %zu bytes\n", stats.peak_memory_usage);
    printf("Current memory usage: %zu bytes\n", stats.current_memory_usage);
    
    // Check for memory leaks
    if (stats.total_allocations != stats.total_deallocations) {
        printf("Warning: Potential memory leak detected!\n");
    }
}
```

## Best Practices

1. **Memory-Efficient Tensor Operations**
   - Use pooled allocation for frequently created/destroyed tensors
   - Enable in-place operations when possible
   - Use streaming for very large tensors
   - Monitor memory statistics regularly

2. **Performance Impact Assessment**
   - Establish baseline metrics before optimization
   - Track both memory and execution time
   - Use continuous monitoring for long-running operations
   - Generate regular performance reports
   - Pay attention to optimization recommendations

3. **General Tips**
   - Start with conservative memory budgets
   - Gradually increase optimization aggressiveness
   - Monitor both performance and memory usage
   - Use the provided tools to identify bottlenecks
   - Document performance improvements 