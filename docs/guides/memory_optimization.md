# Memory Optimization Guide

This guide explains how to optimize memory usage in TinyAI applications, allowing you to run larger models on constrained hardware.

## Overview

TinyAI offers several advanced memory optimization techniques:

1. **Memory-mapped model loading**: Load model weights on demand from storage
2. **Layer-wise computation**: Process layers sequentially to minimize peak memory usage
3. **Adaptive batch sizing**: Automatically adjust batch size based on available memory
4. **Intelligent weight caching**: Cache frequently accessed weights for performance
5. **Dynamic weight unloading**: Release weight memory when no longer needed

These techniques can be combined to significantly reduce memory requirements while maintaining good performance.

## Memory-Mapped Model Loading

### Basic Usage

Memory-mapped model loading allows TinyAI to access model weights directly from storage without loading the entire model into memory.

```c
#include <tinyai/utils/mmap_loader.h>

// Create a default configuration
TinyAIMmapConfig config = tinyaiCreateDefaultMmapConfig();

// Open the model with memory mapping
TinyAIMappedModel* model = tinyaiOpenMappedModel("path/to/model.tmai", &config);
if (!model) {
    printf("Failed to open model\n");
    return 1;
}

// Use the model
// ...

// Close the model when done
tinyaiCloseMappedModel(model);
```

### Configuration Options

You can customize the memory-mapped loading behavior:

```c
TinyAIMmapConfig config = tinyaiCreateDefaultMmapConfig();

// Set maximum cache size (default is 256MB)
config.maxCacheSize = 100 * 1024 * 1024; // 100MB

// Enable/disable prefetching (default is enabled)
config.prefetchEnabled = true;

// Set number of prefetch threads (default is 1)
config.prefetchThreads = 2;

// Enable/disable adaptive caching (default is enabled)
config.adaptiveCaching = true;

// Set minimum layer cache size (default is 4KB)
config.minLayerCacheSize = 8 * 1024; // 8KB
```

### Layer Access

You can explicitly control when layers are loaded and unloaded:

```c
// Get layer count
int layerCount = tinyaiGetMappedLayerCount(model);

// Get layer descriptor
const TinyAILayerDescriptor* desc = tinyaiGetLayerDescriptor(model, layerIndex);

// Load layer weights (will be cached based on configuration)
void* weights = tinyaiGetLayerWeights(model, layerIndex);

// Prefetch layer weights in anticipation of use
tinyaiPrefetchLayerWeights(model, layerIndex);

// Release layer weights when no longer needed
tinyaiReleaseLayerWeights(model, layerIndex);

// Get current memory usage
size_t memUsage = tinyaiGetMappedModelMemoryUsage(model);

// Set layer priority (higher values = more likely to keep in cache)
tinyaiSetLayerPriority(model, layerIndex, 2.0f);
```

## Forward Pass Scheduling

Forward pass scheduling enables layer-wise computation with automatic memory management.

### Basic Usage

```c
#include <tinyai/utils/forward_scheduler.h>

// Create a scheduler with memory limit
TinyAIForwardScheduler* scheduler = tinyaiCreateForwardScheduler(
    model,                  // Mapped model
    TINYAI_EXEC_MEMORY_OPT, // Memory optimization mode
    100 * 1024 * 1024       // 100MB memory limit
);

// Add layers to the schedule (in execution order)
for (int i = 0; i < layerCount; i++) {
    // For sequential network (each layer depends on previous)
    int dependsOn = (i > 0) ? i - 1 : -1;
    TinyAIDependencyType depType = (i > 0) ? TINYAI_DEP_SEQUENTIAL : TINYAI_DEP_NONE;
    
    // Each layer produces output activations of this size
    size_t outputSize = ...; // Size in bytes
    
    tinyaiAddLayerToSchedule(scheduler, i, dependsOn, depType, outputSize);
}

// Prepare for execution
tinyaiPrepareForwardPass(scheduler);

// Execute the forward pass layer by layer
const void* input = ...; // Input data
void* output = ...; // Output buffer
int layerIndex;

while (tinyaiExecuteNextLayer(scheduler, input, output, &layerIndex)) {
    // Layer layerIndex has been executed
    // Memory is automatically managed
}

// Clean up
tinyaiDestroyForwardScheduler(scheduler);
```

### Dependency Types

TinyAI supports different layer dependency patterns:

- `TINYAI_DEP_NONE`: No dependencies (can execute in parallel)
- `TINYAI_DEP_SEQUENTIAL`: Must execute after the previous layer
- `TINYAI_DEP_RESIDUAL`: Depends on an earlier layer (residual connection)
- `TINYAI_DEP_ATTENTION`: Complex dependency pattern (attention mechanism)

Example with residual connections:

```c
// Main branch, layer 0
tinyaiAddLayerToSchedule(scheduler, 0, -1, TINYAI_DEP_NONE, size0);

// Main branch, layer 1
tinyaiAddLayerToSchedule(scheduler, 1, 0, TINYAI_DEP_SEQUENTIAL, size1);

// Residual connection layer, depends on layer 0
tinyaiAddLayerToSchedule(scheduler, 2, 0, TINYAI_DEP_RESIDUAL, size2);

// Output layer, after both main branch and residual
tinyaiAddLayerToSchedule(scheduler, 3, 1, TINYAI_DEP_SEQUENTIAL, size3);
tinyaiAddLayerToSchedule(scheduler, 3, 2, TINYAI_DEP_SEQUENTIAL, size3);
```

### Execution Modes

TinyAI offers different execution modes based on your memory vs. speed requirements:

- `TINYAI_EXEC_NORMAL`: Standard execution with all layers loaded at once
- `TINYAI_EXEC_MEMORY_OPT`: Memory-optimized execution, loading only necessary layers
- `TINYAI_EXEC_STREAMING`: Process inputs in chunks for minimal memory usage
- `TINYAI_EXEC_ADAPTIVE`: Automatically choose best strategy based on model and hardware

### Adaptive Batch Size

You can calculate the optimal batch size based on available memory:

```c
// Calculate optimal batch size
int batchSize = tinyaiCalculateOptimalBatchSize(
    scheduler,    // Forward scheduler
    inputSize,    // Size of a single input in bytes
    outputSize,   // Size of a single output in bytes
    32            // Maximum batch size to consider
);

printf("Optimal batch size: %d\n", batchSize);
```

### Memory Usage Tracking

Monitor memory usage during execution:

```c
// Get current memory usage
size_t memUsage = tinyaiGetSchedulerMemoryUsage(scheduler);

// Get peak memory usage
size_t peakMemUsage = tinyaiGetSchedulerPeakMemoryUsage(scheduler);

printf("Current memory usage: %.2f MB\n", memUsage / (1024.0 * 1024.0));
printf("Peak memory usage: %.2f MB\n", peakMemUsage / (1024.0 * 1024.0));
```

## Advanced Memory Optimization Techniques

### Mixed Precision Quantization

TinyAI supports mixed precision quantization, allowing different layers to use different precision levels:

```c
#include <tinyai/utils/quantize_mixed.h>

// Create mixed precision configuration
TinyAIMixedPrecisionConfig mpConfig;
tinyaiInitMixedPrecisionConfig(&mpConfig);

// Set precision for different layer types
mpConfig.embeddingPrecision = TINYAI_PREC_INT8;   // 8-bit for embeddings
mpConfig.attentionPrecision = TINYAI_PREC_INT4;   // 4-bit for attention
mpConfig.ffnPrecision = TINYAI_PREC_INT4;         // 4-bit for feed-forward
mpConfig.outputPrecision = TINYAI_PREC_FP16;      // FP16 for output layer

// Apply mixed precision configuration to model
tinyaiApplyMixedPrecision(model, &mpConfig);
```

### Model Pruning

Pruning removes unnecessary weights to reduce model size:

```c
#include <tinyai/utils/prune.h>

// Create pruning configuration
TinyAIPruneConfig pruneConfig;
tinyaiInitPruneConfig(&pruneConfig);

// Set pruning parameters
pruneConfig.method = TINYAI_PRUNE_MAGNITUDE;   // Magnitude-based pruning
pruneConfig.sparsity = 0.7;                    // Target 70% sparsity
pruneConfig.blockSize = 4;                     // 4x4 block pruning
pruneConfig.layerSparsity[0] = 0.5;            // Custom sparsity for layer 0

// Apply pruning to the model
tinyaiPruneModel(model, &pruneConfig);

// Calculate memory savings
size_t savedBytes = tinyaiCalculatePruningSavings(model);
printf("Memory saved: %.2f MB\n", savedBytes / (1024.0 * 1024.0));
```

## Best Practices

1. **Use memory mapping for large models**: If your model is larger than available RAM, memory mapping is essential.

2. **Measure memory usage**: Monitor memory usage to identify bottlenecks and optimize accordingly.

3. **Consider mixed precision**: Not all layers need the same precision. Use higher precision only where necessary.

4. **Balance pruning and accuracy**: Excessive pruning can degrade model quality. Test thoroughly after pruning.

5. **Adjust batch size**: Processing larger batches is more efficient, but requires more memory. Find the sweet spot.

6. **Layer prioritization**: Prioritize caching for layers that are accessed frequently or compute-intensive.

7. **Prefetch strategically**: Enable prefetching for sequential access patterns, but disable if access is random.

## Practical Examples

### Running a Large Language Model on Limited RAM

```c
// Open model with memory mapping
TinyAIMmapConfig mmapConfig = tinyaiCreateDefaultMmapConfig();
mmapConfig.maxCacheSize = 100 * 1024 * 1024; // 100MB cache
TinyAIMappedModel* model = tinyaiOpenMappedModel("large_language_model.tmai", &mmapConfig);

// Create forward scheduler with 200MB limit
TinyAIForwardScheduler* scheduler = tinyaiCreateForwardScheduler(
    model, TINYAI_EXEC_MEMORY_OPT, 200 * 1024 * 1024);

// Add layers to schedule (transformer architecture)
// ... (add embedding layer, attention layers, FFN layers, etc.)

// Prepare for execution
tinyaiPrepareForwardPass(scheduler);

// Execute layer by layer
const char* input = "TinyAI is";
char output[1024];
int layerIndex;

while (tinyaiExecuteNextLayer(scheduler, input, output, &layerIndex)) {
    printf("Executed layer %d\n", layerIndex);
    printf("Current memory usage: %.2f MB\n", 
           tinyaiGetSchedulerMemoryUsage(scheduler) / (1024.0 * 1024.0));
}

printf("Generated text: %s\n", output);

// Clean up
tinyaiDestroyForwardScheduler(scheduler);
tinyaiCloseMappedModel(model);
```

This example demonstrates how to run a large language model on a device with limited RAM by using memory mapping and forward pass scheduling.

## Conclusion

TinyAI provides powerful memory optimization techniques that allow you to run large models on resource-constrained devices. By combining these techniques, you can significantly reduce memory requirements while maintaining good performance.

For more information, see the [API Reference](../api/optimization.md) and the [Memory-Mapped Model API Reference](../api/mmap_loader.md).
