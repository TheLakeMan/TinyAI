# TinyAI API Reference

## Overview

This document provides a comprehensive reference for the TinyAI API, covering all core components, their functions, and usage examples.

## Core Components

### Memory Management

```c
/**
 * Memory management configuration
 */
typedef struct {
    size_t initial_pool_size;    // Initial memory pool size
    size_t max_pool_size;        // Maximum memory pool size
    bool track_allocations;      // Whether to track allocations
    bool enable_optimization;    // Whether to enable memory optimization
} TinyAIMemoryConfig;

/**
 * Initialize memory management
 */
bool tinyaiInitMemory(const TinyAIMemoryConfig* config);

/**
 * Allocate memory
 */
void* tinyaiAlloc(size_t size);

/**
 * Free allocated memory
 */
void tinyaiFree(void* ptr);

/**
 * Get memory statistics
 */
TinyAIMemoryStats tinyaiGetMemoryStats();
```

### I/O System

```c
/**
 * File operations
 */
bool tinyaiFileExists(const char* path);
bool tinyaiReadFile(const char* path, void* buffer, size_t size);
bool tinyaiWriteFile(const char* path, const void* buffer, size_t size);

/**
 * Directory operations
 */
bool tinyaiCreateDir(const char* path);
bool tinyaiListDir(const char* path, char*** files, size_t* count);

/**
 * Path manipulation
 */
char* tinyaiJoinPath(const char* path1, const char* path2);
char* tinyaiGetDirName(const char* path);
char* tinyaiGetBaseName(const char* path);
```

### Configuration System

```c
/**
 * Configuration structure
 */
typedef struct {
    const char* key;
    TinyAIConfigType type;
    union {
        int int_value;
        float float_value;
        bool bool_value;
        char* string_value;
    } value;
} TinyAIConfig;

/**
 * Load configuration
 */
bool tinyaiLoadConfig(const char* path, TinyAIConfig** config, size_t* count);

/**
 * Get configuration value
 */
bool tinyaiGetConfigValue(const TinyAIConfig* config, size_t count, 
                         const char* key, void* value);

/**
 * Set configuration value
 */
bool tinyaiSetConfigValue(TinyAIConfig* config, size_t count, 
                         const char* key, const void* value);
```

## Model Components

### Tokenizer

```c
/**
 * Tokenizer configuration
 */
typedef struct {
    const char* vocabulary_path;  // Path to vocabulary file
    size_t max_tokens;           // Maximum number of tokens
    bool case_sensitive;         // Whether to be case sensitive
} TinyAITokenizerConfig;

/**
 * Create tokenizer
 */
TinyAITokenizer* tinyaiCreateTokenizer(const TinyAITokenizerConfig* config);

/**
 * Tokenize text
 */
bool tinyaiTokenize(TinyAITokenizer* tokenizer, const char* text, 
                   int* tokens, size_t* count);

/**
 * Detokenize tokens
 */
bool tinyaiDetokenize(TinyAITokenizer* tokenizer, const int* tokens, 
                     size_t count, char** text);
```

### Text Generation

```c
/**
 * Generation configuration
 */
typedef struct {
    int max_length;              // Maximum generation length
    float temperature;           // Sampling temperature
    int top_k;                   // Top-k sampling parameter
    float top_p;                 // Top-p sampling parameter
} TinyAIGenerationConfig;

/**
 * Generate text
 */
bool tinyaiGenerateText(TinyAIModel* model, const char* prompt,
                       const TinyAIGenerationConfig* config,
                       char** output);
```

## Optimization Components

### Memory Optimizer

```c
/**
 * Memory optimizer configuration
 */
typedef struct {
    bool enable_tensor_reuse;     // Whether to reuse tensor memory
    bool enable_in_place_ops;     // Whether to use in-place operations
    float memory_speed_tradeoff;  // Memory/speed tradeoff (0.0 to 1.0)
    size_t max_memory_budget;     // Maximum memory budget
} TinyAIMemoryOptimizerConfig;

/**
 * Create memory optimizer
 */
TinyAIMemoryOptimizer* tinyaiCreateMemoryOptimizer(
    const TinyAIMemoryOptimizerConfig* config);

/**
 * Optimize memory usage
 */
bool tinyaiOptimizeMemoryUsage(TinyAIMemoryOptimizer* optimizer,
                              TinyAIModel* model,
                              size_t memory_budget);
```

### Progressive Loader

```c
/**
 * Progressive loader configuration
 */
typedef struct {
    bool enable_prefetch;         // Whether to prefetch next layers
    int prefetch_window;          // Number of layers to prefetch
    bool enable_adaptive_window;  // Whether to adapt prefetch window
    size_t max_memory_usage;      // Maximum memory usage
} TinyAIProgressiveLoaderConfig;

/**
 * Create progressive loader
 */
TinyAIProgressiveLoader* tinyaiCreateProgressiveLoader(
    const TinyAIProgressiveLoaderConfig* config);

/**
 * Load layer
 */
bool tinyaiLoadLayer(TinyAIProgressiveLoader* loader, int layer_index);
```

### Layer Scheduler

```c
/**
 * Layer scheduler configuration
 */
typedef struct {
    bool enable_checkpointing;    // Whether to use checkpointing
    float memory_speed_tradeoff;  // Memory/speed tradeoff
    bool recompute_activations;   // Whether to recompute activations
    size_t max_activation_memory; // Maximum activation memory
} TinyAILayerSchedulerConfig;

/**
 * Create layer scheduler
 */
TinyAILayerScheduler* tinyaiCreateLayerScheduler(
    TinyAIModel* model,
    const TinyAILayerSchedulerConfig* config);

/**
 * Execute layer with optimization
 */
bool tinyaiExecuteLayerWithMemoryOptimization(
    TinyAILayerScheduler* scheduler,
    int layer_index,
    TinyAITensor* input,
    TinyAITensor* output);
```

## Utility Components

### Quantization

```c
/**
 * Quantization configuration
 */
typedef struct {
    int bits;                    // Number of bits (4 or 8)
    bool symmetric;              // Whether to use symmetric quantization
    float scale;                 // Quantization scale
    int zero_point;              // Quantization zero point
} TinyAIQuantizationConfig;

/**
 * Quantize tensor
 */
bool tinyaiQuantizeTensor(const TinyAITensor* input,
                         TinyAITensor* output,
                         const TinyAIQuantizationConfig* config);

/**
 * Dequantize tensor
 */
bool tinyaiDequantizeTensor(const TinyAITensor* input,
                           TinyAITensor* output);
```

### Sparse Operations

```c
/**
 * Sparse matrix configuration
 */
typedef struct {
    TinyAISparseFormat format;   // Sparse format (CSR, CSC, etc.)
    int block_size;              // Block size for blocked formats
    bool enable_simd;            // Whether to enable SIMD
} TinyAISparseConfig;

/**
 * Convert to sparse format
 */
bool tinyaiConvertToSparse(const TinyAITensor* dense,
                          TinyAITensor* sparse,
                          const TinyAISparseConfig* config);

/**
 * Sparse matrix multiplication
 */
bool tinyaiSparseMatMul(const TinyAITensor* a,
                       const TinyAITensor* b,
                       TinyAITensor* c);
```

## Error Handling

```c
/**
 * Error codes
 */
typedef enum {
    TINYAI_SUCCESS = 0,
    TINYAI_ERROR_MEMORY,
    TINYAI_ERROR_IO,
    TINYAI_ERROR_INVALID_ARGUMENT,
    TINYAI_ERROR_NOT_IMPLEMENTED,
    TINYAI_ERROR_RUNTIME
} TinyAIErrorCode;

/**
 * Get error message
 */
const char* tinyaiGetErrorMessage(TinyAIErrorCode code);

/**
 * Set error handler
 */
void tinyaiSetErrorHandler(TinyAIErrorHandler handler);
```

## Usage Examples

### Basic Model Usage

```c
// Initialize memory management
TinyAIMemoryConfig mem_config = {
    .initial_pool_size = 1024 * 1024 * 1024,  // 1GB
    .max_pool_size = 2 * 1024 * 1024 * 1024,  // 2GB
    .track_allocations = true,
    .enable_optimization = true
};
tinyaiInitMemory(&mem_config);

// Load model
TinyAIModel* model = tinyaiLoadModel("model.tinyai");

// Create tokenizer
TinyAITokenizerConfig tokenizer_config = {
    .vocabulary_path = "vocabulary.txt",
    .max_tokens = 50000,
    .case_sensitive = false
};
TinyAITokenizer* tokenizer = tinyaiCreateTokenizer(&tokenizer_config);

// Generate text
TinyAIGenerationConfig gen_config = {
    .max_length = 100,
    .temperature = 0.7f,
    .top_k = 50,
    .top_p = 0.9f
};
char* output;
tinyaiGenerateText(model, "Hello, ", &gen_config, &output);
```

### Memory Optimization

```c
// Create memory optimizer
TinyAIMemoryOptimizerConfig opt_config = {
    .enable_tensor_reuse = true,
    .enable_in_place_ops = true,
    .memory_speed_tradeoff = 0.5f,
    .max_memory_budget = 512 * 1024 * 1024  // 512MB
};
TinyAIMemoryOptimizer* optimizer = tinyaiCreateMemoryOptimizer(&opt_config);

// Optimize model
tinyaiOptimizeMemoryUsage(optimizer, model, 256 * 1024 * 1024);  // 256MB budget
```

### Progressive Loading

```c
// Create progressive loader
TinyAIProgressiveLoaderConfig loader_config = {
    .enable_prefetch = true,
    .prefetch_window = 3,
    .enable_adaptive_window = true,
    .max_memory_usage = 256 * 1024 * 1024  // 256MB
};
TinyAIProgressiveLoader* loader = tinyaiCreateProgressiveLoader(&loader_config);

// Load layers as needed
tinyaiLoadLayer(loader, 0);
tinyaiLoadLayer(loader, 1);
```

## Best Practices

1. **Memory Management**
   - Initialize memory management first
   - Set appropriate memory budgets
   - Monitor memory usage
   - Use memory optimization features

2. **Model Usage**
   - Load models once and reuse
   - Use appropriate batch sizes
   - Enable memory optimization
   - Monitor performance

3. **Error Handling**
   - Check return values
   - Handle errors appropriately
   - Clean up resources
   - Log errors for debugging

4. **Performance**
   - Use appropriate configurations
   - Enable optimizations
   - Monitor performance
   - Profile memory usage

### Memory-Efficient Tensor Operations

```c
/**
 * Tensor memory allocation strategy
 */
typedef enum {
    TINYAI_TENSOR_STATIC,    // Static allocation
    TINYAI_TENSOR_POOLED,    // Pooled allocation
    TINYAI_TENSOR_STREAMING  // Streaming allocation
} TinyAITensorAllocStrategy;

/**
 * Tensor memory configuration
 */
typedef struct {
    TinyAITensorAllocStrategy strategy;  // Allocation strategy
    size_t pool_size;                    // Pool size for pooled strategy
    size_t stream_buffer_size;           // Buffer size for streaming
    bool enable_in_place;                // Whether to enable in-place ops
} TinyAITensorMemoryConfig;

/**
 * Create memory-efficient tensor
 */
TinyAITensor* tinyaiCreateTensorWithMemoryConfig(
    const TinyAITensorMemoryConfig* config,
    int ndims,
    const int* dims,
    TinyAIDataType dtype);

/**
 * Perform in-place tensor operation
 */
bool tinyaiTensorInPlaceOp(
    TinyAITensor* tensor,
    TinyAITensorOp op,
    const void* params);

/**
 * Stream tensor data
 */
bool tinyaiStreamTensorData(
    TinyAITensor* tensor,
    const void* data,
    size_t offset,
    size_t size);

/**
 * Get tensor memory statistics
 */
TinyAITensorMemoryStats tinyaiGetTensorMemoryStats(
    const TinyAITensor* tensor);
```

### Performance Impact Assessment

```c
/**
 * Performance tracking configuration
 */
typedef struct {
    bool track_execution_time;   // Whether to track execution time
    bool track_memory_usage;     // Whether to track memory usage
    bool track_cpu_usage;        // Whether to track CPU usage
    bool track_cache_usage;      // Whether to track cache usage
    bool enable_optimization;    // Whether to enable optimization analysis
} TinyAIPerformanceConfig;

/**
 * Performance metrics
 */
typedef struct {
    double execution_time_ms;    // Execution time in milliseconds
    size_t memory_usage_bytes;   // Memory usage in bytes
    float cpu_usage_percent;     // CPU usage percentage
    size_t cache_misses;         // Number of cache misses
    size_t cache_hits;          // Number of cache hits
} TinyAIPerformanceMetrics;

/**
 * Optimization impact
 */
typedef struct {
    float speedup_factor;        // Speedup factor compared to baseline
    float memory_reduction;      // Memory reduction percentage
    float cpu_efficiency;        // CPU efficiency improvement
    char* recommendations;       // Optimization recommendations
} TinyAIOptimizationImpact;

/**
 * Create performance analysis context
 */
TinyAIPerformanceAnalysis* tinyaiCreatePerformanceAnalysis(
    const TinyAIPerformanceConfig* config);

/**
 * Record performance metrics
 */
bool tinyaiRecordMetrics(
    TinyAIPerformanceAnalysis* analysis,
    const TinyAIPerformanceMetrics* metrics);

/**
 * Analyze optimization impact
 */
bool tinyaiAnalyzeOptimizationImpact(
    TinyAIPerformanceAnalysis* analysis,
    const TinyAIPerformanceMetrics* baseline,
    const TinyAIPerformanceMetrics* current);

/**
 * Generate performance report
 */
bool tinyaiGeneratePerformanceReport(
    const TinyAIPerformanceAnalysis* analysis,
    const char* output_path);

/**
 * Get performance trend
 */
float tinyaiGetPerformanceTrend(
    const TinyAIPerformanceAnalysis* analysis);
``` 