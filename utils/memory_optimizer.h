#ifndef TINYAI_MEMORY_OPTIMIZER_H
#define TINYAI_MEMORY_OPTIMIZER_H

#include <stdbool.h>
#include <stddef.h>

// Memory optimizer configuration
typedef struct {
    size_t max_memory_budget;     // Maximum memory to use in bytes
    bool   enable_checkpointing;  // Whether to use activation checkpointing
    float  memory_speed_tradeoff; // 0.0 (prioritize memory) to 1.0 (prioritize speed)
    bool   recompute_activations; // Whether to recompute rather than store activations
    size_t max_activation_memory; // Maximum memory for activations
} TinyAIMemoryOptimizerConfig;

// Memory statistics
typedef struct {
    size_t total_allocated;    // Total memory allocated
    size_t current_allocated;  // Current memory in use
    size_t peak_allocated;     // Peak memory usage
    size_t allocation_count;   // Number of allocations
    size_t free_count;         // Number of frees
    size_t tensor_reuse_count; // Number of tensor reuses
    size_t memory_saved;       // Memory saved through optimizations
} TinyAIMemoryStats;

// Memory optimizer handle
typedef struct TinyAIMemoryOptimizer TinyAIMemoryOptimizer;

// Create a memory optimizer
TinyAIMemoryOptimizer *tinyaiCreateMemoryOptimizer(const TinyAIMemoryOptimizerConfig *config);

// Free memory optimizer
void tinyaiFreeMemoryOptimizer(TinyAIMemoryOptimizer *optimizer);

// Get memory statistics
TinyAIMemoryStats tinyaiGetMemoryOptimizerStats(const TinyAIMemoryOptimizer *optimizer);

// Set memory/speed tradeoff
void tinyaiSetMemorySpeedTradeoff(TinyAIMemoryOptimizer *optimizer, float tradeoff);

// Enable in-place operations
bool tinyaiEnableInPlaceOperations(TinyAIMemoryOptimizer *optimizer, bool enable);

// Execute with tensor reuse
bool tinyaiExecuteWithTensorReuse(TinyAIMemoryOptimizer *optimizer, void *input, void *output);

// Optimize memory usage
bool tinyaiOptimizeMemoryUsage(TinyAIMemoryOptimizer *optimizer);

#endif // TINYAI_MEMORY_OPTIMIZER_H