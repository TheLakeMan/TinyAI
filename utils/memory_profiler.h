#ifndef TINYAI_MEMORY_PROFILER_H
#define TINYAI_MEMORY_PROFILER_H

#include <stdbool.h>
#include <stddef.h>
#include <time.h>

// Memory profile sample
typedef struct {
    size_t timestamp_ms;
    size_t total_memory;
    size_t active_memory;
    size_t peak_memory;
    size_t allocation_count;
    size_t free_count;
    size_t layer_count;
    bool  *layer_loaded;
} MemoryProfileSample;

// Memory profiler state
typedef struct {
    MemoryProfileSample            *samples;
    size_t                          sample_count;
    size_t                          max_samples;
    clock_t                         start_time;
    struct TinyAIProgressiveLoader *loader;
    struct TinyAIMemoryOptimizer   *optimizer;
} MemoryProfiler;

// Initialize memory profiler
MemoryProfiler *init_memory_profiler(const char *model_path, size_t memory_budget,
                                     bool enable_checkpointing, float memory_speed_tradeoff);

// Take a memory profile sample
void take_profile_sample(MemoryProfiler *profiler);

// Free memory profiler resources
void free_memory_profiler(MemoryProfiler *profiler);

// Print memory profile summary
void print_profile_summary(const MemoryProfiler *profiler);

// Generate memory profile report
void generate_profile_report(const MemoryProfiler *profiler, const char *output_path);

#endif // TINYAI_MEMORY_PROFILER_H