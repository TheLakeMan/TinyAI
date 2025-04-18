#ifndef TINYAI_MEMORY_ANALYSIS_H
#define TINYAI_MEMORY_ANALYSIS_H

#include "tinyai.h"
#include <stdbool.h>
#include <stddef.h>

// Memory analysis configuration
typedef struct {
    bool   track_allocations;   // Enable allocation tracking
    bool   track_deallocations; // Enable deallocation tracking
    bool   track_peak_usage;    // Track peak memory usage
    bool   analyze_patterns;    // Analyze memory usage patterns
    size_t sample_interval_ms;  // Sampling interval in milliseconds
    size_t analysis_window_ms;  // Analysis window in milliseconds
} TinyAIMemoryAnalysisConfig;

// Memory allocation record
typedef struct {
    void       *address;   // Allocated memory address
    size_t      size;      // Allocation size
    const char *file;      // Source file
    int         line;      // Source line
    const char *function;  // Function name
    uint64_t    timestamp; // Allocation timestamp
    bool        is_freed;  // Whether the allocation was freed
} TinyAIMemoryAllocation;

// Memory usage pattern
typedef struct {
    size_t total_allocations; // Total number of allocations
    size_t total_freed;       // Total number of freed allocations
    size_t peak_usage;        // Peak memory usage in bytes
    size_t current_usage;     // Current memory usage in bytes
    size_t fragmentation;     // Memory fragmentation percentage
    double allocation_rate;   // Allocations per second
    double deallocation_rate; // Deallocations per second
    double average_lifetime;  // Average allocation lifetime in milliseconds
} TinyAIMemoryPattern;

// Memory analysis context
typedef struct {
    TinyAIMemoryAnalysisConfig config;
    TinyAIMemoryAllocation    *allocations;
    size_t                     max_allocations;
    size_t                     num_allocations;
    TinyAIMemoryPattern        pattern;
    uint64_t                   start_time;
    uint64_t                   last_sample_time;
} TinyAIMemoryAnalysis;

// Create memory analysis context
TinyAIMemoryAnalysis *tinyaiCreateMemoryAnalysis(const TinyAIMemoryAnalysisConfig *config);

// Free memory analysis context
void tinyaiFreeMemoryAnalysis(TinyAIMemoryAnalysis *analysis);

// Record memory allocation
void tinyaiRecordAllocation(TinyAIMemoryAnalysis *analysis, void *address, size_t size,
                            const char *file, int line, const char *function);

// Record memory deallocation
void tinyaiRecordDeallocation(TinyAIMemoryAnalysis *analysis, void *address);

// Take a memory usage sample
void tinyaiTakeMemorySample(TinyAIMemoryAnalysis *analysis);

// Get current memory pattern
TinyAIMemoryPattern tinyaiGetMemoryPattern(const TinyAIMemoryAnalysis *analysis);

// Analyze memory usage patterns
void tinyaiAnalyzeMemoryPatterns(TinyAIMemoryAnalysis *analysis);

// Generate memory usage report
void tinyaiGenerateMemoryReport(const TinyAIMemoryAnalysis *analysis, const char *filename);

// Get memory fragmentation
double tinyaiGetMemoryFragmentation(const TinyAIMemoryAnalysis *analysis);

// Get memory usage trend
double tinyaiGetMemoryUsageTrend(const TinyAIMemoryAnalysis *analysis);

// Get allocation hotspots
void tinyaiGetAllocationHotspots(const TinyAIMemoryAnalysis *analysis,
                                 TinyAIMemoryAllocation **hotspots, size_t *num_hotspots);

// Get memory leak candidates
void tinyaiGetMemoryLeakCandidates(const TinyAIMemoryAnalysis *analysis,
                                   TinyAIMemoryAllocation **leaks, size_t *num_leaks);

// Reset memory analysis
void tinyaiResetMemoryAnalysis(TinyAIMemoryAnalysis *analysis);

// Enable/disable memory analysis
void tinyaiEnableMemoryAnalysis(TinyAIMemoryAnalysis *analysis, bool enable);

// Set memory analysis configuration
void tinyaiSetMemoryAnalysisConfig(TinyAIMemoryAnalysis             *analysis,
                                   const TinyAIMemoryAnalysisConfig *config);

#endif // TINYAI_MEMORY_ANALYSIS_H