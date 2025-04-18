#ifndef TINYAI_PERFORMANCE_IMPACT_H
#define TINYAI_PERFORMANCE_IMPACT_H

#include "tinyai.h"
#include <stdbool.h>
#include <stddef.h>

// Performance impact configuration
typedef struct {
    bool   track_execution_time;  // Track execution time
    bool   track_memory_usage;    // Track memory usage
    bool   track_cpu_usage;       // Track CPU usage
    bool   track_cache_usage;     // Track cache usage
    bool   analyze_optimizations; // Analyze optimization impact
    size_t sample_interval_ms;    // Sampling interval in milliseconds
    size_t analysis_window_ms;    // Analysis window in milliseconds
} TinyAIPerformanceConfig;

// Performance metrics
typedef struct {
    double execution_time;  // Execution time in milliseconds
    size_t memory_usage;    // Memory usage in bytes
    double cpu_usage;       // CPU usage percentage
    size_t cache_misses;    // Number of cache misses
    size_t cache_hits;      // Number of cache hits
    double cache_hit_ratio; // Cache hit ratio
} TinyAIPerformanceMetrics;

// Optimization impact
typedef struct {
    double speedup_factor;      // Speedup compared to baseline
    double memory_reduction;    // Memory reduction percentage
    double cpu_efficiency;      // CPU efficiency improvement
    double cache_improvement;   // Cache performance improvement
    bool   is_beneficial;       // Whether optimization is beneficial
    char   recommendation[256]; // Optimization recommendation
} TinyAIOptimizationImpact;

// Performance analysis context
typedef struct {
    TinyAIPerformanceConfig  config;
    TinyAIPerformanceMetrics baseline;
    TinyAIPerformanceMetrics current;
    TinyAIOptimizationImpact impact;
    uint64_t                 start_time;
    uint64_t                 last_sample_time;
} TinyAIPerformanceAnalysis;

// Create performance analysis context
TinyAIPerformanceAnalysis *tinyaiCreatePerformanceAnalysis(const TinyAIPerformanceConfig *config);

// Free performance analysis context
void tinyaiFreePerformanceAnalysis(TinyAIPerformanceAnalysis *analysis);

// Record performance metrics
void tinyaiRecordMetrics(TinyAIPerformanceAnalysis      *analysis,
                         const TinyAIPerformanceMetrics *metrics);

// Take a performance sample
void tinyaiTakePerformanceSample(TinyAIPerformanceAnalysis *analysis);

// Get current performance metrics
TinyAIPerformanceMetrics tinyaiGetPerformanceMetrics(const TinyAIPerformanceAnalysis *analysis);

// Analyze optimization impact
void tinyaiAnalyzeOptimizationImpact(TinyAIPerformanceAnalysis *analysis);

// Get optimization impact
TinyAIOptimizationImpact tinyaiGetOptimizationImpact(const TinyAIPerformanceAnalysis *analysis);

// Generate performance report
void tinyaiGeneratePerformanceReport(const TinyAIPerformanceAnalysis *analysis,
                                     const char                      *filename);

// Get performance trend
double tinyaiGetPerformanceTrend(const TinyAIPerformanceAnalysis *analysis);

// Reset performance analysis
void tinyaiResetPerformanceAnalysis(TinyAIPerformanceAnalysis *analysis);

// Enable/disable performance analysis
void tinyaiEnablePerformanceAnalysis(TinyAIPerformanceAnalysis *analysis, bool enable);

// Set performance analysis configuration
void tinyaiSetPerformanceConfig(TinyAIPerformanceAnalysis     *analysis,
                                const TinyAIPerformanceConfig *config);

#endif // TINYAI_PERFORMANCE_IMPACT_H