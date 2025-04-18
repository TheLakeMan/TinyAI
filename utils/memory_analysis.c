#include "memory_analysis.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Default configuration
static const TinyAIMemoryAnalysisConfig DEFAULT_CONFIG = {.track_allocations   = true,
                                                          .track_deallocations = true,
                                                          .track_peak_usage    = true,
                                                          .analyze_patterns    = true,
                                                          .sample_interval_ms  = 100,
                                                          .analysis_window_ms  = 1000};

// Get current timestamp in milliseconds
static uint64_t get_timestamp_ms()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
}

// Create memory analysis context
TinyAIMemoryAnalysis *tinyaiCreateMemoryAnalysis(const TinyAIMemoryAnalysisConfig *config)
{
    TinyAIMemoryAnalysis *analysis = malloc(sizeof(TinyAIMemoryAnalysis));
    if (!analysis)
        return NULL;

    // Initialize with default or provided config
    if (config) {
        analysis->config = *config;
    }
    else {
        analysis->config = DEFAULT_CONFIG;
    }

    // Initialize allocation tracking
    analysis->max_allocations = 1024; // Initial capacity
    analysis->allocations     = malloc(sizeof(TinyAIMemoryAllocation) * analysis->max_allocations);
    if (!analysis->allocations) {
        free(analysis);
        return NULL;
    }

    // Initialize pattern tracking
    memset(&analysis->pattern, 0, sizeof(TinyAIMemoryPattern));
    analysis->num_allocations  = 0;
    analysis->start_time       = get_timestamp_ms();
    analysis->last_sample_time = analysis->start_time;

    return analysis;
}

// Free memory analysis context
void tinyaiFreeMemoryAnalysis(TinyAIMemoryAnalysis *analysis)
{
    if (!analysis)
        return;
    free(analysis->allocations);
    free(analysis);
}

// Record memory allocation
void tinyaiRecordAllocation(TinyAIMemoryAnalysis *analysis, void *address, size_t size,
                            const char *file, int line, const char *function)
{
    if (!analysis || !analysis->config.track_allocations)
        return;

    // Check if we need to resize the allocations array
    if (analysis->num_allocations >= analysis->max_allocations) {
        size_t                  new_size = analysis->max_allocations * 2;
        TinyAIMemoryAllocation *new_allocations =
            realloc(analysis->allocations, sizeof(TinyAIMemoryAllocation) * new_size);
        if (!new_allocations)
            return;
        analysis->allocations     = new_allocations;
        analysis->max_allocations = new_size;
    }

    // Record the allocation
    TinyAIMemoryAllocation *alloc = &analysis->allocations[analysis->num_allocations++];
    alloc->address                = address;
    alloc->size                   = size;
    alloc->file                   = file;
    alloc->line                   = line;
    alloc->function               = function;
    alloc->timestamp              = get_timestamp_ms();
    alloc->is_freed               = false;

    // Update pattern statistics
    analysis->pattern.total_allocations++;
    analysis->pattern.current_usage += size;
    if (analysis->pattern.current_usage > analysis->pattern.peak_usage) {
        analysis->pattern.peak_usage = analysis->pattern.current_usage;
    }
}

// Record memory deallocation
void tinyaiRecordDeallocation(TinyAIMemoryAnalysis *analysis, void *address)
{
    if (!analysis || !analysis->config.track_deallocations)
        return;

    // Find the allocation
    for (size_t i = 0; i < analysis->num_allocations; i++) {
        TinyAIMemoryAllocation *alloc = &analysis->allocations[i];
        if (alloc->address == address && !alloc->is_freed) {
            alloc->is_freed = true;
            analysis->pattern.total_freed++;
            analysis->pattern.current_usage -= alloc->size;
            break;
        }
    }
}

// Take a memory usage sample
void tinyaiTakeMemorySample(TinyAIMemoryAnalysis *analysis)
{
    if (!analysis)
        return;

    uint64_t current_time = get_timestamp_ms();
    uint64_t elapsed_ms   = current_time - analysis->last_sample_time;

    if (elapsed_ms >= analysis->config.sample_interval_ms) {
        // Update allocation and deallocation rates
        double elapsed_seconds              = elapsed_ms / 1000.0;
        analysis->pattern.allocation_rate   = analysis->pattern.total_allocations / elapsed_seconds;
        analysis->pattern.deallocation_rate = analysis->pattern.total_freed / elapsed_seconds;

        // Update average lifetime
        uint64_t total_lifetime = 0;
        size_t   count          = 0;
        for (size_t i = 0; i < analysis->num_allocations; i++) {
            if (analysis->allocations[i].is_freed) {
                total_lifetime += analysis->allocations[i].timestamp - analysis->start_time;
                count++;
            }
        }
        if (count > 0) {
            analysis->pattern.average_lifetime = total_lifetime / (double)count;
        }

        analysis->last_sample_time = current_time;
    }
}

// Get current memory pattern
TinyAIMemoryPattern tinyaiGetMemoryPattern(const TinyAIMemoryAnalysis *analysis)
{
    if (!analysis) {
        TinyAIMemoryPattern empty = {0};
        return empty;
    }
    return analysis->pattern;
}

// Analyze memory usage patterns
void tinyaiAnalyzeMemoryPatterns(TinyAIMemoryAnalysis *analysis)
{
    if (!analysis || !analysis->config.analyze_patterns)
        return;

    // Calculate fragmentation
    size_t total_allocated = 0;
    size_t total_free      = 0;
    size_t largest_free    = 0;

    for (size_t i = 0; i < analysis->num_allocations; i++) {
        if (!analysis->allocations[i].is_freed) {
            total_allocated += analysis->allocations[i].size;
        }
    }

    // This is a simplified fragmentation calculation
    // In a real implementation, you'd need to track free blocks
    analysis->pattern.fragmentation =
        (total_free > 0) ? (100 * (total_free - largest_free)) / total_free : 0;
}

// Generate memory usage report
void tinyaiGenerateMemoryReport(const TinyAIMemoryAnalysis *analysis, const char *filename)
{
    if (!analysis || !filename)
        return;

    FILE *file = fopen(filename, "w");
    if (!file)
        return;

    fprintf(file, "Memory Usage Report\n");
    fprintf(file, "==================\n\n");

    fprintf(file, "Total Allocations: %zu\n", analysis->pattern.total_allocations);
    fprintf(file, "Total Freed: %zu\n", analysis->pattern.total_freed);
    fprintf(file, "Current Usage: %zu bytes\n", analysis->pattern.current_usage);
    fprintf(file, "Peak Usage: %zu bytes\n", analysis->pattern.peak_usage);
    fprintf(file, "Fragmentation: %.2f%%\n", analysis->pattern.fragmentation);
    fprintf(file, "Allocation Rate: %.2f/s\n", analysis->pattern.allocation_rate);
    fprintf(file, "Deallocation Rate: %.2f/s\n", analysis->pattern.deallocation_rate);
    fprintf(file, "Average Lifetime: %.2f ms\n", analysis->pattern.average_lifetime);

    // List unfreed allocations
    fprintf(file, "\nUnfreed Allocations:\n");
    for (size_t i = 0; i < analysis->num_allocations; i++) {
        if (!analysis->allocations[i].is_freed) {
            fprintf(file, "  %p: %zu bytes at %s:%d in %s\n", analysis->allocations[i].address,
                    analysis->allocations[i].size, analysis->allocations[i].file,
                    analysis->allocations[i].line, analysis->allocations[i].function);
        }
    }

    fclose(file);
}

// Get memory fragmentation
double tinyaiGetMemoryFragmentation(const TinyAIMemoryAnalysis *analysis)
{
    if (!analysis)
        return 0.0;
    return analysis->pattern.fragmentation;
}

// Get memory usage trend
double tinyaiGetMemoryUsageTrend(const TinyAIMemoryAnalysis *analysis)
{
    if (!analysis || analysis->num_allocations < 2)
        return 0.0;

    // Calculate trend based on recent samples
    size_t recent_samples = 10;
    size_t start_idx      = (analysis->num_allocations > recent_samples)
                                ? analysis->num_allocations - recent_samples
                                : 0;

    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    for (size_t i = start_idx; i < analysis->num_allocations; i++) {
        double x = i - start_idx;
        double y = analysis->allocations[i].size;
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }

    size_t n     = analysis->num_allocations - start_idx;
    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    return slope;
}

// Get allocation hotspots
void tinyaiGetAllocationHotspots(const TinyAIMemoryAnalysis *analysis,
                                 TinyAIMemoryAllocation **hotspots, size_t *num_hotspots)
{
    if (!analysis || !hotspots || !num_hotspots)
        return;

    // Sort allocations by size
    TinyAIMemoryAllocation *sorted =
        malloc(sizeof(TinyAIMemoryAllocation) * analysis->num_allocations);
    if (!sorted)
        return;

    memcpy(sorted, analysis->allocations,
           sizeof(TinyAIMemoryAllocation) * analysis->num_allocations);
    qsort(sorted, analysis->num_allocations, sizeof(TinyAIMemoryAllocation),
          (int (*)(const void *, const void *))compare_allocations);

    // Return top 10 hotspots
    *num_hotspots = (analysis->num_allocations > 10) ? 10 : analysis->num_allocations;
    *hotspots     = malloc(sizeof(TinyAIMemoryAllocation) * (*num_hotspots));
    if (*hotspots) {
        memcpy(*hotspots, sorted, sizeof(TinyAIMemoryAllocation) * (*num_hotspots));
    }

    free(sorted);
}

// Get memory leak candidates
void tinyaiGetMemoryLeakCandidates(const TinyAIMemoryAnalysis *analysis,
                                   TinyAIMemoryAllocation **leaks, size_t *num_leaks)
{
    if (!analysis || !leaks || !num_leaks)
        return;

    // Count unfreed allocations
    size_t count = 0;
    for (size_t i = 0; i < analysis->num_allocations; i++) {
        if (!analysis->allocations[i].is_freed)
            count++;
    }

    *num_leaks = count;
    *leaks     = malloc(sizeof(TinyAIMemoryAllocation) * count);
    if (*leaks) {
        size_t idx = 0;
        for (size_t i = 0; i < analysis->num_allocations; i++) {
            if (!analysis->allocations[i].is_freed) {
                (*leaks)[idx++] = analysis->allocations[i];
            }
        }
    }
}

// Reset memory analysis
void tinyaiResetMemoryAnalysis(TinyAIMemoryAnalysis *analysis)
{
    if (!analysis)
        return;

    analysis->num_allocations = 0;
    memset(&analysis->pattern, 0, sizeof(TinyAIMemoryPattern));
    analysis->start_time       = get_timestamp_ms();
    analysis->last_sample_time = analysis->start_time;
}

// Enable/disable memory analysis
void tinyaiEnableMemoryAnalysis(TinyAIMemoryAnalysis *analysis, bool enable)
{
    if (!analysis)
        return;
    analysis->config.track_allocations   = enable;
    analysis->config.track_deallocations = enable;
}

// Set memory analysis configuration
void tinyaiSetMemoryAnalysisConfig(TinyAIMemoryAnalysis             *analysis,
                                   const TinyAIMemoryAnalysisConfig *config)
{
    if (!analysis || !config)
        return;
    analysis->config = *config;
}

// Comparison function for sorting allocations
static int compare_allocations(const TinyAIMemoryAllocation *a, const TinyAIMemoryAllocation *b)
{
    return (a->size < b->size) - (a->size > b->size);
}