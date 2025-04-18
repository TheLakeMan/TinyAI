#include "../utils/memory_profiler.h"
#include "../utils/benchmark.h"
#include "../utils/memory_optimizer.h"
#include "../utils/progressive_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Profiler configuration
#define PROFILER_SAMPLE_INTERVAL_MS 100
#define PROFILER_DURATION_SECONDS 60
#define PROFILER_MODEL_PATH "data/profiler_model.bin"
#define PROFILER_MEMORY_BUDGET (1000 * 1024 * 1024) // 1GB

// Memory profile data structure
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
    MemoryProfileSample     *samples;
    size_t                   sample_count;
    size_t                   max_samples;
    clock_t                  start_time;
    TinyAIProgressiveLoader *loader;
    TinyAIMemoryOptimizer   *optimizer;
} MemoryProfiler;

// Initialize memory profiler
MemoryProfiler *init_memory_profiler(const char *model_path, size_t memory_budget,
                                     bool enable_checkpointing, float memory_speed_tradeoff)
{
    MemoryProfiler *profiler = malloc(sizeof(MemoryProfiler));
    if (!profiler)
        return NULL;

    // Calculate maximum number of samples
    profiler->max_samples = (PROFILER_DURATION_SECONDS * 1000) / PROFILER_SAMPLE_INTERVAL_MS;
    profiler->samples     = malloc(sizeof(MemoryProfileSample) * profiler->max_samples);
    if (!profiler->samples) {
        free(profiler);
        return NULL;
    }

    profiler->sample_count = 0;
    profiler->start_time   = clock();

    // Create memory optimizer
    TinyAIMemoryOptimizerConfig optimizer_config = {.max_memory_budget     = memory_budget,
                                                    .enable_checkpointing  = enable_checkpointing,
                                                    .memory_speed_tradeoff = memory_speed_tradeoff,
                                                    .recompute_activations = true,
                                                    .max_activation_memory = memory_budget / 2};

    profiler->optimizer = tinyaiCreateMemoryOptimizer(&optimizer_config);
    if (!profiler->optimizer) {
        free(profiler->samples);
        free(profiler);
        return NULL;
    }

    // Create progressive loader
    TinyAIProgressiveLoaderConfig loader_config = {.max_memory_budget      = memory_budget,
                                                   .enable_layer_unloading = true,
                                                   .priority_strategy =
                                                       TINYAI_PRIORITY_ACCESS_PATTERN,
                                                   .prefetch_threshold = 0.7f};

    profiler->loader = tinyaiCreateProgressiveLoader(model_path, &loader_config);
    if (!profiler->loader) {
        tinyaiFreeMemoryOptimizer(profiler->optimizer);
        free(profiler->samples);
        free(profiler);
        return NULL;
    }

    return profiler;
}

// Take a memory profile sample
void take_profile_sample(MemoryProfiler *profiler)
{
    if (profiler->sample_count >= profiler->max_samples)
        return;

    clock_t current_time = clock();
    size_t  elapsed_ms   = (current_time - profiler->start_time) * 1000 / CLOCKS_PER_SEC;

    MemoryProfileSample *sample = &profiler->samples[profiler->sample_count];
    sample->timestamp_ms        = elapsed_ms;

    // Get memory stats
    TinyAIMemoryStats stats  = tinyaiGetProgressiveLoaderMemoryStats(profiler->loader);
    sample->total_memory     = stats.total_allocated;
    sample->active_memory    = stats.current_allocated;
    sample->peak_memory      = stats.peak_allocated;
    sample->allocation_count = stats.allocation_count;
    sample->free_count       = stats.free_count;

    // Get layer loading status
    sample->layer_count  = profiler->loader->layer_count;
    sample->layer_loaded = malloc(sizeof(bool) * sample->layer_count);
    if (sample->layer_loaded) {
        for (size_t i = 0; i < sample->layer_count; i++) {
            sample->layer_loaded[i] = tinyaiIsLayerLoaded(profiler->loader, i);
        }
    }

    profiler->sample_count++;
}

// Free memory profiler resources
void free_memory_profiler(MemoryProfiler *profiler)
{
    if (!profiler)
        return;

    // Free layer loading status arrays
    for (size_t i = 0; i < profiler->sample_count; i++) {
        free(profiler->samples[i].layer_loaded);
    }

    // Free main resources
    tinyaiFreeProgressiveLoader(profiler->loader);
    tinyaiFreeMemoryOptimizer(profiler->optimizer);
    free(profiler->samples);
    free(profiler);
}

// Print memory profile summary
void print_profile_summary(const MemoryProfiler *profiler)
{
    if (!profiler || profiler->sample_count == 0)
        return;

    printf("\nMemory Profile Summary:\n");
    printf("  Duration: %.2f seconds\n",
           profiler->samples[profiler->sample_count - 1].timestamp_ms / 1000.0);
    printf("  Sample Count: %zu\n", profiler->sample_count);

    // Calculate statistics
    size_t max_peak_memory   = 0;
    size_t total_allocations = 0;
    size_t total_frees       = 0;
    double avg_memory_usage  = 0.0;

    for (size_t i = 0; i < profiler->sample_count; i++) {
        const MemoryProfileSample *sample = &profiler->samples[i];
        max_peak_memory =
            (sample->peak_memory > max_peak_memory) ? sample->peak_memory : max_peak_memory;
        total_allocations += sample->allocation_count;
        total_frees += sample->free_count;
        avg_memory_usage += sample->total_memory;
    }

    avg_memory_usage /= profiler->sample_count;

    printf("  Maximum Peak Memory: %.2f MB\n", max_peak_memory / (1024.0 * 1024.0));
    printf("  Average Memory Usage: %.2f MB\n", avg_memory_usage / (1024.0 * 1024.0));
    printf("  Total Allocations: %zu\n", total_allocations);
    printf("  Total Frees: %zu\n", total_frees);
    printf("  Memory Operations per Second: %.2f\n",
           (total_allocations + total_frees) /
               (profiler->samples[profiler->sample_count - 1].timestamp_ms / 1000.0));
}

// Generate memory profile report
void generate_profile_report(const MemoryProfiler *profiler, const char *output_path)
{
    if (!profiler || !output_path)
        return;

    FILE *report = fopen(output_path, "w");
    if (!report) {
        fprintf(stderr, "Failed to open report file: %s\n", output_path);
        return;
    }

    // Write CSV header
    fprintf(report, "Timestamp (ms),Total Memory (bytes),Active Memory (bytes),"
                    "Peak Memory (bytes),Allocations,Frees,Loaded Layers\n");

    // Write sample data
    for (size_t i = 0; i < profiler->sample_count; i++) {
        const MemoryProfileSample *sample = &profiler->samples[i];

        // Count loaded layers
        size_t loaded_layers = 0;
        if (sample->layer_loaded) {
            for (size_t j = 0; j < sample->layer_count; j++) {
                if (sample->layer_loaded[j])
                    loaded_layers++;
            }
        }

        fprintf(report, "%zu,%zu,%zu,%zu,%zu,%zu,%zu\n", sample->timestamp_ms, sample->total_memory,
                sample->active_memory, sample->peak_memory, sample->allocation_count,
                sample->free_count, loaded_layers);
    }

    fclose(report);
    printf("Profile report written to: %s\n", output_path);
}

int main(int argc, char *argv[])
{
    printf("Starting Memory Profiler...\n");

    // Initialize profiler
    MemoryProfiler *profiler = init_memory_profiler(PROFILER_MODEL_PATH, PROFILER_MEMORY_BUDGET,
                                                    true, // enable_checkpointing
                                                    0.5f  // memory_speed_tradeoff
    );

    if (!profiler) {
        fprintf(stderr, "Failed to initialize memory profiler\n");
        return 1;
    }

    // Run profiling
    clock_t start_time = clock();
    while ((clock() - start_time) * 1000 / CLOCKS_PER_SEC < PROFILER_DURATION_SECONDS * 1000) {
        // Simulate model operations
        for (int layer = 0; layer < profiler->loader->layer_count; layer++) {
            if (!tinyaiLoadModelLayer(profiler->loader, layer)) {
                fprintf(stderr, "Failed to load layer %d\n", layer);
                continue;
            }

            // Take profile sample
            take_profile_sample(profiler);

            // Simulate some computation time
            struct timespec sleep_time = {0, PROFILER_SAMPLE_INTERVAL_MS * 1000000};
            nanosleep(&sleep_time, NULL);

            if (!tinyaiUnloadModelLayer(profiler->loader, layer)) {
                fprintf(stderr, "Failed to unload layer %d\n", layer);
                continue;
            }
        }
    }

    // Print summary and generate report
    print_profile_summary(profiler);
    generate_profile_report(profiler, "memory_profile.csv");

    // Cleanup
    free_memory_profiler(profiler);

    printf("\nMemory Profiling Completed!\n");
    return 0;
}