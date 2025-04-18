#include "../utils/benchmark.h"
#include "../utils/memory_optimizer.h"
#include "../utils/progressive_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Benchmark configuration
#define BENCHMARK_ITERATIONS 100
#define BENCHMARK_MODEL_PATH "data/benchmark_model.bin"
#define BENCHMARK_MEMORY_BUDGET (1000 * 1024 * 1024) // 1GB

// Memory usage tracking structure
typedef struct {
    size_t peak_memory;
    size_t average_memory;
    size_t total_allocations;
    size_t total_frees;
    double allocation_time;
    double free_time;
} MemoryBenchmarkStats;

// Run memory benchmark for a specific configuration
static MemoryBenchmarkStats run_memory_benchmark(const char *model_path, size_t memory_budget,
                                                 bool  enable_checkpointing,
                                                 float memory_speed_tradeoff)
{
    MemoryBenchmarkStats stats = {0};
    clock_t              start_time, end_time;

    // Create memory optimizer
    TinyAIMemoryOptimizerConfig config = {.max_memory_budget     = memory_budget,
                                          .enable_checkpointing  = enable_checkpointing,
                                          .memory_speed_tradeoff = memory_speed_tradeoff,
                                          .recompute_activations = true,
                                          .max_activation_memory = memory_budget / 2};

    TinyAIMemoryOptimizer *optimizer = tinyaiCreateMemoryOptimizer(&config);
    if (!optimizer) {
        fprintf(stderr, "Failed to create memory optimizer\n");
        return stats;
    }

    // Create progressive loader
    TinyAIProgressiveLoaderConfig loader_config = {.max_memory_budget      = memory_budget,
                                                   .enable_layer_unloading = true,
                                                   .priority_strategy =
                                                       TINYAI_PRIORITY_ACCESS_PATTERN,
                                                   .prefetch_threshold = 0.7f};

    TinyAIProgressiveLoader *loader = tinyaiCreateProgressiveLoader(model_path, &loader_config);
    if (!loader) {
        fprintf(stderr, "Failed to create progressive loader\n");
        tinyaiFreeMemoryOptimizer(optimizer);
        return stats;
    }

    // Run benchmark iterations
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        // Measure allocation time
        start_time = clock();

        // Load model layers
        for (int layer = 0; layer < loader->layer_count; layer++) {
            if (!tinyaiLoadModelLayer(loader, layer)) {
                fprintf(stderr, "Failed to load layer %d\n", layer);
                continue;
            }
        }

        end_time = clock();
        stats.allocation_time += (double)(end_time - start_time) / CLOCKS_PER_SEC;

        // Get memory stats
        TinyAIMemoryStats memory_stats = tinyaiGetProgressiveLoaderMemoryStats(loader);
        stats.peak_memory              = (memory_stats.peak_allocated > stats.peak_memory)
                                             ? memory_stats.peak_allocated
                                             : stats.peak_memory;
        stats.average_memory += memory_stats.total_allocated;
        stats.total_allocations += memory_stats.allocation_count;
        stats.total_frees += memory_stats.free_count;

        // Measure free time
        start_time = clock();

        // Unload model layers
        for (int layer = 0; layer < loader->layer_count; layer++) {
            if (!tinyaiUnloadModelLayer(loader, layer)) {
                fprintf(stderr, "Failed to unload layer %d\n", layer);
                continue;
            }
        }

        end_time = clock();
        stats.free_time += (double)(end_time - start_time) / CLOCKS_PER_SEC;
    }

    // Calculate averages
    stats.average_memory /= BENCHMARK_ITERATIONS;
    stats.allocation_time /= BENCHMARK_ITERATIONS;
    stats.free_time /= BENCHMARK_ITERATIONS;

    // Cleanup
    tinyaiFreeProgressiveLoader(loader);
    tinyaiFreeMemoryOptimizer(optimizer);

    return stats;
}

// Print benchmark results
static void print_benchmark_results(const char *config_name, const MemoryBenchmarkStats *stats)
{
    printf("\nBenchmark Results for %s:\n", config_name);
    printf("  Peak Memory Usage: %.2f MB\n", stats->peak_memory / (1024.0 * 1024.0));
    printf("  Average Memory Usage: %.2f MB\n", stats->average_memory / (1024.0 * 1024.0));
    printf("  Total Allocations: %zu\n", stats->total_allocations);
    printf("  Total Frees: %zu\n", stats->total_frees);
    printf("  Average Allocation Time: %.4f seconds\n", stats->allocation_time);
    printf("  Average Free Time: %.4f seconds\n", stats->free_time);
    printf("  Memory Operations per Second: %.2f\n",
           (stats->total_allocations + stats->total_frees) /
               (stats->allocation_time + stats->free_time));
}

int main(int argc, char *argv[])
{
    printf("Starting Memory Benchmark...\n");

    // Test different configurations
    struct {
        const char *name;
        size_t      memory_budget;
        bool        enable_checkpointing;
        float       memory_speed_tradeoff;
    } configs[] = {{"Default", BENCHMARK_MEMORY_BUDGET, true, 0.5f},
                   {"Memory-Optimized", BENCHMARK_MEMORY_BUDGET, true, 0.8f},
                   {"Speed-Optimized", BENCHMARK_MEMORY_BUDGET, false, 0.2f},
                   {"Minimal Memory", BENCHMARK_MEMORY_BUDGET / 2, true, 1.0f}};

    for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
        MemoryBenchmarkStats stats =
            run_memory_benchmark(BENCHMARK_MODEL_PATH, configs[i].memory_budget,
                                 configs[i].enable_checkpointing, configs[i].memory_speed_tradeoff);
        print_benchmark_results(configs[i].name, &stats);
    }

    printf("\nMemory Benchmark Completed!\n");
    return 0;
}