#include "tinyai.h"
#include "utils/layer_scheduler.h"
#include "utils/memory_optimizer.h"
#include "utils/progressive_loader.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Benchmark configuration
#define BENCHMARK_DURATION_SECONDS 60
#define SAMPLE_INTERVAL_MS 100
#define WARMUP_ITERATIONS 10
#define MAX_BATCH_SIZES 5
#define MAX_PROMPT_LENGTH 1024

// Performance metrics structure
typedef struct {
    double inference_time;
    double throughput;
    double memory_usage;
    double cpu_usage;
    double cache_hit_rate;
    size_t memory_operations;
    size_t tensor_reuses;
} PerformanceMetrics;

// Benchmark configuration
typedef struct {
    const char *model_path;
    const char *prompt;
    size_t      max_length;
    float       temperature;
    int         batch_sizes[MAX_BATCH_SIZES];
    int         num_batch_sizes;
    bool        enable_simd;
    bool        enable_cache_opt;
    bool        enable_memory_opt;
} BenchmarkConfig;

// Benchmark results
typedef struct {
    PerformanceMetrics metrics[MAX_BATCH_SIZES];
    double             total_duration;
    size_t             total_samples;
    size_t             total_tokens;
} BenchmarkResults;

// Initialize benchmark configuration
static void init_benchmark_config(BenchmarkConfig *config)
{
    config->model_path  = "models/example.tinyai";
    config->prompt      = "The quick brown fox jumps over the lazy dog.";
    config->max_length  = 100;
    config->temperature = 0.7f;

    // Test different batch sizes
    config->batch_sizes[0]  = 1;
    config->batch_sizes[1]  = 4;
    config->batch_sizes[2]  = 8;
    config->batch_sizes[3]  = 16;
    config->batch_sizes[4]  = 32;
    config->num_batch_sizes = 5;

    config->enable_simd       = true;
    config->enable_cache_opt  = true;
    config->enable_memory_opt = true;
}

// Run warmup iterations
static void run_warmup(TinyAIModel *model, const char *prompt,
                       const TinyAIGenerationConfig *gen_config)
{
    printf("Running warmup iterations...\n");
    char output[MAX_PROMPT_LENGTH];

    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        tinyaiGenerateText(model, prompt, gen_config, output);
    }
}

// Collect performance metrics
static void collect_metrics(TinyAIModel *model, PerformanceMetrics *metrics)
{
    // Get inference time and throughput
    TinyAIPerformanceMetrics perf_metrics = tinyaiGetPerformanceMetrics(model);
    metrics->inference_time               = perf_metrics.inference_time;
    metrics->throughput                   = perf_metrics.throughput;

    // Get memory usage
    TinyAIResourceUsage usage = tinyaiGetResourceUsage(model);
    metrics->memory_usage     = usage.memory_usage;
    metrics->cpu_usage        = usage.cpu_usage;

    // Get cache statistics
    TinyAICacheStats cache_stats = tinyaiGetCacheStats(model);
    metrics->cache_hit_rate      = cache_stats.hit_rate;

    // Get memory optimizer statistics
    TinyAIMemoryOptimizer *optimizer = tinyaiGetMemoryOptimizer(model);
    if (optimizer) {
        TinyAIMemoryStats mem_stats = tinyaiGetMemoryOptimizerStats(optimizer);
        metrics->memory_operations  = mem_stats.operations;
        metrics->tensor_reuses      = mem_stats.tensor_reuses;
    }
}

// Run benchmark for a specific batch size
static void run_benchmark_batch(TinyAIModel *model, const BenchmarkConfig *config, int batch_size,
                                PerformanceMetrics *metrics)
{
    // Configure generation
    TinyAIGenerationConfig gen_config = {.max_length  = config->max_length,
                                         .temperature = config->temperature,
                                         .batch_size  = batch_size};

    // Run warmup
    run_warmup(model, config->prompt, &gen_config);

    // Initialize metrics
    memset(metrics, 0, sizeof(PerformanceMetrics));
    size_t total_samples = 0;
    size_t total_tokens  = 0;

    // Run benchmark
    clock_t start_time = clock();
    char    output[MAX_PROMPT_LENGTH];

    while ((clock() - start_time) / CLOCKS_PER_SEC < BENCHMARK_DURATION_SECONDS) {
        // Generate text
        tinyaiGenerateText(model, config->prompt, &gen_config, output);

        // Collect metrics
        PerformanceMetrics sample;
        collect_metrics(model, &sample);

        // Accumulate metrics
        metrics->inference_time += sample.inference_time;
        metrics->throughput += sample.throughput;
        metrics->memory_usage += sample.memory_usage;
        metrics->cpu_usage += sample.cpu_usage;
        metrics->cache_hit_rate += sample.cache_hit_rate;
        metrics->memory_operations += sample.memory_operations;
        metrics->tensor_reuses += sample.tensor_reuses;

        total_samples++;
        total_tokens += strlen(output);

        // Sleep for sample interval
        struct timespec ts = {0, SAMPLE_INTERVAL_MS * 1000000};
        nanosleep(&ts, NULL);
    }

    // Calculate averages
    if (total_samples > 0) {
        metrics->inference_time /= total_samples;
        metrics->throughput /= total_samples;
        metrics->memory_usage /= total_samples;
        metrics->cpu_usage /= total_samples;
        metrics->cache_hit_rate /= total_samples;
    }
}

// Print benchmark results
static void print_results(const BenchmarkConfig *config, const BenchmarkResults *results)
{
    printf("\nPerformance Benchmark Results\n");
    printf("============================\n\n");

    printf("Model: %s\n", config->model_path);
    printf("Prompt: %s\n", config->prompt);
    printf("Duration: %.2f seconds\n", results->total_duration);
    printf("Total Samples: %zu\n", results->total_samples);
    printf("Total Tokens: %zu\n\n", results->total_tokens);

    printf("Batch Size Analysis\n");
    printf("------------------\n");
    for (int i = 0; i < config->num_batch_sizes; i++) {
        const PerformanceMetrics *metrics = &results->metrics[i];
        printf("\nBatch Size: %d\n", config->batch_sizes[i]);
        printf("  Inference Time: %.2f ms\n", metrics->inference_time);
        printf("  Throughput: %.2f tokens/s\n", metrics->throughput);
        printf("  Memory Usage: %.2f%%\n", metrics->memory_usage);
        printf("  CPU Usage: %.2f%%\n", metrics->cpu_usage);
        printf("  Cache Hit Rate: %.2f%%\n", metrics->cache_hit_rate * 100);
        printf("  Memory Operations: %zu\n", metrics->memory_operations);
        printf("  Tensor Reuses: %zu\n", metrics->tensor_reuses);
    }
}

// Main benchmark function
int main(int argc, char **argv)
{
    // Initialize benchmark configuration
    BenchmarkConfig config;
    init_benchmark_config(&config);

    // Initialize memory system
    TinyAIMemoryConfig mem_config = {.initial_pool_size   = 1024 * 1024 * 1024,
                                     .max_pool_size       = 2 * 1024 * 1024 * 1024,
                                     .track_allocations   = true,
                                     .enable_optimization = true};
    tinyaiInitMemory(&mem_config);

    // Load model
    TinyAIModel *model = tinyaiLoadModel(config.model_path);
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", config.model_path);
        return 1;
    }

    // Configure optimizations
    if (config.enable_simd) {
        tinyaiEnableSIMD(model, TINYAI_SIMD_OP_MATRIX_MUL, true);
        tinyaiEnableSIMD(model, TINYAI_SIMD_OP_CONV, true);
        tinyaiSetSIMDOptimizationLevel(model, 3);
    }

    if (config.enable_cache_opt) {
        TinyAICacheConfig cache_config = {.block_size        = 64,
                                          .cache_line_size   = 64,
                                          .enable_prefetch   = true,
                                          .prefetch_distance = 2};
        tinyaiConfigureCacheOptimization(model, &cache_config);
    }

    if (config.enable_memory_opt) {
        TinyAIMemoryOptimizerConfig opt_config = {.enable_tensor_reuse   = true,
                                                  .enable_in_place_ops   = true,
                                                  .memory_speed_tradeoff = 0.7f,
                                                  .max_memory_budget     = 768 * 1024 * 1024};
        TinyAIMemoryOptimizer      *optimizer  = tinyaiCreateMemoryOptimizer(&opt_config);
        tinyaiSetMemoryOptimizer(model, optimizer);
    }

    // Run benchmarks
    BenchmarkResults results;
    memset(&results, 0, sizeof(BenchmarkResults));

    clock_t start_time = clock();

    for (int i = 0; i < config.num_batch_sizes; i++) {
        printf("Running benchmark for batch size %d...\n", config.batch_sizes[i]);
        run_benchmark_batch(model, &config, config.batch_sizes[i], &results.metrics[i]);
    }

    results.total_duration = (double)(clock() - start_time) / CLOCKS_PER_SEC;

    // Print results
    print_results(&config, &results);

    // Cleanup
    tinyaiFreeModel(model);
    tinyaiFreeMemory();

    return 0;
}