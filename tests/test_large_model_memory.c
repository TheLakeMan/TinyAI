#include "utils/benchmark.h"
#include "utils/memory_optimizer.h"
#include "utils/memory_pool.h"
#include "utils/progressive_loader.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Test configuration
#define TEST_MODEL_PATH "data/large_model.bin"
#define TEST_MEMORY_BUDGET (500 * 1024 * 1024) // 500MB
#define TEST_LAYER_COUNT 24

// Test memory optimizer creation and initialization
static void test_optimizer_creation(void)
{
    TinyAIMemoryOptimizerConfig config = {.max_memory_budget     = TEST_MEMORY_BUDGET,
                                          .enable_checkpointing  = true,
                                          .memory_speed_tradeoff = 0.5f,
                                          .recompute_activations = true,
                                          .max_activation_memory = TEST_MEMORY_BUDGET / 2};

    TinyAIMemoryOptimizer *optimizer = tinyaiCreateMemoryOptimizer(&config);
    assert(optimizer != NULL);
    assert(optimizer->config.max_memory_budget == TEST_MEMORY_BUDGET);
    assert(optimizer->config.enable_checkpointing == true);

    tinyaiFreeMemoryOptimizer(optimizer);
    printf("✅ test_optimizer_creation passed\n");
}

// Test memory-aware execution planning
static void test_execution_planning(void)
{
    TinyAIMemoryOptimizerConfig config = {.max_memory_budget     = TEST_MEMORY_BUDGET,
                                          .enable_checkpointing  = true,
                                          .memory_speed_tradeoff = 0.5f,
                                          .recompute_activations = true,
                                          .max_activation_memory = TEST_MEMORY_BUDGET / 2};

    TinyAIMemoryOptimizer *optimizer = tinyaiCreateMemoryOptimizer(&config);
    assert(optimizer != NULL);

    // Create a test model with known memory requirements
    TinyAIModel *model = tinyaiCreateTestModel(TEST_LAYER_COUNT);
    assert(model != NULL);

    // Generate memory-optimized execution plan
    TinyAIExecutionPlan *plan = tinyaiCreateMemoryOptimizedExecutionPlan(optimizer, model);
    assert(plan != NULL);
    assert(plan->layer_count == TEST_LAYER_COUNT);

    // Verify memory estimates
    TinyAIMemoryEstimate estimate = tinyaiEstimateMemoryUsage(optimizer, model);
    assert(estimate.peak_memory <= TEST_MEMORY_BUDGET);
    assert(estimate.average_memory <= TEST_MEMORY_BUDGET * 0.8);

    tinyaiFreeExecutionPlan(plan);
    tinyaiFreeModel(model);
    tinyaiFreeMemoryOptimizer(optimizer);
    printf("✅ test_execution_planning passed\n");
}

// Test activation checkpointing
static void test_checkpointing(void)
{
    TinyAIMemoryOptimizerConfig config = {.max_memory_budget     = TEST_MEMORY_BUDGET,
                                          .enable_checkpointing  = true,
                                          .memory_speed_tradeoff = 0.5f,
                                          .recompute_activations = true,
                                          .max_activation_memory = TEST_MEMORY_BUDGET / 2};

    TinyAIMemoryOptimizer *optimizer = tinyaiCreateMemoryOptimizer(&config);
    assert(optimizer != NULL);

    // Create test activation
    TinyAITensor *activation = tinyaiCreateTestTensor(1000, 1000);
    assert(activation != NULL);

    // Create checkpoint
    TinyAICheckpoint *checkpoint = tinyaiCreateActivationCheckpoint(optimizer, activation);
    assert(checkpoint != NULL);

    // Restore from checkpoint
    TinyAITensor *restored = tinyaiRestoreFromCheckpoint(optimizer, checkpoint);
    assert(restored != NULL);
    assert(tinyaiCompareTensors(activation, restored) == true);

    tinyaiFreeTensor(restored);
    tinyaiFreeCheckpoint(checkpoint);
    tinyaiFreeTensor(activation);
    tinyaiFreeMemoryOptimizer(optimizer);
    printf("✅ test_checkpointing passed\n");
}

// Test memory/speed tradeoff
static void test_memory_speed_tradeoff(void)
{
    TinyAIMemoryOptimizerConfig config = {.max_memory_budget     = TEST_MEMORY_BUDGET,
                                          .enable_checkpointing  = true,
                                          .memory_speed_tradeoff = 0.5f,
                                          .recompute_activations = true,
                                          .max_activation_memory = TEST_MEMORY_BUDGET / 2};

    TinyAIMemoryOptimizer *optimizer = tinyaiCreateMemoryOptimizer(&config);
    assert(optimizer != NULL);

    // Test different tradeoff values
    for (float tradeoff = 0.0f; tradeoff <= 1.0f; tradeoff += 0.2f) {
        tinyaiSetMemorySpeedTradeoff(optimizer, tradeoff);

        TinyAIModel *model = tinyaiCreateTestModel(TEST_LAYER_COUNT);
        assert(model != NULL);

        TinyAIMemoryEstimate estimate = tinyaiEstimateMemoryUsage(optimizer, model);
        assert(estimate.peak_memory <= TEST_MEMORY_BUDGET);

        // Higher tradeoff should result in lower memory usage
        if (tradeoff > 0.0f) {
            assert(estimate.peak_memory < TEST_MEMORY_BUDGET);
        }

        tinyaiFreeModel(model);
    }

    tinyaiFreeMemoryOptimizer(optimizer);
    printf("✅ test_memory_speed_tradeoff passed\n");
}

// Test tensor reuse strategies
static void test_tensor_reuse(void)
{
    TinyAIMemoryOptimizerConfig config = {.max_memory_budget     = TEST_MEMORY_BUDGET,
                                          .enable_checkpointing  = true,
                                          .memory_speed_tradeoff = 0.5f,
                                          .recompute_activations = true,
                                          .max_activation_memory = TEST_MEMORY_BUDGET / 2};

    TinyAIMemoryOptimizer *optimizer = tinyaiCreateMemoryOptimizer(&config);
    assert(optimizer != NULL);

    // Create test tensors
    TinyAITensor *input  = tinyaiCreateTestTensor(1000, 1000);
    TinyAITensor *output = tinyaiCreateTestTensor(1000, 1000);
    assert(input != NULL && output != NULL);

    // Enable in-place operations
    assert(tinyaiEnableInPlaceOperations(optimizer, true) == true);

    // Perform operation with tensor reuse
    assert(tinyaiExecuteWithTensorReuse(optimizer, input, output) == true);

    // Verify memory usage is reduced
    TinyAIMemoryStats stats = tinyaiGetMemoryOptimizerStats(optimizer);
    assert(stats.tensor_reuse_count > 0);
    assert(stats.memory_saved > 0);

    tinyaiFreeTensor(output);
    tinyaiFreeTensor(input);
    tinyaiFreeMemoryOptimizer(optimizer);
    printf("✅ test_tensor_reuse passed\n");
}

// Performance benchmark
static void benchmark_memory_optimization(void)
{
    TinyAIMemoryOptimizerConfig config = {.max_memory_budget     = TEST_MEMORY_BUDGET,
                                          .enable_checkpointing  = true,
                                          .memory_speed_tradeoff = 0.5f,
                                          .recompute_activations = true,
                                          .max_activation_memory = TEST_MEMORY_BUDGET / 2};

    TinyAIMemoryOptimizer *optimizer = tinyaiCreateMemoryOptimizer(&config);
    assert(optimizer != NULL);

    // Benchmark memory optimization
    TinyAIBenchmarkResult result = tinyaiBenchmarkOperation(
        "Memory Optimization", 100,
        (TinyAIBenchmarkOperation){.setup     = NULL,
                                   .operation = (void (*)(void *))tinyaiOptimizeMemoryUsage,
                                   .teardown  = NULL,
                                   .context   = optimizer});

    printf("Memory Optimization Performance:\n");
    printf("  Average time: %.2f ms\n", result.average_time_ms);
    printf("  Min time: %.2f ms\n", result.min_time_ms);
    printf("  Max time: %.2f ms\n", result.max_time_ms);
    printf("  Standard deviation: %.2f ms\n", result.std_dev_ms);

    tinyaiFreeMemoryOptimizer(optimizer);
    printf("✅ benchmark_memory_optimization completed\n");
}

int main(void)
{
    printf("Starting Large Model Memory Optimization Tests...\n");

    test_optimizer_creation();
    test_execution_planning();
    test_checkpointing();
    test_memory_speed_tradeoff();
    test_tensor_reuse();
    benchmark_memory_optimization();

    printf("All Large Model Memory Optimization Tests Passed!\n");
    return 0;
}