#include "../utils/benchmark.h"
#include "../utils/memory_pool.h"
#include "../utils/progressive_loader.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Test configuration
#define TEST_MODEL_PATH "data/test_model.bin"
#define TEST_MEMORY_BUDGET (100 * 1024 * 1024) // 100MB
#define TEST_LAYER_COUNT 12

// Test progressive loader creation and initialization
static void test_loader_creation(void)
{
    TinyAIProgressiveLoaderConfig config = {.max_memory_budget      = TEST_MEMORY_BUDGET,
                                            .enable_layer_unloading = true,
                                            .priority_strategy  = TINYAI_PRIORITY_ACCESS_PATTERN,
                                            .prefetch_threshold = 0.7f};

    TinyAIProgressiveLoader *loader = tinyaiCreateProgressiveLoader(TEST_MODEL_PATH, &config);
    assert(loader != NULL);

    tinyaiFreeProgressiveLoader(loader);
    printf("✅ test_loader_creation passed\n");
}

// Test layer loading and unloading
static void test_layer_management(void)
{
    TinyAIProgressiveLoaderConfig config = {.max_memory_budget      = TEST_MEMORY_BUDGET,
                                            .enable_layer_unloading = true,
                                            .priority_strategy  = TINYAI_PRIORITY_ACCESS_PATTERN,
                                            .prefetch_threshold = 0.7f};

    TinyAIProgressiveLoader *loader = tinyaiCreateProgressiveLoader(TEST_MODEL_PATH, &config);
    assert(loader != NULL);

    // Test loading layers
    for (int i = 0; i < TEST_LAYER_COUNT; i++) {
        assert(tinyaiLoadModelLayer(loader, i) == true);
        assert(tinyaiIsLayerLoaded(loader, i) == true);
    }

    // Test unloading layers
    for (int i = 0; i < TEST_LAYER_COUNT; i++) {
        assert(tinyaiUnloadModelLayer(loader, i) == true);
        assert(tinyaiIsLayerLoaded(loader, i) == false);
    }

    tinyaiFreeProgressiveLoader(loader);
    printf("✅ test_layer_management passed\n");
}

// Test memory budget management
static void test_memory_budget(void)
{
    TinyAIProgressiveLoaderConfig config = {.max_memory_budget      = TEST_MEMORY_BUDGET,
                                            .enable_layer_unloading = true,
                                            .priority_strategy  = TINYAI_PRIORITY_ACCESS_PATTERN,
                                            .prefetch_threshold = 0.7f};

    TinyAIProgressiveLoader *loader = tinyaiCreateProgressiveLoader(TEST_MODEL_PATH, &config);
    assert(loader != NULL);

    // Load layers until we hit memory budget
    size_t total_memory = 0;
    for (int i = 0; i < TEST_LAYER_COUNT; i++) {
        if (tinyaiLoadModelLayer(loader, i)) {
            TinyAIMemoryStats stats = tinyaiGetProgressiveLoaderMemoryStats(loader);
            total_memory            = stats.total_allocated;
            if (total_memory > TEST_MEMORY_BUDGET) {
                // Should have unloaded some layers
                assert(tinyaiGetProgressiveLoaderMemoryStats(loader).total_allocated <=
                       TEST_MEMORY_BUDGET);
                break;
            }
        }
    }

    tinyaiFreeProgressiveLoader(loader);
    printf("✅ test_memory_budget passed\n");
}

// Test layer dependency management
static void test_layer_dependencies(void)
{
    TinyAIProgressiveLoaderConfig config = {.max_memory_budget      = TEST_MEMORY_BUDGET,
                                            .enable_layer_unloading = true,
                                            .priority_strategy  = TINYAI_PRIORITY_ACCESS_PATTERN,
                                            .prefetch_threshold = 0.7f};

    TinyAIProgressiveLoader *loader = tinyaiCreateProgressiveLoader(TEST_MODEL_PATH, &config);
    assert(loader != NULL);

    // Set up dependencies: layer 1 depends on layer 0
    assert(tinyaiAddLayerDependency(loader, 1, 0) == true);

    // Try to unload layer 0 while layer 1 is loaded
    assert(tinyaiLoadModelLayer(loader, 1) == true);
    assert(tinyaiUnloadModelLayer(loader, 0) == false); // Should fail due to dependency

    // Unload layer 1 first, then layer 0
    assert(tinyaiUnloadModelLayer(loader, 1) == true);
    assert(tinyaiUnloadModelLayer(loader, 0) == true);

    tinyaiFreeProgressiveLoader(loader);
    printf("✅ test_layer_dependencies passed\n");
}

// Test access pattern tracking
static void test_access_patterns(void)
{
    TinyAIProgressiveLoaderConfig config = {.max_memory_budget      = TEST_MEMORY_BUDGET,
                                            .enable_layer_unloading = true,
                                            .priority_strategy  = TINYAI_PRIORITY_ACCESS_PATTERN,
                                            .prefetch_threshold = 0.7f};

    TinyAIProgressiveLoader *loader = tinyaiCreateProgressiveLoader(TEST_MODEL_PATH, &config);
    assert(loader != NULL);

    // Simulate access patterns
    for (int i = 0; i < 100; i++) {
        int layer = i % TEST_LAYER_COUNT;
        tinyaiUpdateLayerAccessStats(loader, layer);
    }

    tinyaiFreeProgressiveLoader(loader);
    printf("✅ test_access_patterns passed\n");
}

// Test prefetching
static void test_prefetching(void)
{
    TinyAIProgressiveLoaderConfig config = {.max_memory_budget      = TEST_MEMORY_BUDGET,
                                            .enable_layer_unloading = true,
                                            .priority_strategy  = TINYAI_PRIORITY_ACCESS_PATTERN,
                                            .prefetch_threshold = 0.7f};

    TinyAIProgressiveLoader *loader = tinyaiCreateProgressiveLoader(TEST_MODEL_PATH, &config);
    assert(loader != NULL);

    // Load current layer
    assert(tinyaiLoadModelLayer(loader, 0) == true);

    // Get prefetch recommendations
    int  prefetch_count;
    int *prefetch_layers = tinyaiGetLayersToPreload(loader, 0, &prefetch_count);
    assert(prefetch_layers != NULL);
    assert(prefetch_count > 0);

    free(prefetch_layers);
    tinyaiFreeProgressiveLoader(loader);
    printf("✅ test_prefetching passed\n");
}

// Performance benchmark
static void benchmark_progressive_loading(void)
{
    TinyAIProgressiveLoaderConfig config = {.max_memory_budget      = TEST_MEMORY_BUDGET,
                                            .enable_layer_unloading = true,
                                            .priority_strategy  = TINYAI_PRIORITY_ACCESS_PATTERN,
                                            .prefetch_threshold = 0.7f};

    TinyAIProgressiveLoader *loader = tinyaiCreateProgressiveLoader(TEST_MODEL_PATH, &config);
    assert(loader != NULL);

    // Benchmark layer loading time
    TinyAIBenchmarkOperation op = {.setup     = NULL,
                                   .operation = (void (*)(void *))tinyaiLoadModelLayer,
                                   .teardown  = NULL,
                                   .context   = loader};

    TinyAIBenchmarkResult result = tinyaiBenchmarkOperation("Layer Loading", 100, op);

    printf("Layer Loading Performance:\n");
    printf("  Average time: %.2f ms\n", result.average_time_ms);
    printf("  Min time: %.2f ms\n", result.min_time_ms);
    printf("  Max time: %.2f ms\n", result.max_time_ms);
    printf("  Standard deviation: %.2f ms\n", result.std_dev_ms);

    tinyaiFreeProgressiveLoader(loader);
    printf("✅ benchmark_progressive_loading completed\n");
}

int main(void)
{
    printf("Starting Progressive Loading Tests...\n");

    test_loader_creation();
    test_layer_management();
    test_memory_budget();
    test_layer_dependencies();
    test_access_patterns();
    test_prefetching();
    benchmark_progressive_loading();

    printf("All Progressive Loading Tests Passed!\n");
    return 0;
}