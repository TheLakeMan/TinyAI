/**
 * @file progressive_loader_test.c
 * @brief Test program for the progressive model loader
 *
 * This file contains tests for the progressive loader functionality,
 * demonstrating how to use the loader for managing memory when working
 * with large models.
 */

#include "progressive_loader.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Mock model data for testing
#define TEST_MODEL_SIZE (1024 * 1024 * 100) // 100MB mock model
#define LAYER_COUNT 10
#define LAYER_SIZE (TEST_MODEL_SIZE / LAYER_COUNT)

// Mock model file
typedef struct {
    size_t total_size;
    size_t layer_count;
    size_t layer_sizes[LAYER_COUNT];
    char  *data;
    bool   is_open;
} MockModelFile;

// Mock memory mapped model for testing
typedef struct {
    const char    *file_path;
    void          *mapped_data;
    size_t         size;
    MockModelFile *mock_file;
} MockMappedModel;

// Forward declarations for mock functions
MockModelFile   *create_mock_model_file(const char *path);
void             free_mock_model_file(MockModelFile *model);
MockMappedModel *mock_map_file(const char *path);
void             mock_unmap_file(MockMappedModel *mapped);
void             print_memory_stats(const TinyAIMemoryStats *stats);

// Global mock file for testing
MockModelFile *g_mock_model = NULL;

/**
 * Test creating and initializing a progressive loader
 */
void test_create_progressive_loader()
{
    printf("=== Testing Progressive Loader Creation ===\n");

    // Create default configuration
    TinyAIProgressiveLoaderConfig config = tinyaiCreateDefaultProgressiveLoaderConfig();

    // Modify configuration for testing
    config.max_memory_budget      = 30 * 1024 * 1024; // 30MB memory budget
    config.enable_layer_unloading = true;
    config.priority_strategy      = TINYAI_PRIORITY_LRU;
    config.max_prefetch_layers    = 2;

    // Create the loader with mock file
    const char *mock_path = "mock_model.bin";
    g_mock_model          = create_mock_model_file(mock_path);

    // Create the progressive loader
    TinyAIProgressiveLoader *loader = tinyaiCreateProgressiveLoader(mock_path, &config);
    assert(loader != NULL && "Failed to create progressive loader");

    // Get initial memory statistics
    TinyAIMemoryStats stats = tinyaiGetProgressiveLoaderMemoryStats(loader);
    print_memory_stats(&stats);

    // Verify initial state
    assert(stats.totalModelSize == TEST_MODEL_SIZE);
    assert(stats.totalLayerCount == LAYER_COUNT);
    assert(stats.loadedLayerCount == 0);

    // Clean up
    tinyaiFreeProgressiveLoader(loader);
    printf("Progressive loader creation test passed!\n\n");
}

/**
 * Test loading and unloading layers
 */
void test_load_unload_layers()
{
    printf("=== Testing Layer Loading and Unloading ===\n");

    // Create configuration with small memory budget to force unloading
    TinyAIProgressiveLoaderConfig config = tinyaiCreateDefaultProgressiveLoaderConfig();
    config.max_memory_budget             = LAYER_SIZE * 3; // Only enough for 3 layers
    config.enable_layer_unloading        = true;
    config.priority_strategy             = TINYAI_PRIORITY_LRU;

    // Create the loader
    const char              *mock_path = "mock_model.bin";
    TinyAIProgressiveLoader *loader    = tinyaiCreateProgressiveLoader(mock_path, &config);
    assert(loader != NULL);

    // Load first layer
    printf("Loading layer 0...\n");
    bool success = tinyaiLoadModelLayer(loader, 0);
    assert(success && "Failed to load layer 0");

    // Check layer state
    TinyAILayerState state = tinyaiGetLayerState(loader, 0);
    assert(state == TINYAI_LAYER_LOADED && "Layer 0 should be loaded");

    // Get memory stats after loading first layer
    TinyAIMemoryStats stats = tinyaiGetProgressiveLoaderMemoryStats(loader);
    print_memory_stats(&stats);
    assert(stats.loadedLayerCount == 1);

    // Load more layers to test automatic unloading
    printf("Loading layers 1-5 (should trigger automatic unloading)...\n");
    for (int i = 1; i <= 5; i++) {
        success = tinyaiLoadModelLayer(loader, i);
        assert(success && "Failed to load layer");
        tinyaiUpdateLayerAccessStats(loader, i);

        // Print current memory stats
        stats = tinyaiGetProgressiveLoaderMemoryStats(loader);
        printf("After loading layer %d:\n", i);
        print_memory_stats(&stats);

        // Check that we're not exceeding memory budget
        assert(stats.currentMemoryUsage <= stats.memoryBudget);
    }

    // Verify that some layers were unloaded to make room
    int loaded_count = 0;
    for (int i = 0; i < LAYER_COUNT; i++) {
        if (tinyaiGetLayerState(loader, i) == TINYAI_LAYER_LOADED) {
            loaded_count++;
        }
    }

    printf("Loaded layer count: %d (should be <= 3)\n", loaded_count);
    assert(loaded_count <= 3 && "Too many layers loaded, unloading isn't working");

    // Test explicit unloading
    for (int i = 0; i < LAYER_COUNT; i++) {
        if (tinyaiGetLayerState(loader, i) == TINYAI_LAYER_LOADED) {
            printf("Explicitly unloading layer %d...\n", i);
            success = tinyaiUnloadModelLayer(loader, i);
            assert(success && "Failed to unload layer");
        }
    }

    // Verify all layers are unloaded
    stats = tinyaiGetProgressiveLoaderMemoryStats(loader);
    print_memory_stats(&stats);
    assert(stats.loadedLayerCount == 0 && "Not all layers were unloaded");

    // Clean up
    tinyaiFreeProgressiveLoader(loader);
    printf("Layer loading/unloading test passed!\n\n");
}

/**
 * Test layer dependencies
 */
void test_layer_dependencies()
{
    printf("=== Testing Layer Dependencies ===\n");

    // Create configuration
    TinyAIProgressiveLoaderConfig config = tinyaiCreateDefaultProgressiveLoaderConfig();
    config.max_memory_budget             = TEST_MODEL_SIZE; // Large enough for all layers
    config.enable_dependency_tracking    = true;

    // Create the loader
    const char              *mock_path = "mock_model.bin";
    TinyAIProgressiveLoader *loader    = tinyaiCreateProgressiveLoader(mock_path, &config);
    assert(loader != NULL);

    // Add dependencies: layer 1 depends on layer 0
    bool success = tinyaiAddLayerDependency(loader, 1, 0);
    assert(success && "Failed to add dependency");

    // Add more dependencies
    tinyaiAddLayerDependency(loader, 2, 1);
    tinyaiAddLayerDependency(loader, 3, 2);

    // Check if layers can be unloaded (should be false due to dependencies)
    printf("Loading layer 0...\n");
    tinyaiLoadModelLayer(loader, 0);

    printf("Loading layer 1 (depends on layer 0)...\n");
    tinyaiLoadModelLayer(loader, 1);

    printf("Loading layer 2 (depends on layer 1)...\n");
    tinyaiLoadModelLayer(loader, 2);

    // Check dependency chain
    printf("Checking if base layers can be unloaded (should be false)...\n");
    bool can_unload = tinyaiCanUnloadLayer(loader, 0);
    printf("Can unload layer 0: %s (expected: false)\n", can_unload ? "true" : "false");
    assert(!can_unload && "Should not be able to unload layer 0 due to dependencies");

    // Unload in correct order
    printf("Unloading layers in correct order...\n");
    tinyaiUnloadModelLayer(loader, 2);
    tinyaiUnloadModelLayer(loader, 1);

    // Now layer 0 should be unloadable
    can_unload = tinyaiCanUnloadLayer(loader, 0);
    printf("Can unload layer 0 after dependent layers unloaded: %s (expected: true)\n",
           can_unload ? "true" : "false");
    assert(can_unload && "Should be able to unload layer 0 now");

    // Clean up
    tinyaiFreeProgressiveLoader(loader);
    printf("Layer dependencies test passed!\n\n");
}

/**
 * Test preloading layers
 */
void test_preload_layers()
{
    printf("=== Testing Layer Preloading ===\n");

    // Create configuration
    TinyAIProgressiveLoaderConfig config = tinyaiCreateDefaultProgressiveLoaderConfig();
    config.max_memory_budget             = TEST_MODEL_SIZE; // Large enough for all layers
    config.max_prefetch_layers           = 3;

    // Create the loader
    const char              *mock_path = "mock_model.bin";
    TinyAIProgressiveLoader *loader    = tinyaiCreateProgressiveLoader(mock_path, &config);
    assert(loader != NULL);

    // Preload specific layers
    int layers_to_preload[] = {2, 4, 6};
    printf("Preloading layers 2, 4, and 6...\n");
    bool success = tinyaiPreloadLayers(loader, layers_to_preload, 3);
    assert(success && "Failed to preload layers");

    // Check that the layers were loaded
    for (int i = 0; i < 3; i++) {
        int              layer = layers_to_preload[i];
        TinyAILayerState state = tinyaiGetLayerState(loader, layer);
        printf("Layer %d state: %d (expected: %d LOADED)\n", layer, state, TINYAI_LAYER_LOADED);
        assert(state == TINYAI_LAYER_LOADED && "Layer should be loaded");
    }

    // Test predictive preloading based on access patterns
    printf("Testing predictive preloading...\n");

    // Simulate sequential access pattern
    for (int i = 0; i < 3; i++) {
        tinyaiUpdateLayerAccessStats(loader, i);
    }

    // Get layers to preload based on pattern
    int  count       = 0;
    int *next_layers = tinyaiGetLayersToPreload(loader, 2, &count);
    assert(next_layers != NULL && "Failed to get layers to preload");

    printf("Predicted next layers to load after layer 2: ");
    for (int i = 0; i < count; i++) {
        printf("%d ", next_layers[i]);
    }
    printf("\n");

    // With sequential access, should predict next sequential layers
    assert(count > 0 && "Should predict at least one layer to preload");
    free(next_layers);

    // Clean up
    tinyaiFreeProgressiveLoader(loader);
    printf("Layer preloading test passed!\n\n");
}

/**
 * Test memory budget constraints
 */
void test_memory_budget()
{
    printf("=== Testing Memory Budget Constraints ===\n");

    // Create configuration with limited memory budget
    TinyAIProgressiveLoaderConfig config = tinyaiCreateDefaultProgressiveLoaderConfig();
    config.max_memory_budget             = LAYER_SIZE * 2; // Only enough for 2 layers
    config.enable_layer_unloading        = true;

    // Create the loader
    const char              *mock_path = "mock_model.bin";
    TinyAIProgressiveLoader *loader    = tinyaiCreateProgressiveLoader(mock_path, &config);
    assert(loader != NULL);

    // Get initial memory stats
    TinyAIMemoryStats stats = tinyaiGetProgressiveLoaderMemoryStats(loader);
    print_memory_stats(&stats);

    // Load layers until we hit the memory budget
    printf("Loading layers until we hit memory budget...\n");
    int loaded_count = 0;
    for (int i = 0; i < LAYER_COUNT; i++) {
        bool success = tinyaiLoadModelLayer(loader, i);
        if (success) {
            loaded_count++;
            printf("Successfully loaded layer %d\n", i);
        }
        else {
            printf("Failed to load layer %d (expected if memory budget reached)\n", i);
            break;
        }

        // Check current memory stats
        stats = tinyaiGetProgressiveLoaderMemoryStats(loader);
        print_memory_stats(&stats);
    }

    // Should have loaded at most 2 layers due to budget
    printf("Loaded %d layers (expected <= 2)\n", loaded_count);
    assert(loaded_count <= 2 && "Loaded too many layers despite budget constraint");

    // Update memory budget and try loading more
    printf("Increasing memory budget...\n");
    bool success = tinyaiSetProgressiveLoaderMemoryBudget(loader, LAYER_SIZE * 4);
    assert(success && "Failed to update memory budget");

    // Check updated memory stats
    stats = tinyaiGetProgressiveLoaderMemoryStats(loader);
    print_memory_stats(&stats);

    // Try loading more layers with increased budget
    printf("Loading more layers with increased budget...\n");
    for (int i = 0; i < LAYER_COUNT; i++) {
        if (tinyaiGetLayerState(loader, i) != TINYAI_LAYER_LOADED) {
            success = tinyaiLoadModelLayer(loader, i);
            if (success) {
                printf("Successfully loaded layer %d with increased budget\n", i);
            }

            if (i < 4) {
                assert(success && "Should be able to load layer with increased budget");
            }

            // Only try loading a few more to avoid exceeding the new budget
            if (i >= 4) {
                break;
            }
        }
    }

    // Clean up
    tinyaiFreeProgressiveLoader(loader);
    printf("Memory budget test passed!\n\n");
}

/**
 * Main test program
 */
int main()
{
    printf("=== Progressive Loader Tests ===\n\n");

    test_create_progressive_loader();
    test_load_unload_layers();
    test_layer_dependencies();
    test_preload_layers();
    test_memory_budget();

    // Free global mock model
    if (g_mock_model) {
        free_mock_model_file(g_mock_model);
        g_mock_model = NULL;
    }

    printf("All progressive loader tests passed!\n");
    return 0;
}

// ===== MOCK IMPLEMENTATIONS FOR TESTING =====

/**
 * Print memory statistics
 */
void print_memory_stats(const TinyAIMemoryStats *stats)
{
    printf("Memory Stats:\n");
    printf("  Total Model Size:      %zu bytes (%.2f MB)\n", stats->totalModelSize,
           stats->totalModelSize / (1024.0 * 1024.0));
    printf("  Current Memory Usage:  %zu bytes (%.2f MB)\n", stats->currentMemoryUsage,
           stats->currentMemoryUsage / (1024.0 * 1024.0));
    printf("  Peak Memory Usage:     %zu bytes (%.2f MB)\n", stats->peakMemoryUsage,
           stats->peakMemoryUsage / (1024.0 * 1024.0));
    printf("  Memory Budget:         %zu bytes (%.2f MB)\n", stats->memoryBudget,
           stats->memoryBudget / (1024.0 * 1024.0));
    printf("  Total Layers:          %d\n", stats->totalLayerCount);
    printf("  Loaded Layers:         %d\n", stats->loadedLayerCount);
    printf("  Memory Utilization:    %.2f%%\n", stats->memoryUtilization * 100.0);
    printf("  Average Load Time:     %.2f ms\n", stats->averageLoadTime);
}

/**
 * Create a mock model file for testing
 */
MockModelFile *create_mock_model_file(const char *path)
{
    MockModelFile *model = (MockModelFile *)malloc(sizeof(MockModelFile));
    if (!model)
        return NULL;

    model->total_size  = TEST_MODEL_SIZE;
    model->layer_count = LAYER_COUNT;

    // Set different sizes for each layer to simulate a real model
    for (size_t i = 0; i < LAYER_COUNT; i++) {
        // Make some layers bigger than others
        if (i % 3 == 0) {
            model->layer_sizes[i] = LAYER_SIZE * 1.5;
        }
        else if (i % 3 == 1) {
            model->layer_sizes[i] = LAYER_SIZE * 0.7;
        }
        else {
            model->layer_sizes[i] = LAYER_SIZE;
        }
    }

    // We don't actually allocate the full data for testing
    // Just simulate its existence
    model->data    = NULL;
    model->is_open = true;

    return model;
}

/**
 * Free a mock model file
 */
void free_mock_model_file(MockModelFile *model)
{
    if (model) {
        if (model->data) {
            free(model->data);
        }
        free(model);
    }
}

/**
 * Mock mapping a file into memory
 */
MockMappedModel *mock_map_file(const char *path)
{
    // Use the global mock model
    if (!g_mock_model) {
        g_mock_model = create_mock_model_file(path);
    }

    MockMappedModel *mapped = (MockMappedModel *)malloc(sizeof(MockMappedModel));
    if (!mapped)
        return NULL;

    mapped->file_path   = path;
    mapped->mapped_data = NULL; // We don't allocate real memory
    mapped->size        = TEST_MODEL_SIZE;
    mapped->mock_file   = g_mock_model;

    return mapped;
}

/**
 * Mock unmapping a file from memory
 */
void mock_unmap_file(MockMappedModel *mapped)
{
    if (mapped) {
        free(mapped);
    }
}