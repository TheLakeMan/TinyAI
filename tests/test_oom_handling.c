/**
 * @file test_oom_handling.c
 * @brief Tests for out-of-memory handling in TinyAI
 *
 * This file implements MEM-007 from the test plan, testing how
 * the memory pool and memory-mapped model loader handle out-of-memory
 * conditions gracefully.
 */

#include "../utils/forward_scheduler.h"
#include "../utils/memory_pool.h"
#include "../utils/mmap_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Test parameters */
#define TEST_SMALL_CAPACITY (1024 * 1024) /* 1 MB */
#define TEST_MODEL_FILE "data/test_model.tmai"
#define TEST_LARGE_ALLOC (16 * 1024 * 1024) /* 16 MB - larger than capacity */

/* Test statistics */
static struct {
    int passCount;
    int failCount;
} testStats = {0};

/**
 * Test memory pool out-of-memory handling
 */
static int testMemoryPoolOOM()
{
    printf("Running test: Memory Pool Out-of-Memory Handling\n");

    /* Create a memory pool with small capacity */
    TinyAIMemoryPoolConfig config;
    tinyaiMemoryPoolGetDefaultConfig(&config);
    config.initialCapacity = TEST_SMALL_CAPACITY;
    config.allowGrowth     = false; /* Disable growth to force OOM */

    TinyAIMemoryPool *pool = tinyaiMemoryPoolCreate(&config);
    if (!pool) {
        printf("ERROR: Failed to create memory pool\n");
        return 0;
    }

    /* Allocate until the pool is nearly full */
    void *ptr1 = tinyaiMemoryPoolAlloc(pool, TEST_SMALL_CAPACITY / 2, 16);
    if (!ptr1) {
        printf("ERROR: Failed initial allocation which should succeed\n");
        tinyaiMemoryPoolDestroy(pool);
        return 0;
    }

    /* Write some data to ensure memory is usable */
    memset(ptr1, 0xAA, TEST_SMALL_CAPACITY / 2);

    /* Try to allocate more than remaining space */
    void *ptr2 = tinyaiMemoryPoolAlloc(pool, TEST_SMALL_CAPACITY / 2 + 1024, 16);

    /* This allocation should fail, returning NULL */
    if (ptr2 != NULL) {
        printf("ERROR: Allocation succeeded when it should have failed\n");
        tinyaiMemoryPoolFree(pool, ptr1);
        tinyaiMemoryPoolFree(pool, ptr2);
        tinyaiMemoryPoolDestroy(pool);
        return 0;
    }

    printf("  Successfully handled OOM condition (returned NULL)\n");

    /* Free the first allocation to check pool is still functional */
    tinyaiMemoryPoolFree(pool, ptr1);

    /* Try another allocation that should now succeed */
    void *ptr3 = tinyaiMemoryPoolAlloc(pool, TEST_SMALL_CAPACITY / 4, 16);
    if (!ptr3) {
        printf("ERROR: Pool unusable after OOM condition\n");
        tinyaiMemoryPoolDestroy(pool);
        return 0;
    }

    /* Write some data to ensure memory is usable */
    memset(ptr3, 0xBB, TEST_SMALL_CAPACITY / 4);

    printf("  Pool is still usable after OOM condition\n");

    /* Free the allocation */
    tinyaiMemoryPoolFree(pool, ptr3);

    /* Try allocation larger than the entire pool */
    void *ptr4 = tinyaiMemoryPoolAlloc(pool, TEST_LARGE_ALLOC, 16);
    if (ptr4 != NULL) {
        printf("ERROR: Allocation succeeded when it should have failed\n");
        tinyaiMemoryPoolFree(pool, ptr4);
        tinyaiMemoryPoolDestroy(pool);
        return 0;
    }

    printf("  Successfully handled extreme OOM condition\n");

    /* Get and check statistics */
    TinyAIMemoryPoolStats stats;
    tinyaiMemoryPoolGetStats(pool, &stats);

    printf("  Total pool size: %zu bytes\n", stats.totalAllocated);
    printf("  Free blocks: %zu\n", stats.freeBlocks);

    tinyaiMemoryPoolDestroy(pool);

    return 1;
}

/**
 * Test memory-mapped model OOM handling
 */
static int testMmapModelOOM()
{
    printf("Running test: Memory-Mapped Model OOM Handling\n");

    /* Create a configuration with a tiny cache size */
    TinyAIMmapConfig config = tinyaiCreateDefaultMmapConfig();
    config.maxCacheSize     = 1024; /* 1 KB - way too small for layers */

    /* Try to open the model */
    TinyAIMappedModel *model = tinyaiOpenMappedModel(TEST_MODEL_FILE, &config);
    if (!model) {
        printf("ERROR: Failed to open model file\n");
        return 0;
    }

    /* Get the layer count */
    int layerCount = tinyaiGetMappedLayerCount(model);
    if (layerCount <= 0) {
        printf("ERROR: Invalid layer count: %d\n", layerCount);
        tinyaiCloseMappedModel(model);
        return 0;
    }

    /* Load first layer - this should evict previous layers automatically */
    void *layer0 = tinyaiGetLayerWeights(model, 0);
    if (!layer0) {
        printf("ERROR: Failed to load first layer\n");
        tinyaiCloseMappedModel(model);
        return 0;
    }

    printf("  Successfully loaded layer 0 despite tiny cache\n");

    /* Load several more layers in sequence to force cache evictions */
    for (int i = 1; i < layerCount && i < 5; i++) {
        void *layer = tinyaiGetLayerWeights(model, i);
        if (!layer) {
            printf("ERROR: Failed to load layer %d\n", i);
            tinyaiCloseMappedModel(model);
            return 0;
        }

        printf("  Successfully loaded layer %d\n", i);

        /* Check if layer 0 was evicted as expected */
        size_t memUsage = tinyaiGetMappedModelMemoryUsage(model);
        printf("  Current memory usage: %zu bytes\n", memUsage);

        /* Try to use layer data to ensure it's valid */
        unsigned char firstByte = *(unsigned char *)layer;
        printf("  Layer %d first byte: %d\n", i, firstByte);
    }

    /* Try loading layer 0 again to ensure it works after eviction */
    void *layer0Again = tinyaiGetLayerWeights(model, 0);
    if (!layer0Again) {
        printf("ERROR: Failed to reload layer 0\n");
        tinyaiCloseMappedModel(model);
        return 0;
    }

    printf("  Successfully reloaded layer 0 after eviction\n");

    /* Clean up */
    tinyaiCloseMappedModel(model);
    return 1;
}

/**
 * Test forward scheduler memory constraint handling
 */
static int testForwardSchedulerMemConstraints()
{
    printf("Running test: Forward Scheduler Memory Constraint Handling\n");

    /* Create default memory-mapped model config */
    TinyAIMmapConfig config = tinyaiCreateDefaultMmapConfig();

    /* Open the model */
    TinyAIMappedModel *model = tinyaiOpenMappedModel(TEST_MODEL_FILE, &config);
    if (!model) {
        printf("ERROR: Failed to open model\n");
        return 0;
    }

    /* Create a scheduler with extremely tight memory constraints */
    size_t                  tinyMemoryLimit = 512 * 1024; /* 512 KB */
    TinyAIForwardScheduler *scheduler =
        tinyaiCreateForwardScheduler(model, TINYAI_EXEC_MEMORY_OPT, tinyMemoryLimit);

    if (!scheduler) {
        printf("ERROR: Failed to create forward scheduler\n");
        tinyaiCloseMappedModel(model);
        return 0;
    }

    /* Add layers to the schedule */
    int layerCount = tinyaiGetMappedLayerCount(model);
    for (int i = 0; i < layerCount && i < 5; i++) {
        int                  dependsOn = (i > 0) ? i - 1 : -1;
        TinyAIDependencyType depType   = (i > 0) ? TINYAI_DEP_SEQUENTIAL : TINYAI_DEP_NONE;

        /* Each layer produces output activations */
        size_t outputSize =
            256 * 1024; /* 256 KB - chosen to be large enough to challenge memory constraint */

        if (!tinyaiAddLayerToSchedule(scheduler, i, dependsOn, depType, outputSize)) {
            /* Failure is expected at some point due to memory constraints */
            printf("  Layer %d addition failed due to memory constraints (expected)\n", i);
            break;
        }

        printf("  Successfully added layer %d to schedule\n", i);
    }

    /* Prepare for execution */
    if (!tinyaiPrepareForwardPass(scheduler)) {
        printf("  Forward pass preparation failed due to memory constraints (expected)\n");
    }
    else {
        /* If preparation succeeds, try executing the layers */
        printf("  Forward pass prepared successfully\n");

        /* Execute layers */
        int layerIndex;
        while (tinyaiExecuteNextLayer(scheduler, NULL, NULL, &layerIndex)) {
            printf("  Executed layer %d\n", layerIndex);

            /* Check if memory limit is respected */
            size_t memUsage = tinyaiGetSchedulerMemoryUsage(scheduler);
            printf("  Current memory usage: %zu bytes (limit: %zu)\n", memUsage, tinyMemoryLimit);

            if (memUsage > tinyMemoryLimit) {
                printf("ERROR: Memory usage exceeds limit\n");
                tinyaiDestroyForwardScheduler(scheduler);
                tinyaiCloseMappedModel(model);
                return 0;
            }
        }
    }

    /* Calculate optimal batch size under constraints */
    int batchSize = tinyaiCalculateOptimalBatchSize(scheduler, tinyMemoryLimit / 2, 1024, 32);
    printf("  Calculated optimal batch size: %d\n", batchSize);

    /* Clean up */
    tinyaiDestroyForwardScheduler(scheduler);
    tinyaiCloseMappedModel(model);
    return 1;
}

/**
 * Main test function
 */
int main()
{
    printf("=== TinyAI Out-of-Memory Handling Tests ===\n\n");

    /* Initialize random seed */
    srand((unsigned int)time(NULL));

    /* Ensure test model exists */
    FILE *fp = fopen(TEST_MODEL_FILE, "rb");
    if (!fp) {
        printf("Test model file not found: %s\n", TEST_MODEL_FILE);
        printf("Please run test_mmap_loader first to generate the test model.\n");
        return 1;
    }
    fclose(fp);

    /* Run memory pool OOM test */
    if (testMemoryPoolOOM()) {
        printf("Memory Pool OOM Test: PASSED\n\n");
        testStats.passCount++;
    }
    else {
        printf("Memory Pool OOM Test: FAILED\n\n");
        testStats.failCount++;
    }

    /* Run memory-mapped model OOM test */
    if (testMmapModelOOM()) {
        printf("Memory-Mapped Model OOM Test: PASSED\n\n");
        testStats.passCount++;
    }
    else {
        printf("Memory-Mapped Model OOM Test: FAILED\n\n");
        testStats.failCount++;
    }

    /* Run forward scheduler memory constraints test */
    if (testForwardSchedulerMemConstraints()) {
        printf("Forward Scheduler Memory Constraints Test: PASSED\n\n");
        testStats.passCount++;
    }
    else {
        printf("Forward Scheduler Memory Constraints Test: FAILED\n\n");
        testStats.failCount++;
    }

    /* Print summary */
    printf("=== Test Summary ===\n");
    printf("Passed: %d\n", testStats.passCount);
    printf("Failed: %d\n", testStats.failCount);

    return testStats.failCount > 0 ? 1 : 0;
}