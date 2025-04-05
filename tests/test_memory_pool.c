/**
 * @file test_memory_pool.c
 * @brief Tests for enhanced memory pool system
 */

#include "../utils/memory_pool.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TEST_INITIAL_CAPACITY (1024 * 1024 * 4) /* 4 MB */
#define TEST_MAX_ALLOCATIONS 1000
#define TEST_MIN_ALLOCATION_SIZE 16
#define TEST_MAX_ALLOCATION_SIZE 32768
#define TEST_ITERATIONS 5

/* Global test statistics */
static struct {
    int    passCount;
    int    failCount;
    double totalAllocTime;
    double totalFreeTime;
    size_t peakMemoryUsage;
} testStats = {0};

/* Test basic allocation and deallocation */
static int testBasicAllocFree()
{
    printf("Running test: Basic Allocation and Deallocation\n");

    TinyAIMemoryPoolConfig config;
    tinyaiMemoryPoolGetDefaultConfig(&config);
    config.initialCapacity = TEST_INITIAL_CAPACITY;

    TinyAIMemoryPool *pool = tinyaiMemoryPoolCreate(&config);
    if (!pool) {
        printf("ERROR: Failed to create memory pool\n");
        return 0;
    }

    /* Test a simple allocation */
    void *ptr = tinyaiMemoryPoolAlloc(pool, 1024, 16);
    if (!ptr) {
        printf("ERROR: Failed to allocate memory\n");
        tinyaiMemoryPoolDestroy(pool);
        return 0;
    }

    /* Verify the pointer is aligned */
    if ((uintptr_t)ptr % 16 != 0) {
        printf("ERROR: Memory is not properly aligned\n");
        tinyaiMemoryPoolDestroy(pool);
        return 0;
    }

    /* Write some data to ensure the memory is usable */
    memset(ptr, 0xAA, 1024);

    /* Free the memory */
    tinyaiMemoryPoolFree(pool, ptr);

    /* Get statistics */
    TinyAIMemoryPoolStats stats;
    tinyaiMemoryPoolGetStats(pool, &stats);

    printf("  Test allocated and freed 1024 bytes\n");
    printf("  Total pool size: %zu bytes\n", stats.totalAllocated);
    printf("  Free blocks: %zu\n", stats.freeBlocks);

    tinyaiMemoryPoolDestroy(pool);

    /* Test passed if we got here without crashing */
    return 1;
}

/* Test multiple allocations of different sizes */
static int testMultipleAllocs()
{
    printf("Running test: Multiple Allocations\n");

    TinyAIMemoryPoolConfig config;
    tinyaiMemoryPoolGetDefaultConfig(&config);
    config.initialCapacity = TEST_INITIAL_CAPACITY;

    TinyAIMemoryPool *pool = tinyaiMemoryPoolCreate(&config);
    if (!pool) {
        printf("ERROR: Failed to create memory pool\n");
        return 0;
    }

    /* Allocate several blocks of different sizes */
    void  *ptrs[10];
    size_t sizes[10] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};

    for (int i = 0; i < 10; i++) {
        ptrs[i] = tinyaiMemoryPoolAlloc(pool, sizes[i], 16);
        if (!ptrs[i]) {
            printf("ERROR: Failed to allocate block %d of size %zu\n", i, sizes[i]);
            tinyaiMemoryPoolDestroy(pool);
            return 0;
        }

        /* Write some data to the memory */
        memset(ptrs[i], i + 1, sizes[i]);
    }

    /* Verify the data */
    for (int i = 0; i < 10; i++) {
        unsigned char *p = (unsigned char *)ptrs[i];
        for (size_t j = 0; j < sizes[i]; j++) {
            if (p[j] != (unsigned char)(i + 1)) {
                printf("ERROR: Memory corruption detected in block %d\n", i);
                tinyaiMemoryPoolDestroy(pool);
                return 0;
            }
        }
    }

    /* Free all blocks */
    for (int i = 0; i < 10; i++) {
        tinyaiMemoryPoolFree(pool, ptrs[i]);
    }

    /* Get statistics */
    TinyAIMemoryPoolStats stats;
    tinyaiMemoryPoolGetStats(pool, &stats);

    printf("  Allocated and verified 10 blocks of different sizes\n");
    printf("  Total pool size: %zu bytes\n", stats.totalAllocated);
    printf("  Free blocks: %zu\n", stats.freeBlocks);

    tinyaiMemoryPoolDestroy(pool);

    return 1;
}

/* Test fragmentation and merging */
static int testFragmentationAndMerging()
{
    printf("Running test: Fragmentation and Merging\n");

    TinyAIMemoryPoolConfig config;
    tinyaiMemoryPoolGetDefaultConfig(&config);
    config.initialCapacity = TEST_INITIAL_CAPACITY;

    TinyAIMemoryPool *pool = tinyaiMemoryPoolCreate(&config);
    if (!pool) {
        printf("ERROR: Failed to create memory pool\n");
        return 0;
    }

    /* Allocate many small blocks to fragment the pool */
    void *ptrs[100];
    for (int i = 0; i < 100; i++) {
        ptrs[i] = tinyaiMemoryPoolAlloc(pool, 1024, 16);
        if (!ptrs[i]) {
            printf("ERROR: Failed to allocate block %d\n", i);
            tinyaiMemoryPoolDestroy(pool);
            return 0;
        }
    }

    /* Free alternating blocks to create fragmentation */
    for (int i = 0; i < 100; i += 2) {
        tinyaiMemoryPoolFree(pool, ptrs[i]);
    }

    /* Check fragmentation */
    TinyAIMemoryPoolStats statsBefore;
    tinyaiMemoryPoolGetStats(pool, &statsBefore);

    /* Compact the pool */
    tinyaiMemoryPoolCompact(pool);

    /* Check if compaction helped */
    TinyAIMemoryPoolStats statsAfter;
    tinyaiMemoryPoolGetStats(pool, &statsAfter);

    printf("  Fragmentation before compaction: %zu%%\n", statsBefore.fragmentationScore);
    printf("  Fragmentation after compaction: %zu%%\n", statsAfter.fragmentationScore);
    printf("  Free blocks before: %zu, after: %zu\n", statsBefore.freeBlocks,
           statsAfter.freeBlocks);

    /* Free remaining blocks */
    for (int i = 1; i < 100; i += 2) {
        tinyaiMemoryPoolFree(pool, ptrs[i]);
    }

    /* Check final state after all blocks are freed */
    TinyAIMemoryPoolStats finalStats;
    tinyaiMemoryPoolGetStats(pool, &finalStats);

    printf("  Total free blocks after all deallocations: %zu\n", finalStats.freeBlocks);

    tinyaiMemoryPoolDestroy(pool);

    /* Test passes if compaction reduced fragmentation or free block count */
    return (statsAfter.fragmentationScore <= statsBefore.fragmentationScore ||
            statsAfter.freeBlocks < statsBefore.freeBlocks);
}

/* Test weight allocation for 4-bit quantized matrices */
static int testWeightAllocation()
{
    printf("Running test: 4-Bit Weight Allocation\n");

    TinyAIMemoryPoolConfig config;
    tinyaiMemoryPoolGetDefaultConfig(&config);
    config.initialCapacity = TEST_INITIAL_CAPACITY;

    TinyAIMemoryPool *pool = tinyaiMemoryPoolCreate(&config);
    if (!pool) {
        printf("ERROR: Failed to create memory pool\n");
        return 0;
    }

    /* Allocate weights for a simple matrix (100x100) */
    size_t   rows    = 100;
    size_t   cols    = 100;
    uint8_t *weights = tinyaiMemoryPoolAllocWeights4Bit(pool, rows, cols, true);

    if (!weights) {
        printf("ERROR: Failed to allocate weights\n");
        tinyaiMemoryPoolDestroy(pool);
        return 0;
    }

    /* Check alignment for SIMD */
    if ((uintptr_t)weights % 32 != 0) {
        printf("ERROR: Weights not aligned for SIMD operations\n");
        tinyaiMemoryPoolDestroy(pool);
        return 0;
    }

    /* Write some test data */
    size_t numBytes = (rows * cols + 1) / 2;
    for (size_t i = 0; i < numBytes; i++) {
        weights[i] = (uint8_t)i & 0xFF;
    }

    /* Free the weights */
    tinyaiMemoryPoolFree(pool, weights);

    /* Allocate activations */
    size_t activationSize = 1000;
    float *activations    = tinyaiMemoryPoolAllocActivations(pool, activationSize, true);

    if (!activations) {
        printf("ERROR: Failed to allocate activations\n");
        tinyaiMemoryPoolDestroy(pool);
        return 0;
    }

    /* Check alignment */
    if ((uintptr_t)activations % 32 != 0) {
        printf("ERROR: Activations not aligned for SIMD operations\n");
        tinyaiMemoryPoolDestroy(pool);
        return 0;
    }

    /* Initialize activations */
    for (size_t i = 0; i < activationSize; i++) {
        activations[i] = (float)i / 1000.0f;
    }

    /* Free activations */
    tinyaiMemoryPoolFree(pool, activations);

    /* Get statistics */
    TinyAIMemoryPoolStats stats;
    tinyaiMemoryPoolGetStats(pool, &stats);

    printf("  Successfully allocated and freed weights (100x100) and activations (1000)\n");
    printf("  Total pool size: %zu bytes\n", stats.totalAllocated);
    printf("  Free blocks: %zu\n", stats.freeBlocks);

    tinyaiMemoryPoolDestroy(pool);

    return 1;
}

/* Stress test with many allocations and deallocations */
static int testStressTest()
{
    printf("Running test: Stress Test\n");

    TinyAIMemoryPoolConfig config;
    tinyaiMemoryPoolGetDefaultConfig(&config);
    config.initialCapacity = TEST_INITIAL_CAPACITY;
    config.allowGrowth     = true;

    TinyAIMemoryPool *pool = tinyaiMemoryPoolCreate(&config);
    if (!pool) {
        printf("ERROR: Failed to create memory pool\n");
        return 0;
    }

    /* Allocate a large number of random-sized blocks */
    void  *ptrs[TEST_MAX_ALLOCATIONS];
    size_t sizes[TEST_MAX_ALLOCATIONS];
    int    numAllocs = 0;

    clock_t startAlloc = clock();

    for (int i = 0; i < TEST_MAX_ALLOCATIONS; i++) {
        /* Random size between min and max */
        sizes[i] = TEST_MIN_ALLOCATION_SIZE +
                   (rand() % (TEST_MAX_ALLOCATION_SIZE - TEST_MIN_ALLOCATION_SIZE));

        ptrs[i] = tinyaiMemoryPoolAlloc(pool, sizes[i], 16);
        if (!ptrs[i]) {
            printf("  Could not allocate more after %d allocations\n", i);
            numAllocs = i;
            break;
        }

        /* Write some data to the memory */
        memset(ptrs[i], i & 0xFF, sizes[i]);
        numAllocs++;
    }

    clock_t endAlloc  = clock();
    double  allocTime = (double)(endAlloc - startAlloc) / CLOCKS_PER_SEC;

    /* Get statistics */
    TinyAIMemoryPoolStats stats;
    tinyaiMemoryPoolGetStats(pool, &stats);

    printf("  Allocated %d blocks in %.4f seconds (%.2f allocations/sec)\n", numAllocs, allocTime,
           numAllocs / allocTime);
    printf("  Total pool size: %zu bytes (%.2f MB)\n", stats.totalAllocated,
           (double)stats.totalAllocated / (1024.0 * 1024.0));
    printf("  Used memory: %zu bytes (%.2f MB)\n", stats.totalUsed,
           (double)stats.totalUsed / (1024.0 * 1024.0));

    /* Update stats */
    testStats.totalAllocTime += allocTime;
    if (stats.totalUsed > testStats.peakMemoryUsage) {
        testStats.peakMemoryUsage = stats.totalUsed;
    }

    /* Free all allocated blocks */
    clock_t startFree = clock();

    for (int i = 0; i < numAllocs; i++) {
        tinyaiMemoryPoolFree(pool, ptrs[i]);
    }

    clock_t endFree  = clock();
    double  freeTime = (double)(endFree - startFree) / CLOCKS_PER_SEC;

    printf("  Freed %d blocks in %.4f seconds (%.2f deallocations/sec)\n", numAllocs, freeTime,
           numAllocs / freeTime);

    /* Update stats */
    testStats.totalFreeTime += freeTime;

    /* Get statistics after freeing */
    tinyaiMemoryPoolGetStats(pool, &stats);

    printf("  Free blocks: %zu\n", stats.freeBlocks);
    printf("  Fragmentation: %zu%%\n", stats.fragmentationScore);

    tinyaiMemoryPoolDestroy(pool);

    return 1;
}

/* Test large model simulation */
static int testLargeModelSimulation()
{
    printf("Running test: Large Model Simulation\n");

    TinyAIMemoryPoolConfig config;
    tinyaiMemoryPoolGetDefaultConfig(&config);
    config.initialCapacity = TEST_INITIAL_CAPACITY * 2;
    config.allowGrowth     = true;

    TinyAIMemoryPool *pool = tinyaiMemoryPoolCreate(&config);
    if (!pool) {
        printf("ERROR: Failed to create memory pool\n");
        return 0;
    }

    /* Simulate a large language model with multiple layers */
    const int numLayers        = 8;
    const int hiddenSize       = 512;
    const int intermediateSize = 2048;
    const int vocabSize        = 5000;
    const int maxSeqLength     = 512;

    /* Allocate weights for each layer */
    uint8_t *qkvWeights[numLayers];
    uint8_t *ffnWeights[numLayers];
    float   *activations[numLayers * 2]; /* Two activations per layer */

    /* Allocate all weights and activations */
    for (int i = 0; i < numLayers; i++) {
        /* Query/Key/Value weights - 3 * hiddenSize^2 */
        qkvWeights[i] = tinyaiMemoryPoolAllocWeights4Bit(pool, 3 * hiddenSize, hiddenSize, true);

        if (!qkvWeights[i]) {
            printf("ERROR: Failed to allocate QKV weights for layer %d\n", i);
            tinyaiMemoryPoolDestroy(pool);
            return 0;
        }

        /* Feed-forward weights - intermediateSize * hiddenSize */
        ffnWeights[i] = tinyaiMemoryPoolAllocWeights4Bit(pool, intermediateSize, hiddenSize, true);

        if (!ffnWeights[i]) {
            printf("ERROR: Failed to allocate FFN weights for layer %d\n", i);
            tinyaiMemoryPoolDestroy(pool);
            return 0;
        }

        /* Allocate activations */
        activations[i * 2] =
            tinyaiMemoryPoolAllocActivations(pool, maxSeqLength * hiddenSize, true);

        if (!activations[i * 2]) {
            printf("ERROR: Failed to allocate first activation for layer %d\n", i);
            tinyaiMemoryPoolDestroy(pool);
            return 0;
        }

        activations[i * 2 + 1] =
            tinyaiMemoryPoolAllocActivations(pool, maxSeqLength * hiddenSize, true);

        if (!activations[i * 2 + 1]) {
            printf("ERROR: Failed to allocate second activation for layer %d\n", i);
            tinyaiMemoryPoolDestroy(pool);
            return 0;
        }
    }

    /* Get statistics */
    TinyAIMemoryPoolStats stats;
    tinyaiMemoryPoolGetStats(pool, &stats);

    printf("  Model simulation allocated %d QKV matrices, %d FFN matrices, and %d activation "
           "tensors\n",
           numLayers, numLayers, numLayers * 2);
    printf("  Total memory usage: %zu bytes (%.2f MB)\n", stats.totalUsed,
           (double)stats.totalUsed / (1024.0 * 1024.0));

    /* Free all weights and activations in reverse order */
    for (int i = numLayers - 1; i >= 0; i--) {
        tinyaiMemoryPoolFree(pool, activations[i * 2 + 1]);
        tinyaiMemoryPoolFree(pool, activations[i * 2]);
        tinyaiMemoryPoolFree(pool, ffnWeights[i]);
        tinyaiMemoryPoolFree(pool, qkvWeights[i]);
    }

    /* Get statistics after freeing */
    tinyaiMemoryPoolGetStats(pool, &stats);

    printf("  After freeing: Used memory: %zu bytes\n", stats.totalUsed);
    printf("  Free blocks: %zu\n", stats.freeBlocks);

    tinyaiMemoryPoolDestroy(pool);

    return 1;
}

/* Run a series of allocation performance tests */
static void runPerformanceTests()
{
    printf("\nRunning Performance Tests\n");
    printf("-------------------------\n");

    /* Reset stats */
    testStats.totalAllocTime  = 0;
    testStats.totalFreeTime   = 0;
    testStats.peakMemoryUsage = 0;

    /* Run the stress test multiple times */
    for (int i = 0; i < TEST_ITERATIONS; i++) {
        printf("\nIteration %d:\n", i + 1);
        testStressTest();
    }

    printf("\nPerformance Test Results:\n");
    printf("  Average allocation time: %.4f seconds\n", testStats.totalAllocTime / TEST_ITERATIONS);
    printf("  Average deallocation time: %.4f seconds\n",
           testStats.totalFreeTime / TEST_ITERATIONS);
    printf("  Peak memory usage: %zu bytes (%.2f MB)\n", testStats.peakMemoryUsage,
           (double)testStats.peakMemoryUsage / (1024.0 * 1024.0));
}

/* Main test function */
int main()
{
    srand(time(NULL));

    printf("TinyAI Memory Pool Test Suite\n");
    printf("=============================\n\n");

    /* Run tests */
    int passCount = 0;
    int failCount = 0;

    /* Basic tests */
    if (testBasicAllocFree()) {
        printf("✓ Basic allocation test passed\n\n");
        passCount++;
    }
    else {
        printf("✗ Basic allocation test failed\n\n");
        failCount++;
    }

    if (testMultipleAllocs()) {
        printf("✓ Multiple allocations test passed\n\n");
        passCount++;
    }
    else {
        printf("✗ Multiple allocations test failed\n\n");
        failCount++;
    }

    if (testFragmentationAndMerging()) {
        printf("✓ Fragmentation and merging test passed\n\n");
        passCount++;
    }
    else {
        printf("✗ Fragmentation and merging test failed\n\n");
        failCount++;
    }

    if (testWeightAllocation()) {
        printf("✓ Weight allocation test passed\n\n");
        passCount++;
    }
    else {
        printf("✗ Weight allocation test failed\n\n");
        failCount++;
    }

    if (testLargeModelSimulation()) {
        printf("✓ Large model simulation test passed\n\n");
        passCount++;
    }
    else {
        printf("✗ Large model simulation test failed\n\n");
        failCount++;
    }

    /* Performance tests */
    runPerformanceTests();

    /* Summary */
    printf("\nTest Summary\n");
    printf("===========\n");
    printf("Passed: %d\n", passCount);
    printf("Failed: %d\n", failCount);

    return failCount == 0 ? 0 : 1;
}
