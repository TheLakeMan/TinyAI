/**
 * @file test_mmap_loader.c
 * @brief Tests for memory-mapped model loading and forward pass scheduling
 */

#include "../utils/forward_scheduler.h"
#include "../utils/mmap_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Test model file path */
#define TEST_MODEL_FILE "data/test_model.tmai"

/* Test model dimensions */
#define TEST_MODEL_LAYERS 10
#define TEST_LAYER_SIZE (1024 * 1024) /* 1MB per layer */

/* Memory limit for testing */
#define TEST_MEMORY_LIMIT (5 * 1024 * 1024) /* 5MB */

/* Generate a test model file */
static bool generateTestModel(const char *filepath)
{
    printf("Generating test model file: %s\n", filepath);

    /* Open file for writing */
    FILE *fp = fopen(filepath, "wb");
    if (!fp) {
        printf("Failed to open file for writing: %s\n", filepath);
        return false;
    }

    /* Write header */
    uint32_t header[64] = {0};
    header[0]           = 0x544D4149;           /* "TMAI" magic number */
    header[1]           = 1;                    /* Version */
    header[2]           = TEST_MODEL_LAYERS;    /* Layer count */
    strcpy((char *)(header + 4), "Test Model"); /* Model name */

    if (fwrite(header, sizeof(uint32_t), 64, fp) != 64) {
        printf("Failed to write header\n");
        fclose(fp);
        return false;
    }

    /* Write layer descriptors */
    uint32_t layerHeader[8] = {0};
    size_t   offset         = 256 + TEST_MODEL_LAYERS * 32; /* Header + layer descriptors */

    for (int i = 0; i < TEST_MODEL_LAYERS; i++) {
        layerHeader[0] = (uint32_t)offset; /* Offset */
        layerHeader[1] = TEST_LAYER_SIZE;  /* Size */
        layerHeader[2] = 4;                /* Precision (4-bit) */

        if (fwrite(layerHeader, sizeof(uint32_t), 8, fp) != 8) {
            printf("Failed to write layer descriptor %d\n", i);
            fclose(fp);
            return false;
        }

        offset += TEST_LAYER_SIZE;
    }

    /* Write layer data */
    unsigned char *buffer = (unsigned char *)malloc(TEST_LAYER_SIZE);
    if (!buffer) {
        printf("Failed to allocate buffer\n");
        fclose(fp);
        return false;
    }

    /* Fill buffer with test pattern */
    for (int i = 0; i < TEST_MODEL_LAYERS; i++) {
        /* Fill buffer with layer index */
        memset(buffer, i, TEST_LAYER_SIZE);

        if (fwrite(buffer, 1, TEST_LAYER_SIZE, fp) != TEST_LAYER_SIZE) {
            printf("Failed to write layer data %d\n", i);
            free(buffer);
            fclose(fp);
            return false;
        }
    }

    free(buffer);
    fclose(fp);
    printf("Test model file generated successfully\n");
    return true;
}

/* Test memory-mapped model loading */
static bool testMmapLoading()
{
    printf("Testing memory-mapped model loading...\n");

    /* Create default config */
    TinyAIMmapConfig config = tinyaiCreateDefaultMmapConfig();

    /* Set small cache size to force evictions */
    config.maxCacheSize = 3 * TEST_LAYER_SIZE;

    /* Open the model */
    TinyAIMappedModel *model = tinyaiOpenMappedModel(TEST_MODEL_FILE, &config);
    if (!model) {
        printf("Failed to open model\n");
        return false;
    }

    /* Check layer count */
    int layerCount = tinyaiGetMappedLayerCount(model);
    if (layerCount != TEST_MODEL_LAYERS) {
        printf("Incorrect layer count: %d (expected %d)\n", layerCount, TEST_MODEL_LAYERS);
        tinyaiCloseMappedModel(model);
        return false;
    }

    /* Load each layer and verify contents */
    for (int i = 0; i < layerCount; i++) {
        /* Get layer descriptor */
        const TinyAILayerDescriptor *desc = tinyaiGetLayerDescriptor(model, i);
        if (!desc) {
            printf("Failed to get layer descriptor %d\n", i);
            tinyaiCloseMappedModel(model);
            return false;
        }

        /* Verify layer size */
        if (desc->size != TEST_LAYER_SIZE) {
            printf("Incorrect layer size: %zu (expected %d)\n", desc->size, TEST_LAYER_SIZE);
            tinyaiCloseMappedModel(model);
            return false;
        }

        /* Load layer weights */
        void *weights = tinyaiGetLayerWeights(model, i);
        if (!weights) {
            printf("Failed to load weights for layer %d\n", i);
            tinyaiCloseMappedModel(model);
            return false;
        }

        /* Verify first byte */
        unsigned char firstByte = *(unsigned char *)weights;
        if (firstByte != i) {
            printf("Incorrect first byte: %d (expected %d)\n", firstByte, i);
            tinyaiCloseMappedModel(model);
            return false;
        }

        /* Use the layer */
        printf("Layer %d loaded and verified\n", i);

        /* For even-indexed layers, release them to test cache behavior */
        if (i % 2 == 0) {
            tinyaiReleaseLayerWeights(model, i);
            printf("Layer %d released\n", i);
        }
    }

    /* Close the model */
    tinyaiCloseMappedModel(model);
    printf("Memory-mapped model loading test passed\n");
    return true;
}

/* Test forward pass scheduling */
static bool testForwardScheduling()
{
    printf("Testing forward pass scheduling...\n");

    /* Create default config */
    TinyAIMmapConfig config = tinyaiCreateDefaultMmapConfig();

    /* Open the model */
    TinyAIMappedModel *model = tinyaiOpenMappedModel(TEST_MODEL_FILE, &config);
    if (!model) {
        printf("Failed to open model\n");
        return false;
    }

    /* Create scheduler with memory limit */
    TinyAIForwardScheduler *scheduler =
        tinyaiCreateForwardScheduler(model, TINYAI_EXEC_MEMORY_OPT, TEST_MEMORY_LIMIT);
    if (!scheduler) {
        printf("Failed to create scheduler\n");
        tinyaiCloseMappedModel(model);
        return false;
    }

    /* Add layers to schedule */
    printf("Adding layers to schedule...\n");

    /* Simple linear network */
    for (int i = 0; i < TEST_MODEL_LAYERS; i++) {
        int                  dependsOn = (i > 0) ? i - 1 : -1;
        TinyAIDependencyType depType   = (i > 0) ? TINYAI_DEP_SEQUENTIAL : TINYAI_DEP_NONE;

        /* Each layer produces 1MB of activations */
        size_t outputSize = 1024 * 1024;

        if (!tinyaiAddLayerToSchedule(scheduler, i, dependsOn, depType, outputSize)) {
            printf("Failed to add layer %d to schedule\n", i);
            tinyaiDestroyForwardScheduler(scheduler);
            tinyaiCloseMappedModel(model);
            return false;
        }
    }

    /* Prepare for execution */
    if (!tinyaiPrepareForwardPass(scheduler)) {
        printf("Failed to prepare forward pass\n");
        tinyaiDestroyForwardScheduler(scheduler);
        tinyaiCloseMappedModel(model);
        return false;
    }

    /* Execute forward pass */
    printf("Executing forward pass...\n");
    int layerIndex;
    while (tinyaiExecuteNextLayer(scheduler, NULL, NULL, &layerIndex)) {
        printf("Executed layer %d\n", layerIndex);

        /* Print memory usage */
        size_t memUsage = tinyaiGetSchedulerMemoryUsage(scheduler);
        printf("Current memory usage: %.2f MB\n", memUsage / (1024.0 * 1024.0));

        /* Verify memory usage is below limit */
        if (memUsage > TEST_MEMORY_LIMIT) {
            printf("Memory usage exceeded limit: %zu > %d\n", memUsage, TEST_MEMORY_LIMIT);
            tinyaiDestroyForwardScheduler(scheduler);
            tinyaiCloseMappedModel(model);
            return false;
        }
    }

    /* Print peak memory usage */
    size_t peakMemUsage = tinyaiGetSchedulerPeakMemoryUsage(scheduler);
    printf("Peak memory usage: %.2f MB\n", peakMemUsage / (1024.0 * 1024.0));

    /* Calculate optimal batch size */
    int batchSize = tinyaiCalculateOptimalBatchSize(scheduler, 1024 * 1024, 1024 * 1024, 32);
    printf("Optimal batch size: %d\n", batchSize);

    /* Clean up */
    tinyaiDestroyForwardScheduler(scheduler);
    tinyaiCloseMappedModel(model);
    printf("Forward pass scheduling test passed\n");
    return true;
}

/* Main test function */
int main()
{
    /* Generate test model file */
    if (!generateTestModel(TEST_MODEL_FILE)) {
        printf("Failed to generate test model\n");
        return 1;
    }

    /* Test memory-mapped model loading */
    if (!testMmapLoading()) {
        printf("Memory-mapped model loading test failed\n");
        return 1;
    }

    /* Test forward pass scheduling */
    if (!testForwardScheduling()) {
        printf("Forward pass scheduling test failed\n");
        return 1;
    }

    printf("All tests passed!\n");
    return 0;
}
