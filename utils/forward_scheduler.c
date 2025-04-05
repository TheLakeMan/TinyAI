/**
 * @file forward_scheduler.c
 * @brief Implementation of forward pass scheduler for layer-wise computation with memory
 * optimization
 */

#include "forward_scheduler.h"
#include <float.h>
#include <stdlib.h>
#include <string.h>

/* Maximum number of layers in a model */
#define MAX_EXEC_LAYERS 256

/* Forward scheduler structure definition */
struct TinyAIForwardScheduler {
    /* Model reference */
    TinyAIMappedModel *model;

    /* Execution configuration */
    TinyAIExecutionMode mode;
    size_t              maxMemory;

    /* Layer execution schedule */
    TinyAIExecLayer layers[MAX_EXEC_LAYERS];
    int             layerCount;

    /* Execution state */
    int    currentLayer;
    size_t currentMemoryUsage;
    size_t peakMemoryUsage;

    /* Layer execution callback function and user data */
    bool (*executeLayerFunc)(void *userData, int layerIndex, const void *input, void *output);
    void *userData;
};

/* Find the next executable layer based on dependencies */
static int findNextExecutableLayer(TinyAIForwardScheduler *scheduler)
{
    /* First pass: find all layers that have all dependencies satisfied */
    for (int i = 0; i < scheduler->layerCount; i++) {
        /* Skip already executed layers */
        if (scheduler->layers[i].executed) {
            continue;
        }

        /* Check dependencies based on type */
        switch (scheduler->layers[i].depType) {
        case TINYAI_DEP_NONE:
            /* No dependencies, can always execute */
            return i;

        case TINYAI_DEP_SEQUENTIAL:
            /* Must execute after previous layer */
            if (i == 0 || scheduler->layers[i - 1].executed) {
                return i;
            }
            break;

        case TINYAI_DEP_RESIDUAL:
        case TINYAI_DEP_ATTENTION:
            /* Depends on specific layer */
            if (scheduler->layers[i].dependsOnLayer < 0 ||
                scheduler->layers[scheduler->layers[i].dependsOnLayer].executed) {
                return i;
            }
            break;
        }
    }

    /* No executable layers found */
    return -1;
}

/* Check if a layer's output is needed by any future layers */
static bool isOutputNeededByFutureLayers(TinyAIForwardScheduler *scheduler, int layerIndex)
{
    for (int i = 0; i < scheduler->layerCount; i++) {
        /* Skip already executed layers */
        if (scheduler->layers[i].executed) {
            continue;
        }

        /* Check if this layer depends on the specified layer */
        if ((scheduler->layers[i].depType == TINYAI_DEP_RESIDUAL ||
             scheduler->layers[i].depType == TINYAI_DEP_ATTENTION) &&
            scheduler->layers[i].dependsOnLayer == layerIndex) {
            return true;
        }

        /* For sequential dependencies, check if it's the next layer */
        if (scheduler->layers[i].depType == TINYAI_DEP_SEQUENTIAL && i > 0 && i - 1 == layerIndex &&
            !scheduler->layers[i - 1].executed) {
            return true;
        }
    }

    return false;
}

/* Free memory for unnecessary layer outputs */
static void freeUnneededOutputs(TinyAIForwardScheduler *scheduler)
{
    for (int i = 0; i < scheduler->layerCount; i++) {
        /* Skip unexecuted layers and those already marked as unneeded */
        if (!scheduler->layers[i].executed || !scheduler->layers[i].outputNeeded) {
            continue;
        }

        /* Check if this layer's output is still needed */
        if (!isOutputNeededByFutureLayers(scheduler, i)) {
            /* Output no longer needed, free it */
            if (scheduler->layers[i].outputPtr) {
                free(scheduler->layers[i].outputPtr);
                scheduler->layers[i].outputPtr = NULL;
                scheduler->currentMemoryUsage -= scheduler->layers[i].outputSize;
            }

            /* Mark as unneeded */
            scheduler->layers[i].outputNeeded = false;
        }
    }
}

/* Implementation of public API */

TinyAIForwardScheduler *tinyaiCreateForwardScheduler(TinyAIMappedModel  *model,
                                                     TinyAIExecutionMode mode, size_t maxMemory)
{

    if (!model) {
        return NULL;
    }

    /* Allocate scheduler */
    TinyAIForwardScheduler *scheduler =
        (TinyAIForwardScheduler *)malloc(sizeof(TinyAIForwardScheduler));
    if (!scheduler) {
        return NULL;
    }

    /* Initialize fields */
    memset(scheduler, 0, sizeof(TinyAIForwardScheduler));
    scheduler->model              = model;
    scheduler->mode               = mode;
    scheduler->maxMemory          = maxMemory;
    scheduler->layerCount         = 0;
    scheduler->currentLayer       = -1;
    scheduler->currentMemoryUsage = 0;
    scheduler->peakMemoryUsage    = 0;

    return scheduler;
}

void tinyaiDestroyForwardScheduler(TinyAIForwardScheduler *scheduler)
{
    if (!scheduler) {
        return;
    }

    /* Free all allocated output buffers */
    for (int i = 0; i < scheduler->layerCount; i++) {
        if (scheduler->layers[i].outputPtr) {
            free(scheduler->layers[i].outputPtr);
            scheduler->layers[i].outputPtr = NULL;
        }
    }

    /* Free scheduler structure */
    free(scheduler);
}

bool tinyaiAddLayerToSchedule(TinyAIForwardScheduler *scheduler, int layerIndex, int dependsOnLayer,
                              TinyAIDependencyType depType, size_t outputSize)
{

    if (!scheduler || layerIndex < 0 || scheduler->layerCount >= MAX_EXEC_LAYERS) {
        return false;
    }

    /* Validate dependency */
    if ((depType == TINYAI_DEP_RESIDUAL || depType == TINYAI_DEP_ATTENTION) &&
        (dependsOnLayer < 0 || dependsOnLayer >= scheduler->layerCount)) {
        return false;
    }

    /* Add layer to schedule */
    int index                               = scheduler->layerCount++;
    scheduler->layers[index].layerIndex     = layerIndex;
    scheduler->layers[index].depType        = depType;
    scheduler->layers[index].dependsOnLayer = dependsOnLayer;
    scheduler->layers[index].executed       = false;
    scheduler->layers[index].outputNeeded   = true;
    scheduler->layers[index].memoryUsage    = 0.0f;
    scheduler->layers[index].outputPtr      = NULL;
    scheduler->layers[index].outputSize     = outputSize;

    return true;
}

bool tinyaiPrepareForwardPass(TinyAIForwardScheduler *scheduler)
{
    if (!scheduler) {
        return false;
    }

    /* Reset execution state */
    for (int i = 0; i < scheduler->layerCount; i++) {
        scheduler->layers[i].executed     = false;
        scheduler->layers[i].outputNeeded = true;

        /* Free any existing output buffer */
        if (scheduler->layers[i].outputPtr) {
            free(scheduler->layers[i].outputPtr);
            scheduler->layers[i].outputPtr = NULL;
        }
    }

    scheduler->currentLayer       = -1;
    scheduler->currentMemoryUsage = 0;
    scheduler->peakMemoryUsage    = 0;

    return true;
}

bool tinyaiExecuteNextLayer(TinyAIForwardScheduler *scheduler, const void *input, void *output,
                            int *layerIndex)
{

    if (!scheduler) {
        return false;
    }

    /* Find next executable layer */
    int nextLayer = findNextExecutableLayer(scheduler);
    if (nextLayer < 0) {
        /* No more layers to execute */
        return false;
    }

    /* Set current layer */
    scheduler->currentLayer = nextLayer;

    /* Get layer descriptor */
    TinyAIExecLayer *layer = &scheduler->layers[nextLayer];

    /* Get model layer weights */
    void *weights = NULL;
    if (scheduler->mode == TINYAI_EXEC_MEMORY_OPT || scheduler->mode == TINYAI_EXEC_ADAPTIVE) {
        /* Only load weights when needed in memory optimized mode */
        weights = tinyaiGetLayerWeights(scheduler->model, layer->layerIndex);
        if (!weights) {
            return false;
        }
    }

    /* Allocate output buffer if needed */
    if (!layer->outputPtr && layer->outputSize > 0) {
        layer->outputPtr = malloc(layer->outputSize);
        if (!layer->outputPtr) {
            return false;
        }

        /* Update memory usage */
        scheduler->currentMemoryUsage += layer->outputSize;
        if (scheduler->currentMemoryUsage > scheduler->peakMemoryUsage) {
            scheduler->peakMemoryUsage = scheduler->currentMemoryUsage;
        }
    }

    /* Mark as executed */
    layer->executed = true;

    /* Free memory for outputs that are no longer needed */
    if (scheduler->mode == TINYAI_EXEC_MEMORY_OPT || scheduler->mode == TINYAI_EXEC_ADAPTIVE) {
        freeUnneededOutputs(scheduler);
    }

    /* Release layer weights if in memory optimized mode */
    if (scheduler->mode == TINYAI_EXEC_MEMORY_OPT) {
        tinyaiReleaseLayerWeights(scheduler->model, layer->layerIndex);
    }

    /* Return layer index if requested */
    if (layerIndex) {
        *layerIndex = layer->layerIndex;
    }

    /* For final layer, copy output if provided */
    if (output && nextLayer == scheduler->layerCount - 1 && layer->outputPtr) {
        memcpy(output, layer->outputPtr, layer->outputSize);
    }

    return true;
}

int tinyaiCalculateOptimalBatchSize(TinyAIForwardScheduler *scheduler, size_t inputSize,
                                    size_t outputSize, int maxBatchSize)
{

    if (!scheduler || maxBatchSize <= 0) {
        return 1;
    }

    /* If no memory limit, return maximum batch size */
    if (scheduler->maxMemory == 0) {
        return maxBatchSize;
    }

    /* Calculate per-sample memory usage */
    size_t perSampleMemory = inputSize + outputSize;

    /* Estimate intermediate activations based on layer outputs */
    size_t intermediateMemory = 0;
    for (int i = 0; i < scheduler->layerCount; i++) {
        intermediateMemory += scheduler->layers[i].outputSize;
    }

    /* Calculate model weights memory (fixed, doesn't scale with batch) */
    size_t weightsMemory = tinyaiGetMappedModelMemoryUsage(scheduler->model);

    /* Calculate batch memory requirements */
    size_t availableForBatch = 0;
    if (scheduler->maxMemory > weightsMemory + intermediateMemory) {
        availableForBatch = scheduler->maxMemory - weightsMemory - intermediateMemory;
    }

    /* Calculate batch size */
    int batchSize = availableForBatch / perSampleMemory;

    /* Ensure at least batch size 1 */
    if (batchSize < 1) {
        batchSize = 1;
    }

    /* Cap at maximum requested batch size */
    if (batchSize > maxBatchSize) {
        batchSize = maxBatchSize;
    }

    return batchSize;
}

size_t tinyaiGetSchedulerMemoryUsage(const TinyAIForwardScheduler *scheduler)
{
    if (!scheduler) {
        return 0;
    }

    return scheduler->currentMemoryUsage;
}

size_t tinyaiGetSchedulerPeakMemoryUsage(const TinyAIForwardScheduler *scheduler)
{
    if (!scheduler) {
        return 0;
    }

    return scheduler->peakMemoryUsage;
}

void tinyaiResetScheduler(TinyAIForwardScheduler *scheduler)
{
    if (!scheduler) {
        return;
    }

    /* Reset execution state, but keep allocated memory */
    for (int i = 0; i < scheduler->layerCount; i++) {
        scheduler->layers[i].executed     = false;
        scheduler->layers[i].outputNeeded = true;
    }

    scheduler->currentLayer    = -1;
    scheduler->peakMemoryUsage = scheduler->currentMemoryUsage;
}

bool tinyaiIsLayerExecuted(const TinyAIForwardScheduler *scheduler, int layerIndex)
{
    if (!scheduler || layerIndex < 0 || layerIndex >= scheduler->layerCount) {
        return false;
    }

    return scheduler->layers[layerIndex].executed;
}

void *tinyaiGetLayerOutput(const TinyAIForwardScheduler *scheduler, int layerIndex)
{
    if (!scheduler || layerIndex < 0 || layerIndex >= scheduler->layerCount) {
        return NULL;
    }

    /* Check if layer has been executed and output is available */
    if (!scheduler->layers[layerIndex].executed || !scheduler->layers[layerIndex].outputPtr) {
        return NULL;
    }

    return scheduler->layers[layerIndex].outputPtr;
}

void tinyaiMarkLayerOutputUnneeded(TinyAIForwardScheduler *scheduler, int layerIndex)
{
    if (!scheduler || layerIndex < 0 || layerIndex >= scheduler->layerCount) {
        return;
    }

    TinyAIExecLayer *layer = &scheduler->layers[layerIndex];

    /* Free output buffer if allocated */
    if (layer->outputPtr) {
        free(layer->outputPtr);
        layer->outputPtr = NULL;
        scheduler->currentMemoryUsage -= layer->outputSize;
    }

    /* Mark as unneeded */
    layer->outputNeeded = false;
}
