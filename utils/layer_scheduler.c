/**
 * @file layer_scheduler.c
 * @brief Implementation of layer-wise computation scheduling for memory-efficient inference
 */

#include "layer_scheduler.h"
#include "../core/logging.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

/* Maximum number of layers supported */
#define MAX_LAYERS 1024

/* Maximum number of dependencies per layer */
#define MAX_DEPS_PER_LAYER 64

/* Maximum number of checkpoints */
#define MAX_CHECKPOINTS 256

/* Threshold for adaptive strategy switching */
#define MEMORY_PRESSURE_THRESHOLD 0.9f

/* Typical checkpoint overhead factor */
#define CHECKPOINT_OVERHEAD_FACTOR 1.1f

/* States for topological sorting */
typedef enum {
    TINYAI_NODE_NOT_VISITED = 0,
    TINYAI_NODE_VISITING    = 1,
    TINYAI_NODE_VISITED     = 2
} NodeVisitState;

/* Checkpoint record */
typedef struct {
    int    layerId;
    void  *data;
    size_t size;
    bool   isActive;
} Checkpoint;

/* Layer execution record */
typedef struct {
    int   layerId;
    void *input;
    void *output;
    bool  inputIsCheckpoint;
    int   checkpointId;
} ExecutionRecord;

/* Layer in scheduler */
typedef struct {
    TinyAILayerDesc desc;
    int             id;
    int             numDependencies;
    int             dependencies[MAX_DEPS_PER_LAYER];
    int             numDependents;
    int             dependents[MAX_DEPS_PER_LAYER];
    bool            shouldCheckpoint;
    int             checkpointId;
    bool            visited;
    NodeVisitState  visitState;
} Layer;

/* Layer scheduler structure */
struct TinyAILayerScheduler {
    /* Configuration */
    TinyAILayerSchedulerConfig config;

    /* Layers and execution plan */
    Layer layers[MAX_LAYERS];
    int   numLayers;
    int  *executionOrder;
    int   executionOrderLength;

    /* Checkpoint management */
    Checkpoint checkpoints[MAX_CHECKPOINTS];
    int        numCheckpoints;

    /* Memory management */
    void  *workspace;
    size_t workspaceSize;

    /* Execution statistics */
    TinyAIExecutionStats stats;

    /* Execution state */
    bool             isPrepared;
    ExecutionRecord *executionRecords;
    int              numExecutionRecords;
};

/**
 * Get high resolution time in nanoseconds
 */
static uint64_t getTimeNs()
{
#ifdef _WIN32
    /* Windows implementation */
    static LARGE_INTEGER frequency;
    static int           initialized = 0;
    LARGE_INTEGER        count;

    /* Initialize frequency on first call */
    if (!initialized) {
        QueryPerformanceFrequency(&frequency);
        initialized = 1;
    }

    /* Get current count */
    QueryPerformanceCounter(&count);

    /* Convert to nanoseconds */
    return (uint64_t)((double)count.QuadPart * 1000000000.0 / (double)frequency.QuadPart);
#else
    /* Linux/Unix implementation */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

/**
 * Get default scheduler configuration
 */
void tinyaiLayerSchedulerGetDefaultConfig(TinyAILayerSchedulerConfig *config)
{
    if (!config) {
        return;
    }

    /* Initialize with zeros */
    memset(config, 0, sizeof(TinyAILayerSchedulerConfig));

    /* Set default values */
    config->memoryStrategy         = TINYAI_MEM_STRATEGY_DEFAULT;
    config->checkpointPolicy       = TINYAI_CHECKPOINT_SELECTIVE;
    config->maxMemory              = 0;               /* Unlimited */
    config->preferredWorkspaceSize = 4 * 1024 * 1024; /* 4 MB */
    config->allowInPlace           = true;
    config->optimizeOverlap        = true;
    config->verbose                = false;
}

/**
 * Create a new layer scheduler
 */
TinyAILayerScheduler *tinyaiLayerSchedulerCreate(const TinyAILayerSchedulerConfig *config)
{
    TinyAILayerScheduler *scheduler;

    /* Validate config */
    if (!config) {
        return NULL;
    }

    /* Allocate scheduler */
    scheduler = (TinyAILayerScheduler *)malloc(sizeof(TinyAILayerScheduler));
    if (!scheduler) {
        return NULL;
    }

    /* Initialize with zeros */
    memset(scheduler, 0, sizeof(TinyAILayerScheduler));

    /* Copy configuration */
    memcpy(&scheduler->config, config, sizeof(TinyAILayerSchedulerConfig));

    /* Initialize other fields */
    scheduler->numLayers            = 0;
    scheduler->executionOrder       = NULL;
    scheduler->executionOrderLength = 0;
    scheduler->numCheckpoints       = 0;
    scheduler->workspace            = NULL;
    scheduler->workspaceSize        = 0;
    scheduler->isPrepared           = false;
    scheduler->executionRecords     = NULL;
    scheduler->numExecutionRecords  = 0;

    return scheduler;
}

/**
 * Destroy a layer scheduler
 */
void tinyaiLayerSchedulerDestroy(TinyAILayerScheduler *scheduler)
{
    int i;

    if (!scheduler) {
        return;
    }

    /* Free execution order */
    if (scheduler->executionOrder) {
        free(scheduler->executionOrder);
        scheduler->executionOrder = NULL;
    }

    /* Free execution records */
    if (scheduler->executionRecords) {
        free(scheduler->executionRecords);
        scheduler->executionRecords = NULL;
    }

    /* Free checkpoints */
    for (i = 0; i < scheduler->numCheckpoints; i++) {
        if (scheduler->checkpoints[i].data) {
            free(scheduler->checkpoints[i].data);
            scheduler->checkpoints[i].data = NULL;
        }
    }

    /* Free workspace */
    if (scheduler->workspace) {
        free(scheduler->workspace);
        scheduler->workspace = NULL;
    }

    /* Free scheduler */
    free(scheduler);
}

/**
 * Add a layer to the scheduler
 */
int tinyaiLayerSchedulerAddLayer(TinyAILayerScheduler *scheduler, const TinyAILayerDesc *layer)
{
    Layer *newLayer;

    if (!scheduler || !layer) {
        return -1;
    }

    /* Check if we have room for another layer */
    if (scheduler->numLayers >= MAX_LAYERS) {
        return -1;
    }

    /* Get pointer to the new layer */
    newLayer = &scheduler->layers[scheduler->numLayers];

    /* Copy layer descriptor */
    memcpy(&newLayer->desc, layer, sizeof(TinyAILayerDesc));

    /* Initialize layer */
    newLayer->id               = scheduler->numLayers;
    newLayer->numDependencies  = 0;
    newLayer->numDependents    = 0;
    newLayer->shouldCheckpoint = false;
    newLayer->checkpointId     = -1;
    newLayer->visited          = false;
    newLayer->visitState       = TINYAI_NODE_NOT_VISITED;

    /* Increment layer count */
    scheduler->numLayers++;

    /* Mark scheduler as not prepared */
    scheduler->isPrepared = false;

    return newLayer->id;
}

/**
 * Add a dependency between layers
 */
int tinyaiLayerSchedulerAddDependency(TinyAILayerScheduler *scheduler, int sourceLayerId,
                                      int targetLayerId)
{
    Layer *sourceLayer, *targetLayer;

    if (!scheduler) {
        return -1;
    }

    /* Validate layer IDs */
    if (sourceLayerId < 0 || sourceLayerId >= scheduler->numLayers || targetLayerId < 0 ||
        targetLayerId >= scheduler->numLayers) {
        return -1;
    }

    /* Get pointers to layers */
    sourceLayer = &scheduler->layers[sourceLayerId];
    targetLayer = &scheduler->layers[targetLayerId];

    /* Check if we have room for another dependency */
    if (targetLayer->numDependencies >= MAX_DEPS_PER_LAYER) {
        return -1;
    }

    /* Check if we have room for another dependent */
    if (sourceLayer->numDependents >= MAX_DEPS_PER_LAYER) {
        return -1;
    }

    /* Add dependency to target layer */
    targetLayer->dependencies[targetLayer->numDependencies++] = sourceLayerId;

    /* Add dependent to source layer */
    sourceLayer->dependents[sourceLayer->numDependents++] = targetLayerId;

    /* Mark scheduler as not prepared */
    scheduler->isPrepared = false;

    return 0;
}

/**
 * Add a node to the execution order in topological sort
 */
static int addToExecutionOrder(TinyAILayerScheduler *scheduler, int layerId, int *order,
                               int *orderIndex)
{
    int    i, depId;
    Layer *layer;

    if (!scheduler || !order || !orderIndex) {
        return -1;
    }

    /* Validate layer ID */
    if (layerId < 0 || layerId >= scheduler->numLayers) {
        return -1;
    }

    /* Get pointer to layer */
    layer = &scheduler->layers[layerId];

    /* Check for cycles */
    if (layer->visitState == TINYAI_NODE_VISITING) {
        return -1; /* Cycle detected */
    }

    /* Skip if already visited */
    if (layer->visitState == TINYAI_NODE_VISITED) {
        return 0;
    }

    /* Mark as visiting */
    layer->visitState = TINYAI_NODE_VISITING;

    /* Visit all dependencies */
    for (i = 0; i < layer->numDependencies; i++) {
        depId = layer->dependencies[i];
        if (addToExecutionOrder(scheduler, depId, order, orderIndex) != 0) {
            return -1;
        }
    }

    /* Mark as visited */
    layer->visitState = TINYAI_NODE_VISITED;

    /* Add to execution order */
    order[(*orderIndex)++] = layerId;

    return 0;
}

/**
 * Generate execution order using topological sort
 */
static int generateExecutionOrder(TinyAILayerScheduler *scheduler)
{
    int  i, ret, orderIndex = 0;
    int *order;

    if (!scheduler) {
        return -1;
    }

    /* Allocate memory for execution order */
    order = (int *)malloc(scheduler->numLayers * sizeof(int));
    if (!order) {
        return -1;
    }

    /* Initialize visit state */
    for (i = 0; i < scheduler->numLayers; i++) {
        scheduler->layers[i].visitState = TINYAI_NODE_NOT_VISITED;
    }

    /* Perform topological sort */
    for (i = 0; i < scheduler->numLayers; i++) {
        if (scheduler->layers[i].visitState == TINYAI_NODE_NOT_VISITED) {
            ret = addToExecutionOrder(scheduler, i, order, &orderIndex);
            if (ret != 0) {
                free(order);
                return -1;
            }
        }
    }

    /* Free old execution order if any */
    if (scheduler->executionOrder) {
        free(scheduler->executionOrder);
    }

    /* Set new execution order */
    scheduler->executionOrder       = order;
    scheduler->executionOrderLength = orderIndex;

    return 0;
}

/**
 * Determine which layers should be checkpointed
 */
static void determineCheckpoints(TinyAILayerScheduler *scheduler)
{
    int    i, j, layerId;
    Layer *layer;

    if (!scheduler) {
        return;
    }

    /* Reset checkpoint flags */
    for (i = 0; i < scheduler->numLayers; i++) {
        scheduler->layers[i].shouldCheckpoint = false;
    }

    /* Apply checkpoint policy */
    switch (scheduler->config.checkpointPolicy) {
    case TINYAI_CHECKPOINT_NONE:
        /* No checkpoints */
        break;

    case TINYAI_CHECKPOINT_ALL:
        /* Checkpoint all eligible layers */
        for (i = 0; i < scheduler->numLayers; i++) {
            layer = &scheduler->layers[i];
            if (layer->desc.checkpointEligible) {
                layer->shouldCheckpoint = true;
            }
        }
        break;

    case TINYAI_CHECKPOINT_SELECTIVE:
        /* Checkpoint layers with multiple dependents */
        for (i = 0; i < scheduler->executionOrderLength; i++) {
            layerId = scheduler->executionOrder[i];
            layer   = &scheduler->layers[layerId];

            /* Check if this layer is eligible and has multiple dependents */
            if (layer->desc.checkpointEligible && layer->numDependents > 1) {
                layer->shouldCheckpoint = true;
            }

            /* For memory-constrained strategies, also checkpoint layers with large outputs */
            if (scheduler->config.memoryStrategy == TINYAI_MEM_STRATEGY_MIN_MEMORY) {
                if (layer->desc.checkpointEligible && layer->desc.outputSize > (1024 * 1024)) {
                    layer->shouldCheckpoint = true;
                }
            }
        }
        break;
    }

    /* Override with per-layer settings */
    for (i = 0; i < scheduler->numLayers; i++) {
        if (scheduler->layers[i].checkpointId >= 0) {
            scheduler->layers[i].shouldCheckpoint = true;
        }
    }
}

/**
 * Estimate memory requirements for execution
 */
static void estimateMemoryRequirements(TinyAILayerScheduler *scheduler, size_t *peakMemory,
                                       size_t *totalMemory)
{
    int    i, j, layerId;
    size_t currentMemory    = 0;
    size_t peak             = 0;
    size_t total            = 0;
    size_t checkpointMemory = 0;
    Layer *layer;
    bool  *outputInUse;

    if (!scheduler || !peakMemory || !totalMemory) {
        return;
    }

    /* Allocate output usage tracking array */
    outputInUse = (bool *)calloc(scheduler->numLayers, sizeof(bool));
    if (!outputInUse) {
        *peakMemory  = 0;
        *totalMemory = 0;
        return;
    }

    /* Initialize memory counters */
    currentMemory    = 0;
    peak             = 0;
    total            = 0;
    checkpointMemory = 0;

    /* Simulate execution to estimate memory usage */
    for (i = 0; i < scheduler->executionOrderLength; i++) {
        layerId = scheduler->executionOrder[i];
        layer   = &scheduler->layers[layerId];

        /* Add input memory if not already counted */
        if (!outputInUse[layerId]) {
            currentMemory += layer->desc.inputSize;
            total += layer->desc.inputSize;
        }

        /* Add output memory */
        currentMemory += layer->desc.outputSize;
        total += layer->desc.outputSize;

        /* Add workspace memory */
        currentMemory += layer->desc.workspaceSize;

        /* Track peak memory */
        if (currentMemory > peak) {
            peak = currentMemory;
        }

        /* Release workspace memory */
        currentMemory -= layer->desc.workspaceSize;

        /* Add checkpoint memory if needed */
        if (layer->shouldCheckpoint) {
            checkpointMemory += (size_t)(layer->desc.outputSize * CHECKPOINT_OVERHEAD_FACTOR);
        }

        /* Mark this layer's output as in use */
        outputInUse[layerId] = true;

        /* Check dependents to see if any outputs can be freed */
        for (j = 0; j < layer->numDependents; j++) {
            int  depId       = layer->dependents[j];
            bool stillNeeded = false;
            int  k;

            /* Check if this output is still needed by any remaining layer */
            for (k = i + 1; k < scheduler->executionOrderLength; k++) {
                int    remainingLayerId = scheduler->executionOrder[k];
                Layer *remainingLayer   = &scheduler->layers[remainingLayerId];
                int    d;

                for (d = 0; d < remainingLayer->numDependencies; d++) {
                    if (remainingLayer->dependencies[d] == layerId) {
                        stillNeeded = true;
                        break;
                    }
                }

                if (stillNeeded) {
                    break;
                }
            }

            /* Free output memory if no longer needed */
            if (!stillNeeded && !layer->shouldCheckpoint) {
                currentMemory -= layer->desc.outputSize;
            }
        }
    }

    /* Add checkpoint memory to the peak */
    peak += checkpointMemory;
    total += checkpointMemory;

    /* Free output usage tracking array */
    free(outputInUse);

    /* Return estimates */
    *peakMemory  = peak;
    *totalMemory = total;
}

/**
 * Prepare execution records
 */
static int prepareExecutionRecords(TinyAILayerScheduler *scheduler)
{
    int    i, layerId;
    int    recordCount = 0;
    Layer *layer;

    if (!scheduler) {
        return -1;
    }

    /* Free old execution records if any */
    if (scheduler->executionRecords) {
        free(scheduler->executionRecords);
        scheduler->executionRecords = NULL;
    }

    /* Allocate memory for execution records */
    scheduler->executionRecords =
        (ExecutionRecord *)malloc(scheduler->executionOrderLength * sizeof(ExecutionRecord));
    if (!scheduler->executionRecords) {
        return -1;
    }

    /* Initialize execution records */
    memset(scheduler->executionRecords, 0,
           scheduler->executionOrderLength * sizeof(ExecutionRecord));

    /* Create execution records */
    for (i = 0; i < scheduler->executionOrderLength; i++) {
        layerId = scheduler->executionOrder[i];
        layer   = &scheduler->layers[layerId];

        /* Set layer ID */
        scheduler->executionRecords[recordCount].layerId = layerId;

        /* Input and output pointers will be set during execution */
        scheduler->executionRecords[recordCount].input             = NULL;
        scheduler->executionRecords[recordCount].output            = NULL;
        scheduler->executionRecords[recordCount].inputIsCheckpoint = false;
        scheduler->executionRecords[recordCount].checkpointId      = -1;

        recordCount++;
    }

    scheduler->numExecutionRecords = recordCount;

    return 0;
}

/**
 * Prepare the scheduler for execution
 */
int tinyaiLayerSchedulerPrepare(TinyAILayerScheduler *scheduler)
{
    size_t peakMemory, totalMemory;
    int    ret;

    if (!scheduler) {
        return -1;
    }

    /* Generate execution order using topological sort */
    ret = generateExecutionOrder(scheduler);
    if (ret != 0) {
        return -1;
    }

    /* Determine which layers should be checkpointed */
    determineCheckpoints(scheduler);

    /* Estimate memory requirements */
    estimateMemoryRequirements(scheduler, &peakMemory, &totalMemory);

    /* Validate memory requirements against constraints */
    if (scheduler->config.maxMemory > 0 && peakMemory > scheduler->config.maxMemory) {
        /* Memory requirement exceeds budget */
        /* Try more aggressive checkpointing to reduce memory usage */
        scheduler->config.checkpointPolicy = TINYAI_CHECKPOINT_ALL;
        determineCheckpoints(scheduler);
        estimateMemoryRequirements(scheduler, &peakMemory, &totalMemory);

        /* Still exceeds budget? */
        if (peakMemory > scheduler->config.maxMemory) {
            return -1;
        }
    }

    /* Allocate workspace */
    if (scheduler->workspace) {
        free(scheduler->workspace);
    }

    scheduler->workspaceSize = scheduler->config.preferredWorkspaceSize;
    scheduler->workspace     = malloc(scheduler->workspaceSize);
    if (!scheduler->workspace) {
        return -1;
    }

    /* Prepare execution records */
    ret = prepareExecutionRecords(scheduler);
    if (ret != 0) {
        return -1;
    }

    /* Mark as prepared */
    scheduler->isPrepared = true;

    return 0;
}

/**
 * Find or create a checkpoint
 */
static int findOrCreateCheckpoint(TinyAILayerScheduler *scheduler, int layerId, size_t size)
{
    int i, checkpointId = -1;

    if (!scheduler) {
        return -1;
    }

    /* Check if this layer already has a checkpoint */
    for (i = 0; i < scheduler->numCheckpoints; i++) {
        if (scheduler->checkpoints[i].layerId == layerId) {
            /* Found existing checkpoint */
            checkpointId = i;
            break;
        }
    }

    /* Create new checkpoint if needed */
    if (checkpointId < 0) {
        /* Check if we have room for another checkpoint */
        if (scheduler->numCheckpoints >= MAX_CHECKPOINTS) {
            return -1;
        }

        /* Allocate memory for checkpoint data */
        void *data = malloc(size);
        if (!data) {
            return -1;
        }

        /* Initialize checkpoint */
        checkpointId                                  = scheduler->numCheckpoints;
        scheduler->checkpoints[checkpointId].layerId  = layerId;
        scheduler->checkpoints[checkpointId].data     = data;
        scheduler->checkpoints[checkpointId].size     = size;
        scheduler->checkpoints[checkpointId].isActive = false;

        /* Increment checkpoint count */
        scheduler->numCheckpoints++;
    }

    return checkpointId;
}

/**
 * Execute a layer
 */
static int executeLayer(TinyAILayerScheduler *scheduler, int layerId, void *inputData,
                        void *outputData)
{
    Layer   *layer;
    uint64_t startTime, endTime;
    int      ret;

    if (!scheduler) {
        return -1;
    }

    /* Validate layer ID */
    if (layerId < 0 || layerId >= scheduler->numLayers) {
        return -1;
    }

    /* Get pointer to layer */
    layer = &scheduler->layers[layerId];

    /* Measure execution time */
    startTime = getTimeNs();

    /* Execute layer */
    ret = layer->desc.forward(layer->desc.layerData, inputData, outputData, NULL);

    /* Update execution time */
    endTime = getTimeNs();
    scheduler->stats.totalExecutionTimeNs += (endTime - startTime);
    scheduler->stats.layerExecutionCount++;

    return ret;
}

/**
 * Create a checkpoint for a layer
 */
static int createCheckpoint(TinyAILayerScheduler *scheduler, int layerId, void *outputData)
{
    Layer *layer;
    int    checkpointId;

    if (!scheduler) {
        return -1;
    }

    /* Validate layer ID */
    if (layerId < 0 || layerId >= scheduler->numLayers) {
        return -1;
    }

    /* Get pointer to layer */
    layer = &scheduler->layers[layerId];

    /* Find or create checkpoint */
    checkpointId = findOrCreateCheckpoint(scheduler, layerId, layer->desc.outputSize);
    if (checkpointId < 0) {
        return -1;
    }

    /* Copy output data to checkpoint */
    memcpy(scheduler->checkpoints[checkpointId].data, outputData, layer->desc.outputSize);

    /* Mark checkpoint as active */
    scheduler->checkpoints[checkpointId].isActive = true;

    /* Update statistics */
    scheduler->stats.numCheckpoints++;

    return checkpointId;
}

/**
 * Execute all layers in the scheduler
 */
int tinyaiLayerSchedulerExecute(TinyAILayerScheduler *scheduler, void *inputData, void *outputData,
                                void *userData)
{
    int    i, j, layerId, dependencyId, checkpointId;
    Layer *layer, *dependencyLayer;
    void  *layerInput, *layerOutput;
    int    ret;

    if (!scheduler || !inputData || !outputData) {
        return -1;
    }

    /* Check if scheduler is prepared */
    if (!scheduler->isPrepared) {
        ret = tinyaiLayerSchedulerPrepare(scheduler);
        if (ret != 0) {
            return -1;
        }
    }

    /* Reset statistics */
    memset(&scheduler->stats, 0, sizeof(TinyAIExecutionStats));

    /* Clear checkpoint active flags */
    for (i = 0; i < scheduler->numCheckpoints; i++) {
        scheduler->checkpoints[i].isActive = false;
    }

    /* Execute layers in order */
    for (i = 0; i < scheduler->numExecutionRecords; i++) {
        layerId = scheduler->executionRecords[i].layerId;
        layer   = &scheduler->layers[layerId];

        /* Determine input and output pointers */
        if (i == 0) {
            /* First layer uses provided input */
            layerInput = inputData;
        }
        else {
            /* Get input from previous layer or checkpoint */
            layerInput = NULL;

            /* Check dependencies */
            for (j = 0; j < layer->numDependencies; j++) {
                dependencyId    = layer->dependencies[j];
                dependencyLayer = &scheduler->layers[dependencyId];

                /* Check if this dependency was checkpointed */
                checkpointId = -1;
                for (int k = 0; k < scheduler->numCheckpoints; k++) {
                    if (scheduler->checkpoints[k].layerId == dependencyId &&
                        scheduler->checkpoints[k].isActive) {
                        checkpointId = k;
                        break;
                    }
                }

                if (checkpointId >= 0) {
                    /* Use checkpoint as input */
                    layerInput = scheduler->checkpoints[checkpointId].data;
                    scheduler->executionRecords[i].inputIsCheckpoint = true;
                    scheduler->executionRecords[i].checkpointId      = checkpointId;
                }
                else {
                    /* Need to recompute input */
                    int depIndex = -1;

                    /* Find the execution record for this dependency */
                    for (int k = 0; k < i; k++) {
                        if (scheduler->executionRecords[k].layerId == dependencyId) {
                            depIndex = k;
                            break;
                        }
                    }

                    if (depIndex >= 0) {
                        layerInput = scheduler->executionRecords[depIndex].output;
                    }
                    else {
                        /* Dependency not found, should not happen */
                        return -1;
                    }
                }
            }
        }

        if (i == scheduler->numExecutionRecords - 1) {
            /* Last layer uses provided output */
            layerOutput = outputData;
        }
        else {
            /* Allocate output from workspace */
            if (layer->desc.outputSize > scheduler->workspaceSize) {
                /* Not enough workspace memory */
                return -1;
            }

            layerOutput = scheduler->workspace;
        }

        /* Record input and output */
        scheduler->executionRecords[i].input  = layerInput;
        scheduler->executionRecords[i].output = layerOutput;

        /* Execute layer */
        ret = executeLayer(scheduler, layerId, layerInput, layerOutput);
        if (ret != 0) {
            return -1;
        }

        /* Create checkpoint if needed */
        if (layer->shouldCheckpoint) {
            checkpointId = createCheckpoint(scheduler, layerId, layerOutput);
            if (checkpointId < 0) {
                return -1;
            }
        }
    }

    /* Update statistics */
    scheduler->stats.peakMemoryUsage      = 0; /* Not tracked yet */
    scheduler->stats.totalMemoryAllocated = 0; /* Not tracked yet */

    return 0;
}

/**
 * Set checkpoint policy for a specific layer
 */
int tinyaiLayerSchedulerSetLayerCheckpoint(TinyAILayerScheduler *scheduler, int layerId,
                                           bool shouldCheckpoint)
{
    if (!scheduler) {
        return -1;
    }

    /* Validate layer ID */
    if (layerId < 0 || layerId >= scheduler->numLayers) {
        return -1;
    }

    /* Set checkpoint flag */
    scheduler->layers[layerId].shouldCheckpoint = shouldCheckpoint;

    /* Mark as not prepared */
    scheduler->isPrepared = false;

    return 0;
}

/**
 * Get execution statistics
 */
void tinyaiLayerSchedulerGetStats(TinyAILayerScheduler *scheduler, TinyAIExecutionStats *stats)
{
    if (!scheduler || !stats) {
        return;
    }

    /* Copy statistics */
    memcpy(stats, &scheduler->stats, sizeof(TinyAIExecutionStats));
}

/**
 * Reset scheduler state
 */
void tinyaiLayerSchedulerReset(TinyAILayerScheduler *scheduler)
{
    int i;

    if (!scheduler) {
        return;
    }

    /* Reset execution records */
    if (scheduler->executionRecords) {
        memset(scheduler->executionRecords, 0,
               scheduler->numExecutionRecords * sizeof(ExecutionRecord));
    }

    /* Reset checkpoint active flags */
    for (i = 0; i < scheduler->numCheckpoints; i++) {
        scheduler->checkpoints[i].isActive = false;
    }

    /* Reset statistics */
    memset(&scheduler->stats, 0, sizeof(TinyAIExecutionStats));
}

/**
 * Estimate memory requirements
 */
int tinyaiLayerSchedulerEstimateMemory(TinyAILayerScheduler *scheduler, size_t *peakMemory,
                                       size_t *totalMemory)
{
    if (!scheduler || !peakMemory || !totalMemory) {
        return -1;
    }

    /* Check if scheduler is prepared */
    if (!scheduler->isPrepared) {
        int ret = tinyaiLayerSchedulerPrepare(scheduler);
        if (ret != 0) {
            return -1;
        }
    }

    /* Estimate memory requirements */
    estimateMemoryRequirements(scheduler, peakMemory, totalMemory);

    return 0;
}

/**
 * Set memory strategy
 */
void tinyaiLayerSchedulerSetMemoryStrategy(TinyAILayerScheduler *scheduler,
                                           TinyAIMemoryStrategy  strategy)
{
    if (!scheduler) {
        return;
    }

    /* Set strategy */
    scheduler->config.memoryStrategy = strategy;

    /* Mark as not prepared */
    scheduler->isPrepared = false;
}

/**
 * Set checkpoint policy
 */
void tinyaiLayerSchedulerSetCheckpointPolicy(TinyAILayerScheduler  *scheduler,
                                             TinyAICheckpointPolicy policy)
{
    if (!scheduler) {
        return;
    }

    /* Set policy */
    scheduler->config.checkpointPolicy = policy;

    /* Mark as not prepared */
    scheduler->isPrepared = false;
}

/**
 * Dump scheduler information for debugging
 */
void tinyaiLayerSchedulerDump(TinyAILayerScheduler *scheduler)
{
    int i, j;

    if (!scheduler) {
        return;
    }

    /* Print scheduler information */
    printf("Layer Scheduler:\n");
    printf("  Layers: %d\n", scheduler->numLayers);
    printf("  Execution Order Length: %d\n", scheduler->executionOrderLength);
    printf("  Checkpoints: %d\n", scheduler->numCheckpoints);
    printf("  Workspace Size: %zu bytes\n", scheduler->workspaceSize);
    printf("  Prepared: %s\n", scheduler->isPrepared ? "Yes" : "No");

    /* Print configuration */
    printf("  Configuration:\n");
    printf("    Memory Strategy: %d\n", scheduler->config.memoryStrategy);
    printf("    Checkpoint Policy: %d\n", scheduler->config.checkpointPolicy);
    printf("    Max Memory: %zu bytes\n", scheduler->config.maxMemory);
    printf("    Preferred Workspace Size: %zu bytes\n", scheduler->config.preferredWorkspaceSize);
    printf("    Allow In-Place: %s\n", scheduler->config.allowInPlace ? "Yes" : "No");
    printf("    Optimize Overlap: %s\n", scheduler->config.optimizeOverlap ? "Yes" : "No");
    printf("    Verbose: %s\n", scheduler->config.verbose ? "Yes" : "No");

    /* Print statistics */
    printf("  Statistics:\n");
    printf("    Peak Memory Usage: %zu bytes\n", scheduler->stats.peakMemoryUsage);
    printf("    Total Memory Allocated: %zu bytes\n", scheduler->stats.totalMemoryAllocated);
    printf("    Checkpoints: %zu\n", scheduler->stats.numCheckpoints);
    printf("    Recomputations: %zu\n", scheduler->stats.numRecomputations);
    printf("    Layer Executions: %zu\n", scheduler->stats.layerExecutionCount);
    printf("    Total Execution Time: %lu ns\n", scheduler->stats.totalExecutionTimeNs);

    /* Print layers */
    printf("  Layers:\n");
    for (i = 0; i < scheduler->numLayers; i++) {
        Layer *layer = &scheduler->layers[i];
        printf("    [%d] %s (Type: %d)\n", layer->id, layer->desc.name, layer->desc.type);
        printf("      Input Size: %zu bytes\n", layer->desc.inputSize);
        printf("      Output Size: %zu bytes\n", layer->desc.outputSize);
        printf("      Workspace Size: %zu bytes\n", layer->desc.workspaceSize);
        printf("      Checkpoint: %s\n", layer->shouldCheckpoint ? "Yes" : "No");

        /* Print dependencies */
        if (layer->numDependencies > 0) {
            printf("      Dependencies: ");
            for (j = 0; j < layer->numDependencies; j++) {
                printf("%d ", layer->dependencies[j]);
            }
            printf("\n");
        }

        /* Print dependents */
        if (layer->numDependents > 0) {
            printf("      Dependents: ");
            for (j = 0; j < layer->numDependents; j++) {
                printf("%d ", layer->dependents[j]);
            }
            printf("\n");
        }
    }

    /* Print execution order */
    if (scheduler->executionOrder) {
        printf("  Execution Order: ");
        for (i = 0; i < scheduler->executionOrderLength; i++) {
            printf("%d ", scheduler->executionOrder[i]);
        }
        printf("\n");
    }

    /* Print checkpoints */
    if (scheduler->numCheckpoints > 0) {
        printf("  Checkpoints:\n");
        for (i = 0; i < scheduler->numCheckpoints; i++) {
            Checkpoint *checkpoint = &scheduler->checkpoints[i];
            printf("    [%d] Layer %d, Size: %zu bytes, Active: %s\n", i, checkpoint->layerId,
                   checkpoint->size, checkpoint->isActive ? "Yes" : "No");
        }
    }
}