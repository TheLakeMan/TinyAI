/**
 * @file layer_scheduler.h
 * @brief Layer-wise computation scheduling for memory-efficient inference
 *
 * This header provides functionality for optimizing memory usage during neural
 * network inference by scheduling layer computations in a memory-aware fashion,
 * implementing activation checkpointing, and managing memory/speed trade-offs.
 */

#ifndef TINYAI_LAYER_SCHEDULER_H
#define TINYAI_LAYER_SCHEDULER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Layer types supported by the scheduler
 */
typedef enum {
    TINYAI_LAYER_LINEAR,        /**< Linear/fully connected layer */
    TINYAI_LAYER_CONV,          /**< Convolutional layer */
    TINYAI_LAYER_ATTENTION,     /**< Attention layer */
    TINYAI_LAYER_NORMALIZATION, /**< Normalization layer (e.g., LayerNorm) */
    TINYAI_LAYER_ACTIVATION,    /**< Activation function layer */
    TINYAI_LAYER_POOLING,       /**< Pooling layer */
    TINYAI_LAYER_EMBEDDING,     /**< Embedding layer */
    TINYAI_LAYER_CUSTOM         /**< Custom layer type */
} TinyAILayerType;

/**
 * @brief Memory usage strategy
 */
typedef enum {
    TINYAI_MEM_STRATEGY_DEFAULT,    /**< Default balanced strategy */
    TINYAI_MEM_STRATEGY_MIN_MEMORY, /**< Minimize memory usage (slower) */
    TINYAI_MEM_STRATEGY_MAX_SPEED,  /**< Maximize speed (more memory) */
    TINYAI_MEM_STRATEGY_ADAPTIVE    /**< Adapt based on runtime conditions */
} TinyAIMemoryStrategy;

/**
 * @brief Layer checkpoint policies
 */
typedef enum {
    TINYAI_CHECKPOINT_NONE,      /**< No checkpointing */
    TINYAI_CHECKPOINT_SELECTIVE, /**< Checkpoint selected layers */
    TINYAI_CHECKPOINT_ALL        /**< Checkpoint all eligible layers */
} TinyAICheckpointPolicy;

/**
 * @brief Forward function signature for layer computation
 *
 * @param layerData Layer-specific data
 * @param inputs Input tensor data
 * @param outputs Output tensor data
 * @param userData User-provided data
 * @return 0 on success, non-zero on failure
 */
typedef int (*TinyAILayerForwardFn)(void *layerData, void *inputs, void *outputs, void *userData);

/**
 * @brief Layer descriptor structure
 */
typedef struct {
    TinyAILayerType      type;               /**< Layer type */
    const char          *name;               /**< Layer name */
    size_t               inputSize;          /**< Size of input in bytes */
    size_t               outputSize;         /**< Size of output in bytes */
    size_t               workspaceSize;      /**< Size of workspace needed in bytes */
    TinyAILayerForwardFn forward;            /**< Forward function */
    bool                 checkpointEligible; /**< Whether this layer can be checkpointed */
    void                *layerData;          /**< Layer-specific data */
    bool                 inPlace;            /**< Whether layer can operate in-place */
} TinyAILayerDesc;

/**
 * @brief Scheduler configuration
 */
typedef struct {
    TinyAIMemoryStrategy   memoryStrategy;   /**< Memory usage strategy */
    TinyAICheckpointPolicy checkpointPolicy; /**< Checkpoint policy */
    size_t                 maxMemory; /**< Maximum memory budget in bytes (0 for unlimited) */
    size_t                 preferredWorkspaceSize; /**< Preferred workspace size in bytes */
    bool                   allowInPlace;           /**< Whether to allow in-place operations */
    bool                   optimizeOverlap; /**< Whether to optimize tensor lifetime overlap */
    bool                   verbose;         /**< Whether to print verbose information */
} TinyAILayerSchedulerConfig;

/**
 * @brief Layer scheduler handle
 */
typedef struct TinyAILayerScheduler TinyAILayerScheduler;

/**
 * @brief Execution statistics
 */
typedef struct {
    size_t   peakMemoryUsage;      /**< Peak memory usage in bytes */
    size_t   totalMemoryAllocated; /**< Total memory allocated in bytes */
    size_t   numCheckpoints;       /**< Number of checkpoints created */
    size_t   numRecomputations;    /**< Number of layer recomputations */
    size_t   layerExecutionCount;  /**< Total number of layer executions */
    uint64_t totalExecutionTimeNs; /**< Total execution time in nanoseconds */
} TinyAIExecutionStats;

/**
 * @brief Get default scheduler configuration
 *
 * @param config Pointer to configuration struct to fill
 */
void tinyaiLayerSchedulerGetDefaultConfig(TinyAILayerSchedulerConfig *config);

/**
 * @brief Create a new layer scheduler
 *
 * @param config Scheduler configuration
 * @return New scheduler instance or NULL on failure
 */
TinyAILayerScheduler *tinyaiLayerSchedulerCreate(const TinyAILayerSchedulerConfig *config);

/**
 * @brief Destroy a layer scheduler
 *
 * @param scheduler Scheduler to destroy
 */
void tinyaiLayerSchedulerDestroy(TinyAILayerScheduler *scheduler);

/**
 * @brief Add a layer to the scheduler
 *
 * @param scheduler Target scheduler
 * @param layer Layer descriptor
 * @return Layer ID on success, negative value on failure
 */
int tinyaiLayerSchedulerAddLayer(TinyAILayerScheduler *scheduler, const TinyAILayerDesc *layer);

/**
 * @brief Add a dependency between layers
 *
 * @param scheduler Target scheduler
 * @param sourceLayerId Source layer ID
 * @param targetLayerId Target layer ID
 * @return 0 on success, non-zero on failure
 */
int tinyaiLayerSchedulerAddDependency(TinyAILayerScheduler *scheduler, int sourceLayerId,
                                      int targetLayerId);

/**
 * @brief Prepare the scheduler for execution
 *
 * This function analyzes the layer graph, determines execution order, and
 * allocates required memory.
 *
 * @param scheduler Target scheduler
 * @return 0 on success, non-zero on failure
 */
int tinyaiLayerSchedulerPrepare(TinyAILayerScheduler *scheduler);

/**
 * @brief Execute all layers in the scheduler
 *
 * @param scheduler Target scheduler
 * @param inputData Input data for the first layer
 * @param outputData Output data buffer for the final layer
 * @param userData User data passed to layer forward functions
 * @return 0 on success, non-zero on failure
 */
int tinyaiLayerSchedulerExecute(TinyAILayerScheduler *scheduler, void *inputData, void *outputData,
                                void *userData);

/**
 * @brief Set checkpoint policy for a specific layer
 *
 * @param scheduler Target scheduler
 * @param layerId Layer ID
 * @param shouldCheckpoint Whether to checkpoint this layer
 * @return 0 on success, non-zero on failure
 */
int tinyaiLayerSchedulerSetLayerCheckpoint(TinyAILayerScheduler *scheduler, int layerId,
                                           bool shouldCheckpoint);

/**
 * @brief Get execution statistics
 *
 * @param scheduler Target scheduler
 * @param stats Pointer to stats struct to fill
 */
void tinyaiLayerSchedulerGetStats(TinyAILayerScheduler *scheduler, TinyAIExecutionStats *stats);

/**
 * @brief Reset scheduler state
 *
 * Clears execution state without changing the layer graph.
 *
 * @param scheduler Target scheduler
 */
void tinyaiLayerSchedulerReset(TinyAILayerScheduler *scheduler);

/**
 * @brief Estimate memory requirements
 *
 * @param scheduler Target scheduler
 * @param peakMemory Pointer to store peak memory estimate
 * @param totalMemory Pointer to store total memory estimate
 * @return 0 on success, non-zero on failure
 */
int tinyaiLayerSchedulerEstimateMemory(TinyAILayerScheduler *scheduler, size_t *peakMemory,
                                       size_t *totalMemory);

/**
 * @brief Set memory strategy
 *
 * @param scheduler Target scheduler
 * @param strategy New memory strategy
 */
void tinyaiLayerSchedulerSetMemoryStrategy(TinyAILayerScheduler *scheduler,
                                           TinyAIMemoryStrategy  strategy);

/**
 * @brief Set checkpoint policy
 *
 * @param scheduler Target scheduler
 * @param policy New checkpoint policy
 */
void tinyaiLayerSchedulerSetCheckpointPolicy(TinyAILayerScheduler  *scheduler,
                                             TinyAICheckpointPolicy policy);

/**
 * @brief Dump scheduler information for debugging
 *
 * @param scheduler Target scheduler
 */
void tinyaiLayerSchedulerDump(TinyAILayerScheduler *scheduler);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TINYAI_LAYER_SCHEDULER_H */