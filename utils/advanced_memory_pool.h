/**
 * @file advanced_memory_pool.h
 * @brief Advanced memory pooling system for TinyAI
 *
 * This header provides a hierarchical, size-specific memory pooling system
 * that significantly reduces fragmentation and improves allocation speed
 * for neural network operations. It builds on the base memory pool system
 * but adds specialized pools for different allocation patterns.
 */

#ifndef TINYAI_ADVANCED_MEMORY_POOL_H
#define TINYAI_ADVANCED_MEMORY_POOL_H

#include "memory_pool.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Pool usage pattern types
 */
typedef enum {
    TINYAI_POOL_USAGE_WEIGHTS,     /**< For model weights (mostly read-only) */
    TINYAI_POOL_USAGE_ACTIVATIONS, /**< For activations (read-write, temporary) */
    TINYAI_POOL_USAGE_GENERAL,     /**< For general allocations */
    TINYAI_POOL_USAGE_COUNT        /**< Number of usage patterns */
} TinyAIPoolUsagePattern;

/**
 * @brief Size class for memory pools
 */
typedef enum {
    TINYAI_POOL_SIZE_TINY,   /**< For very small allocations (< 64 bytes) */
    TINYAI_POOL_SIZE_SMALL,  /**< For small allocations (64-256 bytes) */
    TINYAI_POOL_SIZE_MEDIUM, /**< For medium allocations (256-1024 bytes) */
    TINYAI_POOL_SIZE_LARGE,  /**< For large allocations (1-4 KB) */
    TINYAI_POOL_SIZE_XLARGE, /**< For very large allocations (4-64 KB) */
    TINYAI_POOL_SIZE_HUGE,   /**< For huge allocations (64 KB+) */
    TINYAI_POOL_SIZE_COUNT   /**< Number of size classes */
} TinyAIPoolSizeClass;

/**
 * @brief Advanced memory pool configuration
 */
typedef struct {
    /* Base configuration for all pools */
    TinyAIMemoryPoolConfig baseConfig;

    /* Pool size limits */
    size_t sizeClassLimits[TINYAI_POOL_SIZE_COUNT];

    /* Initial capacity per pool type */
    size_t initialCapacity[TINYAI_POOL_USAGE_COUNT][TINYAI_POOL_SIZE_COUNT];

    /* Maximum capacity per pool type */
    size_t maxCapacity[TINYAI_POOL_USAGE_COUNT][TINYAI_POOL_SIZE_COUNT];

    /* Thread safety */
    bool threadSafe;

    /* Enable tensor operation optimization */
    bool optimizeForTensorOps;

    /* Enable automatic pool resizing based on usage patterns */
    bool enableAutoResize;

    /* Aggressive defragmentation */
    bool aggressiveDefrag;
} TinyAIAdvancedPoolConfig;

/**
 * @brief Advanced memory pool statistics
 */
typedef struct {
    /* Stats per pool type */
    TinyAIMemoryPoolStats poolStats[TINYAI_POOL_USAGE_COUNT][TINYAI_POOL_SIZE_COUNT];

    /* Summary stats */
    size_t totalAllocated; /**< Total bytes allocated across all pools */
    size_t totalUsed;      /**< Total bytes used across all pools */
    size_t totalWasted;    /**< Total wasted bytes across all pools */

    /* Cache performance metrics */
    size_t cacheHits;    /**< Number of allocations served from cache */
    size_t cacheMisses;  /**< Number of allocations that missed cache */
    float  cacheHitRate; /**< Cache hit rate (0.0-1.0) */

    /* Pool performance */
    size_t poolSwitches;      /**< Number of times allocations moved between pools */
    double avgAllocationTime; /**< Average allocation time in microseconds */
    double avgFreeTime;       /**< Average free time in microseconds */

    /* Memory pressure indicators */
    uint8_t pressureScore;            /**< Memory pressure score (0-100) */
    bool    outOfMemoryEventOccurred; /**< Whether an out-of-memory event occurred */
} TinyAIAdvancedPoolStats;

/**
 * @brief Advanced memory pool handle
 */
typedef struct TinyAIAdvancedMemoryPool TinyAIAdvancedMemoryPool;

/**
 * @brief Get default advanced memory pool configuration
 *
 * @param config Pointer to configuration struct to fill
 */
void tinyaiAdvancedPoolGetDefaultConfig(TinyAIAdvancedPoolConfig *config);

/**
 * @brief Create a new advanced memory pool system
 *
 * @param config Pool configuration
 * @return Pointer to new advanced memory pool or NULL on failure
 */
TinyAIAdvancedMemoryPool *tinyaiAdvancedPoolCreate(const TinyAIAdvancedPoolConfig *config);

/**
 * @brief Destroy an advanced memory pool and free all its resources
 *
 * @param pool Pool to destroy
 */
void tinyaiAdvancedPoolDestroy(TinyAIAdvancedMemoryPool *pool);

/**
 * @brief Allocate memory from the appropriate pool based on size and usage pattern
 *
 * @param pool Advanced pool to allocate from
 * @param size Number of bytes to allocate
 * @param alignment Memory alignment requirement (must be power of 2)
 * @param usage Usage pattern for the allocation
 * @return Pointer to allocated memory or NULL on failure
 */
void *tinyaiAdvancedPoolAlloc(TinyAIAdvancedMemoryPool *pool, size_t size, size_t alignment,
                              TinyAIPoolUsagePattern usage);

/**
 * @brief Free memory allocated from advanced pool
 *
 * @param pool Advanced pool that the memory was allocated from
 * @param ptr Pointer to allocated memory
 */
void tinyaiAdvancedPoolFree(TinyAIAdvancedMemoryPool *pool, void *ptr);

/**
 * @brief Reallocate memory from advanced pool
 *
 * @param pool Advanced pool that the original memory was allocated from
 * @param ptr Pointer to previously allocated memory (or NULL for new allocation)
 * @param size New size in bytes
 * @param alignment Memory alignment requirement (must be power of 2)
 * @param usage Usage pattern for the allocation
 * @return Pointer to reallocated memory or NULL on failure
 */
void *tinyaiAdvancedPoolRealloc(TinyAIAdvancedMemoryPool *pool, void *ptr, size_t size,
                                size_t alignment, TinyAIPoolUsagePattern usage);

/**
 * @brief Get statistics for the advanced memory pool
 *
 * @param pool Advanced pool to query
 * @param stats Pointer to stats struct to fill
 */
void tinyaiAdvancedPoolGetStats(TinyAIAdvancedMemoryPool *pool, TinyAIAdvancedPoolStats *stats);

/**
 * @brief Reset all pools in the advanced memory pool system
 *
 * @param pool Advanced pool to reset
 */
void tinyaiAdvancedPoolReset(TinyAIAdvancedMemoryPool *pool);

/**
 * @brief Optimize the memory pool distribution based on usage patterns
 *
 * Analyzes allocation patterns and redistributes memory capacity
 * between pools to optimize for the current workload.
 *
 * @param pool Advanced pool to optimize
 * @return true if optimization was successful
 */
bool tinyaiAdvancedPoolOptimize(TinyAIAdvancedMemoryPool *pool);

/**
 * @brief Register a tensor operation with the memory pool
 *
 * Allows the pool to optimize memory layout for specific tensor operations.
 *
 * @param pool Advanced pool
 * @param opType Operation type identifier
 * @param inputSizes Array of input tensor sizes
 * @param numInputs Number of input tensors
 * @param outputSizes Array of output tensor sizes
 * @param numOutputs Number of output tensors
 * @return true if the operation was successfully registered
 */
bool tinyaiAdvancedPoolRegisterTensorOp(TinyAIAdvancedMemoryPool *pool, int opType,
                                        const size_t *inputSizes, int numInputs,
                                        const size_t *outputSizes, int numOutputs);

/**
 * @brief Allocate memory optimized for a specific tensor operation
 *
 * @param pool Advanced pool
 * @param opType Operation type identifier
 * @param isInput Whether this is an input or output tensor
 * @param tensorIndex Index of the tensor in the operation
 * @param size Size in bytes
 * @return Pointer to allocated memory or NULL on failure
 */
void *tinyaiAdvancedPoolAllocForTensorOp(TinyAIAdvancedMemoryPool *pool, int opType, bool isInput,
                                         int tensorIndex, size_t size);

/**
 * @brief Enable or disable thread safety for the pool
 *
 * @param pool Advanced pool
 * @param enable Whether to enable thread safety
 */
void tinyaiAdvancedPoolSetThreadSafety(TinyAIAdvancedMemoryPool *pool, bool enable);

/**
 * @brief Set memory pressure callback function
 *
 * This function will be called when memory pressure reaches a critical level.
 *
 * @param pool Advanced pool
 * @param callback Function to call on high memory pressure
 * @param userData User data to pass to the callback
 */
void tinyaiAdvancedPoolSetPressureCallback(TinyAIAdvancedMemoryPool *pool,
                                           void (*callback)(void *userData, uint8_t pressureLevel),
                                           void *userData);

/**
 * @brief Dump advanced memory pool information for debugging
 *
 * @param pool Advanced pool to dump
 * @param dumpAllocations Whether to dump individual allocations
 */
void tinyaiAdvancedPoolDump(TinyAIAdvancedMemoryPool *pool, bool dumpAllocations);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TINYAI_ADVANCED_MEMORY_POOL_H */