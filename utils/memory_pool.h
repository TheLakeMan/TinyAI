/**
 * @file memory_pool.h
 * @brief Enhanced memory pool system for large models
 *
 * This header provides optimized memory allocation for large 4-bit quantized models.
 * It implements a specialized pool allocator that reduces fragmentation and
 * improves memory locality for model weights and activations.
 */

#ifndef TINYAI_MEMORY_POOL_H
#define TINYAI_MEMORY_POOL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Memory pool configuration
 */
typedef struct {
    size_t initialCapacity;  /**< Initial pool size in bytes */
    size_t maxCapacity;      /**< Maximum pool size in bytes (0 for unlimited) */
    size_t blockSize;        /**< Minimum allocation block size */
    bool   allowGrowth;      /**< Whether the pool can grow beyond initial capacity */
    bool   trackAllocations; /**< Whether to track individual allocations (for debugging) */
} TinyAIMemoryPoolConfig;

/**
 * @brief Memory pool handle
 */
typedef struct TinyAIMemoryPool TinyAIMemoryPool;

/**
 * @brief Memory usage statistics
 */
typedef struct {
    size_t totalAllocated;     /**< Total bytes allocated */
    size_t totalUsed;          /**< Total bytes actually used */
    size_t totalWasted;        /**< Wasted bytes due to alignment and fragmentation */
    size_t largestBlock;       /**< Size of largest free block */
    size_t totalBlocks;        /**< Total number of allocated blocks */
    size_t freeBlocks;         /**< Number of free blocks */
    size_t fragmentationScore; /**< Fragmentation score (0-100, lower is better) */
} TinyAIMemoryPoolStats;

/**
 * @brief Get default memory pool configuration
 *
 * @param config Pointer to configuration struct to fill
 */
void tinyaiMemoryPoolGetDefaultConfig(TinyAIMemoryPoolConfig *config);

/**
 * @brief Create a new memory pool
 *
 * @param config Pool configuration
 * @return Pointer to new memory pool or NULL on failure
 */
TinyAIMemoryPool *tinyaiMemoryPoolCreate(const TinyAIMemoryPoolConfig *config);

/**
 * @brief Destroy a memory pool and free all its memory
 *
 * @param pool Pool to destroy
 */
void tinyaiMemoryPoolDestroy(TinyAIMemoryPool *pool);

/**
 * @brief Allocate memory from pool
 *
 * Allocates aligned memory from the pool. If the pool is full and growth
 * is allowed, it will grow to accommodate the request.
 *
 * @param pool Pool to allocate from
 * @param size Number of bytes to allocate
 * @param alignment Memory alignment requirement (must be power of 2)
 * @return Pointer to allocated memory or NULL on failure
 */
void *tinyaiMemoryPoolAlloc(TinyAIMemoryPool *pool, size_t size, size_t alignment);

/**
 * @brief Reallocate memory block
 *
 * @param pool Pool that the original block was allocated from
 * @param ptr Pointer to previously allocated block (or NULL for new allocation)
 * @param size New size in bytes
 * @param alignment Memory alignment requirement (must be power of 2)
 * @return Pointer to reallocated memory or NULL on failure
 */
void *tinyaiMemoryPoolRealloc(TinyAIMemoryPool *pool, void *ptr, size_t size, size_t alignment);

/**
 * @brief Free memory allocated from pool
 *
 * @param pool Pool that the block was allocated from
 * @param ptr Pointer to allocated memory
 */
void tinyaiMemoryPoolFree(TinyAIMemoryPool *pool, void *ptr);

/**
 * @brief Reset a memory pool, freeing all allocations
 *
 * @param pool Pool to reset
 */
void tinyaiMemoryPoolReset(TinyAIMemoryPool *pool);

/**
 * @brief Get memory pool statistics
 *
 * @param pool Pool to query
 * @param stats Pointer to stats struct to fill
 */
void tinyaiMemoryPoolGetStats(TinyAIMemoryPool *pool, TinyAIMemoryPoolStats *stats);

/**
 * @brief Check if a pointer was allocated from the given pool
 *
 * @param pool Pool to check
 * @param ptr Pointer to check
 * @return true if the pointer was allocated from the pool
 */
bool tinyaiMemoryPoolContains(TinyAIMemoryPool *pool, const void *ptr);

/**
 * @brief Specialized 4-bit weight allocation for models
 *
 * Allocates memory optimized for 4-bit quantized weights.
 * This function handles packing, alignment, and potential SIMD optimizations.
 *
 * @param pool Pool to allocate from
 * @param rows Number of rows in the weight matrix
 * @param cols Number of columns in the weight matrix
 * @param requiresSIMD Whether the allocation will be used with SIMD operations
 * @return Pointer to allocated memory or NULL on failure
 */
uint8_t *tinyaiMemoryPoolAllocWeights4Bit(TinyAIMemoryPool *pool, size_t rows, size_t cols,
                                          bool requiresSIMD);

/**
 * @brief Specialized activation allocation for models
 *
 * Allocates memory optimized for model activations.
 * This function handles alignment for efficient computation.
 *
 * @param pool Pool to allocate from
 * @param size Number of float values to allocate
 * @param requiresSIMD Whether the allocation will be used with SIMD operations
 * @return Pointer to allocated memory or NULL on failure
 */
float *tinyaiMemoryPoolAllocActivations(TinyAIMemoryPool *pool, size_t size, bool requiresSIMD);

/**
 * @brief Compact pool memory to reduce fragmentation
 *
 * @param pool Pool to compact
 * @return true if compaction was successful
 */
bool tinyaiMemoryPoolCompact(TinyAIMemoryPool *pool);

/**
 * @brief Dump memory pool information for debugging
 *
 * @param pool Pool to dump
 * @param dumpAllocations Whether to dump individual allocations
 */
void tinyaiMemoryPoolDump(TinyAIMemoryPool *pool, bool dumpAllocations);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TINYAI_MEMORY_POOL_H */
