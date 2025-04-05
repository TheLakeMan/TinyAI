/**
 * TinyAI Memory Management Header
 * 
 * This header defines memory management functions for TinyAI, including
 * memory tracking and optimized allocation for small embedded systems.
 */

#ifndef TINYAI_MEMORY_H
#define TINYAI_MEMORY_H

#include <stddef.h>

/* ----------------- Memory Allocation ----------------- */

/**
 * Allocate memory
 * 
 * @param size Size in bytes to allocate
 * @return Pointer to allocated memory or NULL on failure
 */
void* tinyaiAlloc(size_t size);

/**
 * Reallocate memory
 * 
 * @param ptr Pointer to memory to reallocate
 * @param size New size in bytes
 * @return Pointer to reallocated memory or NULL on failure
 */
void* tinyaiRealloc(void *ptr, size_t size);

/**
 * Free memory
 * 
 * @param ptr Pointer to memory to free
 */
void tinyaiFree(void *ptr);

/**
 * Allocate zero-initialized memory
 * 
 * @param size Size in bytes to allocate
 * @return Pointer to allocated memory or NULL on failure
 */
void* tinyaiCalloc(size_t count, size_t size);

/* ----------------- Memory Pool ----------------- */

/**
 * Initialize memory pool
 * 
 * @param size Size of memory pool in bytes
 * @return 0 on success, non-zero on error
 */
int tinyaiMemPoolInit(size_t size);

/**
 * Clean up memory pool
 */
void tinyaiMemPoolCleanup();

/**
 * Allocate memory from pool
 * 
 * @param size Size in bytes to allocate
 * @return Pointer to allocated memory or NULL on failure
 */
void* tinyaiMemPoolAlloc(size_t size);

/**
 * Free memory from pool
 * 
 * @param ptr Pointer to memory to free
 */
void tinyaiMemPoolFree(void *ptr);

/**
 * Reset memory pool (free all allocations)
 */
void tinyaiMemPoolReset();

/**
 * Get memory pool statistics
 * 
 * @param totalSize Pointer to store total pool size
 * @param usedSize Pointer to store used pool size
 * @param peakSize Pointer to store peak used pool size
 * @param allocCount Pointer to store allocation count
 */
void tinyaiMemPoolStats(size_t *totalSize, size_t *usedSize, 
                       size_t *peakSize, size_t *allocCount);

/* ----------------- Memory Tracking ----------------- */

/**
 * Initialize memory tracking
 * 
 * @return 0 on success, non-zero on error
 */
int tinyaiMemTrackInit();

/**
 * Clean up memory tracking
 */
void tinyaiMemTrackCleanup();

/**
 * Track memory allocation
 * 
 * @param ptr Pointer to allocated memory
 * @param size Size of allocation
 * @param file Source file name
 * @param line Source line number
 */
void tinyaiMemTrackAlloc(void *ptr, size_t size, const char *file, int line);

/**
 * Track memory free
 * 
 * @param ptr Pointer to freed memory
 */
void tinyaiMemTrackFree(void *ptr);

/**
 * Dump memory leaks
 * 
 * @return Number of leaks found
 */
int tinyaiMemTrackDumpLeaks();

/**
 * Get memory tracking statistics
 * 
 * @param allocCount Pointer to store allocation count
 * @param allocSize Pointer to store total allocation size
 * @param freeCount Pointer to store free count
 * @param freeSize Pointer to store total free size
 */
void tinyaiMemTrackStats(size_t *allocCount, size_t *allocSize, 
                        size_t *freeCount, size_t *freeSize);

/* ----------------- Macros ----------------- */

/* Memory tracking macros */
#ifdef TINYAI_MEMORY_TRACKING
#define TINYAI_MALLOC(size) \
    (tinyaiMemTrackAlloc(tinyaiAlloc(size), (size), __FILE__, __LINE__), \
     tinyaiAlloc(size))
#define TINYAI_REALLOC(ptr, size) \
    (tinyaiMemTrackFree(ptr), \
     tinyaiMemTrackAlloc(tinyaiRealloc(ptr, size), (size), __FILE__, __LINE__), \
     tinyaiRealloc(ptr, size))
#define TINYAI_FREE(ptr) \
    (tinyaiMemTrackFree(ptr), tinyaiFree(ptr))
#define TINYAI_CALLOC(count, size) \
    (tinyaiMemTrackAlloc(tinyaiCalloc(count, size), (count) * (size), __FILE__, __LINE__), \
     tinyaiCalloc(count, size))
#else
#define TINYAI_MALLOC(size) tinyaiAlloc(size)
#define TINYAI_REALLOC(ptr, size) tinyaiRealloc(ptr, size)
#define TINYAI_FREE(ptr) tinyaiFree(ptr)
#define TINYAI_CALLOC(count, size) tinyaiCalloc(count, size)
#endif

#endif /* TINYAI_MEMORY_H */
