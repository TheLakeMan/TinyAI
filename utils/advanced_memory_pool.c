/**
 * @file advanced_memory_pool.c
 * @brief Implementation of advanced memory pooling system for TinyAI
 */

#include "advanced_memory_pool.h"
#include "../core/logging.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

/* Default configuration values */
#define DEFAULT_THREAD_SAFE true
#define DEFAULT_OPTIMIZE_FOR_TENSOR_OPS true
#define DEFAULT_ENABLE_AUTO_RESIZE true
#define DEFAULT_AGGRESSIVE_DEFRAG false

/* Size class limits in bytes */
#define SIZE_TINY_LIMIT 64
#define SIZE_SMALL_LIMIT 256
#define SIZE_MEDIUM_LIMIT 1024
#define SIZE_LARGE_LIMIT 4096
#define SIZE_XLARGE_LIMIT 65536

/* Default capacity per pool type (in bytes) */
/* Weights pools - optimized for read-only access */
#define WEIGHTS_TINY_CAPACITY (64 * 1024)          /* 64 KB */
#define WEIGHTS_SMALL_CAPACITY (256 * 1024)        /* 256 KB */
#define WEIGHTS_MEDIUM_CAPACITY (1024 * 1024)      /* 1 MB */
#define WEIGHTS_LARGE_CAPACITY (4 * 1024 * 1024)   /* 4 MB */
#define WEIGHTS_XLARGE_CAPACITY (16 * 1024 * 1024) /* 16 MB */
#define WEIGHTS_HUGE_CAPACITY (64 * 1024 * 1024)   /* 64 MB */

/* Activations pools - optimized for frequent reuse */
#define ACTIVATIONS_TINY_CAPACITY (128 * 1024)         /* 128 KB */
#define ACTIVATIONS_SMALL_CAPACITY (512 * 1024)        /* 512 KB */
#define ACTIVATIONS_MEDIUM_CAPACITY (2 * 1024 * 1024)  /* 2 MB */
#define ACTIVATIONS_LARGE_CAPACITY (8 * 1024 * 1024)   /* 8 MB */
#define ACTIVATIONS_XLARGE_CAPACITY (32 * 1024 * 1024) /* 32 MB */
#define ACTIVATIONS_HUGE_CAPACITY (128 * 1024 * 1024)  /* 128 MB */

/* General pools - balanced allocation */
#define GENERAL_TINY_CAPACITY (32 * 1024)         /* 32 KB */
#define GENERAL_SMALL_CAPACITY (128 * 1024)       /* 128 KB */
#define GENERAL_MEDIUM_CAPACITY (512 * 1024)      /* 512 KB */
#define GENERAL_LARGE_CAPACITY (2 * 1024 * 1024)  /* 2 MB */
#define GENERAL_XLARGE_CAPACITY (8 * 1024 * 1024) /* 8 MB */
#define GENERAL_HUGE_CAPACITY (32 * 1024 * 1024)  /* 32 MB */

/* Maximum capacity multipliers */
#define MAX_CAPACITY_MULTIPLIER 4

/* Allocation cache */
#define ALLOCATION_CACHE_SIZE 64

/* Memory pressure thresholds */
#define MEMORY_PRESSURE_LOW 30
#define MEMORY_PRESSURE_MEDIUM 60
#define MEMORY_PRESSURE_HIGH 80
#define MEMORY_PRESSURE_CRITICAL 90

/* Operation tracking */
#define MAX_TENSOR_OPS 32
#define MAX_TENSORS_PER_OP 16

/* Cache entry for allocation tracking */
typedef struct {
    void                  *ptr;
    size_t                 size;
    TinyAIMemoryPool      *sourcePool;
    TinyAIPoolUsagePattern usage;
    TinyAIPoolSizeClass    sizeClass;
} AllocationCacheEntry;

/* Tensor operation descriptor */
typedef struct {
    int    opType;
    size_t inputSizes[MAX_TENSORS_PER_OP];
    int    numInputs;
    size_t outputSizes[MAX_TENSORS_PER_OP];
    int    numOutputs;
    void  *optimizedLayouts[MAX_TENSORS_PER_OP * 2]; /* For both inputs and outputs */
} TensorOpDescriptor;

/* Advanced memory pool structure */
struct TinyAIAdvancedMemoryPool {
    /* The actual memory pools organized by usage and size */
    TinyAIMemoryPool *pools[TINYAI_POOL_USAGE_COUNT][TINYAI_POOL_SIZE_COUNT];

    /* Configuration */
    TinyAIAdvancedPoolConfig config;

    /* Allocation cache for fast lookup */
    AllocationCacheEntry allocationCache[ALLOCATION_CACHE_SIZE];
    int                  cacheSize;

    /* Statistics */
    size_t allocCount;
    size_t freeCount;
    size_t cacheHits;
    size_t cacheMisses;
    size_t poolSwitches;

    /* Performance tracking */
    double totalAllocationTime;
    double totalFreeTime;

    /* Memory pressure tracking */
    uint8_t currentPressure;
    bool    outOfMemoryEventOccurred;
    void (*pressureCallback)(void *userData, uint8_t pressureLevel);
    void *pressureCallbackUserData;

    /* Thread safety control */
    bool threadSafetyEnabled;

    /* Tensor operation optimization */
    TensorOpDescriptor tensorOps[MAX_TENSOR_OPS];
    int                numTensorOps;
};

/* Helper functions prototypes */
static TinyAIPoolSizeClass   getSizeClass(size_t size, const TinyAIAdvancedPoolConfig *config);
static TinyAIMemoryPool     *getAppropriatePool(TinyAIAdvancedMemoryPool *pool, size_t size,
                                                TinyAIPoolUsagePattern usage);
static void                  updateMemoryPressure(TinyAIAdvancedMemoryPool *pool);
static void                  addToCache(TinyAIAdvancedMemoryPool *pool, void *ptr, size_t size,
                                        TinyAIMemoryPool *sourcePool, TinyAIPoolUsagePattern usage,
                                        TinyAIPoolSizeClass sizeClass);
static AllocationCacheEntry *findInCache(TinyAIAdvancedMemoryPool *pool, void *ptr);
static void                  removeFromCache(TinyAIAdvancedMemoryPool *pool, void *ptr);
static double                getCurrentTimeMs();

/**
 * Get default advanced memory pool configuration
 */
void tinyaiAdvancedPoolGetDefaultConfig(TinyAIAdvancedPoolConfig *config)
{
    if (!config) {
        return;
    }

    /* Initialize with zeros */
    memset(config, 0, sizeof(TinyAIAdvancedPoolConfig));

    /* Get default base config */
    tinyaiMemoryPoolGetDefaultConfig(&config->baseConfig);

    /* Set size class limits */
    config->sizeClassLimits[TINYAI_POOL_SIZE_TINY]   = SIZE_TINY_LIMIT;
    config->sizeClassLimits[TINYAI_POOL_SIZE_SMALL]  = SIZE_SMALL_LIMIT;
    config->sizeClassLimits[TINYAI_POOL_SIZE_MEDIUM] = SIZE_MEDIUM_LIMIT;
    config->sizeClassLimits[TINYAI_POOL_SIZE_LARGE]  = SIZE_LARGE_LIMIT;
    config->sizeClassLimits[TINYAI_POOL_SIZE_XLARGE] = SIZE_XLARGE_LIMIT;
    config->sizeClassLimits[TINYAI_POOL_SIZE_HUGE]   = (size_t)-1; /* No upper limit */

    /* Set initial capacities */
    /* Weights pools */
    config->initialCapacity[TINYAI_POOL_USAGE_WEIGHTS][TINYAI_POOL_SIZE_TINY] =
        WEIGHTS_TINY_CAPACITY;
    config->initialCapacity[TINYAI_POOL_USAGE_WEIGHTS][TINYAI_POOL_SIZE_SMALL] =
        WEIGHTS_SMALL_CAPACITY;
    config->initialCapacity[TINYAI_POOL_USAGE_WEIGHTS][TINYAI_POOL_SIZE_MEDIUM] =
        WEIGHTS_MEDIUM_CAPACITY;
    config->initialCapacity[TINYAI_POOL_USAGE_WEIGHTS][TINYAI_POOL_SIZE_LARGE] =
        WEIGHTS_LARGE_CAPACITY;
    config->initialCapacity[TINYAI_POOL_USAGE_WEIGHTS][TINYAI_POOL_SIZE_XLARGE] =
        WEIGHTS_XLARGE_CAPACITY;
    config->initialCapacity[TINYAI_POOL_USAGE_WEIGHTS][TINYAI_POOL_SIZE_HUGE] =
        WEIGHTS_HUGE_CAPACITY;

    /* Activations pools */
    config->initialCapacity[TINYAI_POOL_USAGE_ACTIVATIONS][TINYAI_POOL_SIZE_TINY] =
        ACTIVATIONS_TINY_CAPACITY;
    config->initialCapacity[TINYAI_POOL_USAGE_ACTIVATIONS][TINYAI_POOL_SIZE_SMALL] =
        ACTIVATIONS_SMALL_CAPACITY;
    config->initialCapacity[TINYAI_POOL_USAGE_ACTIVATIONS][TINYAI_POOL_SIZE_MEDIUM] =
        ACTIVATIONS_MEDIUM_CAPACITY;
    config->initialCapacity[TINYAI_POOL_USAGE_ACTIVATIONS][TINYAI_POOL_SIZE_LARGE] =
        ACTIVATIONS_LARGE_CAPACITY;
    config->initialCapacity[TINYAI_POOL_USAGE_ACTIVATIONS][TINYAI_POOL_SIZE_XLARGE] =
        ACTIVATIONS_XLARGE_CAPACITY;
    config->initialCapacity[TINYAI_POOL_USAGE_ACTIVATIONS][TINYAI_POOL_SIZE_HUGE] =
        ACTIVATIONS_HUGE_CAPACITY;

    /* General pools */
    config->initialCapacity[TINYAI_POOL_USAGE_GENERAL][TINYAI_POOL_SIZE_TINY] =
        GENERAL_TINY_CAPACITY;
    config->initialCapacity[TINYAI_POOL_USAGE_GENERAL][TINYAI_POOL_SIZE_SMALL] =
        GENERAL_SMALL_CAPACITY;
    config->initialCapacity[TINYAI_POOL_USAGE_GENERAL][TINYAI_POOL_SIZE_MEDIUM] =
        GENERAL_MEDIUM_CAPACITY;
    config->initialCapacity[TINYAI_POOL_USAGE_GENERAL][TINYAI_POOL_SIZE_LARGE] =
        GENERAL_LARGE_CAPACITY;
    config->initialCapacity[TINYAI_POOL_USAGE_GENERAL][TINYAI_POOL_SIZE_XLARGE] =
        GENERAL_XLARGE_CAPACITY;
    config->initialCapacity[TINYAI_POOL_USAGE_GENERAL][TINYAI_POOL_SIZE_HUGE] =
        GENERAL_HUGE_CAPACITY;

    /* Set max capacities (for now, just multiply initial by MAX_CAPACITY_MULTIPLIER) */
    for (int usage = 0; usage < TINYAI_POOL_USAGE_COUNT; usage++) {
        for (int size = 0; size < TINYAI_POOL_SIZE_COUNT; size++) {
            config->maxCapacity[usage][size] =
                config->initialCapacity[usage][size] * MAX_CAPACITY_MULTIPLIER;
        }
    }

    /* Set feature flags */
    config->threadSafe           = DEFAULT_THREAD_SAFE;
    config->optimizeForTensorOps = DEFAULT_OPTIMIZE_FOR_TENSOR_OPS;
    config->enableAutoResize     = DEFAULT_ENABLE_AUTO_RESIZE;
    config->aggressiveDefrag     = DEFAULT_AGGRESSIVE_DEFRAG;
}

/**
 * Create a new advanced memory pool system
 */
TinyAIAdvancedMemoryPool *tinyaiAdvancedPoolCreate(const TinyAIAdvancedPoolConfig *config)
{
    if (!config) {
        return NULL;
    }

    /* Allocate the advanced pool structure */
    TinyAIAdvancedMemoryPool *advPool =
        (TinyAIAdvancedMemoryPool *)malloc(sizeof(TinyAIAdvancedMemoryPool));
    if (!advPool) {
        return NULL;
    }

    /* Initialize the structure */
    memset(advPool, 0, sizeof(TinyAIAdvancedMemoryPool));

    /* Copy configuration */
    memcpy(&advPool->config, config, sizeof(TinyAIAdvancedPoolConfig));

    /* Create individual pools for each usage/size combination */
    for (int usage = 0; usage < TINYAI_POOL_USAGE_COUNT; usage++) {
        for (int size = 0; size < TINYAI_POOL_SIZE_COUNT; size++) {
            /* Skip creation of pools with zero capacity */
            if (config->initialCapacity[usage][size] == 0) {
                advPool->pools[usage][size] = NULL;
                continue;
            }

            /* Configure this pool */
            TinyAIMemoryPoolConfig poolConfig = config->baseConfig;
            poolConfig.initialCapacity        = config->initialCapacity[usage][size];
            poolConfig.maxCapacity            = config->maxCapacity[usage][size];

            /* Adjust block size based on size class for efficiency */
            switch (size) {
            case TINYAI_POOL_SIZE_TINY:
                poolConfig.blockSize = 16; /* Small blocks */
                break;
            case TINYAI_POOL_SIZE_SMALL:
                poolConfig.blockSize = 32;
                break;
            case TINYAI_POOL_SIZE_MEDIUM:
                poolConfig.blockSize = 64;
                break;
            case TINYAI_POOL_SIZE_LARGE:
                poolConfig.blockSize = 128;
                break;
            case TINYAI_POOL_SIZE_XLARGE:
                poolConfig.blockSize = 256;
                break;
            case TINYAI_POOL_SIZE_HUGE:
                poolConfig.blockSize = 512; /* Larger blocks */
                break;
            }

            /* Create the pool */
            advPool->pools[usage][size] = tinyaiMemoryPoolCreate(&poolConfig);
            if (!advPool->pools[usage][size]) {
                /* Failed to create pool, clean up and return NULL */
                tinyaiAdvancedPoolDestroy(advPool);
                return NULL;
            }
        }
    }

    /* Initialize other properties */
    advPool->threadSafetyEnabled      = config->threadSafe;
    advPool->cacheSize                = 0;
    advPool->currentPressure          = 0;
    advPool->outOfMemoryEventOccurred = false;
    advPool->pressureCallback         = NULL;
    advPool->pressureCallbackUserData = NULL;

    return advPool;
}

/**
 * Destroy an advanced memory pool and free all its resources
 */
void tinyaiAdvancedPoolDestroy(TinyAIAdvancedMemoryPool *pool)
{
    if (!pool) {
        return;
    }

    /* Destroy all individual pools */
    for (int usage = 0; usage < TINYAI_POOL_USAGE_COUNT; usage++) {
        for (int size = 0; size < TINYAI_POOL_SIZE_COUNT; size++) {
            if (pool->pools[usage][size]) {
                tinyaiMemoryPoolDestroy(pool->pools[usage][size]);
                pool->pools[usage][size] = NULL;
            }
        }
    }

    /* Free the advanced pool structure */
    free(pool);
}

/**
 * Get current time in milliseconds
 */
static double getCurrentTimeMs()
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

    /* Convert to milliseconds */
    return (double)count.QuadPart * 1000.0 / (double)frequency.QuadPart;
#else
    /* Linux/Unix implementation */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
#endif
}

/**
 * Determine the size class for a given allocation size
 */
static TinyAIPoolSizeClass getSizeClass(size_t size, const TinyAIAdvancedPoolConfig *config)
{
    for (int i = 0; i < TINYAI_POOL_SIZE_COUNT - 1; i++) {
        if (size <= config->sizeClassLimits[i]) {
            return (TinyAIPoolSizeClass)i;
        }
    }
    return TINYAI_POOL_SIZE_HUGE;
}

/**
 * Get the appropriate pool for a given size and usage
 */
static TinyAIMemoryPool *getAppropriatePool(TinyAIAdvancedMemoryPool *pool, size_t size,
                                            TinyAIPoolUsagePattern usage)
{
    TinyAIPoolSizeClass sizeClass = getSizeClass(size, &pool->config);
    return pool->pools[usage][sizeClass];
}

/**
 * Add an allocation to the cache
 */
static void addToCache(TinyAIAdvancedMemoryPool *pool, void *ptr, size_t size,
                       TinyAIMemoryPool *sourcePool, TinyAIPoolUsagePattern usage,
                       TinyAIPoolSizeClass sizeClass)
{
    if (pool->cacheSize >= ALLOCATION_CACHE_SIZE) {
        /* Cache is full, remove oldest entry (simple FIFO for now) */
        for (int i = 0; i < ALLOCATION_CACHE_SIZE - 1; i++) {
            pool->allocationCache[i] = pool->allocationCache[i + 1];
        }
        pool->cacheSize--;
    }

    /* Add new entry at the end */
    int index                               = pool->cacheSize;
    pool->allocationCache[index].ptr        = ptr;
    pool->allocationCache[index].size       = size;
    pool->allocationCache[index].sourcePool = sourcePool;
    pool->allocationCache[index].usage      = usage;
    pool->allocationCache[index].sizeClass  = sizeClass;

    pool->cacheSize++;
}

/**
 * Find an allocation in the cache
 */
static AllocationCacheEntry *findInCache(TinyAIAdvancedMemoryPool *pool, void *ptr)
{
    for (int i = 0; i < pool->cacheSize; i++) {
        if (pool->allocationCache[i].ptr == ptr) {
            return &pool->allocationCache[i];
        }
    }
    return NULL;
}

/**
 * Remove an allocation from the cache
 */
static void removeFromCache(TinyAIAdvancedMemoryPool *pool, void *ptr)
{
    int found = -1;

    /* Find the entry */
    for (int i = 0; i < pool->cacheSize; i++) {
        if (pool->allocationCache[i].ptr == ptr) {
            found = i;
            break;
        }
    }

    if (found >= 0) {
        /* Remove entry by shifting all entries after it */
        for (int i = found; i < pool->cacheSize - 1; i++) {
            pool->allocationCache[i] = pool->allocationCache[i + 1];
        }
        pool->cacheSize--;
    }
}

/**
 * Update memory pressure metric
 */
static void updateMemoryPressure(TinyAIAdvancedMemoryPool *pool)
{
    size_t totalUsed     = 0;
    size_t totalCapacity = 0;

    /* Calculate total usage across all pools */
    for (int usage = 0; usage < TINYAI_POOL_USAGE_COUNT; usage++) {
        for (int size = 0; size < TINYAI_POOL_SIZE_COUNT; size++) {
            if (pool->pools[usage][size]) {
                TinyAIMemoryPoolStats stats;
                tinyaiMemoryPoolGetStats(pool->pools[usage][size], &stats);

                totalUsed += stats.totalUsed;
                totalCapacity += stats.totalAllocated;
            }
        }
    }

    /* Calculate pressure as a percentage */
    if (totalCapacity > 0) {
        pool->currentPressure = (uint8_t)((totalUsed * 100) / totalCapacity);
    }
    else {
        pool->currentPressure = 0;
    }

    /* Trigger callback if pressure is high and callback is registered */
    if (pool->pressureCallback && pool->currentPressure >= MEMORY_PRESSURE_HIGH) {
        pool->pressureCallback(pool->pressureCallbackUserData, pool->currentPressure);
    }
}

/**
 * Allocate memory from the appropriate pool based on size and usage pattern
 */
void *tinyaiAdvancedPoolAlloc(TinyAIAdvancedMemoryPool *pool, size_t size, size_t alignment,
                              TinyAIPoolUsagePattern usage)
{
    if (!pool || size == 0 || usage >= TINYAI_POOL_USAGE_COUNT) {
        return NULL;
    }

    /* Track performance */
    double startTime = getCurrentTimeMs();

    /* Get the appropriate pool */
    TinyAIPoolSizeClass sizeClass = getSizeClass(size, &pool->config);
    TinyAIMemoryPool   *memPool   = pool->pools[usage][sizeClass];

    if (!memPool) {
        /* Try to use general pool as fallback */
        memPool = pool->pools[TINYAI_POOL_USAGE_GENERAL][sizeClass];
        if (!memPool) {
            /* No suitable pool found */
            return NULL;
        }

        /* Track the pool switch */
        pool->poolSwitches++;
    }

    /* Allocate memory */
    void *ptr = tinyaiMemoryPoolAlloc(memPool, size, alignment);

    if (ptr) {
        /* Add to cache for faster lookups */
        addToCache(pool, ptr, size, memPool, usage, sizeClass);

        /* Update statistics */
        pool->allocCount++;

        /* Calculate allocation duration */
        double duration = getCurrentTimeMs() - startTime;
        pool->totalAllocationTime += duration;

        /* Update memory pressure */
        updateMemoryPressure(pool);
    }
    else {
        /* Allocation failed, mark out of memory event */
        pool->outOfMemoryEventOccurred = true;

        /* Try with general pool as fallback if not already tried */
        if (usage != TINYAI_POOL_USAGE_GENERAL) {
            memPool = pool->pools[TINYAI_POOL_USAGE_GENERAL][sizeClass];
            if (memPool) {
                ptr = tinyaiMemoryPoolAlloc(memPool, size, alignment);
                if (ptr) {
                    /* Add to cache */
                    addToCache(pool, ptr, size, memPool, TINYAI_POOL_USAGE_GENERAL, sizeClass);

                    /* Update statistics */
                    pool->allocCount++;
                    pool->poolSwitches++;

                    /* Calculate allocation duration */
                    double duration = getCurrentTimeMs() - startTime;
                    pool->totalAllocationTime += duration;

                    /* Update memory pressure */
                    updateMemoryPressure(pool);
                }
            }
        }
    }

    return ptr;
}

/**
 * Free memory allocated from advanced pool
 */
void tinyaiAdvancedPoolFree(TinyAIAdvancedMemoryPool *pool, void *ptr)
{
    if (!pool || !ptr) {
        return;
    }

    /* Track performance */
    double startTime = getCurrentTimeMs();

    /* Look up allocation in cache */
    AllocationCacheEntry *entry = findInCache(pool, ptr);

    if (entry) {
        /* Cache hit - use the source pool information */
        tinyaiMemoryPoolFree(entry->sourcePool, ptr);

        /* Remove from cache */
        removeFromCache(pool, ptr);

        /* Update statistics */
        pool->freeCount++;
        pool->cacheHits++;
    }
    else {
        /* Cache miss - need to scan all pools */
        pool->cacheMisses++;

        bool found = false;

        /* Try each pool until we find the one that contains this pointer */
        for (int usage = 0; usage < TINYAI_POOL_USAGE_COUNT && !found; usage++) {
            for (int size = 0; size < TINYAI_POOL_SIZE_COUNT && !found; size++) {
                TinyAIMemoryPool *memPool = pool->pools[usage][size];
                if (memPool && tinyaiMemoryPoolContains(memPool, ptr)) {
                    /* Found the pool, free the memory */
                    tinyaiMemoryPoolFree(memPool, ptr);

                    /* Update statistics */
                    pool->freeCount++;
                    found = true;
                    break;
                }
            }
        }

        if (!found) {
            /* Pointer not found in any pool - this is an error */
            /* Log warning or error */
            return;
        }
    }

    /* Calculate free duration */
    double duration = getCurrentTimeMs() - startTime;
    pool->totalFreeTime += duration;

    /* Update memory pressure */
    updateMemoryPressure(pool);
}

/**
 * Reallocate memory from advanced pool
 */
void *tinyaiAdvancedPoolRealloc(TinyAIAdvancedMemoryPool *pool, void *ptr, size_t size,
                                size_t alignment, TinyAIPoolUsagePattern usage)
{
    if (!pool) {
        return NULL;
    }

    /* If ptr is NULL, this is equivalent to alloc */
    if (!ptr) {
        return tinyaiAdvancedPoolAlloc(pool, size, alignment, usage);
    }

    /* If size is 0, this is equivalent to free */
    if (size == 0) {
        tinyaiAdvancedPoolFree(pool, ptr);
        return NULL;
    }

    /* Look up allocation in cache */
    AllocationCacheEntry *entry = findInCache(pool, ptr);

    if (entry) {
        /* Cache hit - we know which pool it came from */
        TinyAIPoolSizeClass newSizeClass = getSizeClass(size, &pool->config);

        /* Check if we need to switch pools */
        if (newSizeClass != entry->sizeClass || usage != entry->usage) {
            /* Size or usage pattern changed, allocate from new pool and copy data */
            void *newPtr = tinyaiAdvancedPoolAlloc(pool, size, alignment, usage);
            if (!newPtr) {
                return NULL; /* Allocation failed */
            }

            /* Copy data from old location to new location */
            size_t copySize = (size < entry->size) ? size : entry->size;
            memcpy(newPtr, ptr, copySize);

            /* Free old allocation */
            tinyaiAdvancedPoolFree(pool, ptr);

            /* Track pool switch */
            pool->poolSwitches++;

            return newPtr;
        }
        else {
            /* Same pool, use its realloc directly */
            void *newPtr = tinyaiMemoryPoolRealloc(entry->sourcePool, ptr, size, alignment);
            if (!newPtr) {
                return NULL; /* Reallocation failed */
            }

            /* Update cache if pointer changed */
            if (newPtr != ptr) {
                removeFromCache(pool, ptr);
                addToCache(pool, newPtr, size, entry->sourcePool, entry->usage, entry->sizeClass);
            }
            else {
                /* Just update the size */
                entry->size = size;
            }

            return newPtr;
        }
    }
    else {
        /* Cache miss - need to scan all pools */
        pool->cacheMisses++;

        /* Find which pool this pointer belongs to */
        TinyAIMemoryPool      *sourcePool      = NULL;
        TinyAIPoolUsagePattern sourceUsage     = TINYAI_POOL_USAGE_GENERAL;
        TinyAIPoolSizeClass    sourceSizeClass = TINYAI_POOL_SIZE_TINY;

        for (int u = 0; u < TINYAI_POOL_USAGE_COUNT && !sourcePool; u++) {
            for (int s = 0; s < TINYAI_POOL_SIZE_COUNT && !sourcePool; s++) {
                TinyAIMemoryPool *memPool = pool->pools[u][s];
                if (memPool && tinyaiMemoryPoolContains(memPool, ptr)) {
                    sourcePool      = memPool;
                    sourceUsage     = (TinyAIPoolUsagePattern)u;
                    sourceSizeClass = (TinyAIPoolSizeClass)s;
                    break;
                }
            }
        }

        if (!sourcePool) {
            /* Pointer not found in any pool - this is an error */
            return NULL;
        }

        /* Determine new size class */
        TinyAIPoolSizeClass newSizeClass = getSizeClass(size, &pool->config);

        /* Check if we need to switch pools */
        if (newSizeClass != sourceSizeClass || usage != sourceUsage) {
            /* Size or usage pattern changed, allocate from new pool and copy data */
            void *newPtr = tinyaiAdvancedPoolAlloc(pool, size, alignment, usage);
            if (!newPtr) {
                return NULL; /* Allocation failed */
            }

            /* Get size of original allocation (approximate) */
            size_t origSize =
                (sourceSizeClass > 0) ? pool->config.sizeClassLimits[sourceSizeClass - 1] : 0;

            /* Copy data from old location to new location */
            size_t copySize = (size < origSize) ? size : origSize;
            memcpy(newPtr, ptr, copySize);

            /* Free old allocation */
            tinyaiMemoryPoolFree(sourcePool, ptr);

            /* Track pool switch */
            pool->poolSwitches++;

            return newPtr;
        }
        else {
            /* Same pool, use its realloc directly */
            void *newPtr = tinyaiMemoryPoolRealloc(sourcePool, ptr, size, alignment);
            if (!newPtr) {
                return NULL; /* Reallocation failed */
            }

            /* Add to cache */
            if (newPtr != ptr) {
                addToCache(pool, newPtr, size, sourcePool, sourceUsage, sourceSizeClass);
            }

            return newPtr;
        }
    }
}

/**
 * Get statistics for the advanced memory pool
 */
void tinyaiAdvancedPoolGetStats(TinyAIAdvancedMemoryPool *pool, TinyAIAdvancedPoolStats *stats)
{
    if (!pool || !stats) {
        return;
    }

    /* Clear stats structure */
    memset(stats, 0, sizeof(TinyAIAdvancedPoolStats));

    /* Gather statistics from all pools */
    for (int usage = 0; usage < TINYAI_POOL_USAGE_COUNT; usage++) {
        for (int size = 0; size < TINYAI_POOL_SIZE_COUNT; size++) {
            TinyAIMemoryPool *memPool = pool->pools[usage][size];
            if (memPool) {
                tinyaiMemoryPoolGetStats(memPool, &stats->poolStats[usage][size]);

                /* Accumulate summary stats */
                stats->totalAllocated += stats->poolStats[usage][size].totalAllocated;
                stats->totalUsed += stats->poolStats[usage][size].totalUsed;
                stats->totalWasted += stats->poolStats[usage][size].totalWasted;
            }
        }
    }

    /* Set cache performance metrics */
    stats->cacheHits    = pool->cacheHits;
    stats->cacheMisses  = pool->cacheMisses;
    stats->cacheHitRate = (pool->cacheHits + pool->cacheMisses > 0)
                              ? (float)pool->cacheHits / (pool->cacheHits + pool->cacheMisses)
                              : 0.0f;

    /* Set pool performance metrics */
    stats->poolSwitches = pool->poolSwitches;
    stats->avgAllocationTime =
        (pool->allocCount > 0) ? pool->totalAllocationTime / pool->allocCount : 0.0;
    stats->avgFreeTime = (pool->freeCount > 0) ? pool->totalFreeTime / pool->freeCount : 0.0;

    /* Set memory pressure indicators */
    stats->pressureScore            = pool->currentPressure;
    stats->outOfMemoryEventOccurred = pool->outOfMemoryEventOccurred;
}

/**
 * Reset all pools in the advanced memory pool system
 */
void tinyaiAdvancedPoolReset(TinyAIAdvancedMemoryPool *pool)
{
    if (!pool) {
        return;
    }

    /* Reset all individual pools */
    for (int usage = 0; usage < TINYAI_POOL_USAGE_COUNT; usage++) {
        for (int size = 0; size < TINYAI_POOL_SIZE_COUNT; size++) {
            if (pool->pools[usage][size]) {
                tinyaiMemoryPoolReset(pool->pools[usage][size]);
            }
        }
    }

    /* Clear cache */
    pool->cacheSize = 0;

    /* Reset statistics */
    pool->allocCount               = 0;
    pool->freeCount                = 0;
    pool->cacheHits                = 0;
    pool->cacheMisses              = 0;
    pool->poolSwitches             = 0;
    pool->totalAllocationTime      = 0.0;
    pool->totalFreeTime            = 0.0;
    pool->currentPressure          = 0;
    pool->outOfMemoryEventOccurred = false;
}

/**
 * Optimize the memory pool distribution based on usage patterns
 */
bool tinyaiAdvancedPoolOptimize(TinyAIAdvancedMemoryPool *pool)
{
    if (!pool || !pool->config.enableAutoResize) {
        return false;
    }

    /* Analyze usage patterns for each pool */
    for (int usage = 0; usage < TINYAI_POOL_USAGE_COUNT; usage++) {
        for (int size = 0; size < TINYAI_POOL_SIZE_COUNT; size++) {
            TinyAIMemoryPool *memPool = pool->pools[usage][size];
            if (!memPool) {
                continue;
            }

            /* Get current stats */
            TinyAIMemoryPoolStats stats;
            tinyaiMemoryPoolGetStats(memPool, &stats);

            /* Calculate usage ratio */
            float usageRatio = (float)stats.totalUsed / stats.totalAllocated;

            /* Decide if we need to resize this pool */
            if (usageRatio > 0.85f) {
                /* Pool is almost full, try to grow it */
                size_t growSize = stats.totalAllocated / 2; /* Grow by 50% */

                /* Create a new config with increased capacity */
                TinyAIMemoryPoolConfig newConfig = pool->config.baseConfig;
                newConfig.initialCapacity        = stats.totalAllocated + growSize;

                /* Check if growing would exceed max capacity */
                if (pool->config.maxCapacity[usage][size] > 0 &&
                    newConfig.initialCapacity > pool->config.maxCapacity[usage][size]) {
                    newConfig.initialCapacity = pool->config.maxCapacity[usage][size];
                }

                /* Only grow if there's actually room to grow */
                if (newConfig.initialCapacity > stats.totalAllocated) {
                    /* Create a new pool with increased capacity */
                    TinyAIMemoryPool *newPool = tinyaiMemoryPoolCreate(&newConfig);
                    if (newPool) {
                        /* Transfer all allocations to the new pool (this is expensive) */
                        /* For now, just keep using the old pool and let the system handle it */
                        tinyaiMemoryPoolDestroy(newPool);
                    }
                }
            }
            else if (usageRatio < 0.25f &&
                     stats.totalAllocated > pool->config.initialCapacity[usage][size]) {
                /* Pool is mostly empty and larger than initial size, consider shrinking */
                /* Note: Shrinking is complex and might require compaction, not implemented here */
            }

            /* If configured, compact the pool to reduce fragmentation */
            if (pool->config.aggressiveDefrag) {
                tinyaiMemoryPoolCompact(memPool);
            }
        }
    }

    return true;
}

/**
 * Register a tensor operation with the memory pool
 */
bool tinyaiAdvancedPoolRegisterTensorOp(TinyAIAdvancedMemoryPool *pool, int opType,
                                        const size_t *inputSizes, int numInputs,
                                        const size_t *outputSizes, int numOutputs)
{
    if (!pool || !pool->config.optimizeForTensorOps || !inputSizes || numInputs <= 0 ||
        numInputs > MAX_TENSORS_PER_OP || !outputSizes || numOutputs <= 0 ||
        numOutputs > MAX_TENSORS_PER_OP) {
        return false;
    }

    /* Check if we've reached the maximum number of registered ops */
    if (pool->numTensorOps >= MAX_TENSOR_OPS) {
        return false;
    }

    /* Add new operation */
    int opIndex = pool->numTensorOps;

    /* Set operation type */
    pool->tensorOps[opIndex].opType = opType;

    /* Copy input sizes */
    pool->tensorOps[opIndex].numInputs = numInputs;
    for (int i = 0; i < numInputs; i++) {
        pool->tensorOps[opIndex].inputSizes[i] = inputSizes[i];
    }

    /* Copy output sizes */
    pool->tensorOps[opIndex].numOutputs = numOutputs;
    for (int i = 0; i < numOutputs; i++) {
        pool->tensorOps[opIndex].outputSizes[i] = outputSizes[i];
    }

    /* Initialize optimized layouts to NULL */
    for (int i = 0; i < MAX_TENSORS_PER_OP * 2; i++) {
        pool->tensorOps[opIndex].optimizedLayouts[i] = NULL;
    }

    /* Increment count */
    pool->numTensorOps++;

    return true;
}

/**
 * Find a registered tensor operation
 */
static TensorOpDescriptor *findTensorOp(TinyAIAdvancedMemoryPool *pool, int opType)
{
    for (int i = 0; i < pool->numTensorOps; i++) {
        if (pool->tensorOps[i].opType == opType) {
            return &pool->tensorOps[i];
        }
    }
    return NULL;
}

/**
 * Allocate memory optimized for a specific tensor operation
 */
void *tinyaiAdvancedPoolAllocForTensorOp(TinyAIAdvancedMemoryPool *pool, int opType, bool isInput,
                                         int tensorIndex, size_t size)
{
    if (!pool || !pool->config.optimizeForTensorOps || size == 0) {
        return NULL;
    }

    /* Find the tensor operation */
    TensorOpDescriptor *op = findTensorOp(pool, opType);
    if (!op) {
        /* Operation not found, fall back to regular allocation */
        return tinyaiAdvancedPoolAlloc(pool, size, 32, TINYAI_POOL_USAGE_ACTIVATIONS);
    }

    /* Check if the tensor index is valid */
    int maxIndex = isInput ? op->numInputs : op->numOutputs;
    if (tensorIndex < 0 || tensorIndex >= maxIndex) {
        /* Invalid tensor index */
        return NULL;
    }

    /* Calculate layout index */
    int layoutIndex = isInput ? tensorIndex : op->numInputs + tensorIndex;

    /* Check if we already have an optimized layout for this tensor */
    if (op->optimizedLayouts[layoutIndex]) {
        /* We have a pre-allocated optimized layout, check if it's big enough */
        /* For simplicity, we assume the layout is always big enough - a real implementation
           would need to track sizes and might need to reallocate */
        return op->optimizedLayouts[layoutIndex];
    }

    /* Allocate memory for this tensor */
    void *memory = tinyaiAdvancedPoolAlloc(pool, size, 32, TINYAI_POOL_USAGE_ACTIVATIONS);
    if (!memory) {
        return NULL;
    }

    /* Store this layout for future reuse */
    op->optimizedLayouts[layoutIndex] = memory;

    return memory;
}

/**
 * Enable or disable thread safety for the pool
 */
void tinyaiAdvancedPoolSetThreadSafety(TinyAIAdvancedMemoryPool *pool, bool enable)
{
    if (pool) {
        pool->threadSafetyEnabled = enable;
    }
}

/**
 * Set memory pressure callback function
 */
void tinyaiAdvancedPoolSetPressureCallback(TinyAIAdvancedMemoryPool *pool,
                                           void (*callback)(void *userData, uint8_t pressureLevel),
                                           void *userData)
{
    if (pool) {
        pool->pressureCallback         = callback;
        pool->pressureCallbackUserData = userData;
    }
}

/**
 * Dump advanced memory pool information for debugging
 */
void tinyaiAdvancedPoolDump(TinyAIAdvancedMemoryPool *pool, bool dumpAllocations)
{
    if (!pool) {
        return;
    }

    /* Print general pool information */
    printf("=== Advanced Memory Pool Statistics ===\n");
    printf("Total allocations: %zu\n", pool->allocCount);
    printf("Total frees: %zu\n", pool->freeCount);
    printf("Cache hits: %zu\n", pool->cacheHits);
    printf("Cache misses: %zu\n", pool->cacheMisses);
    printf("Pool switches: %zu\n", pool->poolSwitches);
    printf("Current memory pressure: %u%%\n", pool->currentPressure);
    printf("Out of memory event occurred: %s\n", pool->outOfMemoryEventOccurred ? "Yes" : "No");

    /* Print statistics for each pool */
    printf("\n=== Individual Pool Statistics ===\n");
    const char *usageNames[TINYAI_POOL_USAGE_COUNT] = {"Weights", "Activations", "General"};

    const char *sizeNames[TINYAI_POOL_SIZE_COUNT] = {"Tiny",  "Small",   "Medium",
                                                     "Large", "X-Large", "Huge"};

    for (int usage = 0; usage < TINYAI_POOL_USAGE_COUNT; usage++) {
        for (int size = 0; size < TINYAI_POOL_SIZE_COUNT; size++) {
            TinyAIMemoryPool *memPool = pool->pools[usage][size];
            if (memPool) {
                TinyAIMemoryPoolStats stats;
                tinyaiMemoryPoolGetStats(memPool, &stats);

                printf("\n  %s-%s Pool:\n", usageNames[usage], sizeNames[size]);
                printf("    Total allocated: %zu bytes\n", stats.totalAllocated);
                printf("    Total used: %zu bytes\n", stats.totalUsed);
                printf("    Total wasted: %zu bytes\n", stats.totalWasted);
                printf("    Largest free block: %zu bytes\n", stats.largestBlock);
                printf("    Free blocks: %zu\n", stats.freeBlocks);
                printf("    Fragmentation score: %zu\n", stats.fragmentationScore);

                if (dumpAllocations) {
                    /* Delegate to the memory pool's dump function */
                    printf("\n    Allocations:\n");
                    tinyaiMemoryPoolDump(memPool, true);
                }
            }
        }
    }

    /* Print tensor operation information if any */
    if (pool->numTensorOps > 0) {
        printf("\n=== Registered Tensor Operations ===\n");
        for (int i = 0; i < pool->numTensorOps; i++) {
            TensorOpDescriptor *op = &pool->tensorOps[i];
            printf("  Operation #%d (Type %d):\n", i, op->opType);
            printf("    Inputs: %d, Outputs: %d\n", op->numInputs, op->numOutputs);
        }
    }
}