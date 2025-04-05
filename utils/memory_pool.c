/**
 * @file memory_pool.c
 * @brief Implementation of enhanced memory pool system for large models
 */

#include "memory_pool.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Alignment helpers */
#define ALIGN_UP(x, alignment) (((x) + (alignment - 1)) & ~(alignment - 1))
#define IS_ALIGNED(x, alignment) (((uintptr_t)(x) & (alignment - 1)) == 0)

/* Default pool settings */
#define DEFAULT_INITIAL_CAPACITY (1024 * 1024 * 16) /* 16 MB */
#define DEFAULT_MAX_CAPACITY (1024 * 1024 * 256)    /* 256 MB */
#define DEFAULT_BLOCK_SIZE 64
#define DEFAULT_ALLOW_GROWTH true
#define DEFAULT_TRACK_ALLOCS true

/* Constants for allocation tracking */
#define MAX_ALLOCATION_RECORDS 10000
#define ALLOCATION_MAGIC 0xA110CA73 /* "ALLOCAT" in hex */

/* SIMD Alignment requirements */
#define SIMD_ALIGNMENT 32 /* AVX needs 32-byte alignment */
#define DEFAULT_ALIGNMENT 16

/* Memory block structure */
typedef struct MemoryBlock {
    void               *address;   /* Pointer to the beginning of the block */
    size_t              size;      /* Total size of the block including header */
    size_t              usedSize;  /* Size actually used by the allocation */
    bool                isFree;    /* Whether the block is free */
    struct MemoryBlock *next;      /* Next block in the chain */
    struct MemoryBlock *prev;      /* Previous block in the chain */
    uint32_t            magic;     /* Magic number for validation */
    const char         *allocFile; /* Source file of allocation (for debugging) */
    int                 allocLine; /* Source line of allocation (for debugging) */
} MemoryBlock;

/* Memory region representing a continuous allocation from the system */
typedef struct MemoryRegion {
    void                *memory; /* The actual memory */
    size_t               size;   /* Size of the memory region */
    struct MemoryRegion *next;   /* Next region in the chain */
} MemoryRegion;

/* Memory pool structure */
struct TinyAIMemoryPool {
    MemoryRegion *regions;          /* List of memory regions */
    MemoryBlock  *blocks;           /* List of allocated and free blocks */
    size_t        totalSize;        /* Total size of all regions */
    size_t        usedSize;         /* Total used size */
    size_t        blockSize;        /* Minimum block size */
    size_t        maxCapacity;      /* Maximum capacity (0 for unlimited) */
    bool          allowGrowth;      /* Whether to allow growth */
    bool          trackAllocations; /* Whether to track allocations */

    /* Statistics */
    size_t numAllocations; /* Number of active allocations */
    size_t numFreeBlocks;  /* Number of free blocks */
    size_t peakUsage;      /* Peak memory usage */
};

/* Get default memory pool configuration */
void tinyaiMemoryPoolGetDefaultConfig(TinyAIMemoryPoolConfig *config)
{
    if (config) {
        config->initialCapacity  = DEFAULT_INITIAL_CAPACITY;
        config->maxCapacity      = DEFAULT_MAX_CAPACITY;
        config->blockSize        = DEFAULT_BLOCK_SIZE;
        config->allowGrowth      = DEFAULT_ALLOW_GROWTH;
        config->trackAllocations = DEFAULT_TRACK_ALLOCS;
    }
}

/* Create a memory block */
static MemoryBlock *createMemoryBlock(void *address, size_t size, bool isFree)
{
    MemoryBlock *block = (MemoryBlock *)address;

    block->address   = address;
    block->size      = size;
    block->usedSize  = isFree ? 0 : size;
    block->isFree    = isFree;
    block->next      = NULL;
    block->prev      = NULL;
    block->magic     = ALLOCATION_MAGIC;
    block->allocFile = NULL;
    block->allocLine = 0;

    return block;
}

/* Initialize a memory region */
static MemoryRegion *createMemoryRegion(size_t size)
{
    /* Allocate region structure and memory in one go */
    size_t totalSize = sizeof(MemoryRegion) + size;
    void  *memory    = malloc(totalSize);

    if (!memory) {
        return NULL;
    }

    MemoryRegion *region = (MemoryRegion *)memory;
    region->memory       = (char *)memory + sizeof(MemoryRegion);
    region->size         = size;
    region->next         = NULL;

    return region;
}

/* Find a free block of sufficient size */
static MemoryBlock *findFreeBlock(TinyAIMemoryPool *pool, size_t size, size_t alignment)
{
    MemoryBlock *block = pool->blocks;

    while (block) {
        if (block->isFree) {
            /* Check if this block is big enough */
            void *alignedAddr = (void *)ALIGN_UP(
                (uintptr_t)((char *)block->address + sizeof(MemoryBlock)), alignment);
            size_t availableSize = block->size - ((char *)alignedAddr - (char *)block->address);

            if (availableSize >= size) {
                return block;
            }
        }
        block = block->next;
    }

    return NULL;
}

/* Split a block if possible */
static void splitBlock(TinyAIMemoryPool *pool, MemoryBlock *block, size_t size)
{
    /* Only split if the remaining size is at least blockSize + sizeof(MemoryBlock) */
    size_t minSplitSize = pool->blockSize + sizeof(MemoryBlock);

    if (block->size - size >= minSplitSize) {
        /* Create new block at the end of this allocation */
        void        *newBlockAddr = (char *)block->address + size;
        MemoryBlock *newBlock     = createMemoryBlock(newBlockAddr, block->size - size, true);

        /* Update the original block */
        block->size = size;

        /* Insert the new block into the list */
        newBlock->next = block->next;
        newBlock->prev = block;

        if (block->next) {
            block->next->prev = newBlock;
        }

        block->next = newBlock;

        /* Update pool statistics */
        pool->numFreeBlocks++;
    }
}

/* Add a memory region to the pool */
static bool addMemoryRegion(TinyAIMemoryPool *pool, size_t size)
{
    /* Check if adding this region would exceed the max capacity */
    if (pool->maxCapacity > 0 && pool->totalSize + size > pool->maxCapacity) {
        size = pool->maxCapacity - pool->totalSize;

        /* If we can't allocate anything meaningful, return failure */
        if (size < pool->blockSize + sizeof(MemoryBlock)) {
            return false;
        }
    }

    /* Create a new region */
    MemoryRegion *region = createMemoryRegion(size);
    if (!region) {
        return false;
    }

    /* Add region to the chain */
    region->next  = pool->regions;
    pool->regions = region;

    /* Create a free block for the whole region */
    MemoryBlock *block = createMemoryBlock(region->memory, size, true);

    /* Add block to the chain */
    if (pool->blocks) {
        block->next        = pool->blocks;
        pool->blocks->prev = block;
    }

    pool->blocks = block;

    /* Update pool statistics */
    pool->totalSize += size;
    pool->numFreeBlocks++;

    return true;
}

/* Merge adjacent free blocks */
static void mergeAdjacentFreeBlocks(TinyAIMemoryPool *pool)
{
    MemoryBlock *block = pool->blocks;

    while (block && block->next) {
        if (block->isFree && block->next->isFree) {
            /* Check if blocks are adjacent */
            if ((char *)block->address + block->size == (char *)block->next->address) {
                /* Merge blocks */
                MemoryBlock *nextBlock = block->next;

                block->size += nextBlock->size;
                block->next = nextBlock->next;

                if (nextBlock->next) {
                    nextBlock->next->prev = block;
                }

                /* Update statistics */
                pool->numFreeBlocks--;

                /* Continue from the current block to catch multiple adjacents */
                continue;
            }
        }

        block = block->next;
    }
}

/* Create a new memory pool */
TinyAIMemoryPool *tinyaiMemoryPoolCreate(const TinyAIMemoryPoolConfig *config)
{
    TinyAIMemoryPoolConfig defaultConfig;

    /* Use default config if none provided */
    if (!config) {
        tinyaiMemoryPoolGetDefaultConfig(&defaultConfig);
        config = &defaultConfig;
    }

    /* Allocate the pool structure */
    TinyAIMemoryPool *pool = (TinyAIMemoryPool *)malloc(sizeof(TinyAIMemoryPool));
    if (!pool) {
        return NULL;
    }

    /* Initialize pool */
    memset(pool, 0, sizeof(TinyAIMemoryPool));
    pool->blockSize        = config->blockSize;
    pool->maxCapacity      = config->maxCapacity;
    pool->allowGrowth      = config->allowGrowth;
    pool->trackAllocations = config->trackAllocations;

    /* Allocate initial memory region */
    if (!addMemoryRegion(pool, config->initialCapacity)) {
        free(pool);
        return NULL;
    }

    return pool;
}

/* Destroy a memory pool */
void tinyaiMemoryPoolDestroy(TinyAIMemoryPool *pool)
{
    if (!pool) {
        return;
    }

    /* Free all regions */
    MemoryRegion *region = pool->regions;
    while (region) {
        MemoryRegion *next = region->next;
        free(region);
        region = next;
    }

    /* Free the pool structure */
    free(pool);
}

/* Allocate memory from pool */
void *tinyaiMemoryPoolAlloc(TinyAIMemoryPool *pool, size_t size, size_t alignment)
{
    if (!pool || size == 0) {
        return NULL;
    }

    /* Ensure alignment is at least DEFAULT_ALIGNMENT and a power of 2 */
    if (alignment < DEFAULT_ALIGNMENT) {
        alignment = DEFAULT_ALIGNMENT;
    }

    assert((alignment & (alignment - 1)) == 0); /* Power of 2 check */

    /* Adjust size to include header and alignment and ensure it's at least blockSize */
    size_t totalSize = sizeof(MemoryBlock) + size + (alignment - 1);
    if (totalSize < pool->blockSize) {
        totalSize = pool->blockSize;
    }

    /* Find a free block */
    MemoryBlock *block = findFreeBlock(pool, totalSize, alignment);

    /* If no suitable block is found, try to grow the pool */
    if (!block) {
        if (!pool->allowGrowth) {
            return NULL;
        }

        /* Calculate growth size - double the current size or enough for this allocation */
        size_t growthSize = pool->totalSize;
        if (growthSize < totalSize) {
            growthSize = totalSize;
        }

        /* Add a new region */
        if (!addMemoryRegion(pool, growthSize)) {
            return NULL;
        }

        /* Try to find a block again */
        block = findFreeBlock(pool, totalSize, alignment);
        if (!block) {
            return NULL;
        }
    }

    /* Calculate aligned address for user data */
    void *alignedAddr =
        (void *)ALIGN_UP((uintptr_t)((char *)block->address + sizeof(MemoryBlock)), alignment);
    size_t headerOffset = (char *)alignedAddr - (char *)block->address;

    /* Split the block if possible */
    size_t usedSize = headerOffset + size;
    splitBlock(pool, block, usedSize);

    /* Mark the block as used */
    block->isFree   = false;
    block->usedSize = usedSize;

    /* Update statistics */
    pool->numFreeBlocks--;
    pool->numAllocations++;
    pool->usedSize += usedSize;

    if (pool->usedSize > pool->peakUsage) {
        pool->peakUsage = pool->usedSize;
    }

    return alignedAddr;
}

/* Free memory allocated from pool */
void tinyaiMemoryPoolFree(TinyAIMemoryPool *pool, void *ptr)
{
    if (!pool || !ptr) {
        return;
    }

    /* Find the block containing this pointer */
    MemoryBlock *block = pool->blocks;
    while (block) {
        /* Check if this pointer is within this block */
        void *blockStart = (char *)block->address + sizeof(MemoryBlock);
        void *blockEnd   = (char *)block->address + block->size;

        if (ptr >= blockStart && ptr < blockEnd) {
            if (block->magic != ALLOCATION_MAGIC) {
                fprintf(stderr, "Memory corruption detected: invalid magic number\n");
                return;
            }

            if (block->isFree) {
                fprintf(stderr, "Double free detected\n");
                return;
            }

            /* Mark the block as free */
            block->isFree   = true;
            block->usedSize = 0;

            /* Update statistics */
            pool->numFreeBlocks++;
            pool->numAllocations--;
            pool->usedSize -= block->size;

            /* Try to merge adjacent free blocks */
            mergeAdjacentFreeBlocks(pool);

            return;
        }

        block = block->next;
    }

    fprintf(stderr, "Attempt to free pointer not allocated from this pool\n");
}

/* Reset a memory pool */
void tinyaiMemoryPoolReset(TinyAIMemoryPool *pool)
{
    if (!pool) {
        return;
    }

    /* Mark all blocks as free */
    MemoryBlock *block = pool->blocks;
    while (block) {
        block->isFree   = true;
        block->usedSize = 0;
        block           = block->next;
    }

    /* Reset statistics */
    pool->numAllocations = 0;
    pool->numFreeBlocks  = 0;
    pool->usedSize       = 0;

    /* Merge all adjacent free blocks */
    mergeAdjacentFreeBlocks(pool);

    /* Count the free blocks */
    block = pool->blocks;
    while (block) {
        if (block->isFree) {
            pool->numFreeBlocks++;
        }
        block = block->next;
    }
}

/* Get memory pool statistics */
void tinyaiMemoryPoolGetStats(TinyAIMemoryPool *pool, TinyAIMemoryPoolStats *stats)
{
    if (!pool || !stats) {
        return;
    }

    /* Calculate statistics */
    stats->totalAllocated = pool->totalSize;
    stats->totalUsed      = pool->usedSize;
    stats->totalWasted    = pool->totalSize - pool->usedSize;
    stats->totalBlocks    = pool->numAllocations + pool->numFreeBlocks;
    stats->freeBlocks     = pool->numFreeBlocks;

    /* Find largest free block */
    size_t       largestBlock = 0;
    MemoryBlock *block        = pool->blocks;
    while (block) {
        if (block->isFree && block->size > largestBlock) {
            largestBlock = block->size;
        }
        block = block->next;
    }
    stats->largestBlock = largestBlock;

    /* Calculate fragmentation score (0-100) */
    if (pool->totalSize > 0) {
        /* Consider both the number of free blocks and the largest block size */
        float blockFrag = 0.0f;
        if (stats->totalBlocks > 1) {
            blockFrag = (float)stats->freeBlocks / (float)stats->totalBlocks;
        }

        float sizeFrag = 0.0f;
        if (stats->totalWasted > 0) {
            sizeFrag = 1.0f - ((float)stats->largestBlock / (float)stats->totalWasted);
        }

        /* Combine the metrics (higher means more fragmented) */
        float fragmentation       = (blockFrag * 0.5f) + (sizeFrag * 0.5f);
        stats->fragmentationScore = (size_t)(fragmentation * 100.0f);
    }
    else {
        stats->fragmentationScore = 0;
    }
}

/* Check if a pointer was allocated from the given pool */
bool tinyaiMemoryPoolContains(TinyAIMemoryPool *pool, const void *ptr)
{
    if (!pool || !ptr) {
        return false;
    }

    /* Check if the pointer is within any of the blocks */
    MemoryBlock *block = pool->blocks;
    while (block) {
        void *blockStart = (char *)block->address + sizeof(MemoryBlock);
        void *blockEnd   = (char *)block->address + block->size;

        if (ptr >= blockStart && ptr < blockEnd && !block->isFree) {
            return true;
        }

        block = block->next;
    }

    return false;
}

/* Specialized 4-bit weight allocation */
uint8_t *tinyaiMemoryPoolAllocWeights4Bit(TinyAIMemoryPool *pool, size_t rows, size_t cols,
                                          bool requiresSIMD)
{
    if (!pool || rows == 0 || cols == 0) {
        return NULL;
    }

    /* For 4-bit weights, we need cols/2 bytes per row (2 weights per byte) */
    size_t bytesPerRow = (cols + 1) / 2;
    size_t totalBytes  = rows * bytesPerRow;

    /* Allocate with SIMD alignment if required */
    size_t alignment = requiresSIMD ? SIMD_ALIGNMENT : DEFAULT_ALIGNMENT;

    return (uint8_t *)tinyaiMemoryPoolAlloc(pool, totalBytes, alignment);
}

/* Specialized activation allocation */
float *tinyaiMemoryPoolAllocActivations(TinyAIMemoryPool *pool, size_t size, bool requiresSIMD)
{
    if (!pool || size == 0) {
        return NULL;
    }

    /* Allocate with SIMD alignment if required */
    size_t alignment = requiresSIMD ? SIMD_ALIGNMENT : DEFAULT_ALIGNMENT;

    return (float *)tinyaiMemoryPoolAlloc(pool, size * sizeof(float), alignment);
}

/* Reallocate memory block */
void *tinyaiMemoryPoolRealloc(TinyAIMemoryPool *pool, void *ptr, size_t size, size_t alignment)
{
    if (!pool) {
        return NULL;
    }

    /* If ptr is NULL, this is equivalent to alloc */
    if (!ptr) {
        return tinyaiMemoryPoolAlloc(pool, size, alignment);
    }

    /* If size is 0, this is equivalent to free */
    if (size == 0) {
        tinyaiMemoryPoolFree(pool, ptr);
        return NULL;
    }

    /* Ensure alignment is adequate and a power of 2 */
    if (alignment < DEFAULT_ALIGNMENT) {
        alignment = DEFAULT_ALIGNMENT;
    }

    assert((alignment & (alignment - 1)) == 0); /* Power of 2 check */

    /* Find the block containing this pointer */
    MemoryBlock *block = pool->blocks;
    while (block) {
        void *blockStart = (char *)block->address + sizeof(MemoryBlock);
        void *blockEnd   = (char *)block->address + block->size;

        if (ptr >= blockStart && ptr < blockEnd && !block->isFree) {
            /* Calculate the current user data size */
            size_t headerOffset = (char *)ptr - (char *)block->address;
            size_t currentSize  = block->size - headerOffset;

            /* If the existing block is big enough, just return it */
            if (currentSize >= size) {
                return ptr;
            }

            /* Otherwise allocate a new block and copy the data */
            void *newPtr = tinyaiMemoryPoolAlloc(pool, size, alignment);
            if (!newPtr) {
                return NULL;
            }

            memcpy(newPtr, ptr, currentSize);
            tinyaiMemoryPoolFree(pool, ptr);

            return newPtr;
        }

        block = block->next;
    }

    /* If we get here, the pointer wasn't allocated from this pool */
    return NULL;
}

/* Compact pool memory to reduce fragmentation */
bool tinyaiMemoryPoolCompact(TinyAIMemoryPool *pool)
{
    if (!pool) {
        return false;
    }

    /* First merge adjacent free blocks */
    mergeAdjacentFreeBlocks(pool);

    /* TODO: Implement more advanced compaction if needed */

    return true;
}

/* Dump memory pool information for debugging */
void tinyaiMemoryPoolDump(TinyAIMemoryPool *pool, bool dumpAllocations)
{
    if (!pool) {
        return;
    }

    printf("Memory Pool Summary:\n");
    printf("  Total Size: %zu bytes (%.2f MB)\n", pool->totalSize,
           (double)pool->totalSize / (1024.0 * 1024.0));
    printf("  Used Size: %zu bytes (%.2f MB)\n", pool->usedSize,
           (double)pool->usedSize / (1024.0 * 1024.0));
    printf("  Active Allocations: %zu\n", pool->numAllocations);
    printf("  Free Blocks: %zu\n", pool->numFreeBlocks);
    printf("  Peak Usage: %zu bytes (%.2f MB)\n", pool->peakUsage,
           (double)pool->peakUsage / (1024.0 * 1024.0));

    /* Get detailed statistics */
    TinyAIMemoryPoolStats stats;
    tinyaiMemoryPoolGetStats(pool, &stats);

    printf("  Fragmentation Score: %zu%%\n", stats.fragmentationScore);
    printf("  Largest Free Block: %zu bytes (%.2f MB)\n", stats.largestBlock,
           (double)stats.largestBlock / (1024.0 * 1024.0));
    printf("  Wasted Memory: %zu bytes (%.2f MB)\n", stats.totalWasted,
           (double)stats.totalWasted / (1024.0 * 1024.0));

    if (dumpAllocations) {
        printf("\nBlock Details:\n");
        MemoryBlock *block      = pool->blocks;
        int          blockCount = 0;

        while (block) {
            printf("  Block %d: addr=%p size=%zu used=%zu %s\n", blockCount++, block->address,
                   block->size, block->usedSize, block->isFree ? "FREE" : "USED");

            block = block->next;
        }
    }
}
