/**
 * TinyAI Memory Management Implementation
 */

#include "memory.h"
#include <stdlib.h> // For malloc, realloc, free, calloc
#include <stdio.h>  // For printf (debugging/leaks)
#include <string.h> // For memset

/* ----------------- Standard Allocation Wrappers ----------------- */

void* tinyaiAlloc(size_t size) {
    // Basic wrapper around malloc
    return malloc(size);
}

void* tinyaiRealloc(void *ptr, size_t size) {
    // Basic wrapper around realloc
    return realloc(ptr, size);
}

void tinyaiFree(void *ptr) {
    // Basic wrapper around free
    free(ptr);
}

void* tinyaiCalloc(size_t count, size_t size) {
    // Basic wrapper around calloc
    return calloc(count, size);
}

/* ----------------- Memory Pool (Simple Bump Allocator) ----------------- */

static unsigned char *g_memPool = NULL;
static size_t g_memPoolTotalSize = 0;
static size_t g_memPoolUsedSize = 0;
static size_t g_memPoolPeakSize = 0;
static size_t g_memPoolAllocCount = 0;
static size_t g_memPoolOffset = 0; // Current allocation offset

int tinyaiMemPoolInit(size_t size) {
    if (g_memPool != NULL) {
        fprintf(stderr, "Error: Memory pool already initialized.\n");
        return 1; // Already initialized
    }
    if (size == 0) {
        fprintf(stderr, "Error: Memory pool size cannot be zero.\n");
        return 1; // Invalid size
    }

    g_memPool = (unsigned char *)malloc(size);
    if (g_memPool == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory pool of size %zu bytes.\n", size);
        return 1; // Allocation failed
    }

    g_memPoolTotalSize = size;
    g_memPoolUsedSize = 0;
    g_memPoolPeakSize = 0;
    g_memPoolAllocCount = 0;
    g_memPoolOffset = 0;

    // Optional: Zero initialize the pool
    // memset(g_memPool, 0, size); 

    return 0; // Success
}

void tinyaiMemPoolCleanup() {
    if (g_memPool != NULL) {
        free(g_memPool);
        g_memPool = NULL;
        g_memPoolTotalSize = 0;
        g_memPoolUsedSize = 0;
        g_memPoolPeakSize = 0;
        g_memPoolAllocCount = 0;
        g_memPoolOffset = 0;
    }
}

void* tinyaiMemPoolAlloc(size_t size) {
    if (g_memPool == NULL) {
        fprintf(stderr, "Error: Memory pool not initialized.\n");
        return NULL;
    }
    if (size == 0) {
        return NULL; // Cannot allocate zero bytes
    }

    // Basic alignment (e.g., align to pointer size)
    size_t alignment = sizeof(void*);
    size_t aligned_offset = (g_memPoolOffset + alignment - 1) & ~(alignment - 1);
    
    if (aligned_offset + size > g_memPoolTotalSize) {
        fprintf(stderr, "Error: Memory pool out of memory. Requested %zu, available %zu.\n", 
                size, g_memPoolTotalSize - aligned_offset);
        return NULL; // Out of memory
    }

    void *ptr = g_memPool + aligned_offset;
    g_memPoolOffset = aligned_offset + size;
    g_memPoolUsedSize = g_memPoolOffset; // In bump allocator, used size is the current offset
    g_memPoolAllocCount++;

    if (g_memPoolUsedSize > g_memPoolPeakSize) {
        g_memPoolPeakSize = g_memPoolUsedSize;
    }

    // Optional: Zero initialize allocated memory
    // memset(ptr, 0, size);

    return ptr;
}

void tinyaiMemPoolFree(void *ptr) {
    // Simple bump allocator: Free is typically a no-op or only allows freeing the last allocation.
    // For simplicity, we make it a no-op here. Reset the pool to free everything.
    // A more complex pool would need to manage free blocks.
    (void)ptr; // Suppress unused parameter warning
}

void tinyaiMemPoolReset() {
    if (g_memPool != NULL) {
        g_memPoolUsedSize = 0;
        g_memPoolAllocCount = 0;
        g_memPoolOffset = 0;
        // Optional: Zero the pool memory if desired on reset
        // memset(g_memPool, 0, g_memPoolTotalSize);
    }
}

void tinyaiMemPoolStats(size_t *totalSize, size_t *usedSize, 
                       size_t *peakSize, size_t *allocCount) {
    if (totalSize) *totalSize = g_memPoolTotalSize;
    if (usedSize) *usedSize = g_memPoolUsedSize;
    if (peakSize) *peakSize = g_memPoolPeakSize;
    if (allocCount) *allocCount = g_memPoolAllocCount;
}


/* ----------------- Memory Tracking (Simple Linked List) ----------------- */

#ifdef TINYAI_MEMORY_TRACKING

typedef struct MemTrackNode {
    void *ptr;
    size_t size;
    const char *file;
    int line;
    struct MemTrackNode *next;
} MemTrackNode;

static MemTrackNode *g_memTrackHead = NULL;
static size_t g_memTrackAllocCount = 0;
static size_t g_memTrackAllocSize = 0;
static size_t g_memTrackFreeCount = 0;
// Note: Tracking total freed size isn't straightforward without storing original size on free

int tinyaiMemTrackInit() {
    g_memTrackHead = NULL;
    g_memTrackAllocCount = 0;
    g_memTrackAllocSize = 0;
    g_memTrackFreeCount = 0;
    return 0; // Success
}

void tinyaiMemTrackCleanup() {
    // Dump leaks before cleaning up
    tinyaiMemTrackDumpLeaks(); 

    // Free any remaining nodes in the tracking list itself
    MemTrackNode *current = g_memTrackHead;
    while (current != NULL) {
        MemTrackNode *next = current->next;
        free(current); // Free the tracking node itself
        current = next;
    }
    g_memTrackHead = NULL;
}

void tinyaiMemTrackAlloc(void *ptr, size_t size, const char *file, int line) {
    if (ptr == NULL) {
        // Don't track failed allocations
        return;
    }

    // Allocate a node to store tracking info
    MemTrackNode *newNode = (MemTrackNode *)malloc(sizeof(MemTrackNode));
    if (newNode == NULL) {
        fprintf(stderr, "FATAL: Failed to allocate memory for tracking node!\n");
        // In a real scenario, might need a more robust way to handle this
        return; 
    }

    newNode->ptr = ptr;
    newNode->size = size;
    newNode->file = file;
    newNode->line = line;
    newNode->next = g_memTrackHead; // Prepend to list
    g_memTrackHead = newNode;

    g_memTrackAllocCount++;
    g_memTrackAllocSize += size;
}

void tinyaiMemTrackFree(void *ptr) {
    if (ptr == NULL) {
        return; // Don't track freeing NULL
    }

    MemTrackNode *current = g_memTrackHead;
    MemTrackNode *prev = NULL;

    while (current != NULL) {
        if (current->ptr == ptr) {
            // Found the allocation to remove
            if (prev == NULL) {
                // Removing the head node
                g_memTrackHead = current->next;
            } else {
                prev->next = current->next;
            }

            g_memTrackFreeCount++;
            // g_memTrackAllocSize -= current->size; // Adjust current allocated size
            
            free(current); // Free the tracking node itself
            return;
        }
        prev = current;
        current = current->next;
    }

    // If we reach here, ptr was not found in the tracked list
    // This could indicate a double free or freeing untracked memory
    fprintf(stderr, "Warning: Attempting to free untracked or already freed memory: %p\n", ptr);
}

int tinyaiMemTrackDumpLeaks() {
    int leakCount = 0;
    MemTrackNode *current = g_memTrackHead;
    size_t totalLeakedSize = 0;

    if (current != NULL) {
        printf("--- TinyAI Memory Leak Report ---\n");
    }

    while (current != NULL) {
        printf("  Leak detected: %p (%zu bytes) allocated at %s:%d\n",
               current->ptr, current->size, current->file, current->line);
        leakCount++;
        totalLeakedSize += current->size;
        current = current->next;
    }

    if (leakCount > 0) {
        printf("--- Total Leaked: %d blocks, %zu bytes ---\n", leakCount, totalLeakedSize);
    } else {
         printf("--- TinyAI Memory Leak Report: No leaks detected ---\n");
    }

    return leakCount;
}

void tinyaiMemTrackStats(size_t *allocCount, size_t *allocSize, 
                        size_t *freeCount, size_t *freeSize) {
    // Note: freeSize is tricky to track accurately without storing size on free
    if (allocCount) *allocCount = g_memTrackAllocCount;
    if (allocSize) *allocSize = g_memTrackAllocSize; // Total ever allocated
    if (freeCount) *freeCount = g_memTrackFreeCount;
    if (freeSize) *freeSize = 0; // Placeholder - not accurately tracked here
}

#else // TINYAI_MEMORY_TRACKING not defined

// Provide empty stub functions if tracking is disabled

int tinyaiMemTrackInit() { return 0; }
void tinyaiMemTrackCleanup() {}
void tinyaiMemTrackAlloc(void *ptr, size_t size, const char *file, int line) { (void)ptr; (void)size; (void)file; (void)line; }
void tinyaiMemTrackFree(void *ptr) { (void)ptr; }
int tinyaiMemTrackDumpLeaks() { return 0; }
void tinyaiMemTrackStats(size_t *allocCount, size_t *allocSize, 
                        size_t *freeCount, size_t *freeSize) {
    if (allocCount) *allocCount = 0;
    if (allocSize) *allocSize = 0;
    if (freeCount) *freeCount = 0;
    if (freeSize) *freeSize = 0;
}

#endif // TINYAI_MEMORY_TRACKING
