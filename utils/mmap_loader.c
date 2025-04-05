/**
 * @file mmap_loader.c
 * @brief Implementation of memory-mapped model loading utilities for TinyAI
 */

#include "mmap_loader.h"
#include <fcntl.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Include appropriate headers for memory mapping based on platform */
#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

/* Include threading support */
#ifdef _WIN32
#include <process.h>
typedef HANDLE ThreadHandle;
#else
#include <pthread.h>
typedef pthread_t ThreadHandle;
#endif

/* Model header magic number */
#define TINYAI_MODEL_MAGIC 0x544D4149 /* "TMAI" in ASCII */

/* Maximum number of layers in a model */
#define MAX_LAYERS 256

/* Memory-mapped model structure */
struct TinyAIMappedModel {
    /* File mapping info */
#ifdef _WIN32
    HANDLE fileHandle;
    HANDLE mappingHandle;
#else
    int fileDescriptor;
#endif
    void  *mappedData;
    size_t mappedSize;

    /* Model metadata */
    char     modelName[64];
    uint32_t layerCount;
    uint32_t version;

    /* Layer descriptors */
    TinyAILayerDescriptor layers[MAX_LAYERS];

    /* Caching info */
    TinyAIMmapConfig config;
    size_t           currentCacheSize;

    /* Prefetching */
    ThreadHandle prefetchThread;
    bool         prefetchRunning;
    int          nextPrefetchLayer;

    /* Lock for thread safety */
#ifdef _WIN32
    CRITICAL_SECTION lock;
#else
    pthread_mutex_t lock;
#endif

    /* Timestamp for access tracking */
    uint64_t timestamp;
};

/* Get current timestamp in milliseconds */
static uint64_t getCurrentTimestamp()
{
#ifdef _WIN32
    return GetTickCount64();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
#endif
}

/* Lock the model for thread safety */
static void lockModel(TinyAIMappedModel *model)
{
#ifdef _WIN32
    EnterCriticalSection(&model->lock);
#else
    pthread_mutex_lock(&model->lock);
#endif
}

/* Unlock the model */
static void unlockModel(TinyAIMappedModel *model)
{
#ifdef _WIN32
    LeaveCriticalSection(&model->lock);
#else
    pthread_mutex_unlock(&model->lock);
#endif
}

/* Calculate priority score for a layer (for cache eviction decision) */
static float calculatePriorityScore(const TinyAILayerDescriptor *layer, uint64_t currentTime)
{
    /* Blend of priority, recency and frequency */
    float recencyScore   = 1.0f / (1.0f + (currentTime - layer->lastAccessed) / 1000.0f);
    float frequencyScore = layer->accessCount / 100.0f; /* Normalize */

    /* Combine scores - priority is most important, then recency, then frequency */
    return layer->priority * 0.6f + recencyScore * 0.3f + frequencyScore * 0.1f;
}

/* Find the least important layer for cache eviction */
static int findLayerToEvict(TinyAIMappedModel *model)
{
    int      worstLayerIndex = -1;
    float    worstScore      = 3.402823466e+38f; /* Approximately FLT_MAX */
    uint64_t currentTime     = model->timestamp;

    for (int i = 0; i < model->layerCount; i++) {
        if (model->layers[i].isActive && model->layers[i].cachedWeights) {
            float score = calculatePriorityScore(&model->layers[i], currentTime);
            if (score < worstScore) {
                worstScore      = score;
                worstLayerIndex = i;
            }
        }
    }

    return worstLayerIndex;
}

/* Free memory allocated for a layer's cached weights */
static void freeLayerCache(TinyAIMappedModel *model, int layerIndex)
{
    if (layerIndex < 0 || layerIndex >= model->layerCount) {
        return;
    }

    TinyAILayerDescriptor *layer = &model->layers[layerIndex];
    if (layer->cachedWeights) {
        free(layer->cachedWeights);
        layer->cachedWeights = NULL;
        model->currentCacheSize -= layer->size;
        layer->isActive = false;
    }
}

/* Make space in the cache for a new layer */
static bool ensureCacheSpace(TinyAIMappedModel *model, size_t requiredSize)
{
    /* If we have enough space, just return */
    if (model->currentCacheSize + requiredSize <= model->config.maxCacheSize) {
        return true;
    }

    /* Need to evict layers until we have enough space */
    while (model->currentCacheSize + requiredSize > model->config.maxCacheSize) {
        int layerToEvict = findLayerToEvict(model);
        if (layerToEvict < 0) {
            /* No layers to evict, cannot make space */
            return false;
        }

        freeLayerCache(model, layerToEvict);
    }

    return true;
}

/* Load a layer's weights into memory */
static bool loadLayerWeights(TinyAIMappedModel *model, int layerIndex)
{
    if (layerIndex < 0 || layerIndex >= model->layerCount) {
        return false;
    }

    TinyAILayerDescriptor *layer = &model->layers[layerIndex];

    /* Check if already cached */
    if (layer->cachedWeights) {
        /* Update access statistics */
        layer->lastAccessed = model->timestamp;
        layer->accessCount++;
        layer->isActive = true;
        return true;
    }

    /* Make space in the cache */
    if (!ensureCacheSpace(model, layer->size)) {
        return false;
    }

    /* Allocate memory for cached weights */
    layer->cachedWeights = malloc(layer->size);
    if (!layer->cachedWeights) {
        return false;
    }

    /* Copy from memory-mapped file */
    memcpy(layer->cachedWeights, (uint8_t *)model->mappedData + layer->offset, layer->size);

    /* Update cache size and access statistics */
    model->currentCacheSize += layer->size;
    layer->lastAccessed = model->timestamp;
    layer->accessCount++;
    layer->isActive = true;

    return true;
}

/* Thread function for prefetching layers */
#ifdef _WIN32
static unsigned __stdcall prefetchThreadFunc(void *param)
{
#else
static void *prefetchThreadFunc(void *param)
{
#endif
    TinyAIMappedModel *model = (TinyAIMappedModel *)param;

    while (model->prefetchRunning) {
        /* Lock the model */
        lockModel(model);

        /* Find the next layer to prefetch */
        int layerToPrefetch = model->nextPrefetchLayer;

        /* If we've prefetched all layers, start over */
        if (layerToPrefetch >= model->layerCount) {
            layerToPrefetch = 0;
        }

        model->nextPrefetchLayer = layerToPrefetch + 1;

        /* Skip if already cached */
        bool shouldPrefetch = !model->layers[layerToPrefetch].cachedWeights;

        /* Unlock before potentially long operation */
        unlockModel(model);

        /* Prefetch the layer if needed */
        if (shouldPrefetch) {
            /* Try to load, but don't force eviction of higher priority layers */
            lockModel(model);
            loadLayerWeights(model, layerToPrefetch);
            unlockModel(model);
        }

        /* Sleep a bit to avoid hogging resources */
#ifdef _WIN32
        Sleep(10);
#else
        usleep(10000);
#endif
    }

#ifdef _WIN32
    _endthreadex(0);
    return 0;
#else
    return NULL;
#endif
}

/* Start the prefetch thread */
static bool startPrefetchThread(TinyAIMappedModel *model)
{
    if (!model->config.prefetchEnabled) {
        return true;
    }

    model->prefetchRunning   = true;
    model->nextPrefetchLayer = 0;

#ifdef _WIN32
    model->prefetchThread = (HANDLE)_beginthreadex(NULL, 0, prefetchThreadFunc, model, 0, NULL);
    return model->prefetchThread != NULL;
#else
    return pthread_create(&model->prefetchThread, NULL, prefetchThreadFunc, model) == 0;
#endif
}

/* Stop the prefetch thread */
static void stopPrefetchThread(TinyAIMappedModel *model)
{
    if (!model->prefetchRunning) {
        return;
    }

    model->prefetchRunning = false;

#ifdef _WIN32
    WaitForSingleObject(model->prefetchThread, INFINITE);
    CloseHandle(model->prefetchThread);
#else
    pthread_join(model->prefetchThread, NULL);
#endif
}

/* Implementation of public API */

TinyAIMappedModel *tinyaiOpenMappedModel(const char *filepath, const TinyAIMmapConfig *config)
{
    if (!filepath) {
        return NULL;
    }

    /* Allocate model structure */
    TinyAIMappedModel *model = (TinyAIMappedModel *)malloc(sizeof(TinyAIMappedModel));
    if (!model) {
        return NULL;
    }
    memset(model, 0, sizeof(TinyAIMappedModel));

    /* Copy configuration */
    if (config) {
        model->config = *config;
    }
    else {
        model->config = tinyaiCreateDefaultMmapConfig();
    }

#ifdef _WIN32
    /* Initialize critical section */
    InitializeCriticalSection(&model->lock);

    /* Open the file */
    model->fileHandle = CreateFile(filepath, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                                   FILE_ATTRIBUTE_NORMAL, NULL);

    if (model->fileHandle == INVALID_HANDLE_VALUE) {
        free(model);
        return NULL;
    }

    /* Get file size */
    DWORD highSize;
    DWORD lowSize     = GetFileSize(model->fileHandle, &highSize);
    model->mappedSize = (((size_t)highSize) << 32) | lowSize;

    /* Create file mapping */
    model->mappingHandle =
        CreateFileMapping(model->fileHandle, NULL, PAGE_READONLY, highSize, lowSize, NULL);

    if (!model->mappingHandle) {
        CloseHandle(model->fileHandle);
        free(model);
        return NULL;
    }

    /* Map the file into memory */
    model->mappedData =
        MapViewOfFile(model->mappingHandle, FILE_MAP_READ, 0, 0, 0 /* Map the entire file */
        );

    if (!model->mappedData) {
        CloseHandle(model->mappingHandle);
        CloseHandle(model->fileHandle);
        free(model);
        return NULL;
    }
#else
    /* Initialize mutex */
    pthread_mutex_init(&model->lock, NULL);

    /* Open the file */
    model->fileDescriptor = open(filepath, O_RDONLY);
    if (model->fileDescriptor < 0) {
        free(model);
        return NULL;
    }

    /* Get file size */
    struct stat st;
    if (fstat(model->fileDescriptor, &st) < 0) {
        close(model->fileDescriptor);
        free(model);
        return NULL;
    }
    model->mappedSize = st.st_size;

    /* Map the file into memory */
    model->mappedData =
        mmap(NULL, model->mappedSize, PROT_READ, MAP_PRIVATE, model->fileDescriptor, 0);

    if (model->mappedData == MAP_FAILED) {
        close(model->fileDescriptor);
        free(model);
        return NULL;
    }
#endif

    /* Read model header and verify magic number */
    uint32_t *header = (uint32_t *)model->mappedData;
    if (header[0] != TINYAI_MODEL_MAGIC) {
        /* Not a valid TinyAI model file */
        tinyaiCloseMappedModel(model);
        return NULL;
    }

    /* Extract model metadata */
    model->version    = header[1];
    model->layerCount = header[2];
    if (model->layerCount > MAX_LAYERS) {
        /* Too many layers */
        tinyaiCloseMappedModel(model);
        return NULL;
    }

    /* Copy model name */
    strncpy(model->modelName, (char *)(header + 4), 64);
    model->modelName[63] = '\0';

    /* Read layer descriptors */
    uint8_t *ptr = (uint8_t *)model->mappedData + 256; /* Header size */
    for (uint32_t i = 0; i < model->layerCount; i++) {
        uint32_t *layerHeader          = (uint32_t *)ptr;
        model->layers[i].offset        = layerHeader[0];
        model->layers[i].size          = layerHeader[1];
        model->layers[i].layerIndex    = i;
        model->layers[i].precision     = layerHeader[2];
        model->layers[i].cachedWeights = NULL;
        model->layers[i].isActive      = false;
        model->layers[i].priority      = 1.0f;
        model->layers[i].lastAccessed  = 0;
        model->layers[i].accessCount   = 0;

        ptr += 32; /* Layer descriptor size */
    }

    /* Initialize cache */
    model->currentCacheSize = 0;
    model->timestamp        = getCurrentTimestamp();

    /* Start prefetch thread if enabled */
    if (model->config.prefetchEnabled) {
        if (!startPrefetchThread(model)) {
            /* Failed to start prefetch thread */
            tinyaiCloseMappedModel(model);
            return NULL;
        }
    }

    return model;
}

void tinyaiCloseMappedModel(TinyAIMappedModel *model)
{
    if (!model) {
        return;
    }

    /* Stop prefetch thread */
    if (model->prefetchRunning) {
        stopPrefetchThread(model);
    }

    /* Free cached layer weights */
    for (uint32_t i = 0; i < model->layerCount; i++) {
        if (model->layers[i].cachedWeights) {
            free(model->layers[i].cachedWeights);
            model->layers[i].cachedWeights = NULL;
        }
    }

#ifdef _WIN32
    /* Unmap and close file */
    if (model->mappedData) {
        UnmapViewOfFile(model->mappedData);
    }
    if (model->mappingHandle) {
        CloseHandle(model->mappingHandle);
    }
    if (model->fileHandle != INVALID_HANDLE_VALUE) {
        CloseHandle(model->fileHandle);
    }

    /* Delete critical section */
    DeleteCriticalSection(&model->lock);
#else
    /* Unmap and close file */
    if (model->mappedData && model->mappedData != MAP_FAILED) {
        munmap(model->mappedData, model->mappedSize);
    }
    if (model->fileDescriptor >= 0) {
        close(model->fileDescriptor);
    }

    /* Destroy mutex */
    pthread_mutex_destroy(&model->lock);
#endif

    /* Free model structure */
    free(model);
}

int tinyaiGetMappedLayerCount(const TinyAIMappedModel *model)
{
    if (!model) {
        return -1;
    }

    return model->layerCount;
}

const TinyAILayerDescriptor *tinyaiGetLayerDescriptor(const TinyAIMappedModel *model,
                                                      int                      layerIndex)
{
    if (!model || layerIndex < 0 || layerIndex >= model->layerCount) {
        return NULL;
    }

    return &model->layers[layerIndex];
}

void *tinyaiGetLayerWeights(TinyAIMappedModel *model, int layerIndex)
{
    if (!model || layerIndex < 0 || layerIndex >= model->layerCount) {
        return NULL;
    }

    /* Update timestamp */
    model->timestamp = getCurrentTimestamp();

    /* Lock for thread safety */
    lockModel(model);

    /* Load layer weights if needed */
    bool success = loadLayerWeights(model, layerIndex);

    /* Get the cached weights */
    void *weights = success ? model->layers[layerIndex].cachedWeights : NULL;

    /* Unlock */
    unlockModel(model);

    return weights;
}

bool tinyaiPrefetchLayerWeights(TinyAIMappedModel *model, int layerIndex)
{
    if (!model || layerIndex < 0 || layerIndex >= model->layerCount) {
        return false;
    }

    /* Lock for thread safety */
    lockModel(model);

    /* Try to load the layer */
    bool success = loadLayerWeights(model, layerIndex);

    /* Unlock */
    unlockModel(model);

    return success;
}

void tinyaiReleaseLayerWeights(TinyAIMappedModel *model, int layerIndex)
{
    if (!model || layerIndex < 0 || layerIndex >= model->layerCount) {
        return;
    }

    /* Lock for thread safety */
    lockModel(model);

    /* Free layer cache */
    freeLayerCache(model, layerIndex);

    /* Unlock */
    unlockModel(model);
}

TinyAIMmapConfig tinyaiCreateDefaultMmapConfig(void)
{
    TinyAIMmapConfig config;

    /* Default to 256MB cache */
    config.maxCacheSize = 256 * 1024 * 1024;

    /* Enable prefetching by default */
    config.prefetchEnabled = true;

    /* Use 1 prefetch thread by default */
    config.prefetchThreads = 1;

    /* Enable adaptive caching by default */
    config.adaptiveCaching = true;

    /* Minimum 4KB per layer cache */
    config.minLayerCacheSize = 4 * 1024;

    return config;
}

size_t tinyaiGetMappedModelMemoryUsage(const TinyAIMappedModel *model)
{
    if (!model) {
        return 0;
    }

    return model->currentCacheSize;
}

void tinyaiSetLayerPriority(TinyAIMappedModel *model, int layerIndex, float priority)
{
    if (!model || layerIndex < 0 || layerIndex >= model->layerCount) {
        return;
    }

    /* Lock for thread safety */
    lockModel(model);

    /* Set priority */
    model->layers[layerIndex].priority = priority;

    /* Unlock */
    unlockModel(model);
}

void tinyaiResetLayerPriorities(TinyAIMappedModel *model)
{
    if (!model) {
        return;
    }

    /* Lock for thread safety */
    lockModel(model);

    /* Reset priorities */
    for (uint32_t i = 0; i < model->layerCount; i++) {
        model->layers[i].priority = 1.0f;
    }

    /* Unlock */
    unlockModel(model);
}
