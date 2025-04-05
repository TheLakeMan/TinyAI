/**
 * @file mmap_loader.h
 * @brief Memory-mapped model loading utilities for TinyAI
 *
 * This header provides utilities for memory-mapped access to model weights,
 * allowing models to be used directly from storage without fully loading into RAM.
 */

#ifndef TINYAI_MMAP_LOADER_H
#define TINYAI_MMAP_LOADER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Structure representing a memory-mapped model file
 */
typedef struct TinyAIMappedModel TinyAIMappedModel;

/**
 * Layer access descriptor for memory-mapped models
 */
typedef struct {
    size_t   offset;        /* Offset of layer weights in the mapped file */
    size_t   size;          /* Size of layer weights in bytes */
    int      layerIndex;    /* Index of the layer in the model */
    int      precision;     /* Precision of the layer weights (bits) */
    void    *cachedWeights; /* Pointer to cached weights (NULL if not cached) */
    bool     isActive;      /* Whether this layer is currently loaded */
    float    priority;      /* Priority for cache retention (higher = more likely to keep) */
    uint64_t lastAccessed;  /* Timestamp of last access */
    uint64_t accessCount;   /* Number of times this layer has been accessed */
} TinyAILayerDescriptor;

/**
 * Memory-mapped model configuration
 */
typedef struct {
    size_t maxCacheSize;      /* Maximum size of in-memory cache in bytes */
    bool   prefetchEnabled;   /* Whether to enable weight prefetching */
    int    prefetchThreads;   /* Number of threads to use for prefetching */
    bool   adaptiveCaching;   /* Whether to use adaptive caching based on access patterns */
    size_t minLayerCacheSize; /* Minimum cache size per layer in bytes */
} TinyAIMmapConfig;

/**
 * Open a memory-mapped model file
 *
 * @param filepath Path to the model file
 * @param config Memory-mapped loading configuration
 * @return Pointer to the memory-mapped model structure, or NULL on failure
 */
TinyAIMappedModel *tinyaiOpenMappedModel(const char *filepath, const TinyAIMmapConfig *config);

/**
 * Close a memory-mapped model and release resources
 *
 * @param model Pointer to the memory-mapped model to close
 */
void tinyaiCloseMappedModel(TinyAIMappedModel *model);

/**
 * Get the number of layers in a memory-mapped model
 *
 * @param model Pointer to the memory-mapped model
 * @return Number of layers, or -1 on error
 */
int tinyaiGetMappedLayerCount(const TinyAIMappedModel *model);

/**
 * Get descriptor for a specific layer in the model
 *
 * @param model Pointer to the memory-mapped model
 * @param layerIndex Index of the layer to get
 * @return Layer descriptor, or NULL on error
 */
const TinyAILayerDescriptor *tinyaiGetLayerDescriptor(const TinyAIMappedModel *model,
                                                      int                      layerIndex);

/**
 * Get pointer to a layer's weights, loading from disk if necessary
 *
 * @param model Pointer to the memory-mapped model
 * @param layerIndex Index of the layer to load
 * @return Pointer to the layer weights, or NULL on error
 */
void *tinyaiGetLayerWeights(TinyAIMappedModel *model, int layerIndex);

/**
 * Prefetch a layer's weights into memory in anticipation of use
 *
 * @param model Pointer to the memory-mapped model
 * @param layerIndex Index of the layer to prefetch
 * @return true if prefetching was successful, false otherwise
 */
bool tinyaiPrefetchLayerWeights(TinyAIMappedModel *model, int layerIndex);

/**
 * Release a layer's weights from memory to free up space
 *
 * @param model Pointer to the memory-mapped model
 * @param layerIndex Index of the layer to unload
 */
void tinyaiReleaseLayerWeights(TinyAIMappedModel *model, int layerIndex);

/**
 * Create a default memory-mapped model configuration
 *
 * @return Default configuration
 */
TinyAIMmapConfig tinyaiCreateDefaultMmapConfig(void);

/**
 * Get current memory usage of the mapped model cache
 *
 * @param model Pointer to the memory-mapped model
 * @return Memory usage in bytes
 */
size_t tinyaiGetMappedModelMemoryUsage(const TinyAIMappedModel *model);

/**
 * Set layer access priority for caching
 *
 * @param model Pointer to the memory-mapped model
 * @param layerIndex Index of the layer to set priority for
 * @param priority Priority value (higher = more likely to keep in cache)
 */
void tinyaiSetLayerPriority(TinyAIMappedModel *model, int layerIndex, float priority);

/**
 * Reset all layer cache priorities to default values
 *
 * @param model Pointer to the memory-mapped model
 */
void tinyaiResetLayerPriorities(TinyAIMappedModel *model);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_MMAP_LOADER_H */
