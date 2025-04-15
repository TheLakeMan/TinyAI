/**
 * @file progressive_loader.h
 * @brief Progressive model loading utilities for TinyAI
 *
 * This header provides utilities for progressively loading model weights,
 * allowing large models to be utilized with a minimal memory footprint by
 * loading and unloading layers on demand.
 */

#ifndef TINYAI_PROGRESSIVE_LOADER_H
#define TINYAI_PROGRESSIVE_LOADER_H

#include "mmap_loader.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Layer state in the progressive loader
 */
typedef enum {
    TINYAI_LAYER_UNLOADED,    /**< Layer is not currently loaded in memory */
    TINYAI_LAYER_LOADING,     /**< Layer is in the process of being loaded */
    TINYAI_LAYER_LOADED,      /**< Layer is fully loaded in memory */
    TINYAI_LAYER_UNLOADING,   /**< Layer is in the process of being unloaded */
    TINYAI_LAYER_PREFETCHING, /**< Layer is being prefetched in the background */
    TINYAI_LAYER_ERROR        /**< Error state */
} TinyAILayerState;

/**
 * @brief Priority strategies for loading/unloading layers
 */
typedef enum {
    TINYAI_PRIORITY_LRU,   /**< Least Recently Used */
    TINYAI_PRIORITY_MFU,   /**< Most Frequently Used */
    TINYAI_PRIORITY_FIFO,  /**< First In, First Out */
    TINYAI_PRIORITY_CUSTOM /**< Custom priority using user-defined values */
} TinyAIPriorityStrategy;

/**
 * @brief Usage patterns for predictive layer loading
 */
typedef enum {
    TINYAI_USAGE_UNKNOWN,    /**< Unknown usage pattern */
    TINYAI_USAGE_SEQUENTIAL, /**< Sequential access pattern */
    TINYAI_USAGE_REPEATED,   /**< Repeated access to specific layers */
    TINYAI_USAGE_RANDOM      /**< Random access pattern */
} TinyAIUsagePattern;

/**
 * @brief Memory statistics for progressive loader
 */
typedef struct {
    size_t totalModelSize;     /**< Total size of model in bytes */
    size_t currentMemoryUsage; /**< Current memory usage in bytes */
    size_t peakMemoryUsage;    /**< Peak memory usage in bytes */
    size_t memoryBudget;       /**< Maximum memory budget in bytes */
    int    totalLayerCount;    /**< Total number of layers in model */
    int    loadedLayerCount;   /**< Number of currently loaded layers */
    float  memoryUtilization;  /**< Memory utilization ratio (0.0-1.0) */
    float  averageLoadTime;    /**< Average time to load a layer in ms */
} TinyAIMemoryStats;

/**
 * @brief Progressive loader configuration
 */
typedef struct {
    size_t max_memory_budget;                 /**< Maximum memory budget in bytes */
    bool   enable_layer_unloading;            /**< Whether to unload layers when memory is full */
    TinyAIPriorityStrategy priority_strategy; /**< Strategy to use for unloading */
    float prefetch_threshold;         /**< Memory utilization threshold for prefetching (0.0-1.0) */
    int   max_prefetch_layers;        /**< Maximum number of layers to prefetch */
    bool  enable_compression;         /**< Whether to enable weight compression */
    bool  enable_dependency_tracking; /**< Track dependencies between layers */
    int   cache_alignment;            /**< Memory alignment for cache optimization */
} TinyAIProgressiveLoaderConfig;

/**
 * @brief Progressive loader for model weights
 */
typedef struct TinyAIProgressiveLoader TinyAIProgressiveLoader;

/**
 * @brief Create a default configuration for progressive loader
 *
 * @return Default configuration with sensible settings
 */
TinyAIProgressiveLoaderConfig tinyaiCreateDefaultProgressiveLoaderConfig(void);

/**
 * @brief Create a progressive loader for a model
 *
 * @param model_path Path to the model file
 * @param config Configuration settings
 * @return Pointer to the created loader or NULL on failure
 */
TinyAIProgressiveLoader *tinyaiCreateProgressiveLoader(const char                    *model_path,
                                                       TinyAIProgressiveLoaderConfig *config);

/**
 * @brief Create a progressive loader from an existing memory mapped model
 *
 * @param mapped_model Pointer to an already opened memory mapped model
 * @param config Configuration settings
 * @return Pointer to the created loader or NULL on failure
 */
TinyAIProgressiveLoader *
tinyaiCreateProgressiveLoaderFromMapped(TinyAIMappedModel             *mapped_model,
                                        TinyAIProgressiveLoaderConfig *config);

/**
 * @brief Free a progressive loader and release all resources
 *
 * @param loader Pointer to the loader to free
 */
void tinyaiFreeProgressiveLoader(TinyAIProgressiveLoader *loader);

/**
 * @brief Load a specific layer from the model
 *
 * @param loader Pointer to the progressive loader
 * @param layer_index Index of the layer to load
 * @return true if successful, false on failure
 */
bool tinyaiLoadModelLayer(TinyAIProgressiveLoader *loader, int layer_index);

/**
 * @brief Unload a specific layer from the model
 *
 * @param loader Pointer to the progressive loader
 * @param layer_index Index of the layer to unload
 * @return true if successful, false on failure
 */
bool tinyaiUnloadModelLayer(TinyAIProgressiveLoader *loader, int layer_index);

/**
 * @brief Get a pointer to a layer's weights, loading from disk if necessary
 *
 * @param loader Pointer to the progressive loader
 * @param layer_index Index of the layer to get weights for
 * @return Pointer to the layer weights or NULL on failure
 */
void *tinyaiGetProgressiveLayerWeights(TinyAIProgressiveLoader *loader, int layer_index);

/**
 * @brief Get memory usage statistics for the progressive loader
 *
 * @param loader Pointer to the progressive loader
 * @return Memory statistics structure
 */
TinyAIMemoryStats tinyaiGetProgressiveLoaderMemoryStats(const TinyAIProgressiveLoader *loader);

/**
 * @brief Set memory budget for the progressive loader
 *
 * @param loader Pointer to the progressive loader
 * @param budget_bytes New memory budget in bytes
 * @return true if successful, false on failure
 */
bool tinyaiSetProgressiveLoaderMemoryBudget(TinyAIProgressiveLoader *loader, size_t budget_bytes);

/**
 * @brief Add a dependency relationship between layers
 *
 * @param loader Pointer to the progressive loader
 * @param dependent_layer Index of the dependent layer
 * @param dependency_layer Index of the dependency layer
 * @return true if successful, false on failure
 */
bool tinyaiAddLayerDependency(TinyAIProgressiveLoader *loader, int dependent_layer,
                              int dependency_layer);

/**
 * @brief Check if a layer can be safely unloaded
 *
 * @param loader Pointer to the progressive loader
 * @param layer_index Index of the layer to check
 * @return true if the layer can be unloaded, false otherwise
 */
bool tinyaiCanUnloadLayer(const TinyAIProgressiveLoader *loader, int layer_index);

/**
 * @brief Update access statistics for a layer
 *
 * @param loader Pointer to the progressive loader
 * @param layer_index Index of the layer to update
 */
void tinyaiUpdateLayerAccessStats(TinyAIProgressiveLoader *loader, int layer_index);

/**
 * @brief Get layers to preload based on usage patterns
 *
 * @param loader Pointer to the progressive loader
 * @param current_layer Current layer index
 * @param count Pointer to store the number of layers to preload
 * @return Array of layer indices to preload, must be freed by caller
 */
int *tinyaiGetLayersToPreload(TinyAIProgressiveLoader *loader, int current_layer, int *count);

/**
 * @brief Analyze usage pattern from access history
 *
 * @param loader Pointer to the progressive loader
 * @return Detected usage pattern
 */
TinyAIUsagePattern tinyaiGetUsagePattern(const TinyAIProgressiveLoader *loader);

/**
 * @brief Set custom priority value for a layer
 *
 * @param loader Pointer to the progressive loader
 * @param layer_index Index of the layer to set priority for
 * @param priority Priority value (higher values mean higher priority)
 * @return true if successful, false on failure
 */
bool tinyaiSetLayerCustomPriority(TinyAIProgressiveLoader *loader, int layer_index, float priority);

/**
 * @brief Get the current state of a layer
 *
 * @param loader Pointer to the progressive loader
 * @param layer_index Index of the layer to check
 * @return Current state of the layer
 */
TinyAILayerState tinyaiGetLayerState(const TinyAIProgressiveLoader *loader, int layer_index);

/**
 * @brief Optimize memory allocation across layers based on access patterns
 *
 * @param loader Pointer to the progressive loader
 * @return true if successful, false on failure
 */
bool tinyaiOptimizeLayerMemoryAllocation(TinyAIProgressiveLoader *loader);

/**
 * @brief Preload a specified sequence of layers
 *
 * @param loader Pointer to the progressive loader
 * @param layer_indices Array of layer indices to preload
 * @param count Number of layers in the array
 * @return true if all layers were loaded successfully, false otherwise
 */
bool tinyaiPreloadLayers(TinyAIProgressiveLoader *loader, const int *layer_indices, int count);

/**
 * @brief Clear all loaded layers to free memory
 *
 * @param loader Pointer to the progressive loader
 * @return true if successful, false on failure
 */
bool tinyaiClearAllLayers(TinyAIProgressiveLoader *loader);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_PROGRESSIVE_LOADER_H */