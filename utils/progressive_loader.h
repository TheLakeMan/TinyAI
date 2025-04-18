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
 * @brief Layer priority levels
 */
typedef enum {
    TINYAI_LAYER_PRIORITY_LOW      = 0,
    TINYAI_LAYER_PRIORITY_MEDIUM   = 1,
    TINYAI_LAYER_PRIORITY_HIGH     = 2,
    TINYAI_LAYER_PRIORITY_CRITICAL = 3
} TinyAILayerPriority;

/**
 * @brief Layer state
 */
typedef enum {
    TINYAI_LAYER_STATE_UNLOADED  = 0,
    TINYAI_LAYER_STATE_LOADING   = 1,
    TINYAI_LAYER_STATE_LOADED    = 2,
    TINYAI_LAYER_STATE_UNLOADING = 3
} TinyAILayerState;

/**
 * @brief Layer information
 */
typedef struct {
    size_t              layer_id;         // Unique identifier for the layer
    size_t              memory_usage;     // Memory required by the layer
    TinyAILayerPriority priority;         // Current priority level
    TinyAILayerState    state;            // Current state
    size_t              access_count;     // Number of times accessed
    uint64_t            last_access_time; // Timestamp of last access
    size_t             *dependencies;     // Array of dependent layer IDs
    size_t              num_dependencies; // Number of dependencies
} TinyAILayerInfo;

/**
 * @brief Progressive loader configuration
 */
typedef struct {
    size_t max_memory;        // Maximum memory budget
    size_t min_memory;        // Minimum memory to keep free
    size_t load_threshold;    // Memory threshold for loading
    size_t unload_threshold;  // Memory threshold for unloading
    size_t priority_window;   // Time window for priority calculation
    bool   enable_prefetch;   // Whether to enable prefetching
    size_t prefetch_distance; // Number of layers to prefetch
} TinyAIProgressiveConfig;

/**
 * @brief Progressive loader context
 */
typedef struct {
    TinyAIProgressiveConfig config;
    TinyAILayerInfo        *layers;
    size_t                  num_layers;
    size_t                  current_memory;
    size_t                  peak_memory;
    uint64_t                start_time;
    bool                    is_initialized;
} TinyAIProgressiveLoader;

/**
 * @brief Create a progressive loader for a model
 *
 * @param model_path Path to the model file
 * @param config Configuration settings
 * @return Pointer to the created loader or NULL on failure
 */
TinyAIProgressiveLoader *tinyaiCreateProgressiveLoader(const char                    *model_path,
                                                       const TinyAIProgressiveConfig *config);

/**
 * @brief Create a progressive loader from an existing memory mapped model
 *
 * @param mapped_model Pointer to an already opened memory mapped model
 * @param config Configuration settings
 * @return Pointer to the created loader or NULL on failure
 */
TinyAIProgressiveLoader *
tinyaiCreateProgressiveLoaderFromMapped(TinyAIMappedModel             *mapped_model,
                                        const TinyAIProgressiveConfig *config);

/**
 * @brief Free a progressive loader and release all resources
 *
 * @param loader Pointer to the loader to free
 */
void tinyaiFreeProgressiveLoader(TinyAIProgressiveLoader *loader);

/**
 * @brief Initialize layer information
 *
 * @param loader Pointer to the progressive loader
 * @param layer_id Unique identifier for the layer
 * @param memory_usage Memory required by the layer
 * @param priority Current priority level
 * @param dependencies Array of dependent layer IDs
 * @param num_dependencies Number of dependencies
 * @return true if successful, false on failure
 */
bool tinyaiInitLayerInfo(TinyAIProgressiveLoader *loader, size_t layer_id, size_t memory_usage,
                         TinyAILayerPriority priority, const size_t *dependencies,
                         size_t num_dependencies);

/**
 * @brief Request layer loading
 *
 * @param loader Pointer to the progressive loader
 * @param layer_id Unique identifier for the layer
 * @return true if successful, false on failure
 */
bool tinyaiRequestLayer(TinyAIProgressiveLoader *loader, size_t layer_id);

/**
 * @brief Unload layer
 *
 * @param loader Pointer to the progressive loader
 * @param layer_id Unique identifier for the layer
 * @return true if successful, false on failure
 */
bool tinyaiUnloadLayer(TinyAIProgressiveLoader *loader, size_t layer_id);

/**
 * @brief Get layer state
 *
 * @param loader Pointer to the progressive loader
 * @param layer_id Unique identifier for the layer
 * @return Current state of the layer
 */
TinyAILayerState tinyaiGetLayerState(const TinyAIProgressiveLoader *loader, size_t layer_id);

/**
 * @brief Update layer priority
 *
 * @param loader Pointer to the progressive loader
 * @param layer_id Unique identifier for the layer
 * @param priority New priority level
 * @return true if successful, false on failure
 */
bool tinyaiUpdateLayerPriority(TinyAIProgressiveLoader *loader, size_t layer_id,
                               TinyAILayerPriority priority);

/**
 * @brief Get memory usage
 *
 * @param loader Pointer to the progressive loader
 * @return Current memory usage in bytes
 */
size_t tinyaiGetMemoryUsage(const TinyAIProgressiveLoader *loader);

/**
 * @brief Get peak memory usage
 *
 * @param loader Pointer to the progressive loader
 * @return Peak memory usage in bytes
 */
size_t tinyaiGetPeakMemoryUsage(const TinyAIProgressiveLoader *loader);

/**
 * @brief Check if layer can be loaded
 *
 * @param loader Pointer to the progressive loader
 * @param layer_id Unique identifier for the layer
 * @return true if the layer can be loaded, false otherwise
 */
bool tinyaiCanLoadLayer(const TinyAIProgressiveLoader *loader, size_t layer_id);

/**
 * @brief Get layer dependencies
 *
 * @param loader Pointer to the progressive loader
 * @param layer_id Unique identifier for the layer
 * @param num_dependencies Pointer to store the number of dependencies
 * @return Array of dependent layer IDs
 */
const size_t *tinyaiGetLayerDependencies(const TinyAIProgressiveLoader *loader, size_t layer_id,
                                         size_t *num_dependencies);

/**
 * @brief Update layer access
 *
 * @param loader Pointer to the progressive loader
 * @param layer_id Unique identifier for the layer
 */
void tinyaiUpdateLayerAccess(TinyAIProgressiveLoader *loader, size_t layer_id);

/**
 * @brief Reset loader state
 *
 * @param loader Pointer to the progressive loader
 */
void tinyaiResetProgressiveLoader(TinyAIProgressiveLoader *loader);

/**
 * @brief Enable/disable prefetching
 *
 * @param loader Pointer to the progressive loader
 * @param enable true to enable prefetching, false to disable
 */
void tinyaiEnablePrefetching(TinyAIProgressiveLoader *loader, bool enable);

/**
 * @brief Set prefetch distance
 *
 * @param loader Pointer to the progressive loader
 * @param distance Number of layers to prefetch
 */
void tinyaiSetPrefetchDistance(TinyAIProgressiveLoader *loader, size_t distance);

/**
 * @brief Get loader configuration
 *
 * @param loader Pointer to the progressive loader
 * @return Configuration settings
 */
const TinyAIProgressiveConfig *tinyaiGetLoaderConfig(const TinyAIProgressiveLoader *loader);

/**
 * @brief Set loader configuration
 *
 * @param loader Pointer to the progressive loader
 * @param config New configuration settings
 * @return true if successful, false on failure
 */
bool tinyaiSetLoaderConfig(TinyAIProgressiveLoader *loader, const TinyAIProgressiveConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_PROGRESSIVE_LOADER_H */