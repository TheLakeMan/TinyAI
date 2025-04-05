/**
 * @file prune.h
 * @brief Model pruning utilities for TinyAI
 *
 * This header provides utilities for pruning neural network models
 * to reduce their size and computational requirements.
 */

#ifndef TINYAI_PRUNE_H
#define TINYAI_PRUNE_H

#include "quantize.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Pruning method to use
 */
typedef enum {
    TINYAI_PRUNE_MAGNITUDE,  /* Prune by absolute magnitude (remove smallest weights) */
    TINYAI_PRUNE_THRESHOLD,  /* Prune by threshold (remove weights below threshold) */
    TINYAI_PRUNE_STRUCTURED, /* Structured pruning (remove entire filters/channels) */
    TINYAI_PRUNE_RANDOM      /* Random pruning (for baseline comparison) */
} TinyAIPruneMethod;

/**
 * Configuration for pruning a model
 */
typedef struct {
    TinyAIPruneMethod method;              /* Pruning method to use */
    float             pruneRate;           /* Target sparsity (0.0-1.0) */
    float             threshold;           /* Threshold for THRESHOLD method */
    bool              layerWisePruning;    /* Whether to prune each layer separately */
    bool              retrainAfterPrune;   /* Whether to do fine-tuning after pruning */
    int               numRetrainSteps;     /* Number of retraining steps after pruning */
    float             retrainLearningRate; /* Learning rate for retraining */
} TinyAIPruneConfig;

/**
 * Prune a model with the given configuration
 *
 * @param srcModelPath Path to source model file
 * @param dstModelPath Path to save pruned model
 * @param config Pruning configuration
 * @return true on success, false on failure
 */
bool tinyaiPruneModel(const char *srcModelPath, const char *dstModelPath,
                      const TinyAIPruneConfig *config);

/**
 * Apply magnitude-based pruning to a weight matrix
 *
 * @param weights Pointer to weight matrix (will be modified in-place)
 * @param rows Number of rows in weight matrix
 * @param cols Number of columns in weight matrix
 * @param pruneRate Target sparsity (0.0-1.0)
 * @return true on success, false on failure
 */
bool tinyaiPruneMatrixByMagnitude(float *weights, int rows, int cols, float pruneRate);

/**
 * Apply threshold-based pruning to a weight matrix
 *
 * @param weights Pointer to weight matrix (will be modified in-place)
 * @param rows Number of rows in weight matrix
 * @param cols Number of columns in weight matrix
 * @param threshold Threshold value (weights with magnitude below this are pruned)
 * @return true on success, false on failure
 */
bool tinyaiPruneMatrixByThreshold(float *weights, int rows, int cols, float threshold);

/**
 * Apply structured pruning to a weight matrix (remove entire filters/channels)
 *
 * @param weights Pointer to weight matrix (will be modified in-place)
 * @param rows Number of rows in weight matrix
 * @param cols Number of columns in weight matrix
 * @param pruneRate Target sparsity (0.0-1.0)
 * @param isConvFilter Whether the weights are from a convolution filter
 * @param filterShape Array containing filter dimensions [depth, height, width, filters]
 * @return true on success, false on failure
 */
bool tinyaiPruneMatrixStructured(float *weights, int rows, int cols, float pruneRate,
                                 bool isConvFilter, const int *filterShape);

/**
 * Apply random pruning to a weight matrix
 *
 * @param weights Pointer to weight matrix (will be modified in-place)
 * @param rows Number of rows in weight matrix
 * @param cols Number of columns in weight matrix
 * @param pruneRate Target sparsity (0.0-1.0)
 * @return true on success, false on failure
 */
bool tinyaiPruneMatrixRandom(float *weights, int rows, int cols, float pruneRate);

/**
 * Calculate the sparsity of a weight matrix
 *
 * @param weights Pointer to weight matrix
 * @param rows Number of rows in weight matrix
 * @param cols Number of columns in weight matrix
 * @param threshold Threshold for considering a weight as zero
 * @return Sparsity as a fraction (0.0-1.0)
 */
float tinyaiCalculateSparsity(const float *weights, int rows, int cols, float threshold);

/**
 * Create a default pruning configuration
 *
 * @return Default configuration (must be freed with free())
 */
TinyAIPruneConfig *tinyaiCreateDefaultPruneConfig(void);

/**
 * Estimate memory savings from pruning
 *
 * @param modelPath Path to model file
 * @param pruneRate Target sparsity (0.0-1.0)
 * @param useCSR Whether to use CSR format for sparse matrices
 * @param originalSize Output parameter for original model size in bytes
 * @param prunedSize Output parameter for pruned model size in bytes
 * @return true on success, false on failure
 */
bool tinyaiEstimatePrunedSize(const char *modelPath, float pruneRate, bool useCSR,
                              size_t *originalSize, size_t *prunedSize);

/**
 * Apply fine-tuning after pruning
 *
 * @param modelPath Path to pruned model file
 * @param dataPath Path to calibration data
 * @param config Pruning configuration with retraining parameters
 * @return true on success, false on failure
 */
bool tinyaiRetrainPrunedModel(const char *modelPath, const char *dataPath,
                              const TinyAIPruneConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_PRUNE_H */
