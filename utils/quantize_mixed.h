/**
 * @file quantize_mixed.h
 * @brief Mixed precision quantization utilities for TinyAI
 *
 * This header provides utilities for mixed precision quantization,
 * allowing different parts of a model to use different bit-widths
 * based on their sensitivity and importance.
 */

#ifndef TINYAI_QUANTIZE_MIXED_H
#define TINYAI_QUANTIZE_MIXED_H

#include "quantize.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Supported quantization precisions
 */
typedef enum {
    TINYAI_PRECISION_FP32, /* Full precision floating point (32-bit) */
    TINYAI_PRECISION_FP16, /* Half precision floating point (16-bit) */
    TINYAI_PRECISION_INT8, /* 8-bit integer quantization */
    TINYAI_PRECISION_INT4, /* 4-bit integer quantization */
    TINYAI_PRECISION_INT2  /* 2-bit integer quantization */
} TinyAIPrecisionType;

/**
 * Mixed precision quantization configuration for a layer
 */
typedef struct {
    TinyAIPrecisionType weightPrecision; /* Precision for weights */
    TinyAIPrecisionType biasPrecision;   /* Precision for biases */
    TinyAIPrecisionType activPrecision;  /* Precision for activations */
    float               weightThreshold; /* Per-layer weight clipping threshold */
    float               biasThreshold;   /* Per-layer bias clipping threshold */
    float               activThreshold;  /* Per-layer activation clipping threshold */
} TinyAILayerQuantConfig;

/**
 * Matrix with mixed precision elements
 */
typedef struct {
    void               *data;      /* Pointer to matrix data */
    size_t              dataSize;  /* Size of data in bytes */
    int                 rows;      /* Number of rows */
    int                 cols;      /* Number of columns */
    TinyAIPrecisionType precision; /* Precision of the data */
    float               scale;     /* Scale factor for quantization */
    float               zeroPoint; /* Zero point for quantization */
} TinyAIMixedPrecMatrix;

/**
 * Mixed precision model configuration
 */
typedef struct {
    int                     numLayers;          /* Number of layers in the model */
    TinyAILayerQuantConfig *layerConfigs;       /* Per-layer quantization configs */
    bool                    perChannelQuantize; /* Whether to use per-channel quantization */
    bool                    useSymmetric;       /* Whether to use symmetric quantization */
    int                     calibrationSize;    /* Calibration dataset size */
    float                  *calibrationData;    /* Representative data for calibration */
} TinyAIMixedPrecConfig;

/**
 * Create a mixed precision matrix from floating point data
 *
 * @param data Source floating point data
 * @param rows Number of rows
 * @param cols Number of columns
 * @param precision Target precision for the matrix
 * @param threshold Clipping threshold for quantization (0.0 for auto)
 * @return Quantized mixed precision matrix (NULL on failure)
 */
TinyAIMixedPrecMatrix *tinyaiCreateMixedPrecMatrix(const float *data, int rows, int cols,
                                                   TinyAIPrecisionType precision, float threshold);

/**
 * Free a mixed precision matrix
 *
 * @param matrix Matrix to free
 */
void tinyaiFreeMixedPrecMatrix(TinyAIMixedPrecMatrix *matrix);

/**
 * Convert a mixed precision matrix to floating point
 *
 * @param matrix Mixed precision matrix to convert
 * @param output Output floating point array (must be pre-allocated)
 * @return true on success, false on failure
 */
bool tinyaiMixedPrecToFloat(const TinyAIMixedPrecMatrix *matrix, float *output);

/**
 * Determine optimal precision for each layer using sensitivity analysis
 *
 * @param modelPath Path to original model file
 * @param calibrationData Representative input data for calibration
 * @param calibrationSize Number of calibration samples
 * @param config Output quantization configuration
 * @return true on success, false on failure
 */
bool tinyaiDetermineOptimalPrecision(const char *modelPath, const float *calibrationData,
                                     int calibrationSize, TinyAIMixedPrecConfig *config);

/**
 * Apply mixed precision quantization to a model
 *
 * @param srcModelPath Path to source model file
 * @param dstModelPath Path to save quantized model
 * @param config Mixed precision configuration
 * @return true on success, false on failure
 */
bool tinyaiQuantizeModelMixedPrecision(const char *srcModelPath, const char *dstModelPath,
                                       const TinyAIMixedPrecConfig *config);

/**
 * Matrix multiplication with mixed precision matrices
 *
 * @param a First matrix
 * @param b Second matrix
 * @param output Output matrix (must be pre-allocated)
 * @return true on success, false on failure
 */
bool tinyaiMixedPrecMatMul(const TinyAIMixedPrecMatrix *a, const TinyAIMixedPrecMatrix *b,
                           TinyAIMixedPrecMatrix *output);

/**
 * Free a mixed precision model configuration
 *
 * @param config Configuration to free
 */
void tinyaiFreeMixedPrecConfig(TinyAIMixedPrecConfig *config);

/**
 * Get the size in bits for a given precision type
 *
 * @param precision Precision type
 * @return Size in bits (0 if invalid)
 */
int tinyaiGetPrecisionBits(TinyAIPrecisionType precision);

/**
 * Calculate memory usage for a mixed precision matrix
 *
 * @param matrix Mixed precision matrix
 * @return Memory usage in bytes
 */
size_t tinyaiGetMixedPrecMatrixMemoryUsage(const TinyAIMixedPrecMatrix *matrix);

/**
 * Create a default mixed precision configuration
 *
 * @param numLayers Number of layers in the model
 * @return Default configuration (NULL on failure)
 */
TinyAIMixedPrecConfig *tinyaiCreateDefaultMixedPrecConfig(int numLayers);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_QUANTIZE_MIXED_H */
