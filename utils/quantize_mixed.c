/**
 * @file quantize_mixed.c
 * @brief Implementation of mixed precision quantization utilities for TinyAI
 */

#include "quantize_mixed.h"
#include "memory_pool.h"
#include "simd_ops.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Helper function to determine number of bits needed for storage */
static size_t getPrecisionBytes(TinyAIPrecisionType precision, size_t numElements)
{
    switch (precision) {
    case TINYAI_PRECISION_FP32:
        return numElements * sizeof(float);
    case TINYAI_PRECISION_FP16:
        return numElements * sizeof(uint16_t); /* FP16 stored as uint16_t */
    case TINYAI_PRECISION_INT8:
        return numElements * sizeof(int8_t);
    case TINYAI_PRECISION_INT4:
        return (numElements + 1) / 2; /* 4-bit packed, 2 values per byte */
    case TINYAI_PRECISION_INT2:
        return (numElements + 3) / 4; /* 2-bit packed, 4 values per byte */
    default:
        return 0;
    }
}

/* Helper function to get minmax for different precisions */
static void getPrecisionMinMax(TinyAIPrecisionType precision, float *min, float *max)
{
    switch (precision) {
    case TINYAI_PRECISION_FP32:
        *min = -FLT_MAX;
        *max = FLT_MAX;
        break;
    case TINYAI_PRECISION_FP16:
        /* IEEE 754 half-precision floating-point format */
        *min = -65504.0f; /* Min positive normal */
        *max = 65504.0f;  /* Max positive normal */
        break;
    case TINYAI_PRECISION_INT8:
        *min = -128.0f;
        *max = 127.0f;
        break;
    case TINYAI_PRECISION_INT4:
        *min = -8.0f;
        *max = 7.0f;
        break;
    case TINYAI_PRECISION_INT2:
        *min = -2.0f;
        *max = 1.0f;
        break;
    default:
        *min = 0.0f;
        *max = 0.0f;
        break;
    }
}

/* Helper function to compute scale and zero point */
static void computeScaleAndZeroPoint(const float *data, int size, float min, float max,
                                     float *scale, float *zeroPoint)
{
    float dataMin = FLT_MAX;
    float dataMax = -FLT_MAX;

    /* Find min/max values in the data */
    for (int i = 0; i < size; i++) {
        if (data[i] < dataMin)
            dataMin = data[i];
        if (data[i] > dataMax)
            dataMax = data[i];
    }

    /* Apply threshold if needed to prevent outliers from skewing the range */
    if (dataMax - dataMin < 1e-6f) {
        /* Avoid division by zero */
        dataMin = -1.0f;
        dataMax = 1.0f;
    }

    /* Compute scale and zero point */
    *scale     = (dataMax - dataMin) / (max - min);
    *zeroPoint = min - dataMin / *scale;
}

/* Helper function to quantize FP32 to different precisions */
static void quantizeToTarget(const float *src, void *dst, int size, TinyAIPrecisionType precision,
                             float scale, float zeroPoint)
{
    float min, max;
    getPrecisionMinMax(precision, &min, &max);

    switch (precision) {
    case TINYAI_PRECISION_FP32:
        /* Direct copy for FP32 */
        memcpy(dst, src, size * sizeof(float));
        break;

    case TINYAI_PRECISION_FP16: {
        /* Quantize to FP16 (simple version, not full IEEE 754 compliant) */
        uint16_t *dstFp16 = (uint16_t *)dst;
        for (int i = 0; i < size; i++) {
            /* Simple conversion to FP16 using truncation */
            float value = src[i];
            /* Clamp to FP16 range */
            if (value > max)
                value = max;
            if (value < min)
                value = min;

            /* Convert to FP16 (simplified) */
            uint16_t sign     = (value < 0) ? 0x8000 : 0;
            float    absValue = fabsf(value);

            /* Handle special cases */
            if (absValue < 6.1e-5f) {
                /* Flush to zero for very small values */
                dstFp16[i] = sign;
            }
            else if (absValue > max) {
                /* Infinity */
                dstFp16[i] = sign | 0x7C00;
            }
            else {
                /* Normal case */
                int   exponent = (int)floor(log2f(absValue));
                float mantissa = absValue / powf(2.0f, exponent) - 1.0f;

                /* Adjust for FP16 bias and range */
                exponent += 15; /* FP16 bias */
                if (exponent < 1)
                    exponent = 1; /* Denormal */
                if (exponent > 30)
                    exponent = 30; /* Max exponent */

                uint16_t expBits = (uint16_t)(exponent << 10);
                uint16_t manBits = (uint16_t)(mantissa * 1024.0f);

                dstFp16[i] = sign | expBits | manBits;
            }
        }
        break;
    }

    case TINYAI_PRECISION_INT8: {
        /* Quantize to INT8 */
        int8_t *dstInt8 = (int8_t *)dst;
        for (int i = 0; i < size; i++) {
            float value = src[i] / scale + zeroPoint;
            /* Clamp to INT8 range */
            if (value > max)
                value = max;
            if (value < min)
                value = min;
            dstInt8[i] = (int8_t)roundf(value);
        }
        break;
    }

    case TINYAI_PRECISION_INT4: {
        /* Quantize to INT4 (packed, 2 values per byte) */
        uint8_t *dstInt4 = (uint8_t *)dst;
        for (int i = 0; i < size; i += 2) {
            float value1 = src[i] / scale + zeroPoint;
            /* Clamp to INT4 range */
            if (value1 > max)
                value1 = max;
            if (value1 < min)
                value1 = min;
            int8_t quantized1 = (int8_t)roundf(value1);

            int8_t quantized2 = 0;
            if (i + 1 < size) {
                float value2 = src[i + 1] / scale + zeroPoint;
                if (value2 > max)
                    value2 = max;
                if (value2 < min)
                    value2 = min;
                quantized2 = (int8_t)roundf(value2);
            }

            /* Pack two 4-bit values into one byte */
            dstInt4[i / 2] = (quantized1 & 0x0F) | ((quantized2 & 0x0F) << 4);
        }
        break;
    }

    case TINYAI_PRECISION_INT2: {
        /* Quantize to INT2 (packed, 4 values per byte) */
        uint8_t *dstInt2 = (uint8_t *)dst;
        for (int i = 0; i < size; i += 4) {
            uint8_t packedByte = 0;

            for (int j = 0; j < 4; j++) {
                if (i + j < size) {
                    float value = src[i + j] / scale + zeroPoint;
                    /* Clamp to INT2 range */
                    if (value > max)
                        value = max;
                    if (value < min)
                        value = min;
                    uint8_t quantized = (uint8_t)((int8_t)roundf(value) & 0x03);
                    packedByte |= (quantized << (j * 2));
                }
            }

            dstInt2[i / 4] = packedByte;
        }
        break;
    }

    default:
        /* Unsupported precision */
        break;
    }
}

/* Helper function to dequantize to FP32 */
static void dequantizeToFloat(const void *src, float *dst, int size, TinyAIPrecisionType precision,
                              float scale, float zeroPoint)
{
    switch (precision) {
    case TINYAI_PRECISION_FP32:
        /* Direct copy for FP32 */
        memcpy(dst, src, size * sizeof(float));
        break;

    case TINYAI_PRECISION_FP16: {
        /* Dequantize from FP16 */
        const uint16_t *srcFp16 = (const uint16_t *)src;
        for (int i = 0; i < size; i++) {
            uint16_t value    = srcFp16[i];
            uint16_t sign     = value & 0x8000;
            uint16_t exp      = (value & 0x7C00) >> 10;
            uint16_t mantissa = value & 0x03FF;

            float result;
            if (exp == 0) {
                /* Zero or denormal */
                if (mantissa == 0) {
                    result = 0.0f;
                }
                else {
                    /* Denormal */
                    result = mantissa / 1024.0f * powf(2.0f, -14);
                }
            }
            else if (exp == 31) {
                /* Infinity or NaN */
                if (mantissa == 0) {
                    result = INFINITY;
                }
                else {
                    result = NAN;
                }
            }
            else {
                /* Normal number */
                result = (1.0f + mantissa / 1024.0f) * powf(2.0f, exp - 15);
            }

            dst[i] = sign ? -result : result;
        }
        break;
    }

    case TINYAI_PRECISION_INT8: {
        /* Dequantize from INT8 */
        const int8_t *srcInt8 = (const int8_t *)src;
        for (int i = 0; i < size; i++) {
            dst[i] = ((float)srcInt8[i] - zeroPoint) * scale;
        }
        break;
    }

    case TINYAI_PRECISION_INT4: {
        /* Dequantize from INT4 (packed, 2 values per byte) */
        const uint8_t *srcInt4 = (const uint8_t *)src;
        for (int i = 0; i < size; i += 2) {
            uint8_t packedByte = srcInt4[i / 2];

            /* Extract first 4-bit value (lower nibble) */
            int8_t value1 = (int8_t)(packedByte & 0x0F);
            /* Sign extend if negative (when highest bit in nibble is set) */
            if (value1 & 0x08)
                value1 |= 0xF0;
            dst[i] = ((float)value1 - zeroPoint) * scale;

            /* Extract second 4-bit value (upper nibble) if within bounds */
            if (i + 1 < size) {
                int8_t value2 = (int8_t)((packedByte >> 4) & 0x0F);
                if (value2 & 0x08)
                    value2 |= 0xF0;
                dst[i + 1] = ((float)value2 - zeroPoint) * scale;
            }
        }
        break;
    }

    case TINYAI_PRECISION_INT2: {
        /* Dequantize from INT2 (packed, 4 values per byte) */
        const uint8_t *srcInt2 = (const uint8_t *)src;
        for (int i = 0; i < size; i += 4) {
            uint8_t packedByte = srcInt2[i / 4];

            for (int j = 0; j < 4; j++) {
                if (i + j < size) {
                    /* Extract 2-bit value */
                    int8_t value = (int8_t)((packedByte >> (j * 2)) & 0x03);
                    /* Sign extend if negative (when highest bit is set) */
                    if (value & 0x02)
                        value |= 0xFC;
                    dst[i + j] = ((float)value - zeroPoint) * scale;
                }
            }
        }
        break;
    }

    default:
        /* Unsupported precision */
        break;
    }
}

/* Public API Implementation */

TinyAIMixedPrecMatrix *tinyaiCreateMixedPrecMatrix(const float *data, int rows, int cols,
                                                   TinyAIPrecisionType precision, float threshold)
{
    if (!data || rows <= 0 || cols <= 0) {
        return NULL;
    }

    /* Allocate matrix structure */
    TinyAIMixedPrecMatrix *matrix = (TinyAIMixedPrecMatrix *)malloc(sizeof(TinyAIMixedPrecMatrix));
    if (!matrix) {
        return NULL;
    }

    /* Initialize matrix properties */
    matrix->rows      = rows;
    matrix->cols      = cols;
    matrix->precision = precision;

    /* Calculate required storage size */
    size_t numElements = (size_t)rows * cols;
    size_t dataSize    = getPrecisionBytes(precision, numElements);
    if (dataSize == 0) {
        free(matrix);
        return NULL;
    }

    /* Allocate data storage */
    matrix->data = malloc(dataSize);
    if (!matrix->data) {
        free(matrix);
        return NULL;
    }
    matrix->dataSize = dataSize;

    /* Compute quantization parameters */
    float min, max;
    getPrecisionMinMax(precision, &min, &max);

    /* Apply threshold if provided */
    if (threshold > 0.0f) {
        /* Find absolute max value for threshold clipping */
        float absMax = 0.0f;
        for (size_t i = 0; i < numElements; i++) {
            float absVal = fabsf(data[i]);
            if (absVal > absMax) {
                absMax = absVal;
            }
        }

        /* Use user-provided threshold instead of absolute max */
        if (absMax > threshold) {
            absMax = threshold;
        }

        /* Create normalized data for quantization */
        float *normalizedData = (float *)malloc(numElements * sizeof(float));
        if (!normalizedData) {
            free(matrix->data);
            free(matrix);
            return NULL;
        }

        /* Clip values to threshold */
        for (size_t i = 0; i < numElements; i++) {
            if (data[i] > absMax) {
                normalizedData[i] = absMax;
            }
            else if (data[i] < -absMax) {
                normalizedData[i] = -absMax;
            }
            else {
                normalizedData[i] = data[i];
            }
        }

        /* Compute quantization parameters and quantize */
        computeScaleAndZeroPoint(normalizedData, numElements, min, max, &matrix->scale,
                                 &matrix->zeroPoint);
        quantizeToTarget(normalizedData, matrix->data, numElements, precision, matrix->scale,
                         matrix->zeroPoint);

        free(normalizedData);
    }
    else {
        /* No threshold, use original data */
        computeScaleAndZeroPoint(data, numElements, min, max, &matrix->scale, &matrix->zeroPoint);
        quantizeToTarget(data, matrix->data, numElements, precision, matrix->scale,
                         matrix->zeroPoint);
    }

    return matrix;
}

void tinyaiFreeMixedPrecMatrix(TinyAIMixedPrecMatrix *matrix)
{
    if (matrix) {
        if (matrix->data) {
            free(matrix->data);
        }
        free(matrix);
    }
}

bool tinyaiMixedPrecToFloat(const TinyAIMixedPrecMatrix *matrix, float *output)
{
    if (!matrix || !output) {
        return false;
    }

    size_t numElements = (size_t)matrix->rows * matrix->cols;
    dequantizeToFloat(matrix->data, output, numElements, matrix->precision, matrix->scale,
                      matrix->zeroPoint);

    return true;
}

bool tinyaiDetermineOptimalPrecision(const char *modelPath, const float *calibrationData,
                                     int calibrationSize, TinyAIMixedPrecConfig *config)
{
    /* Implementation would analyze model layer by layer to determine optimal precision */
    /* This is a complex process typically involving:
     * 1. Layer-wise sensitivity analysis
     * 2. Iterative precision adjustment
     * 3. Accuracy evaluation on calibration dataset
     */

    /* Simplified version for demonstration */
    if (!modelPath || !calibrationData || calibrationSize <= 0 || !config ||
        !config->layerConfigs) {
        return false;
    }

    /* Default to 4-bit for all except first and last layers */
    for (int i = 0; i < config->numLayers; i++) {
        if (i == 0 || i == config->numLayers - 1) {
            /* First and last layers are more sensitive, use 8-bit */
            config->layerConfigs[i].weightPrecision = TINYAI_PRECISION_INT8;
            config->layerConfigs[i].biasPrecision =
                TINYAI_PRECISION_FP32; /* Biases often kept at higher precision */
            config->layerConfigs[i].activPrecision = TINYAI_PRECISION_INT8;
        }
        else {
            /* Internal layers use 4-bit */
            config->layerConfigs[i].weightPrecision = TINYAI_PRECISION_INT4;
            config->layerConfigs[i].biasPrecision   = TINYAI_PRECISION_FP16;
            config->layerConfigs[i].activPrecision  = TINYAI_PRECISION_INT8;
        }

        /* Set default thresholds */
        config->layerConfigs[i].weightThreshold = 0.0f; /* Auto */
        config->layerConfigs[i].biasThreshold   = 0.0f; /* Auto */
        config->layerConfigs[i].activThreshold  = 0.0f; /* Auto */
    }

    return true;
}

bool tinyaiQuantizeModelMixedPrecision(const char *srcModelPath, const char *dstModelPath,
                                       const TinyAIMixedPrecConfig *config)
{
    /* Full implementation would:
     * 1. Load the source model
     * 2. Apply mixed precision quantization layer by layer
     * 3. Save the quantized model to the destination path
     */

    /* Placeholder for a complex model quantization process */
    if (!srcModelPath || !dstModelPath || !config) {
        return false;
    }

    /* Return success for now - full implementation would be model format specific */
    return true;
}

bool tinyaiMixedPrecMatMul(const TinyAIMixedPrecMatrix *a, const TinyAIMixedPrecMatrix *b,
                           TinyAIMixedPrecMatrix *output)
{
    if (!a || !b || !output || a->cols != b->rows || output->rows != a->rows ||
        output->cols != b->cols) {
        return false;
    }

    /* Convert inputs to float for simplicity */
    size_t aElements = (size_t)a->rows * a->cols;
    size_t bElements = (size_t)b->rows * b->cols;
    float *aFloat    = (float *)malloc(aElements * sizeof(float));
    float *bFloat    = (float *)malloc(bElements * sizeof(float));

    if (!aFloat || !bFloat) {
        if (aFloat)
            free(aFloat);
        if (bFloat)
            free(bFloat);
        return false;
    }

    /* Dequantize input matrices */
    dequantizeToFloat(a->data, aFloat, aElements, a->precision, a->scale, a->zeroPoint);
    dequantizeToFloat(b->data, bFloat, bElements, b->precision, b->scale, b->zeroPoint);

    /* Perform matrix multiplication in floating point */
    size_t outputSize  = (size_t)output->rows * output->cols;
    float *resultFloat = (float *)malloc(outputSize * sizeof(float));
    if (!resultFloat) {
        free(aFloat);
        free(bFloat);
        return false;
    }

    /* Simple naive matrix multiply */
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a->cols; k++) {
                sum += aFloat[i * a->cols + k] * bFloat[k * b->cols + j];
            }
            resultFloat[i * output->cols + j] = sum;
        }
    }

    /* Quantize result to output precision */
    float min, max;
    getPrecisionMinMax(output->precision, &min, &max);
    computeScaleAndZeroPoint(resultFloat, outputSize, min, max, &output->scale, &output->zeroPoint);
    quantizeToTarget(resultFloat, output->data, outputSize, output->precision, output->scale,
                     output->zeroPoint);

    /* Clean up */
    free(aFloat);
    free(bFloat);
    free(resultFloat);

    return true;
}

void tinyaiFreeMixedPrecConfig(TinyAIMixedPrecConfig *config)
{
    if (config) {
        if (config->layerConfigs) {
            free(config->layerConfigs);
        }
        if (config->calibrationData) {
            free(config->calibrationData);
        }
        free(config);
    }
}

int tinyaiGetPrecisionBits(TinyAIPrecisionType precision)
{
    switch (precision) {
    case TINYAI_PRECISION_FP32:
        return 32;
    case TINYAI_PRECISION_FP16:
        return 16;
    case TINYAI_PRECISION_INT8:
        return 8;
    case TINYAI_PRECISION_INT4:
        return 4;
    case TINYAI_PRECISION_INT2:
        return 2;
    default:
        return 0;
    }
}

size_t tinyaiGetMixedPrecMatrixMemoryUsage(const TinyAIMixedPrecMatrix *matrix)
{
    if (!matrix) {
        return 0;
    }

    /* Calculate overhead and data size */
    size_t overhead = sizeof(TinyAIMixedPrecMatrix);
    size_t dataSize = matrix->dataSize;

    return overhead + dataSize;
}

TinyAIMixedPrecConfig *tinyaiCreateDefaultMixedPrecConfig(int numLayers)
{
    if (numLayers <= 0) {
        return NULL;
    }

    /* Allocate configuration structure */
    TinyAIMixedPrecConfig *config = (TinyAIMixedPrecConfig *)malloc(sizeof(TinyAIMixedPrecConfig));
    if (!config) {
        return NULL;
    }

    /* Initialize with default values */
    config->numLayers          = numLayers;
    config->perChannelQuantize = true;
    config->useSymmetric       = true;
    config->calibrationSize    = 0;
    config->calibrationData    = NULL;

    /* Allocate layer configurations */
    config->layerConfigs =
        (TinyAILayerQuantConfig *)malloc(numLayers * sizeof(TinyAILayerQuantConfig));
    if (!config->layerConfigs) {
        free(config);
        return NULL;
    }

    /* Initialize each layer with default settings */
    for (int i = 0; i < numLayers; i++) {
        /* Default to 4-bit weights, 16-bit biases, 8-bit activations */
        config->layerConfigs[i].weightPrecision = TINYAI_PRECISION_INT4;
        config->layerConfigs[i].biasPrecision   = TINYAI_PRECISION_FP16;
        config->layerConfigs[i].activPrecision  = TINYAI_PRECISION_INT8;
        config->layerConfigs[i].weightThreshold = 0.0f; /* Auto */
        config->layerConfigs[i].biasThreshold   = 0.0f; /* Auto */
        config->layerConfigs[i].activThreshold  = 0.0f; /* Auto */
    }

    return config;
}
