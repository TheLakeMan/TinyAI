/**
 * @file sparse_ops.c
 * @brief Implementation of sparse matrix operations for TinyAI
 */

#include "sparse_ops.h"
#include "memory.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Check if SIMD is available */
#if defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h>
#define TINYAI_SIMD_AVX
#elif defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__) || defined(__SSSE3__) ||          \
    defined(__SSE4_1__) || defined(__SSE4_2__)
#include <emmintrin.h>
#include <xmmintrin.h>
#define TINYAI_SIMD_SSE
#endif

/**
 * Create a CSR matrix from dense matrix data with a sparsity threshold
 */
TinyAICSRMatrix *tinyaiCreateCSRMatrixFromDense(const float *dense, int32_t rows, int32_t cols,
                                                float threshold)
{
    if (!dense || rows <= 0 || cols <= 0) {
        return NULL;
    }

    /* First pass: count non-zero elements */
    int32_t nnz = 0;
    for (int32_t i = 0; i < rows; i++) {
        for (int32_t j = 0; j < cols; j++) {
            if (fabsf(dense[i * cols + j]) >= threshold) {
                nnz++;
            }
        }
    }

    /* Allocate CSR matrix */
    TinyAICSRMatrix *csr = (TinyAICSRMatrix *)malloc(sizeof(TinyAICSRMatrix));
    if (!csr) {
        return NULL;
    }

    csr->rows = rows;
    csr->cols = cols;
    csr->nnz  = nnz;

    /* Allocate arrays */
    csr->values     = (float *)malloc(nnz * sizeof(float));
    csr->colIndices = (int32_t *)malloc(nnz * sizeof(int32_t));
    csr->rowPtrs    = (int32_t *)malloc((rows + 1) * sizeof(int32_t));

    if (!csr->values || !csr->colIndices || !csr->rowPtrs) {
        tinyaiCSRMatrixFree(csr);
        return NULL;
    }

    /* Second pass: fill in CSR matrix */
    csr->rowPtrs[0] = 0;
    int32_t idx     = 0;

    for (int32_t i = 0; i < rows; i++) {
        for (int32_t j = 0; j < cols; j++) {
            float val = dense[i * cols + j];
            if (fabsf(val) >= threshold) {
                csr->values[idx]     = val;
                csr->colIndices[idx] = j;
                idx++;
            }
        }
        csr->rowPtrs[i + 1] = idx;
    }

    return csr;
}

/**
 * Create a 4-bit quantized CSR matrix from dense matrix data with sparsity threshold
 */
TinyAICSRMatrix4Bit *tinyaiCreateCSRMatrix4BitFromDense(const float *dense, int32_t rows,
                                                        int32_t cols, float threshold)
{
    if (!dense || rows <= 0 || cols <= 0) {
        return NULL;
    }

    /* First pass: count non-zero elements and find min/max values */
    int32_t nnz    = 0;
    float   minVal = FLT_MAX;
    float   maxVal = -FLT_MAX;

    for (int32_t i = 0; i < rows; i++) {
        for (int32_t j = 0; j < cols; j++) {
            float val = dense[i * cols + j];
            if (fabsf(val) >= threshold) {
                nnz++;
                if (val < minVal)
                    minVal = val;
                if (val > maxVal)
                    maxVal = val;
            }
        }
    }

    /* Allocate CSR matrix */
    TinyAICSRMatrix4Bit *csr = (TinyAICSRMatrix4Bit *)malloc(sizeof(TinyAICSRMatrix4Bit));
    if (!csr) {
        return NULL;
    }

    csr->rows = rows;
    csr->cols = cols;
    csr->nnz  = nnz;

    /* Calculate quantization parameters */
    float range    = maxVal - minVal;
    csr->scale     = range / 15.0f; /* 4-bit range is 0-15 */
    csr->zeroPoint = minVal;

    /* Allocate arrays */
    /* Each qvalue takes 4 bits, so we need nnz/2 bytes (rounded up) */
    size_t qvaluesSize = (nnz + 1) / 2;
    csr->qvalues       = (uint8_t *)malloc(qvaluesSize);
    csr->colIndices    = (int32_t *)malloc(nnz * sizeof(int32_t));
    csr->rowPtrs       = (int32_t *)malloc((rows + 1) * sizeof(int32_t));

    if (!csr->qvalues || !csr->colIndices || !csr->rowPtrs) {
        tinyaiCSRMatrix4BitFree(csr);
        return NULL;
    }

    /* Initialize qvalues to zero */
    memset(csr->qvalues, 0, qvaluesSize);

    /* Second pass: fill in CSR matrix */
    csr->rowPtrs[0] = 0;
    int32_t idx     = 0;

    for (int32_t i = 0; i < rows; i++) {
        for (int32_t j = 0; j < cols; j++) {
            float val = dense[i * cols + j];
            if (fabsf(val) >= threshold) {
                /* Quantize to 4 bits */
                float   normalized = (val - csr->zeroPoint) / csr->scale;
                uint8_t qval       = (uint8_t)roundf(fmaxf(0.0f, fminf(normalized, 15.0f)));

                /* Pack two 4-bit values per byte */
                if (idx % 2 == 0) {
                    csr->qvalues[idx / 2] = qval;
                }
                else {
                    csr->qvalues[idx / 2] |= (qval << 4);
                }

                csr->colIndices[idx] = j;
                idx++;
            }
        }
        csr->rowPtrs[i + 1] = idx;
    }

    return csr;
}

/**
 * Convert a CSR matrix to dense format
 */
bool tinyaiCSRMatrixToDense(const TinyAICSRMatrix *csr, float *dense)
{
    if (!csr || !dense) {
        return false;
    }

    /* Initialize dense matrix to zeros */
    memset(dense, 0, csr->rows * csr->cols * sizeof(float));

    /* Fill in non-zero values */
    for (int32_t i = 0; i < csr->rows; i++) {
        int32_t rowStart = csr->rowPtrs[i];
        int32_t rowEnd   = csr->rowPtrs[i + 1];

        for (int32_t j = rowStart; j < rowEnd; j++) {
            int32_t col                = csr->colIndices[j];
            dense[i * csr->cols + col] = csr->values[j];
        }
    }

    return true;
}

/**
 * Convert a 4-bit quantized CSR matrix to dense format
 */
bool tinyaiCSRMatrix4BitToDense(const TinyAICSRMatrix4Bit *csr, float *dense)
{
    if (!csr || !dense) {
        return false;
    }

    /* Initialize dense matrix to zeros */
    memset(dense, 0, csr->rows * csr->cols * sizeof(float));

    /* Fill in non-zero values */
    for (int32_t i = 0; i < csr->rows; i++) {
        int32_t rowStart = csr->rowPtrs[i];
        int32_t rowEnd   = csr->rowPtrs[i + 1];

        for (int32_t j = rowStart; j < rowEnd; j++) {
            /* Extract 4-bit value */
            uint8_t qval;
            if (j % 2 == 0) {
                qval = csr->qvalues[j / 2] & 0x0F;
            }
            else {
                qval = (csr->qvalues[j / 2] >> 4) & 0x0F;
            }

            /* Dequantize */
            float val = qval * csr->scale + csr->zeroPoint;

            int32_t col                = csr->colIndices[j];
            dense[i * csr->cols + col] = val;
        }
    }

    return true;
}

/**
 * Free memory used by a CSR matrix
 */
void tinyaiCSRMatrixFree(TinyAICSRMatrix *csr)
{
    if (!csr) {
        return;
    }

    if (csr->values) {
        free(csr->values);
    }

    if (csr->colIndices) {
        free(csr->colIndices);
    }

    if (csr->rowPtrs) {
        free(csr->rowPtrs);
    }

    free(csr);
}

/**
 * Free memory used by a 4-bit quantized CSR matrix
 */
void tinyaiCSRMatrix4BitFree(TinyAICSRMatrix4Bit *csr)
{
    if (!csr) {
        return;
    }

    if (csr->qvalues) {
        free(csr->qvalues);
    }

    if (csr->colIndices) {
        free(csr->colIndices);
    }

    if (csr->rowPtrs) {
        free(csr->rowPtrs);
    }

    free(csr);
}

/**
 * Perform sparse matrix-vector multiplication: y = A * x
 */
bool tinyaiCSRMatrixVectorMul(const TinyAICSRMatrix *csr, const float *x, float *y)
{
    if (!csr || !x || !y) {
        return false;
    }

    /* Initialize output vector to zero */
    memset(y, 0, csr->rows * sizeof(float));

    /* Perform CSR matrix-vector multiplication */
    for (int32_t i = 0; i < csr->rows; i++) {
        int32_t rowStart = csr->rowPtrs[i];
        int32_t rowEnd   = csr->rowPtrs[i + 1];

        for (int32_t j = rowStart; j < rowEnd; j++) {
            int32_t col = csr->colIndices[j];
            y[i] += csr->values[j] * x[col];
        }
    }

    return true;
}

/**
 * Perform 4-bit quantized sparse matrix-vector multiplication: y = A * x
 */
bool tinyaiCSRMatrix4BitVectorMul(const TinyAICSRMatrix4Bit *csr, const float *x, float *y)
{
    if (!csr || !x || !y) {
        return false;
    }

    /* Initialize output vector to zero */
    memset(y, 0, csr->rows * sizeof(float));

    /* Perform CSR matrix-vector multiplication */
    for (int32_t i = 0; i < csr->rows; i++) {
        int32_t rowStart = csr->rowPtrs[i];
        int32_t rowEnd   = csr->rowPtrs[i + 1];

        for (int32_t j = rowStart; j < rowEnd; j++) {
            /* Extract 4-bit value */
            uint8_t qval;
            if (j % 2 == 0) {
                qval = csr->qvalues[j / 2] & 0x0F;
            }
            else {
                qval = (csr->qvalues[j / 2] >> 4) & 0x0F;
            }

            /* Dequantize */
            float val = qval * csr->scale + csr->zeroPoint;

            int32_t col = csr->colIndices[j];
            y[i] += val * x[col];
        }
    }

    return true;
}

#ifdef TINYAI_SIMD_AVX
/**
 * Perform SIMD-accelerated sparse matrix-vector multiplication: y = A * x (AVX version)
 */
bool tinyaiCSRMatrixVectorMulSIMD(const TinyAICSRMatrix *csr, const float *x, float *y)
{
    if (!csr || !x || !y) {
        return false;
    }

    /* Initialize output vector to zero */
    memset(y, 0, csr->rows * sizeof(float));

    /* Perform CSR matrix-vector multiplication with AVX */
    for (int32_t i = 0; i < csr->rows; i++) {
        int32_t rowStart = csr->rowPtrs[i];
        int32_t rowEnd   = csr->rowPtrs[i + 1];

        /* Process 8 elements at a time using AVX */
        int32_t j   = rowStart;
        __m256  sum = _mm256_setzero_ps();

        /* Process 8 elements at a time */
        for (; j + 7 < rowEnd; j += 8) {
            /* Load values and indices */
            __m256 values = _mm256_loadu_ps(&csr->values[j]);

            /* Gather x values based on column indices */
            int32_t col0 = csr->colIndices[j];
            int32_t col1 = csr->colIndices[j + 1];
            int32_t col2 = csr->colIndices[j + 2];
            int32_t col3 = csr->colIndices[j + 3];
            int32_t col4 = csr->colIndices[j + 4];
            int32_t col5 = csr->colIndices[j + 5];
            int32_t col6 = csr->colIndices[j + 6];
            int32_t col7 = csr->colIndices[j + 7];

            __m256 xvals = _mm256_set_ps(x[col7], x[col6], x[col5], x[col4], x[col3], x[col2],
                                         x[col1], x[col0]);

            /* Multiply and accumulate */
            sum = _mm256_add_ps(sum, _mm256_mul_ps(values, xvals));
        }

        /* Reduce sum vector to single value */
        float temp[8];
        _mm256_storeu_ps(temp, sum);
        float rowSum = 0.0f;
        for (int k = 0; k < 8; k++) {
            rowSum += temp[k];
        }

        /* Process remaining elements */
        for (; j < rowEnd; j++) {
            int32_t col = csr->colIndices[j];
            rowSum += csr->values[j] * x[col];
        }

        y[i] = rowSum;
    }

    return true;
}

/**
 * Perform SIMD-accelerated 4-bit quantized sparse matrix-vector multiplication: y = A * x (AVX
 * version)
 */
bool tinyaiCSRMatrix4BitVectorMulSIMD(const TinyAICSRMatrix4Bit *csr, const float *x, float *y)
{
    if (!csr || !x || !y) {
        return false;
    }

    /* Initialize output vector to zero */
    memset(y, 0, csr->rows * sizeof(float));

    /* Create constants for dequantization */
    __m256 vscale = _mm256_set1_ps(csr->scale);
    __m256 vzero  = _mm256_set1_ps(csr->zeroPoint);

    /* Perform CSR matrix-vector multiplication with AVX */
    for (int32_t i = 0; i < csr->rows; i++) {
        int32_t rowStart = csr->rowPtrs[i];
        int32_t rowEnd   = csr->rowPtrs[i + 1];

        /* Process in chunks of 8 (for AVX) */
        int32_t j   = rowStart;
        __m256  sum = _mm256_setzero_ps();

        /* Ensure we process aligned chunks of 16 elements (8 bytes in 4-bit format) */
        int32_t alignedEnd = rowStart + ((rowEnd - rowStart) / 16) * 16;

        for (; j < alignedEnd; j += 16) {
            /* Load 8 bytes (16 quantized values) */
            uint8_t qvals[8];
            for (int k = 0; k < 8; k++) {
                qvals[k] = csr->qvalues[(j + k * 2) / 2];
            }

            /* First 8 values (lower 4 bits of each byte) */
            int32_t cols1[8];
            float   vals1[8];

            for (int k = 0; k < 8; k++) {
                /* Extract and dequantize values */
                uint8_t qval = qvals[k] & 0x0F;
                vals1[k]     = qval * csr->scale + csr->zeroPoint;

                /* Get column indices */
                cols1[k] = csr->colIndices[j + k * 2];
            }

            /* Load x values for first 8 */
            __m256 xvals1 = _mm256_set_ps(x[cols1[7]], x[cols1[6]], x[cols1[5]], x[cols1[4]],
                                          x[cols1[3]], x[cols1[2]], x[cols1[1]], x[cols1[0]]);

            /* Load dequantized values for first 8 */
            __m256 vvals1 = _mm256_loadu_ps(vals1);

            /* Multiply and accumulate first 8 */
            sum = _mm256_add_ps(sum, _mm256_mul_ps(vvals1, xvals1));

            /* Second 8 values (upper 4 bits of each byte) */
            int32_t cols2[8];
            float   vals2[8];

            for (int k = 0; k < 8; k++) {
                /* Extract and dequantize values */
                uint8_t qval = (qvals[k] >> 4) & 0x0F;
                vals2[k]     = qval * csr->scale + csr->zeroPoint;

                /* Get column indices */
                cols2[k] = csr->colIndices[j + k * 2 + 1];
            }

            /* Load x values for second 8 */
            __m256 xvals2 = _mm256_set_ps(x[cols2[7]], x[cols2[6]], x[cols2[5]], x[cols2[4]],
                                          x[cols2[3]], x[cols2[2]], x[cols2[1]], x[cols2[0]]);

            /* Load dequantized values for second 8 */
            __m256 vvals2 = _mm256_loadu_ps(vals2);

            /* Multiply and accumulate second 8 */
            sum = _mm256_add_ps(sum, _mm256_mul_ps(vvals2, xvals2));
        }

        /* Reduce sum vector to single value */
        float temp[8];
        _mm256_storeu_ps(temp, sum);
        float rowSum = 0.0f;
        for (int k = 0; k < 8; k++) {
            rowSum += temp[k];
        }

        /* Process remaining elements */
        for (; j < rowEnd; j++) {
            /* Extract 4-bit value */
            uint8_t qval;
            if (j % 2 == 0) {
                qval = csr->qvalues[j / 2] & 0x0F;
            }
            else {
                qval = (csr->qvalues[j / 2] >> 4) & 0x0F;
            }

            /* Dequantize */
            float val = qval * csr->scale + csr->zeroPoint;

            int32_t col = csr->colIndices[j];
            rowSum += val * x[col];
        }

        y[i] = rowSum;
    }

    return true;
}

#elif defined(TINYAI_SIMD_SSE)
/**
 * Perform SIMD-accelerated sparse matrix-vector multiplication: y = A * x (SSE version)
 */
bool tinyaiCSRMatrixVectorMulSIMD(const TinyAICSRMatrix *csr, const float *x, float *y)
{
    if (!csr || !x || !y) {
        return false;
    }

    /* Initialize output vector to zero */
    memset(y, 0, csr->rows * sizeof(float));

    /* Perform CSR matrix-vector multiplication with SSE */
    for (int32_t i = 0; i < csr->rows; i++) {
        int32_t rowStart = csr->rowPtrs[i];
        int32_t rowEnd   = csr->rowPtrs[i + 1];

        /* Process 4 elements at a time using SSE */
        int32_t j   = rowStart;
        __m128  sum = _mm_setzero_ps();

        /* Process 4 elements at a time */
        for (; j + 3 < rowEnd; j += 4) {
            /* Load values and indices */
            __m128 values = _mm_loadu_ps(&csr->values[j]);

            /* Gather x values based on column indices */
            int32_t col0 = csr->colIndices[j];
            int32_t col1 = csr->colIndices[j + 1];
            int32_t col2 = csr->colIndices[j + 2];
            int32_t col3 = csr->colIndices[j + 3];

            __m128 xvals = _mm_set_ps(x[col3], x[col2], x[col1], x[col0]);

            /* Multiply and accumulate */
            sum = _mm_add_ps(sum, _mm_mul_ps(values, xvals));
        }

        /* Reduce sum vector to single value */
        float temp[4];
        _mm_storeu_ps(temp, sum);
        float rowSum = temp[0] + temp[1] + temp[2] + temp[3];

        /* Process remaining elements */
        for (; j < rowEnd; j++) {
            int32_t col = csr->colIndices[j];
            rowSum += csr->values[j] * x[col];
        }

        y[i] = rowSum;
    }

    return true;
}

/**
 * Perform SIMD-accelerated 4-bit quantized sparse matrix-vector multiplication: y = A * x (SSE
 * version)
 */
bool tinyaiCSRMatrix4BitVectorMulSIMD(const TinyAICSRMatrix4Bit *csr, const float *x, float *y)
{
    if (!csr || !x || !y) {
        return false;
    }

    /* Initialize output vector to zero */
    memset(y, 0, csr->rows * sizeof(float));

    /* Create constants for dequantization */
    __m128 vscale = _mm_set1_ps(csr->scale);
    __m128 vzero  = _mm_set1_ps(csr->zeroPoint);

    /* Perform CSR matrix-vector multiplication with SSE */
    for (int32_t i = 0; i < csr->rows; i++) {
        int32_t rowStart = csr->rowPtrs[i];
        int32_t rowEnd   = csr->rowPtrs[i + 1];

        /* Process in chunks of 4 (for SSE) */
        int32_t j   = rowStart;
        __m128  sum = _mm_setzero_ps();

        /* Ensure we process aligned chunks of 8 elements (4 bytes in 4-bit format) */
        int32_t alignedEnd = rowStart + ((rowEnd - rowStart) / 8) * 8;

        for (; j < alignedEnd; j += 8) {
            /* Load 4 bytes (8 quantized values) */
            uint8_t qvals[4];
            for (int k = 0; k < 4; k++) {
                qvals[k] = csr->qvalues[(j + k * 2) / 2];
            }

            /* First 4 values (lower 4 bits of each byte) */
            int32_t cols1[4];
            float   vals1[4];

            for (int k = 0; k < 4; k++) {
                /* Extract and dequantize values */
                uint8_t qval = qvals[k] & 0x0F;
                vals1[k]     = qval * csr->scale + csr->zeroPoint;

                /* Get column indices */
                cols1[k] = csr->colIndices[j + k * 2];
            }

            /* Load x values for first 4 */
            __m128 xvals1 = _mm_set_ps(x[cols1[3]], x[cols1[2]], x[cols1[1]], x[cols1[0]]);

            /* Load dequantized values for first 4 */
            __m128 vvals1 = _mm_loadu_ps(vals1);

            /* Multiply and accumulate first 4 */
            sum = _mm_add_ps(sum, _mm_mul_ps(vvals1, xvals1));

            /* Second 4 values (upper 4 bits of each byte) */
            int32_t cols2[4];
            float   vals2[4];

            for (int k = 0; k < 4; k++) {
                /* Extract and dequantize values */
                uint8_t qval = (qvals[k] >> 4) & 0x0F;
                vals2[k]     = qval * csr->scale + csr->zeroPoint;

                /* Get column indices */
                cols2[k] = csr->colIndices[j + k * 2 + 1];
            }

            /* Load x values for second 4 */
            __m128 xvals2 = _mm_set_ps(x[cols2[3]], x[cols2[2]], x[cols2[1]], x[cols2[0]]);

            /* Load dequantized values for second 4 */
            __m128 vvals2 = _mm_loadu_ps(vals2);

            /* Multiply and accumulate second 4 */
            sum = _mm_add_ps(sum, _mm_mul_ps(vvals2, xvals2));
        }

        /* Reduce sum vector to single value */
        float temp[4];
        _mm_storeu_ps(temp, sum);
        float rowSum = temp[0] + temp[1] + temp[2] + temp[3];

        /* Process remaining elements */
        for (; j < rowEnd; j++) {
            /* Extract 4-bit value */
            uint8_t qval;
            if (j % 2 == 0) {
                qval = csr->qvalues[j / 2] & 0x0F;
            }
            else {
                qval = (csr->qvalues[j / 2] >> 4) & 0x0F;
            }

            /* Dequantize */
            float val = qval * csr->scale + csr->zeroPoint;

            int32_t col = csr->colIndices[j];
            rowSum += val * x[col];
        }

        y[i] = rowSum;
    }

    return true;
}
#else
/* Non-SIMD fallback implementations */
bool tinyaiCSRMatrixVectorMulSIMD(const TinyAICSRMatrix *csr, const float *x, float *y)
{
    return tinyaiCSRMatrixVectorMul(csr, x, y);
}

bool tinyaiCSRMatrix4BitVectorMulSIMD(const TinyAICSRMatrix4Bit *csr, const float *x, float *y)
{
    return tinyaiCSRMatrix4BitVectorMul(csr, x, y);
}
#endif

/**
 * Calculate memory usage of CSR matrix in bytes
 */
size_t tinyaiCSRMatrixMemoryUsage(const TinyAICSRMatrix *csr)
{
    if (!csr) {
        return 0;
    }

    size_t memoryUsage = 0;

    /* Size of the struct itself */
    memoryUsage += sizeof(TinyAICSRMatrix);

    /* Size of arrays */
    memoryUsage += csr->nnz * sizeof(float);          /* values */
    memoryUsage += csr->nnz * sizeof(int32_t);        /* colIndices */
    memoryUsage += (csr->rows + 1) * sizeof(int32_t); /* rowPtrs */

    return memoryUsage;
}

/**
 * Calculate memory usage of 4-bit quantized CSR matrix in bytes
 */
size_t tinyaiCSRMatrix4BitMemoryUsage(const TinyAICSRMatrix4Bit *csr)
{
    if (!csr) {
        return 0;
    }

    size_t memoryUsage = 0;

    /* Size of the struct itself */
    memoryUsage += sizeof(TinyAICSRMatrix4Bit);

    /* Size of arrays */
    memoryUsage += (csr->nnz + 1) / 2;                /* qvalues (4-bit packed) */
    memoryUsage += csr->nnz * sizeof(int32_t);        /* colIndices */
    memoryUsage += (csr->rows + 1) * sizeof(int32_t); /* rowPtrs */

    return memoryUsage;
}

/**
 * Calculate compression ratio compared to dense matrix storage
 */
float tinyaiCSRMatrixCompressionRatio(const TinyAICSRMatrix *csr)
{
    if (!csr) {
        return 0.0f;
    }

    /* Size of dense matrix in bytes */
    size_t denseSize = csr->rows * csr->cols * sizeof(float);

    /* Size of CSR matrix in bytes */
    size_t sparseSize = tinyaiCSRMatrixMemoryUsage(csr);

    return (float)denseSize / (float)sparseSize;
}

/**
 * Calculate compression ratio for 4-bit quantized CSR matrix compared to dense matrix storage
 */
float tinyaiCSRMatrix4BitCompressionRatio(const TinyAICSRMatrix4Bit *csr)
{
    if (!csr) {
        return 0.0f;
    }

    /* Size of dense matrix in bytes */
    size_t denseSize = csr->rows * csr->cols * sizeof(float);

    /* Size of 4-bit quantized CSR matrix in bytes */
    size_t sparseSize = tinyaiCSRMatrix4BitMemoryUsage(csr);

    return (float)denseSize / (float)sparseSize;
}
