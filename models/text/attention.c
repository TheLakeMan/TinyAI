/**
 * @file attention.c
 * @brief SIMD-accelerated attention mechanism for transformer models
 *
 * This file implements attention mechanisms with SIMD optimization.
 */

#include "attention.h"
#include "../../core/memory.h"
#include "../../utils/simd_ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* SIMD detection and platform-specific includes - same as in simd_ops.c */
#if defined(_MSC_VER)
/* Windows/MSVC */
#include <intrin.h>
#define HAS_SSE2_SUPPORT 1
#if (_MSC_VER >= 1600) /* Visual Studio 2010 and later */
#define HAS_AVX_SUPPORT 1
#endif
#if (_MSC_VER >= 1700) /* Visual Studio 2012 and later */
#define HAS_AVX2_SUPPORT 1
#endif
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
/* GCC/Clang on x86 */
#include <cpuid.h>
#if defined(__SSE2__)
#include <emmintrin.h>
#define HAS_SSE2_SUPPORT 1
#endif
#if defined(__AVX__)
#include <immintrin.h>
#define HAS_AVX_SUPPORT 1
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#define HAS_AVX2_SUPPORT 1
#endif
#endif

/**
 * Memory pool calculations for scratch memory
 */
static size_t calculateScratchMemorySize(const TinyAIAttentionParams *params)
{
    uint32_t hiddenDim = params->hiddenDim;
    uint32_t seqLength = params->seqLength;
    uint32_t numHeads  = params->numHeads;
    uint32_t headDim   = params->headDim;

    /* Calculate size for all temporary buffers */
    size_t qkvSize     = 3 * seqLength * hiddenDim * sizeof(float);        /* Query, Key, Value */
    size_t scoresSize  = numHeads * seqLength * seqLength * sizeof(float); /* Attention scores */
    size_t softmaxSize = numHeads * seqLength * seqLength * sizeof(float); /* Softmax scores */
    size_t contextSize = seqLength * hiddenDim * sizeof(float);            /* Context vectors */

    return qkvSize + scoresSize + softmaxSize + contextSize;
}

/**
 * Initialize self-attention structure
 */
int tinyaiInitSelfAttention(TinyAISelfAttention *attention, const TinyAIAttentionParams *params)
{
    if (!attention || !params) {
        return -1;
    }

    /* Copy parameters */
    memcpy(&attention->params, params, sizeof(TinyAIAttentionParams));

    /* Initialize weight matrices to zeros */
    memset(&attention->queryWeight, 0, sizeof(TinyAIMatrix4bit));
    memset(&attention->keyWeight, 0, sizeof(TinyAIMatrix4bit));
    memset(&attention->valueWeight, 0, sizeof(TinyAIMatrix4bit));
    memset(&attention->outputWeight, 0, sizeof(TinyAIMatrix4bit));

    /* Initialize bias pointers to NULL */
    attention->queryBias  = NULL;
    attention->keyBias    = NULL;
    attention->valueBias  = NULL;
    attention->outputBias = NULL;

    /* Allocate scratch memory */
    size_t scratchSize       = calculateScratchMemorySize(params);
    attention->scratchMemory = (float *)TINYAI_MALLOC(scratchSize);
    if (!attention->scratchMemory) {
        return -1;
    }

    return 0;
}

/**
 * Free self-attention resources
 */
void tinyaiDestroySelfAttention(TinyAISelfAttention *attention)
{
    if (!attention) {
        return;
    }

    /* Free weight data if owned by this structure */
    if (attention->queryWeight.data) {
        TINYAI_FREE(attention->queryWeight.data);
    }
    if (attention->keyWeight.data) {
        TINYAI_FREE(attention->keyWeight.data);
    }
    if (attention->valueWeight.data) {
        TINYAI_FREE(attention->valueWeight.data);
    }
    if (attention->outputWeight.data) {
        TINYAI_FREE(attention->outputWeight.data);
    }

    /* Free biases if owned by this structure */
    if (attention->queryBias) {
        TINYAI_FREE(attention->queryBias);
    }
    if (attention->keyBias) {
        TINYAI_FREE(attention->keyBias);
    }
    if (attention->valueBias) {
        TINYAI_FREE(attention->valueBias);
    }
    if (attention->outputBias) {
        TINYAI_FREE(attention->outputBias);
    }

    /* Free scratch memory */
    if (attention->scratchMemory) {
        TINYAI_FREE(attention->scratchMemory);
    }

    /* Reset the structure */
    memset(attention, 0, sizeof(TinyAISelfAttention));
}

/**
 * Set weights for self-attention
 */
int tinyaiSetAttentionWeights(TinyAISelfAttention *attention, const TinyAIMatrix4bit *queryWeight,
                              const TinyAIMatrix4bit *keyWeight,
                              const TinyAIMatrix4bit *valueWeight,
                              const TinyAIMatrix4bit *outputWeight, const float *queryBias,
                              const float *keyBias, const float *valueBias, const float *outputBias)
{
    if (!attention || !queryWeight || !keyWeight || !valueWeight || !outputWeight) {
        return -1;
    }

    uint32_t hiddenDim = attention->params.hiddenDim;

    /* Free existing weight data if needed */
    if (attention->queryWeight.data) {
        TINYAI_FREE(attention->queryWeight.data);
    }
    if (attention->keyWeight.data) {
        TINYAI_FREE(attention->keyWeight.data);
    }
    if (attention->valueWeight.data) {
        TINYAI_FREE(attention->valueWeight.data);
    }
    if (attention->outputWeight.data) {
        TINYAI_FREE(attention->outputWeight.data);
    }

    /* Calculate the size of each matrix in bytes */
    /* For 4-bit quantized weights, we use 2 values per byte */
    size_t querySize  = (queryWeight->rows * queryWeight->cols + 1) / 2;
    size_t keySize    = (keyWeight->rows * keyWeight->cols + 1) / 2;
    size_t valueSize  = (valueWeight->rows * valueWeight->cols + 1) / 2;
    size_t outputSize = (outputWeight->rows * outputWeight->cols + 1) / 2;

    /* Allocate and copy weight data */
    attention->queryWeight.data  = (uint8_t *)TINYAI_MALLOC(querySize);
    attention->keyWeight.data    = (uint8_t *)TINYAI_MALLOC(keySize);
    attention->valueWeight.data  = (uint8_t *)TINYAI_MALLOC(valueSize);
    attention->outputWeight.data = (uint8_t *)TINYAI_MALLOC(outputSize);

    if (!attention->queryWeight.data || !attention->keyWeight.data ||
        !attention->valueWeight.data || !attention->outputWeight.data) {
        /* Clean up on error */
        tinyaiDestroySelfAttention(attention);
        return -1;
    }

    /* Copy weight data */
    memcpy(attention->queryWeight.data, queryWeight->data, querySize);
    memcpy(attention->keyWeight.data, keyWeight->data, keySize);
    memcpy(attention->valueWeight.data, valueWeight->data, valueSize);
    memcpy(attention->outputWeight.data, outputWeight->data, outputSize);

    /* Copy matrix metadata */
    attention->queryWeight.rows      = queryWeight->rows;
    attention->queryWeight.cols      = queryWeight->cols;
    attention->queryWeight.scale     = queryWeight->scale;
    attention->queryWeight.zeroPoint = queryWeight->zeroPoint;

    attention->keyWeight.rows      = keyWeight->rows;
    attention->keyWeight.cols      = keyWeight->cols;
    attention->keyWeight.scale     = keyWeight->scale;
    attention->keyWeight.zeroPoint = keyWeight->zeroPoint;

    attention->valueWeight.rows      = valueWeight->rows;
    attention->valueWeight.cols      = valueWeight->cols;
    attention->valueWeight.scale     = valueWeight->scale;
    attention->valueWeight.zeroPoint = valueWeight->zeroPoint;

    attention->outputWeight.rows      = outputWeight->rows;
    attention->outputWeight.cols      = outputWeight->cols;
    attention->outputWeight.scale     = outputWeight->scale;
    attention->outputWeight.zeroPoint = outputWeight->zeroPoint;

    /* Free existing bias data if needed */
    if (attention->queryBias) {
        TINYAI_FREE(attention->queryBias);
    }
    if (attention->keyBias) {
        TINYAI_FREE(attention->keyBias);
    }
    if (attention->valueBias) {
        TINYAI_FREE(attention->valueBias);
    }
    if (attention->outputBias) {
        TINYAI_FREE(attention->outputBias);
    }

    /* Handle biases (optional) */
    if (queryBias) {
        attention->queryBias = (float *)TINYAI_MALLOC(hiddenDim * sizeof(float));
        if (!attention->queryBias) {
            tinyaiDestroySelfAttention(attention);
            return -1;
        }
        memcpy(attention->queryBias, queryBias, hiddenDim * sizeof(float));
    }

    if (keyBias) {
        attention->keyBias = (float *)TINYAI_MALLOC(hiddenDim * sizeof(float));
        if (!attention->keyBias) {
            tinyaiDestroySelfAttention(attention);
            return -1;
        }
        memcpy(attention->keyBias, keyBias, hiddenDim * sizeof(float));
    }

    if (valueBias) {
        attention->valueBias = (float *)TINYAI_MALLOC(hiddenDim * sizeof(float));
        if (!attention->valueBias) {
            tinyaiDestroySelfAttention(attention);
            return -1;
        }
        memcpy(attention->valueBias, valueBias, hiddenDim * sizeof(float));
    }

    if (outputBias) {
        attention->outputBias = (float *)TINYAI_MALLOC(hiddenDim * sizeof(float));
        if (!attention->outputBias) {
            tinyaiDestroySelfAttention(attention);
            return -1;
        }
        memcpy(attention->outputBias, outputBias, hiddenDim * sizeof(float));
    }

    return 0;
}

/**
 * Calculate memory offsets for different components in scratch memory
 */
static void getMemoryOffsets(const TinyAIAttentionParams *params, float **query, float **key,
                             float **value, float **scores, float **softmaxScores, float **context,
                             float *scratchMemory)
{
    uint32_t hiddenDim = params->hiddenDim;
    uint32_t seqLength = params->seqLength;
    uint32_t numHeads  = params->numHeads;

    /* Calculate offsets for each component */
    *query         = scratchMemory;
    *key           = *query + seqLength * hiddenDim;
    *value         = *key + seqLength * hiddenDim;
    *scores        = *value + seqLength * hiddenDim;
    *softmaxScores = *scores + numHeads * seqLength * seqLength;
    *context       = *softmaxScores + numHeads * seqLength * seqLength;
}

/**
 * SIMD-accelerated query-key-value projection using AVX2
 */
#if defined(HAS_AVX2_SUPPORT)
static int tinyaiSimdQKVProjectionAVX2(const float *input, const TinyAIMatrix4bit *queryWeight,
                                       const TinyAIMatrix4bit *keyWeight,
                                       const TinyAIMatrix4bit *valueWeight, const float *queryBias,
                                       const float *keyBias, const float *valueBias, float *query,
                                       float *key, float *value, uint32_t seqLength,
                                       uint32_t hiddenDim, uint32_t numHeads, uint32_t headDim)
{
    /* For each position in the sequence */
    for (uint32_t i = 0; i < seqLength; i++) {
        /* Get input vector for this position */
        const float *inputVec = input + i * hiddenDim;

        /* Output vectors for this position */
        float *queryVec = query + i * hiddenDim;
        float *keyVec   = key + i * hiddenDim;
        float *valueVec = value + i * hiddenDim;

        /* Perform matrix multiplication for query, key, value projections */
        /* We'll use tinyaiSimdMatMul4Bit with AVX2 which will be automatically selected */
        /* Query projection */
        float queryScale = queryWeight->scale;
        tinyaiSimdMatMul4Bit(queryVec, queryWeight->data, inputVec, hiddenDim, hiddenDim,
                             &queryScale);

        /* Key projection */
        float keyScale = keyWeight->scale;
        tinyaiSimdMatMul4Bit(keyVec, keyWeight->data, inputVec, hiddenDim, hiddenDim, &keyScale);

        /* Value projection */
        float valueScale = valueWeight->scale;
        tinyaiSimdMatMul4Bit(valueVec, valueWeight->data, inputVec, hiddenDim, hiddenDim,
                             &valueScale);

        /* Add biases if provided */
        if (queryBias) {
            /* Use SIMD vector addition */
            for (uint32_t j = 0; j < hiddenDim; j += 8) {
                __m256 biasVec   = _mm256_loadu_ps(queryBias + j);
                __m256 valVec    = _mm256_loadu_ps(queryVec + j);
                __m256 resultVec = _mm256_add_ps(valVec, biasVec);
                _mm256_storeu_ps(queryVec + j, resultVec);
            }
        }

        if (keyBias) {
            /* Use SIMD vector addition */
            for (uint32_t j = 0; j < hiddenDim; j += 8) {
                __m256 biasVec   = _mm256_loadu_ps(keyBias + j);
                __m256 valVec    = _mm256_loadu_ps(keyVec + j);
                __m256 resultVec = _mm256_add_ps(valVec, biasVec);
                _mm256_storeu_ps(keyVec + j, resultVec);
            }
        }

        if (valueBias) {
            /* Use SIMD vector addition */
            for (uint32_t j = 0; j < hiddenDim; j += 8) {
                __m256 biasVec   = _mm256_loadu_ps(valueBias + j);
                __m256 valVec    = _mm256_loadu_ps(valueVec + j);
                __m256 resultVec = _mm256_add_ps(valVec, biasVec);
                _mm256_storeu_ps(valueVec + j, resultVec);
            }
        }
    }

    return 0;
}
#endif

/**
 * SIMD-accelerated query-key-value projection using SSE2
 */
#if defined(HAS_SSE2_SUPPORT)
static int tinyaiSimdQKVProjectionSSE2(const float *input, const TinyAIMatrix4bit *queryWeight,
                                       const TinyAIMatrix4bit *keyWeight,
                                       const TinyAIMatrix4bit *valueWeight, const float *queryBias,
                                       const float *keyBias, const float *valueBias, float *query,
                                       float *key, float *value, uint32_t seqLength,
                                       uint32_t hiddenDim, uint32_t numHeads, uint32_t headDim)
{
    /* For each position in the sequence */
    for (uint32_t i = 0; i < seqLength; i++) {
        /* Get input vector for this position */
        const float *inputVec = input + i * hiddenDim;

        /* Output vectors for this position */
        float *queryVec = query + i * hiddenDim;
        float *keyVec   = key + i * hiddenDim;
        float *valueVec = value + i * hiddenDim;

        /* Perform matrix multiplication for query, key, value projections */
        /* We'll use tinyaiSimdMatMul4Bit which will use SSE2 */
        /* Query projection */
        float queryScale = queryWeight->scale;
        tinyaiSimdMatMul4Bit(queryVec, queryWeight->data, inputVec, hiddenDim, hiddenDim,
                             &queryScale);

        /* Key projection */
        float keyScale = keyWeight->scale;
        tinyaiSimdMatMul4Bit(keyVec, keyWeight->data, inputVec, hiddenDim, hiddenDim, &keyScale);

        /* Value projection */
        float valueScale = valueWeight->scale;
        tinyaiSimdMatMul4Bit(valueVec, valueWeight->data, inputVec, hiddenDim, hiddenDim,
                             &valueScale);

        /* Add biases if provided */
        if (queryBias) {
            /* Use SIMD vector addition */
            for (uint32_t j = 0; j < hiddenDim; j += 4) {
                __m128 biasVec   = _mm_loadu_ps(queryBias + j);
                __m128 valVec    = _mm_loadu_ps(queryVec + j);
                __m128 resultVec = _mm_add_ps(valVec, biasVec);
                _mm_storeu_ps(queryVec + j, resultVec);
            }
        }

        if (keyBias) {
            /* Use SIMD vector addition */
            for (uint32_t j = 0; j < hiddenDim; j += 4) {
                __m128 biasVec   = _mm_loadu_ps(keyBias + j);
                __m128 valVec    = _mm_loadu_ps(keyVec + j);
                __m128 resultVec = _mm_add_ps(valVec, biasVec);
                _mm_storeu_ps(keyVec + j, resultVec);
            }
        }

        if (valueBias) {
            /* Use SIMD vector addition */
            for (uint32_t j = 0; j < hiddenDim; j += 4) {
                __m128 biasVec   = _mm_loadu_ps(valueBias + j);
                __m128 valVec    = _mm_loadu_ps(valueVec + j);
                __m128 resultVec = _mm_add_ps(valVec, biasVec);
                _mm_storeu_ps(valueVec + j, resultVec);
            }
        }
    }

    return 0;
}
#endif

/**
 * Reference implementation of query-key-value projection
 */
static int tinyaiQKVProjectionReference(const float *input, const TinyAIMatrix4bit *queryWeight,
                                        const TinyAIMatrix4bit *keyWeight,
                                        const TinyAIMatrix4bit *valueWeight, const float *queryBias,
                                        const float *keyBias, const float *valueBias, float *query,
                                        float *key, float *value, uint32_t seqLength,
                                        uint32_t hiddenDim, uint32_t numHeads, uint32_t headDim)
{
    /* For each position in the sequence */
    for (uint32_t i = 0; i < seqLength; i++) {
        /* Get input vector for this position */
        const float *inputVec = input + i * hiddenDim;

        /* Dequantize weight matrices */
        TinyAIMatrixFP32 *queryWeightFP32 = tinyaiDequantize4bitToFP32(queryWeight);
        TinyAIMatrixFP32 *keyWeightFP32   = tinyaiDequantize4bitToFP32(keyWeight);
        TinyAIMatrixFP32 *valueWeightFP32 = tinyaiDequantize4bitToFP32(valueWeight);

        if (!queryWeightFP32 || !keyWeightFP32 || !valueWeightFP32) {
            if (queryWeightFP32)
                tinyaiDestroyMatrixFP32(queryWeightFP32);
            if (keyWeightFP32)
                tinyaiDestroyMatrixFP32(keyWeightFP32);
            if (valueWeightFP32)
                tinyaiDestroyMatrixFP32(valueWeightFP32);
            return -1;
        }

        /* Output vectors for this position */
        float *queryVec = query + i * hiddenDim;
        float *keyVec   = key + i * hiddenDim;
        float *valueVec = value + i * hiddenDim;

        /* Perform matrix multiplication for query projection */
        for (uint32_t j = 0; j < hiddenDim; j++) {
            float sum = queryBias ? queryBias[j] : 0.0f;
            for (uint32_t k = 0; k < hiddenDim; k++) {
                sum += inputVec[k] * queryWeightFP32->data[k * hiddenDim + j];
            }
            queryVec[j] = sum;
        }

        /* Perform matrix multiplication for key projection */
        for (uint32_t j = 0; j < hiddenDim; j++) {
            float sum = keyBias ? keyBias[j] : 0.0f;
            for (uint32_t k = 0; k < hiddenDim; k++) {
                sum += inputVec[k] * keyWeightFP32->data[k * hiddenDim + j];
            }
            keyVec[j] = sum;
        }

        /* Perform matrix multiplication for value projection */
        for (uint32_t j = 0; j < hiddenDim; j++) {
            float sum = valueBias ? valueBias[j] : 0.0f;
            for (uint32_t k = 0; k < hiddenDim; k++) {
                sum += inputVec[k] * valueWeightFP32->data[k * hiddenDim + j];
            }
            valueVec[j] = sum;
        }

        /* Clean up */
        tinyaiDestroyMatrixFP32(queryWeightFP32);
        tinyaiDestroyMatrixFP32(keyWeightFP32);
        tinyaiDestroyMatrixFP32(valueWeightFP32);
    }

    return 0;
}

/**
 * SIMD-accelerated query-key-value projection (public API)
 */
int tinyaiSimdQKVProjection(const float *input, const TinyAIMatrix4bit *queryWeight,
                            const TinyAIMatrix4bit *keyWeight, const TinyAIMatrix4bit *valueWeight,
                            const float *queryBias, const float *keyBias, const float *valueBias,
                            float *query, float *key, float *value, uint32_t seqLength,
                            uint32_t hiddenDim, uint32_t numHeads, uint32_t headDim)
{
    /* Check if SIMD is available and which version */
    extern bool g_simdInitialized;
    extern bool g_hasSSE2;
    extern bool g_hasAVX;
    extern bool g_hasAVX2;

    /* Use the most advanced SIMD version available */
#if defined(HAS_AVX2_SUPPORT)
    if (g_hasAVX2) {
        return tinyaiSimdQKVProjectionAVX2(input, queryWeight, keyWeight, valueWeight, queryBias,
                                           keyBias, valueBias, query, key, value, seqLength,
                                           hiddenDim, numHeads, headDim);
    }
#endif

#if defined(HAS_SSE2_SUPPORT)
    if (g_hasSSE2) {
        return tinyaiSimdQKVProjectionSSE2(input, queryWeight, keyWeight, valueWeight, queryBias,
                                           keyBias, valueBias, query, key, value, seqLength,
                                           hiddenDim, numHeads, headDim);
    }
#endif

    /* Fallback to reference implementation */
    return tinyaiQKVProjectionReference(input, queryWeight, keyWeight, valueWeight, queryBias,
                                        keyBias, valueBias, query, key, value, seqLength, hiddenDim,
                                        numHeads, headDim);
}

/**
 * SIMD-accelerated attention score computation (Q*K^T) using AVX2
 */
#if defined(HAS_AVX2_SUPPORT)
static int tinyaiSimdAttentionScoresAVX2(const float *query, const float *key, float *scores,
                                         uint32_t seqLength, uint32_t numHeads, uint32_t headDim,
                                         float scaleFactor, bool useCausalMask)
{
    /* Process each attention head separately */
    for (uint32_t h = 0; h < numHeads; h++) {
        /* Compute QK^T for this head */
        for (uint32_t i = 0; i < seqLength; i++) {     /* Query sequence position */
            for (uint32_t j = 0; j < seqLength; j++) { /* Key sequence position */
                /* Skip future tokens for causal mask */
                if (useCausalMask && j > i) {
                    scores[h * seqLength * seqLength + i * seqLength + j] = -INFINITY;
                    continue;
                }

                /* Get query and key vectors for this head at positions i and j */
                const float *queryVec = query + i * numHeads * headDim + h * headDim;
                const float *keyVec   = key + j * numHeads * headDim + h * headDim;

                /* Compute dot product with AVX2 */
                __m256 sumVec = _mm256_setzero_ps();

                /* Process 8 elements at a time */
                for (uint32_t k = 0; k < headDim; k += 8) {
                    __m256 queryChunk = _mm256_loadu_ps(queryVec + k);
                    __m256 keyChunk   = _mm256_loadu_ps(keyVec + k);

                    /* Multiply and accumulate */
                    sumVec = _mm256_add_ps(sumVec, _mm256_mul_ps(queryChunk, keyChunk));
                }

                /* Horizontal sum */
                float sum[8];
                _mm256_storeu_ps(sum, sumVec);
                float dotProduct =
                    sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];

                /* Add remaining elements */
                for (uint32_t k = (headDim / 8) * 8; k < headDim; k++) {
                    dotProduct += queryVec[k] * keyVec[k];
                }

                /* Scale and store */
                scores[h * seqLength * seqLength + i * seqLength + j] = dotProduct * scaleFactor;
            }
        }
    }

    return 0;
}
#endif

/**
 * SIMD-accelerated attention score computation (Q*K^T) using SSE2
 */
#if defined(HAS_SSE2_SUPPORT)
static int tinyaiSimdAttentionScoresSSE2(const float *query, const float *key, float *scores,
                                         uint32_t seqLength, uint32_t numHeads, uint32_t headDim,
                                         float scaleFactor, bool useCausalMask)
{
    /* Process each attention head separately */
    for (uint32_t h = 0; h < numHeads; h++) {
        /* Compute QK^T for this head */
        for (uint32_t i = 0; i < seqLength; i++) {     /* Query sequence position */
            for (uint32_t j = 0; j < seqLength; j++) { /* Key sequence position */
                /* Skip future tokens for causal mask */
                if (useCausalMask && j > i) {
                    scores[h * seqLength * seqLength + i * seqLength + j] = -INFINITY;
                    continue;
                }

                /* Get query and key vectors for this head at positions i and j */
                const float *queryVec = query + i * numHeads * headDim + h * headDim;
                const float *keyVec   = key + j * numHeads * headDim + h * headDim;

                /* Compute dot product with SSE2 */
                __m128 sumVec = _mm_setzero_ps();

                /* Process 4 elements at a time */
                for (uint32_t k = 0; k < headDim; k += 4) {
                    __m128 queryChunk = _mm_loadu_ps(queryVec + k);
                    __m128 keyChunk   = _mm_loadu_ps(keyVec + k);

                    /* Multiply and accumulate */
                    sumVec = _mm_add_ps(sumVec, _mm_mul_ps(queryChunk, keyChunk));
                }

                /* Horizontal sum */
                float sum[4];
                _mm_storeu_ps(sum, sumVec);
                float dotProduct = sum[0] + sum[1] + sum[2] + sum[3];

                /* Add remaining elements */
                for (uint32_t k = (headDim / 4) * 4; k < headDim; k++) {
                    dotProduct += queryVec[k] * keyVec[k];
                }

                /* Scale and store */
                scores[h * seqLength * seqLength + i * seqLength + j] = dotProduct * scaleFactor;
            }
        }
    }

    return 0;
}
#endif

/**
 * Reference implementation of attention score computation
 */
static int tinyaiAttentionScoresReference(const float *query, const float *key, float *scores,
                                          uint32_t seqLength, uint32_t numHeads, uint32_t headDim,
                                          float scaleFactor, bool useCausalMask)
{
    /* Process each attention head separately */
    for (uint32_t h = 0; h < numHeads; h++) {
        /* Compute QK^T for this head */
        for (uint32_t i = 0; i < seqLength; i++) {     /* Query sequence position */
            for (uint32_t j = 0; j < seqLength; j++) { /* Key sequence position */
                /* Skip future tokens for causal mask */
                if (useCausalMask && j > i) {
                    scores[h * seqLength * seqLength + i * seqLength + j] = -INFINITY;
                    continue;
                }

                /* Get query and key vectors for this head at positions i and j */
                const float *queryVec = query + i * numHeads * headDim + h * headDim;
                const float *keyVec   = key + j * numHeads * headDim + h * headDim;

                /* Compute dot product */
                float dotProduct = 0.0f;

                for (uint32_t k = 0; k < headDim; k++) {
                    dotProduct += queryVec[k] * keyVec[k];
                }

                /* Scale and store */
                scores[h * seqLength * seqLength + i * seqLength + j] = dotProduct * scaleFactor;
            }
        }
    }

    return 0;
}

/**
 * SIMD-accelerated attention score computation (public API)
 */
int tinyaiSimdAttentionScores(const float *query, const float *key, float *scores,
                              uint32_t seqLength, uint32_t numHeads, uint32_t headDim,
                              float scaleFactor, bool useCausalMask)
{
    /* Check if SIMD is available and which version */
    extern bool g_simdInitialized;
    extern bool g_hasSSE2;
    extern bool g_hasAVX;
    extern bool g_hasAVX2;

    /* Use the most advanced SIMD version available */
#if defined(HAS_AVX2_SUPPORT)
    if (g_hasAVX2) {
        return tinyaiSimdAttentionScoresAVX2(query, key, scores, seqLength, numHeads, headDim,
                                             scaleFactor, useCausalMask);
    }
#endif

#if defined(HAS_SSE2_SUPPORT)
    if (g_hasSSE2) {
        return tinyaiSimdAttentionScoresSSE2(query, key, scores, seqLength, numHeads, headDim,
                                             scaleFactor, useCausalMask);
    }
#endif

    /* Fallback to reference implementation */
    return tinyaiAttentionScoresReference(query, key, scores, seqLength, numHeads, headDim,
                                          scaleFactor, useCausalMask);
}

/**
 * Implementation of the full self-attention forward pass (using the component functions above)
 */
int tinyaiSelfAttentionForward(TinyAISelfAttention *attention, const float *input, float *output)
{
    if (!attention || !input || !output) {
        return -1;
    }

    /* Get attention parameters */
    TinyAIAttentionParams *params    = &attention->params;
    uint32_t               seqLength = params->seqLength;
    uint32_t               hiddenDim = params->hiddenDim;
    uint32_t               numHeads  = params->numHeads;
    uint32_t               headDim   = params->headDim;

    /* Get memory areas from scratch memory for intermediate results */
    float *query, *key, *value, *scores, *softmaxScores, *context;
    getMemoryOffsets(params, &query, &key, &value, &scores, &softmaxScores, &context,
                     attention->scratchMemory);

    /* Perform QKV projection */
    tinyaiSimdQKVProjection(input, &attention->queryWeight, &attention->keyWeight,
                            &attention->valueWeight, attention->queryBias, attention->keyBias,
                            attention->valueBias, query, key, value, seqLength, hiddenDim, numHeads,
                            headDim);

    /* Compute attention scores (Q * K^T / sqrt(headDim)) */
    float scaleFactor = params->scaleFactor;
    tinyaiSimdAttentionScores(query, key, scores, seqLength, numHeads, headDim, scaleFactor,
                              params->useCausalMask);

    /* Apply softmax to attention scores */
    tinyaiSimdAttentionSoftmax(scores, softmaxScores, seqLength, numHeads);

    /* Compute context vectors (softmax(QK^T) * V) */
    tinyaiSimdAttentionContext(softmaxScores, value, context, seqLength, numHeads, headDim);

    /* Final output projection */
    tinyaiSimdOutputProjection(context, &attention->outputWeight, attention->outputBias, output,
                               seqLength, hiddenDim);

    return 0;
}

/**
 * SIMD-accelerated softmax computation using AVX2
 */
#if defined(HAS_AVX2_SUPPORT)
static int tinyaiSimdAttentionSoftmaxAVX2(const float *scores, float *softmaxScores,
                                          uint32_t seqLength, uint32_t numHeads)
{
    /* Process each attention head separately */
    for (uint32_t h = 0; h < numHeads; h++) {
        /* Process each query position */
        for (uint32_t i = 0; i < seqLength; i++) {
            /* Get pointer to scores for this head and query position */
            const float *rowScores  = scores + h * seqLength * seqLength + i * seqLength;
            float       *rowSoftmax = softmaxScores + h * seqLength * seqLength + i * seqLength;

            /* Find max score for numerical stability */
            float maxScore = -INFINITY;
            for (uint32_t j = 0; j < seqLength; j++) {
                if (rowScores[j] > maxScore) {
                    maxScore = rowScores[j];
                }
            }

            /* Compute exp(score - maxScore) and sum */
            __m256 maxVec = _mm256_set1_ps(maxScore);
            __m256 sumVec = _mm256_setzero_ps();

            /* Process in chunks of 8 */
            for (uint32_t j = 0; j < seqLength; j += 8) {
                /* Clamp for last iteration if needed */
                uint32_t remainingItems = seqLength - j;
                uint32_t itemsToProcess = remainingItems < 8 ? remainingItems : 8;

                if (itemsToProcess == 8) {
                    /* Process full 8 elements */
                    __m256 scoresVec = _mm256_loadu_ps(rowScores + j);
                    __m256 expInput  = _mm256_sub_ps(scoresVec, maxVec);

                    /* Compute exp(x) using AVX2 */
                    /* We use the approximation exp(x) ≈ 2^(1.442695*x) */
                    /* First, compute 1.442695*x (multiply by log2(e)) */
                    __m256 log2eVec = _mm256_set1_ps(1.442695f);
                    __m256 scaled   = _mm256_mul_ps(expInput, log2eVec);

                    /* Use _mm256_exp_ps if available, otherwise approximate */
                    /* For simplicity, we use exp(x) = exp2f(x * log2(e)) */
                    __m256 expVec;

                    /* This is a simple approximation, in practice you would use
                     * a more accurate approximation or library function */
                    float expResults[8];
                    float scaledValues[8];
                    _mm256_storeu_ps(scaledValues, scaled);

                    /* Use scalar operations to compute exp */
                    _mm256_storeu_ps(expResults, expInput);
                    for (int k = 0; k < 8; k++) {
                        expResults[k] = expf(expResults[k]);
                    }
                    expVec = _mm256_loadu_ps(expResults);

                    /* Store results and accumulate sum */
                    _mm256_storeu_ps(rowSoftmax + j, expVec);
                    sumVec = _mm256_add_ps(sumVec, expVec);
                }
                else {
                    /* Process remaining items individually */
                    for (uint32_t k = 0; k < itemsToProcess; k++) {
                        float expValue    = expf(rowScores[j + k] - maxScore);
                        rowSoftmax[j + k] = expValue;
                        /* Use scalar addition for the sum */
                        float sumArr[8];
                        _mm256_storeu_ps(sumArr, sumVec);
                        sumArr[0] += expValue;
                        sumVec = _mm256_loadu_ps(sumArr);
                    }
                }
            }

            /* Finalize sum */
            float sumArr[8];
            _mm256_storeu_ps(sumArr, sumVec);
            float sum = sumArr[0] + sumArr[1] + sumArr[2] + sumArr[3] + sumArr[4] + sumArr[5] +
                        sumArr[6] + sumArr[7];

            /* Normalize by sum */
            __m256 sumRecipVec = _mm256_set1_ps(1.0f / sum);

            /* Normalize in chunks of 8 */
            for (uint32_t j = 0; j < seqLength; j += 8) {
                uint32_t remainingItems = seqLength - j;
                uint32_t itemsToProcess = remainingItems < 8 ? remainingItems : 8;

                if (itemsToProcess == 8) {
                    /* Full vector processing */
                    __m256 valuesVec     = _mm256_loadu_ps(rowSoftmax + j);
                    __m256 normalizedVec = _mm256_mul_ps(valuesVec, sumRecipVec);
                    _mm256_storeu_ps(rowSoftmax + j, normalizedVec);
                }
                else {
                    /* Individual processing for remainder */
                    for (uint32_t k = 0; k < itemsToProcess; k++) {
                        rowSoftmax[j + k] /= sum;
                    }
                }
            }
        }
    }

    return 0;
}
#endif

/**
 * SIMD-accelerated softmax computation using SSE2
 */
#if defined(HAS_SSE2_SUPPORT)
static int tinyaiSimdAttentionSoftmaxSSE2(const float *scores, float *softmaxScores,
                                          uint32_t seqLength, uint32_t numHeads)
{
    /* Process each attention head separately */
    for (uint32_t h = 0; h < numHeads; h++) {
        /* Process each query position */
        for (uint32_t i = 0; i < seqLength; i++) {
            /* Get pointer to scores for this head and query position */
            const float *rowScores  = scores + h * seqLength * seqLength + i * seqLength;
            float       *rowSoftmax = softmaxScores + h * seqLength * seqLength + i * seqLength;

            /* Find max score for numerical stability */
            float maxScore = -INFINITY;
            for (uint32_t j = 0; j < seqLength; j++) {
                if (rowScores[j] > maxScore) {
                    maxScore = rowScores[j];
                }
            }

            /* Compute exp(score - maxScore) and sum */
            __m128 maxVec = _mm_set1_ps(maxScore);
            __m128 sumVec = _mm_setzero_ps();

            /* Process in chunks of 4 */
            for (uint32_t j = 0; j < seqLength; j += 4) {
                /* Clamp for last iteration if needed */
                uint32_t remainingItems = seqLength - j;
                uint32_t itemsToProcess = remainingItems < 4 ? remainingItems : 4;

                if (itemsToProcess == 4) {
                    /* Process full 4 elements */
                    __m128 scoresVec = _mm_loadu_ps(rowScores + j);
                    __m128 expInput  = _mm_sub_ps(scoresVec, maxVec);

                    /* Compute exp using scalar operations for simplicity */
                    float expInputArr[4];
                    _mm_storeu_ps(expInputArr, expInput);

                    float expResults[4];
                    for (int k = 0; k < 4; k++) {
                        expResults[k] = expf(expInputArr[k]);
                    }

                    __m128 expVec = _mm_loadu_ps(expResults);

                    /* Store results and accumulate sum */
                    _mm_storeu_ps(rowSoftmax + j, expVec);
                    sumVec = _mm_add_ps(sumVec, expVec);
                }
                else {
                    /* Process remaining items individually */
                    for (uint32_t k = 0; k < itemsToProcess; k++) {
                        float expValue    = expf(rowScores[j + k] - maxScore);
                        rowSoftmax[j + k] = expValue;
                        /* Use scalar addition for the sum */
                        float sumArr[4];
                        _mm_storeu_ps(sumArr, sumVec);
                        sumArr[0] += expValue;
                        sumVec = _mm_loadu_ps(sumArr);
                    }
                }
            }

            /* Finalize sum */
            float sumArr[4];
            _mm_storeu_ps(sumArr, sumVec);
            float sum = sumArr[0] + sumArr[1] + sumArr[2] + sumArr[3];

            /* Normalize by sum */
            __m128 sumRecipVec = _mm_set1_ps(1.0f / sum);

            /* Normalize in chunks of 4 */
            for (uint32_t j = 0; j < seqLength; j += 4) {
                uint32_t remainingItems = seqLength - j;
                uint32_t itemsToProcess = remainingItems < 4 ? remainingItems : 4;

                if (itemsToProcess == 4) {
                    /* Full vector processing */
                    __m128 valuesVec     = _mm_loadu_ps(rowSoftmax + j);
                    __m128 normalizedVec = _mm_mul_ps(valuesVec, sumRecipVec);
                    _mm_storeu_ps(rowSoftmax + j, normalizedVec);
                }
                else {
                    /* Individual processing for remainder */
                    for (uint32_t k = 0; k < itemsToProcess; k++) {
                        rowSoftmax[j + k] /= sum;
                    }
                }
            }
        }
    }

    return 0;
}
#endif

/**
 * Reference implementation of softmax computation for attention scores
 */
static int tinyaiAttentionSoftmaxReference(const float *scores, float *softmaxScores,
                                           uint32_t seqLength, uint32_t numHeads)
{
    /* Process each attention head separately */
    for (uint32_t h = 0; h < numHeads; h++) {
        /* Process each query position */
        for (uint32_t i = 0; i < seqLength; i++) {
            /* Get pointers to scores for this head and query position */
            const float *rowScores  = scores + h * seqLength * seqLength + i * seqLength;
            float       *rowSoftmax = softmaxScores + h * seqLength * seqLength + i * seqLength;

            /* Find max score for numerical stability */
            float maxScore = -INFINITY;
            for (uint32_t j = 0; j < seqLength; j++) {
                if (rowScores[j] > maxScore) {
                    maxScore = rowScores[j];
                }
            }

            /* Compute exp(score - maxScore) and sum */
            float sum = 0.0f;
            for (uint32_t j = 0; j < seqLength; j++) {
                float expValue = expf(rowScores[j] - maxScore);
                rowSoftmax[j]  = expValue;
                sum += expValue;
            }

            /* Normalize by sum */
            if (sum > 0.0f) { /* Prevent division by zero */
                float invSum = 1.0f / sum;
                for (uint32_t j = 0; j < seqLength; j++) {
                    rowSoftmax[j] *= invSum;
                }
            }
        }
    }

    return 0;
}

/**
 * SIMD-accelerated softmax computation (public API)
 */
int tinyaiSimdAttentionSoftmax(const float *scores, float *softmaxScores, uint32_t seqLength,
                               uint32_t numHeads)
{
    /* Check if SIMD is available and which version */
    extern bool g_simdInitialized;
    extern bool g_hasSSE2;
    extern bool g_hasAVX;
    extern bool g_hasAVX2;

    /* Use the most advanced SIMD version available */
#if defined(HAS_AVX2_SUPPORT)
    if (g_hasAVX2) {
        return tinyaiSimdAttentionSoftmaxAVX2(scores, softmaxScores, seqLength, numHeads);
    }
#endif

#if defined(HAS_SSE2_SUPPORT)
    if (g_hasSSE2) {
        return tinyaiSimdAttentionSoftmaxSSE2(scores, softmaxScores, seqLength, numHeads);
    }
#endif

    /* Fallback to reference implementation */
    return tinyaiAttentionSoftmaxReference(scores, softmaxScores, seqLength, numHeads);
}

/**
 * SIMD-accelerated attention context computation using AVX2
 */
#if defined(HAS_AVX2_SUPPORT)
static int tinyaiSimdAttentionContextAVX2(const float *softmaxScores, const float *value,
                                          float *context, uint32_t seqLength, uint32_t numHeads,
                                          uint32_t headDim)
{
    /* Process each head */
    for (uint32_t h = 0; h < numHeads; h++) {
        /* Process each query position */
        for (uint32_t i = 0; i < seqLength; i++) {
            /* Get softmax scores for this query position */
            const float *scores = softmaxScores + h * seqLength * seqLength + i * seqLength;

            /* Initialize context vector for this position to zero */
            float *contextVec = context + i * numHeads * headDim + h * headDim;
            for (uint32_t d = 0; d < headDim; d += 8) {
                _mm256_storeu_ps(contextVec + d, _mm256_setzero_ps());
            }

            /* Compute weighted sum of value vectors */
            for (uint32_t j = 0; j < seqLength; j++) {
                /* Get value vector for this key position */
                const float *valueVec = value + j * numHeads * headDim + h * headDim;

                /* Get attention weight */
                float  weight    = scores[j];
                __m256 weightVec = _mm256_set1_ps(weight);

                /* Process in chunks of 8 */
                for (uint32_t d = 0; d < headDim; d += 8) {
                    /* Load current context and value vectors */
                    __m256 contextChunk = _mm256_loadu_ps(contextVec + d);
                    __m256 valueChunk   = _mm256_loadu_ps(valueVec + d);

                    /* Multiply value by weight and add to context */
                    __m256 weightedValue = _mm256_mul_ps(valueChunk, weightVec);
                    contextChunk         = _mm256_add_ps(contextChunk, weightedValue);

                    /* Store result back to context */
                    _mm256_storeu_ps(contextVec + d, contextChunk);
                }
            }
        }
    }

    return 0;
}
#endif

/**
 * SIMD-accelerated attention context computation using SSE2
 */
#if defined(HAS_SSE2_SUPPORT)
static int tinyaiSimdAttentionContextSSE2(const float *softmaxScores, const float *value,
                                          float *context, uint32_t seqLength, uint32_t numHeads,
                                          uint32_t headDim)
{
    /* Process each head */
    for (uint32_t h = 0; h < numHeads; h++) {
        /* Process each query position */
        for (uint32_t i = 0; i < seqLength; i++) {
            /* Get softmax scores for this query position */
            const float *scores = softmaxScores + h * seqLength * seqLength + i * seqLength;

            /* Initialize context vector for this position to zero */
            float *contextVec = context + i * numHeads * headDim + h * headDim;
            for (uint32_t d = 0; d < headDim; d += 4) {
                _mm_storeu_ps(contextVec + d, _mm_setzero_ps());
            }

            /* Compute weighted sum of value vectors */
            for (uint32_t j = 0; j < seqLength; j++) {
                /* Get value vector for this key position */
                const float *valueVec = value + j * numHeads * headDim + h * headDim;

                /* Get attention weight */
                float  weight    = scores[j];
                __m128 weightVec = _mm_set1_ps(weight);

                /* Process in chunks of 4 */
                for (uint32_t d = 0; d < headDim; d += 4) {
                    /* Load current context and value vectors */
                    __m128 contextChunk = _mm_loadu_ps(contextVec + d);
                    __m128 valueChunk   = _mm_loadu_ps(valueVec + d);

                    /* Multiply value by weight and add to context */
                    __m128 weightedValue = _mm_mul_ps(valueChunk, weightVec);
                    contextChunk         = _mm_add_ps(contextChunk, weightedValue);

                    /* Store result back to context */
                    _mm_storeu_ps(contextVec + d, contextChunk);
                }
            }
        }
    }

    return 0;
}
#endif

/**
 * Reference implementation of attention context computation
 */
static int tinyaiAttentionContextReference(const float *softmaxScores, const float *value,
                                           float *context, uint32_t seqLength, uint32_t numHeads,
                                           uint32_t headDim)
{
    /* Process each head */
    for (uint32_t h = 0; h < numHeads; h++) {
        /* Process each query position */
        for (uint32_t i = 0; i < seqLength; i++) {
            /* Get softmax scores for this query position */
            const float *scores = softmaxScores + h * seqLength * seqLength + i * seqLength;

            /* Initialize context vector for this position to zero */
            float *contextVec = context + i * numHeads * headDim + h * headDim;
            for (uint32_t d = 0; d < headDim; d++) {
                contextVec[d] = 0.0f;
            }

            /* Compute weighted sum of value vectors */
            for (uint32_t j = 0; j < seqLength; j++) {
                /* Get value vector for this key position */
                const float *valueVec = value + j * numHeads * headDim + h * headDim;

                /* Get attention weight */
                float weight = scores[j];

                /* Weighted sum */
                for (uint32_t d = 0; d < headDim; d++) {
                    contextVec[d] += weight * valueVec[d];
                }
            }
        }
    }

    return 0;
}

/**
 * SIMD-accelerated attention context computation (public API)
 */
int tinyaiSimdAttentionContext(const float *softmaxScores, const float *value, float *context,
                               uint32_t seqLength, uint32_t numHeads, uint32_t headDim)
{
    /* Check if SIMD is available and which version */
    extern bool g_simdInitialized;
    extern bool g_hasSSE2;
    extern bool g_hasAVX;
    extern bool g_hasAVX2;

    /* Use the most advanced SIMD version available */
#if defined(HAS_AVX2_SUPPORT)
    if (g_hasAVX2) {
        return tinyaiSimdAttentionContextAVX2(softmaxScores, value, context, seqLength, numHeads,
                                              headDim);
    }
#endif

#if defined(HAS_SSE2_SUPPORT)
    if (g_hasSSE2) {
        return tinyaiSimdAttentionContextSSE2(softmaxScores, value, context, seqLength, numHeads,
                                              headDim);
    }
#endif

    /* Fallback to reference implementation */
    return tinyaiAttentionContextReference(softmaxScores, value, context, seqLength, numHeads,
                                           headDim);
}

/**
 * SIMD-accelerated output projection using AVX2
 */
#if defined(HAS_AVX2_SUPPORT)
static int tinyaiSimdOutputProjectionAVX2(const float            *context,
                                          const TinyAIMatrix4bit *outputWeight,
                                          const float *outputBias, float *output,
                                          uint32_t seqLength, uint32_t hiddenDim)
{
    /* Process each sequence position */
    for (uint32_t i = 0; i < seqLength; i++) {
        /* Get context vector for this position */
        const float *contextVec = context + i * hiddenDim;

        /* Get output vector for this position */
        float *outputVec = output + i * hiddenDim;

        /* Use SIMD matrix multiplication for projection */
        float scale = outputWeight->scale;
        tinyaiSimdMatMul4Bit(outputVec, outputWeight->data, contextVec, hiddenDim, hiddenDim,
                             &scale);

        /* Add bias if provided */
        if (outputBias) {
            /* Process in chunks of 8 */
            for (uint32_t j = 0; j < hiddenDim; j += 8) {
                /* Load bias and output vectors */
                __m256 biasVec = _mm256_loadu_ps(outputBias + j);
                __m256 outVec  = _mm256_loadu_ps(outputVec + j);

                /* Add bias to output */
                __m256 resultVec = _mm256_add_ps(outVec, biasVec);

                /* Store result */
                _mm256_storeu_ps(outputVec + j, resultVec);
            }
        }
    }

    return 0;
}
#endif

/**
 * SIMD-accelerated output projection using SSE2
 */
#if defined(HAS_SSE2_SUPPORT)
static int tinyaiSimdOutputProjectionSSE2(const float            *context,
                                          const TinyAIMatrix4bit *outputWeight,
                                          const float *outputBias, float *output,
                                          uint32_t seqLength, uint32_t hiddenDim)
{
    /* Process each sequence position */
    for (uint32_t i = 0; i < seqLength; i++) {
        /* Get context vector for this position */
        const float *contextVec = context + i * hiddenDim;

        /* Get output vector for this position */
        float *outputVec = output + i * hiddenDim;

        /* Use SIMD matrix multiplication for projection */
        float scale = outputWeight->scale;
        tinyaiSimdMatMul4Bit(outputVec, outputWeight->data, contextVec, hiddenDim, hiddenDim,
                             &scale);

        /* Add bias if provided */
        if (outputBias) {
            /* Process in chunks of 4 */
            for (uint32_t j = 0; j < hiddenDim; j += 4) {
                /* Load bias and output vectors */
                __m128 biasVec = _mm_loadu_ps(outputBias + j);
                __m128 outVec  = _mm_loadu_ps(outputVec + j);

                /* Add bias to output */
                __m128 resultVec = _mm_add_ps(outVec, biasVec);

                /* Store result */
                _mm_storeu_ps(outputVec + j, resultVec);
            }
        }
    }

    return 0;
}
#endif

/**
 * Reference implementation of output projection
 */
static int tinyaiOutputProjectionReference(const float            *context,
                                           const TinyAIMatrix4bit *outputWeight,
                                           const float *outputBias, float *output,
                                           uint32_t seqLength, uint32_t hiddenDim)
{
    /* Process each sequence position */
    for (uint32_t i = 0; i < seqLength; i++) {
        /* Get context vector for this position */
        const float *contextVec = context + i * hiddenDim;

        /* Get output vector for this position */
        float *outputVec = output + i * hiddenDim;

        /* Dequantize output weight matrix */
        TinyAIMatrixFP32 *outputWeightFP32 = tinyaiDequantize4bitToFP32(outputWeight);
        if (!outputWeightFP32) {
            return -1;
        }

        /* Matrix multiplication */
        for (uint32_t j = 0; j < hiddenDim; j++) {
            float sum = outputBias ? outputBias[j] : 0.0f;

            for (uint32_t k = 0; k < hiddenDim; k++) {
                sum += contextVec[k] * outputWeightFP32->data[k * hiddenDim + j];
            }

            outputVec[j] = sum;
        }

        /* Clean up */
        tinyaiDestroyMatrixFP32(outputWeightFP32);
    }

    return 0;
}

/**
 * SIMD-accelerated output projection (public API)
 */
int tinyaiSimdOutputProjection(const float *context, const TinyAIMatrix4bit *outputWeight,
                               const float *outputBias, float *output, uint32_t seqLength,
                               uint32_t hiddenDim)
{
    /* Check if SIMD is available and which version */
    extern bool g_simdInitialized;
    extern bool g_hasSSE2;
    extern bool g_hasAVX;
    extern bool g_hasAVX2;

    /* Use the most advanced SIMD version available */
#if defined(HAS_AVX2_SUPPORT)
    if (g_hasAVX2) {
        return tinyaiSimdOutputProjectionAVX2(context, outputWeight, outputBias, output, seqLength,
                                              hiddenDim);
    }
#endif

#if defined(HAS_SSE2_SUPPORT)
    if (g_hasSSE2) {
        return tinyaiSimdOutputProjectionSSE2(context, outputWeight, outputBias, output, seqLength,
                                              hiddenDim);
    }
#endif

    /* Fallback to reference implementation */
    return tinyaiOutputProjectionReference(context, outputWeight, outputBias, output, seqLength,
                                           hiddenDim);
}
