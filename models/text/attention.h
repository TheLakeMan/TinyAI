/**
 * @file attention.h
 * @brief SIMD-accelerated attention mechanism for transformer models
 *
 * This file contains declarations for attention mechanisms with SIMD optimization.
 */

#ifndef TINYAI_ATTENTION_H
#define TINYAI_ATTENTION_H

#include "../../utils/quantize.h"
#include <stdbool.h>
#include <stdint.h>

/**
 * Attention parameters structure
 */
typedef struct {
    uint32_t batchSize;     /* Batch size (usually 1 for inference) */
    uint32_t seqLength;     /* Sequence length */
    uint32_t numHeads;      /* Number of attention heads */
    uint32_t headDim;       /* Dimension of each head */
    uint32_t hiddenDim;     /* Hidden dimension (numHeads * headDim) */
    bool     useCausalMask; /* Whether to use causal masking */
    float    scaleFactor;   /* Scale factor for QK^T product (usually 1/sqrt(headDim)) */
} TinyAIAttentionParams;

/**
 * Self-attention structure
 */
typedef struct {
    TinyAIAttentionParams params;        /* Attention parameters */
    TinyAIMatrix4bit      queryWeight;   /* Query projection weights */
    TinyAIMatrix4bit      keyWeight;     /* Key projection weights */
    TinyAIMatrix4bit      valueWeight;   /* Value projection weights */
    TinyAIMatrix4bit      outputWeight;  /* Output projection weights */
    float                *queryBias;     /* Query projection bias */
    float                *keyBias;       /* Key projection bias */
    float                *valueBias;     /* Value projection bias */
    float                *outputBias;    /* Output projection bias */
    float                *scratchMemory; /* Scratch memory for intermediate results */
} TinyAISelfAttention;

/**
 * Initialize self-attention structure
 *
 * @param attention Attention structure to initialize
 * @param params Attention parameters
 * @return 0 on success, -1 on error
 */
int tinyaiInitSelfAttention(TinyAISelfAttention *attention, const TinyAIAttentionParams *params);

/**
 * Free self-attention resources
 *
 * @param attention Attention structure to free
 */
void tinyaiDestroySelfAttention(TinyAISelfAttention *attention);

/**
 * Set weights for self-attention
 *
 * @param attention Attention structure
 * @param queryWeight Query projection weights (4-bit quantized)
 * @param keyWeight Key projection weights (4-bit quantized)
 * @param valueWeight Value projection weights (4-bit quantized)
 * @param outputWeight Output projection weights (4-bit quantized)
 * @param queryBias Query projection bias
 * @param keyBias Key projection bias
 * @param valueBias Value projection bias
 * @param outputBias Output projection bias
 * @return 0 on success, -1 on error
 */
int tinyaiSetAttentionWeights(TinyAISelfAttention *attention, const TinyAIMatrix4bit *queryWeight,
                              const TinyAIMatrix4bit *keyWeight,
                              const TinyAIMatrix4bit *valueWeight,
                              const TinyAIMatrix4bit *outputWeight, const float *queryBias,
                              const float *keyBias, const float *valueBias,
                              const float *outputBias);

/**
 * Perform self-attention operation with SIMD acceleration
 *
 * @param attention Attention structure
 * @param input Input tensor [seqLength x hiddenDim]
 * @param output Output tensor [seqLength x hiddenDim]
 * @return 0 on success, -1 on error
 */
int tinyaiSelfAttentionForward(TinyAISelfAttention *attention, const float *input, float *output);

/**
 * SIMD-accelerated query-key-value projection
 *
 * @param input Input tensor [seqLength x hiddenDim]
 * @param queryWeight Query projection weights (4-bit quantized)
 * @param keyWeight Key projection weights (4-bit quantized)
 * @param valueWeight Value projection weights (4-bit quantized)
 * @param queryBias Query projection bias
 * @param keyBias Key projection bias
 * @param valueBias Value projection bias
 * @param query Output query tensor [seqLength x (numHeads*headDim)]
 * @param key Output key tensor [seqLength x (numHeads*headDim)]
 * @param value Output value tensor [seqLength x (numHeads*headDim)]
 * @param seqLength Sequence length
 * @param hiddenDim Hidden dimension
 * @param numHeads Number of attention heads
 * @param headDim Dimension of each head
 * @return 0 on success, -1 on error
 */
int tinyaiSimdQKVProjection(const float *input, const TinyAIMatrix4bit *queryWeight,
                            const TinyAIMatrix4bit *keyWeight, const TinyAIMatrix4bit *valueWeight,
                            const float *queryBias, const float *keyBias, const float *valueBias,
                            float *query, float *key, float *value, uint32_t seqLength,
                            uint32_t hiddenDim, uint32_t numHeads, uint32_t headDim);

/**
 * SIMD-accelerated attention score computation (Q*K^T)
 *
 * @param query Query tensor [seqLength x (numHeads*headDim)]
 * @param key Key tensor [seqLength x (numHeads*headDim)]
 * @param scores Output scores tensor [numHeads x seqLength x seqLength]
 * @param seqLength Sequence length
 * @param numHeads Number of attention heads
 * @param headDim Dimension of each head
 * @param scaleFactor Scale factor (usually 1/sqrt(headDim))
 * @param useCausalMask Whether to use causal masking
 * @return 0 on success, -1 on error
 */
int tinyaiSimdAttentionScores(const float *query, const float *key, float *scores,
                              uint32_t seqLength, uint32_t numHeads, uint32_t headDim,
                              float scaleFactor, bool useCausalMask);

/**
 * SIMD-accelerated softmax computation for attention scores
 *
 * @param scores Attention scores [numHeads x seqLength x seqLength]
 * @param softmaxScores Output softmax scores [numHeads x seqLength x seqLength]
 * @param seqLength Sequence length
 * @param numHeads Number of attention heads
 * @return 0 on success, -1 on error
 */
int tinyaiSimdAttentionSoftmax(const float *scores, float *softmaxScores, uint32_t seqLength,
                               uint32_t numHeads);

/**
 * SIMD-accelerated attention context computation (softmax(Q*K^T)*V)
 *
 * @param softmaxScores Softmax scores [numHeads x seqLength x seqLength]
 * @param value Value tensor [seqLength x (numHeads*headDim)]
 * @param context Output context tensor [seqLength x (numHeads*headDim)]
 * @param seqLength Sequence length
 * @param numHeads Number of attention heads
 * @param headDim Dimension of each head
 * @return 0 on success, -1 on error
 */
int tinyaiSimdAttentionContext(const float *softmaxScores, const float *value, float *context,
                               uint32_t seqLength, uint32_t numHeads, uint32_t headDim);

/**
 * SIMD-accelerated output projection
 *
 * @param context Context tensor [seqLength x (numHeads*headDim)]
 * @param outputWeight Output projection weights (4-bit quantized)
 * @param outputBias Output projection bias
 * @param output Output tensor [seqLength x hiddenDim]
 * @param seqLength Sequence length
 * @param hiddenDim Hidden dimension
 * @return 0 on success, -1 on error
 */
int tinyaiSimdOutputProjection(const float *context, const TinyAIMatrix4bit *outputWeight,
                               const float *outputBias, float *output, uint32_t seqLength,
                               uint32_t hiddenDim);

#endif /* TINYAI_ATTENTION_H */
