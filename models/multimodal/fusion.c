/**
 * @file fusion.c
 * @brief Implementation of cross-modal fusion operations for TinyAI
 */

#include "fusion.h"
#include "../../utils/simd_ops.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Concatenation-based fusion of multiple modality features
 *
 * @param outputs Array of feature vectors from different modalities
 * @param outDims Array of feature dimensions from different modalities
 * @param numModalities Number of modalities to fuse
 * @param fusedOutput Output buffer for fused features
 * @param fusedDim Dimension of the fused feature space
 * @return true on success, false on failure
 */
bool tinyaiFusionConcat(const float **outputs, const int *outDims, int numModalities,
                        float *fusedOutput, int fusedDim)
{
    if (!outputs || !outDims || !fusedOutput || numModalities <= 0 || fusedDim <= 0) {
        return false;
    }

    /* Calculate expected output dimension */
    int expectedFusedDim = 0;
    for (int i = 0; i < numModalities; i++) {
        expectedFusedDim += outDims[i];
    }

    /* Verify that the provided fusedDim matches the expected dimension */
    if (fusedDim != expectedFusedDim) {
        fprintf(stderr, "Fusion dimension mismatch: expected %d, got %d\n", expectedFusedDim,
                fusedDim);
        return false;
    }

    /* Concatenate feature vectors */
    int offset = 0;
    for (int i = 0; i < numModalities; i++) {
        if (!outputs[i]) {
            fprintf(stderr, "Null input for modality %d\n", i);
            return false;
        }

        /* Copy this modality's features to the appropriate position in fusedOutput */
        memcpy(fusedOutput + offset, outputs[i], outDims[i] * sizeof(float));
        offset += outDims[i];
    }

    return true;
}

/**
 * Addition-based fusion of multiple modality features
 *
 * @param outputs Array of feature vectors from different modalities
 * @param outDims Array of feature dimensions from different modalities
 * @param numModalities Number of modalities to fuse
 * @param fusedOutput Output buffer for fused features
 * @param fusedDim Dimension of the fused feature space
 * @return true on success, false on failure
 */
bool tinyaiFusionAdd(const float **outputs, const int *outDims, int numModalities,
                     float *fusedOutput, int fusedDim)
{
    if (!outputs || !outDims || !fusedOutput || numModalities <= 0 || fusedDim <= 0) {
        return false;
    }

    /* Verify that all modalities have the same dimension */
    for (int i = 0; i < numModalities; i++) {
        if (outDims[i] != fusedDim) {
            fprintf(stderr,
                    "Dimension mismatch for addition fusion: modality %d has dim %d, expected %d\n",
                    i, outDims[i], fusedDim);
            return false;
        }

        if (!outputs[i]) {
            fprintf(stderr, "Null input for modality %d\n", i);
            return false;
        }
    }

    /* Initialize output with zeros */
    memset(fusedOutput, 0, fusedDim * sizeof(float));

    /* Add all modality features */
    for (int i = 0; i < numModalities; i++) {
        for (int j = 0; j < fusedDim; j++) {
            fusedOutput[j] += outputs[i][j];
        }
    }

    return true;
}

/**
 * Multiplication-based fusion of multiple modality features
 *
 * @param outputs Array of feature vectors from different modalities
 * @param outDims Array of feature dimensions from different modalities
 * @param numModalities Number of modalities to fuse
 * @param fusedOutput Output buffer for fused features
 * @param fusedDim Dimension of the fused feature space
 * @return true on success, false on failure
 */
bool tinyaiFusionMultiply(const float **outputs, const int *outDims, int numModalities,
                          float *fusedOutput, int fusedDim)
{
    if (!outputs || !outDims || !fusedOutput || numModalities <= 0 || fusedDim <= 0) {
        return false;
    }

    /* Verify that all modalities have the same dimension */
    for (int i = 0; i < numModalities; i++) {
        if (outDims[i] != fusedDim) {
            fprintf(stderr,
                    "Dimension mismatch for multiplication fusion: modality %d has dim %d, "
                    "expected %d\n",
                    i, outDims[i], fusedDim);
            return false;
        }

        if (!outputs[i]) {
            fprintf(stderr, "Null input for modality %d\n", i);
            return false;
        }
    }

    /* Initialize output with ones */
    for (int j = 0; j < fusedDim; j++) {
        fusedOutput[j] = 1.0f;
    }

    /* Multiply all modality features */
    for (int i = 0; i < numModalities; i++) {
        for (int j = 0; j < fusedDim; j++) {
            fusedOutput[j] *= outputs[i][j];
        }
    }

    return true;
}

/**
 * Softmax function
 */
static void softmax(float *input, int size)
{
    if (!input || size <= 0) {
        return;
    }

    /* Find maximum for numerical stability */
    float maxVal = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > maxVal) {
            maxVal = input[i];
        }
    }

    /* Compute softmax */
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        input[i] = expf(input[i] - maxVal);
        sum += input[i];
    }

    /* Normalize */
    if (sum > 0.0f) {
        for (int i = 0; i < size; i++) {
            input[i] /= sum;
        }
    }
}

/**
 * Attention-based fusion of multiple modality features
 *
 * @param outputs Array of feature vectors from different modalities
 * @param outDims Array of feature dimensions from different modalities
 * @param numModalities Number of modalities to fuse
 * @param weights Attention weights for each modality (can be NULL for equal weighting)
 * @param fusedOutput Output buffer for fused features
 * @param fusedDim Dimension of the fused feature space
 * @param useQuantization Whether to use 4-bit quantized weights
 * @param useSIMD Whether to use SIMD acceleration
 * @return true on success, false on failure
 */
bool tinyaiFusionAttention(const float **outputs, const int *outDims, int numModalities,
                           const float *weights, float *fusedOutput, int fusedDim,
                           bool useQuantization, bool useSIMD)
{
    if (!outputs || !outDims || !fusedOutput || numModalities <= 0 || fusedDim <= 0) {
        return false;
    }

    /* Verify that all modalities have the same dimension */
    for (int i = 0; i < numModalities; i++) {
        if (outDims[i] != fusedDim) {
            fprintf(
                stderr,
                "Dimension mismatch for attention fusion: modality %d has dim %d, expected %d\n", i,
                outDims[i], fusedDim);
            return false;
        }

        if (!outputs[i]) {
            fprintf(stderr, "Null input for modality %d\n", i);
            return false;
        }
    }

    /* If weights are provided, use them directly */
    if (weights) {
        /* Initialize output with zeros */
        memset(fusedOutput, 0, fusedDim * sizeof(float));

        /* Apply attention weights */
        for (int i = 0; i < numModalities; i++) {
            for (int j = 0; j < fusedDim; j++) {
                fusedOutput[j] += weights[i] * outputs[i][j];
            }
        }
    }
    else {
        /* Compute attention weights based on feature magnitudes */
        float *attentionWeights = (float *)malloc(numModalities * sizeof(float));
        if (!attentionWeights) {
            fprintf(stderr, "Failed to allocate attention weights\n");
            return false;
        }

        /* Compute squared magnitudes */
        for (int i = 0; i < numModalities; i++) {
            float squaredMag = 0.0f;
            for (int j = 0; j < outDims[i]; j++) {
                squaredMag += outputs[i][j] * outputs[i][j];
            }
            attentionWeights[i] = squaredMag;
        }

        /* Apply softmax to get attention weights */
        softmax(attentionWeights, numModalities);

        /* Initialize output with zeros */
        memset(fusedOutput, 0, fusedDim * sizeof(float));

        /* Apply attention weights */
        for (int i = 0; i < numModalities; i++) {
            for (int j = 0; j < fusedDim; j++) {
                fusedOutput[j] += attentionWeights[i] * outputs[i][j];
            }
        }

        free(attentionWeights);
    }

    return true;
}

/**
 * Cross-attention between two modalities
 *
 * Performs cross-attention where each modality attends to the other's features
 *
 * @param output1 Feature vector from modality 1
 * @param dim1 Dimension of modality 1 features
 * @param output2 Feature vector from modality 2
 * @param dim2 Dimension of modality 2 features
 * @param fusedOutput Output buffer for fused features
 * @param fusedDim Dimension of the fused feature space
 * @param weights Attention weights (can be NULL for learned attention)
 * @param useQuantization Whether to use 4-bit quantized weights
 * @param useSIMD Whether to use SIMD acceleration
 * @return true on success, false on failure
 */
bool tinyaiFusionCrossAttention(const float *output1, int dim1, const float *output2, int dim2,
                                float *fusedOutput, int fusedDim, const float *weights,
                                bool useQuantization, bool useSIMD)
{
    if (!output1 || !output2 || !fusedOutput || dim1 <= 0 || dim2 <= 0 || fusedDim <= 0) {
        return false;
    }

    /* Compute attention matrix: dim1 x dim2 */
    float *attentionMatrix = (float *)malloc(dim1 * dim2 * sizeof(float));
    if (!attentionMatrix) {
        fprintf(stderr, "Failed to allocate attention matrix\n");
        return false;
    }

    /* Calculate dot products for attention scores */
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            attentionMatrix[i * dim2 + j] = output1[i] * output2[j];
        }
    }

    /* Apply softmax row-wise */
    for (int i = 0; i < dim1; i++) {
        softmax(&attentionMatrix[i * dim2], dim2);
    }

    /* Compute attention-weighted features */
    float *attended1 = (float *)malloc(dim1 * sizeof(float));
    float *attended2 = (float *)malloc(dim2 * sizeof(float));
    if (!attended1 || !attended2) {
        fprintf(stderr, "Failed to allocate attended features\n");
        free(attentionMatrix);
        if (attended1)
            free(attended1);
        if (attended2)
            free(attended2);
        return false;
    }

    /* Initialize to zeros */
    memset(attended1, 0, dim1 * sizeof(float));
    memset(attended2, 0, dim2 * sizeof(float));

    /* Compute attended features */
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            attended1[i] += attentionMatrix[i * dim2 + j] * output2[j];
        }
    }

    /* Compute attention from modality 2 to modality 1 */
    float *attentionMatrix2 = (float *)malloc(dim2 * dim1 * sizeof(float));
    if (!attentionMatrix2) {
        fprintf(stderr, "Failed to allocate second attention matrix\n");
        free(attentionMatrix);
        free(attended1);
        free(attended2);
        return false;
    }

    /* Transpose the attention matrix */
    for (int i = 0; i < dim2; i++) {
        for (int j = 0; j < dim1; j++) {
            attentionMatrix2[i * dim1 + j] = attentionMatrix[j * dim2 + i];
        }
    }

    /* Apply softmax row-wise */
    for (int i = 0; i < dim2; i++) {
        softmax(&attentionMatrix2[i * dim1], dim1);
    }

    /* Compute attended features for modality 2 */
    for (int i = 0; i < dim2; i++) {
        for (int j = 0; j < dim1; j++) {
            attended2[i] += attentionMatrix2[i * dim1 + j] * output1[j];
        }
    }

    /* Concatenate the attended features */
    if (fusedDim == dim1 + dim2) {
        memcpy(fusedOutput, attended1, dim1 * sizeof(float));
        memcpy(fusedOutput + dim1, attended2, dim2 * sizeof(float));
    }
    else {
        fprintf(stderr, "Fusion dimension mismatch for cross-attention: %d, expected %d\n",
                fusedDim, dim1 + dim2);
        free(attentionMatrix);
        free(attentionMatrix2);
        free(attended1);
        free(attended2);
        return false;
    }

    /* Clean up */
    free(attentionMatrix);
    free(attentionMatrix2);
    free(attended1);
    free(attended2);

    return true;
}

/**
 * Projection of modality features to a common dimension
 *
 * @param input Input feature vector
 * @param inputDim Input feature dimension
 * @param output Output feature vector
 * @param outputDim Output feature dimension
 * @param weights Projection weights
 * @param bias Projection bias (can be NULL)
 * @param useQuantization Whether to use 4-bit quantized weights
 * @param useSIMD Whether to use SIMD acceleration
 * @return true on success, false on failure
 */
bool tinyaiFusionProject(const float *input, int inputDim, float *output, int outputDim,
                         const void *weights, const float *bias, bool useQuantization, bool useSIMD)
{
    if (!input || !output || !weights || inputDim <= 0 || outputDim <= 0) {
        return false;
    }

    /* Initialize output to zeros */
    memset(output, 0, outputDim * sizeof(float));

    if (useQuantization) {
        /* Use 4-bit quantized weights */
        const uint8_t *quantWeights = (const uint8_t *)weights;

        /* Use SIMD if available */
        if (useSIMD) {
            /* Scale factors for dequantization should be passed as part of the weights */
            const float *scaleFactors =
                (const float *)(quantWeights + (inputDim * outputDim + 1) / 2);

            /* Use SIMD acceleration for matrix-vector multiplication */
            tinyaiSimdMatMul4Bit(output, quantWeights, input, outputDim, inputDim, scaleFactors);
        }
        else {
            /* Reference implementation for 4-bit quantized weights */
            const float *scaleFactors =
                (const float *)(quantWeights + (inputDim * outputDim + 1) / 2);

            for (int i = 0; i < outputDim; i++) {
                float sum = 0.0f;
                for (int j = 0; j < inputDim; j += 2) {
                    int byteIdx = (i * inputDim + j) / 2;
                    int blockIdx =
                        (i * inputDim + j) / 32; /* Assuming 16 values per scale factor */

                    /* Extract two 4-bit weights from one byte */
                    uint8_t byte = quantWeights[byteIdx];
                    int8_t  w1   = (int8_t)((byte & 0xF0) >> 4) - 8; /* Convert 4-bit to signed */

                    /* Apply first weight */
                    sum += (float)w1 * scaleFactors[blockIdx] * input[j];

                    /* Apply second weight if we're not at the end */
                    if (j + 1 < inputDim) {
                        int8_t w2 = (int8_t)(byte & 0x0F) - 8; /* Convert 4-bit to signed */
                        sum += (float)w2 * scaleFactors[blockIdx] * input[j + 1];
                    }
                }
                output[i] = sum;
            }
        }
    }
    else {
        /* Use full-precision weights */
        const float *floatWeights = (const float *)weights;

        /* Perform matrix-vector multiplication */
        for (int i = 0; i < outputDim; i++) {
            float sum = 0.0f;
            for (int j = 0; j < inputDim; j++) {
                sum += floatWeights[i * inputDim + j] * input[j];
            }
            output[i] = sum;
        }
    }

    /* Add bias if provided */
    if (bias) {
        for (int i = 0; i < outputDim; i++) {
            output[i] += bias[i];
        }
    }

    return true;
}
