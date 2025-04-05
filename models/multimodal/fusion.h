/**
 * @file fusion.h
 * @brief Cross-modal fusion operations for TinyAI
 *
 * This header defines the fusion operations used to combine
 * features from different modalities in multimodal models.
 */

#ifndef TINYAI_FUSION_H
#define TINYAI_FUSION_H

#include "multimodal_model.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

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
                        float *fusedOutput, int fusedDim);

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
                     float *fusedOutput, int fusedDim);

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
                          float *fusedOutput, int fusedDim);

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
                           bool useQuantization, bool useSIMD);

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
                                bool useQuantization, bool useSIMD);

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
                         const void *weights, const float *bias, bool useQuantization,
                         bool useSIMD);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_FUSION_H */
