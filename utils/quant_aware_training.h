/**
 * @file quant_aware_training.h
 * @brief Quantization-aware training utilities for TinyAI
 *
 * This header provides utilities for implementing quantization-aware training,
 * which helps models maintain accuracy when quantized post-training.
 */

#ifndef TINYAI_QUANT_AWARE_TRAINING_H
#define TINYAI_QUANT_AWARE_TRAINING_H

#include "quantize.h"
#include "quantize_mixed.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Configuration for quantization-aware training
 */
typedef struct {
    TinyAIPrecisionType weightPrecision;           /* Precision for weights during training */
    TinyAIPrecisionType activationPrecision;       /* Precision for activations during training */
    bool                useSymmetricQuantization;  /* Whether to use symmetric quantization */
    bool                usePerChannelQuantization; /* Whether to use per-channel quantization */
    float               learningRate;              /* Learning rate for training */
    int                 batchSize;                 /* Batch size for training */
    int                 numEpochs;                 /* Number of training epochs */
    const char         *datasetPath;               /* Path to training dataset */
    const char         *validationDatasetPath;     /* Path to validation dataset */
    bool                enableStraightThroughEstimator; /* Whether to use STE for gradients */
    bool                useNoiseInjection; /* Add noise to simulate quantization effects */
    float               noiseStrength;     /* Strength of injected noise (0.0-1.0) */
} TinyAIQuantAwareTrainingConfig;

/**
 * Initialize a model for quantization-aware training
 *
 * @param modelPath Path to model file
 * @param config Training configuration
 * @return true on success, false on failure
 */
bool tinyaiInitQuantAwareTraining(const char                           *modelPath,
                                  const TinyAIQuantAwareTrainingConfig *config);

/**
 * Train a model with quantization awareness
 *
 * @param modelPath Path to model file
 * @param outputModelPath Path to save trained model
 * @param config Training configuration
 * @return true on success, false on failure
 */
bool tinyaiTrainWithQuantAwareness(const char *modelPath, const char *outputModelPath,
                                   const TinyAIQuantAwareTrainingConfig *config);

/**
 * Create a straight-through estimator for a quantization operation
 * (Used for backpropagating through non-differentiable quantization)
 *
 * @param realValue The real-valued input
 * @param quantizedValue The quantized output
 * @param precision Quantization precision
 * @return Gradient for backpropagation
 */
float tinyaiStraightThroughEstimator(float realValue, float quantizedValue,
                                     TinyAIPrecisionType precision);

/**
 * Simulate quantization effects during training by adding controlled noise
 *
 * @param weights Weight matrix to modify
 * @param numElements Number of elements in the matrix
 * @param precision Target precision for simulation
 * @param strength Noise strength (0.0-1.0)
 * @return true on success, false on failure
 */
bool tinyaiSimulateQuantizationNoise(float *weights, int numElements, TinyAIPrecisionType precision,
                                     float strength);

/**
 * Quantize weights during forward pass in training
 *
 * @param weights Real-valued weights
 * @param numElements Number of elements in the matrix
 * @param precision Target precision
 * @param outQuantized Output buffer for quantized weights (can be same as weights)
 * @return true on success, false on failure
 */
bool tinyaiQuantizeForForwardPass(float *weights, int numElements, TinyAIPrecisionType precision,
                                  float *outQuantized);

/**
 * Evaluate model accuracy with quantization awareness
 *
 * @param modelPath Path to model file
 * @param datasetPath Path to evaluation dataset
 * @param config Quantization configuration
 * @param accuracy Output parameter for accuracy
 * @return true on success, false on failure
 */
bool tinyaiEvaluateQuantizedAccuracy(const char *modelPath, const char *datasetPath,
                                     const TinyAIQuantAwareTrainingConfig *config, float *accuracy);

/**
 * Create a default quantization-aware training configuration
 *
 * @return Default configuration (must be freed with free())
 */
TinyAIQuantAwareTrainingConfig *tinyaiCreateDefaultQuantAwareTrainingConfig(void);

/**
 * Export quantization-aware trained model to TinyAI format
 *
 * @param modelPath Path to trained model
 * @param exportPath Path to save exported model
 * @param config Quantization configuration
 * @return true on success, false on failure
 */
bool tinyaiExportQuantAwareModel(const char *modelPath, const char *exportPath,
                                 const TinyAIQuantAwareTrainingConfig *config);

/**
 * Fine-tune a pre-trained model with quantization awareness
 *
 * @param modelPath Path to pre-trained model
 * @param outputModelPath Path to save fine-tuned model
 * @param config Training configuration
 * @param freezeLayers Number of layers to freeze from the bottom (0 = train all)
 * @return true on success, false on failure
 */
bool tinyaiFineTuneWithQuantAwareness(const char *modelPath, const char *outputModelPath,
                                      const TinyAIQuantAwareTrainingConfig *config,
                                      int                                   freezeLayers);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_QUANT_AWARE_TRAINING_H */
