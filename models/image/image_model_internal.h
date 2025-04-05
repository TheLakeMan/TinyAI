/**
 * @file image_model_internal.h
 * @brief Internal definitions for image model implementation
 *
 * This file contains internal structures and definitions used by
 * the image model implementation. These are not part of the public API.
 */

#ifndef TINYAI_IMAGE_MODEL_INTERNAL_H
#define TINYAI_IMAGE_MODEL_INTERNAL_H

#include "image_model.h"
#include <float.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Layer type definitions */
#define LAYER_TYPE_INPUT 0
#define LAYER_TYPE_CONV 1
#define LAYER_TYPE_DEPTHWISE 2
#define LAYER_TYPE_POOLING 3
#define LAYER_TYPE_DENSE 4
#define LAYER_TYPE_FLATTEN 5
#define LAYER_TYPE_DROPOUT 6
#define LAYER_TYPE_ACTIVATION 7

/* Activation type definitions */
#define ACTIVATION_NONE 0
#define ACTIVATION_RELU 1
#define ACTIVATION_SIGMOID 2
#define ACTIVATION_TANH 3

/**
 * Internal layer structure - matches the definition in image_model.c
 */
typedef struct Layer {
    int  type;
    char name[32];
    int  inputWidth;
    int  inputHeight;
    int  inputChannels;
    int  outputWidth;
    int  outputHeight;
    int  outputChannels;
    int  kernelSize;
    int  stride;
    int  padding;
    int  activation;

    /* Layer weights and parameters */
    uint8_t *weights; /* 4-bit quantized weights (if applicable) */
    float   *biases;  /* Biases for conv/dense layers */
    float   *scales;  /* Scale factors for quantized weights */

    /* Memory requirements */
    size_t weightBytes; /* Size of weights in bytes */
    size_t biasBytes;   /* Size of biases in bytes */
    size_t outputBytes; /* Size of output in bytes */
} Layer;

/**
 * Internal model structure - matches the definition in image_model.c
 */
struct TinyAIImageModel {
    int modelType;
    int inputWidth;
    int inputHeight;
    int inputChannels;
    int numClasses;

    /* Layers */
    Layer layers[50]; /* Assuming MAX_LAYERS = 50 */
    int   numLayers;

    /* Memory */
    void *memoryPool;
    bool  useExternalMemory;
    bool  useSIMD;
    bool  useQuantization;

    /* Labels */
    char **labels;
    int    numLabels;

    /* Preprocessing parameters */
    TinyAIImagePreprocessParams preprocess;
};

/**
 * Forward pass function for image model
 * @param model The model to use
 * @param input Input data (preprocessed image data)
 * @param output Output buffer for classification results
 * @return true on success, false on failure
 */
bool tinyaiImageModelForwardPass(const TinyAIImageModel *model, const float *input, float *output);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_IMAGE_MODEL_INTERNAL_H */
