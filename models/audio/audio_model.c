/**
 * @file audio_model.c
 * @brief Implementation of audio model functionality for TinyAI
 */

#include "audio_model.h"
#include "../../core/memory.h"
#include "../../utils/quantize.h"
#include "audio_features.h"
#include "audio_utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Audio model structure definition
 */
struct TinyAIAudioModel {
    /* Configuration */
    TinyAIAudioModelConfig config;

    /* Model parameters */
    float   *weights;          /* Weights for the model (non-quantized) */
    uint8_t *quantizedWeights; /* 4-bit quantized weights if enabled */
    float   *biases;           /* Biases for the model */

    /* Model architecture */
    int inputDim;   /* Input dimension */
    int hiddenSize; /* Size of hidden layer */
    int numClasses; /* Number of output classes */
    int numLayers;  /* Number of model layers */

    /* Runtime options */
    bool useQuantization; /* Whether to use 4-bit quantization */
    bool useSIMD;         /* Whether to use SIMD acceleration */

    /* Memory management */
    void *memoryPool;        /* Memory pool for efficient allocation */
    bool  useExternalMemory; /* Whether memory pool is external */
};

/**
 * Create an audio model
 * @param config Configuration for the audio model
 * @return New audio model, or NULL on failure
 */
TinyAIAudioModel *tinyaiAudioModelCreate(const TinyAIAudioModelConfig *config)
{
    if (!config) {
        fprintf(stderr, "Audio model config is NULL\n");
        return NULL;
    }

    /* Allocate model structure */
    TinyAIAudioModel *model = (TinyAIAudioModel *)malloc(sizeof(TinyAIAudioModel));
    if (!model) {
        fprintf(stderr, "Failed to allocate audio model structure\n");
        return NULL;
    }

    /* Initialize to zeros */
    memset(model, 0, sizeof(TinyAIAudioModel));

    /* Copy configuration */
    memcpy(&model->config, config, sizeof(TinyAIAudioModelConfig));

    /* Set model architecture parameters */
    model->hiddenSize      = config->hiddenSize;
    model->numLayers       = config->numLayers;
    model->numClasses      = config->numClasses;
    model->useQuantization = config->use4BitQuantization;
    model->useSIMD         = config->useSIMD;

    /* Calculate input dimension based on feature type */
    switch (config->featuresConfig.type) {
    case TINYAI_AUDIO_FEATURES_MFCC:
        model->inputDim = config->featuresConfig.numCoefficients;
        if (config->featuresConfig.includeDelta) {
            model->inputDim *= 2;
        }
        if (config->featuresConfig.includeDeltaDelta) {
            model->inputDim += config->featuresConfig.numCoefficients;
        }
        break;

    case TINYAI_AUDIO_FEATURES_MEL:
        model->inputDim = config->featuresConfig.numFilters;
        break;

    case TINYAI_AUDIO_FEATURES_SPECTROGRAM:
        /* For spectrogram, input dimension is typically fftSize/2+1 */
        model->inputDim = 512; /* Default to 1024-point FFT */
        break;

    case TINYAI_AUDIO_FEATURES_RAW:
        /* For raw audio, input dimension is the frame length */
        model->inputDim = config->featuresConfig.frameLength;
        break;

    default:
        fprintf(stderr, "Unknown feature type: %d\n", config->featuresConfig.type);
        free(model);
        return NULL;
    }

    /* Calculate memory requirements */
    size_t weightBytes = 0;
    size_t biasBytes   = 0;

    /* Simple feed-forward architecture with uniform layer sizes */
    int layerSizes[32]; /* Up to 32 layers */
    layerSizes[0] = model->inputDim;
    for (int i = 1; i < model->numLayers; i++) {
        layerSizes[i] = model->hiddenSize;
    }
    layerSizes[model->numLayers] = model->numClasses;

    /* Calculate memory for weights */
    for (int i = 0; i < model->numLayers; i++) {
        weightBytes += (size_t)layerSizes[i] * layerSizes[i + 1] * sizeof(float);
        biasBytes += (size_t)layerSizes[i + 1] * sizeof(float);
    }

    /* If using quantization, adjust weight memory */
    if (model->useQuantization) {
        model->quantizedWeights = (uint8_t *)malloc((weightBytes + 1) / 2); /* 4-bit quantization */
        if (!model->quantizedWeights) {
            fprintf(stderr, "Failed to allocate quantized weights\n");
            free(model);
            return NULL;
        }
    }
    else {
        model->weights = (float *)malloc(weightBytes);
        if (!model->weights) {
            fprintf(stderr, "Failed to allocate weights\n");
            free(model);
            return NULL;
        }
    }

    /* Allocate biases */
    model->biases = (float *)malloc(biasBytes);
    if (!model->biases) {
        fprintf(stderr, "Failed to allocate biases\n");
        if (model->useQuantization) {
            free(model->quantizedWeights);
        }
        else {
            free(model->weights);
        }
        free(model);
        return NULL;
    }

    /* Initialize weights and biases */
    if (config->weightsFile) {
        /* Load weights from file - placeholder, would need to be implemented */
        fprintf(stderr, "Loading weights from file not yet implemented\n");
        /* Initialize with random values for now */
        if (!model->useQuantization) {
            for (size_t i = 0; i < weightBytes / sizeof(float); i++) {
                model->weights[i] =
                    ((float)rand() / RAND_MAX) * 0.1f - 0.05f; /* Small random values */
            }
        }
        else {
            /* For quantized weights, first create float weights then quantize */
            float *tempWeights = (float *)malloc(weightBytes);
            if (!tempWeights) {
                fprintf(stderr, "Failed to allocate temporary weights\n");
                free(model->biases);
                if (model->useQuantization) {
                    free(model->quantizedWeights);
                }
                else {
                    free(model->weights);
                }
                free(model);
                return NULL;
            }

            /* Fill with random values */
            for (size_t i = 0; i < weightBytes / sizeof(float); i++) {
                tempWeights[i] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
            }

            /* Quantize */
            tinyaiQuantizeWeights(tempWeights, model->quantizedWeights, weightBytes / sizeof(float),
                                  4);

            /* Clean up */
            free(tempWeights);
        }
    }
    else {
        /* Initialize with random values */
        if (!model->useQuantization) {
            for (size_t i = 0; i < weightBytes / sizeof(float); i++) {
                model->weights[i] =
                    ((float)rand() / RAND_MAX) * 0.1f - 0.05f; /* Small random values */
            }
        }
        else {
            /* For quantized weights, first create float weights then quantize */
            float *tempWeights = (float *)malloc(weightBytes);
            if (!tempWeights) {
                fprintf(stderr, "Failed to allocate temporary weights\n");
                free(model->biases);
                if (model->useQuantization) {
                    free(model->quantizedWeights);
                }
                else {
                    free(model->weights);
                }
                free(model);
                return NULL;
            }

            /* Fill with random values */
            for (size_t i = 0; i < weightBytes / sizeof(float); i++) {
                tempWeights[i] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
            }

            /* Quantize */
            tinyaiQuantizeWeights(tempWeights, model->quantizedWeights, weightBytes / sizeof(float),
                                  4);

            /* Clean up */
            free(tempWeights);
        }
    }

    /* Initialize biases to zeros */
    memset(model->biases, 0, biasBytes);

    return model;
}

/**
 * Free an audio model
 * @param model The model to free
 */
void tinyaiAudioModelFree(TinyAIAudioModel *model)
{
    if (!model) {
        return;
    }

    /* Free weights */
    if (model->useQuantization) {
        free(model->quantizedWeights);
    }
    else {
        free(model->weights);
    }

    /* Free biases */
    free(model->biases);

    /* Free memory pool if owned */
    if (model->memoryPool && !model->useExternalMemory) {
        free(model->memoryPool);
    }

    /* Free model structure */
    free(model);
}

/**
 * Process audio data with the model
 * @param model The audio model to use
 * @param audio The audio data to process
 * @param output The output structure to fill
 * @return true on success, false on failure
 */
bool tinyaiAudioModelProcess(TinyAIAudioModel *model, const TinyAIAudioData *audio,
                             TinyAIAudioModelOutput *output)
{
    if (!model || !audio || !output) {
        return false;
    }

    /* Extract features from audio */
    TinyAIAudioFeatures features;
    memset(&features, 0, sizeof(TinyAIAudioFeatures));

    /* Extract features based on configuration */
    switch (model->config.featuresConfig.type) {
    case TINYAI_AUDIO_FEATURES_MFCC:
        if (!tinyaiAudioExtractMFCC(audio, &model->config.featuresConfig, NULL, &features)) {
            fprintf(stderr, "Failed to extract MFCC features\n");
            return false;
        }
        break;

    case TINYAI_AUDIO_FEATURES_MEL:
        if (!tinyaiAudioExtractMelSpectrogram(audio, &model->config.featuresConfig, NULL,
                                              &features)) {
            fprintf(stderr, "Failed to extract Mel spectrogram features\n");
            return false;
        }
        break;

    case TINYAI_AUDIO_FEATURES_SPECTROGRAM:
        if (!tinyaiAudioExtractSpectrogram(audio, &model->config.featuresConfig, NULL, &features)) {
            fprintf(stderr, "Failed to extract spectrogram features\n");
            return false;
        }
        break;

    case TINYAI_AUDIO_FEATURES_RAW:
        /* For raw audio, just convert samples to float and reshape */
        fprintf(stderr, "Raw audio features not yet implemented\n");
        tinyaiAudioFeaturesFree(&features);
        return false;

    default:
        fprintf(stderr, "Unknown feature type: %d\n", model->config.featuresConfig.type);
        return false;
    }

    /* Initialize output if not already initialized */
    if (!output->logits || !output->probabilities) {
        if (!tinyaiAudioModelOutputInit(output, model->numClasses)) {
            fprintf(stderr, "Failed to initialize audio model output\n");
            tinyaiAudioFeaturesFree(&features);
            return false;
        }
    }

    /* Validate feature dimensions */
    if (features.numFeatures != model->inputDim) {
        fprintf(stderr, "Feature dimension mismatch: got %d, expected %d\n", features.numFeatures,
                model->inputDim);
        tinyaiAudioFeaturesFree(&features);
        return false;
    }

    /* For now, we'll just use a very simple model that averages the features across time */
    /* This is a placeholder for a real model implementation */
    float *avgFeatures = (float *)malloc(model->inputDim * sizeof(float));
    if (!avgFeatures) {
        fprintf(stderr, "Failed to allocate average features\n");
        tinyaiAudioFeaturesFree(&features);
        return false;
    }

    /* Initialize to zeros */
    memset(avgFeatures, 0, model->inputDim * sizeof(float));

    /* Sum features across time */
    for (int t = 0; t < features.numFrames; t++) {
        for (int f = 0; f < features.numFeatures; f++) {
            avgFeatures[f] += features.data[t * features.numFeatures + f];
        }
    }

    /* Divide by number of frames to get average */
    for (int f = 0; f < features.numFeatures; f++) {
        avgFeatures[f] /= features.numFrames;
    }

    /* Apply simple linear transformation to get logits */
    /* This is a placeholder for a real model forward pass */
    for (int c = 0; c < model->numClasses; c++) {
        output->logits[c] = model->biases[c];
        for (int f = 0; f < model->inputDim; f++) {
            if (model->useQuantization) {
                /* For quantized weights, would need dequantization here */
                /* For now, just use a simple random value for demonstration */
                output->logits[c] += avgFeatures[f] * (((float)rand() / RAND_MAX) * 2.0f - 1.0f);
            }
            else {
                /* For full precision, would multiply by weights */
                /* For now, just use a simple random value for demonstration */
                output->logits[c] += avgFeatures[f] * (((float)rand() / RAND_MAX) * 2.0f - 1.0f);
            }
        }
    }

    /* Apply softmax to get probabilities */
    float maxLogit = output->logits[0];
    for (int c = 1; c < model->numClasses; c++) {
        if (output->logits[c] > maxLogit) {
            maxLogit = output->logits[c];
        }
    }

    float sumExp = 0.0f;
    for (int c = 0; c < model->numClasses; c++) {
        output->probabilities[c] = expf(output->logits[c] - maxLogit);
        sumExp += output->probabilities[c];
    }

    for (int c = 0; c < model->numClasses; c++) {
        output->probabilities[c] /= sumExp;
    }

    /* Find predicted class */
    output->predictedClass = 0;
    output->confidence     = output->probabilities[0];
    for (int c = 1; c < model->numClasses; c++) {
        if (output->probabilities[c] > output->confidence) {
            output->predictedClass = c;
            output->confidence     = output->probabilities[c];
        }
    }

    /* Clean up */
    free(avgFeatures);
    tinyaiAudioFeaturesFree(&features);

    return true;
}

/**
 * Initialize audio model output structure
 * @param output The output structure to initialize
 * @param numClasses Number of output classes
 * @return true on success, false on failure
 */
bool tinyaiAudioModelOutputInit(TinyAIAudioModelOutput *output, int numClasses)
{
    if (!output || numClasses <= 0) {
        return false;
    }

    /* Clear the structure */
    memset(output, 0, sizeof(TinyAIAudioModelOutput));

    /* Allocate logits */
    output->logits = (float *)malloc(numClasses * sizeof(float));
    if (!output->logits) {
        fprintf(stderr, "Failed to allocate logits\n");
        return false;
    }

    /* Allocate probabilities */
    output->probabilities = (float *)malloc(numClasses * sizeof(float));
    if (!output->probabilities) {
        fprintf(stderr, "Failed to allocate probabilities\n");
        free(output->logits);
        output->logits = NULL;
        return false;
    }

    return true;
}

/**
 * Free audio model output
 * @param output The output to free
 */
void tinyaiAudioModelOutputFree(TinyAIAudioModelOutput *output)
{
    if (!output) {
        return;
    }

    /* Free logits */
    if (output->logits) {
        free(output->logits);
        output->logits = NULL;
    }

    /* Free probabilities */
    if (output->probabilities) {
        free(output->probabilities);
        output->probabilities = NULL;
    }

    /* Clear other fields */
    output->predictedClass = 0;
    output->confidence     = 0.0f;
}

/**
 * Enable SIMD acceleration for audio model
 * @param model The model to configure
 * @param enable Whether to enable SIMD
 * @return true on success, false on failure
 */
bool tinyaiAudioModelEnableSIMD(TinyAIAudioModel *model, bool enable)
{
    if (!model) {
        return false;
    }

    model->useSIMD = enable;

    return true;
}
