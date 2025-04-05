/**
 * @file image_classifier.h
 * @brief Header for image classification in TinyAI
 */

#ifndef TINYAI_IMAGE_CLASSIFIER_H
#define TINYAI_IMAGE_CLASSIFIER_H

#include "../../models/image/image_model.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Prediction result structure
 */
typedef struct {
    int   class_id;   /* Class index */
    float confidence; /* Confidence score (0-1) */
    char *label;      /* Class label (owned by classifier) */
} TinyAIPrediction;

/**
 * Image classifier configuration
 */
typedef struct {
    /* Model paths */
    const char *modelPath;   /* Path to model structure */
    const char *weightsPath; /* Path to model weights */
    const char *labelsPath;  /* Path to class labels file */

    /* Classification parameters */
    int   inputSize;           /* Input image size in pixels */
    int   topK;                /* Number of top predictions to return */
    float confidenceThreshold; /* Minimum confidence for predictions */

    /* Optimization options */
    bool useQuantization; /* Whether to use quantization */
    bool useSIMD;         /* Whether to use SIMD acceleration */
} TinyAIClassifierConfig;

/**
 * Image classifier
 */
typedef struct TinyAIImageClassifier TinyAIImageClassifier;

/**
 * Create a new image classifier
 *
 * @param config Classifier configuration
 * @return New classifier or NULL on error
 */
TinyAIImageClassifier *tinyaiClassifierCreate(const TinyAIClassifierConfig *config);

/**
 * Free an image classifier
 *
 * @param classifier Classifier to free
 */
void tinyaiClassifierFree(TinyAIImageClassifier *classifier);

/**
 * Classify an image file
 *
 * @param classifier Classifier to use
 * @param imagePath Path to image file
 * @param predictions Output array to store predictions
 * @param maxPredictions Maximum number of predictions to store
 * @param numPredictions Output parameter for number of predictions stored
 * @return True on success, false on failure
 */
bool tinyaiClassifyImage(TinyAIImageClassifier *classifier, const char *imagePath,
                         TinyAIPrediction *predictions, int maxPredictions, int *numPredictions);

/**
 * Classify an in-memory image
 *
 * @param classifier Classifier to use
 * @param image Image data
 * @param predictions Output array to store predictions
 * @param maxPredictions Maximum number of predictions to store
 * @param numPredictions Output parameter for number of predictions stored
 * @return True on success, false on failure
 */
bool tinyaiClassifyImageData(TinyAIImageClassifier *classifier, const TinyAIImage *image,
                             TinyAIPrediction *predictions, int maxPredictions,
                             int *numPredictions);

/**
 * Get the last inference time in milliseconds
 *
 * @param classifier Classifier
 * @return Inference time in milliseconds or -1 if not available
 */
double tinyaiClassifierGetInferenceTime(const TinyAIImageClassifier *classifier);

/**
 * Get memory usage statistics
 *
 * @param classifier Classifier
 * @param modelMemory Output parameter for model memory (in bytes)
 * @param totalMemory Output parameter for total memory (in bytes)
 * @return True on success, false on failure
 */
bool tinyaiClassifierGetMemoryUsage(const TinyAIImageClassifier *classifier, size_t *modelMemory,
                                    size_t *totalMemory);

/**
 * Format prediction as a string
 *
 * @param prediction Prediction to format
 * @param buffer Output buffer to store formatted string
 * @param bufferSize Size of output buffer
 * @return True on success, false on failure
 */
bool tinyaiFormatPrediction(const TinyAIPrediction *prediction, char *buffer, size_t bufferSize);

/**
 * Enable or disable SIMD acceleration
 *
 * @param classifier Classifier
 * @param enable Whether to enable SIMD
 * @return True on success, false on failure
 */
bool tinyaiClassifierEnableSIMD(TinyAIImageClassifier *classifier, bool enable);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_IMAGE_CLASSIFIER_H */
