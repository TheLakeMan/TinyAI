/**
 * @file image_model.h
 * @brief Public API for image model functionality in TinyAI
 */

#ifndef TINYAI_IMAGE_MODEL_H
#define TINYAI_IMAGE_MODEL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Image format enumeration
 */
typedef enum {
    TINYAI_IMAGE_FORMAT_GRAYSCALE,
    TINYAI_IMAGE_FORMAT_RGB,
    TINYAI_IMAGE_FORMAT_BGR,
    TINYAI_IMAGE_FORMAT_RGBA
} TinyAIImageFormat;

/**
 * Image model type enumeration
 */
typedef enum {
    TINYAI_IMAGE_MODEL_TINY_CNN,
    TINYAI_IMAGE_MODEL_MOBILENET,
    TINYAI_IMAGE_MODEL_EFFICIENTNET,
    TINYAI_IMAGE_MODEL_CUSTOM
} TinyAIImageModelType;

/**
 * Image structure
 */
typedef struct {
    int               width;    /* Image width */
    int               height;   /* Image height */
    TinyAIImageFormat format;   /* Image format */
    uint8_t          *data;     /* Pixel data */
    bool              ownsData; /* Whether we own the data (and should free it) */
} TinyAIImage;

/**
 * Image classification result
 */
typedef struct {
    int         classId;    /* Class ID */
    float       confidence; /* Confidence score (0-1) */
    const char *label;      /* Class label (if available) */
} TinyAIImageClassResult;

/**
 * Image preprocessing parameters
 */
typedef struct {
    int   targetWidth;  /* Target width for resizing */
    int   targetHeight; /* Target height for resizing */
    float meanR;        /* Mean value for red channel */
    float meanG;        /* Mean value for green channel */
    float meanB;        /* Mean value for blue channel */
    float stdR;         /* Standard deviation for red channel */
    float stdG;         /* Standard deviation for green channel */
    float stdB;         /* Standard deviation for blue channel */
    bool  centerCrop;   /* Whether to center crop */
    float cropRatio;    /* Ratio for center cropping */
} TinyAIImagePreprocessParams;

/**
 * Image model parameters for creation
 */
typedef struct {
    TinyAIImageModelType modelType;       /* Type of model to create */
    int                  inputWidth;      /* Input width */
    int                  inputHeight;     /* Input height */
    int                  inputChannels;   /* Input channels */
    int                  numClasses;      /* Number of output classes */
    const char          *weightsFile;     /* Path to weights file (optional) */
    const char          *labelsFile;      /* Path to labels file (optional) */
    bool                 useQuantization; /* Whether to use 4-bit quantization */
    bool                 useSIMD;         /* Whether to use SIMD acceleration */
    void                *customParams;    /* Custom parameters (for CUSTOM model type) */
} TinyAIImageModelParams;

/**
 * Forward declaration of image model struct
 */
typedef struct TinyAIImageModel TinyAIImageModel;

/**
 * Create an image model
 * @param params Parameters for model creation
 * @return Newly allocated model, or NULL on failure
 */
TinyAIImageModel *tinyaiImageModelCreate(const TinyAIImageModelParams *params);

/**
 * Free an image model
 * @param model The model to free
 */
void tinyaiImageModelFree(TinyAIImageModel *model);

/**
 * Classify an image
 * @param model The model to use for classification
 * @param image The image to classify
 * @param topK Number of top results to return
 * @param results Array to store results (must be pre-allocated for topK results)
 * @return Number of results on success, negative on failure
 */
int tinyaiImageModelClassify(TinyAIImageModel *model, const TinyAIImage *image, int topK,
                             TinyAIImageClassResult *results);

/**
 * Set custom memory pool for model
 * @param model The model to set memory pool for
 * @param memoryPool Memory pool to use
 * @return true on success, false on failure
 */
bool tinyaiImageModelSetMemoryPool(TinyAIImageModel *model, void *memoryPool);

/**
 * Enable or disable SIMD acceleration
 * @param model The model to configure
 * @param enable Whether to enable SIMD
 * @return true on success, false on failure
 */
bool tinyaiImageModelEnableSIMD(TinyAIImageModel *model, bool enable);

/**
 * Get memory usage statistics
 * @param model The model to query
 * @param weightMemory Output parameter for weight memory (in bytes)
 * @param activationMemory Output parameter for activation memory (in bytes)
 * @return true on success, false on failure
 */
bool tinyaiImageModelGetMemoryUsage(const TinyAIImageModel *model, size_t *weightMemory,
                                    size_t *activationMemory);

/**
 * Get preprocessing parameters
 * @param model The model to query
 * @param params Output parameter for preprocessing parameters
 * @return true on success, false on failure
 */
bool tinyaiImageModelGetPreprocessParams(const TinyAIImageModel      *model,
                                         TinyAIImagePreprocessParams *params);

/**
 * Print model summary
 * @param model The model to print summary for
 */
void tinyaiImageModelPrintSummary(const TinyAIImageModel *model);

/**
 * Get the number of weights in the model
 * @param model The model to query
 * @return Number of weights
 */
size_t tinyaiImageModelGetNumWeights(const TinyAIImageModel *model);

/**
 * Get the number of operations per forward pass
 * @param model The model to query
 * @return Number of operations
 */
size_t tinyaiImageModelGetNumOperations(const TinyAIImageModel *model);

#ifdef __cplusplus
}
#endif

/*
 * Include image utilities
 * This header contains declarations for functions such as:
 * - tinyaiImageCreate
 * - tinyaiImageFree
 * - tinyaiImageCopy
 * - tinyaiImageResize
 * and other image manipulation functions
 */
#include "image_utils.h"

#endif /* TINYAI_IMAGE_MODEL_H */
