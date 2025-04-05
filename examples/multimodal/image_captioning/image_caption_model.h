/**
 * @file image_caption_model.h
 * @brief Header for image captioning using multimodal model in TinyAI
 */

#ifndef TINYAI_IMAGE_CAPTION_MODEL_H
#define TINYAI_IMAGE_CAPTION_MODEL_H

#include "../../../models/multimodal/multimodal_model.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Image captioning model configuration
 */
typedef struct {
    int   imageWidth;      /* Input image width */
    int   imageHeight;     /* Input image height */
    int   maxTokenLength;  /* Maximum token length for generated caption */
    int   textEmbedDim;    /* Text embedding dimension */
    int   imageFeatureDim; /* Image feature dimension */
    int   fusionDim;       /* Dimension of fused representation */
    bool  useQuantization; /* Whether to use 4-bit quantization */
    bool  useSIMD;         /* Whether to use SIMD acceleration */
    char *weightsFile;     /* Path to weights file */
    char *vocabFile;       /* Path to vocabulary file */
} TinyAIImageCaptionConfig;

/**
 * Image captioning model handle
 */
typedef struct TinyAIImageCaptionModel TinyAIImageCaptionModel;

/**
 * Create an image captioning model
 *
 * @param config Configuration for the model
 * @return Model handle, or NULL on failure
 */
TinyAIImageCaptionModel *tinyaiImageCaptionModelCreate(const TinyAIImageCaptionConfig *config);

/**
 * Free an image captioning model
 *
 * @param model Model to free
 */
void tinyaiImageCaptionModelFree(TinyAIImageCaptionModel *model);

/**
 * Generate a caption for an image
 *
 * @param model Model to use
 * @param imagePath Path to the image file
 * @param caption Buffer to store the generated caption
 * @param maxLength Maximum length of the caption buffer
 * @return true on success, false on failure
 */
bool tinyaiImageCaptionGenerate(TinyAIImageCaptionModel *model, const char *imagePath,
                                char *caption, int maxLength);

/**
 * Generate a caption for an image directly from image data
 *
 * @param model Model to use
 * @param image Image to caption
 * @param caption Buffer to store the generated caption
 * @param maxLength Maximum length of the caption buffer
 * @return true on success, false on failure
 */
bool tinyaiImageCaptionGenerateFromImage(TinyAIImageCaptionModel *model, const TinyAIImage *image,
                                         char *caption, int maxLength);

/**
 * Get model's memory usage statistics
 *
 * @param model Model to query
 * @param weightMemory Output parameter for weight memory (in bytes)
 * @param activationMemory Output parameter for activation memory (in bytes)
 * @return true on success, false on failure
 */
bool tinyaiImageCaptionModelGetMemoryUsage(const TinyAIImageCaptionModel *model,
                                           size_t *weightMemory, size_t *activationMemory);

/**
 * Enable or disable SIMD acceleration
 *
 * @param model Model to configure
 * @param enable Whether to enable SIMD
 * @return true on success, false on failure
 */
bool tinyaiImageCaptionModelEnableSIMD(TinyAIImageCaptionModel *model, bool enable);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_IMAGE_CAPTION_MODEL_H */
