/**
 * @file image_utils.h
 * @brief Image utility functions for TinyAI
 */

#ifndef TINYAI_IMAGE_UTILS_H
#define TINYAI_IMAGE_UTILS_H

#include "image_model.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Create a new image
 * @param width Image width
 * @param height Image height
 * @param format Image format
 * @return Newly allocated image, or NULL on failure
 */
TinyAIImage *tinyaiImageCreate(int width, int height, TinyAIImageFormat format);

/**
 * Free an image
 * @param image The image to free
 */
void tinyaiImageFree(TinyAIImage *image);

/**
 * Create a copy of an image
 * @param image The image to copy
 * @return Newly allocated copy of image, or NULL on failure
 */
TinyAIImage *tinyaiImageCopy(const TinyAIImage *image);

/**
 * Resize an image using bilinear interpolation
 * @param image The image to resize
 * @param newWidth New width
 * @param newHeight New height
 * @return Newly allocated resized image, or NULL on failure
 */
TinyAIImage *tinyaiImageResize(const TinyAIImage *image, int newWidth, int newHeight);

/**
 * Convert image to grayscale
 * @param image The image to convert
 * @return Newly allocated grayscale image, or NULL on failure
 */
TinyAIImage *tinyaiImageToGrayscale(const TinyAIImage *image);

/**
 * Convert image to float array
 * @param image The image to convert
 * @param output Float array to store result (must be pre-allocated)
 * @param normalize Whether to normalize to 0-1 range
 * @return true on success, false on failure
 */
bool tinyaiImageToFloatArray(const TinyAIImage *image, float *output, bool normalize);

/**
 * Set default preprocessing parameters
 * @param params The parameters structure to initialize
 */
void tinyaiImagePreprocessParamsDefault(TinyAIImagePreprocessParams *params);

/**
 * Preprocess an image for model input
 * @param image The image to preprocess
 * @param params Preprocessing parameters
 * @return Newly allocated preprocessed image, or NULL on failure
 */
TinyAIImage *tinyaiImagePreprocess(const TinyAIImage                 *image,
                                   const TinyAIImagePreprocessParams *params);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_IMAGE_UTILS_H */