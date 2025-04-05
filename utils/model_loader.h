/**
 * @file model_loader.h
 * @brief Header for model weight loading utilities for TinyAI
 */

#ifndef TINYAI_MODEL_LOADER_H
#define TINYAI_MODEL_LOADER_H

#include "../models/image/image_model.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Load model weights from a file
 * @param model The model to load weights into
 * @param filepath Path to the weights file
 * @param convertPrecision Whether to convert between precisions if needed
 * @return true on success, false on failure
 */
bool tinyaiLoadModelWeights(TinyAIImageModel *model, const char *filepath, bool convertPrecision);

/**
 * Save model weights to a file
 * @param model The model to save weights from
 * @param filepath Path to save the weights to
 * @return true on success, false on failure
 */
bool tinyaiSaveModelWeights(TinyAIImageModel *model, const char *filepath);

/**
 * Generate a TinyAI model weight file from a standard format model (e.g., ONNX, TFLite)
 * @param srcFilepath Source model file path
 * @param destFilepath Destination TinyAI weight file path
 * @param modelType Target model type
 * @param quantize Whether to quantize weights to 4-bit
 * @return true on success, false on failure
 */
bool tinyaiConvertModelWeights(const char *srcFilepath, const char *destFilepath,
                               TinyAIImageModelType modelType, bool quantize);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_MODEL_LOADER_H */
