/**
 * @file image_captioner.h
 * @brief Header for multimodal image captioning in TinyAI
 */

#ifndef TINYAI_IMAGE_CAPTIONER_H
#define TINYAI_IMAGE_CAPTIONER_H

#include "../../../models/image/image_model.h"
#include "../../../models/multimodal/multimodal_model.h"
#include "../../../models/text/generate.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Fusion method for combining image and text features
 */
typedef enum {
    TINYAI_FUSION_CONCATENATION,  /* Concatenate features */
    TINYAI_FUSION_ADDITION,       /* Add features */
    TINYAI_FUSION_MULTIPLICATION, /* Multiply features */
    TINYAI_FUSION_ATTENTION       /* Use attention mechanism */
} TinyAIFusionMethod;

/**
 * Image captioner configuration
 */
typedef struct {
    /* Model paths */
    const char *visionModelPath;     /* Path to vision model structure */
    const char *visionWeightsPath;   /* Path to vision model weights */
    const char *languageModelPath;   /* Path to language model structure */
    const char *languageWeightsPath; /* Path to language model weights */
    const char *tokenizerPath;       /* Path to tokenizer vocabulary */

    /* Generation parameters */
    int   maxTokens;   /* Maximum tokens in caption */
    int   beamWidth;   /* Beam search width (1 for greedy) */
    float temperature; /* Sampling temperature */

    /* Fusion settings */
    TinyAIFusionMethod fusionMethod; /* Method to fuse image and text features */

    /* Optimization options */
    bool useQuantization; /* Whether to use quantization */
    bool useSIMD;         /* Whether to use SIMD acceleration */
} TinyAICaptionerConfig;

/**
 * Image captioner
 */
typedef struct TinyAIImageCaptioner TinyAIImageCaptioner;

/**
 * Callback for streaming text generation
 * @param token The generated token text
 * @param is_partial Whether this is a partial UTF-8 character
 * @param user_data User-provided data pointer
 * @return True to continue generation, false to stop
 */
typedef bool (*TinyAICaptionerStreamCallback)(const char *token, bool is_partial, void *user_data);

/**
 * Create a new image captioner
 *
 * @param config Captioner configuration
 * @return New captioner or NULL on error
 */
TinyAIImageCaptioner *tinyaiCaptionerCreate(const TinyAICaptionerConfig *config);

/**
 * Free an image captioner
 *
 * @param captioner Captioner to free
 */
void tinyaiCaptionerFree(TinyAIImageCaptioner *captioner);

/**
 * Generate a caption for an image file
 *
 * @param captioner Captioner to use
 * @param imagePath Path to image file
 * @param stream_callback Callback for streaming tokens (NULL for non-streaming)
 * @param user_data User data to pass to the callback
 * @return Generated caption (caller must free) or NULL on error
 */
char *tinyaiGenerateCaption(TinyAIImageCaptioner *captioner, const char *imagePath,
                            TinyAICaptionerStreamCallback stream_callback, void *user_data);

/**
 * Generate a caption for an in-memory image
 *
 * @param captioner Captioner to use
 * @param image Image data
 * @param stream_callback Callback for streaming tokens (NULL for non-streaming)
 * @param user_data User data to pass to the callback
 * @return Generated caption (caller must free) or NULL on error
 */
char *tinyaiGenerateCaptionFromImage(TinyAIImageCaptioner *captioner, const TinyAIImage *image,
                                     TinyAICaptionerStreamCallback stream_callback,
                                     void                         *user_data);

/**
 * Get the encoding time for the vision model in milliseconds
 *
 * @param captioner Captioner
 * @return Encoding time in milliseconds or -1 if not available
 */
double tinyaiCaptionerGetVisionEncodingTime(const TinyAIImageCaptioner *captioner);

/**
 * Get the generation time for the text model in milliseconds
 *
 * @param captioner Captioner
 * @return Generation time in milliseconds or -1 if not available
 */
double tinyaiCaptionerGetTextGenerationTime(const TinyAIImageCaptioner *captioner);

/**
 * Get memory usage statistics
 *
 * @param captioner Captioner
 * @param visionModelMemory Output parameter for vision model memory (in bytes)
 * @param languageModelMemory Output parameter for language model memory (in bytes)
 * @param totalMemory Output parameter for total memory (in bytes)
 * @return True on success, false on failure
 */
bool tinyaiCaptionerGetMemoryUsage(const TinyAIImageCaptioner *captioner, size_t *visionModelMemory,
                                   size_t *languageModelMemory, size_t *totalMemory);

/**
 * Set fusion method for image and text features
 *
 * @param captioner Captioner
 * @param method Fusion method to use
 * @return True on success, false on failure
 */
bool tinyaiCaptionerSetFusionMethod(TinyAIImageCaptioner *captioner, TinyAIFusionMethod method);

/**
 * Set generation parameters
 *
 * @param captioner Captioner
 * @param temperature Sampling temperature (0.0-1.5)
 * @param maxTokens Maximum tokens to generate
 * @param beamWidth Beam search width (1 for greedy search)
 * @return True on success, false on failure
 */
bool tinyaiCaptionerSetGenerationParams(TinyAIImageCaptioner *captioner, float temperature,
                                        int maxTokens, int beamWidth);

/**
 * Enable or disable SIMD acceleration
 *
 * @param captioner Captioner
 * @param enable Whether to enable SIMD
 * @return True on success, false on failure
 */
bool tinyaiCaptionerEnableSIMD(TinyAIImageCaptioner *captioner, bool enable);

/**
 * Convert fusion method string to enum
 *
 * @param methodStr String representation of fusion method
 * @return Fusion method enum value or TINYAI_FUSION_ATTENTION if not recognized
 */
TinyAIFusionMethod tinyaiGetFusionMethodFromString(const char *methodStr);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_IMAGE_CAPTIONER_H */
