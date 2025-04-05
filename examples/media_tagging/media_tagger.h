/**
 * @file media_tagger.h
 * @brief Header for media tagging system in TinyAI
 *
 * This header defines the media tagging API for TinyAI, which provides
 * automated tagging of media files (images, audio, text) with relevant metadata.
 */

#ifndef TINYAI_MEDIA_TAGGER_H
#define TINYAI_MEDIA_TAGGER_H

#include "../../models/image/image_model.h"
#include "../../models/multimodal/multimodal_model.h"
#include "../../models/text/generate.h"
#include "../../models/text/tokenizer.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Media types supported by the tagger
 */
typedef enum {
    TINYAI_MEDIA_TYPE_IMAGE,
    TINYAI_MEDIA_TYPE_AUDIO,
    TINYAI_MEDIA_TYPE_TEXT,
    TINYAI_MEDIA_TYPE_UNKNOWN
} TinyAIMediaType;

/**
 * Tag category types
 */
typedef enum {
    TINYAI_TAG_CATEGORY_SCENE,   /* Scene description */
    TINYAI_TAG_CATEGORY_OBJECT,  /* Objects present */
    TINYAI_TAG_CATEGORY_EMOTION, /* Emotional content */
    TINYAI_TAG_CATEGORY_STYLE,   /* Artistic style */
    TINYAI_TAG_CATEGORY_TOPIC,   /* Topics/subjects */
    TINYAI_TAG_CATEGORY_CUSTOM,  /* Custom category */
    TINYAI_TAG_CATEGORY_ALL      /* All categories */
} TinyAITagCategory;

/**
 * Tag structure representing a single tag
 */
typedef struct {
    char             *text;       /* Tag text */
    float             confidence; /* Confidence score (0-1) */
    TinyAITagCategory category;   /* Tag category */
} TinyAITag;

/**
 * Media tagging configuration
 */
typedef struct {
    /* Model paths */
    const char *imageModelPath;      /* Path to image model structure */
    const char *imageWeightsPath;    /* Path to image model weights */
    const char *textModelPath;       /* Path to text model structure */
    const char *textWeightsPath;     /* Path to text model weights */
    const char *tokenizerPath;       /* Path to tokenizer vocabulary */
    const char *multimodalModelPath; /* Path to multimodal model (optional) */

    /* Operation parameters */
    int               maxTags;             /* Maximum number of tags to generate */
    float             confidenceThreshold; /* Minimum confidence to include a tag */
    TinyAITagCategory categories;          /* Categories to include (bitwise OR) */

    /* Performance configuration */
    bool useQuantization; /* Whether to use 4-bit quantization */
    bool useSIMD;         /* Whether to use SIMD acceleration */
    int  imageWidth;      /* Input image width */
    int  imageHeight;     /* Input image height */
    int  maxTextLength;   /* Maximum text length for processing */
} TinyAIMediaTaggerConfig;

/**
 * Media tagger handle
 */
typedef struct TinyAIMediaTagger TinyAIMediaTagger;

/**
 * Create a media tagger
 *
 * @param config Configuration for the tagger
 * @return New tagger or NULL on error
 */
TinyAIMediaTagger *tinyaiMediaTaggerCreate(const TinyAIMediaTaggerConfig *config);

/**
 * Free a media tagger
 *
 * @param tagger Tagger to free
 */
void tinyaiMediaTaggerFree(TinyAIMediaTagger *tagger);

/**
 * Detect media type from file extension
 *
 * @param filepath Path to the media file
 * @return Detected media type
 */
TinyAIMediaType tinyaiMediaTaggerDetectType(const char *filepath);

/**
 * Set categories to include in tagging
 *
 * @param tagger Tagger to configure
 * @param categories Tag categories to include (bitwise OR of TinyAITagCategory values)
 * @return true on success, false on failure
 */
bool tinyaiMediaTaggerSetCategories(TinyAIMediaTagger *tagger, TinyAITagCategory categories);

/**
 * Set confidence threshold for tags
 *
 * @param tagger Tagger to configure
 * @param threshold Minimum confidence threshold (0-1)
 * @return true on success, false on failure
 */
bool tinyaiMediaTaggerSetThreshold(TinyAIMediaTagger *tagger, float threshold);

/**
 * Tag a media file
 *
 * @param tagger Tagger to use
 * @param filepath Path to the media file
 * @param tags Array to store generated tags (must be pre-allocated for maxTags)
 * @param maxTags Maximum number of tags to generate
 * @param mediaType Optional pointer to store detected media type
 * @return Number of tags generated or -1 on error
 */
int tinyaiMediaTaggerTagFile(TinyAIMediaTagger *tagger, const char *filepath, TinyAITag *tags,
                             int maxTags, TinyAIMediaType *mediaType);

/**
 * Tag an image
 *
 * @param tagger Tagger to use
 * @param image Image to tag
 * @param tags Array to store generated tags (must be pre-allocated for maxTags)
 * @param maxTags Maximum number of tags to generate
 * @return Number of tags generated or -1 on error
 */
int tinyaiMediaTaggerTagImage(TinyAIMediaTagger *tagger, const TinyAIImage *image, TinyAITag *tags,
                              int maxTags);

/**
 * Tag text content
 *
 * @param tagger Tagger to use
 * @param text Text to tag
 * @param tags Array to store generated tags (must be pre-allocated for maxTags)
 * @param maxTags Maximum number of tags to generate
 * @return Number of tags generated or -1 on error
 */
int tinyaiMediaTaggerTagText(TinyAIMediaTagger *tagger, const char *text, TinyAITag *tags,
                             int maxTags);

/**
 * Generate description for a tagged media file
 *
 * @param tagger Tagger to use
 * @param tags Array of tags
 * @param numTags Number of tags in the array
 * @param description Buffer to store generated description (must be pre-allocated)
 * @param maxLength Maximum length of the description buffer
 * @param mediaType Type of media for context-appropriate description
 * @return true on success, false on failure
 */
bool tinyaiMediaTaggerGenerateDescription(TinyAIMediaTagger *tagger, const TinyAITag *tags,
                                          int numTags, char *description, int maxLength,
                                          TinyAIMediaType mediaType);

/**
 * Save tags to a file
 *
 * @param tags Array of tags
 * @param numTags Number of tags in the array
 * @param filepath Path to save the tags file
 * @param format Format to save tags in ("txt", "json", "xml")
 * @return true on success, false on failure
 */
bool tinyaiMediaTaggerSaveTags(const TinyAITag *tags, int numTags, const char *filepath,
                               const char *format);

/**
 * Free tag resources
 *
 * @param tags Array of tags
 * @param numTags Number of tags in the array
 */
void tinyaiMediaTaggerFreeTags(TinyAITag *tags, int numTags);

/**
 * Get memory usage statistics
 *
 * @param tagger Tagger to query
 * @param weightMemory Output parameter for weight memory (in bytes)
 * @param activationMemory Output parameter for activation memory (in bytes)
 * @return true on success, false on failure
 */
bool tinyaiMediaTaggerGetMemoryUsage(const TinyAIMediaTagger *tagger, size_t *weightMemory,
                                     size_t *activationMemory);

/**
 * Enable or disable SIMD acceleration
 *
 * @param tagger Tagger to configure
 * @param enable Whether to enable SIMD
 * @return true on success, false on failure
 */
bool tinyaiMediaTaggerEnableSIMD(TinyAIMediaTagger *tagger, bool enable);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_MEDIA_TAGGER_H */
