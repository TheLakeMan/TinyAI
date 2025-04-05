/**
 * @file image_captioner.c
 * @brief Implementation of image captioning in TinyAI
 */

#include "image_captioner.h"
#include "../../../core/io.h"
#include "../../../models/image/image_model.h"
#include "../../../models/multimodal/multimodal_model.h"
#include "../../../models/text/generate.h"
#include "../../../utils/quantize.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Image captioner structure
 */
struct TinyAIImageCaptioner {
    TinyAIMultimodalModel *model;           /* Multimodal model */
    TinyAITokenizer       *tokenizer;       /* Tokenizer */
    TinyAICaptionStyle     captionStyle;    /* Caption style */
    char                  *customPrompt;    /* Custom prompt template */
    int                    maxTokens;       /* Maximum tokens in caption */
    bool                   useQuantization; /* Whether to use quantization */
    bool                   useSIMD;         /* Whether to use SIMD */
    int                    imageWidth;      /* Image width */
    int                    imageHeight;     /* Image height */
};

/* Predefined prompts for different caption styles */
static const char *PROMPT_DESCRIPTIVE = "Describe this image in detail:";
static const char *PROMPT_CONCISE     = "Provide a brief caption for this image:";
static const char *PROMPT_CREATIVE    = "Write a creative and imaginative caption for this image:";
static const char *PROMPT_TECHNICAL   = "Provide a technical description of this image:";

/**
 * Create an image captioner
 */
TinyAIImageCaptioner *tinyaiImageCaptionerCreate(const TinyAIImageCaptionerConfig *config)
{
    if (!config || !config->modelPath || !config->weightsPath || !config->tokenizerPath) {
        fprintf(stderr, "Invalid image captioner configuration\n");
        return NULL;
    }

    /* Allocate captioner structure */
    TinyAIImageCaptioner *captioner = (TinyAIImageCaptioner *)malloc(sizeof(TinyAIImageCaptioner));
    if (!captioner) {
        fprintf(stderr, "Failed to allocate image captioner\n");
        return NULL;
    }

    /* Initialize with defaults */
    memset(captioner, 0, sizeof(TinyAIImageCaptioner));
    captioner->captionStyle    = config->captionStyle;
    captioner->maxTokens       = config->maxTokens > 0 ? config->maxTokens : 100;
    captioner->useQuantization = config->useQuantization;
    captioner->useSIMD         = config->useSIMD;
    captioner->imageWidth      = config->imageWidth > 0 ? config->imageWidth : 224;
    captioner->imageHeight     = config->imageHeight > 0 ? config->imageHeight : 224;

    /* Set custom prompt if provided */
    if (config->customPrompt) {
        captioner->customPrompt = strdup(config->customPrompt);
    }

    /* Load tokenizer */
    captioner->tokenizer = tinyaiTokenizerCreate(config->tokenizerPath);
    if (!captioner->tokenizer) {
        fprintf(stderr, "Failed to create tokenizer from %s\n", config->tokenizerPath);
        tinyaiImageCaptionerFree(captioner);
        return NULL;
    }

    /* Create and initialize multimodal model */
    TinyAIMultimodalModelConfig modelConfig = {.modelPath       = config->modelPath,
                                               .weightsPath     = config->weightsPath,
                                               .tokenizerPath   = config->tokenizerPath,
                                               .imageWidth      = captioner->imageWidth,
                                               .imageHeight     = captioner->imageHeight,
                                               .useQuantization = captioner->useQuantization,
                                               .useSIMD         = captioner->useSIMD};

    captioner->model = tinyaiMultimodalModelCreate(&modelConfig);
    if (!captioner->model) {
        fprintf(stderr, "Failed to create multimodal model\n");
        tinyaiImageCaptionerFree(captioner);
        return NULL;
    }

    return captioner;
}

/**
 * Free an image captioner
 */
void tinyaiImageCaptionerFree(TinyAIImageCaptioner *captioner)
{
    if (!captioner) {
        return;
    }

    /* Free the model */
    if (captioner->model) {
        tinyaiMultimodalModelFree(captioner->model);
    }

    /* Free the tokenizer */
    if (captioner->tokenizer) {
        tinyaiTokenizerFree(captioner->tokenizer);
    }

    /* Free custom prompt */
    if (captioner->customPrompt) {
        free(captioner->customPrompt);
    }

    /* Free the captioner structure */
    free(captioner);
}

/**
 * Get prompt template based on caption style
 */
static const char *getCaptionPrompt(TinyAIImageCaptioner *captioner)
{
    switch (captioner->captionStyle) {
    case TINYAI_CAPTION_STYLE_DESCRIPTIVE:
        return PROMPT_DESCRIPTIVE;
    case TINYAI_CAPTION_STYLE_CONCISE:
        return PROMPT_CONCISE;
    case TINYAI_CAPTION_STYLE_CREATIVE:
        return PROMPT_CREATIVE;
    case TINYAI_CAPTION_STYLE_TECHNICAL:
        return PROMPT_TECHNICAL;
    case TINYAI_CAPTION_STYLE_CUSTOM:
        return captioner->customPrompt ? captioner->customPrompt : PROMPT_DESCRIPTIVE;
    default:
        return PROMPT_DESCRIPTIVE;
    }
}

/**
 * Generate a caption for an image file
 */
bool tinyaiImageCaptionerCaptionFile(TinyAIImageCaptioner *captioner, const char *imagePath,
                                     char *caption, int maxLength)
{
    if (!captioner || !imagePath || !caption || maxLength <= 0) {
        return false;
    }

    /* Load the image */
    TinyAIImage *image = tinyaiImageLoadFromFile(imagePath);
    if (!image) {
        fprintf(stderr, "Failed to load image from %s\n", imagePath);
        return false;
    }

    /* Generate caption */
    bool result = tinyaiImageCaptionerCaptionImage(captioner, image, caption, maxLength);

    /* Clean up */
    tinyaiImageFree(image);

    return result;
}

/**
 * Generate a caption for an image
 */
bool tinyaiImageCaptionerCaptionImage(TinyAIImageCaptioner *captioner, const TinyAIImage *image,
                                      char *caption, int maxLength)
{
    if (!captioner || !image || !caption || maxLength <= 0 || !captioner->model ||
        !captioner->tokenizer) {
        return false;
    }

    /* Get the prompt for the current caption style */
    const char *promptTemplate = getCaptionPrompt(captioner);

    /* Generate the caption using the multimodal model */
    TinyAIMultimodalInferenceParams params;
    memset(&params, 0, sizeof(TinyAIMultimodalInferenceParams));
    params.image       = image;
    params.textPrompt  = promptTemplate;
    params.maxTokens   = captioner->maxTokens;
    params.temperature = 0.7f; /* Balanced creativity and coherence */
    params.topP        = 0.9f; /* Good for captioning */

    /* Perform inference */
    char *generatedText = tinyaiMultimodalModelGenerateText(captioner->model, &params);
    if (!generatedText) {
        fprintf(stderr, "Failed to generate caption\n");
        return false;
    }

    /* Copy to output buffer with truncation */
    strncpy(caption, generatedText, maxLength - 1);
    caption[maxLength - 1] = '\0';

    /* Clean up */
    free(generatedText);

    return true;
}

/**
 * Set the caption style
 */
bool tinyaiImageCaptionerSetStyle(TinyAIImageCaptioner *captioner, TinyAICaptionStyle style,
                                  const char *customPrompt)
{
    if (!captioner) {
        return false;
    }

    captioner->captionStyle = style;

    /* Update custom prompt if provided */
    if (style == TINYAI_CAPTION_STYLE_CUSTOM) {
        /* Free existing custom prompt if any */
        if (captioner->customPrompt) {
            free(captioner->customPrompt);
            captioner->customPrompt = NULL;
        }

        /* Set new custom prompt */
        if (customPrompt) {
            captioner->customPrompt = strdup(customPrompt);
            if (!captioner->customPrompt) {
                fprintf(stderr, "Failed to allocate memory for custom prompt\n");
                return false;
            }
        }
        else {
            /* Fall back to descriptive if no custom prompt is provided */
            captioner->captionStyle = TINYAI_CAPTION_STYLE_DESCRIPTIVE;
        }
    }

    return true;
}

/**
 * Get memory usage statistics
 */
bool tinyaiImageCaptionerGetMemoryUsage(const TinyAIImageCaptioner *captioner, size_t *weightMemory,
                                        size_t *activationMemory)
{
    if (!captioner || !weightMemory || !activationMemory) {
        return false;
    }

    /* Get memory usage from the multimodal model */
    if (captioner->model) {
        return tinyaiMultimodalModelGetMemoryUsage(captioner->model, weightMemory,
                                                   activationMemory);
    }

    /* Fall back to zero if no model is available */
    *weightMemory     = 0;
    *activationMemory = 0;
    return true;
}

/**
 * Enable or disable SIMD acceleration
 */
bool tinyaiImageCaptionerEnableSIMD(TinyAIImageCaptioner *captioner, bool enable)
{
    if (!captioner) {
        return false;
    }

    captioner->useSIMD = enable;

    /* Apply to the model if available */
    if (captioner->model) {
        return tinyaiMultimodalModelEnableSIMD(captioner->model, enable);
    }

    return true;
}
