/**
 * @file visual_qa.c
 * @brief Implementation of visual question answering in TinyAI
 */

#include "visual_qa.h"
#include "../../../core/io.h"
#include "../../../models/image/image_model.h"
#include "../../../models/multimodal/multimodal_model.h"
#include "../../../models/text/generate.h"
#include "../../../utils/quantize.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Visual QA structure
 */
struct TinyAIVisualQA {
    TinyAIMultimodalModel *model;           /* Multimodal model */
    TinyAITokenizer       *tokenizer;       /* Tokenizer */
    TinyAIAnswerStyle      answerStyle;     /* Answer style */
    char                  *customTemplate;  /* Custom prompt template */
    int                    maxTokens;       /* Maximum tokens in answer */
    bool                   useQuantization; /* Whether to use quantization */
    bool                   useSIMD;         /* Whether to use SIMD */
    int                    imageWidth;      /* Image width */
    int                    imageHeight;     /* Image height */
};

/* Predefined prompt templates for different answer styles */
static const char *TEMPLATE_CONCISE =
    "Answer the following question about this image briefly and directly: ";
static const char *TEMPLATE_DETAILED = "Answer the following question about this image in detail: ";
static const char *TEMPLATE_FACTUAL =
    "Answer the following question about this image with factual information only: ";
static const char *TEMPLATE_CASUAL =
    "Answer this question about the image in a friendly, conversational way: ";

/**
 * Create a visual QA system
 */
TinyAIVisualQA *tinyaiVisualQACreate(const TinyAIVisualQAConfig *config)
{
    if (!config || !config->modelPath || !config->weightsPath || !config->tokenizerPath) {
        fprintf(stderr, "Invalid visual QA configuration\n");
        return NULL;
    }

    /* Allocate visual QA structure */
    TinyAIVisualQA *vqa = (TinyAIVisualQA *)malloc(sizeof(TinyAIVisualQA));
    if (!vqa) {
        fprintf(stderr, "Failed to allocate visual QA system\n");
        return NULL;
    }

    /* Initialize with defaults */
    memset(vqa, 0, sizeof(TinyAIVisualQA));
    vqa->answerStyle     = config->answerStyle;
    vqa->maxTokens       = config->maxTokens > 0 ? config->maxTokens : 100;
    vqa->useQuantization = config->useQuantization;
    vqa->useSIMD         = config->useSIMD;
    vqa->imageWidth      = config->imageWidth > 0 ? config->imageWidth : 224;
    vqa->imageHeight     = config->imageHeight > 0 ? config->imageHeight : 224;

    /* Set custom template if provided */
    if (config->customTemplate) {
        vqa->customTemplate = strdup(config->customTemplate);
    }

    /* Load tokenizer */
    vqa->tokenizer = tinyaiTokenizerCreate(config->tokenizerPath);
    if (!vqa->tokenizer) {
        fprintf(stderr, "Failed to create tokenizer from %s\n", config->tokenizerPath);
        tinyaiVisualQAFree(vqa);
        return NULL;
    }

    /* Create and initialize multimodal model */
    TinyAIMultimodalModelConfig modelConfig = {.modelPath       = config->modelPath,
                                               .weightsPath     = config->weightsPath,
                                               .tokenizerPath   = config->tokenizerPath,
                                               .imageWidth      = vqa->imageWidth,
                                               .imageHeight     = vqa->imageHeight,
                                               .useQuantization = vqa->useQuantization,
                                               .useSIMD         = vqa->useSIMD};

    vqa->model = tinyaiMultimodalModelCreate(&modelConfig);
    if (!vqa->model) {
        fprintf(stderr, "Failed to create multimodal model\n");
        tinyaiVisualQAFree(vqa);
        return NULL;
    }

    return vqa;
}

/**
 * Free a visual QA system
 */
void tinyaiVisualQAFree(TinyAIVisualQA *vqa)
{
    if (!vqa) {
        return;
    }

    /* Free the model */
    if (vqa->model) {
        tinyaiMultimodalModelFree(vqa->model);
    }

    /* Free the tokenizer */
    if (vqa->tokenizer) {
        tinyaiTokenizerFree(vqa->tokenizer);
    }

    /* Free custom template */
    if (vqa->customTemplate) {
        free(vqa->customTemplate);
    }

    /* Free the VQA structure */
    free(vqa);
}

/**
 * Get prompt template based on answer style
 */
static const char *getPromptTemplate(TinyAIVisualQA *vqa)
{
    switch (vqa->answerStyle) {
    case TINYAI_ANSWER_STYLE_CONCISE:
        return TEMPLATE_CONCISE;
    case TINYAI_ANSWER_STYLE_DETAILED:
        return TEMPLATE_DETAILED;
    case TINYAI_ANSWER_STYLE_FACTUAL:
        return TEMPLATE_FACTUAL;
    case TINYAI_ANSWER_STYLE_CASUAL:
        return TEMPLATE_CASUAL;
    case TINYAI_ANSWER_STYLE_CUSTOM:
        return vqa->customTemplate ? vqa->customTemplate : TEMPLATE_CONCISE;
    default:
        return TEMPLATE_CONCISE;
    }
}

/**
 * Create full prompt from template and question
 */
static char *createFullPrompt(const char *template, const char *question)
{
    size_t templateLen = strlen(template);
    size_t questionLen = strlen(question);

    char *prompt = (char *)malloc(templateLen + questionLen + 1);
    if (!prompt) {
        return NULL;
    }

    strcpy(prompt, template);
    strcat(prompt, question);

    return prompt;
}

/**
 * Answer a question about an image file
 */
bool tinyaiVisualQAAnswerQuestion(TinyAIVisualQA *vqa, const char *imagePath, const char *question,
                                  char *answer, int maxLength)
{
    if (!vqa || !imagePath || !question || !answer || maxLength <= 0) {
        return false;
    }

    /* Load the image */
    TinyAIImage *image = tinyaiImageLoadFromFile(imagePath);
    if (!image) {
        fprintf(stderr, "Failed to load image from %s\n", imagePath);
        return false;
    }

    /* Generate answer */
    bool result = tinyaiVisualQAAnswerQuestionForImage(vqa, image, question, answer, maxLength);

    /* Clean up */
    tinyaiImageFree(image);

    return result;
}

/**
 * Answer a question about an image
 */
bool tinyaiVisualQAAnswerQuestionForImage(TinyAIVisualQA *vqa, const TinyAIImage *image,
                                          const char *question, char *answer, int maxLength)
{
    if (!vqa || !image || !question || !answer || maxLength <= 0 || !vqa->model ||
        !vqa->tokenizer) {
        return false;
    }

    /* Get the prompt template for the current answer style */
    const char *promptTemplate = getPromptTemplate(vqa);

    /* Create full prompt */
    char *fullPrompt = createFullPrompt(promptTemplate, question);
    if (!fullPrompt) {
        fprintf(stderr, "Failed to create prompt\n");
        return false;
    }

    /* Generate the answer using the multimodal model */
    TinyAIMultimodalInferenceParams params;
    memset(&params, 0, sizeof(TinyAIMultimodalInferenceParams));
    params.image       = image;
    params.textPrompt  = fullPrompt;
    params.maxTokens   = vqa->maxTokens;
    params.temperature = 0.5f; /* Keep it more factual than creative */
    params.topP        = 0.9f;

    /* Perform inference */
    char *generatedText = tinyaiMultimodalModelGenerateText(vqa->model, &params);

    /* Clean up prompt */
    free(fullPrompt);

    if (!generatedText) {
        fprintf(stderr, "Failed to generate answer\n");
        return false;
    }

    /* Copy to output buffer with truncation */
    strncpy(answer, generatedText, maxLength - 1);
    answer[maxLength - 1] = '\0';

    /* Clean up */
    free(generatedText);

    return true;
}

/**
 * Set the answer style
 */
bool tinyaiVisualQASetStyle(TinyAIVisualQA *vqa, TinyAIAnswerStyle style,
                            const char *customTemplate)
{
    if (!vqa) {
        return false;
    }

    vqa->answerStyle = style;

    /* Update custom template if provided */
    if (style == TINYAI_ANSWER_STYLE_CUSTOM) {
        /* Free existing custom template if any */
        if (vqa->customTemplate) {
            free(vqa->customTemplate);
            vqa->customTemplate = NULL;
        }

        /* Set new custom template */
        if (customTemplate) {
            vqa->customTemplate = strdup(customTemplate);
            if (!vqa->customTemplate) {
                fprintf(stderr, "Failed to allocate memory for custom template\n");
                return false;
            }
        }
        else {
            /* Fall back to concise if no custom template is provided */
            vqa->answerStyle = TINYAI_ANSWER_STYLE_CONCISE;
        }
    }

    return true;
}

/**
 * Get memory usage statistics
 */
bool tinyaiVisualQAGetMemoryUsage(const TinyAIVisualQA *vqa, size_t *weightMemory,
                                  size_t *activationMemory)
{
    if (!vqa || !weightMemory || !activationMemory) {
        return false;
    }

    /* Get memory usage from the multimodal model */
    if (vqa->model) {
        return tinyaiMultimodalModelGetMemoryUsage(vqa->model, weightMemory, activationMemory);
    }

    /* Fall back to zero if no model is available */
    *weightMemory     = 0;
    *activationMemory = 0;
    return true;
}

/**
 * Enable or disable SIMD acceleration
 */
bool tinyaiVisualQAEnableSIMD(TinyAIVisualQA *vqa, bool enable)
{
    if (!vqa) {
        return false;
    }

    vqa->useSIMD = enable;

    /* Apply to the model if available */
    if (vqa->model) {
        return tinyaiMultimodalModelEnableSIMD(vqa->model, enable);
    }

    return true;
}
