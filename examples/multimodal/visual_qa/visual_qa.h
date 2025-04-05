/**
 * @file visual_qa.h
 * @brief Header for visual question answering in TinyAI
 *
 * This header defines the visual question answering API for TinyAI, which allows
 * answering natural language questions about images.
 */

#ifndef TINYAI_VISUAL_QA_H
#define TINYAI_VISUAL_QA_H

#include "../../../models/image/image_model.h"
#include "../../../models/multimodal/multimodal_model.h"
#include "../../../models/text/generate.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Answer styles
 */
typedef enum {
    TINYAI_ANSWER_STYLE_CONCISE,  /* Short, direct answers */
    TINYAI_ANSWER_STYLE_DETAILED, /* More detailed explanations */
    TINYAI_ANSWER_STYLE_FACTUAL,  /* Focus on factual information */
    TINYAI_ANSWER_STYLE_CASUAL,   /* Conversational, friendly style */
    TINYAI_ANSWER_STYLE_CUSTOM    /* Custom style with prompt template */
} TinyAIAnswerStyle;

/**
 * Visual QA configuration
 */
typedef struct {
    /* Model paths */
    const char *modelPath;     /* Path to multimodal model structure */
    const char *weightsPath;   /* Path to model weights */
    const char *tokenizerPath; /* Path to tokenizer vocabulary */

    /* QA parameters */
    TinyAIAnswerStyle answerStyle;    /* Answer generation style */
    const char       *customTemplate; /* Custom prompt template (for CUSTOM style) */
    int               maxTokens;      /* Maximum tokens in answer */

    /* Performance configuration */
    bool useQuantization; /* Whether to use quantization */
    bool useSIMD;         /* Whether to use SIMD acceleration */
    int  imageWidth;      /* Image width for model input */
    int  imageHeight;     /* Image height for model input */
} TinyAIVisualQAConfig;

/**
 * Visual QA handle
 */
typedef struct TinyAIVisualQA TinyAIVisualQA;

/**
 * Create a visual QA system
 *
 * @param config Configuration for the visual QA system
 * @return New visual QA system or NULL on error
 */
TinyAIVisualQA *tinyaiVisualQACreate(const TinyAIVisualQAConfig *config);

/**
 * Free a visual QA system
 *
 * @param vqa Visual QA system to free
 */
void tinyaiVisualQAFree(TinyAIVisualQA *vqa);

/**
 * Answer a question about an image file
 *
 * @param vqa Visual QA system to use
 * @param imagePath Path to image file
 * @param question Text of the question
 * @param answer Buffer to store answer (must be pre-allocated)
 * @param maxLength Maximum length of answer buffer
 * @return True on success, false on failure
 */
bool tinyaiVisualQAAnswerQuestion(TinyAIVisualQA *vqa, const char *imagePath, const char *question,
                                  char *answer, int maxLength);

/**
 * Answer a question about an image
 *
 * @param vqa Visual QA system to use
 * @param image Image to query
 * @param question Text of the question
 * @param answer Buffer to store answer (must be pre-allocated)
 * @param maxLength Maximum length of answer buffer
 * @return True on success, false on failure
 */
bool tinyaiVisualQAAnswerQuestionForImage(TinyAIVisualQA *vqa, const TinyAIImage *image,
                                          const char *question, char *answer, int maxLength);

/**
 * Set the answer style
 *
 * @param vqa Visual QA system to configure
 * @param style Answer style to use
 * @param customTemplate Custom prompt template (for CUSTOM style)
 * @return True on success, false on failure
 */
bool tinyaiVisualQASetStyle(TinyAIVisualQA *vqa, TinyAIAnswerStyle style,
                            const char *customTemplate);

/**
 * Get memory usage statistics
 *
 * @param vqa Visual QA system to query
 * @param weightMemory Output parameter for weight memory (in bytes)
 * @param activationMemory Output parameter for activation memory (in bytes)
 * @return True on success, false on failure
 */
bool tinyaiVisualQAGetMemoryUsage(const TinyAIVisualQA *vqa, size_t *weightMemory,
                                  size_t *activationMemory);

/**
 * Enable or disable SIMD acceleration
 *
 * @param vqa Visual QA system to configure
 * @param enable Whether to enable SIMD
 * @return True on success, false on failure
 */
bool tinyaiVisualQAEnableSIMD(TinyAIVisualQA *vqa, bool enable);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_VISUAL_QA_H */
