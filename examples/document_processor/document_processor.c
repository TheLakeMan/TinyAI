/**
 * @file document_processor.c
 * @brief Implementation of document processor in TinyAI
 */

#include "document_processor.h"
#include "../../core/io.h"
#include "../../core/memory.h"
#include "../../models/text/generate.h"
#include "../../models/text/tokenizer.h"
#include "../../utils/quantize.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Document processor structure
 */
struct TinyAIDocumentProcessor {
    TinyAIDocumentProcessorMode mode;            /* Processing mode */
    TinyAIModel                *model;           /* Text generation model */
    TinyAITokenizer            *tokenizer;       /* Tokenizer */
    int                         maxInputLength;  /* Maximum input document length */
    int                         maxOutputLength; /* Maximum output length */
    int                         numClasses;      /* Number of classes for classification */
    char                      **classLabels;     /* Class labels for classification */
    bool                        useSIMD;         /* Whether SIMD is enabled */
};

/**
 * Create a document processor
 */
TinyAIDocumentProcessor *tinyaiDocumentProcessorCreate(const TinyAIDocumentProcessorConfig *config)
{
    if (!config || !config->modelPath || !config->weightsPath || !config->tokenizerPath) {
        fprintf(stderr, "Invalid document processor configuration\n");
        return NULL;
    }

    /* Allocate processor structure */
    TinyAIDocumentProcessor *processor =
        (TinyAIDocumentProcessor *)malloc(sizeof(TinyAIDocumentProcessor));
    if (!processor) {
        fprintf(stderr, "Failed to allocate document processor\n");
        return NULL;
    }

    /* Initialize the processor */
    memset(processor, 0, sizeof(TinyAIDocumentProcessor));
    processor->mode            = config->mode;
    processor->maxInputLength  = config->maxInputLength;
    processor->maxOutputLength = config->maxOutputLength;
    processor->useSIMD         = config->useSIMD;

    /* Load tokenizer */
    processor->tokenizer = tinyaiTokenizerCreate(config->tokenizerPath);
    if (!processor->tokenizer) {
        fprintf(stderr, "Failed to create tokenizer from %s\n", config->tokenizerPath);
        free(processor);
        return NULL;
    }

    /* Load model */
    processor->model = tinyaiLoadModel(config->modelPath, config->weightsPath, NULL);
    if (!processor->model) {
        fprintf(stderr, "Failed to load model from %s and %s\n", config->modelPath,
                config->weightsPath);
        tinyaiTokenizerFree(processor->tokenizer);
        free(processor);
        return NULL;
    }

    /* If model is quantized and config requests it, quantize the model */
    if (config->useQuantization) {
        if (tinyaiQuantizeModel(processor->model) != 0) {
            fprintf(stderr, "Warning: Failed to quantize model\n");
        }
    }

    /* Set up classification-specific data if needed */
    if (config->mode == TINYAI_DOC_MODE_CLASSIFY && config->numClasses > 0 && config->classLabels) {
        processor->numClasses  = config->numClasses;
        processor->classLabels = (char **)malloc(config->numClasses * sizeof(char *));
        if (!processor->classLabels) {
            fprintf(stderr, "Failed to allocate class labels\n");
            tinyaiDestroyModel(processor->model);
            tinyaiTokenizerFree(processor->tokenizer);
            free(processor);
            return NULL;
        }

        /* Copy class labels */
        for (int i = 0; i < config->numClasses; i++) {
            if (config->classLabels[i]) {
                processor->classLabels[i] = strdup(config->classLabels[i]);
                if (!processor->classLabels[i]) {
                    fprintf(stderr, "Failed to allocate class label %d\n", i);
                    for (int j = 0; j < i; j++) {
                        free(processor->classLabels[j]);
                    }
                    free(processor->classLabels);
                    tinyaiDestroyModel(processor->model);
                    tinyaiTokenizerFree(processor->tokenizer);
                    free(processor);
                    return NULL;
                }
            }
            else {
                processor->classLabels[i] = NULL;
            }
        }
    }

    return processor;
}

/**
 * Free a document processor
 */
void tinyaiDocumentProcessorFree(TinyAIDocumentProcessor *processor)
{
    if (!processor) {
        return;
    }

    /* Free class labels if they exist */
    if (processor->classLabels) {
        for (int i = 0; i < processor->numClasses; i++) {
            free(processor->classLabels[i]);
        }
        free(processor->classLabels);
    }

    /* Free model and tokenizer */
    if (processor->model) {
        tinyaiDestroyModel(processor->model);
    }
    if (processor->tokenizer) {
        tinyaiTokenizerFree(processor->tokenizer);
    }

    /* Free processor structure */
    free(processor);
}

/**
 * Read text from a file
 */
static char *readTextFromFile(const char *filePath)
{
    FILE *file = fopen(filePath, "r");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", filePath);
        return NULL;
    }

    /* Get file size */
    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    /* Allocate buffer */
    char *buffer = (char *)malloc(fileSize + 1);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate buffer for file content\n");
        fclose(file);
        return NULL;
    }

    /* Read file content */
    size_t readSize  = fread(buffer, 1, fileSize, file);
    buffer[readSize] = '\0';

    fclose(file);
    return buffer;
}

/**
 * Process a document file
 */
bool tinyaiDocumentProcessFile(TinyAIDocumentProcessor *processor, const char *filePath,
                               char *outputBuffer, int outputSize)
{
    if (!processor || !filePath || !outputBuffer || outputSize <= 0) {
        return false;
    }

    /* Read file content */
    char *text = readTextFromFile(filePath);
    if (!text) {
        return false;
    }

    /* Process the text */
    bool result = tinyaiDocumentProcessText(processor, text, outputBuffer, outputSize);

    /* Clean up */
    free(text);

    return result;
}

/**
 * Process document text
 */
bool tinyaiDocumentProcessText(TinyAIDocumentProcessor *processor, const char *text,
                               char *outputBuffer, int outputSize)
{
    if (!processor || !text || !outputBuffer || outputSize <= 0) {
        return false;
    }

    /* Call the appropriate function based on the mode */
    switch (processor->mode) {
    case TINYAI_DOC_MODE_CLASSIFY: {
        /* Classify the document and format the result */
        TinyAIDocumentClassResult results[10]; /* Assuming max 10 top results */
        int                       numResults = tinyaiDocumentClassify(processor, text, results, 10);
        if (numResults <= 0) {
            return false;
        }

        /* Format the results */
        int offset = 0;
        offset += snprintf(outputBuffer + offset, outputSize - offset, "Classification Results:\n");
        for (int i = 0; i < numResults && offset < outputSize; i++) {
            offset += snprintf(outputBuffer + offset, outputSize - offset,
                               "  Class: %s (ID: %d), Confidence: %.2f%%\n",
                               results[i].label ? results[i].label : "Unknown", results[i].classId,
                               results[i].confidence * 100.0f);
        }
        return true;
    }

    case TINYAI_DOC_MODE_SUMMARIZE:
        return tinyaiDocumentSummarize(processor, text, outputBuffer, outputSize);

    case TINYAI_DOC_MODE_EXTRACT_INFO:
        /* For extraction mode, we need an additional prompt, so this doesn't work
           directly. The user should call tinyaiDocumentExtractInfo instead. */
        snprintf(outputBuffer, outputSize,
                 "Error: For information extraction, call tinyaiDocumentExtractInfo instead.");
        return false;

    default:
        snprintf(outputBuffer, outputSize, "Error: Unknown document processing mode");
        return false;
    }
}

/**
 * Classify a document
 */
int tinyaiDocumentClassify(TinyAIDocumentProcessor *processor, const char *text,
                           TinyAIDocumentClassResult *results, int maxResults)
{
    if (!processor || !text || !results || maxResults <= 0) {
        return -1;
    }

    /* For classification, we'll use a simple approach where we:
       1. Tokenize the input text
       2. Run the model to get logits for class prediction
       3. Convert logits to probabilities and select top classes
    */

    /* Tokenize input */
    int  textLength = 0;
    int *tokens     = tinyaiTokenizerEncodeText(processor->tokenizer, text, &textLength);
    if (!tokens || textLength == 0) {
        fprintf(stderr, "Failed to tokenize input text\n");
        return -1;
    }

    /* Truncate if needed */
    if (textLength > processor->maxInputLength) {
        textLength = processor->maxInputLength;
    }

    /* Allocate buffer for output logits */
    int    vocabSize    = tinyaiTokenizerGetVocabSize(processor->tokenizer);
    float *outputLogits = (float *)malloc(vocabSize * sizeof(float));
    if (!outputLogits) {
        fprintf(stderr, "Failed to allocate output logits buffer\n");
        free(tokens);
        return -1;
    }

    /* Run the model */
    if (tinyaiModelForward(processor->model, tokens, textLength, outputLogits) != 0) {
        fprintf(stderr, "Failed to run model forward pass\n");
        free(outputLogits);
        free(tokens);
        return -1;
    }

    /* For classification, we'll use the first N logits as class probabilities
       where N is the number of classes */
    int numResults = processor->numClasses < maxResults ? processor->numClasses : maxResults;

    /* Initialize results */
    for (int i = 0; i < numResults; i++) {
        results[i].classId    = i;
        results[i].confidence = 0.0f;
        results[i].label      = processor->classLabels ? processor->classLabels[i] : NULL;
    }

    /* Apply softmax to get probabilities */
    float maxLogit = outputLogits[0];
    for (int i = 1; i < processor->numClasses; i++) {
        if (outputLogits[i] > maxLogit) {
            maxLogit = outputLogits[i];
        }
    }

    float sumExp = 0.0f;
    for (int i = 0; i < processor->numClasses; i++) {
        sumExp += expf(outputLogits[i] - maxLogit);
    }

    for (int i = 0; i < processor->numClasses; i++) {
        float probability = expf(outputLogits[i] - maxLogit) / sumExp;

        /* Find where this class should be inserted in the results array */
        int insertPos = numResults;
        for (int j = 0; j < numResults; j++) {
            if (probability > results[j].confidence) {
                insertPos = j;
                break;
            }
        }

        /* If this class should be in the results, insert it and shift others down */
        if (insertPos < numResults) {
            for (int j = numResults - 1; j > insertPos; j--) {
                results[j] = results[j - 1];
            }
            results[insertPos].classId    = i;
            results[insertPos].confidence = probability;
            results[insertPos].label = processor->classLabels ? processor->classLabels[i] : NULL;
        }
    }

    /* Clean up */
    free(outputLogits);
    free(tokens);

    return numResults;
}

/**
 * Summarize a document
 */
bool tinyaiDocumentSummarize(TinyAIDocumentProcessor *processor, const char *text, char *summary,
                             int maxLength)
{
    if (!processor || !text || !summary || maxLength <= 0) {
        return false;
    }

    /* For summarization, we'll add a prefix prompt to the text */
    const char *summarizePrompt = "Summarize the following text:\n\n";

    /* Calculate total prompt length */
    size_t promptLen  = strlen(summarizePrompt);
    size_t textLen    = strlen(text);
    char  *fullPrompt = (char *)malloc(promptLen + textLen + 1);
    if (!fullPrompt) {
        fprintf(stderr, "Failed to allocate memory for prompt\n");
        return false;
    }

    /* Create full prompt */
    strcpy(fullPrompt, summarizePrompt);
    strcat(fullPrompt, text);

    /* Tokenize input */
    int  promptLength = 0;
    int *promptTokens = tinyaiTokenizerEncodeText(processor->tokenizer, fullPrompt, &promptLength);
    free(fullPrompt);

    if (!promptTokens || promptLength == 0) {
        fprintf(stderr, "Failed to tokenize prompt\n");
        return false;
    }

    /* Truncate if needed */
    if (promptLength > processor->maxInputLength) {
        promptLength = processor->maxInputLength;
    }

    /* Set up generation parameters */
    TinyAIGenerationParams genParams;
    memset(&genParams, 0, sizeof(TinyAIGenerationParams));
    genParams.maxTokens      = processor->maxOutputLength;
    genParams.samplingMethod = TINYAI_SAMPLING_TOP_P;
    genParams.temperature    = 0.7f;
    genParams.topP           = 0.9f;
    genParams.promptTokens   = promptTokens;
    genParams.promptLength   = promptLength;

    /* Allocate buffer for output tokens */
    int *outputTokens = (int *)malloc(processor->maxOutputLength * sizeof(int));
    if (!outputTokens) {
        fprintf(stderr, "Failed to allocate output tokens buffer\n");
        free(promptTokens);
        return false;
    }

    /* Generate summary */
    int numTokens =
        tinyaiGenerateText(processor->model, &genParams, outputTokens, processor->maxOutputLength);

    free(promptTokens);

    if (numTokens <= 0) {
        fprintf(stderr, "Failed to generate summary\n");
        free(outputTokens);
        return false;
    }

    /* Decode output tokens */
    char *summaryText = tinyaiTokenizerDecode(processor->tokenizer, outputTokens, numTokens);
    free(outputTokens);

    if (!summaryText) {
        fprintf(stderr, "Failed to decode summary tokens\n");
        return false;
    }

    /* Copy to output buffer, ensuring we don't exceed maxLength */
    strncpy(summary, summaryText, maxLength - 1);
    summary[maxLength - 1] = '\0';

    free(summaryText);
    return true;
}

/**
 * Extract information from a document
 */
bool tinyaiDocumentExtractInfo(TinyAIDocumentProcessor *processor, const char *text,
                               const char *prompt, char *result, int maxLength)
{
    if (!processor || !text || !prompt || !result || maxLength <= 0) {
        return false;
    }

    /* For extraction, combine the prompt with the text */
    size_t promptLen = strlen(prompt);
    size_t textLen   = strlen(text);

    /* Add newlines and separator */
    const char *separator    = "\n\nDocument:\n\n";
    size_t      separatorLen = strlen(separator);

    char *fullPrompt = (char *)malloc(promptLen + separatorLen + textLen + 1);
    if (!fullPrompt) {
        fprintf(stderr, "Failed to allocate memory for extraction prompt\n");
        return false;
    }

    /* Create full prompt */
    strcpy(fullPrompt, prompt);
    strcat(fullPrompt, separator);
    strcat(fullPrompt, text);

    /* Tokenize input */
    int  promptLength = 0;
    int *promptTokens = tinyaiTokenizerEncodeText(processor->tokenizer, fullPrompt, &promptLength);
    free(fullPrompt);

    if (!promptTokens || promptLength == 0) {
        fprintf(stderr, "Failed to tokenize extraction prompt\n");
        return false;
    }

    /* Truncate if needed */
    if (promptLength > processor->maxInputLength) {
        promptLength = processor->maxInputLength;
    }

    /* Set up generation parameters */
    TinyAIGenerationParams genParams;
    memset(&genParams, 0, sizeof(TinyAIGenerationParams));
    genParams.maxTokens      = processor->maxOutputLength;
    genParams.samplingMethod = TINYAI_SAMPLING_TOP_P;
    genParams.temperature    = 0.7f;
    genParams.topP           = 0.9f;
    genParams.promptTokens   = promptTokens;
    genParams.promptLength   = promptLength;

    /* Allocate buffer for output tokens */
    int *outputTokens = (int *)malloc(processor->maxOutputLength * sizeof(int));
    if (!outputTokens) {
        fprintf(stderr, "Failed to allocate output tokens buffer\n");
        free(promptTokens);
        return false;
    }

    /* Generate extraction */
    int numTokens =
        tinyaiGenerateText(processor->model, &genParams, outputTokens, processor->maxOutputLength);

    free(promptTokens);

    if (numTokens <= 0) {
        fprintf(stderr, "Failed to generate extraction\n");
        free(outputTokens);
        return false;
    }

    /* Decode output tokens */
    char *extractedText = tinyaiTokenizerDecode(processor->tokenizer, outputTokens, numTokens);
    free(outputTokens);

    if (!extractedText) {
        fprintf(stderr, "Failed to decode extraction tokens\n");
        return false;
    }

    /* Copy to output buffer, ensuring we don't exceed maxLength */
    strncpy(result, extractedText, maxLength - 1);
    result[maxLength - 1] = '\0';

    free(extractedText);
    return true;
}

/**
 * Get memory usage statistics
 */
bool tinyaiDocumentProcessorGetMemoryUsage(const TinyAIDocumentProcessor *processor,
                                           size_t *weightMemory, size_t *activationMemory)
{
    if (!processor || !weightMemory || !activationMemory) {
        return false;
    }

    /* For now, we just estimate memory usage based on the model size */
    /* In a real implementation, we would query the model for actual usage */
    *weightMemory     = processor->model ? tinyaiGetModelWeightMemory(processor->model) : 0;
    *activationMemory = processor->model ? tinyaiGetModelActivationMemory(processor->model) : 0;

    return true;
}

/**
 * Enable or disable SIMD acceleration
 */
bool tinyaiDocumentProcessorEnableSIMD(TinyAIDocumentProcessor *processor, bool enable)
{
    if (!processor) {
        return false;
    }

    processor->useSIMD = enable;

    /* Apply SIMD settings to the model */
    /* This would call into the model's SIMD control function in a real implementation */
    /* For now, we just update our flag */

    return true;
}
