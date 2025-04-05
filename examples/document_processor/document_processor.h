/**
 * @file document_processor.h
 * @brief Header for document processor in TinyAI
 *
 * This header defines the document processor API for TinyAI, which provides
 * document classification, summarization, and information extraction.
 */

#ifndef TINYAI_DOCUMENT_PROCESSOR_H
#define TINYAI_DOCUMENT_PROCESSOR_H

#include "../../models/text/generate.h"
#include "../../models/text/tokenizer.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Document processor modes
 */
typedef enum {
    TINYAI_DOC_MODE_CLASSIFY,    /* Document classification */
    TINYAI_DOC_MODE_SUMMARIZE,   /* Document summarization */
    TINYAI_DOC_MODE_EXTRACT_INFO /* Information extraction */
} TinyAIDocumentProcessorMode;

/**
 * Document processor configuration
 */
typedef struct {
    TinyAIDocumentProcessorMode mode;            /* Processing mode */
    const char                 *modelPath;       /* Path to model structure */
    const char                 *weightsPath;     /* Path to model weights */
    const char                 *tokenizerPath;   /* Path to tokenizer vocab */
    bool                        useQuantization; /* Whether to use 4-bit quantization */
    bool                        useSIMD;         /* Whether to use SIMD acceleration */
    int                         maxInputLength;  /* Maximum input document length in tokens */
    int                         maxOutputLength; /* Maximum output length in tokens */
    int                         numClasses;      /* Number of classes for classification mode */
    const char                **classLabels;     /* Class labels for classification mode */
} TinyAIDocumentProcessorConfig;

/**
 * Document processor handle
 */
typedef struct TinyAIDocumentProcessor TinyAIDocumentProcessor;

/**
 * Classification result
 */
typedef struct {
    int         classId;    /* Class ID */
    float       confidence; /* Confidence score (0-1) */
    const char *label;      /* Class label (if available) */
} TinyAIDocumentClassResult;

/**
 * Create a document processor
 *
 * @param config Configuration for the processor
 * @return New processor or NULL on error
 */
TinyAIDocumentProcessor *tinyaiDocumentProcessorCreate(const TinyAIDocumentProcessorConfig *config);

/**
 * Free a document processor
 *
 * @param processor Processor to free
 */
void tinyaiDocumentProcessorFree(TinyAIDocumentProcessor *processor);

/**
 * Process a document file
 *
 * @param processor Processor to use
 * @param filePath Path to document file
 * @param outputBuffer Buffer to store output (must be pre-allocated)
 * @param outputSize Size of output buffer
 * @return True on success, false on failure
 */
bool tinyaiDocumentProcessFile(TinyAIDocumentProcessor *processor, const char *filePath,
                               char *outputBuffer, int outputSize);

/**
 * Process document text
 *
 * @param processor Processor to use
 * @param text Document text
 * @param outputBuffer Buffer to store output (must be pre-allocated)
 * @param outputSize Size of output buffer
 * @return True on success, false on failure
 */
bool tinyaiDocumentProcessText(TinyAIDocumentProcessor *processor, const char *text,
                               char *outputBuffer, int outputSize);

/**
 * Classify a document
 *
 * @param processor Processor to use
 * @param text Document text
 * @param results Array to store classification results (must be pre-allocated)
 * @param maxResults Maximum number of results to return
 * @return Number of results or -1 on error
 */
int tinyaiDocumentClassify(TinyAIDocumentProcessor *processor, const char *text,
                           TinyAIDocumentClassResult *results, int maxResults);

/**
 * Summarize a document
 *
 * @param processor Processor to use
 * @param text Document text
 * @param summary Buffer to store summary (must be pre-allocated)
 * @param maxLength Maximum length of summary
 * @return True on success, false on failure
 */
bool tinyaiDocumentSummarize(TinyAIDocumentProcessor *processor, const char *text, char *summary,
                             int maxLength);

/**
 * Extract information from a document
 *
 * @param processor Processor to use
 * @param text Document text
 * @param prompt Specific information to extract (e.g., "Extract all dates")
 * @param result Buffer to store extracted information (must be pre-allocated)
 * @param maxLength Maximum length of result
 * @return True on success, false on failure
 */
bool tinyaiDocumentExtractInfo(TinyAIDocumentProcessor *processor, const char *text,
                               const char *prompt, char *result, int maxLength);

/**
 * Get memory usage statistics
 *
 * @param processor Processor to query
 * @param weightMemory Output parameter for weight memory (in bytes)
 * @param activationMemory Output parameter for activation memory (in bytes)
 * @return True on success, false on failure
 */
bool tinyaiDocumentProcessorGetMemoryUsage(const TinyAIDocumentProcessor *processor,
                                           size_t *weightMemory, size_t *activationMemory);

/**
 * Enable or disable SIMD acceleration
 *
 * @param processor Processor to configure
 * @param enable Whether to enable SIMD
 * @return True on success, false on failure
 */
bool tinyaiDocumentProcessorEnableSIMD(TinyAIDocumentProcessor *processor, bool enable);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_DOCUMENT_PROCESSOR_H */
