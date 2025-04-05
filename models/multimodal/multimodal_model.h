/**
 * @file multimodal_model.h
 * @brief Public API for multimodal model functionality in TinyAI
 *
 * This header defines the public API for multimodal models in TinyAI,
 * which can process inputs from multiple modalities (text, image, etc.)
 * and produce outputs that combine information from all modalities.
 */

#ifndef TINYAI_MULTIMODAL_MODEL_H
#define TINYAI_MULTIMODAL_MODEL_H

#include "../image/image_model.h"
#include "../text/generate.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Multimodal model type enumeration
 */
typedef enum {
    TINYAI_MULTIMODAL_FUSION,     /* Simple fusion of modalities */
    TINYAI_MULTIMODAL_CROSS_ATTN, /* Cross-attention between modalities */
    TINYAI_MULTIMODAL_CUSTOM      /* Custom multimodal architecture */
} TinyAIMultimodalModelType;

/**
 * Multimodal fusion method enumeration
 */
typedef enum {
    TINYAI_FUSION_CONCAT,   /* Concatenation of features */
    TINYAI_FUSION_ADD,      /* Addition of features */
    TINYAI_FUSION_MULTIPLY, /* Multiplication of features */
    TINYAI_FUSION_ATTENTION /* Attention-based fusion */
} TinyAIFusionMethod;

/**
 * Supported input modalities
 */
typedef enum {
    TINYAI_MODALITY_TEXT,  /* Text modality */
    TINYAI_MODALITY_IMAGE, /* Image modality */
    TINYAI_MODALITY_AUDIO  /* Audio modality (future support) */
} TinyAIModality;

/**
 * Input modality configuration
 */
typedef struct {
    TinyAIModality modality; /* Type of modality */
    union {
        struct {
            int maxTokens; /* Maximum number of tokens for text */
            int embedDim;  /* Embedding dimension for text */
        } text;
        struct {
            int width;    /* Image width */
            int height;   /* Image height */
            int channels; /* Image channels */
        } image;
        struct {
            int sampleRate; /* Audio sample rate */
            int duration;   /* Audio duration in seconds */
        } audio;
    } config;
} TinyAIModalityConfig;

/**
 * Forward declaration of multimodal model struct
 */
typedef struct TinyAIMultimodalModel TinyAIMultimodalModel;

/**
 * Multimodal input container
 */
typedef struct {
    void *textInput;  /* Text input (token IDs) */
    int   textLength; /* Number of tokens */

    TinyAIImage *imageInput; /* Image input */

    void *audioInput;  /* Audio input (for future use) */
    int   audioLength; /* Audio length */
} TinyAIMultimodalInput;

/**
 * Multimodal output container
 */
typedef struct {
    float *embeddings; /* Fused embeddings */
    int    embedDim;   /* Embedding dimension */
    int    length;     /* Number of embedding vectors */

    float *textLogits; /* Text output logits (if applicable) */
    int    vocabSize;  /* Size of vocabulary (if applicable) */

    float *imageFeatures; /* Image features (if applicable) */
    int    numClasses;    /* Number of image classes (if applicable) */
} TinyAIMultimodalOutput;

/**
 * Multimodal model parameters for creation
 */
typedef struct {
    TinyAIMultimodalModelType modelType;       /* Type of multimodal model */
    TinyAIModalityConfig     *modalityConfigs; /* Array of modality configurations */
    int                       numModalities;   /* Number of modalities */
    TinyAIFusionMethod        fusionMethod;    /* Method for fusing modalities */
    int                       fusionDim;       /* Dimension of fused representation */
    int                       numLayers;       /* Number of fusion layers */
    const char               *weightsFile;     /* Path to weights file (optional) */
    bool                      useQuantization; /* Whether to use 4-bit quantization */
    bool                      useSIMD;         /* Whether to use SIMD acceleration */
    void                     *customParams;    /* Custom parameters */
} TinyAIMultimodalModelParams;

/**
 * Create a multimodal model
 * @param params Parameters for model creation
 * @return Newly allocated model, or NULL on failure
 */
TinyAIMultimodalModel *tinyaiMultimodalModelCreate(const TinyAIMultimodalModelParams *params);

/**
 * Free a multimodal model
 * @param model The model to free
 */
void tinyaiMultimodalModelFree(TinyAIMultimodalModel *model);

/**
 * Process multimodal input
 * @param model The model to use
 * @param input Multimodal input containing different modalities
 * @param output Output structure to store results (must be pre-allocated)
 * @return true on success, false on failure
 */
bool tinyaiMultimodalModelProcess(TinyAIMultimodalModel *model, const TinyAIMultimodalInput *input,
                                  TinyAIMultimodalOutput *output);

/**
 * Initialize multimodal input
 * @param input Multimodal input structure to initialize
 * @return true on success, false on failure
 */
bool tinyaiMultimodalInputInit(TinyAIMultimodalInput *input);

/**
 * Free multimodal input
 * @param input Multimodal input to free
 * @param freeContents Whether to free the contained inputs
 */
void tinyaiMultimodalInputFree(TinyAIMultimodalInput *input, bool freeContents);

/**
 * Initialize multimodal output
 * @param output Multimodal output structure to initialize
 * @param embedDim Dimension of embeddings
 * @param length Number of embedding vectors
 * @param vocabSize Size of vocabulary (0 if not applicable)
 * @param numClasses Number of image classes (0 if not applicable)
 * @return true on success, false on failure
 */
bool tinyaiMultimodalOutputInit(TinyAIMultimodalOutput *output, int embedDim, int length,
                                int vocabSize, int numClasses);

/**
 * Free multimodal output
 * @param output Multimodal output to free
 */
void tinyaiMultimodalOutputFree(TinyAIMultimodalOutput *output);

/**
 * Set custom memory pool for model
 * @param model The model to set memory pool for
 * @param memoryPool Memory pool to use
 * @return true on success, false on failure
 */
bool tinyaiMultimodalModelSetMemoryPool(TinyAIMultimodalModel *model, void *memoryPool);

/**
 * Enable or disable SIMD acceleration
 * @param model The model to configure
 * @param enable Whether to enable SIMD
 * @return true on success, false on failure
 */
bool tinyaiMultimodalModelEnableSIMD(TinyAIMultimodalModel *model, bool enable);

/**
 * Get memory usage statistics
 * @param model The model to query
 * @param weightMemory Output parameter for weight memory (in bytes)
 * @param activationMemory Output parameter for activation memory (in bytes)
 * @return true on success, false on failure
 */
bool tinyaiMultimodalModelGetMemoryUsage(const TinyAIMultimodalModel *model, size_t *weightMemory,
                                         size_t *activationMemory);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_MULTIMODAL_MODEL_H */
