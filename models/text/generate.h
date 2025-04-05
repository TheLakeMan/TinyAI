/**
 * TinyAI Text Generation Header
 * 
 * This header defines the text generation API for TinyAI, allowing
 * loading and running 4-bit quantized neural language models.
 */

#ifndef TINYAI_GENERATE_H
#define TINYAI_GENERATE_H

#include <stdint.h>
#include "tokenizer.h"
#include "../../utils/quantize.h"

/* ----------------- Constants ----------------- */

/* Model type constants */
#define TINYAI_MODEL_TYPE_RNN         0
#define TINYAI_MODEL_TYPE_TRANSFORMER 1

/* Layer type constants */
#define TINYAI_LAYER_EMBEDDING        0
#define TINYAI_LAYER_DENSE            1
#define TINYAI_LAYER_RNN              2
#define TINYAI_LAYER_ATTENTION        3
#define TINYAI_LAYER_LAYERNORM        4
#define TINYAI_LAYER_OUTPUT           5

/* Activation function constants */
#define TINYAI_ACTIVATION_NONE        0
#define TINYAI_ACTIVATION_RELU        1
#define TINYAI_ACTIVATION_SIGMOID     2
#define TINYAI_ACTIVATION_TANH        3
#define TINYAI_ACTIVATION_GELU        4

/* Sampling method constants */
#define TINYAI_SAMPLING_GREEDY        0
#define TINYAI_SAMPLING_TEMPERATURE   1
#define TINYAI_SAMPLING_TOP_K         2
#define TINYAI_SAMPLING_TOP_P         3

/* ----------------- Types ----------------- */

/**
 * Layer type enumeration
 */
typedef uint32_t TinyAILayerType;

/**
 * Activation function enumeration
 */
typedef uint32_t TinyAIActivation;

/**
 * Model layer structure
 */
typedef struct {
    TinyAILayerType type;          /* Layer type */
    TinyAIActivation activation;   /* Activation function */
    uint32_t inputSize;            /* Input size */
    uint32_t outputSize;           /* Output size */
    TinyAIMatrix4bit weights;      /* Layer weights (4-bit quantized) */
    float *biases;                 /* Layer biases */
} TinyAILayer;

/**
 * Model structure
 */
typedef struct {
    uint32_t type;                 /* Model type */
    uint32_t layerCount;           /* Number of layers */
    TinyAILayer *layers;           /* Layers array */
    TinyAITokenizer *tokenizer;    /* Tokenizer */
    uint32_t hiddenSize;           /* Hidden size */
    uint32_t contextSize;          /* Maximum context size */
    float *activations[2];         /* Ping-pong activation buffers */
    int activeBuffer;              /* Active buffer index */
} TinyAIModel;

/**
 * Generation parameters structure
 */
typedef struct {
    int maxTokens;                 /* Maximum tokens to generate */
    uint32_t samplingMethod;       /* Sampling method */
    float temperature;             /* Temperature for sampling */
    uint32_t topK;                 /* Top-K for sampling */
    float topP;                    /* Top-P (nucleus) for sampling */
    uint32_t seed;                 /* Random seed (0 for random) */
    int *promptTokens;             /* Prompt tokens (can be NULL) */
    int promptLength;              /* Prompt length */
} TinyAIGenerationParams;

/* ----------------- API Functions ----------------- */

/**
 * Create a new text generation model
 * 
 * @param type Model type
 * @param hiddenSize Hidden size
 * @param contextSize Context size
 * @param tokenizer Tokenizer (ownership not transferred)
 * @return New model or NULL on error
 */
TinyAIModel* tinyaiCreateModel(uint32_t type, uint32_t hiddenSize, 
                             uint32_t contextSize, TinyAITokenizer *tokenizer);

/**
 * Destroy a text generation model
 * 
 * @param model Model to destroy
 */
void tinyaiDestroyModel(TinyAIModel *model);

/**
 * Add a layer to a model
 * 
 * @param model Model to add to
 * @param type Layer type
 * @param inputSize Input size
 * @param outputSize Output size
 * @param activation Activation function
 * @return 0 on success, non-zero on error
 */
int tinyaiAddLayer(TinyAIModel *model, TinyAILayerType type, 
                 uint32_t inputSize, uint32_t outputSize, 
                 TinyAIActivation activation);

/**
 * Load model weights from a file
 * 
 * @param model Model to load into
 * @param path File path
 * @return 0 on success, non-zero on error
 */
int tinyaiLoadModelWeights(TinyAIModel *model, const char *path);

/**
 * Load a complete model from files
 * 
 * @param modelPath Model structure file path
 * @param weightsPath Model weights file path
 * @param tokenizerPath Tokenizer file path
 * @return Loaded model or NULL on error
 */
TinyAIModel* tinyaiLoadModel(const char *modelPath, const char *weightsPath, 
                           const char *tokenizerPath);

/**
 * Perform a single forward pass through the model
 * 
 * @param model Model to use
 * @param input Input token IDs
 * @param inputLength Number of input tokens
 * @param output Output logits (must be allocated to at least vocab size)
 * @return 0 on success, non-zero on error
 */
int tinyaiModelForward(TinyAIModel *model, const int *input, 
                     int inputLength, float *output);

/**
 * Sample the next token from output probabilities
 * 
 * @param output Output logits from model
 * @param vocabSize Vocabulary size
 * @param params Generation parameters
 * @return Sampled token ID
 */
int tinyaiSampleToken(const float *output, int vocabSize, 
                    const TinyAIGenerationParams *params);

/**
 * Generate text from a model
 * 
 * @param model Model to use
 * @param params Generation parameters
 * @param outputTokens Output token buffer (must be allocated)
 * @param maxOutputTokens Maximum output tokens
 * @return Number of tokens generated
 */
int tinyaiGenerateText(TinyAIModel *model, const TinyAIGenerationParams *params,
                     int *outputTokens, int maxOutputTokens);

/**
 * Convert a model to 4-bit quantization
 * 
 * @param model Model to quantize
 * @return 0 on success, non-zero on error
 */
int tinyaiQuantizeModel(TinyAIModel *model);

#endif /* TINYAI_GENERATE_H */
