/**
 * TinyAI Text Generation Implementation
 *
 * This file implements the text generation model for TinyAI,
 * using 4-bit quantization for extreme memory efficiency.
 */

#include "generate.h"
#include "../../core/config.h"
#include "../../core/memory.h"
#include "../../utils/quantize.h"
#include "tokenizer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ----------------- Internal Constants and Definitions ----------------- */

/* Block size for matrix multiplication */
#define BLOCK_SIZE 32

/* KV cache entry */
typedef struct {
    float *key;   /* Key vector */
    float *value; /* Value vector */
} KVCacheEntry;

/* ----------------- Static Variables ----------------- */

/* Random number generator state */
static unsigned int randState = 1;

/* ----------------- Helper Functions ----------------- */

/**
 * Initialize random number generator
 */
static void seedRandom(uint32_t seed) { randState = (seed == 0) ? (unsigned int)time(NULL) : seed; }

/**
 * Generate a random number between 0 and 1
 */
static float randomFloat()
{
    randState = randState * 1664525 + 1013904223; /* Linear congruential generator */
    return (randState & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

/**
 * Apply temperature to logits
 */
static void applyTemperature(float *logits, uint32_t size, float temperature)
{
    if (temperature <= 0.0f) {
        temperature = 1.0f;
    }

    for (uint32_t i = 0; i < size; i++) {
        logits[i] /= temperature;
    }
}

/**
 * Convert logits to probabilities using softmax
 */
static void softmax(float *logits, uint32_t size)
{
    /* Find max for numerical stability */
    float maxLogit = logits[0];
    for (uint32_t i = 1; i < size; i++) {
        if (logits[i] > maxLogit) {
            maxLogit = logits[i];
        }
    }

    /* Compute softmax */
    float sum = 0.0f;
    for (uint32_t i = 0; i < size; i++) {
        logits[i] = expf(logits[i] - maxLogit);
        sum += logits[i];
    }

    /* Normalize */
    if (sum > 0.0f) {
        for (uint32_t i = 0; i < size; i++) {
            logits[i] /= sum;
        }
    }
}

/**
 * Perform top-K sampling
 */
static int sampleTopK(const float *probs, uint32_t size, uint32_t k)
{
    if (k >= size) {
        /* No need for top-K if K is greater than the vocab size */
        float sum = 0.0f;
        for (uint32_t i = 0; i < size; i++) {
            sum += probs[i];
        }

        float r      = randomFloat() * sum;
        float cumSum = 0.0f;

        for (uint32_t i = 0; i < size; i++) {
            cumSum += probs[i];
            if (r < cumSum) {
                return i;
            }
        }

        return size - 1;
    }

    /* Create a copy of probabilities */
    float *probsCopy = (float *)TINYAI_MALLOC(size * sizeof(float));
    if (!probsCopy) {
        return 0; /* Error, return first token */
    }

    memcpy(probsCopy, probs, size * sizeof(float));

    /* Find top K indices */
    uint32_t *topIndices = (uint32_t *)TINYAI_MALLOC(k * sizeof(uint32_t));
    if (!topIndices) {
        TINYAI_FREE(probsCopy);
        return 0;
    }

    for (uint32_t i = 0; i < k; i++) {
        /* Find max probability */
        float    maxProb = -1.0f;
        uint32_t maxIdx  = 0;

        for (uint32_t j = 0; j < size; j++) {
            if (probsCopy[j] > maxProb) {
                maxProb = probsCopy[j];
                maxIdx  = j;
            }
        }

        topIndices[i]     = maxIdx;
        probsCopy[maxIdx] = -1.0f; /* Mark as used */
    }

    /* Sample from top K */
    float sum = 0.0f;
    for (uint32_t i = 0; i < k; i++) {
        sum += probs[topIndices[i]];
    }

    float r      = randomFloat() * sum;
    float cumSum = 0.0f;

    int result = topIndices[0]; /* Default to first top-K token */

    for (uint32_t i = 0; i < k; i++) {
        cumSum += probs[topIndices[i]];
        if (r < cumSum) {
            result = topIndices[i];
            break;
        }
    }

    /* Clean up */
    TINYAI_FREE(probsCopy);
    TINYAI_FREE(topIndices);

    return result;
}

/**
 * Perform top-P (nucleus) sampling
 */
static int sampleTopP(const float *probs, uint32_t size, float p)
{
    if (p >= 1.0f) {
        /* No need for top-P if P is greater than or equal to 1.0 */
        float sum = 0.0f;
        for (uint32_t i = 0; i < size; i++) {
            sum += probs[i];
        }

        float r      = randomFloat() * sum;
        float cumSum = 0.0f;

        for (uint32_t i = 0; i < size; i++) {
            cumSum += probs[i];
            if (r < cumSum) {
                return i;
            }
        }

        return size - 1;
    }

    /* Create a copy of probabilities with indices */
    struct ProbIndex {
        float    prob;
        uint32_t index;
    } *probIndices = (struct ProbIndex *)TINYAI_MALLOC(size * sizeof(struct ProbIndex));

    if (!probIndices) {
        return 0; /* Error, return first token */
    }

    for (uint32_t i = 0; i < size; i++) {
        probIndices[i].prob  = probs[i];
        probIndices[i].index = i;
    }

    /* Sort by probability (descending) */
    for (uint32_t i = 0; i < size - 1; i++) {
        for (uint32_t j = i + 1; j < size; j++) {
            if (probIndices[j].prob > probIndices[i].prob) {
                struct ProbIndex temp = probIndices[i];
                probIndices[i]        = probIndices[j];
                probIndices[j]        = temp;
            }
        }
    }

    /* Find cutoff index for top-P */
    float    cumSum    = 0.0f;
    uint32_t cutoffIdx = 0;

    for (uint32_t i = 0; i < size; i++) {
        cumSum += probIndices[i].prob;
        if (cumSum >= p) {
            cutoffIdx = i + 1;
            break;
        }
    }

    if (cutoffIdx == 0) {
        cutoffIdx = size; /* Use all tokens if can't reach p */
    }

    /* Sample from top-P */
    float sum = 0.0f;
    for (uint32_t i = 0; i < cutoffIdx; i++) {
        sum += probIndices[i].prob;
    }

    float r = randomFloat() * sum;
    cumSum  = 0.0f;

    int result = probIndices[0].index; /* Default to highest probability token */

    for (uint32_t i = 0; i < cutoffIdx; i++) {
        cumSum += probIndices[i].prob;
        if (r < cumSum) {
            result = probIndices[i].index;
            break;
        }
    }

    /* Clean up */
    TINYAI_FREE(probIndices);

    return result;
}

/**
 * Copy a layer's weights from FP32 to 4-bit
 */
static int copyLayerWeights(TinyAILayer *layer, const float *weights, const float *biases)
{
    if (!layer || !weights) {
        return -1;
    }

    /* Create temporary FP32 matrix */
    TinyAIMatrixFP32 weightsFP32;
    weightsFP32.rows = layer->inputSize;
    weightsFP32.cols = layer->outputSize;
    weightsFP32.data = (float *)weights;

    /* Quantize to 4-bit */
    TinyAIMatrix4bit *weights4bit = tinyaiQuantizeFP32To4bit(&weightsFP32);
    if (!weights4bit) {
        return -1;
    }

    /* Copy to layer */
    memcpy(&layer->weights, weights4bit, sizeof(TinyAIMatrix4bit));

    /* Add biases if provided */
    if (biases) {
        layer->biases = (float *)TINYAI_MALLOC(layer->outputSize * sizeof(float));
        if (!layer->biases) {
            tinyaiDestroyMatrix4bit(weights4bit);
            return -1;
        }

        memcpy(layer->biases, biases, layer->outputSize * sizeof(float));
    }

    /* Clean up */
    TINYAI_FREE(
        weights4bit); /* Just free the struct, not the data which is now owned by the layer */

    return 0;
}

/* ----------------- Model Implementation ----------------- */

/**
 * Create a new text generation model
 */
TinyAIModel *tinyaiCreateModel(uint32_t type, uint32_t hiddenSize, uint32_t contextSize,
                               TinyAITokenizer *tokenizer)
{
    if (!tokenizer) {
        return NULL;
    }

    TinyAIModel *model = (TinyAIModel *)TINYAI_MALLOC(sizeof(TinyAIModel));
    if (!model) {
        return NULL;
    }

    /* Initialize the model */
    model->type        = type;
    model->layerCount  = 0;
    model->layers      = NULL;
    model->tokenizer   = tokenizer;
    model->hiddenSize  = hiddenSize;
    model->contextSize = contextSize;

    /* Allocate activation buffers */
    model->activations[0] = (float *)TINYAI_MALLOC(contextSize * hiddenSize * sizeof(float));
    model->activations[1] = (float *)TINYAI_MALLOC(contextSize * hiddenSize * sizeof(float));

    if (!model->activations[0] || !model->activations[1]) {
        if (model->activations[0])
            TINYAI_FREE(model->activations[0]);
        if (model->activations[1])
            TINYAI_FREE(model->activations[1]);
        TINYAI_FREE(model);
        return NULL;
    }

    model->activeBuffer = 0;

    return model;
}

/**
 * Destroy a text generation model
 */
void tinyaiDestroyModel(TinyAIModel *model)
{
    if (!model) {
        return;
    }

    /* Free layers */
    if (model->layers) {
        for (uint32_t i = 0; i < model->layerCount; i++) {
            if (model->layers[i].weights.data) {
                TINYAI_FREE(model->layers[i].weights.data);
            }
            if (model->layers[i].biases) {
                TINYAI_FREE(model->layers[i].biases);
            }
        }
        TINYAI_FREE(model->layers);
    }

    /* Free activation buffers */
    if (model->activations[0]) {
        TINYAI_FREE(model->activations[0]);
    }
    if (model->activations[1]) {
        TINYAI_FREE(model->activations[1]);
    }

    /* Note: We don't free the tokenizer as it might be used elsewhere */

    /* Free the model */
    TINYAI_FREE(model);
}

/**
 * Add a layer to a model
 */
int tinyaiAddLayer(TinyAIModel *model, TinyAILayerType type, uint32_t inputSize,
                   uint32_t outputSize, TinyAIActivation activation)
{
    if (!model) {
        return -1;
    }

    /* Allocate or reallocate the layers array */
    TinyAILayer *newLayers =
        (TinyAILayer *)TINYAI_MALLOC((model->layerCount + 1) * sizeof(TinyAILayer));
    if (!newLayers) {
        return -1;
    }

    /* Copy existing layers */
    if (model->layers) {
        memcpy(newLayers, model->layers, model->layerCount * sizeof(TinyAILayer));
        TINYAI_FREE(model->layers);
    }

    model->layers = newLayers;

    /* Initialize the new layer */
    TinyAILayer *layer  = &model->layers[model->layerCount];
    layer->type         = type;
    layer->activation   = activation;
    layer->inputSize    = inputSize;
    layer->outputSize   = outputSize;
    layer->weights.data = NULL;
    layer->biases       = NULL;

    model->layerCount++;

    return 0;
}

/**
 * Load model weights from a file
 */
int tinyaiLoadModelWeights(TinyAIModel *model, const char *path)
{
    if (!model || !path) {
        return -1;
    }

    FILE *file = fopen(path, "rb");
    if (!file) {
        return -1;
    }

    /* Read header */
    uint32_t magic, version, layerCount;
    if (fread(&magic, sizeof(magic), 1, file) != 1 || magic != 0x4D494E54) { /* "TINY" */
        fclose(file);
        return -1;
    }

    if (fread(&version, sizeof(version), 1, file) != 1 ||
        fread(&layerCount, sizeof(layerCount), 1, file) != 1 || layerCount != model->layerCount) {
        fclose(file);
        return -1;
    }

    /* Read layer weights */
    for (uint32_t i = 0; i < model->layerCount; i++) {
        TinyAILayer *layer = &model->layers[i];

        /* Read layer type and sizes */
        uint32_t layerType, inputSize, outputSize;
        if (fread(&layerType, sizeof(layerType), 1, file) != 1 ||
            fread(&inputSize, sizeof(inputSize), 1, file) != 1 ||
            fread(&outputSize, sizeof(outputSize), 1, file) != 1) {
            fclose(file);
            return -1;
        }

        /* Verify layer details */
        if (layerType != layer->type || inputSize != layer->inputSize ||
            outputSize != layer->outputSize) {
            fclose(file);
            return -1;
        }

        /* Allocate weight matrix */
        size_t dataSize =
            (layer->inputSize * layer->outputSize + 1) / 2; /* 4-bit, 2 values per byte */
        layer->weights.data = (uint8_t *)TINYAI_MALLOC(dataSize);
        if (!layer->weights.data) {
            fclose(file);
            return -1;
        }

        /* Read weights */
        if (fread(&layer->weights.scale, sizeof(layer->weights.scale), 1, file) != 1 ||
            fread(&layer->weights.zeroPoint, sizeof(layer->weights.zeroPoint), 1, file) != 1 ||
            fread(layer->weights.data, 1, dataSize, file) != dataSize) {
            fclose(file);
            return -1;
        }

        layer->weights.rows = layer->inputSize;
        layer->weights.cols = layer->outputSize;

        /* Allocate and read biases */
        layer->biases = (float *)TINYAI_MALLOC(layer->outputSize * sizeof(float));
        if (!layer->biases) {
            fclose(file);
            return -1;
        }

        if (fread(layer->biases, sizeof(float), layer->outputSize, file) != layer->outputSize) {
            fclose(file);
            return -1;
        }
    }

    fclose(file);
    return 0;
}

/**
 * Load a complete model from files
 */
TinyAIModel *tinyaiLoadModel(const char *modelPath, const char *weightsPath,
                             const char *tokenizerPath)
{
    /* Load model structure */
    FILE *file = fopen(modelPath, "rb");
    if (!file) {
        return NULL;
    }

    /* Read header */
    uint32_t magic, version, type, hiddenSize, contextSize, layerCount;
    if (fread(&magic, sizeof(magic), 1, file) != 1 || magic != 0x4D494E54 || /* "TINY" */
        fread(&version, sizeof(version), 1, file) != 1 ||
        fread(&type, sizeof(type), 1, file) != 1 ||
        fread(&hiddenSize, sizeof(hiddenSize), 1, file) != 1 ||
        fread(&contextSize, sizeof(contextSize), 1, file) != 1 ||
        fread(&layerCount, sizeof(layerCount), 1, file) != 1) {
        fclose(file);
        return NULL;
    }

    /* Load tokenizer */
    TinyAITokenizer *tokenizer = tinyaiCreateTokenizer();
    if (!tokenizer) {
        fclose(file);
        return NULL;
    }

    if (tinyaiLoadVocabulary(tokenizer, tokenizerPath) != 0) {
        tinyaiDestroyTokenizer(tokenizer);
        fclose(file);
        return NULL;
    }

    /* Create model */
    TinyAIModel *model = tinyaiCreateModel(type, hiddenSize, contextSize, tokenizer);
    if (!model) {
        tinyaiDestroyTokenizer(tokenizer);
        fclose(file);
        return NULL;
    }

    /* Read layer definitions */
    for (uint32_t i = 0; i < layerCount; i++) {
        uint32_t layerType, inputSize, outputSize, activation;
        if (fread(&layerType, sizeof(layerType), 1, file) != 1 ||
            fread(&inputSize, sizeof(inputSize), 1, file) != 1 ||
            fread(&outputSize, sizeof(outputSize), 1, file) != 1 ||
            fread(&activation, sizeof(activation), 1, file) != 1) {
            tinyaiDestroyModel(model);
            fclose(file);
            return NULL;
        }

        tinyaiAddLayer(model, (TinyAILayerType)layerType, inputSize, outputSize,
                       (TinyAIActivation)activation);
    }

    fclose(file);

    /* Load weights */
    if (tinyaiLoadModelWeights(model, weightsPath) != 0) {
        tinyaiDestroyModel(model);
        return NULL;
    }

    return model;
}

/* Include SIMD operations and cache optimizations */
#include "../../utils/cache_opt.h"
#include "../../utils/simd_ops.h"

/**
 * Cache-optimized matrix-vector multiplication using SIMD if available
 */
static void matrixVectorMultiplySIMD(const float *matrix, const float *vector, float *output,
                                     int rows, int cols, bool useSIMD)
{
    if (useSIMD) {
        /* Use SIMD implementation from simd_ops.h */
        tinyai_simd_matmul_f32(matrix, vector, output, rows, cols, 1);
    }
    else {
        /* Get cache optimization parameters */
        TinyAICacheOptConfig config = tinyai_cache_opt_init_default();
        tinyai_cache_opt_matrix_multiply(rows, 1, cols, &config);

        /* Use cache-friendly blocking */
        TINYAI_LOOP_TILING_2D(i, 0, rows, k, 0, cols, config.blockSizeX, config.blockSizeY, {
            /* Inside the innermost loops, accumulate partial sums */
            output[i] += matrix[i * cols + k] * vector[k];
        });
    }
}

/**
 * Perform a single forward pass through the model
 */
int tinyaiModelForward(TinyAIModel *model, const int *input, int inputLength, float *output)
{
    if (!model || !input || !output || inputLength <= 0) {
        return -1;
    }

    /* Context limitation */
    if (inputLength > model->contextSize) {
        inputLength = model->contextSize;
    }

    /* Check if SIMD is available */
    bool useSIMD =
        tinyai_simd_detect_capabilities() > 0; /* Non-zero means some SIMD is available */

    /* Process based on model type */
    switch (model->type) {
    case TINYAI_MODEL_TYPE_RNN:
        /* Simple RNN implementation */
        {
            /* Embedding layer */
            float *embeddings = model->activations[model->activeBuffer];
            memset(embeddings, 0, model->hiddenSize * sizeof(float));

            /* Process only the last token for output */
            int lastToken = input[inputLength - 1];

            /* Check token range */
            if (lastToken < 0 || lastToken >= model->tokenizer->tokenCount) {
                lastToken = TINYAI_TOKEN_UNKNOWN;
            }

            /* Apply the model layers */
            for (uint32_t i = 0; i < model->layerCount; i++) {
                TinyAILayer *layer = &model->layers[i];

                /* Flip activation buffers */
                model->activeBuffer = 1 - model->activeBuffer;
                float *input        = model->activations[1 - model->activeBuffer];
                float *output       = model->activations[model->activeBuffer];

                /* Apply layer based on type */
                switch (layer->type) {
                case TINYAI_LAYER_EMBEDDING:
                    /* Copy embedding vector for the token */
                    {
                        TinyAIMatrixFP32 *matrix = tinyaiDequantize4bitToFP32(&layer->weights);

                        if (!matrix) {
                            return -1;
                        }

                        /* Copy embedding for the token */
                        memcpy(output, matrix->data + lastToken * layer->outputSize,
                               layer->outputSize * sizeof(float));

                        tinyaiDestroyMatrixFP32(matrix);
                    }
                    break;

                case TINYAI_LAYER_DENSE:
                    /* Dense layer implementation */
                    {
                        TinyAIMatrixFP32 *matrix = tinyaiDequantize4bitToFP32(&layer->weights);

                        if (!matrix) {
                            return -1;
                        }

                        /* Matrix multiplication */
                        for (uint32_t j = 0; j < layer->outputSize; j++) {
                            float sum = 0.0f;
                            for (uint32_t k = 0; k < layer->inputSize; k++) {
                                sum += input[k] * matrix->data[k * layer->outputSize + j];
                            }

                            /* Add bias */
                            if (layer->biases) {
                                sum += layer->biases[j];
                            }

                            /* Store result */
                            output[j] = sum;
                        }

                        tinyaiDestroyMatrixFP32(matrix);

                        /* Apply activation function */
                        switch (layer->activation) {
                        case TINYAI_ACTIVATION_RELU:
                            for (uint32_t j = 0; j < layer->outputSize; j++) {
                                if (output[j] < 0.0f) {
                                    output[j] = 0.0f;
                                }
                            }
                            break;

                        case TINYAI_ACTIVATION_SIGMOID:
                            for (uint32_t j = 0; j < layer->outputSize; j++) {
                                output[j] = 1.0f / (1.0f + expf(-output[j]));
                            }
                            break;

                        case TINYAI_ACTIVATION_TANH:
                            for (uint32_t j = 0; j < layer->outputSize; j++) {
                                output[j] = tanhf(output[j]);
                            }
                            break;

                        case TINYAI_ACTIVATION_GELU:
                            for (uint32_t j = 0; j < layer->outputSize; j++) {
                                float x   = output[j];
                                output[j] = 0.5f * x *
                                            (1.0f + tanhf(sqrtf(2.0f / 3.14159f) *
                                                          (x + 0.044715f * x * x * x)));
                            }
                            break;

                        default:
                            /* No activation (linear) */
                            break;
                        }
                    }
                    break;

                case TINYAI_LAYER_OUTPUT:
                    /* Output layer */
                    {
                        TinyAIMatrixFP32 *matrix = tinyaiDequantize4bitToFP32(&layer->weights);

                        if (!matrix) {
                            return -1;
                        }

                        /* Matrix multiplication */
                        for (uint32_t j = 0; j < layer->outputSize; j++) {
                            float sum = 0.0f;
                            for (uint32_t k = 0; k < layer->inputSize; k++) {
                                sum += input[k] * matrix->data[k * layer->outputSize + j];
                            }

                            /* Add bias */
                            if (layer->biases) {
                                sum += layer->biases[j];
                            }

                            /* Store result directly to output */
                            output[j] = sum;
                        }

                        tinyaiDestroyMatrixFP32(matrix);
                    }
                    break;

                default:
                    /* Unsupported layer type */
                    return -1;
                }
            }

            /* Copy final activations to output */
            memcpy(output, model->activations[model->activeBuffer],
                   model->tokenizer->tokenCount * sizeof(float));
        }
        break;

    case TINYAI_MODEL_TYPE_TRANSFORMER:
        /* Transformer implementation */
        /* In a real implementation, we would need much more code for transformers */
        /* This is a very simplified version for demonstration purposes */
        {
            /* Embedding layer */
            float *embeddings = model->activations[model->activeBuffer];
            memset(embeddings, 0, inputLength * model->hiddenSize * sizeof(float));

            /* Apply the model layers */
            for (uint32_t i = 0; i < model->layerCount; i++) {
                TinyAILayer *layer = &model->layers[i];

                /* Flip activation buffers */
                model->activeBuffer = 1 - model->activeBuffer;
                float *input        = model->activations[1 - model->activeBuffer];
                float *output       = model->activations[model->activeBuffer];

                /* Apply layer based on type */
                switch (layer->type) {
                case TINYAI_LAYER_EMBEDDING:
                    /* Apply token embeddings */
                    {
                        TinyAIMatrixFP32 *matrix = tinyaiDequantize4bitToFP32(&layer->weights);

                        if (!matrix) {
                            return -1;
                        }

                        /* Embed each token */
                        for (int j = 0; j < inputLength; j++) {
                            int token = input[j];

                            /* Check token range */
                            if (token < 0 || token >= model->tokenizer->tokenCount) {
                                token = TINYAI_TOKEN_UNKNOWN;
                            }

                            /* Copy embedding for this token */
                            memcpy(output + j * layer->outputSize,
                                   matrix->data + token * layer->outputSize,
                                   layer->outputSize * sizeof(float));
                        }

                        tinyaiDestroyMatrixFP32(matrix);
                    }
                    break;

                case TINYAI_LAYER_ATTENTION:
                    /* Simplified self-attention layer */
                    /* A real implementation would be much more complex */
                    {
                        /* For this simplified version, just pass through */
                        memcpy(output, input, inputLength * layer->outputSize * sizeof(float));
                    }
                    break;

                case TINYAI_LAYER_DENSE:
                    /* Dense layer for each position */
                    {
                        for (int j = 0; j < inputLength; j++) {
                            TinyAIMatrixFP32 *matrix = tinyaiDequantize4bitToFP32(&layer->weights);

                            if (!matrix) {
                                return -1;
                            }

                            /* Matrix multiplication for this position */
                            for (uint32_t k = 0; k < layer->outputSize; k++) {
                                float sum = 0.0f;
                                for (uint32_t l = 0; l < layer->inputSize; l++) {
                                    sum += input[j * layer->inputSize + l] *
                                           matrix->data[l * layer->outputSize + k];
                                }

                                /* Add bias */
                                if (layer->biases) {
                                    sum += layer->biases[k];
                                }

                                /* Store result */
                                output[j * layer->outputSize + k] = sum;
                            }

                            tinyaiDestroyMatrixFP32(matrix);

                            /* Apply activation function */
                            switch (layer->activation) {
                            case TINYAI_ACTIVATION_RELU:
                                for (uint32_t k = 0; k < layer->outputSize; k++) {
                                    float *val = &output[j * layer->outputSize + k];
                                    if (*val < 0.0f)
                                        *val = 0.0f;
                                }
                                break;

                            case TINYAI_ACTIVATION_SIGMOID:
                                for (uint32_t k = 0; k < layer->outputSize; k++) {
                                    float *val = &output[j * layer->outputSize + k];
                                    *val       = 1.0f / (1.0f + expf(-*val));
                                }
                                break;

                            case TINYAI_ACTIVATION_TANH:
                                for (uint32_t k = 0; k < layer->outputSize; k++) {
                                    float *val = &output[j * layer->outputSize + k];
                                    *val       = tanhf(*val);
                                }
                                break;

                            case TINYAI_ACTIVATION_GELU:
                                for (uint32_t k = 0; k < layer->outputSize; k++) {
                                    float *val = &output[j * layer->outputSize + k];
                                    float  x   = *val;
                                    *val       = 0.5f * x *
                                           (1.0f + tanhf(sqrtf(2.0f / 3.14159f) *
                                                         (x + 0.044715f * x * x * x)));
                                }
                                break;

                            default:
                                /* No activation (linear) */
                                break;
                            }
                        }
                    }
                    break;

                case TINYAI_LAYER_LAYERNORM:
                    /* Layer normalization */
                    /* Apply to each position separately */
                    for (int j = 0; j < inputLength; j++) {
                        /* Calculate mean */
                        float mean = 0.0f;
                        for (uint32_t k = 0; k < layer->inputSize; k++) {
                            mean += input[j * layer->inputSize + k];
                        }
                        mean /= layer->inputSize;

                        /* Calculate variance */
                        float variance = 0.0f;
                        for (uint32_t k = 0; k < layer->inputSize; k++) {
                            float diff = input[j * layer->inputSize + k] - mean;
                            variance += diff * diff;
                        }
                        variance /= layer->inputSize;

                        /* Normalize */
                        for (uint32_t k = 0; k < layer->inputSize; k++) {
                            float val =
                                (input[j * layer->inputSize + k] - mean) / sqrtf(variance + 1e-5f);

                            /* Scale and shift (using weights and biases) */
                            if (layer->biases) {
                                val = val * layer->biases[k] + layer->biases[layer->inputSize + k];
                            }

                            output[j * layer->inputSize + k] = val;
                        }
                    }
                    break;

                case TINYAI_LAYER_OUTPUT:
                    /* Output layer */
                    /* For transformers, use only the last position */
                    {
                        TinyAIMatrixFP32 *matrix = tinyaiDequantize4bitToFP32(&layer->weights);

                        if (!matrix) {
                            return -1;
                        }

                        /* For the output layer, we use only the last token's representation */
                        float *lastTokenRep = input + (inputLength - 1) * layer->inputSize;

                        /* Matrix multiplication */
                        for (uint32_t j = 0; j < layer->outputSize; j++) {
                            float sum = 0.0f;
                            for (uint32_t k = 0; k < layer->inputSize; k++) {
                                sum += lastTokenRep[k] * matrix->data[k * layer->outputSize + j];
                            }

                            /* Add bias */
                            if (layer->biases) {
                                sum += layer->biases[j];
                            }

                            /* Store result */
                            output[j] = sum;
                        }

                        tinyaiDestroyMatrixFP32(matrix);
                    }
                    break;

                default:
                    /* Unsupported layer type */
                    return -1;
                }
            }

            /* Copy final logits to output */
            if (model->tokenizer->tokenCount <= model->layerCount) {
                /* Vocabulary too small, there's a problem */
                return -1;
            }

            /* Get the final output layer's values */
            float *finalOutput = model->activations[model->activeBuffer];
            memcpy(output, finalOutput, model->tokenizer->tokenCount * sizeof(float));
        }
        break;

    default:
        /* Unsupported model type */
        return -1;
    }

    return 0;
}

/**
 * Sample the next token from output probabilities
 */
int tinyaiSampleToken(const float *output, int vocabSize, const TinyAIGenerationParams *params)
{
    if (!output || !params) {
        return 0; /* Default to first token on error */
    }

    /* Allocate buffer for probabilities */
    float *probs = (float *)TINYAI_MALLOC(vocabSize * sizeof(float));
    if (!probs) {
        return 0; /* Default to first token on error */
    }

    /* Copy and apply temperature */
    memcpy(probs, output, vocabSize * sizeof(float));
    applyTemperature(probs, vocabSize, params->temperature);

    /* Convert to probabilities using softmax */
    softmax(probs, vocabSize);

    /* Sample token based on method */
    int token;

    switch (params->samplingMethod) {
    case TINYAI_SAMPLING_GREEDY:
        /* Choose highest probability token */
        token = 0;
        for (int i = 1; i < vocabSize; i++) {
            if (probs[i] > probs[token]) {
                token = i;
            }
        }
        break;

    case TINYAI_SAMPLING_TOP_K:
        token = sampleTopK(probs, vocabSize, params->topK);
        break;

    case TINYAI_SAMPLING_TOP_P:
        token = sampleTopP(probs, vocabSize, params->topP);
        break;

    case TINYAI_SAMPLING_TEMPERATURE:
        /* Already applied temperature, sample directly */
        {
            float r      = randomFloat();
            float cumSum = 0.0f;

            token = 0; /* Default */

            for (int i = 0; i < vocabSize; i++) {
                cumSum += probs[i];
                if (r < cumSum) {
                    token = i;
                    break;
                }
            }
        }
        break;

    default:
        /* Unknown sampling method, use greedy */
        token = 0;
        for (int i = 1; i < vocabSize; i++) {
            if (probs[i] > probs[token]) {
                token = i;
            }
        }
        break;
    }

    /* Clean up */
    TINYAI_FREE(probs);

    return token;
}

/**
 * Generate text from a model
 */
int tinyaiGenerateText(TinyAIModel *model, const TinyAIGenerationParams *params, int *outputTokens,
                       int maxOutputTokens)
{
    if (!model || !params || !outputTokens || maxOutputTokens <= 0) {
        return 0;
    }

    /* Initialize random number generator */
    seedRandom(params->seed);

    /* Check if prompt is provided */
    if (!params->promptTokens || params->promptLength == 0) {
        /* Start with BOS token */
        outputTokens[0] = TINYAI_TOKEN_BOS;

        int numTokens = 1;

        /* Generate tokens */
        while (numTokens < maxOutputTokens && numTokens < params->maxTokens) {
            /* Forward pass */
            float *logits = (float *)TINYAI_MALLOC(model->tokenizer->tokenCount * sizeof(float));
            if (!logits) {
                /* Memory allocation failed */
                break;
            }

            /* Get logits for next token */
            int result = tinyaiModelForward(model, outputTokens, numTokens, logits);
            if (result != 0) {
                TINYAI_FREE(logits);
                break;
            }

            /* Sample next token */
            int nextToken = tinyaiSampleToken(logits, model->tokenizer->tokenCount, params);
            TINYAI_FREE(logits);

            /* Check for EOS token */
            if (nextToken == TINYAI_TOKEN_EOS) {
                break;
            }

            /* Add token to output */
            outputTokens[numTokens++] = nextToken;
        }

        return numTokens;
    }
    else {
        /* Start with prompt */
        if (params->promptLength > maxOutputTokens) {
            /* Prompt too long */
            return 0;
        }

        /* Copy prompt */
        memcpy(outputTokens, params->promptTokens, params->promptLength * sizeof(int));
        int numTokens = params->promptLength;

        /* Generate tokens */
        while (numTokens < maxOutputTokens && numTokens < params->maxTokens) {
            /* Forward pass */
            float *logits = (float *)TINYAI_MALLOC(model->tokenizer->tokenCount * sizeof(float));
            if (!logits) {
                /* Memory allocation failed */
                break;
            }

            /* Get logits for next token */
            int contextSize = numTokens;
            if (contextSize > model->contextSize) {
                contextSize = model->contextSize;
            }

            int result = tinyaiModelForward(model, outputTokens + numTokens - contextSize,
                                            contextSize, logits);
            if (result != 0) {
                TINYAI_FREE(logits);
                break;
            }

            /* Sample next token */
            int nextToken = tinyaiSampleToken(logits, model->tokenizer->tokenCount, params);
            TINYAI_FREE(logits);

            /* Check for EOS token */
            if (nextToken == TINYAI_TOKEN_EOS) {
                break;
            }

            /* Add token to output */
            outputTokens[numTokens++] = nextToken;
        }

        return numTokens;
    }
}

/**
 * Convert a model to 4-bit quantization
 */
int tinyaiQuantizeModel(TinyAIModel *model)
{
    if (!model) {
        return -1;
    }

    /* Each layer is already 4-bit quantized during loading */
    /* This function is a placeholder for higher-precision models that need conversion */

    return 0;
}
