/**
 * @file multimodal_model.c
 * @brief Implementation of the multimodal model functionality in TinyAI
 */

#include "multimodal_model.h"
#include "../../core/memory.h"
#include "fusion.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Forward declaration of Layer struct for internal use */
typedef struct Layer Layer;

/* Define the FusionLayer structure */
typedef struct {
    int  type;          /* Fusion type (concat, add, multiply, attention) */
    char name[32];      /* Layer name */
    int *inputDims;     /* Input dimensions for each modality */
    int  numModalities; /* Number of modalities to fuse */
    int  outputDim;     /* Output dimension */

    /* Layer weights and parameters for attention-based fusion */
    uint8_t *weights; /* 4-bit quantized weights (if applicable) */
    float   *biases;  /* Biases for attention */
    float   *scales;  /* Scale factors for quantized weights */

    /* Memory requirements */
    size_t weightBytes; /* Size of weights in bytes */
    size_t biasBytes;   /* Size of biases in bytes */
    size_t outputBytes; /* Size of output in bytes */
} FusionLayer;

/* Define the ModalityEncoder structure */
typedef struct {
    int  type;      /* Modality type (text, image, audio) */
    char name[32];  /* Encoder name */
    int  inputDim;  /* Input dimension */
    int  outputDim; /* Output dimension */

    /* Encoder type-specific parameters */
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

    /* Encoder parameters */
    uint8_t *weights; /* 4-bit quantized weights (if applicable) */
    float   *biases;  /* Biases for encoder */
    float   *scales;  /* Scale factors for quantized weights */

    /* Memory requirements */
    size_t weightBytes; /* Size of weights in bytes */
    size_t biasBytes;   /* Size of biases in bytes */
    size_t outputBytes; /* Size of output in bytes */
} ModalityEncoder;

/* Define the MultimodalModel structure */
struct TinyAIMultimodalModel {
    int                modelType;
    int                numModalities;
    int                fusionDim;
    TinyAIFusionMethod fusionMethod;

    /* Modality encoders */
    ModalityEncoder *modalityEncoders;

    /* Fusion layers */
    FusionLayer *fusionLayers;
    int          numFusionLayers;

    /* Memory */
    void *memoryPool;
    bool  useExternalMemory;
    bool  useSIMD;
    bool  useQuantization;
};

/* Private functions */

/**
 * Initialize a modality encoder
 */
static bool initModalityEncoder(ModalityEncoder *encoder, TinyAIModalityConfig *config,
                                int outputDim, bool useQuantization)
{
    if (!encoder || !config) {
        return false;
    }

    /* Set basic encoder info */
    memset(encoder, 0, sizeof(ModalityEncoder));
    encoder->type = config->modality;

    /* Set modality-specific parameters */
    switch (config->modality) {
    case TINYAI_MODALITY_TEXT:
        snprintf(encoder->name, sizeof(encoder->name), "text_encoder");
        encoder->inputDim              = config->config.text.maxTokens;
        encoder->outputDim             = outputDim;
        encoder->config.text.maxTokens = config->config.text.maxTokens;
        encoder->config.text.embedDim  = config->config.text.embedDim;

        /* For text, we need embeddings and projection */
        size_t embedSize = (size_t)config->config.text.maxTokens * config->config.text.embedDim;
        size_t projSize  = (size_t)config->config.text.embedDim * outputDim;

        if (useQuantization) {
            encoder->weightBytes = ((embedSize + projSize) + 1) / 2; /* 4-bit quantization */
        }
        else {
            encoder->weightBytes = (embedSize + projSize) * sizeof(float);
        }

        encoder->biasBytes   = outputDim * sizeof(float);
        encoder->outputBytes = outputDim * sizeof(float);
        break;

    case TINYAI_MODALITY_IMAGE:
        snprintf(encoder->name, sizeof(encoder->name), "image_encoder");
        encoder->inputDim = config->config.image.width * config->config.image.height *
                            config->config.image.channels;
        encoder->outputDim             = outputDim;
        encoder->config.image.width    = config->config.image.width;
        encoder->config.image.height   = config->config.image.height;
        encoder->config.image.channels = config->config.image.channels;

        /* For image, we need CNN and projection */
        /* This is a simplified approximation */
        size_t cnnSize  = (size_t)encoder->inputDim * 64; /* Simplified CNN */
        size_t projSize = (size_t)64 * outputDim;         /* Projection */

        if (useQuantization) {
            encoder->weightBytes = ((cnnSize + projSize) + 1) / 2; /* 4-bit quantization */
        }
        else {
            encoder->weightBytes = (cnnSize + projSize) * sizeof(float);
        }

        encoder->biasBytes   = (64 + outputDim) * sizeof(float);
        encoder->outputBytes = outputDim * sizeof(float);
        break;

    case TINYAI_MODALITY_AUDIO:
        snprintf(encoder->name, sizeof(encoder->name), "audio_encoder");
        encoder->inputDim  = config->config.audio.sampleRate * config->config.audio.duration;
        encoder->outputDim = outputDim;
        encoder->config.audio.sampleRate = config->config.audio.sampleRate;
        encoder->config.audio.duration   = config->config.audio.duration;

        /* For audio, we need feature extraction and projection */
        /* This is a simplified approximation */
        size_t featureSize = (size_t)encoder->inputDim * 64; /* Simplified features */
        size_t projSize    = (size_t)64 * outputDim;         /* Projection */

        if (useQuantization) {
            encoder->weightBytes = ((featureSize + projSize) + 1) / 2; /* 4-bit quantization */
        }
        else {
            encoder->weightBytes = (featureSize + projSize) * sizeof(float);
        }

        encoder->biasBytes   = (64 + outputDim) * sizeof(float);
        encoder->outputBytes = outputDim * sizeof(float);
        break;

    default:
        fprintf(stderr, "Unknown modality type: %d\n", config->modality);
        return false;
    }

    return true;
}

/**
 * Initialize a fusion layer
 */
static bool initFusionLayer(FusionLayer *layer, TinyAIFusionMethod fusionMethod, int *inputDims,
                            int numModalities, int outputDim, bool useQuantization)
{
    if (!layer || !inputDims || numModalities <= 0) {
        return false;
    }

    /* Set basic layer info */
    memset(layer, 0, sizeof(FusionLayer));
    layer->type          = fusionMethod;
    layer->numModalities = numModalities;
    layer->outputDim     = outputDim;

    /* Allocate input dimensions array */
    layer->inputDims = (int *)malloc(numModalities * sizeof(int));
    if (!layer->inputDims) {
        fprintf(stderr, "Failed to allocate input dimensions array\n");
        return false;
    }

    /* Copy input dimensions */
    memcpy(layer->inputDims, inputDims, numModalities * sizeof(int));

    /* Set name based on fusion method */
    switch (fusionMethod) {
    case TINYAI_FUSION_CONCAT:
        snprintf(layer->name, sizeof(layer->name), "fusion_concat");
        break;
    case TINYAI_FUSION_ADD:
        snprintf(layer->name, sizeof(layer->name), "fusion_add");
        break;
    case TINYAI_FUSION_MULTIPLY:
        snprintf(layer->name, sizeof(layer->name), "fusion_multiply");
        break;
    case TINYAI_FUSION_ATTENTION:
        snprintf(layer->name, sizeof(layer->name), "fusion_attention");
        break;
    default:
        fprintf(stderr, "Unknown fusion method: %d\n", fusionMethod);
        free(layer->inputDims);
        return false;
    }

    /* Calculate memory requirements based on fusion method */
    size_t totalInputDim = 0;
    for (int i = 0; i < numModalities; i++) {
        totalInputDim += inputDims[i];
    }

    /* For attention, we need weights */
    if (fusionMethod == TINYAI_FUSION_ATTENTION) {
        size_t attnSize = (size_t)totalInputDim * outputDim;

        if (useQuantization) {
            layer->weightBytes = (attnSize + 1) / 2; /* 4-bit quantization */
        }
        else {
            layer->weightBytes = attnSize * sizeof(float);
        }

        layer->biasBytes = outputDim * sizeof(float);
    }
    else {
        layer->weightBytes = 0;
        layer->biasBytes   = 0;
    }

    /* Output size */
    layer->outputBytes = outputDim * sizeof(float);

    return true;
}

/**
 * Create a multimodal model
 * @param params Parameters for model creation
 * @return Newly allocated model, or NULL on failure
 */
TinyAIMultimodalModel *tinyaiMultimodalModelCreate(const TinyAIMultimodalModelParams *params)
{
    if (!params || params->numModalities <= 0 || !params->modalityConfigs) {
        return NULL;
    }

    /* Allocate model structure */
    TinyAIMultimodalModel *model = (TinyAIMultimodalModel *)malloc(sizeof(TinyAIMultimodalModel));
    if (!model) {
        fprintf(stderr, "Failed to allocate model structure\n");
        return NULL;
    }

    /* Initialize the model */
    memset(model, 0, sizeof(TinyAIMultimodalModel));
    model->modelType       = params->modelType;
    model->numModalities   = params->numModalities;
    model->fusionMethod    = params->fusionMethod;
    model->fusionDim       = params->fusionDim;
    model->useQuantization = params->useQuantization;
    model->useSIMD         = params->useSIMD;
    model->numFusionLayers = params->numLayers;

    /* Allocate modality encoders */
    model->modalityEncoders =
        (ModalityEncoder *)malloc(params->numModalities * sizeof(ModalityEncoder));
    if (!model->modalityEncoders) {
        fprintf(stderr, "Failed to allocate modality encoders\n");
        free(model);
        return NULL;
    }

    /* Initialize each modality encoder */
    for (int i = 0; i < params->numModalities; i++) {
        if (!initModalityEncoder(&model->modalityEncoders[i], &params->modalityConfigs[i],
                                 params->fusionDim, model->useQuantization)) {
            fprintf(stderr, "Failed to initialize modality encoder %d\n", i);
            free(model->modalityEncoders);
            free(model);
            return NULL;
        }
    }

    /* Allocate fusion layers if needed */
    if (params->numLayers > 0) {
        model->fusionLayers = (FusionLayer *)malloc(params->numLayers * sizeof(FusionLayer));
        if (!model->fusionLayers) {
            fprintf(stderr, "Failed to allocate fusion layers\n");
            free(model->modalityEncoders);
            free(model);
            return NULL;
        }

        /* Initialize fusion layers */
        int *inputDims = (int *)malloc(params->numModalities * sizeof(int));
        if (!inputDims) {
            fprintf(stderr, "Failed to allocate input dimensions array\n");
            free(model->fusionLayers);
            free(model->modalityEncoders);
            free(model);
            return NULL;
        }

        /* Get output dimensions of each encoder */
        for (int i = 0; i < params->numModalities; i++) {
            inputDims[i] = model->modalityEncoders[i].outputDim;
        }

        /* Initialize fusion layers */
        for (int i = 0; i < params->numLayers; i++) {
            if (!initFusionLayer(&model->fusionLayers[i], params->fusionMethod, inputDims,
                                 params->numModalities, params->fusionDim,
                                 model->useQuantization)) {
                fprintf(stderr, "Failed to initialize fusion layer %d\n", i);
                free(inputDims);
                for (int j = 0; j < i; j++) {
                    free(model->fusionLayers[j].inputDims);
                }
                free(model->fusionLayers);
                free(model->modalityEncoders);
                free(model);
                return NULL;
            }
        }

        free(inputDims);
    }

    /* Calculate total memory requirements */
    size_t totalWeightBytes     = 0;
    size_t totalActivationBytes = 0;

    /* Add memory for encoders */
    for (int i = 0; i < model->numModalities; i++) {
        totalWeightBytes +=
            model->modalityEncoders[i].weightBytes + model->modalityEncoders[i].biasBytes;
        if (model->modalityEncoders[i].outputBytes > totalActivationBytes) {
            totalActivationBytes = model->modalityEncoders[i].outputBytes;
        }
    }

    /* Add memory for fusion layers */
    for (int i = 0; i < model->numFusionLayers; i++) {
        totalWeightBytes += model->fusionLayers[i].weightBytes + model->fusionLayers[i].biasBytes;
        if (model->fusionLayers[i].outputBytes > totalActivationBytes) {
            totalActivationBytes = model->fusionLayers[i].outputBytes;
        }
    }

    /* Allocate memory pool if needed */
    if (!model->memoryPool && !params->customParams) {
        model->memoryPool =
            tinyaiMemoryPoolCreate(totalWeightBytes, totalActivationBytes, model->useSIMD);
        if (!model->memoryPool) {
            fprintf(stderr, "Failed to allocate memory pool\n");
            if (model->fusionLayers) {
                for (int i = 0; i < model->numFusionLayers; i++) {
                    free(model->fusionLayers[i].inputDims);
                }
                free(model->fusionLayers);
            }
            free(model->modalityEncoders);
            free(model);
            return NULL;
        }

        model->useExternalMemory = false;
    }
    else if (params->customParams) {
        /* Use provided memory pool */
        model->memoryPool        = params->customParams;
        model->useExternalMemory = true;
    }

    /* Load weights if specified */
    if (params->weightsFile) {
        /* Load weights from file */
        /* This will be implemented in a separate function */
    }

    return model;
}

/**
 * Free a multimodal model
 * @param model The model to free
 */
void tinyaiMultimodalModelFree(TinyAIMultimodalModel *model)
{
    if (!model) {
        return;
    }

    /* Free fusion layers */
    if (model->fusionLayers) {
        for (int i = 0; i < model->numFusionLayers; i++) {
            free(model->fusionLayers[i].inputDims);
        }
        free(model->fusionLayers);
    }

    /* Free modality encoders */
    free(model->modalityEncoders);

    /* Free memory pool if we own it */
    if (model->memoryPool && !model->useExternalMemory) {
        tinyaiMemoryPoolFree(model->memoryPool);
    }

    /* Free the model structure */
    free(model);
}

/**
 * Process individual modality using its encoder
 */
static bool processModality(const ModalityEncoder *encoder, const void *input, int inputLength,
                            float *output, bool useSIMD)
{
    if (!encoder || !input || !output) {
        return false;
    }

    /* This is a placeholder implementation */
    /* In a real implementation, we would use the appropriate encoder for each modality */

    /* For now, just set output to zeros */
    memset(output, 0, encoder->outputDim * sizeof(float));

    return true;
}

/**
 * Process multimodal input
 * @param model The model to use
 * @param input Multimodal input containing different modalities
 * @param output Output structure to store results (must be pre-allocated)
 * @return true on success, false on failure
 */
bool tinyaiMultimodalModelProcess(TinyAIMultimodalModel *model, const TinyAIMultimodalInput *input,
                                  TinyAIMultimodalOutput *output)
{
    if (!model || !input || !output) {
        return false;
    }

    /* Allocate temporary buffer for encoder outputs */
    float **encoderOutputs = (float **)malloc(model->numModalities * sizeof(float *));
    if (!encoderOutputs) {
        fprintf(stderr, "Failed to allocate encoder outputs\n");
        return false;
    }

    for (int i = 0; i < model->numModalities; i++) {
        encoderOutputs[i] = (float *)malloc(model->modalityEncoders[i].outputDim * sizeof(float));
        if (!encoderOutputs[i]) {
            fprintf(stderr, "Failed to allocate encoder output %d\n", i);
            for (int j = 0; j < i; j++) {
                free(encoderOutputs[j]);
            }
            free(encoderOutputs);
            return false;
        }
    }

    /* Process each modality */
    for (int i = 0; i < model->numModalities; i++) {
        const void *modalityInput = NULL;
        int         inputLength   = 0;

        /* Get input for this modality */
        switch (model->modalityEncoders[i].type) {
        case TINYAI_MODALITY_TEXT:
            modalityInput = input->textInput;
            inputLength   = input->textLength;
            break;

        case TINYAI_MODALITY_IMAGE:
            modalityInput = input->imageInput;
            inputLength   = 1; /* For image, we just have one input */
            break;

        case TINYAI_MODALITY_AUDIO:
            modalityInput = input->audioInput;
            inputLength   = input->audioLength;
            break;

        default:
            fprintf(stderr, "Unknown modality type: %d\n", model->modalityEncoders[i].type);
            for (int j = 0; j < model->numModalities; j++) {
                free(encoderOutputs[j]);
            }
            free(encoderOutputs);
            return false;
        }

        /* Process this modality */
        if (!processModality(&model->modalityEncoders[i], modalityInput, inputLength,
                             encoderOutputs[i], model->useSIMD)) {
            fprintf(stderr, "Failed to process modality %d\n", i);
            for (int j = 0; j < model->numModalities; j++) {
                free(encoderOutputs[j]);
            }
            free(encoderOutputs);
            return false;
        }
    }

    /* Initialize output structure */
    output->embedDim = model->fusionDim;
    output->length   = 1; /* For now, we just have one output */

    /* Allocate memory for output embeddings if not already allocated */
    if (!output->embeddings) {
        output->embeddings = (float *)malloc(output->embedDim * output->length * sizeof(float));
        if (!output->embeddings) {
            fprintf(stderr, "Failed to allocate output embeddings\n");
            for (int i = 0; i < model->numModalities; i++) {
                free(encoderOutputs[i]);
            }
            free(encoderOutputs);
            return false;
        }
    }

    /* Prepare inputs for fusion */
    const float **fusionInputs = (const float **)malloc(model->numModalities * sizeof(float *));
    if (!fusionInputs) {
        fprintf(stderr, "Failed to allocate fusion inputs\n");
        for (int i = 0; i < model->numModalities; i++) {
            free(encoderOutputs[i]);
        }
        free(encoderOutputs);
        return false;
    }

    int *fusionDims = (int *)malloc(model->numModalities * sizeof(int));
    if (!fusionDims) {
        fprintf(stderr, "Failed to allocate fusion dimensions\n");
        free(fusionInputs);
        for (int i = 0; i < model->numModalities; i++) {
            free(encoderOutputs[i]);
        }
        free(encoderOutputs);
        return false;
    }

    /* Set up fusion inputs and dimensions */
    for (int i = 0; i < model->numModalities; i++) {
        fusionInputs[i] = encoderOutputs[i];
        fusionDims[i]   = model->modalityEncoders[i].outputDim;
    }

    /* Apply fusion method */
    bool fusionSuccess = false;
    switch (model->fusionMethod) {
    case TINYAI_FUSION_CONCAT:
        fusionSuccess = tinyaiFusionConcat(fusionInputs, fusionDims, model->numModalities,
                                           output->embeddings, output->embedDim);
        break;

    case TINYAI_FUSION_ADD:
        fusionSuccess = tinyaiFusionAdd(fusionInputs, fusionDims, model->numModalities,
                                        output->embeddings, output->embedDim);
        break;

    case TINYAI_FUSION_MULTIPLY:
        fusionSuccess = tinyaiFusionMultiply(fusionInputs, fusionDims, model->numModalities,
                                             output->embeddings, output->embedDim);
        break;

    case TINYAI_FUSION_ATTENTION:
        /* For attention, we need to use the fusion layers weights */
        fusionSuccess = tinyaiFusionAttention(fusionInputs, fusionDims, model->numModalities,
                                              NULL, /* No pre-defined weights, use learned ones */
                                              output->embeddings, output->embedDim,
                                              model->useQuantization, model->useSIMD);
        break;

    default:
        fprintf(stderr, "Unknown fusion method: %d\n", model->fusionMethod);
        fusionSuccess = false;
    }

    /* Clean up */
    free(fusionDims);
    free(fusionInputs);
    for (int i = 0; i < model->numModalities; i++) {
        free(encoderOutputs[i]);
    }
    free(encoderOutputs);

    return fusionSuccess;
}

/**
 * Initialize multimodal input
 * @param input Multimodal input structure to initialize
 * @return true on success, false on failure
 */
bool tinyaiMultimodalInputInit(TinyAIMultimodalInput *input)
{
    if (!input) {
        return false;
    }

    /* Initialize to zeros */
    memset(input, 0, sizeof(TinyAIMultimodalInput));

    return true;
}

/**
 * Free multimodal input
 * @param input Multimodal input to free
 * @param freeContents Whether to free the contained inputs
 */
void tinyaiMultimodalInputFree(TinyAIMultimodalInput *input, bool freeContents)
{
    if (!input) {
        return;
    }

    if (freeContents) {
        /* Free text input */
        if (input->textInput) {
            free(input->textInput);
        }

        /* Free image input */
        if (input->imageInput) {
            tinyaiImageFree(input->imageInput);
        }

        /* Free audio input */
        if (input->audioInput) {
            free(input->audioInput);
        }
    }

    /* Clear the structure */
    memset(input, 0, sizeof(TinyAIMultimodalInput));
}

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
                                int vocabSize, int numClasses)
{
    if (!output || embedDim <= 0 || length <= 0) {
        return false;
    }

    /* Initialize the output structure */
    memset(output, 0, sizeof(TinyAIMultimodalOutput));
    output->embedDim   = embedDim;
    output->length     = length;
    output->vocabSize  = vocabSize;
    output->numClasses = numClasses;

    /* Allocate memory for embeddings */
    output->embeddings = (float *)malloc(embedDim * length * sizeof(float));
    if (!output->embeddings) {
        fprintf(stderr, "Failed to allocate embeddings\n");
        return false;
    }

    /* Allocate memory for text logits if needed */
    if (vocabSize > 0) {
        output->textLogits = (float *)malloc(vocabSize * sizeof(float));
        if (!output->textLogits) {
            fprintf(stderr, "Failed to allocate text logits\n");
            free(output->embeddings);
            output->embeddings = NULL;
            return false;
        }
    }

    /* Allocate memory for image features if needed */
    if (numClasses > 0) {
        output->imageFeatures = (float *)malloc(numClasses * sizeof(float));
        if (!output->imageFeatures) {
            fprintf(stderr, "Failed to allocate image features\n");
            if (output->textLogits) {
                free(output->textLogits);
                output->textLogits = NULL;
            }
            free(output->embeddings);
            output->embeddings = NULL;
            return false;
        }
    }

    return true;
}

/**
 * Free multimodal output
 * @param output Multimodal output to free
 */
void tinyaiMultimodalOutputFree(TinyAIMultimodalOutput *output)
{
    if (!output) {
        return;
    }

    /* Free embeddings */
    if (output->embeddings) {
        free(output->embeddings);
    }

    /* Free text logits */
    if (output->textLogits) {
        free(output->textLogits);
    }

    /* Free image features */
    if (output->imageFeatures) {
        free(output->imageFeatures);
    }

    /* Clear the structure */
    memset(output, 0, sizeof(TinyAIMultimodalOutput));
}

/**
 * Set custom memory pool for model
 * @param model The model to set memory pool for
 * @param memoryPool Memory pool to use
 * @return true on success, false on failure
 */
bool tinyaiMultimodalModelSetMemoryPool(TinyAIMultimodalModel *model, void *memoryPool)
{
    if (!model || !memoryPool) {
        return false;
    }

    /* Free existing memory pool if we own it */
    if (model->memoryPool && !model->useExternalMemory) {
        tinyaiMemoryPoolFree(model->memoryPool);
    }

    model->memoryPool        = memoryPool;
    model->useExternalMemory = true;

    return true;
}

/**
 * Enable or disable SIMD acceleration
 * @param model The model to configure
 * @param enable Whether to enable SIMD
 * @return true on success, false on failure
 */
bool tinyaiMultimodalModelEnableSIMD(TinyAIMultimodalModel *model, bool enable)
{
    if (!model) {
        return false;
    }

    model->useSIMD = enable;

    /* If using our own memory pool, update it */
    if (model->memoryPool && !model->useExternalMemory) {
        tinyaiMemoryPoolUpdateSIMD(model->memoryPool, enable);
    }

    return true;
}

/**
 * Get memory usage statistics
 * @param model The model to query
 * @param weightMemory Output parameter for weight memory (in bytes)
 * @param activationMemory Output parameter for activation memory (in bytes)
 * @return true on success, false on failure
 */
bool tinyaiMultimodalModelGetMemoryUsage(const TinyAIMultimodalModel *model, size_t *weightMemory,
                                         size_t *activationMemory)
{
    if (!model || !weightMemory || !activationMemory) {
        return false;
    }

    /* Calculate total weight memory */
    size_t totalWeightBytes = 0;

    /* Add memory for encoders */
    for (int i = 0; i < model->numModalities; i++) {
        totalWeightBytes +=
            model->modalityEncoders[i].weightBytes + model->modalityEncoders[i].biasBytes;
    }

    /* Add memory for fusion layers */
    for (int i = 0; i < model->numFusionLayers; i++) {
        totalWeightBytes += model->fusionLayers[i].weightBytes + model->fusionLayers[i].biasBytes;
    }

    /* Calculate maximum activation memory */
    size_t maxActivationBytes = 0;

    /* Consider encoder activations */
    for (int i = 0; i < model->numModalities; i++) {
        if (model->modalityEncoders[i].outputBytes > maxActivationBytes) {
            maxActivationBytes = model->modalityEncoders[i].outputBytes;
        }
    }

    /* Consider fusion layer activations */
    for (int i = 0; i < model->numFusionLayers; i++) {
        if (model->fusionLayers[i].outputBytes > maxActivationBytes) {
            maxActivationBytes = model->fusionLayers[i].outputBytes;
        }
    }

    *weightMemory     = totalWeightBytes;
    *activationMemory = maxActivationBytes;

    return true;
}
