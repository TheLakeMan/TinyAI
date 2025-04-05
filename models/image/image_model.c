/**
 * @file image_model.c
 * @brief Implementation of the image model functionality in TinyAI
 */

#include "image_model.h"
#include "../../core/memory.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Layer type definitions */
#define LAYER_TYPE_INPUT 0
#define LAYER_TYPE_CONV 1
#define LAYER_TYPE_DEPTHWISE 2
#define LAYER_TYPE_POOLING 3
#define LAYER_TYPE_DENSE 4
#define LAYER_TYPE_FLATTEN 5
#define LAYER_TYPE_DROPOUT 6
#define LAYER_TYPE_ACTIVATION 7

/* Activation type definitions */
#define ACTIVATION_NONE 0
#define ACTIVATION_RELU 1
#define ACTIVATION_SIGMOID 2
#define ACTIVATION_TANH 3

/* Forward declaration of Layer struct for internal use */
typedef struct Layer Layer;

/* Define the Layer structure */
struct Layer {
    int  type;
    char name[32];
    int  inputWidth;
    int  inputHeight;
    int  inputChannels;
    int  outputWidth;
    int  outputHeight;
    int  outputChannels;
    int  kernelSize;
    int  stride;
    int  padding;
    int  activation;

    /* Layer weights and parameters */
    uint8_t *weights; /* 4-bit quantized weights (if applicable) */
    float   *biases;  /* Biases for conv/dense layers */
    float   *scales;  /* Scale factors for quantized weights */

    /* Memory requirements */
    size_t weightBytes; /* Size of weights in bytes */
    size_t biasBytes;   /* Size of biases in bytes */
    size_t outputBytes; /* Size of output in bytes */
};

/* Define the Model structure */
struct TinyAIImageModel {
    int modelType;
    int inputWidth;
    int inputHeight;
    int inputChannels;
    int numClasses;

    /* Layers */
    Layer layers[50]; /* Assuming MAX_LAYERS = 50 */
    int   numLayers;

    /* Memory */
    void *memoryPool;
    bool  useExternalMemory;
    bool  useSIMD;
    bool  useQuantization;

    /* Labels */
    char **labels;
    int    numLabels;

    /* Preprocessing parameters */
    TinyAIImagePreprocessParams preprocess;
};

/* Private functions */

/**
 * Initialize a convolutional layer
 */
static bool initConvLayer(Layer *layer, int inputWidth, int inputHeight, int inputChannels,
                          int outputChannels, int kernelSize, int stride, int padding,
                          int activation, bool useQuantization)
{
    if (!layer) {
        return false;
    }

    /* Set basic layer info */
    memset(layer, 0, sizeof(Layer));
    layer->type = LAYER_TYPE_CONV;
    snprintf(layer->name, sizeof(layer->name), "conv_%dx%d_%d", kernelSize, kernelSize,
             outputChannels);

    /* Set dimensions */
    layer->inputWidth    = inputWidth;
    layer->inputHeight   = inputHeight;
    layer->inputChannels = inputChannels;
    layer->kernelSize    = kernelSize;
    layer->stride        = stride;
    layer->padding       = padding;
    layer->activation    = activation;

    /* Calculate output dimensions */
    layer->outputWidth    = (inputWidth + 2 * padding - kernelSize) / stride + 1;
    layer->outputHeight   = (inputHeight + 2 * padding - kernelSize) / stride + 1;
    layer->outputChannels = outputChannels;

    /* Calculate memory requirements */
    size_t numWeights = (size_t)kernelSize * kernelSize * inputChannels * outputChannels;
    if (useQuantization) {
        /* 4-bit quantization: 2 weights per byte */
        layer->weightBytes = (numWeights + 1) / 2;
    }
    else {
        /* Full precision: 4 bytes per weight */
        layer->weightBytes = numWeights * sizeof(float);
    }

    /* Biases and scales */
    layer->biasBytes = outputChannels * sizeof(float);

    /* Output buffer size */
    layer->outputBytes =
        (size_t)layer->outputWidth * layer->outputHeight * layer->outputChannels * sizeof(float);

    return true;
}

/**
 * Initialize a depthwise separable convolutional layer
 */
static bool initDepthwiseLayer(Layer *layer, int inputWidth, int inputHeight, int inputChannels,
                               int multiplier, int kernelSize, int stride, int padding,
                               int activation, bool useQuantization)
{
    if (!layer) {
        return false;
    }

    /* Set basic layer info */
    memset(layer, 0, sizeof(Layer));
    layer->type = LAYER_TYPE_DEPTHWISE;
    snprintf(layer->name, sizeof(layer->name), "depthwise_%dx%d_%d", kernelSize, kernelSize,
             multiplier);

    /* Set dimensions */
    layer->inputWidth    = inputWidth;
    layer->inputHeight   = inputHeight;
    layer->inputChannels = inputChannels;
    layer->kernelSize    = kernelSize;
    layer->stride        = stride;
    layer->padding       = padding;
    layer->activation    = activation;

    /* Calculate output dimensions */
    layer->outputWidth    = (inputWidth + 2 * padding - kernelSize) / stride + 1;
    layer->outputHeight   = (inputHeight + 2 * padding - kernelSize) / stride + 1;
    layer->outputChannels = inputChannels * multiplier;

    /* Calculate memory requirements */
    size_t numWeights = (size_t)kernelSize * kernelSize * inputChannels * multiplier;
    if (useQuantization) {
        /* 4-bit quantization: 2 weights per byte */
        layer->weightBytes = (numWeights + 1) / 2;
    }
    else {
        /* Full precision: 4 bytes per weight */
        layer->weightBytes = numWeights * sizeof(float);
    }

    /* Biases and scales */
    layer->biasBytes = layer->outputChannels * sizeof(float);

    /* Output buffer size */
    layer->outputBytes =
        (size_t)layer->outputWidth * layer->outputHeight * layer->outputChannels * sizeof(float);

    return true;
}

/**
 * Initialize a pooling layer
 */
static bool initPoolingLayer(Layer *layer, int inputWidth, int inputHeight, int inputChannels,
                             int kernelSize, int stride, int padding)
{
    if (!layer) {
        return false;
    }

    /* Set basic layer info */
    memset(layer, 0, sizeof(Layer));
    layer->type = LAYER_TYPE_POOLING;
    snprintf(layer->name, sizeof(layer->name), "pool_%dx%d", kernelSize, kernelSize);

    /* Set dimensions */
    layer->inputWidth    = inputWidth;
    layer->inputHeight   = inputHeight;
    layer->inputChannels = inputChannels;
    layer->kernelSize    = kernelSize;
    layer->stride        = stride;
    layer->padding       = padding;
    layer->activation    = ACTIVATION_NONE;

    /* Calculate output dimensions */
    layer->outputWidth    = (inputWidth + 2 * padding - kernelSize) / stride + 1;
    layer->outputHeight   = (inputHeight + 2 * padding - kernelSize) / stride + 1;
    layer->outputChannels = inputChannels;

    /* No weights in pooling layer */
    layer->weightBytes = 0;
    layer->biasBytes   = 0;

    /* Output buffer size */
    layer->outputBytes =
        (size_t)layer->outputWidth * layer->outputHeight * layer->outputChannels * sizeof(float);

    return true;
}

/**
 * Initialize a fully connected (dense) layer
 */
static bool initDenseLayer(Layer *layer, int inputSize, int outputSize, int activation,
                           bool useQuantization)
{
    if (!layer) {
        return false;
    }

    /* Set basic layer info */
    memset(layer, 0, sizeof(Layer));
    layer->type = LAYER_TYPE_DENSE;
    snprintf(layer->name, sizeof(layer->name), "dense_%d", outputSize);

    /* Set dimensions
     * For dense layers, we represent the input and output as 1D */
    layer->inputWidth     = inputSize;
    layer->inputHeight    = 1;
    layer->inputChannels  = 1;
    layer->outputWidth    = outputSize;
    layer->outputHeight   = 1;
    layer->outputChannels = 1;
    layer->activation     = activation;

    /* Calculate memory requirements */
    size_t numWeights = (size_t)inputSize * outputSize;
    if (useQuantization) {
        /* 4-bit quantization: 2 weights per byte */
        layer->weightBytes = (numWeights + 1) / 2;
    }
    else {
        /* Full precision: 4 bytes per weight */
        layer->weightBytes = numWeights * sizeof(float);
    }

    /* Biases */
    layer->biasBytes = outputSize * sizeof(float);

    /* Output buffer size */
    layer->outputBytes = outputSize * sizeof(float);

    return true;
}

/**
 * Initialize a flatten layer
 */
static bool initFlattenLayer(Layer *layer, int inputWidth, int inputHeight, int inputChannels)
{
    if (!layer) {
        return false;
    }

    /* Set basic layer info */
    memset(layer, 0, sizeof(Layer));
    layer->type = LAYER_TYPE_FLATTEN;
    snprintf(layer->name, sizeof(layer->name), "flatten");

    /* Set dimensions */
    layer->inputWidth     = inputWidth;
    layer->inputHeight    = inputHeight;
    layer->inputChannels  = inputChannels;
    layer->outputWidth    = inputWidth * inputHeight * inputChannels;
    layer->outputHeight   = 1;
    layer->outputChannels = 1;
    layer->activation     = ACTIVATION_NONE;

    /* No weights in flatten layer */
    layer->weightBytes = 0;
    layer->biasBytes   = 0;

    /* Output buffer size */
    layer->outputBytes = (size_t)layer->outputWidth * sizeof(float);

    return true;
}

/**
 * Create a MobileNet model
 */
static TinyAIImageModel *createMobileNetModel(const TinyAIImageModelParams *params)
{
    if (!params || params->inputWidth <= 0 || params->inputHeight <= 0 ||
        params->inputChannels <= 0 || params->numClasses <= 0) {
        return NULL;
    }

    /* Allocate model structure */
    TinyAIImageModel *model = (TinyAIImageModel *)malloc(sizeof(TinyAIImageModel));
    if (!model) {
        fprintf(stderr, "Failed to allocate model structure\n");
        return NULL;
    }

    /* Initialize the model */
    memset(model, 0, sizeof(TinyAIImageModel));
    model->modelType       = TINYAI_IMAGE_MODEL_MOBILENET;
    model->inputWidth      = params->inputWidth;
    model->inputHeight     = params->inputHeight;
    model->inputChannels   = params->inputChannels;
    model->numClasses      = params->numClasses;
    model->useQuantization = params->useQuantization;
    model->useSIMD         = params->useSIMD;

    /* Set default preprocessing parameters */
    tinyaiImagePreprocessParamsDefault(&model->preprocess);
    model->preprocess.targetWidth  = params->inputWidth;
    model->preprocess.targetHeight = params->inputHeight;

    /* MobileNet standard preprocessing */
    model->preprocess.meanR = 127.5f;
    model->preprocess.meanG = 127.5f;
    model->preprocess.meanB = 127.5f;
    model->preprocess.stdR  = 127.5f;
    model->preprocess.stdG  = 127.5f;
    model->preprocess.stdB  = 127.5f;

    /* Configure the MobileNet architecture */
    int layerIdx    = 0;
    int curWidth    = params->inputWidth;
    int curHeight   = params->inputHeight;
    int curChannels = params->inputChannels;

    /* Initial convolution layer */
    if (!initConvLayer(&model->layers[layerIdx++], curWidth, curHeight, curChannels, 32, 3, 2, 1,
                       ACTIVATION_RELU, model->useQuantization)) {
        free(model);
        return NULL;
    }

    /* Update dimensions after first conv */
    curWidth    = model->layers[layerIdx - 1].outputWidth;
    curHeight   = model->layers[layerIdx - 1].outputHeight;
    curChannels = model->layers[layerIdx - 1].outputChannels;

    /* First depthwise separable block */
    if (!initDepthwiseLayer(&model->layers[layerIdx++], curWidth, curHeight, curChannels, 1, 3, 1,
                            1, ACTIVATION_RELU, model->useQuantization)) {
        free(model);
        return NULL;
    }

    curWidth    = model->layers[layerIdx - 1].outputWidth;
    curHeight   = model->layers[layerIdx - 1].outputHeight;
    curChannels = model->layers[layerIdx - 1].outputChannels;

    if (!initConvLayer(&model->layers[layerIdx++], curWidth, curHeight, curChannels, 64, 1, 1, 0,
                       ACTIVATION_RELU, model->useQuantization)) {
        free(model);
        return NULL;
    }

    curWidth    = model->layers[layerIdx - 1].outputWidth;
    curHeight   = model->layers[layerIdx - 1].outputHeight;
    curChannels = model->layers[layerIdx - 1].outputChannels;

    /* Second depthwise separable block with stride 2 */
    if (!initDepthwiseLayer(&model->layers[layerIdx++], curWidth, curHeight, curChannels, 1, 3, 2,
                            1, ACTIVATION_RELU, model->useQuantization)) {
        free(model);
        return NULL;
    }

    curWidth    = model->layers[layerIdx - 1].outputWidth;
    curHeight   = model->layers[layerIdx - 1].outputHeight;
    curChannels = model->layers[layerIdx - 1].outputChannels;

    if (!initConvLayer(&model->layers[layerIdx++], curWidth, curHeight, curChannels, 128, 1, 1, 0,
                       ACTIVATION_RELU, model->useQuantization)) {
        free(model);
        return NULL;
    }

    curWidth    = model->layers[layerIdx - 1].outputWidth;
    curHeight   = model->layers[layerIdx - 1].outputHeight;
    curChannels = model->layers[layerIdx - 1].outputChannels;

    /* Third depthwise separable block */
    if (!initDepthwiseLayer(&model->layers[layerIdx++], curWidth, curHeight, curChannels, 1, 3, 1,
                            1, ACTIVATION_RELU, model->useQuantization)) {
        free(model);
        return NULL;
    }

    curWidth    = model->layers[layerIdx - 1].outputWidth;
    curHeight   = model->layers[layerIdx - 1].outputHeight;
    curChannels = model->layers[layerIdx - 1].outputChannels;

    if (!initConvLayer(&model->layers[layerIdx++], curWidth, curHeight, curChannels, 128, 1, 1, 0,
                       ACTIVATION_RELU, model->useQuantization)) {
        free(model);
        return NULL;
    }

    curWidth    = model->layers[layerIdx - 1].outputWidth;
    curHeight   = model->layers[layerIdx - 1].outputHeight;
    curChannels = model->layers[layerIdx - 1].outputChannels;

    /* Fourth depthwise separable block with stride 2 */
    if (!initDepthwiseLayer(&model->layers[layerIdx++], curWidth, curHeight, curChannels, 1, 3, 2,
                            1, ACTIVATION_RELU, model->useQuantization)) {
        free(model);
        return NULL;
    }

    curWidth    = model->layers[layerIdx - 1].outputWidth;
    curHeight   = model->layers[layerIdx - 1].outputHeight;
    curChannels = model->layers[layerIdx - 1].outputChannels;

    if (!initConvLayer(&model->layers[layerIdx++], curWidth, curHeight, curChannels, 256, 1, 1, 0,
                       ACTIVATION_RELU, model->useQuantization)) {
        free(model);
        return NULL;
    }

    curWidth    = model->layers[layerIdx - 1].outputWidth;
    curHeight   = model->layers[layerIdx - 1].outputHeight;
    curChannels = model->layers[layerIdx - 1].outputChannels;

    /* Fifth depthwise separable block */
    if (!initDepthwiseLayer(&model->layers[layerIdx++], curWidth, curHeight, curChannels, 1, 3, 1,
                            1, ACTIVATION_RELU, model->useQuantization)) {
        free(model);
        return NULL;
    }

    curWidth    = model->layers[layerIdx - 1].outputWidth;
    curHeight   = model->layers[layerIdx - 1].outputHeight;
    curChannels = model->layers[layerIdx - 1].outputChannels;

    if (!initConvLayer(&model->layers[layerIdx++], curWidth, curHeight, curChannels, 256, 1, 1, 0,
                       ACTIVATION_RELU, model->useQuantization)) {
        free(model);
        return NULL;
    }

    curWidth    = model->layers[layerIdx - 1].outputWidth;
    curHeight   = model->layers[layerIdx - 1].outputHeight;
    curChannels = model->layers[layerIdx - 1].outputChannels;

    /* Global average pooling */
    if (!initPoolingLayer(&model->layers[layerIdx++], curWidth, curHeight, curChannels, curWidth, 1,
                          0)) {
        free(model);
        return NULL;
    }

    curWidth    = model->layers[layerIdx - 1].outputWidth;
    curHeight   = model->layers[layerIdx - 1].outputHeight;
    curChannels = model->layers[layerIdx - 1].outputChannels;

    /* Flatten */
    if (!initFlattenLayer(&model->layers[layerIdx++], curWidth, curHeight, curChannels)) {
        free(model);
        return NULL;
    }

    int flattenSize = model->layers[layerIdx - 1].outputWidth;

    /* Final dense layer */
    if (!initDenseLayer(&model->layers[layerIdx++], flattenSize, params->numClasses,
                        ACTIVATION_NONE, model->useQuantization)) {
        free(model);
        return NULL;
    }

    /* Set the layer count */
    model->numLayers = layerIdx;

    /* Calculate total memory requirements */
    size_t totalWeightBytes     = 0;
    size_t totalActivationBytes = 0;

    for (int i = 0; i < model->numLayers; i++) {
        totalWeightBytes += model->layers[i].weightBytes + model->layers[i].biasBytes;
        if (model->layers[i].outputBytes > totalActivationBytes) {
            totalActivationBytes = model->layers[i].outputBytes;
        }
    }

    /* Allocate memory pool if needed */
    if (!model->memoryPool && !params->customParams) {
        model->memoryPool =
            tinyaiMemoryPoolCreate(totalWeightBytes, totalActivationBytes, model->useSIMD);
        if (!model->memoryPool) {
            fprintf(stderr, "Failed to allocate memory pool\n");
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
        /* Load weights from file, handled by model_loader.c */
        /* This will be implemented in tinyaiLoadModelWeights */
    }

    /* Load labels if specified */
    if (params->labelsFile) {
        /* TODO: Implement label loading from file */
    }

    return model;
}

/**
 * Create a TinyCNN model (simpler model for testing)
 */
static TinyAIImageModel *createTinyCNNModel(const TinyAIImageModelParams *params)
{
    if (!params || params->inputWidth <= 0 || params->inputHeight <= 0 ||
        params->inputChannels <= 0 || params->numClasses <= 0) {
        return NULL;
    }

    /* Allocate model structure */
    TinyAIImageModel *model = (TinyAIImageModel *)malloc(sizeof(TinyAIImageModel));
    if (!model) {
        fprintf(stderr, "Failed to allocate model structure\n");
        return NULL;
    }

    /* Initialize the model */
    memset(model, 0, sizeof(TinyAIImageModel));
    model->modelType       = TINYAI_IMAGE_MODEL_TINY_CNN;
    model->inputWidth      = params->inputWidth;
    model->inputHeight     = params->inputHeight;
    model->inputChannels   = params->inputChannels;
    model->numClasses      = params->numClasses;
    model->useQuantization = params->useQuantization;
    model->useSIMD         = params->useSIMD;

    /* Set default preprocessing parameters */
    tinyaiImagePreprocessParamsDefault(&model->preprocess);
    model->preprocess.targetWidth  = params->inputWidth;
    model->preprocess.targetHeight = params->inputHeight;

    /* Configure the TinyCNN architecture (simpler than MobileNet) */
    int layerIdx    = 0;
    int curWidth    = params->inputWidth;
    int curHeight   = params->inputHeight;
    int curChannels = params->inputChannels;

    /* First conv layer */
    if (!initConvLayer(&model->layers[layerIdx++], curWidth, curHeight, curChannels, 16, 3, 1, 1,
                       ACTIVATION_RELU, model->useQuantization)) {
        free(model);
        return NULL;
    }

    curWidth    = model->layers[layerIdx - 1].outputWidth;
    curHeight   = model->layers[layerIdx - 1].outputHeight;
    curChannels = model->layers[layerIdx - 1].outputChannels;

    /* Max pooling */
    if (!initPoolingLayer(&model->layers[layerIdx++], curWidth, curHeight, curChannels, 2, 2, 0)) {
        free(model);
        return NULL;
    }

    curWidth    = model->layers[layerIdx - 1].outputWidth;
    curHeight   = model->layers[layerIdx - 1].outputHeight;
    curChannels = model->layers[layerIdx - 1].outputChannels;

    /* Second conv layer */
    if (!initConvLayer(&model->layers[layerIdx++], curWidth, curHeight, curChannels, 32, 3, 1, 1,
                       ACTIVATION_RELU, model->useQuantization)) {
        free(model);
        return NULL;
    }

    curWidth    = model->layers[layerIdx - 1].outputWidth;
    curHeight   = model->layers[layerIdx - 1].outputHeight;
    curChannels = model->layers[layerIdx - 1].outputChannels;

    /* Max pooling */
    if (!initPoolingLayer(&model->layers[layerIdx++], curWidth, curHeight, curChannels, 2, 2, 0)) {
        free(model);
        return NULL;
    }

    curWidth    = model->layers[layerIdx - 1].outputWidth;
    curHeight   = model->layers[layerIdx - 1].outputHeight;
    curChannels = model->layers[layerIdx - 1].outputChannels;

    /* Flatten */
    if (!initFlattenLayer(&model->layers[layerIdx++], curWidth, curHeight, curChannels)) {
        free(model);
        return NULL;
    }

    int flattenSize = model->layers[layerIdx - 1].outputWidth;

    /* Dense layer */
    if (!initDenseLayer(&model->layers[layerIdx++], flattenSize, 64, ACTIVATION_RELU,
                        model->useQuantization)) {
        free(model);
        return NULL;
    }

    /* Final dense layer */
    if (!initDenseLayer(&model->layers[layerIdx++], 64, params->numClasses, ACTIVATION_NONE,
                        model->useQuantization)) {
        free(model);
        return NULL;
    }

    /* Set the layer count */
    model->numLayers = layerIdx;

    /* Calculate total memory requirements */
    size_t totalWeightBytes     = 0;
    size_t totalActivationBytes = 0;

    for (int i = 0; i < model->numLayers; i++) {
        totalWeightBytes += model->layers[i].weightBytes + model->layers[i].biasBytes;
        if (model->layers[i].outputBytes > totalActivationBytes) {
            totalActivationBytes = model->layers[i].outputBytes;
        }
    }

    /* Allocate memory pool if needed */
    if (!model->memoryPool && !params->customParams) {
        model->memoryPool =
            tinyaiMemoryPoolCreate(totalWeightBytes, totalActivationBytes, model->useSIMD);
        if (!model->memoryPool) {
            fprintf(stderr, "Failed to allocate memory pool\n");
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

    return model;
}

/* Public API functions */

/**
 * Create an image model
 * @param params Parameters for model creation
 * @return Newly allocated model, or NULL on failure
 */
TinyAIImageModel *tinyaiImageModelCreate(const TinyAIImageModelParams *params)
{
    if (!params) {
        return NULL;
    }

    switch (params->modelType) {
    case TINYAI_IMAGE_MODEL_MOBILENET:
        return createMobileNetModel(params);

    case TINYAI_IMAGE_MODEL_TINY_CNN:
        return createTinyCNNModel(params);

    case TINYAI_IMAGE_MODEL_EFFICIENTNET:
        /* Not yet implemented */
        fprintf(stderr, "EfficientNet model not yet implemented\n");
        return NULL;

    case TINYAI_IMAGE_MODEL_CUSTOM:
        /* Custom model implementation would go here */
        fprintf(stderr, "Custom model requires implementation\n");
        return NULL;

    default:
        fprintf(stderr, "Unknown model type\n");
        return NULL;
    }
}

/**
 * Free an image model
 * @param model The model to free
 */
void tinyaiImageModelFree(TinyAIImageModel *model)
{
    if (!model) {
        return;
    }

    /* Free labels if we have them */
    if (model->labels) {
        for (int i = 0; i < model->numLabels; i++) {
            if (model->labels[i]) {
                free(model->labels[i]);
            }
        }
        free(model->labels);
    }

    /* Free memory pool if we own it */
    if (model->memoryPool && !model->useExternalMemory) {
        tinyaiMemoryPoolFree(model->memoryPool);
    }

    /* Free the model structure */
    free(model);
}

/**
 * Classify an image using the model
 * @param model The model to use for classification
 * @param image The image to classify
 * @param topK Number of top results to return
 * @param results Array to store results (must be pre-allocated for topK results)
 * @return Number of results on success, negative on failure
 */
int tinyaiImageModelClassify(TinyAIImageModel *model, const TinyAIImage *image, int topK,
                             TinyAIImageClassResult *results)
{
    if (!model || !image || !results || topK <= 0) {
        return -1;
    }

    /* Convert image to float array compatible with model input */
    float *imageData = (float *)malloc(model->inputWidth * model->inputHeight *
                                       model->inputChannels * sizeof(float));
    if (!imageData) {
        fprintf(stderr, "Failed to allocate memory for image data\n");
        return -1;
    }

    /* Preprocess image if needed */
    TinyAIImage *preprocessedImage = tinyaiImagePreprocess(image, &model->preprocess);
    if (!preprocessedImage) {
        fprintf(stderr, "Failed to preprocess image\n");
        free(imageData);
        return -1;
    }

    /* Convert image to float array */
    if (!tinyaiImageToFloatArray(preprocessedImage, imageData, true)) {
        fprintf(stderr, "Failed to convert image to float array\n");
        tinyaiImageFree(preprocessedImage);
        free(imageData);
        return -1;
    }

    /* If we created a new image for preprocessing, free it */
    if (preprocessedImage != image) {
        tinyaiImageFree(preprocessedImage);
    }

    /* Allocate memory for model output */
    float *outputData = (float *)malloc(model->numClasses * sizeof(float));
    if (!outputData) {
        fprintf(stderr, "Failed to allocate memory for model output\n");
        free(imageData);
        return -1;
    }

    /* Run forward pass */
    if (!tinyaiImageModelForwardPass(model, imageData, outputData)) {
        fprintf(stderr, "Forward pass failed\n");
        free(outputData);
        free(imageData);
        return -1;
    }

    /* Copy results to the provided array */
    /* For now, we're just taking the raw output values as confidences */
    /* In a real implementation, you might want to apply softmax here */

    /* Create array of indices for sorting */
    int *indices = (int *)malloc(model->numClasses * sizeof(int));
    if (!indices) {
        fprintf(stderr, "Failed to allocate memory for sorting indices\n");
        free(outputData);
        free(imageData);
        return -1;
    }

    /* Initialize indices */
    for (int i = 0; i < model->numClasses; i++) {
        indices[i] = i;
    }

    /* Simple bubble sort to get top-K results (more efficient sorting could be used) */
    for (int i = 0; i < model->numClasses - 1; i++) {
        for (int j = 0; j < model->numClasses - i - 1; j++) {
            if (outputData[indices[j]] < outputData[indices[j + 1]]) {
                /* Swap indices */
                int temp       = indices[j];
                indices[j]     = indices[j + 1];
                indices[j + 1] = temp;
            }
        }
    }

    /* Fill in results with top-K classes */
    int numResults = (topK < model->numClasses) ? topK : model->numClasses;
    for (int i = 0; i < numResults; i++) {
        int classIdx          = indices[i];
        results[i].classId    = classIdx;
        results[i].confidence = outputData[classIdx];
        results[i].label =
            (model->labels && classIdx < model->numLabels) ? model->labels[classIdx] : NULL;
    }

    /* Free allocated memory */
    free(indices);
    free(outputData);
    free(imageData);

    return numResults;
}

/**
 * Set custom memory pool for model
 * @param model The model to set memory pool for
 * @param memoryPool Memory pool to use
 * @return true on success, false on failure
 */
bool tinyaiImageModelSetMemoryPool(TinyAIImageModel *model, void *memoryPool)
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
bool tinyaiImageModelEnableSIMD(TinyAIImageModel *model, bool enable)
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
bool tinyaiImageModelGetMemoryUsage(const TinyAIImageModel *model, size_t *weightMemory,
                                    size_t *activationMemory)
{
    if (!model || !weightMemory || !activationMemory) {
        return false;
    }

    /* Calculate total weight memory */
    size_t totalWeightBytes = 0;
    for (int i = 0; i < model->numLayers; i++) {
        totalWeightBytes += model->layers[i].weightBytes + model->layers[i].biasBytes;
    }

    /* Calculate maximum activation memory */
    size_t maxActivationBytes = 0;
    for (int i = 0; i < model->numLayers; i++) {
        if (model->layers[i].outputBytes > maxActivationBytes) {
            maxActivationBytes = model->layers[i].outputBytes;
        }
    }

    *weightMemory     = totalWeightBytes;
    *activationMemory = maxActivationBytes;

    return true;
}

/**
 * Get preprocessing parameters
 * @param model The model to query
 * @param params Output parameter for preprocessing parameters
 * @return true on success, false on failure
 */
bool tinyaiImageModelGetPreprocessParams(const TinyAIImageModel      *model,
                                         TinyAIImagePreprocessParams *params)
{
    if (!model || !params) {
        return false;
    }

    *params = model->preprocess;
    return true;
}

/**
 * Forward pass function declaration - implemented in forward_pass.c
 */
bool tinyaiImageModelForwardPass(const TinyAIImageModel *model, const float *input, float *output);
