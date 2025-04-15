/**
 * @file model_loader.c
 * @brief Utilities for loading pre-trained model weights in TinyAI
 */

#include "model_loader.h"
#include "../core/memory.h"
#include "../models/image/image_model.h"
#include "quantize.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Forward declaration for types from image_model.h */
typedef struct TinyAIImagePreprocessParams TinyAIImagePreprocessParams;

/* This struct declaration is matching the internal struct from image_model.c */
typedef struct Layer {
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
} Layer;

/**
 * Header for TinyAI model weight files
 */
typedef struct {
    char     magic[8];    /* Magic string "TINYAIWT" */
    uint32_t version;     /* Version number (currently 1) */
    uint32_t modelType;   /* Type of model (matches TinyAIImageModelType) */
    uint32_t inputWidth;  /* Input width */
    uint32_t inputHeight; /* Input height */
    uint32_t channels;    /* Input channels */
    uint32_t numLayers;   /* Number of layers */
    uint32_t numClasses;  /* Number of output classes */
    uint8_t  quantized;   /* Whether weights are quantized (0=full precision, 1=4-bit) */
    uint8_t  padding[3];  /* Padding to align to 4 bytes */
} TinyAIModelHeader;

/**
 * Layer information in weight file
 */
typedef struct {
    uint32_t layerType;    /* Type of layer (matches LayerType enum) */
    uint32_t inputDim[3];  /* Input dimensions [width, height, channels] */
    uint32_t outputDim[3]; /* Output dimensions [width, height, channels] */
    uint32_t kernelSize;   /* Kernel size for conv layers */
    uint32_t stride;       /* Stride for conv/pool layers */
    uint32_t padding;      /* Padding for conv layers */
    uint32_t weightsSize;  /* Size of weights blob in bytes */
    uint32_t biasSize;     /* Size of bias blob in bytes */
} TinyAILayerInfo;

/* External functions from image_model.c that we need */
extern int   tinyaiGetImageModelType(const TinyAIImageModel *model);
extern int   tinyaiGetImageModelInputWidth(const TinyAIImageModel *model);
extern int   tinyaiGetImageModelInputHeight(const TinyAIImageModel *model);
extern int   tinyaiGetImageModelInputChannels(const TinyAIImageModel *model);
extern int   tinyaiGetImageModelNumClasses(const TinyAIImageModel *model);
extern int   tinyaiGetImageModelNumLayers(const TinyAIImageModel *model);
extern bool  tinyaiIsImageModelQuantized(const TinyAIImageModel *model);
extern void *tinyaiGetImageModelMemoryPool(const TinyAIImageModel *model);
extern bool  tinyaiGetImageModelUseSIMD(const TinyAIImageModel *model);
extern bool  tinyaiGetLayerInfo(const TinyAIImageModel *model, int layerIdx, Layer *layer);
extern bool  tinyaiSetLayerWeights(TinyAIImageModel *model, int layerIdx, void *weights,
                                   size_t weightsSize);
extern bool  tinyaiSetLayerBiases(TinyAIImageModel *model, int layerIdx, float *biases,
                                  size_t biasSize);
extern bool  tinyaiSetLayerScales(TinyAIImageModel *model, int layerIdx, float *scales,
                                  size_t scaleSize);

/**
 * Validate if a model is compatible with weight file
 * @param model The model to validate
 * @param header Header from weight file
 * @return true if compatible, false otherwise
 */
static bool validateModelCompatibility(TinyAIImageModel *model, const TinyAIModelHeader *header)
{
    if (!model || !header) {
        return false;
    }

    /* Check basic compatibility */
    if (tinyaiGetImageModelType(model) != header->modelType) {
        fprintf(stderr, "Model type mismatch: %d vs %d\n", tinyaiGetImageModelType(model),
                header->modelType);
        return false;
    }

    if (tinyaiGetImageModelInputWidth(model) != header->inputWidth ||
        tinyaiGetImageModelInputHeight(model) != header->inputHeight ||
        tinyaiGetImageModelInputChannels(model) != header->channels) {
        fprintf(stderr, "Input dimensions mismatch: %dx%dx%d vs %dx%dx%d\n",
                tinyaiGetImageModelInputWidth(model), tinyaiGetImageModelInputHeight(model),
                tinyaiGetImageModelInputChannels(model), header->inputWidth, header->inputHeight,
                header->channels);
        return false;
    }

    if (tinyaiGetImageModelNumClasses(model) != header->numClasses) {
        fprintf(stderr, "Number of classes mismatch: %d vs %d\n",
                tinyaiGetImageModelNumClasses(model), header->numClasses);
        return false;
    }

    /* Additional validation can be added here */
    return true;
}

/**
 * Load model weights from a file
 * @param model The model to load weights into
 * @param filepath Path to the weights file
 * @param convertPrecision Whether to convert between precisions if needed
 * @return true on success, false on failure
 */
bool tinyaiLoadModelWeights(TinyAIImageModel *model, const char *filepath, bool convertPrecision)
{
    if (!model || !filepath) {
        return false;
    }

    FILE *file = fopen(filepath, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open weights file: %s\n", filepath);
        return false;
    }

    /* Read header */
    TinyAIModelHeader header;
    if (fread(&header, sizeof(header), 1, file) != 1) {
        fprintf(stderr, "Failed to read header from weights file\n");
        fclose(file);
        return false;
    }

    /* Validate magic string */
    if (strncmp(header.magic, "TINYAIWT", 8) != 0) {
        fprintf(stderr, "Invalid magic string in weights file\n");
        fclose(file);
        return false;
    }

    /* Validate compatibility */
    if (!validateModelCompatibility(model, &header)) {
        fprintf(stderr, "Model is not compatible with weights file\n");
        fclose(file);
        return false;
    }

    /* Check if we need to convert precision */
    bool needsConversion = (tinyaiIsImageModelQuantized(model) != (header.quantized == 1));
    if (needsConversion && !convertPrecision) {
        fprintf(stderr, "Precision mismatch and conversion not allowed\n");
        fclose(file);
        return false;
    }

    /* Load each layer */
    for (int i = 0; i < header.numLayers; i++) {
        TinyAILayerInfo layerInfo;

        /* Read layer info */
        if (fread(&layerInfo, sizeof(layerInfo), 1, file) != 1) {
            fprintf(stderr, "Failed to read layer info for layer %d\n", i);
            fclose(file);
            return false;
        }

        /* Skip layers with no weights (e.g. pooling, flatten) */
        if (layerInfo.weightsSize == 0 && layerInfo.biasSize == 0) {
            continue;
        }

        /* Find matching layer in the model */
        bool layerFound = false;
        int  numLayers  = tinyaiGetImageModelNumLayers(model);

        for (int j = 0; j < numLayers; j++) {
            Layer layer;
            if (!tinyaiGetLayerInfo(model, j, &layer)) {
                continue;
            }

            /* Check if layer dimensions match */
            if (layer.type == layerInfo.layerType && layer.inputWidth == layerInfo.inputDim[0] &&
                layer.inputHeight == layerInfo.inputDim[1] &&
                layer.inputChannels == layerInfo.inputDim[2] &&
                layer.outputWidth == layerInfo.outputDim[0] &&
                layer.outputHeight == layerInfo.outputDim[1] &&
                layer.outputChannels == layerInfo.outputDim[2]) {

                /* We found a matching layer */
                layerFound = true;

                /* Handle weights */
                if (layerInfo.weightsSize > 0) {
                    void *memoryPool = tinyaiGetImageModelMemoryPool(model);
                    bool  useSIMD    = tinyaiGetImageModelUseSIMD(model);

                    if (needsConversion) {
                        /* Need to convert between quantized and full precision */
                        if (tinyaiIsImageModelQuantized(model)) {
                            /* Convert from full precision to 4-bit */
                            float *tempWeights = (float *)malloc(layerInfo.weightsSize);
                            if (!tempWeights) {
                                fprintf(stderr, "Failed to allocate temp memory for conversion\n");
                                fclose(file);
                                return false;
                            }

                            /* Read full precision weights */
                            if (fread(tempWeights, 1, layerInfo.weightsSize, file) !=
                                layerInfo.weightsSize) {
                                fprintf(stderr, "Failed to read weights for layer %d\n", i);
                                free(tempWeights);
                                fclose(file);
                                return false;
                            }

                            /* Quantize to 4-bit */
                            size_t   numWeights       = layerInfo.weightsSize / sizeof(float);
                            uint8_t *quantizedWeights = (uint8_t *)malloc((numWeights + 1) / 2);
                            if (!quantizedWeights) {
                                fprintf(stderr, "Failed to allocate quantized weights\n");
                                free(tempWeights);
                                fclose(file);
                                return false;
                            }

                            tinyaiQuantizeWeights(tempWeights, quantizedWeights, numWeights);
                            tinyaiSetLayerWeights(model, j, quantizedWeights, (numWeights + 1) / 2);

                            free(quantizedWeights);
                            free(tempWeights);
                        }
                        else {
                            /* Convert from 4-bit to full precision */
                            uint8_t *tempWeights = (uint8_t *)malloc(layerInfo.weightsSize);
                            if (!tempWeights) {
                                fprintf(stderr, "Failed to allocate temp memory for conversion\n");
                                fclose(file);
                                return false;
                            }

                            /* Read quantized weights */
                            if (fread(tempWeights, 1, layerInfo.weightsSize, file) !=
                                layerInfo.weightsSize) {
                                fprintf(stderr, "Failed to read weights for layer %d\n", i);
                                free(tempWeights);
                                fclose(file);
                                return false;
                            }

                            /* Dequantize to float */
                            size_t numWeights =
                                layerInfo.weightsSize * 2; /* 4-bit: 2 weights per byte */
                            float *dequantizedWeights = (float *)malloc(numWeights * sizeof(float));
                            if (!dequantizedWeights) {
                                fprintf(stderr, "Failed to allocate dequantized weights\n");
                                free(tempWeights);
                                fclose(file);
                                return false;
                            }

                            tinyaiDequantizeWeights(tempWeights, dequantizedWeights, numWeights);
                            tinyaiSetLayerWeights(model, j, dequantizedWeights,
                                                  numWeights * sizeof(float));

                            free(dequantizedWeights);
                            free(tempWeights);
                        }
                    }
                    else {
                        /* No conversion needed, just read the weights directly */
                        void *weights = malloc(layerInfo.weightsSize);
                        if (!weights) {
                            fprintf(stderr, "Failed to allocate weights memory\n");
                            fclose(file);
                            return false;
                        }

                        if (fread(weights, 1, layerInfo.weightsSize, file) !=
                            layerInfo.weightsSize) {
                            fprintf(stderr, "Failed to read weights for layer %d\n", i);
                            free(weights);
                            fclose(file);
                            return false;
                        }

                        /* Set weights into the model */
                        tinyaiSetLayerWeights(model, j, weights, layerInfo.weightsSize);
                        free(weights);
                    }
                }

                /* Handle biases */
                if (layerInfo.biasSize > 0) {
                    float *biases = (float *)malloc(layerInfo.biasSize);
                    if (!biases) {
                        fprintf(stderr, "Failed to allocate biases memory\n");
                        fclose(file);
                        return false;
                    }

                    /* Read biases (always in full precision) */
                    if (fread(biases, 1, layerInfo.biasSize, file) != layerInfo.biasSize) {
                        fprintf(stderr, "Failed to read biases for layer %d\n", i);
                        free(biases);
                        fclose(file);
                        return false;
                    }

                    /* Set biases into the model */
                    tinyaiSetLayerBiases(model, j, biases, layerInfo.biasSize);
                    free(biases);
                }

                break; /* Found and processed the layer, move to next */
            }
        }

        if (!layerFound) {
            fprintf(stderr, "No matching layer found for layer %d in weights file\n", i);
            fclose(file);
            return false;
        }
    }

    fclose(file);
    return true;
}

/**
 * Save model weights to a file
 * @param model The model to save weights from
 * @param filepath Path to save the weights to
 * @return true on success, false on failure
 */
bool tinyaiSaveModelWeights(TinyAIImageModel *model, const char *filepath)
{
    if (!model || !filepath) {
        return false;
    }

    FILE *file = fopen(filepath, "wb");
    if (!file) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filepath);
        return false;
    }

    /* Prepare header */
    TinyAIModelHeader header;
    memset(&header, 0, sizeof(header));

    /* Set magic string and version */
    memcpy(header.magic, "TINYAIWT", 8);
    header.version = 1;

    /* Set model info */
    header.modelType   = tinyaiGetImageModelType(model);
    header.inputWidth  = tinyaiGetImageModelInputWidth(model);
    header.inputHeight = tinyaiGetImageModelInputHeight(model);
    header.channels    = tinyaiGetImageModelInputChannels(model);
    header.numLayers   = tinyaiGetImageModelNumLayers(model);
    header.numClasses  = tinyaiGetImageModelNumClasses(model);
    header.quantized   = tinyaiIsImageModelQuantized(model) ? 1 : 0;

    /* Write header */
    if (fwrite(&header, sizeof(header), 1, file) != 1) {
        fprintf(stderr, "Failed to write header\n");
        fclose(file);
        return false;
    }

    /* Write each layer */
    int numLayers = tinyaiGetImageModelNumLayers(model);
    for (int i = 0; i < numLayers; i++) {
        Layer layer;
        if (!tinyaiGetLayerInfo(model, i, &layer)) {
            fprintf(stderr, "Failed to get layer info for layer %d\n", i);
            fclose(file);
            return false;
        }

        /* Prepare layer info */
        TinyAILayerInfo layerInfo;
        memset(&layerInfo, 0, sizeof(layerInfo));

        layerInfo.layerType    = layer.type;
        layerInfo.inputDim[0]  = layer.inputWidth;
        layerInfo.inputDim[1]  = layer.inputHeight;
        layerInfo.inputDim[2]  = layer.inputChannels;
        layerInfo.outputDim[0] = layer.outputWidth;
        layerInfo.outputDim[1] = layer.outputHeight;
        layerInfo.outputDim[2] = layer.outputChannels;
        layerInfo.kernelSize   = layer.kernelSize;
        layerInfo.stride       = layer.stride;
        layerInfo.padding      = layer.padding;
        layerInfo.weightsSize  = layer.weightBytes;
        layerInfo.biasSize     = layer.biasBytes;

        /* Write layer info */
        if (fwrite(&layerInfo, sizeof(layerInfo), 1, file) != 1) {
            fprintf(stderr, "Failed to write layer info for layer %d\n", i);
            fclose(file);
            return false;
        }

        /* Write weights if present */
        if (layer.weights && layer.weightBytes > 0) {
            if (fwrite(layer.weights, 1, layer.weightBytes, file) != layer.weightBytes) {
                fprintf(stderr, "Failed to write weights for layer %d\n", i);
                fclose(file);
                return false;
            }
        }

        /* Write biases if present */
        if (layer.biases && layer.biasBytes > 0) {
            if (fwrite(layer.biases, 1, layer.biasBytes, file) != layer.biasBytes) {
                fprintf(stderr, "Failed to write biases for layer %d\n", i);
                fclose(file);
                return false;
            }
        }
    }

    fclose(file);
    return true;
}

/**
 * Generate a TinyAI model weight file from a standard format model (e.g., ONNX, TFLite)
 * @param srcFilepath Source model file path
 * @param destFilepath Destination TinyAI weight file path
 * @param modelType Target model type
 * @param quantize Whether to quantize weights to 4-bit
 * @return true on success, false on failure
 */
bool tinyaiConvertModelWeights(const char *srcFilepath, const char *destFilepath,
                               TinyAIImageModelType modelType, bool quantize)
{
    /* This is a placeholder function that would normally implement
       conversion from standard formats. In a real implementation,
       this would parse ONNX/TFLite/etc. and extract weights. */

    fprintf(stderr, "Model conversion not yet implemented for %s\n", srcFilepath);
    return false;
}
