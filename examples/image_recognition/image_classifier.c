/**
 * @file image_classifier.c
 * @brief Implementation of image classification in TinyAI
 */

#include "image_classifier.h"
#include "../../core/io.h"
#include "../../models/image/image_model.h"
#include "../../utils/quantize.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Default values */
#define DEFAULT_INPUT_SIZE 224
#define DEFAULT_TOP_K 5
#define DEFAULT_CONFIDENCE_THRESHOLD 0.1f

/* Maximum number of class labels to support */
#define MAX_LABELS 1000

/* Maximum length of a class label */
#define MAX_LABEL_LENGTH 128

/**
 * Internal structure for image classifier
 */
struct TinyAIImageClassifier {
    TinyAIModel *model;               /* Image model */
    char       **labels;              /* Class labels */
    int          numLabels;           /* Number of class labels */
    int          inputSize;           /* Input image size */
    int          topK;                /* Number of top predictions to return */
    float        confidenceThreshold; /* Minimum confidence threshold */
    bool         useQuantization;     /* Whether to use quantization */
    bool         useSIMD;             /* Whether to use SIMD acceleration */
    double       lastInferenceTime;   /* Last inference time in milliseconds */
};

/**
 * Prediction entry for sorting
 */
typedef struct {
    int   class_id;
    float confidence;
} PredictionEntry;

/**
 * Compare prediction entries for sorting (descending by confidence)
 */
static int comparePredictions(const void *a, const void *b)
{
    const PredictionEntry *predA = (const PredictionEntry *)a;
    const PredictionEntry *predB = (const PredictionEntry *)b;

    if (predA->confidence > predB->confidence)
        return -1;
    if (predA->confidence < predB->confidence)
        return 1;
    return 0;
}

/**
 * Load class labels from a file
 */
static char **loadLabels(const char *filePath, int *numLabels)
{
    if (!filePath || !numLabels) {
        return NULL;
    }

    /* Read labels file */
    char *fileContent = tinyaiReadTextFile(filePath);
    if (!fileContent) {
        fprintf(stderr, "Failed to read labels file: %s\n", filePath);
        return NULL;
    }

    /* Allocate array for labels */
    char **labels = (char **)malloc(MAX_LABELS * sizeof(char *));
    if (!labels) {
        free(fileContent);
        fprintf(stderr, "Failed to allocate memory for labels\n");
        return NULL;
    }

    /* Parse labels */
    int   count = 0;
    char *line  = strtok(fileContent, "\n");

    while (line && count < MAX_LABELS) {
        /* Skip empty lines */
        if (strlen(line) == 0) {
            line = strtok(NULL, "\n");
            continue;
        }

        /* Allocate and copy label */
        labels[count] = (char *)malloc(MAX_LABEL_LENGTH);
        if (!labels[count]) {
            fprintf(stderr, "Failed to allocate memory for label %d\n", count);
            break;
        }

        strncpy(labels[count], line, MAX_LABEL_LENGTH - 1);
        labels[count][MAX_LABEL_LENGTH - 1] = '\0';

        count++;
        line = strtok(NULL, "\n");
    }

    /* Set number of labels */
    *numLabels = count;

    /* Clean up */
    free(fileContent);

    return labels;
}

/**
 * Free class labels
 */
static void freeLabels(char **labels, int numLabels)
{
    if (labels) {
        for (int i = 0; i < numLabels; i++) {
            free(labels[i]);
        }
        free(labels);
    }
}

/**
 * Create a new image classifier
 */
TinyAIImageClassifier *tinyaiClassifierCreate(const TinyAIClassifierConfig *config)
{
    if (!config || !config->modelPath || !config->weightsPath || !config->labelsPath) {
        fprintf(stderr, "Invalid classifier configuration\n");
        return NULL;
    }

    /* Allocate classifier structure */
    TinyAIImageClassifier *classifier =
        (TinyAIImageClassifier *)malloc(sizeof(TinyAIImageClassifier));
    if (!classifier) {
        fprintf(stderr, "Failed to allocate classifier\n");
        return NULL;
    }

    /* Initialize with defaults */
    memset(classifier, 0, sizeof(TinyAIImageClassifier));
    classifier->inputSize = config->inputSize > 0 ? config->inputSize : DEFAULT_INPUT_SIZE;
    classifier->topK      = config->topK > 0 ? config->topK : DEFAULT_TOP_K;
    classifier->confidenceThreshold = config->confidenceThreshold >= 0.0f
                                          ? config->confidenceThreshold
                                          : DEFAULT_CONFIDENCE_THRESHOLD;
    classifier->useQuantization     = config->useQuantization;
    classifier->useSIMD             = config->useSIMD;
    classifier->lastInferenceTime   = -1.0;

    /* Load class labels */
    classifier->labels = loadLabels(config->labelsPath, &classifier->numLabels);
    if (!classifier->labels) {
        free(classifier);
        fprintf(stderr, "Failed to load class labels from %s\n", config->labelsPath);
        return NULL;
    }

    /* Load model */
    classifier->model = tinyaiLoadModel(config->modelPath, config->weightsPath, NULL);
    if (!classifier->model) {
        freeLabels(classifier->labels, classifier->numLabels);
        free(classifier);
        fprintf(stderr, "Failed to load model from %s and %s\n", config->modelPath,
                config->weightsPath);
        return NULL;
    }

    /* Apply quantization if requested */
    if (classifier->useQuantization) {
        if (tinyaiQuantizeModel(classifier->model) != 0) {
            fprintf(stderr, "Warning: Model quantization failed\n");
            /* Continue with unquantized model */
        }
    }

    return classifier;
}

/**
 * Free an image classifier
 */
void tinyaiClassifierFree(TinyAIImageClassifier *classifier)
{
    if (!classifier) {
        return;
    }

    /* Free the model */
    if (classifier->model) {
        tinyaiDestroyModel(classifier->model);
    }

    /* Free class labels */
    freeLabels(classifier->labels, classifier->numLabels);

    /* Free the classifier structure */
    free(classifier);
}

/**
 * Process model outputs to get predictions
 */
static bool processOutputs(TinyAIImageClassifier *classifier, const float *outputs, int outputSize,
                           TinyAIPrediction *predictions, int maxPredictions, int *numPredictions)
{
    if (!classifier || !outputs || !predictions || !numPredictions) {
        return false;
    }

    /* Limit outputs to number of labels */
    int validOutputs = outputSize;
    if (validOutputs > classifier->numLabels) {
        validOutputs = classifier->numLabels;
    }

    /* Create entries for sorting */
    PredictionEntry *entries = (PredictionEntry *)malloc(validOutputs * sizeof(PredictionEntry));
    if (!entries) {
        fprintf(stderr, "Failed to allocate memory for prediction entries\n");
        return false;
    }

    /* Fill entries */
    for (int i = 0; i < validOutputs; i++) {
        entries[i].class_id   = i;
        entries[i].confidence = outputs[i];
    }

    /* Sort predictions by confidence (descending) */
    qsort(entries, validOutputs, sizeof(PredictionEntry), comparePredictions);

    /* Copy top predictions */
    int count = 0;
    for (int i = 0; i < validOutputs && count < maxPredictions; i++) {
        /* Skip predictions below threshold */
        if (entries[i].confidence < classifier->confidenceThreshold) {
            continue;
        }

        /* Copy prediction */
        predictions[count].class_id   = entries[i].class_id;
        predictions[count].confidence = entries[i].confidence;

        /* Copy label (if available) */
        if (entries[i].class_id < classifier->numLabels) {
            predictions[count].label = classifier->labels[entries[i].class_id];
        }
        else {
            predictions[count].label = "Unknown";
        }

        count++;
    }

    /* Set number of predictions */
    *numPredictions = count;

    /* Clean up */
    free(entries);

    return true;
}

/**
 * Classify an image file
 */
bool tinyaiClassifyImage(TinyAIImageClassifier *classifier, const char *imagePath,
                         TinyAIPrediction *predictions, int maxPredictions, int *numPredictions)
{
    if (!classifier || !imagePath || !predictions || maxPredictions <= 0 || !numPredictions) {
        return false;
    }

    /* Load the image */
    TinyAIImage *image = tinyaiImageLoadFromFile(imagePath);
    if (!image) {
        fprintf(stderr, "Failed to load image from %s\n", imagePath);
        return false;
    }

    /* Classify the image */
    bool result =
        tinyaiClassifyImageData(classifier, image, predictions, maxPredictions, numPredictions);

    /* Clean up */
    tinyaiImageFree(image);

    return result;
}

/**
 * Classify an in-memory image
 */
bool tinyaiClassifyImageData(TinyAIImageClassifier *classifier, const TinyAIImage *image,
                             TinyAIPrediction *predictions, int maxPredictions, int *numPredictions)
{
    if (!classifier || !classifier->model || !image || !predictions || maxPredictions <= 0 ||
        !numPredictions) {
        return false;
    }

    /* Resize image if needed */
    TinyAIImage *processedImage = image;
    TinyAIImage *resizedImage   = NULL;

    if (image->width != classifier->inputSize || image->height != classifier->inputSize) {
        resizedImage = tinyaiImageResize(image, classifier->inputSize, classifier->inputSize);
        if (!resizedImage) {
            fprintf(stderr, "Failed to resize image to %dx%d\n", classifier->inputSize,
                    classifier->inputSize);
            return false;
        }
        processedImage = resizedImage;
    }

    /* Start timing */
    clock_t start = clock();

    /* Run inference */
    float *outputs    = NULL;
    int    outputSize = 0;

    bool success =
        tinyaiRunImageInference(classifier->model, processedImage, &outputs, &outputSize);

    /* End timing */
    clock_t end                   = clock();
    classifier->lastInferenceTime = 1000.0 * ((double)(end - start)) / CLOCKS_PER_SEC;

    /* Process results */
    if (success && outputs) {
        success = processOutputs(classifier, outputs, outputSize, predictions, maxPredictions,
                                 numPredictions);
        free(outputs);
    }
    else {
        success = false;
    }

    /* Clean up */
    if (resizedImage) {
        tinyaiImageFree(resizedImage);
    }

    return success;
}

/**
 * Get the last inference time in milliseconds
 */
double tinyaiClassifierGetInferenceTime(const TinyAIImageClassifier *classifier)
{
    if (!classifier) {
        return -1.0;
    }

    return classifier->lastInferenceTime;
}

/**
 * Get memory usage statistics
 */
bool tinyaiClassifierGetMemoryUsage(const TinyAIImageClassifier *classifier, size_t *modelMemory,
                                    size_t *totalMemory)
{
    if (!classifier || !modelMemory || !totalMemory) {
        return false;
    }

    /* Calculate model memory */
    size_t mModel = 0;
    if (classifier->model) {
        mModel = tinyaiGetModelSizeBytes(classifier->model);
    }

    /* Calculate label memory */
    size_t mLabels = 0;
    for (int i = 0; i < classifier->numLabels; i++) {
        if (classifier->labels[i]) {
            mLabels += strlen(classifier->labels[i]) + 1;
        }
    }
    mLabels += classifier->numLabels * sizeof(char *);

    /* Calculate total memory */
    size_t mTotal = mModel + mLabels + sizeof(TinyAIImageClassifier);

    /* Set output parameters */
    *modelMemory = mModel;
    *totalMemory = mTotal;

    return true;
}

/**
 * Format prediction as a string
 */
bool tinyaiFormatPrediction(const TinyAIPrediction *prediction, char *buffer, size_t bufferSize)
{
    if (!prediction || !buffer || bufferSize <= 0) {
        return false;
    }

    /* Format as "Label (XX.X%)" */
    int written = snprintf(buffer, bufferSize, "%s (%.1f%%)",
                           prediction->label ? prediction->label : "Unknown",
                           prediction->confidence * 100.0f);

    return written > 0 && (size_t)written < bufferSize;
}

/**
 * Enable or disable SIMD acceleration
 */
bool tinyaiClassifierEnableSIMD(TinyAIImageClassifier *classifier, bool enable)
{
    if (!classifier) {
        return false;
    }

    classifier->useSIMD = enable;

    /* Apply to model if available */
    if (classifier->model) {
        return tinyaiEnableModelSIMD(classifier->model, enable);
    }

    return true;
}
