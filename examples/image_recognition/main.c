/**
 * @file main.c
 * @brief Main program for the TinyAI image recognition example
 */

#include "../../core/io.h"
#include "image_classifier.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#define COLOR_RESET "\033[0m"
#define COLOR_BOLD "\033[1m"
#define COLOR_RED "\033[31m"
#define COLOR_GREEN "\033[32m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_BLUE "\033[34m"
#define COLOR_CYAN "\033[36m"
#else
#include <unistd.h>
#define COLOR_RESET "\033[0m"
#define COLOR_BOLD "\033[1m"
#define COLOR_RED "\033[31m"
#define COLOR_GREEN "\033[32m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_BLUE "\033[34m"
#define COLOR_CYAN "\033[36m"
#endif

/* Maximum path length */
#define MAX_PATH_LENGTH 512

/* Maximum number of predictions to display */
#define MAX_PREDICTIONS 100

/* Buffer size for formatting predictions */
#define PRED_BUFFER_SIZE 256

/* Read a file with a list of image paths */
static char **readBatchFile(const char *filePath, int *numImages)
{
    if (!filePath || !numImages) {
        return NULL;
    }

    /* Read file content */
    char *fileContent = tinyaiReadTextFile(filePath);
    if (!fileContent) {
        fprintf(stderr, "Failed to read batch file: %s\n", filePath);
        return NULL;
    }

    /* Count lines */
    int lineCount = 0;
    for (char *p = fileContent; *p; p++) {
        if (*p == '\n') {
            lineCount++;
        }
    }

    /* Add one more if file doesn't end with newline */
    if (fileContent[0] != '\0' && fileContent[strlen(fileContent) - 1] != '\n') {
        lineCount++;
    }

    /* Allocate array for image paths */
    char **imagePaths = (char **)malloc(lineCount * sizeof(char *));
    if (!imagePaths) {
        free(fileContent);
        fprintf(stderr, "Failed to allocate memory for image paths\n");
        return NULL;
    }

    /* Parse image paths */
    int   count = 0;
    char *line  = strtok(fileContent, "\n");

    while (line && count < lineCount) {
        /* Skip empty lines */
        if (strlen(line) == 0) {
            line = strtok(NULL, "\n");
            continue;
        }

        /* Allocate and copy image path */
        imagePaths[count] = (char *)malloc(MAX_PATH_LENGTH);
        if (!imagePaths[count]) {
            /* Clean up on error */
            for (int i = 0; i < count; i++) {
                free(imagePaths[i]);
            }
            free(imagePaths);
            free(fileContent);
            fprintf(stderr, "Failed to allocate memory for image path %d\n", count);
            return NULL;
        }

        strncpy(imagePaths[count], line, MAX_PATH_LENGTH - 1);
        imagePaths[count][MAX_PATH_LENGTH - 1] = '\0';

        count++;
        line = strtok(NULL, "\n");
    }

    /* Set number of images */
    *numImages = count;

    /* Clean up */
    free(fileContent);

    return imagePaths;
}

/* Free batch image paths */
static void freeBatchImagePaths(char **imagePaths, int numImages)
{
    if (imagePaths) {
        for (int i = 0; i < numImages; i++) {
            free(imagePaths[i]);
        }
        free(imagePaths);
    }
}

/* Process a single image */
static bool processImage(TinyAIImageClassifier *classifier, const char *imagePath, FILE *outputFile)
{
    if (!classifier || !imagePath) {
        return false;
    }

    /* Start timing */
    clock_t start = clock();

    /* Run classification */
    TinyAIPrediction predictions[MAX_PREDICTIONS];
    int              numPredictions = 0;

    bool success =
        tinyaiClassifyImage(classifier, imagePath, predictions, MAX_PREDICTIONS, &numPredictions);

    /* End timing */
    clock_t end      = clock();
    double  loadTime = ((double)(end - start)) / CLOCKS_PER_SEC;

    if (!success) {
        fprintf(stderr, "Failed to classify image: %s\n", imagePath);
        return false;
    }

    /* Print processing information */
    if (!outputFile) {
        printf("\n%sProcessing image: %s%s\n", COLOR_BOLD, imagePath, COLOR_RESET);
        printf("Inference time: %.2f ms\n", tinyaiClassifierGetInferenceTime(classifier));
        printf("Total processing time: %.2f seconds\n", loadTime);
    }
    else {
        fprintf(outputFile, "Image: %s\n", imagePath);
        fprintf(outputFile, "Inference time: %.2f ms\n",
                tinyaiClassifierGetInferenceTime(classifier));
    }

    /* Print predictions */
    if (!outputFile) {
        printf("\n%sTop %d Predictions:%s\n", COLOR_BOLD, numPredictions, COLOR_RESET);
    }
    else {
        fprintf(outputFile, "\nTop %d Predictions:\n", numPredictions);
    }

    char buffer[PRED_BUFFER_SIZE];

    for (int i = 0; i < numPredictions; i++) {
        if (tinyaiFormatPrediction(&predictions[i], buffer, PRED_BUFFER_SIZE)) {
            if (!outputFile) {
                printf("%d. %s\n", i + 1, buffer);
            }
            else {
                fprintf(outputFile, "%d. %s\n", i + 1, buffer);
            }
        }
    }

    if (!outputFile) {
        printf("\n");
    }
    else {
        fprintf(outputFile, "\n");
    }

    return true;
}

/* Print usage instructions */
static void printUsage(const char *progname)
{
    printf("TinyAI Image Recognition Example\n\n");
    printf("Usage: %s [options] <image_file>\n\n", progname);
    printf("Options:\n");
    printf("  --model <file>       Path to model structure file\n");
    printf("  --weights <file>     Path to model weights file\n");
    printf("  --labels <file>      Path to class labels file\n");
    printf("  --top-k <n>          Return top-K predictions (default: 5)\n");
    printf("  --threshold <value>  Minimum confidence threshold (default: 0.1)\n");
    printf("  --input-size <size>  Input image size in pixels (default: 224)\n");
    printf("  --batch <file>       Process multiple images listed in a file\n");
    printf("  --output <file>      Save results to file instead of stdout\n");
    printf("  --quantized          Use 4-bit quantization (default: enabled)\n");
    printf("  --no-quantize        Disable quantization\n");
    printf("  --simd               Enable SIMD acceleration\n");
    printf("  --no-simd            Disable SIMD acceleration\n");
    printf("  --help               Show this help message\n");
}

/* Print memory usage statistics */
static void printMemoryUsage(TinyAIImageClassifier *classifier)
{
    size_t modelMem, totalMem;

    if (tinyaiClassifierGetMemoryUsage(classifier, &modelMem, &totalMem)) {
        printf("\n%sMemory Usage:%s\n", COLOR_BOLD, COLOR_RESET);
        printf("  Model: %6.2f MB\n", modelMem / (1024.0 * 1024.0));
        printf("  Total: %6.2f MB\n", totalMem / (1024.0 * 1024.0));
    }
}

/* Main function */
int main(int argc, char *argv[])
{
    /* Default settings */
    const char *model_path           = NULL;
    const char *weights_path         = NULL;
    const char *labels_path          = NULL;
    const char *batch_file           = NULL;
    const char *output_file          = NULL;
    const char *image_path           = NULL;
    int         top_k                = 5;
    float       confidence_threshold = 0.1f;
    int         input_size           = 224;
    bool        use_quantization     = true;
    bool        use_simd             = true;

    /* Parse command line arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        }
        else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            weights_path = argv[++i];
        }
        else if (strcmp(argv[i], "--labels") == 0 && i + 1 < argc) {
            labels_path = argv[++i];
        }
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            top_k = atoi(argv[++i]);
            if (top_k <= 0) {
                fprintf(stderr, "Invalid top-k value: %s\n", argv[i]);
                return 1;
            }
        }
        else if (strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
            confidence_threshold = (float)atof(argv[++i]);
            if (confidence_threshold < 0.0f || confidence_threshold > 1.0f) {
                fprintf(stderr, "Invalid threshold (must be between 0.0 and 1.0): %s\n", argv[i]);
                return 1;
            }
        }
        else if (strcmp(argv[i], "--input-size") == 0 && i + 1 < argc) {
            input_size = atoi(argv[++i]);
            if (input_size <= 0) {
                fprintf(stderr, "Invalid input size: %s\n", argv[i]);
                return 1;
            }
        }
        else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            batch_file = argv[++i];
        }
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        }
        else if (strcmp(argv[i], "--quantized") == 0) {
            use_quantization = true;
        }
        else if (strcmp(argv[i], "--no-quantize") == 0) {
            use_quantization = false;
        }
        else if (strcmp(argv[i], "--simd") == 0) {
            use_simd = true;
        }
        else if (strcmp(argv[i], "--no-simd") == 0) {
            use_simd = false;
        }
        else if (argv[i][0] != '-') {
            /* Image path */
            image_path = argv[i];
        }
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    /* Check for required arguments */
    if (!model_path || !weights_path || !labels_path) {
        fprintf(stderr, "Error: Model, weights, and labels paths are required\n");
        printUsage(argv[0]);
        return 1;
    }

    /* Check for batch file or image path */
    if (!batch_file && !image_path) {
        fprintf(stderr, "Error: Either an image file or batch file must be specified\n");
        printUsage(argv[0]);
        return 1;
    }

    /* Open output file if specified */
    FILE *output = NULL;
    if (output_file) {
        output = fopen(output_file, "w");
        if (!output) {
            fprintf(stderr, "Error: Failed to open output file: %s\n", output_file);
            return 1;
        }
    }

    /* Initialize classifier */
    printf("Loading model from %s and %s...\n", model_path, weights_path);

    TinyAIClassifierConfig config;
    memset(&config, 0, sizeof(TinyAIClassifierConfig));
    config.modelPath           = model_path;
    config.weightsPath         = weights_path;
    config.labelsPath          = labels_path;
    config.inputSize           = input_size;
    config.topK                = top_k;
    config.confidenceThreshold = confidence_threshold;
    config.useQuantization     = use_quantization;
    config.useSIMD             = use_simd;

    /* Create classifier */
    clock_t                start      = clock();
    TinyAIImageClassifier *classifier = tinyaiClassifierCreate(&config);

    if (!classifier) {
        fprintf(stderr, "Failed to create classifier\n");
        if (output) {
            fclose(output);
        }
        return 1;
    }

    clock_t end      = clock();
    double  initTime = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Model loaded successfully in %.2f seconds\n", initTime);

    /* Print memory usage */
    printMemoryUsage(classifier);

    /* Process images */
    if (batch_file) {
        /* Process multiple images from batch file */
        int    numImages  = 0;
        char **imagePaths = readBatchFile(batch_file, &numImages);

        if (!imagePaths || numImages == 0) {
            fprintf(stderr, "Failed to read batch file or no valid images found\n");
            tinyaiClassifierFree(classifier);
            if (output) {
                fclose(output);
            }
            return 1;
        }

        printf("Processing %d images from %s...\n", numImages, batch_file);

        int successCount = 0;
        for (int i = 0; i < numImages; i++) {
            if (processImage(classifier, imagePaths[i], output)) {
                successCount++;
            }
        }

        printf("\nSuccessfully classified %d out of %d images\n", successCount, numImages);

        /* Clean up */
        freeBatchImagePaths(imagePaths, numImages);
    }
    else {
        /* Process single image */
        processImage(classifier, image_path, output);
    }

    /* Close output file */
    if (output) {
        fclose(output);
        printf("Results saved to %s\n", output_file);
    }

    /* Clean up */
    tinyaiClassifierFree(classifier);

    return 0;
}
