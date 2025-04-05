/**
 * @file main.c
 * @brief Main program for document processor example
 *
 * This example demonstrates how to use TinyAI's document processing capabilities
 * for classification, summarization, and information extraction.
 */

#include "document_processor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_OUTPUT_LENGTH 4096

/**
 * Read text from a file
 */
static char *readTextFromFile(const char *filePath)
{
    FILE *file = fopen(filePath, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open file: %s\n", filePath);
        return NULL;
    }

    /* Get file size */
    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    /* Allocate buffer */
    char *buffer = (char *)malloc(fileSize + 1);
    if (!buffer) {
        fprintf(stderr, "Error: Failed to allocate buffer for file content\n");
        fclose(file);
        return NULL;
    }

    /* Read file content */
    size_t readSize  = fread(buffer, 1, fileSize, file);
    buffer[readSize] = '\0';

    fclose(file);
    return buffer;
}

void print_usage(const char *progname)
{
    printf("Usage: %s [options] <mode> <document_file>\n", progname);
    printf("\nModes:\n");
    printf("  classify   Classify the document into categories\n");
    printf("  summarize  Generate a summary of the document\n");
    printf("  extract    Extract specific information from the document\n");
    printf("\nOptions:\n");
    printf("  --model <file>       Model structure file path\n");
    printf("  --weights <file>     Model weights file path\n");
    printf("  --vocab <file>       Tokenizer vocabulary file path\n");
    printf("  --classes <file>     Classes file for classification (one class per line)\n");
    printf("  --prompt <text>      Extraction prompt for extract mode\n");
    printf("  --max-input <n>      Maximum input length in tokens (default: 1024)\n");
    printf("  --max-output <n>     Maximum output length in tokens (default: 256)\n");
    printf("  --simd               Enable SIMD acceleration\n");
    printf("  --quantized          Use 4-bit quantization\n");
    printf("  --help               Show this help message\n");
}

char **load_classes(const char *filepath, int *num_classes)
{
    FILE *file = fopen(filepath, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open classes file %s\n", filepath);
        return NULL;
    }

    /* Count the number of lines in the file */
    int  count = 0;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        count++;
    }

    /* Allocate array for class labels */
    char **classes = (char **)malloc(count * sizeof(char *));
    if (!classes) {
        fprintf(stderr, "Error: Memory allocation failed for classes array\n");
        fclose(file);
        return NULL;
    }

    /* Reset file position and read class labels */
    rewind(file);
    int i = 0;
    while (fgets(buffer, sizeof(buffer), file) != NULL && i < count) {
        /* Remove newline character */
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0';
        }

        /* Allocate and copy class label */
        classes[i] = strdup(buffer);
        if (!classes[i]) {
            fprintf(stderr, "Error: Memory allocation failed for class label\n");
            for (int j = 0; j < i; j++) {
                free(classes[j]);
            }
            free(classes);
            fclose(file);
            return NULL;
        }
        i++;
    }

    fclose(file);
    *num_classes = count;
    return classes;
}

void free_classes(char **classes, int num_classes)
{
    if (classes) {
        for (int i = 0; i < num_classes; i++) {
            free(classes[i]);
        }
        free(classes);
    }
}

int main(int argc, char *argv[])
{
    /* Default settings */
    char                       *model_path        = NULL;
    char                       *weights_path      = NULL;
    char                       *vocab_path        = NULL;
    char                       *classes_path      = NULL;
    char                       *document_path     = NULL;
    char                       *extraction_prompt = NULL;
    TinyAIDocumentProcessorMode mode              = TINYAI_DOC_MODE_CLASSIFY;
    int                         max_input_length  = 1024;
    int                         max_output_length = 256;
    bool                        use_simd          = false;
    bool                        use_quantization  = false;
    bool                        mode_set          = false;

    /* Parse command line arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        }
        else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            weights_path = argv[++i];
        }
        else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
            vocab_path = argv[++i];
        }
        else if (strcmp(argv[i], "--classes") == 0 && i + 1 < argc) {
            classes_path = argv[++i];
        }
        else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            extraction_prompt = argv[++i];
        }
        else if (strcmp(argv[i], "--max-input") == 0 && i + 1 < argc) {
            max_input_length = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--max-output") == 0 && i + 1 < argc) {
            max_output_length = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--simd") == 0) {
            use_simd = true;
        }
        else if (strcmp(argv[i], "--quantized") == 0) {
            use_quantization = true;
        }
        else if (strcmp(argv[i], "classify") == 0) {
            mode     = TINYAI_DOC_MODE_CLASSIFY;
            mode_set = true;
        }
        else if (strcmp(argv[i], "summarize") == 0) {
            mode     = TINYAI_DOC_MODE_SUMMARIZE;
            mode_set = true;
        }
        else if (strcmp(argv[i], "extract") == 0) {
            mode     = TINYAI_DOC_MODE_EXTRACT_INFO;
            mode_set = true;
        }
        else if (argv[i][0] != '-' && !document_path) {
            document_path = argv[i];
        }
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    /* Validate required arguments */
    if (!mode_set || !document_path) {
        fprintf(stderr, "Error: Mode and document file are required\n");
        print_usage(argv[0]);
        return 1;
    }

    if (!model_path || !weights_path || !vocab_path) {
        fprintf(stderr, "Error: Model, weights, and vocabulary paths are required\n");
        print_usage(argv[0]);
        return 1;
    }

    if (mode == TINYAI_DOC_MODE_CLASSIFY && !classes_path) {
        fprintf(stderr, "Error: Classes file is required for classification mode\n");
        print_usage(argv[0]);
        return 1;
    }

    if (mode == TINYAI_DOC_MODE_EXTRACT_INFO && !extraction_prompt) {
        fprintf(stderr, "Error: Extraction prompt is required for extraction mode\n");
        print_usage(argv[0]);
        return 1;
    }

    /* Load classes for classification if needed */
    char **class_labels = NULL;
    int    num_classes  = 0;

    if (mode == TINYAI_DOC_MODE_CLASSIFY && classes_path) {
        class_labels = load_classes(classes_path, &num_classes);
        if (!class_labels) {
            return 1;
        }
    }

    /* Initialize document processor configuration */
    TinyAIDocumentProcessorConfig config;
    memset(&config, 0, sizeof(config));
    config.mode            = mode;
    config.modelPath       = model_path;
    config.weightsPath     = weights_path;
    config.tokenizerPath   = vocab_path;
    config.useQuantization = use_quantization;
    config.useSIMD         = use_simd;
    config.maxInputLength  = max_input_length;
    config.maxOutputLength = max_output_length;
    config.numClasses      = num_classes;
    config.classLabels     = (const char **)class_labels;

    /* Create document processor */
    printf("Initializing document processor...\n");
    TinyAIDocumentProcessor *processor = tinyaiDocumentProcessorCreate(&config);
    if (!processor) {
        fprintf(stderr, "Error: Failed to create document processor\n");
        free_classes(class_labels, num_classes);
        return 1;
    }

    /* Get memory usage statistics */
    size_t weight_memory, activation_memory;
    if (tinyaiDocumentProcessorGetMemoryUsage(processor, &weight_memory, &activation_memory)) {
        printf("Memory usage:\n");
        printf("  Weights: %.2f MB\n", weight_memory / (1024.0 * 1024.0));
        printf("  Activations: %.2f MB\n", activation_memory / (1024.0 * 1024.0));
        printf("  Total: %.2f MB\n", (weight_memory + activation_memory) / (1024.0 * 1024.0));
    }

    /* Allocate output buffer */
    char *output_buffer = (char *)malloc(MAX_OUTPUT_LENGTH * sizeof(char));
    if (!output_buffer) {
        fprintf(stderr, "Error: Failed to allocate output buffer\n");
        tinyaiDocumentProcessorFree(processor);
        free_classes(class_labels, num_classes);
        return 1;
    }

    /* Measure processing time */
    clock_t start_time = clock();

    /* Process the document based on the mode */
    bool success = false;
    if (mode == TINYAI_DOC_MODE_EXTRACT_INFO) {
        printf("Extracting information from %s using prompt: \"%s\"\n", document_path,
               extraction_prompt);
        success =
            tinyaiDocumentProcessFile(processor, document_path, output_buffer, MAX_OUTPUT_LENGTH);
        if (!success) {
            /* If the main process file fails for extraction (as expected), try direct extraction */
            char *document_text = readTextFromFile(document_path);
            if (document_text) {
                success = tinyaiDocumentExtractInfo(processor, document_text, extraction_prompt,
                                                    output_buffer, MAX_OUTPUT_LENGTH);
                free(document_text);
            }
        }
    }
    else {
        printf("Processing document: %s\n", document_path);
        success =
            tinyaiDocumentProcessFile(processor, document_path, output_buffer, MAX_OUTPUT_LENGTH);
    }

    /* Calculate processing time */
    clock_t end_time        = clock();
    double  processing_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    /* Print results */
    if (success) {
        printf("\n===== RESULT =====\n");
        printf("%s\n", output_buffer);
        printf("=================\n\n");
        printf("Processing completed in %.2f seconds\n", processing_time);
    }
    else {
        fprintf(stderr, "Error: Document processing failed\n");
    }

    /* Clean up */
    free(output_buffer);
    tinyaiDocumentProcessorFree(processor);
    free_classes(class_labels, num_classes);

    return success ? 0 : 1;
}
