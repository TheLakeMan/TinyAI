/**
 * @file main.c
 * @brief Main program for visual question answering example
 *
 * This example demonstrates how to use TinyAI's multimodal capabilities
 * to answer questions about images using natural language.
 */

#include "visual_qa.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_ANSWER_LENGTH 2048
#define MAX_QUESTION_LENGTH 512
#define MAX_PATH_LENGTH 256

/* Print usage instructions */
void print_usage(const char *progname)
{
    printf("Usage: %s [options] <image.jpg> \"Your question about the image\"\n", progname);
    printf("\nOptions:\n");
    printf("  --model <file>         Model structure file\n");
    printf("  --weights <file>       Model weights file\n");
    printf("  --tokenizer <file>     Tokenizer vocabulary file\n");
    printf("  --style <style>        Answer style (concise, detailed, factual, casual)\n");
    printf("  --custom-template <text> Custom prompt template for answers\n");
    printf("  --max-tokens <n>       Maximum tokens in answer (default: 100)\n");
    printf("  --output <file>        Output file (default: output to terminal only)\n");
    printf("  --batch <file>         Batch file with image paths and questions\n");
    printf("  --quantized            Use 4-bit quantization\n");
    printf("  --simd                 Use SIMD acceleration\n");
    printf("  --help                 Show this help message\n");
}

/* Process a single image and question */
bool process_query(TinyAIVisualQA *vqa, const char *imagePath, const char *question,
                   const char *outputPath, bool append)
{
    char answer[MAX_ANSWER_LENGTH];

    printf("Processing question for image: %s\n", imagePath);
    printf("Question: %s\n", question);

    /* Generate answer */
    clock_t start_time = clock();

    bool success =
        tinyaiVisualQAAnswerQuestion(vqa, imagePath, question, answer, MAX_ANSWER_LENGTH);

    clock_t end_time   = clock();
    double  time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    if (!success) {
        fprintf(stderr, "Error: Failed to generate answer for image %s\n", imagePath);
        return false;
    }

    /* Print answer */
    printf("\nAnswer: %s\n", answer);
    printf("Generation time: %.3f seconds\n", time_taken);

    /* Save answer if output file is specified */
    if (outputPath) {
        FILE *file = fopen(outputPath, append ? "a" : "w");
        if (file) {
            fprintf(file, "Image: %s\nQuestion: %s\nAnswer: %s\n\n", imagePath, question, answer);
            fclose(file);
            printf("Answer saved to %s\n", outputPath);
        }
        else {
            fprintf(stderr, "Error: Failed to open output file %s\n", outputPath);
        }
    }

    return true;
}

/* Process a batch file containing image paths and questions */
bool process_batch_file(TinyAIVisualQA *vqa, const char *batchFilePath, const char *outputPath)
{
    FILE *batchFile = fopen(batchFilePath, "r");
    if (!batchFile) {
        fprintf(stderr, "Error: Failed to open batch file %s\n", batchFilePath);
        return false;
    }

    char line[MAX_PATH_LENGTH + MAX_QUESTION_LENGTH];
    int  queries_processed = 0;
    char imagePath[MAX_PATH_LENGTH];
    char question[MAX_QUESTION_LENGTH];

    printf("Processing batch file: %s\n", batchFilePath);

    while (fgets(line, sizeof(line), batchFile)) {
        /* Skip empty lines and comments */
        if (line[0] == '\n' || line[0] == '#' || line[0] == '\0') {
            continue;
        }

        /* Parse line format: image_path|question */
        char *separator = strchr(line, '|');
        if (!separator) {
            fprintf(stderr, "Warning: Invalid line format, skipping: %s", line);
            continue;
        }

        /* Extract image path and question */
        *separator = '\0';
        strncpy(imagePath, line, MAX_PATH_LENGTH - 1);
        imagePath[MAX_PATH_LENGTH - 1] = '\0';

        /* Trim whitespace */
        char *end = imagePath + strlen(imagePath) - 1;
        while (end > imagePath && (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r')) {
            *end = '\0';
            end--;
        }

        strncpy(question, separator + 1, MAX_QUESTION_LENGTH - 1);
        question[MAX_QUESTION_LENGTH - 1] = '\0';

        /* Trim whitespace */
        end = question + strlen(question) - 1;
        while (end > question && (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r')) {
            *end = '\0';
            end--;
        }

        /* Process the query */
        if (process_query(vqa, imagePath, question, outputPath, queries_processed > 0)) {
            queries_processed++;
        }
    }

    fclose(batchFile);
    printf("\nProcessed %d queries from batch file\n", queries_processed);

    return queries_processed > 0;
}

int main(int argc, char *argv[])
{
    /* Default settings */
    const char       *model_path       = NULL;
    const char       *weights_path     = NULL;
    const char       *tokenizer_path   = NULL;
    const char       *output_path      = NULL;
    const char       *batch_file       = NULL;
    const char       *custom_template  = NULL;
    TinyAIAnswerStyle style            = TINYAI_ANSWER_STYLE_CONCISE;
    int               max_tokens       = 100;
    bool              use_quantization = false;
    bool              use_simd         = false;

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
        else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            tokenizer_path = argv[++i];
        }
        else if (strcmp(argv[i], "--style") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "concise") == 0) {
                style = TINYAI_ANSWER_STYLE_CONCISE;
            }
            else if (strcmp(argv[i], "detailed") == 0) {
                style = TINYAI_ANSWER_STYLE_DETAILED;
            }
            else if (strcmp(argv[i], "factual") == 0) {
                style = TINYAI_ANSWER_STYLE_FACTUAL;
            }
            else if (strcmp(argv[i], "casual") == 0) {
                style = TINYAI_ANSWER_STYLE_CASUAL;
            }
            else {
                fprintf(stderr, "Error: Unknown answer style: %s\n", argv[i]);
                return 1;
            }
        }
        else if (strcmp(argv[i], "--custom-template") == 0 && i + 1 < argc) {
            custom_template = argv[++i];
            style           = TINYAI_ANSWER_STYLE_CUSTOM;
        }
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
            if (max_tokens <= 0) {
                fprintf(stderr, "Error: Invalid max tokens value\n");
                return 1;
            }
        }
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        }
        else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            batch_file = argv[++i];
        }
        else if (strcmp(argv[i], "--quantized") == 0) {
            use_quantization = true;
        }
        else if (strcmp(argv[i], "--simd") == 0) {
            use_simd = true;
        }
    }

    /* Validate required arguments */
    if (!model_path || !weights_path || !tokenizer_path) {
        fprintf(stderr, "Error: Model, weights, and tokenizer paths are required\n");
        print_usage(argv[0]);
        return 1;
    }

    /* Check if we have an image and question (for single query mode) */
    const char *image_path = NULL;
    const char *question   = NULL;

    if (!batch_file) {
        int arg_count = 0;
        for (int i = 1; i < argc; i++) {
            if (argv[i][0] != '-') {
                if (arg_count == 0) {
                    image_path = argv[i];
                }
                else if (arg_count == 1) {
                    question = argv[i];
                }
                arg_count++;
            }
            else if (argv[i][0] == '-' && i + 1 < argc && argv[i + 1][0] != '-') {
                /* Skip option value */
                i++;
            }
        }

        if (!image_path || !question) {
            fprintf(stderr, "Error: Both image path and question are required\n");
            print_usage(argv[0]);
            return 1;
        }
    }

    /* Create visual QA configuration */
    TinyAIVisualQAConfig config;
    memset(&config, 0, sizeof(TinyAIVisualQAConfig));
    config.modelPath       = model_path;
    config.weightsPath     = weights_path;
    config.tokenizerPath   = tokenizer_path;
    config.answerStyle     = style;
    config.customTemplate  = custom_template;
    config.maxTokens       = max_tokens;
    config.useQuantization = use_quantization;
    config.useSIMD         = use_simd;

    /* Create visual QA system */
    printf("Initializing visual QA system...\n");
    clock_t start_time = clock();

    TinyAIVisualQA *vqa = tinyaiVisualQACreate(&config);
    if (!vqa) {
        fprintf(stderr, "Error: Failed to create visual QA system\n");
        return 1;
    }

    clock_t init_time = clock();
    printf("Initialization completed in %.2f seconds\n",
           (double)(init_time - start_time) / CLOCKS_PER_SEC);

    /* Get memory usage statistics */
    size_t weightMemory, activationMemory;
    if (tinyaiVisualQAGetMemoryUsage(vqa, &weightMemory, &activationMemory)) {
        printf("Memory usage:\n");
        printf("  Weights: %.2f MB\n", weightMemory / (1024.0 * 1024.0));
        printf("  Activations: %.2f MB\n", activationMemory / (1024.0 * 1024.0));
        printf("  Total: %.2f MB\n", (weightMemory + activationMemory) / (1024.0 * 1024.0));
    }

    /* Process queries */
    bool success = false;

    if (batch_file) {
        /* Process batch file */
        success = process_batch_file(vqa, batch_file, output_path);
    }
    else {
        /* Process single query */
        success = process_query(vqa, image_path, question, output_path, false);
    }

    /* Summary */
    clock_t end_time = clock();
    printf("\nTotal processing time: %.2f seconds\n",
           (double)(end_time - start_time) / CLOCKS_PER_SEC);

    /* Clean up */
    tinyaiVisualQAFree(vqa);

    return success ? 0 : 1;
}
