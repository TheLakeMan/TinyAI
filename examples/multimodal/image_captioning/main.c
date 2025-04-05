/**
 * @file main.c
 * @brief Main program for image captioning example
 *
 * This example demonstrates how to use TinyAI's multimodal capabilities
 * to generate natural language captions for images.
 */

#include "image_captioner.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_CAPTION_LENGTH 2048

/* Print usage instructions */
void print_usage(const char *progname)
{
    printf("Usage: %s [options] <image1.jpg> [image2.jpg ...]\n", progname);
    printf("\nOptions:\n");
    printf("  --model <file>         Model structure file\n");
    printf("  --weights <file>       Model weights file\n");
    printf("  --tokenizer <file>     Tokenizer vocabulary file\n");
    printf("  --style <style>        Caption style (descriptive, concise, creative, technical)\n");
    printf("  --custom-prompt <text> Custom prompt for captioning\n");
    printf("  --max-tokens <n>       Maximum tokens in caption (default: 100)\n");
    printf("  --output <file>        Output file (default: captions.txt)\n");
    printf("  --quantized            Use 4-bit quantization\n");
    printf("  --simd                 Use SIMD acceleration\n");
    printf("  --compare              Compare different caption styles\n");
    printf("  --help                 Show this help message\n");
}

/* Process a single image */
bool process_image(TinyAIImageCaptioner *captioner, const char *imagePath, const char *outputPath,
                   bool append)
{
    char caption[MAX_CAPTION_LENGTH];

    printf("Processing image: %s\n", imagePath);

    /* Generate caption */
    clock_t start_time = clock();

    bool success =
        tinyaiImageCaptionerCaptionFile(captioner, imagePath, caption, MAX_CAPTION_LENGTH);

    clock_t end_time   = clock();
    double  time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    if (!success) {
        fprintf(stderr, "Error: Failed to generate caption for %s\n", imagePath);
        return false;
    }

    /* Print caption */
    printf("\nCaption: %s\n", caption);
    printf("Generation time: %.3f seconds\n", time_taken);

    /* Save caption if output file is specified */
    if (outputPath) {
        FILE *file = fopen(outputPath, append ? "a" : "w");
        if (file) {
            fprintf(file, "Image: %s\nCaption: %s\n\n", imagePath, caption);
            fclose(file);
            printf("Caption saved to %s\n", outputPath);
        }
        else {
            fprintf(stderr, "Error: Failed to open output file %s\n", outputPath);
        }
    }

    return true;
}

/* Compare different caption styles for a single image */
bool compare_styles(TinyAIImageCaptioner *captioner, const char *imagePath, const char *outputPath)
{
    TinyAICaptionStyle styles[] = {TINYAI_CAPTION_STYLE_DESCRIPTIVE, TINYAI_CAPTION_STYLE_CONCISE,
                                   TINYAI_CAPTION_STYLE_CREATIVE, TINYAI_CAPTION_STYLE_TECHNICAL};

    const char *style_names[] = {"Descriptive", "Concise", "Creative", "Technical"};

    char  caption[MAX_CAPTION_LENGTH];
    FILE *file = NULL;

    /* Open output file if specified */
    if (outputPath) {
        file = fopen(outputPath, "w");
        if (!file) {
            fprintf(stderr, "Error: Failed to open output file %s\n", outputPath);
        }
        else {
            fprintf(file, "Image: %s\n\n", imagePath);
        }
    }

    printf("\nComparing caption styles for image: %s\n", imagePath);

    /* Generate captions with different styles */
    for (int i = 0; i < 4; i++) {
        /* Set caption style */
        if (!tinyaiImageCaptionerSetStyle(captioner, styles[i], NULL)) {
            fprintf(stderr, "Error: Failed to set caption style\n");
            continue;
        }

        /* Generate caption */
        clock_t start_time = clock();
        bool    success =
            tinyaiImageCaptionerCaptionFile(captioner, imagePath, caption, MAX_CAPTION_LENGTH);
        clock_t end_time   = clock();
        double  time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

        if (!success) {
            fprintf(stderr, "Error: Failed to generate %s caption\n", style_names[i]);
            continue;
        }

        /* Print caption */
        printf("\n%s caption: %s\n", style_names[i], caption);
        printf("Generation time: %.3f seconds\n", time_taken);

        /* Save caption if output file is open */
        if (file) {
            fprintf(file, "%s caption: %s\n\n", style_names[i], caption);
        }
    }

    /* Close output file if open */
    if (file) {
        fclose(file);
        printf("\nComparison saved to %s\n", outputPath);
    }

    return true;
}

int main(int argc, char *argv[])
{
    /* Default settings */
    const char        *model_path       = NULL;
    const char        *weights_path     = NULL;
    const char        *tokenizer_path   = NULL;
    const char        *output_path      = NULL;
    const char        *custom_prompt    = NULL;
    TinyAICaptionStyle style            = TINYAI_CAPTION_STYLE_DESCRIPTIVE;
    int                max_tokens       = 100;
    bool               use_quantization = false;
    bool               use_simd         = false;
    bool               compare_mode     = false;

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
            if (strcmp(argv[i], "descriptive") == 0) {
                style = TINYAI_CAPTION_STYLE_DESCRIPTIVE;
            }
            else if (strcmp(argv[i], "concise") == 0) {
                style = TINYAI_CAPTION_STYLE_CONCISE;
            }
            else if (strcmp(argv[i], "creative") == 0) {
                style = TINYAI_CAPTION_STYLE_CREATIVE;
            }
            else if (strcmp(argv[i], "technical") == 0) {
                style = TINYAI_CAPTION_STYLE_TECHNICAL;
            }
            else {
                fprintf(stderr, "Error: Unknown caption style: %s\n", argv[i]);
                return 1;
            }
        }
        else if (strcmp(argv[i], "--custom-prompt") == 0 && i + 1 < argc) {
            custom_prompt = argv[++i];
            style         = TINYAI_CAPTION_STYLE_CUSTOM;
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
        else if (strcmp(argv[i], "--quantized") == 0) {
            use_quantization = true;
        }
        else if (strcmp(argv[i], "--simd") == 0) {
            use_simd = true;
        }
        else if (strcmp(argv[i], "--compare") == 0) {
            compare_mode = true;
        }
    }

    /* Validate required arguments */
    if (!model_path || !weights_path || !tokenizer_path) {
        fprintf(stderr, "Error: Model, weights, and tokenizer paths are required\n");
        print_usage(argv[0]);
        return 1;
    }

    /* Check if we have images to process */
    bool hasImages = false;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-' && !hasImages) {
            hasImages = true;
            break;
        }
        else if (argv[i][0] == '-' && i + 1 < argc && argv[i + 1][0] != '-') {
            /* Skip option value */
            i++;
        }
    }

    if (!hasImages) {
        fprintf(stderr, "Error: No images specified\n");
        print_usage(argv[0]);
        return 1;
    }

    /* Create captioner configuration */
    TinyAIImageCaptionerConfig config;
    memset(&config, 0, sizeof(TinyAIImageCaptionerConfig));
    config.modelPath       = model_path;
    config.weightsPath     = weights_path;
    config.tokenizerPath   = tokenizer_path;
    config.captionStyle    = style;
    config.customPrompt    = custom_prompt;
    config.maxTokens       = max_tokens;
    config.useQuantization = use_quantization;
    config.useSIMD         = use_simd;

    /* Create image captioner */
    printf("Initializing image captioner...\n");
    clock_t start_time = clock();

    TinyAIImageCaptioner *captioner = tinyaiImageCaptionerCreate(&config);
    if (!captioner) {
        fprintf(stderr, "Error: Failed to create image captioner\n");
        return 1;
    }

    clock_t init_time = clock();
    printf("Initialization completed in %.2f seconds\n",
           (double)(init_time - start_time) / CLOCKS_PER_SEC);

    /* Get memory usage statistics */
    size_t weightMemory, activationMemory;
    if (tinyaiImageCaptionerGetMemoryUsage(captioner, &weightMemory, &activationMemory)) {
        printf("Memory usage:\n");
        printf("  Weights: %.2f MB\n", weightMemory / (1024.0 * 1024.0));
        printf("  Activations: %.2f MB\n", activationMemory / (1024.0 * 1024.0));
        printf("  Total: %.2f MB\n", (weightMemory + activationMemory) / (1024.0 * 1024.0));
    }

    /* Process images */
    int images_processed = 0;

    for (int i = 1; i < argc; i++) {
        /* Skip options and their values */
        if (argv[i][0] == '-') {
            if (strcmp(argv[i], "--quantized") != 0 && strcmp(argv[i], "--simd") != 0 &&
                strcmp(argv[i], "--compare") != 0) {
                i++; /* Skip option value */
            }
            continue;
        }

        /* Process image */
        if (compare_mode) {
            compare_styles(captioner, argv[i], output_path);
        }
        else {
            process_image(captioner, argv[i], output_path, images_processed > 0);
        }

        images_processed++;
    }

    /* Summary */
    clock_t end_time = clock();
    printf("\nProcessed %d images in %.2f seconds\n", images_processed,
           (double)(end_time - start_time) / CLOCKS_PER_SEC);

    /* Clean up */
    tinyaiImageCaptionerFree(captioner);

    return 0;
}
