/**
 * @file main.c
 * @brief Main program for media tagging example
 *
 * This example demonstrates how to use TinyAI's media tagging capabilities
 * to automatically tag and describe images, audio, and text files.
 */

#include "media_tagger.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#endif

#define MAX_TAGS 50
#define MAX_DESCRIPTION_LENGTH 2048
#define MAX_PATH_LENGTH 256

/* Print usage instructions */
void print_usage(const char *progname)
{
    printf("Usage: %s [options] file1 [file2 file3 ...]\n", progname);
    printf("\nOptions:\n");
    printf("  --image-model <file>      Image model structure file\n");
    printf("  --image-weights <file>    Image model weights file\n");
    printf("  --text-model <file>       Text model structure file\n");
    printf("  --text-weights <file>     Text model weights file\n");
    printf("  --tokenizer <file>        Tokenizer vocabulary file\n");
    printf("  --output <dir>            Output directory for tag files (default: current dir)\n");
    printf("  --format <format>         Output format: txt, json, xml (default: json)\n");
    printf("  --threshold <value>       Confidence threshold (0.0-1.0, default: 0.5)\n");
    printf("  --max-tags <n>            Maximum number of tags (default: 20)\n");
    printf("  --generate-description    Generate descriptions for each media file\n");
    printf("  --quantized               Use 4-bit quantization for models\n");
    printf("  --simd                    Use SIMD acceleration\n");
    printf("  --batch <directory>       Process all supported files in directory\n");
    printf("  --help                    Show this help message\n");
}

/* Process a single file */
bool process_file(TinyAIMediaTagger *tagger, const char *filepath, const char *outputDir,
                  const char *format, bool generateDescription)
{
    TinyAITag       tags[MAX_TAGS];
    char            outputPath[MAX_PATH_LENGTH];
    char            description[MAX_DESCRIPTION_LENGTH];
    TinyAIMediaType mediaType;

    printf("Processing: %s\n", filepath);

    /* Tag the file */
    int numTags = tinyaiMediaTaggerTagFile(tagger, filepath, tags, MAX_TAGS, &mediaType);
    if (numTags <= 0) {
        fprintf(stderr, "Error: Failed to tag file %s\n", filepath);
        return false;
    }

    printf("Identified as: %s\n", mediaType == TINYAI_MEDIA_TYPE_IMAGE   ? "Image"
                                  : mediaType == TINYAI_MEDIA_TYPE_AUDIO ? "Audio"
                                  : mediaType == TINYAI_MEDIA_TYPE_TEXT  ? "Text"
                                                                         : "Unknown");

    printf("Tags (%d):\n", numTags);
    for (int i = 0; i < numTags; i++) {
        printf("  %s (%.2f)\n", tags[i].text, tags[i].confidence);
    }

    /* Generate description if requested */
    if (generateDescription) {
        if (tinyaiMediaTaggerGenerateDescription(tagger, tags, numTags, description,
                                                 MAX_DESCRIPTION_LENGTH, mediaType)) {
            printf("Description:\n%s\n", description);
        }
        else {
            fprintf(stderr, "Warning: Could not generate description\n");
        }
    }

    /* Determine output file path */
    const char *basename = strrchr(filepath, '/');
    if (!basename) {
        basename = strrchr(filepath, '\\');
    }
    basename = basename ? basename + 1 : filepath;

    /* Replace extension with format */
    snprintf(outputPath, sizeof(outputPath), "%s/%s.tags.%s", outputDir ? outputDir : ".", basename,
             format);

    /* Save tags to file */
    if (tinyaiMediaTaggerSaveTags(tags, numTags, outputPath, format)) {
        printf("Tags saved to: %s\n", outputPath);
    }
    else {
        fprintf(stderr, "Error: Failed to save tags to %s\n", outputPath);
    }

    /* Free tags */
    tinyaiMediaTaggerFreeTags(tags, numTags);

    return true;
}

/* Process all supported files in a directory */
int process_directory(TinyAIMediaTagger *tagger, const char *dirPath, const char *outputDir,
                      const char *format, bool generateDescription)
{
    char filepath[MAX_PATH_LENGTH];
    int  filesProcessed = 0;

    printf("Processing directory: %s\n", dirPath);

#ifdef _WIN32
    /* Windows implementation using FindFirstFile/FindNextFile */
    WIN32_FIND_DATA findData;
    char            searchPath[MAX_PATH_LENGTH];
    snprintf(searchPath, sizeof(searchPath), "%s\\*", dirPath);

    HANDLE hFind = FindFirstFile(searchPath, &findData);
    if (hFind == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Error: Could not open directory %s\n", dirPath);
        return 0;
    }

    do {
        /* Skip . and .. */
        if (strcmp(findData.cFileName, ".") == 0 || strcmp(findData.cFileName, "..") == 0) {
            continue;
        }

        /* Skip directories */
        if (findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            continue;
        }

        /* Construct full path */
        snprintf(filepath, sizeof(filepath), "%s\\%s", dirPath, findData.cFileName);

        /* Check if file has a supported extension */
        TinyAIMediaType type = tinyaiMediaTaggerDetectType(filepath);
        if (type != TINYAI_MEDIA_TYPE_UNKNOWN) {
            if (process_file(tagger, filepath, outputDir, format, generateDescription)) {
                filesProcessed++;
            }
        }
    } while (FindNextFile(hFind, &findData));

    FindClose(hFind);
#else
    /* POSIX implementation using dirent */
    DIR           *dir;
    struct dirent *entry;

    dir = opendir(dirPath);
    if (!dir) {
        fprintf(stderr, "Error: Could not open directory %s\n", dirPath);
        return 0;
    }

    while ((entry = readdir(dir)) != NULL) {
        /* Skip . and .. */
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        /* Construct full path */
        snprintf(filepath, sizeof(filepath), "%s/%s", dirPath, entry->d_name);

        /* Check if file has a supported extension */
        TinyAIMediaType type = tinyaiMediaTaggerDetectType(filepath);
        if (type != TINYAI_MEDIA_TYPE_UNKNOWN) {
            if (process_file(tagger, filepath, outputDir, format, generateDescription)) {
                filesProcessed++;
            }
        }
    }

    closedir(dir);
#endif

    printf("Processed %d files from directory %s\n", filesProcessed, dirPath);
    return filesProcessed;
}

int main(int argc, char *argv[])
{
    /* Default settings */
    const char *image_model_path     = NULL;
    const char *image_weights_path   = NULL;
    const char *text_model_path      = NULL;
    const char *text_weights_path    = NULL;
    const char *tokenizer_path       = NULL;
    const char *output_dir           = NULL;
    const char *format               = "json";
    const char *batch_dir            = NULL;
    float       threshold            = 0.5f;
    int         max_tags             = 20;
    bool        generate_description = false;
    bool        use_quantization     = false;
    bool        use_simd             = false;
    int         files_processed      = 0;

    /* Parse command line arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "--image-model") == 0 && i + 1 < argc) {
            image_model_path = argv[++i];
        }
        else if (strcmp(argv[i], "--image-weights") == 0 && i + 1 < argc) {
            image_weights_path = argv[++i];
        }
        else if (strcmp(argv[i], "--text-model") == 0 && i + 1 < argc) {
            text_model_path = argv[++i];
        }
        else if (strcmp(argv[i], "--text-weights") == 0 && i + 1 < argc) {
            text_weights_path = argv[++i];
        }
        else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            tokenizer_path = argv[++i];
        }
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_dir = argv[++i];
        }
        else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
            format = argv[++i];
        }
        else if (strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
            threshold = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--max-tags") == 0 && i + 1 < argc) {
            max_tags = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--generate-description") == 0) {
            generate_description = true;
        }
        else if (strcmp(argv[i], "--quantized") == 0) {
            use_quantization = true;
        }
        else if (strcmp(argv[i], "--simd") == 0) {
            use_simd = true;
        }
        else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            batch_dir = argv[++i];
        }
    }

    /* Validate required arguments */
    if ((!image_model_path || !image_weights_path) && (!text_model_path || !text_weights_path)) {
        fprintf(stderr, "Error: At least one model type (image or text) must be fully specified\n");
        print_usage(argv[0]);
        return 1;
    }

    /* For text model, tokenizer is required */
    if ((text_model_path || text_weights_path) && !tokenizer_path) {
        fprintf(stderr, "Error: Tokenizer is required when using text model\n");
        return 1;
    }

    /* Check if we have files to process */
    if (!batch_dir && argc <= 1) {
        fprintf(stderr, "Error: No files specified\n");
        print_usage(argv[0]);
        return 1;
    }

    /* Validate format */
    if (strcmp(format, "txt") != 0 && strcmp(format, "json") != 0 && strcmp(format, "xml") != 0) {
        fprintf(stderr, "Error: Unsupported format: %s (must be txt, json, or xml)\n", format);
        return 1;
    }

    /* Create tagger configuration */
    TinyAIMediaTaggerConfig config;
    memset(&config, 0, sizeof(TinyAIMediaTaggerConfig));
    config.imageModelPath      = image_model_path;
    config.imageWeightsPath    = image_weights_path;
    config.textModelPath       = text_model_path;
    config.textWeightsPath     = text_weights_path;
    config.tokenizerPath       = tokenizer_path;
    config.maxTags             = max_tags;
    config.confidenceThreshold = threshold;
    config.categories          = TINYAI_TAG_CATEGORY_ALL;
    config.useQuantization     = use_quantization;
    config.useSIMD             = use_simd;

    /* Create media tagger */
    printf("Initializing media tagger...\n");
    clock_t start_time = clock();

    TinyAIMediaTagger *tagger = tinyaiMediaTaggerCreate(&config);
    if (!tagger) {
        fprintf(stderr, "Error: Failed to create media tagger\n");
        return 1;
    }

    clock_t init_time = clock();
    printf("Initialization completed in %.2f seconds\n",
           (double)(init_time - start_time) / CLOCKS_PER_SEC);

    /* Get memory usage statistics */
    size_t weightMemory, activationMemory;
    if (tinyaiMediaTaggerGetMemoryUsage(tagger, &weightMemory, &activationMemory)) {
        printf("Memory usage:\n");
        printf("  Weights: %.2f KB\n", weightMemory / 1024.0);
        printf("  Activations: %.2f KB\n", activationMemory / 1024.0);
        printf("  Total: %.2f KB\n", (weightMemory + activationMemory) / 1024.0);
    }

    /* Process files */
    if (batch_dir) {
        /* Process all files in directory */
        files_processed =
            process_directory(tagger, batch_dir, output_dir, format, generate_description);
    }
    else {
        /* Process individual files */
        for (int i = 1; i < argc; i++) {
            /* Skip options and their values */
            if (argv[i][0] == '-') {
                if (strcmp(argv[i], "--generate-description") != 0 &&
                    strcmp(argv[i], "--quantized") != 0 && strcmp(argv[i], "--simd") != 0) {
                    i++; /* Skip option value */
                }
                continue;
            }

            /* Process file */
            if (process_file(tagger, argv[i], output_dir, format, generate_description)) {
                files_processed++;
            }
        }
    }

    /* Summary */
    clock_t end_time = clock();
    printf("\nProcessed %d files in %.2f seconds\n", files_processed,
           (double)(end_time - start_time) / CLOCKS_PER_SEC);

    /* Clean up */
    tinyaiMediaTaggerFree(tagger);

    return files_processed > 0 ? 0 : 1;
}
