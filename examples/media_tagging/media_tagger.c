/**
 * @file media_tagger.c
 * @brief Implementation of media tagging system in TinyAI
 */

#include "media_tagger.h"
#include "../../core/io.h"
#include "../../core/memory.h"
#include "../../models/image/image_model.h"
#include "../../models/multimodal/multimodal_model.h"
#include "../../models/text/generate.h"
#include "../../models/text/tokenizer.h"
#include "../../utils/quantize.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* File extensions for media type detection */
static const char *IMAGE_EXTENSIONS[] = {".jpg", ".jpeg", ".png",  ".gif",
                                         ".bmp", ".tiff", ".webp", NULL};
static const char *AUDIO_EXTENSIONS[] = {".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", NULL};
static const char *TEXT_EXTENSIONS[]  = {".txt", ".md",  ".html", ".xml", ".json",
                                         ".csv", ".doc", ".pdf",  NULL};

/**
 * Internal structure for media tagger
 */
struct TinyAIMediaTagger {
    /* Models */
    TinyAIImageModel      *imageModel;      /* Image classification model */
    TinyAIModel           *textModel;       /* Text generation model */
    TinyAITokenizer       *tokenizer;       /* Tokenizer */
    TinyAIMultimodalModel *multimodalModel; /* Multimodal model (optional) */

    /* Configuration */
    int               maxTags;             /* Maximum number of tags to generate */
    float             confidenceThreshold; /* Minimum confidence threshold */
    TinyAITagCategory categories;          /* Tag categories to include */
    bool              useQuantization;     /* Whether to use quantization */
    bool              useSIMD;             /* Whether to use SIMD */
    int               imageWidth;          /* Image width */
    int               imageHeight;         /* Image height */
    int               maxTextLength;       /* Maximum text length */
};

/**
 * Check if string ends with specified suffix (case insensitive)
 */
static bool endsWith(const char *str, const char *suffix)
{
    if (!str || !suffix) {
        return false;
    }

    size_t str_len    = strlen(str);
    size_t suffix_len = strlen(suffix);

    if (suffix_len > str_len) {
        return false;
    }

    const char *str_end = str + str_len - suffix_len;

    /* Case insensitive comparison */
    while (*str_end && *suffix) {
        if (tolower((unsigned char)*str_end) != tolower((unsigned char)*suffix)) {
            return false;
        }
        str_end++;
        suffix++;
    }

    return true;
}

/**
 * Create a new tag with the given text, confidence, and category
 */
static TinyAITag createTag(const char *text, float confidence, TinyAITagCategory category)
{
    TinyAITag tag;
    tag.text       = strdup(text);
    tag.confidence = confidence;
    tag.category   = category;
    return tag;
}

/**
 * Create a media tagger
 */
TinyAIMediaTagger *tinyaiMediaTaggerCreate(const TinyAIMediaTaggerConfig *config)
{
    if (!config) {
        fprintf(stderr, "Error: Invalid configuration for media tagger\n");
        return NULL;
    }

    /* Check if either image or text models are provided */
    if ((!config->imageModelPath || !config->imageWeightsPath) &&
        (!config->textModelPath || !config->textWeightsPath)) {
        fprintf(stderr, "Error: At least one of image or text models must be provided\n");
        return NULL;
    }

    /* Allocate tagger structure */
    TinyAIMediaTagger *tagger = (TinyAIMediaTagger *)malloc(sizeof(TinyAIMediaTagger));
    if (!tagger) {
        fprintf(stderr, "Error: Failed to allocate media tagger\n");
        return NULL;
    }

    /* Initialize with defaults */
    memset(tagger, 0, sizeof(TinyAIMediaTagger));
    tagger->maxTags = config->maxTags > 0 ? config->maxTags : 20;
    tagger->confidenceThreshold =
        config->confidenceThreshold >= 0.0f ? config->confidenceThreshold : 0.5f;
    tagger->categories      = config->categories ? config->categories : TINYAI_TAG_CATEGORY_ALL;
    tagger->useQuantization = config->useQuantization;
    tagger->useSIMD         = config->useSIMD;
    tagger->imageWidth      = config->imageWidth > 0 ? config->imageWidth : 224;
    tagger->imageHeight     = config->imageHeight > 0 ? config->imageHeight : 224;
    tagger->maxTextLength   = config->maxTextLength > 0 ? config->maxTextLength : 1024;

    /* Initialize tokenizer if available */
    if (config->tokenizerPath) {
        tagger->tokenizer = tinyaiTokenizerCreate(config->tokenizerPath);
        if (!tagger->tokenizer) {
            fprintf(stderr, "Warning: Failed to create tokenizer from %s\n", config->tokenizerPath);
        }
    }

    /* Initialize image model if available */
    if (config->imageModelPath && config->imageWeightsPath) {
        /* Configure image model parameters */
        TinyAIImageModelParams imageParams;
        memset(&imageParams, 0, sizeof(TinyAIImageModelParams));
        imageParams.modelType       = TINYAI_IMAGE_MODEL_MOBILENET; /* Default model type */
        imageParams.inputWidth      = tagger->imageWidth;
        imageParams.inputHeight     = tagger->imageHeight;
        imageParams.inputChannels   = 3;    /* RGB */
        imageParams.numClasses      = 1000; /* Default for standard image models */
        imageParams.weightsFile     = config->imageWeightsPath;
        imageParams.useQuantization = tagger->useQuantization;
        imageParams.useSIMD         = tagger->useSIMD;

        /* Create image model */
        tagger->imageModel = tinyaiImageModelCreate(&imageParams);
        if (!tagger->imageModel) {
            fprintf(stderr, "Warning: Failed to create image model\n");
        }
    }

    /* Initialize text model if available */
    if (config->textModelPath && config->textWeightsPath && tagger->tokenizer) {
        /* Load text model */
        tagger->textModel =
            tinyaiLoadModel(config->textModelPath, config->textWeightsPath, config->tokenizerPath);
        if (!tagger->textModel) {
            fprintf(stderr, "Warning: Failed to create text model\n");
        }

        /* Apply quantization if requested */
        if (tagger->useQuantization && tagger->textModel) {
            if (tinyaiQuantizeModel(tagger->textModel) != 0) {
                fprintf(stderr, "Warning: Failed to quantize text model\n");
            }
        }
    }

    /* Initialize multimodal model if available */
    /* Note: This is a placeholder for future implementation */

    /* Check if at least one model was loaded successfully */
    if (!tagger->imageModel && !tagger->textModel) {
        fprintf(stderr, "Error: Failed to create any models for media tagger\n");
        tinyaiMediaTaggerFree(tagger);
        return NULL;
    }

    return tagger;
}

/**
 * Free a media tagger
 */
void tinyaiMediaTaggerFree(TinyAIMediaTagger *tagger)
{
    if (!tagger) {
        return;
    }

    /* Free image model */
    if (tagger->imageModel) {
        tinyaiImageModelFree(tagger->imageModel);
    }

    /* Free text model */
    if (tagger->textModel) {
        tinyaiDestroyModel(tagger->textModel);
    }

    /* Free tokenizer */
    if (tagger->tokenizer) {
        tinyaiTokenizerFree(tagger->tokenizer);
    }

    /* Free multimodal model */
    if (tagger->multimodalModel) {
        tinyaiMultimodalModelFree(tagger->multimodalModel);
    }

    /* Free tagger structure */
    free(tagger);
}

/**
 * Detect media type from file extension
 */
TinyAIMediaType tinyaiMediaTaggerDetectType(const char *filepath)
{
    if (!filepath) {
        return TINYAI_MEDIA_TYPE_UNKNOWN;
    }

    /* Check for image extensions */
    for (int i = 0; IMAGE_EXTENSIONS[i] != NULL; i++) {
        if (endsWith(filepath, IMAGE_EXTENSIONS[i])) {
            return TINYAI_MEDIA_TYPE_IMAGE;
        }
    }

    /* Check for audio extensions */
    for (int i = 0; AUDIO_EXTENSIONS[i] != NULL; i++) {
        if (endsWith(filepath, AUDIO_EXTENSIONS[i])) {
            return TINYAI_MEDIA_TYPE_AUDIO;
        }
    }

    /* Check for text extensions */
    for (int i = 0; TEXT_EXTENSIONS[i] != NULL; i++) {
        if (endsWith(filepath, TEXT_EXTENSIONS[i])) {
            return TINYAI_MEDIA_TYPE_TEXT;
        }
    }

    return TINYAI_MEDIA_TYPE_UNKNOWN;
}

/**
 * Set categories to include in tagging
 */
bool tinyaiMediaTaggerSetCategories(TinyAIMediaTagger *tagger, TinyAITagCategory categories)
{
    if (!tagger) {
        return false;
    }

    tagger->categories = categories;
    return true;
}

/**
 * Set confidence threshold for tags
 */
bool tinyaiMediaTaggerSetThreshold(TinyAIMediaTagger *tagger, float threshold)
{
    if (!tagger || threshold < 0.0f || threshold > 1.0f) {
        return false;
    }

    tagger->confidenceThreshold = threshold;
    return true;
}

/**
 * Read text from a file
 */
static char *readTextFromFile(const char *filepath)
{
    FILE *file = fopen(filepath, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open file: %s\n", filepath);
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

/**
 * Tag a media file
 */
int tinyaiMediaTaggerTagFile(TinyAIMediaTagger *tagger, const char *filepath, TinyAITag *tags,
                             int maxTags, TinyAIMediaType *mediaType)
{
    if (!tagger || !filepath || !tags || maxTags <= 0) {
        return -1;
    }

    /* Detect media type */
    TinyAIMediaType type = tinyaiMediaTaggerDetectType(filepath);
    if (mediaType) {
        *mediaType = type;
    }

    /* Process based on media type */
    switch (type) {
    case TINYAI_MEDIA_TYPE_IMAGE:
        if (tagger->imageModel) {
            /* Load image */
            TinyAIImage *image = tinyaiImageLoadFromFile(filepath);
            if (!image) {
                fprintf(stderr, "Error: Failed to load image from %s\n", filepath);
                return -1;
            }

            /* Tag image */
            int numTags = tinyaiMediaTaggerTagImage(tagger, image, tags, maxTags);

            /* Clean up */
            tinyaiImageFree(image);
            return numTags;
        }
        break;

    case TINYAI_MEDIA_TYPE_TEXT:
        if (tagger->textModel && tagger->tokenizer) {
            /* Load text */
            char *text = readTextFromFile(filepath);
            if (!text) {
                return -1;
            }

            /* Tag text */
            int numTags = tinyaiMediaTaggerTagText(tagger, text, tags, maxTags);

            /* Clean up */
            free(text);
            return numTags;
        }
        break;

    case TINYAI_MEDIA_TYPE_AUDIO:
        /* Audio tagging not fully implemented yet */
        fprintf(stderr, "Warning: Audio tagging not fully implemented\n");
        break;

    default:
        fprintf(stderr, "Error: Unknown or unsupported media type for %s\n", filepath);
        break;
    }

    return -1;
}

/**
 * Tag an image
 */
int tinyaiMediaTaggerTagImage(TinyAIMediaTagger *tagger, const TinyAIImage *image, TinyAITag *tags,
                              int maxTags)
{
    if (!tagger || !image || !tags || maxTags <= 0 || !tagger->imageModel) {
        return -1;
    }

    /* Preprocess image if needed */
    TinyAIImage *processedImage = NULL;
    if (image->width != tagger->imageWidth || image->height != tagger->imageHeight) {
        processedImage = tinyaiImageResize(image, tagger->imageWidth, tagger->imageHeight);
        if (!processedImage) {
            fprintf(stderr, "Error: Failed to resize image\n");
            return -1;
        }
    }
    else {
        processedImage = tinyaiImageCopy(image);
        if (!processedImage) {
            fprintf(stderr, "Error: Failed to copy image\n");
            return -1;
        }
    }

    /* Allocate results array */
    int numClasses = maxTags < 20 ? maxTags : 20; /* Limit to reasonable number */
    TinyAIImageClassResult *results =
        (TinyAIImageClassResult *)malloc(numClasses * sizeof(TinyAIImageClassResult));
    if (!results) {
        fprintf(stderr, "Error: Failed to allocate classification results\n");
        tinyaiImageFree(processedImage);
        return -1;
    }

    /* Classify image */
    int numResults =
        tinyaiImageModelClassify(tagger->imageModel, processedImage, numClasses, results);

    /* Convert classification results to tags */
    int numTags = 0;
    for (int i = 0; i < numResults && numTags < maxTags; i++) {
        /* Skip tags below threshold */
        if (results[i].confidence < tagger->confidenceThreshold) {
            continue;
        }

        /* Create tag based on result */
        tags[numTags] = createTag(results[i].label ? results[i].label : "unknown",
                                  results[i].confidence, TINYAI_TAG_CATEGORY_OBJECT);

        numTags++;
    }

    /* Clean up */
    free(results);
    tinyaiImageFree(processedImage);

    return numTags;
}

/**
 * Extract keywords from text using text model
 */
static char **extractKeywords(TinyAIMediaTagger *tagger, const char *text, int *numKeywords)
{
    if (!tagger || !text || !numKeywords || !tagger->textModel || !tagger->tokenizer) {
        return NULL;
    }

    /* Initialize with empty result */
    *numKeywords = 0;

    /* Create prompt for keyword extraction */
    const char *extractPrompt = "Extract key topics and entities from this text:\n\n";
    size_t      promptLen     = strlen(extractPrompt);
    size_t      textLen       = strlen(text);
    char       *fullPrompt    = (char *)malloc(promptLen + textLen + 1);
    if (!fullPrompt) {
        fprintf(stderr, "Error: Failed to allocate memory for keyword prompt\n");
        return NULL;
    }

    /* Combine prompt and text */
    strcpy(fullPrompt, extractPrompt);
    strcat(fullPrompt, text);

    /* Tokenize prompt */
    int  promptLength = 0;
    int *promptTokens = tinyaiTokenizerEncodeText(tagger->tokenizer, fullPrompt, &promptLength);
    free(fullPrompt);

    if (!promptTokens || promptLength == 0) {
        fprintf(stderr, "Error: Failed to tokenize text for keyword extraction\n");
        return NULL;
    }

    /* Truncate if needed */
    if (promptLength > tagger->maxTextLength) {
        promptLength = tagger->maxTextLength;
    }

    /* Set up generation parameters */
    TinyAIGenerationParams genParams;
    memset(&genParams, 0, sizeof(TinyAIGenerationParams));
    genParams.maxTokens      = 100; /* Limit output tokens */
    genParams.samplingMethod = TINYAI_SAMPLING_GREEDY;
    genParams.promptTokens   = promptTokens;
    genParams.promptLength   = promptLength;

    /* Allocate buffer for output tokens */
    int *outputTokens = (int *)malloc(genParams.maxTokens * sizeof(int));
    if (!outputTokens) {
        fprintf(stderr, "Error: Failed to allocate output tokens buffer\n");
        free(promptTokens);
        return NULL;
    }

    /* Generate keywords */
    int numTokens =
        tinyaiGenerateText(tagger->textModel, &genParams, outputTokens, genParams.maxTokens);

    free(promptTokens);

    if (numTokens <= 0) {
        fprintf(stderr, "Error: Failed to generate keywords\n");
        free(outputTokens);
        return NULL;
    }

    /* Decode output tokens */
    char *keywordText = tinyaiTokenizerDecode(tagger->tokenizer, outputTokens, numTokens);
    free(outputTokens);

    if (!keywordText) {
        fprintf(stderr, "Error: Failed to decode keyword tokens\n");
        return NULL;
    }

    /* Split into individual keywords (by newline or comma) */
    char **keywords = (char **)malloc(100 * sizeof(char *)); /* Allocate for up to 100 keywords */
    if (!keywords) {
        fprintf(stderr, "Error: Failed to allocate keywords array\n");
        free(keywordText);
        return NULL;
    }

    char *saveptr;
    char *token = strtok_r(keywordText, ",\n", &saveptr);
    while (token && *numKeywords < 100) {
        /* Skip empty tokens */
        while (*token && isspace((unsigned char)*token)) {
            token++;
        }

        if (*token) {
            keywords[*numKeywords] = strdup(token);
            (*numKeywords)++;
        }

        token = strtok_r(NULL, ",\n", &saveptr);
    }

    free(keywordText);
    return keywords;
}

/**
 * Tag text content
 */
int tinyaiMediaTaggerTagText(TinyAIMediaTagger *tagger, const char *text, TinyAITag *tags,
                             int maxTags)
{
    if (!tagger || !text || !tags || maxTags <= 0 || !tagger->textModel || !tagger->tokenizer) {
        return -1;
    }

    /* Extract keywords from text */
    int    numKeywords = 0;
    char **keywords    = extractKeywords(tagger, text, &numKeywords);
    if (!keywords) {
        return -1;
    }

    /* Convert keywords to tags */
    int numTags = 0;
    for (int i = 0; i < numKeywords && numTags < maxTags; i++) {
        /* Create tag from keyword */
        tags[numTags] = createTag(keywords[i], 1.0f, /* Default confidence for keywords */
                                  TINYAI_TAG_CATEGORY_TOPIC);

        numTags++;
        free(keywords[i]);
    }

    free(keywords);
    return numTags;
}

/**
 * Generate description for a tagged media file
 */
bool tinyaiMediaTaggerGenerateDescription(TinyAIMediaTagger *tagger, const TinyAITag *tags,
                                          int numTags, char *description, int maxLength,
                                          TinyAIMediaType mediaType)
{
    if (!tagger || !tags || numTags <= 0 || !description || maxLength <= 0 || !tagger->textModel ||
        !tagger->tokenizer) {
        return false;
    }

    /* Build a prompt from tags */
    const char *mediaTypeStr;
    switch (mediaType) {
    case TINYAI_MEDIA_TYPE_IMAGE:
        mediaTypeStr = "image";
        break;
    case TINYAI_MEDIA_TYPE_AUDIO:
        mediaTypeStr = "audio file";
        break;
    case TINYAI_MEDIA_TYPE_TEXT:
        mediaTypeStr = "document";
        break;
    default:
        mediaTypeStr = "content";
        break;
    }

    /* Create the prompt */
    char prompt[2048];
    int  offset =
        snprintf(prompt, sizeof(prompt),
                 "Generate a brief description of this %s based on these tags:\n", mediaTypeStr);

    /* Add tags to prompt */
    for (int i = 0; i < numTags && offset < sizeof(prompt) - 1; i++) {
        offset += snprintf(prompt + offset, sizeof(prompt) - offset, "- %s (%.2f)\n", tags[i].text,
                           tags[i].confidence);
    }

    /* Add instruction for description */
    offset += snprintf(prompt + offset, sizeof(prompt) - offset, "\nDescription:");

    /* Tokenize prompt */
    int  promptLength = 0;
    int *promptTokens = tinyaiTokenizerEncodeText(tagger->tokenizer, prompt, &promptLength);

    if (!promptTokens || promptLength == 0) {
        fprintf(stderr, "Error: Failed to tokenize prompt for description generation\n");
        return false;
    }

    /* Set up generation parameters */
    TinyAIGenerationParams genParams;
    memset(&genParams, 0, sizeof(TinyAIGenerationParams));
    genParams.maxTokens      = 100; /* Limit description length */
    genParams.samplingMethod = TINYAI_SAMPLING_TOP_P;
    genParams.temperature    = 0.7f;
    genParams.topP           = 0.9f;
    genParams.promptTokens   = promptTokens;
    genParams.promptLength   = promptLength;

    /* Allocate buffer for output tokens */
    int *outputTokens = (int *)malloc(genParams.maxTokens * sizeof(int));
    if (!outputTokens) {
        fprintf(stderr, "Error: Failed to allocate output tokens buffer\n");
        free(promptTokens);
        return false;
    }

    /* Generate description */
    int numTokens =
        tinyaiGenerateText(tagger->textModel, &genParams, outputTokens, genParams.maxTokens);

    free(promptTokens);

    if (numTokens <= 0) {
        fprintf(stderr, "Error: Failed to generate description\n");
        free(outputTokens);
        return false;
    }

    /* Decode output tokens */
    char *descriptionText = tinyaiTokenizerDecode(tagger->tokenizer, outputTokens, numTokens);
    free(outputTokens);

    if (!descriptionText) {
        fprintf(stderr, "Error: Failed to decode description tokens\n");
        return false;
    }

    /* Copy to output buffer with truncation */
    strncpy(description, descriptionText, maxLength - 1);
    description[maxLength - 1] = '\0';

    free(descriptionText);
    return true;
}

/**
 * Save tags to a text file
 */
static bool saveTagsToText(const TinyAITag *tags, int numTags, const char *filepath)
{
    FILE *file = fopen(filepath, "w");
    if (!file) {
        fprintf(stderr, "Error: Failed to open file for writing: %s\n", filepath);
        return false;
    }

    /* Write tags to file */
    for (int i = 0; i < numTags; i++) {
        fprintf(file, "%s,%.2f,%d\n", tags[i].text, tags[i].confidence, tags[i].category);
    }

    fclose(file);
    return true;
}

/**
 * Save tags to a JSON file
 */
static bool saveTagsToJson(const TinyAITag *tags, int numTags, const char *filepath)
{
    FILE *file = fopen(filepath, "w");
    if (!file) {
        fprintf(stderr, "Error: Failed to open file for writing: %s\n", filepath);
        return false;
    }

    /* Write JSON header */
    fprintf(file, "{\n  \"tags\": [\n");

    /* Write tags as JSON objects */
    for (int i = 0; i < numTags; i++) {
        fprintf(file, "    {\n");
        fprintf(file, "      \"text\": \"%s\",\n", tags[i].text);
        fprintf(file, "      \"confidence\": %.4f,\n", tags[i].confidence);
        fprintf(file, "      \"category\": %d\n", tags[i].category);
        fprintf(file, "    }%s\n", (i < numTags - 1) ? "," : "");
    }

    /* Write JSON footer */
    fprintf(file, "  ]\n}\n");

    fclose(file);
    return true;
}

/**
 * Save tags to an XML file
 */
static bool saveTagsToXml(const TinyAITag *tags, int numTags, const char *filepath)
{
    FILE *file = fopen(filepath, "w");
    if (!file) {
        fprintf(stderr, "Error: Failed to open file for writing: %s\n", filepath);
        return false;
    }

    /* Write XML header */
    fprintf(file, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    fprintf(file, "<tags>\n");

    /* Write tags as XML elements */
    for (int i = 0; i < numTags; i++) {
        fprintf(file, "  <tag>\n");
        fprintf(file, "    <text>%s</text>\n", tags[i].text);
        fprintf(file, "    <confidence>%.4f</confidence>\n", tags[i].confidence);
        fprintf(file, "    <category>%d</category>\n", tags[i].category);
        fprintf(file, "  </tag>\n");
    }

    /* Write XML footer */
    fprintf(file, "</tags>\n");

    fclose(file);
    return true;
}

/**
 * Save tags to a file
 */
bool tinyaiMediaTaggerSaveTags(const TinyAITag *tags, int numTags, const char *filepath,
                               const char *format)
{
    if (!tags || numTags <= 0 || !filepath || !format) {
        return false;
    }

    /* Choose format based on string */
    if (strcmp(format, "txt") == 0) {
        return saveTagsToText(tags, numTags, filepath);
    }
    else if (strcmp(format, "json") == 0) {
        return saveTagsToJson(tags, numTags, filepath);
    }
    else if (strcmp(format, "xml") == 0) {
        return saveTagsToXml(tags, numTags, filepath);
    }
    else {
        fprintf(stderr, "Error: Unsupported tag file format: %s\n", format);
        return false;
    }
}

/**
 * Free tag resources
 */
void tinyaiMediaTaggerFreeTags(TinyAITag *tags, int numTags)
{
    if (!tags) {
        return;
    }

    for (int i = 0; i < numTags; i++) {
        free(tags[i].text);
    }
}

/**
 * Get memory usage statistics
 */
bool tinyaiMediaTaggerGetMemoryUsage(const TinyAIMediaTagger *tagger, size_t *weightMemory,
                                     size_t *activationMemory)
{
    if (!tagger || !weightMemory || !activationMemory) {
        return false;
    }

    /* Initialize with zeros */
    *weightMemory     = 0;
    *activationMemory = 0;

    /* Add image model memory */
    if (tagger->imageModel) {
        size_t imageWeights, imageActivations;
        if (tinyaiImageModelGetMemoryUsage(tagger->imageModel, &imageWeights, &imageActivations)) {
            *weightMemory += imageWeights;
            *activationMemory += imageActivations;
        }
    }

    /* Add text model memory (estimated) */
    if (tagger->textModel) {
        /* Simplified estimate for text model memory */
        *weightMemory += 2 * 1024 * 1024;     /* 2MB for weights */
        *activationMemory += 1 * 1024 * 1024; /* 1MB for activations */
    }

    /* Add multimodal model memory (if applicable) */
    if (tagger->multimodalModel) {
        size_t multimodalWeights, multimodalActivations;
        if (tinyaiMultimodalModelGetMemoryUsage(tagger->multimodalModel, &multimodalWeights,
                                                &multimodalActivations)) {
            *weightMemory += multimodalWeights;
            *activationMemory += multimodalActivations;
        }
    }

    return true;
}

/**
 * Enable or disable SIMD acceleration
 */
bool tinyaiMediaTaggerEnableSIMD(TinyAIMediaTagger *tagger, bool enable)
{
    if (!tagger) {
        return false;
    }

    tagger->useSIMD = enable;

    /* Apply SIMD settings to image model if available */
    if (tagger->imageModel) {
        tinyaiImageModelEnableSIMD(tagger->imageModel, enable);
    }

    /* Apply SIMD settings to multimodal model if available */
    if (tagger->multimodalModel) {
        tinyaiMultimodalModelEnableSIMD(tagger->multimodalModel, enable);
    }

    /* For text model, there's no direct SIMD control function,
       but the flag will be used in future operations */

    return true;
}
