/**
 * @file main.c
 * @brief Keyword Spotting example for TinyAI
 */

#include "../../../core/memory.h"
#include "../../../models/audio/audio_model.h"
#include "../../../models/audio/audio_utils.h"
#include "kws.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Default model path */
#define DEFAULT_MODEL_PATH "data/models/kws_model.bin"

/* Display usage information */
static void showUsage(const char *progName)
{
    printf("TinyAI Keyword Spotting Example\n");
    printf("--------------------------------\n");
    printf("Usage: %s [input_wav] [options]\n", progName);
    printf("\n");
    printf("Options:\n");
    printf("  --keyword <word>        Set keyword to detect (required if not using "
           "--list-keywords)\n");
    printf("  --threshold <value>     Set detection threshold (0.0-1.0, default 0.5)\n");
    printf("  --frame-size <ms>       Set frame size in milliseconds (default 25)\n");
    printf("  --frame-shift <ms>      Set frame shift in milliseconds (default 10)\n");
    printf("  --no-deltas             Disable delta features\n");
    printf("  --no-smoothing          Disable detection smoothing\n");
    printf("  --coefficients <num>    Set number of MFCC coefficients (default 13)\n");
    printf("  --model <path>          Set path to model file (default %s)\n", DEFAULT_MODEL_PATH);
    printf("  --mic                   Use microphone input instead of file\n");
    printf("  --list-keywords         List available keywords in the model\n");
    printf("  --visualize             Show visualization of detection results\n");
    printf("  --width <chars>         Set visualization width in characters (default 80)\n");
    printf("  --help                  Show this usage information\n");
    printf("\n");
    printf("Example:\n");
    printf("  %s input.wav --keyword \"hello\" --visualize\n", progName);
    printf("  %s --mic --keyword \"tinyai\"\n", progName);
}

/* Parse command line arguments */
static bool parseArgs(int argc, char **argv, char **inputFile, char **keyword, char **modelPath,
                      float *threshold, bool *useDeltas, bool *smoothDetections, int *frameSize,
                      int *frameShift, int *coefficients, bool *useMic, bool *listKeywords,
                      bool *visualize, int *visWidth)
{
    if (argc < 2) {
        return false;
    }

    /* Set default values */
    *inputFile        = NULL;
    *keyword          = NULL;
    *modelPath        = DEFAULT_MODEL_PATH;
    *threshold        = -1.0f; /* Use default */
    *useDeltas        = true;
    *smoothDetections = true;
    *frameSize        = 25;
    *frameShift       = 10;
    *coefficients     = 13;
    *useMic           = false;
    *listKeywords     = false;
    *visualize        = false;
    *visWidth         = 80;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            return false;
        }
        else if (strcmp(argv[i], "--keyword") == 0 && i + 1 < argc) {
            *keyword = argv[++i];
        }
        else if (strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
            *threshold = atof(argv[++i]);
            if (*threshold < 0.0f)
                *threshold = 0.0f;
            if (*threshold > 1.0f)
                *threshold = 1.0f;
        }
        else if (strcmp(argv[i], "--frame-size") == 0 && i + 1 < argc) {
            *frameSize = atoi(argv[++i]);
            if (*frameSize < 10)
                *frameSize = 10;
            if (*frameSize > 100)
                *frameSize = 100;
        }
        else if (strcmp(argv[i], "--frame-shift") == 0 && i + 1 < argc) {
            *frameShift = atoi(argv[++i]);
            if (*frameShift < 5)
                *frameShift = 5;
            if (*frameShift > *frameSize)
                *frameShift = *frameSize;
        }
        else if (strcmp(argv[i], "--coefficients") == 0 && i + 1 < argc) {
            *coefficients = atoi(argv[++i]);
            if (*coefficients < 8)
                *coefficients = 8;
            if (*coefficients > 20)
                *coefficients = 20;
        }
        else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            *modelPath = argv[++i];
        }
        else if (strcmp(argv[i], "--no-deltas") == 0) {
            *useDeltas = false;
        }
        else if (strcmp(argv[i], "--no-smoothing") == 0) {
            *smoothDetections = false;
        }
        else if (strcmp(argv[i], "--mic") == 0) {
            *useMic = true;
        }
        else if (strcmp(argv[i], "--list-keywords") == 0) {
            *listKeywords = true;
        }
        else if (strcmp(argv[i], "--visualize") == 0) {
            *visualize = true;
        }
        else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            *visWidth = atoi(argv[++i]);
            if (*visWidth < 10)
                *visWidth = 10;
            if (*visWidth > 200)
                *visWidth = 200;
        }
        else if (*inputFile == NULL && argv[i][0] != '-') {
            *inputFile = argv[i];
        }
    }

    /* Validate inputs */
    if (!*listKeywords && !*keyword) {
        fprintf(stderr, "Error: Keyword is required (use --keyword or --list-keywords)\n");
        return false;
    }

    /* When using microphone, no input file is needed */
    if (*useMic) {
        *inputFile = NULL;
        return true;
    }

    /* For listing keywords, no input file is needed */
    if (*listKeywords) {
        return true;
    }

    /* Otherwise, input file is required */
    return (*inputFile != NULL);
}

/**
 * Record audio from microphone (simulation for this example)
 * @param durationMs Duration to record in milliseconds
 * @return Audio data, or NULL on failure (caller must free)
 */
static TinyAIAudioData *recordFromMicrophone(int durationMs)
{
    /* This is a stub function for the example
     * In a real implementation, this would capture from the microphone
     */

    /* Create fake audio data for simulation */
    TinyAIAudioData *audio = (TinyAIAudioData *)malloc(sizeof(TinyAIAudioData));
    if (!audio) {
        return NULL;
    }

    /* Set properties */
    audio->format.sampleRate    = 16000;
    audio->format.channels      = 1;
    audio->format.bitsPerSample = 32; /* float */
    audio->durationMs           = durationMs;

    /* Create buffer */
    int numSamples  = (audio->format.sampleRate * durationMs) / 1000;
    audio->dataSize = numSamples * sizeof(float);
    audio->data     = malloc(audio->dataSize);
    if (!audio->data) {
        free(audio);
        return NULL;
    }

    /* Generate fake audio with occasional "keywords" */
    float *samples = (float *)audio->data;
    for (int i = 0; i < numSamples; i++) {
        /* Base background noise */
        samples[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 0.1f;

        /* Add occasional "keywords" */
        if (i % 8000 == 0 && i + 3000 < numSamples) {
            /* Simulate a keyword with higher energy */
            for (int j = 0; j < 3000 && i + j < numSamples; j++) {
                samples[i + j] += ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 0.5f;
            }
        }
    }

    printf("Recorded %d ms of audio (%d samples).\n", durationMs, numSamples);

    return audio;
}

/**
 * List available keywords in the model
 * @param state Keyword spotting state
 */
static void listAvailableKeywords(TinyAIKWSState *state)
{
    if (!state) {
        return;
    }

    /* Get available keywords */
    char *keywords[TINYAI_KWS_MAX_KEYWORDS];
    int   numKeywords;

    if (!tinyaiKWSGetAvailableKeywords(state, keywords, TINYAI_KWS_MAX_KEYWORDS, &numKeywords)) {
        printf("Error: Failed to get available keywords.\n");
        return;
    }

    /* Print available keywords */
    printf("Available keywords (%d):\n", numKeywords);
    for (int i = 0; i < numKeywords; i++) {
        printf("  %d. \"%s\"\n", i + 1, keywords[i]);
    }
}

int main(int argc, char **argv)
{
    char *inputFile;
    char *keyword;
    char *modelPath;
    float threshold;
    bool  useDeltas;
    bool  smoothDetections;
    int   frameSize;
    int   frameShift;
    int   coefficients;
    bool  useMic;
    bool  listKeywords;
    bool  visualize;
    int   visWidth;

    /* Parse arguments */
    if (!parseArgs(argc, argv, &inputFile, &keyword, &modelPath, &threshold, &useDeltas,
                   &smoothDetections, &frameSize, &frameShift, &coefficients, &useMic,
                   &listKeywords, &visualize, &visWidth)) {
        showUsage(argv[0]);
        return 1;
    }

    /* Initialize KWS configuration */
    TinyAIKWSConfig config;
    tinyaiKWSInitConfig(&config);

    /* Override defaults with command-line options */
    if (threshold >= 0.0f) {
        config.detectionThreshold = threshold;
    }
    config.frameSize           = frameSize;
    config.frameShift          = frameShift;
    config.useDeltas           = useDeltas;
    config.numMfccCoefficients = coefficients;
    config.smoothDetections    = smoothDetections;

    /* Show configuration */
    printf("TinyAI Keyword Spotting\n");
    printf("Model: %s\n", modelPath);
    printf("Configuration:\n");
    printf("  Detection threshold: %.2f\n", config.detectionThreshold);
    printf("  Frame size: %d ms\n", config.frameSize);
    printf("  Frame shift: %d ms\n", config.frameShift);
    printf("  MFCC coefficients: %d\n", config.numMfccCoefficients);
    printf("  Use deltas: %s\n", config.useDeltas ? "yes" : "no");
    printf("  Smoothing: %s\n", config.smoothDetections ? "yes" : "no");
    printf("\n");

    /* Create KWS state */
    TinyAIKWSState *kwsState = tinyaiKWSCreate(&config, modelPath);
    if (!kwsState) {
        printf("Error: Failed to create KWS state\n");
        return 1;
    }

    /* If just listing keywords, show them and exit */
    if (listKeywords) {
        listAvailableKeywords(kwsState);
        tinyaiKWSFree(kwsState);
        return 0;
    }

    /* Add keyword to detect */
    if (!tinyaiKWSAddKeyword(kwsState, keyword, threshold)) {
        printf("Error: Failed to add keyword \"%s\"\n", keyword);
        tinyaiKWSFree(kwsState);
        return 1;
    }

    /* Get audio data */
    TinyAIAudioData *audio = NULL;

    if (useMic) {
        /* Record from microphone (simulation) */
        printf("Recording from microphone (simulation)...\n");
        audio = recordFromMicrophone(5000); /* 5 seconds */
    }
    else {
        /* Load audio file */
        printf("Loading audio file: %s\n", inputFile);
        audio = (TinyAIAudioData *)malloc(sizeof(TinyAIAudioData));
        if (audio) {
            if (!tinyaiAudioLoadFile(inputFile, (TinyAIAudioFileFormat)(-1), audio)) {
                printf("Error: Failed to load audio file: %s\n", inputFile);
                free(audio);
                audio = NULL;
            }
        }
    }

    if (!audio) {
        printf("Error: Failed to get audio data\n");
        tinyaiKWSFree(kwsState);
        return 1;
    }

    /* Print audio information */
    printf("Audio information:\n");
    printf("  Sample rate: %d Hz\n", audio->format.sampleRate);
    printf("  Channels: %d\n", audio->format.channels);
    printf("  Duration: %.2f seconds\n", (float)audio->durationMs / 1000.0f);
    printf("\n");

    /* Process audio */
    printf("Detecting keyword \"%s\"...\n", keyword);
    TinyAIKWSDetection *detections;
    int                 numDetections;

    if (!tinyaiKWSProcessAudio(kwsState, audio, &detections, &numDetections)) {
        printf("Error: Failed to process audio\n");
        tinyaiAudioDataFree(audio);
        tinyaiKWSFree(kwsState);
        return 1;
    }

    /* Print detection results */
    printf("\nDetection results for keyword \"%s\":\n", keyword);
    printf("  Total detections: %d\n", numDetections);

    if (numDetections > 0) {
        printf("\nDetected keywords:\n");
        for (int i = 0; i < numDetections; i++) {
            printf("  %d. at %.2f-%.2f sec (confidence: %.2f)\n", i + 1, detections[i].startTime,
                   detections[i].endTime, detections[i].confidence);
        }
    }
    else {
        printf("  No keywords detected\n");
    }

    /* Visualize results if requested */
    if (visualize && numDetections > 0) {
        printf("\n");
        tinyaiKWSVisualizeDetections(kwsState, audio, detections, numDetections, visWidth);
    }

    /* Cleanup */
    if (detections) {
        free(detections);
    }
    tinyaiAudioDataFree(audio);
    tinyaiKWSFree(kwsState);

    return 0;
}
