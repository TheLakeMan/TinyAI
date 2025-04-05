/**
 * @file main.c
 * @brief Voice Activity Detection example for TinyAI
 */

#include "../../../core/memory.h"
#include "../../../models/audio/audio_model.h"
#include "../../../models/audio/audio_utils.h"
#include "vad.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Display usage information */
static void showUsage(const char *progName)
{
    printf("TinyAI Voice Activity Detection Example\n");
    printf("----------------------------------------\n");
    printf("Usage: %s <input_wav> [options]\n", progName);
    printf("\n");
    printf("Options:\n");
    printf("  --sensitivity <value>   Set sensitivity (0.0-1.0, default 0.5)\n");
    printf("  --frame-size <ms>       Set frame size in milliseconds (default 20)\n");
    printf("  --frame-shift <ms>      Set frame shift in milliseconds (default 10)\n");
    printf("  --no-zcr                Disable zero-crossing rate analysis\n");
    printf("  --no-smoothing          Disable smoothing\n");
    printf("  --visualize             Show visualization of detection results\n");
    printf("  --width <chars>         Set visualization width in characters (default 80)\n");
    printf("  --help                  Show this usage information\n");
    printf("\n");
    printf("Example:\n");
    printf("  %s input.wav --sensitivity 0.7 --visualize\n", progName);
}

/* Parse command line arguments */
static bool parseArgs(int argc, char **argv, char **inputFile, TinyAIVADConfig *config,
                      bool *visualize, int *visWidth)
{
    if (argc < 2) {
        return false;
    }

    /* Set default values */
    *inputFile = NULL;
    *visualize = false;
    *visWidth  = 80;
    tinyaiVADInitConfig(config);

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            return false;
        }
        else if (strcmp(argv[i], "--sensitivity") == 0 && i + 1 < argc) {
            config->sensitivity = atof(argv[++i]);
            if (config->sensitivity < 0.0f)
                config->sensitivity = 0.0f;
            if (config->sensitivity > 1.0f)
                config->sensitivity = 1.0f;
        }
        else if (strcmp(argv[i], "--frame-size") == 0 && i + 1 < argc) {
            config->frameSize = atoi(argv[++i]);
            if (config->frameSize < 10)
                config->frameSize = 10;
            if (config->frameSize > 100)
                config->frameSize = 100;
        }
        else if (strcmp(argv[i], "--frame-shift") == 0 && i + 1 < argc) {
            config->frameShift = atoi(argv[++i]);
            if (config->frameShift < 5)
                config->frameShift = 5;
            if (config->frameShift > config->frameSize)
                config->frameShift = config->frameSize;
        }
        else if (strcmp(argv[i], "--no-zcr") == 0) {
            config->useZcr = false;
        }
        else if (strcmp(argv[i], "--no-smoothing") == 0) {
            config->smoothing = false;
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

    return (*inputFile != NULL);
}

int main(int argc, char **argv)
{
    char           *inputFile;
    TinyAIVADConfig config;
    bool            visualize;
    int             visWidth;

    /* Parse arguments */
    if (!parseArgs(argc, argv, &inputFile, &config, &visualize, &visWidth)) {
        showUsage(argv[0]);
        return 1;
    }

    /* Show configuration */
    printf("TinyAI Voice Activity Detection\n");
    printf("Input file: %s\n", inputFile);
    printf("Configuration:\n");
    printf("  Sensitivity: %.2f\n", config.sensitivity);
    printf("  Frame size: %d ms\n", config.frameSize);
    printf("  Frame shift: %d ms\n", config.frameShift);
    printf("  Use ZCR: %s\n", config.useZcr ? "yes" : "no");
    printf("  Smoothing: %s\n", config.smoothing ? "yes" : "no");
    printf("\n");

    /* Load audio file */
    TinyAIAudioData audio;
    if (!tinyaiAudioLoadFile(inputFile, (TinyAIAudioFileFormat)(-1), &audio)) {
        printf("Error: Failed to load audio file: %s\n", inputFile);
        return 1;
    }

    /* Print audio information */
    printf("Audio information:\n");
    printf("  Sample rate: %d Hz\n", audio.format.sampleRate);
    printf("  Channels: %d\n", audio.format.channels);
    printf("  Duration: %.2f seconds\n", (float)audio.durationMs / 1000.0f);
    printf("\n");

    /* Create VAD state */
    TinyAIVADState *vadState = tinyaiVADCreate(&config, audio.format.sampleRate);
    if (!vadState) {
        printf("Error: Failed to create VAD state\n");
        tinyaiAudioDataFree(&audio);
        return 1;
    }

    /* Process audio */
    bool *activity;
    int   activitySize;
    if (!tinyaiVADProcessAudio(vadState, &audio, &activity, &activitySize)) {
        printf("Error: Failed to process audio\n");
        tinyaiVADFree(vadState);
        tinyaiAudioDataFree(&audio);
        return 1;
    }

    /* Print results */
    int totalActive = 0;
    for (int i = 0; i < activitySize; i++) {
        if (activity[i]) {
            totalActive++;
        }
    }
    float activePercent = 100.0f * (float)totalActive / (float)activitySize;

    printf("Voice activity detection results:\n");
    printf("  Total frames: %d\n", activitySize);
    printf("  Voice frames: %d (%.1f%%)\n", totalActive, activePercent);
    printf("  Silence frames: %d (%.1f%%)\n", activitySize - totalActive, 100.0f - activePercent);
    printf("\n");

    /* Visualize results if requested */
    if (visualize) {
        tinyaiVADVisualize(activity, activitySize, visWidth);
    }

    /* Calculate voice segments */
    int segmentCount        = 0;
    int currentSegmentStart = -1;

    for (int i = 0; i < activitySize; i++) {
        if (activity[i] && currentSegmentStart < 0) {
            /* Start of new segment */
            currentSegmentStart = i;
        }
        else if (!activity[i] && currentSegmentStart >= 0) {
            /* End of segment */
            float startTime = (float)(currentSegmentStart * config.frameShift) / 1000.0f;
            float endTime   = (float)(i * config.frameShift) / 1000.0f;
            float duration  = endTime - startTime;

            printf("Voice segment %d: %.2f - %.2f sec (%.2f sec)\n", ++segmentCount, startTime,
                   endTime, duration);

            currentSegmentStart = -1;
        }
    }

    /* Handle case where last segment extends to end of audio */
    if (currentSegmentStart >= 0) {
        float startTime = (float)(currentSegmentStart * config.frameShift) / 1000.0f;
        float endTime   = (float)(activitySize * config.frameShift) / 1000.0f;
        float duration  = endTime - startTime;

        printf("Voice segment %d: %.2f - %.2f sec (%.2f sec)\n", ++segmentCount, startTime, endTime,
               duration);
    }

    printf("\nDetected %d voice segments\n", segmentCount);

    /* Memory cleanup */
    free(activity);
    tinyaiVADFree(vadState);
    tinyaiAudioDataFree(&audio);

    return 0;
}
