/**
 * @file main.c
 * @brief Speech Recognition example for TinyAI
 */

#include "../../../core/memory.h"
#include "../../../models/audio/audio_model.h"
#include "../../../models/audio/audio_utils.h"
#include "asr.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Define M_PI if not defined */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Default paths */
#define DEFAULT_ACOUSTIC_MODEL_PATH "data/models/acoustic_model.bin"
#define DEFAULT_LANGUAGE_MODEL_PATH "data/models/language_model.bin"
#define DEFAULT_OUTPUT_PATH "transcript.txt"

/* Display usage information */
static void showUsage(const char *progName)
{
    printf("TinyAI Speech Recognition Example\n");
    printf("----------------------------------\n");
    printf("Usage: %s [input_wav] [options]\n", progName);
    printf("\n");
    printf("Options:\n");
    printf("  --output <file>      Set output transcript file path (default: %s)\n",
           DEFAULT_OUTPUT_PATH);
    printf("  --model-am <path>    Set acoustic model path (default: %s)\n",
           DEFAULT_ACOUSTIC_MODEL_PATH);
    printf("  --model-lm <path>    Set language model path (default: %s)\n",
           DEFAULT_LANGUAGE_MODEL_PATH);
    printf("  --mode <mode>        Set recognition mode: fast, balanced, accurate (default: "
           "balanced)\n");
    printf("  --lm-weight <value>  Set language model weight (0.0-1.0, default: 0.5)\n");
    printf("  --beam-width <num>   Set beam width for decoding (default: 8)\n");
    printf("  --mic                Use microphone input (simulation) instead of file\n");
    printf("  --no-punctuation     Disable punctuation inference\n");
    printf("  --verbose            Enable verbose output\n");
    printf("  --timestamps         Include word timestamps in output\n");
    printf("  --list-models        Display information about available models\n");
    printf("  --help               Show this usage information\n");
    printf("\n");
    printf("Example:\n");
    printf("  %s input.wav --output transcript.txt --mode accurate\n", progName);
    printf("  %s --mic --verbose\n", progName);
}

/* Parse command-line arguments */
static bool parseArgs(int argc, char **argv, char **inputFile, char **outputFile,
                      char **acousticModelPath, char **languageModelPath, TinyAIASRMode *mode,
                      float *lmWeight, int *beamWidth, bool *useMic, bool *enablePunctuation,
                      bool *verbose, bool *includeTimestamps, bool *listModels)
{
    if (argc < 2) {
        return false;
    }

    /* Set default values */
    *inputFile         = NULL;
    *outputFile        = DEFAULT_OUTPUT_PATH;
    *acousticModelPath = DEFAULT_ACOUSTIC_MODEL_PATH;
    *languageModelPath = DEFAULT_LANGUAGE_MODEL_PATH;
    *mode              = TINYAI_ASR_MODE_BALANCED;
    *lmWeight          = 0.5f;
    *beamWidth         = 8;
    *useMic            = false;
    *enablePunctuation = true;
    *verbose           = false;
    *includeTimestamps = false;
    *listModels        = false;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            return false;
        }
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            *outputFile = argv[++i];
        }
        else if (strcmp(argv[i], "--model-am") == 0 && i + 1 < argc) {
            *acousticModelPath = argv[++i];
        }
        else if (strcmp(argv[i], "--model-lm") == 0 && i + 1 < argc) {
            *languageModelPath = argv[++i];
        }
        else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "fast") == 0) {
                *mode = TINYAI_ASR_MODE_FAST;
            }
            else if (strcmp(argv[i], "accurate") == 0) {
                *mode = TINYAI_ASR_MODE_ACCURATE;
            }
            else {
                *mode = TINYAI_ASR_MODE_BALANCED;
            }
        }
        else if (strcmp(argv[i], "--lm-weight") == 0 && i + 1 < argc) {
            *lmWeight = atof(argv[++i]);
            if (*lmWeight < 0.0f)
                *lmWeight = 0.0f;
            if (*lmWeight > 1.0f)
                *lmWeight = 1.0f;
        }
        else if (strcmp(argv[i], "--beam-width") == 0 && i + 1 < argc) {
            *beamWidth = atoi(argv[++i]);
            if (*beamWidth < 1)
                *beamWidth = 1;
            if (*beamWidth > TINYAI_ASR_MAX_BEAM_WIDTH)
                *beamWidth = TINYAI_ASR_MAX_BEAM_WIDTH;
        }
        else if (strcmp(argv[i], "--mic") == 0) {
            *useMic = true;
        }
        else if (strcmp(argv[i], "--no-punctuation") == 0) {
            *enablePunctuation = false;
        }
        else if (strcmp(argv[i], "--verbose") == 0) {
            *verbose = true;
        }
        else if (strcmp(argv[i], "--timestamps") == 0) {
            *includeTimestamps = true;
        }
        else if (strcmp(argv[i], "--list-models") == 0) {
            *listModels = true;
        }
        else if (*inputFile == NULL && argv[i][0] != '-') {
            *inputFile = argv[i];
        }
    }

    /* Special cases */
    if (*listModels) {
        /* Only show models info, don't need input file */
        return true;
    }

    /* When using microphone, no input file is needed */
    if (*useMic) {
        *inputFile = NULL;
        return true;
    }

    /* Otherwise, input file is required */
    return (*inputFile != NULL);
}

/**
 * Generate simulated audio data for testing
 * @param durationMs Duration in milliseconds
 * @param sampleRate Sample rate in Hz
 * @return Simulated audio data (caller must free)
 */
static TinyAIAudioData *generateSimulatedAudio(int durationMs, int sampleRate)
{
    /* Create audio data */
    TinyAIAudioData *audio = (TinyAIAudioData *)malloc(sizeof(TinyAIAudioData));
    if (!audio) {
        return NULL;
    }

    /* Set format */
    audio->format.sampleRate    = sampleRate;
    audio->format.channels      = 1;
    audio->format.bitsPerSample = 32; /* float */
    audio->durationMs           = durationMs;

    /* Calculate buffer size */
    int numSamples  = (sampleRate * durationMs) / 1000;
    audio->dataSize = numSamples * sizeof(float);

    /* Allocate buffer */
    audio->data = malloc(audio->dataSize);
    if (!audio->data) {
        free(audio);
        return NULL;
    }

    /* Generate simulated speech samples */
    float *samples = (float *)audio->data;

    /* Create patterns that look somewhat like speech */
    const int wordsPerSec     = 2;
    const int samplesPerWord  = sampleRate / wordsPerSec;
    const int silenceDuration = samplesPerWord / 3;

    for (int i = 0; i < numSamples; i++) {
        /* Determine if this is in a "word" or "silence" */
        int posInPattern = i % samplesPerWord;

        if (posInPattern < (samplesPerWord - silenceDuration)) {
            /* "Word" - create a formant-like pattern with noise */
            float baseFreq = 100.0f + (i % samplesPerWord) * 0.5f; /* Rising pitch */
            float formant1 = sinf(2.0f * M_PI * baseFreq * i / sampleRate);
            float formant2 = 0.5f * sinf(2.0f * M_PI * 2.0f * baseFreq * i / sampleRate);
            float noise    = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 0.1f;

            /* Envelope to smooth transitions */
            float envelope = 1.0f;
            if (posInPattern < 1000) {
                envelope = (float)posInPattern / 1000.0f; /* Attack */
            }
            else if (posInPattern > (samplesPerWord - silenceDuration - 1000)) {
                envelope =
                    (float)(samplesPerWord - silenceDuration - posInPattern) / 1000.0f; /* Decay */
            }

            samples[i] = (formant1 + formant2 + noise) * envelope * 0.5f;
        }
        else {
            /* "Silence" - just low-level noise */
            samples[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 0.01f;
        }
    }

    return audio;
}

/**
 * Record audio from microphone (simulation)
 * @param durationMs Duration to record in milliseconds
 * @return Simulated microphone data (caller must free)
 */
static TinyAIAudioData *recordFromMicrophone(int durationMs)
{
    printf("Recording from microphone (simulation)...\n");

    /* Generate simulated audio */
    TinyAIAudioData *audio = generateSimulatedAudio(durationMs, 16000); /* 16kHz audio */

    if (audio) {
        printf("Recorded %d ms of audio (%d samples at %d Hz).\n", durationMs,
               (int)(audio->dataSize / sizeof(float)), audio->format.sampleRate);
    }

    return audio;
}

int main(int argc, char **argv)
{
    char         *inputFile;
    char         *outputFile;
    char         *acousticModelPath;
    char         *languageModelPath;
    TinyAIASRMode mode;
    float         lmWeight;
    int           beamWidth;
    bool          useMic;
    bool          enablePunctuation;
    bool          verbose;
    bool          includeTimestamps;
    bool          listModels;

    /* Seed random number generator for simulations */
    srand(time(NULL));

    /* Parse arguments */
    if (!parseArgs(argc, argv, &inputFile, &outputFile, &acousticModelPath, &languageModelPath,
                   &mode, &lmWeight, &beamWidth, &useMic, &enablePunctuation, &verbose,
                   &includeTimestamps, &listModels)) {
        showUsage(argv[0]);
        return 1;
    }

    /* Just list models and exit if requested */
    if (listModels) {
        tinyaiASRPrintModelInfo();
        return 0;
    }

    /* Configure recognition */
    TinyAIASRConfig config;
    tinyaiASRInitConfig(&config);

    /* Update configuration based on command-line options */
    config.mode                 = mode;
    config.lmType               = TINYAI_ASR_LM_BIGRAM; /* Use bigram LM by default */
    config.lmWeight             = lmWeight;
    config.beamWidth            = beamWidth;
    config.enablePunctuation    = enablePunctuation;
    config.enableVerboseOutput  = verbose;
    config.enableWordTimestamps = includeTimestamps;

    /* Display configuration */
    if (verbose) {
        printf("TinyAI Speech Recognition\n");
        printf("Acoustic model: %s\n", acousticModelPath);
        printf("Language model: %s\n", languageModelPath);
        printf("Configuration:\n");
        printf("  Mode: %s\n", mode == TINYAI_ASR_MODE_FAST       ? "Fast"
                               : mode == TINYAI_ASR_MODE_ACCURATE ? "Accurate"
                                                                  : "Balanced");
        printf("  Language model weight: %.2f\n", config.lmWeight);
        printf("  Beam width: %d\n", config.beamWidth);
        printf("  Punctuation: %s\n", config.enablePunctuation ? "Enabled" : "Disabled");
        printf("  Word timestamps: %s\n", config.enableWordTimestamps ? "Enabled" : "Disabled");
        printf("\n");
    }

    /* Create ASR state */
    TinyAIASRState *asrState = tinyaiASRCreate(&config, acousticModelPath, languageModelPath);
    if (!asrState) {
        printf("Error: Failed to create ASR state.\n");
        return 1;
    }

    /* Get audio data */
    TinyAIAudioData *audio = NULL;

    if (useMic) {
        /* Record from microphone (simulation) */
        audio = recordFromMicrophone(10000); /* 10 seconds */
    }
    else {
        /* Load audio file */
        if (verbose) {
            printf("Loading audio file: %s\n", inputFile);
        }

        audio = (TinyAIAudioData *)malloc(sizeof(TinyAIAudioData));
        if (audio) {
            if (!tinyaiAudioLoadFile(inputFile, (TinyAIAudioFileFormat)(-1), audio)) {
                printf("Error: Failed to load audio file: %s\n", inputFile);
                free(audio);
                audio = NULL;
            }
        }

        /* If file loading failed, generate simulated audio for testing */
        if (!audio) {
            if (verbose) {
                printf("Generating simulated audio for testing.\n");
            }
            audio = generateSimulatedAudio(5000, 16000); /* 5 seconds, 16kHz */
        }
    }

    if (!audio) {
        printf("Error: Failed to get audio data.\n");
        tinyaiASRFree(asrState);
        return 1;
    }

    /* Display audio info */
    if (verbose) {
        printf("Audio information:\n");
        printf("  Sample rate: %d Hz\n", audio->format.sampleRate);
        printf("  Channels: %d\n", audio->format.channels);
        printf("  Duration: %.2f seconds\n", (float)audio->durationMs / 1000.0f);
        printf("\n");
    }

    /* Perform recognition */
    printf("Recognizing speech (this might take a few seconds)...\n");

    TinyAIASRResult result;
    if (!tinyaiASRProcessAudio(asrState, audio, &result)) {
        printf("Error: Speech recognition failed.\n");
        tinyaiAudioDataFree(audio);
        tinyaiASRFree(asrState);
        return 1;
    }

    /* Display results */
    printf("\nRecognition result:\n");
    printf("%s\n", result.transcript);

    /* Save to file if requested */
    if (outputFile) {
        printf("Saving transcript to: %s\n", outputFile);
        if (!tinyaiASRSaveResult(&result, outputFile, includeTimestamps)) {
            printf("Error: Failed to save transcript.\n");
        }
    }

    /* Get word error rate against dummy reference (for testing) */
    const char *dummyReference = "this is a dummy reference for testing purposes";
    float       wer            = tinyaiASRCalculateWER(&result, dummyReference);
    if (verbose) {
        printf("\nSimulated Word Error Rate: %.1f%%\n", wer * 100.0f);
    }

    /* Cleanup */
    tinyaiAudioDataFree(audio);
    tinyaiASRFree(asrState);

    return 0;
}
