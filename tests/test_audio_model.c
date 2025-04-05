/**
 * @file test_audio_model.c
 * @brief Test program for audio model functionality in TinyAI
 */

#include "../models/audio/audio_features.h"
#include "../models/audio/audio_model.h"
#include "../models/audio/audio_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/**
 * Create a sample audio file for testing
 * @param filename Output file path
 * @param durationMs Duration in milliseconds
 * @param frequency Tone frequency in Hz
 * @return true on success, false on failure
 */
static bool createSampleAudio(const char *filename, int durationMs, float frequency)
{
    int    sampleRate = 16000; /* 16kHz sample rate */
    int    numSamples = (int)(durationMs * sampleRate / 1000);
    float *samples    = (float *)malloc(numSamples * sizeof(float));
    if (!samples) {
        fprintf(stderr, "Failed to allocate sample buffer\n");
        return false;
    }

    /* Generate a simple sine wave */
    for (int i = 0; i < numSamples; i++) {
        float t    = (float)i / sampleRate;
        samples[i] = 0.5f * sinf(2.0f * 3.14159f * frequency * t);
    }

    /* Create audio data */
    TinyAIAudioData   audio;
    TinyAIAudioFormat format;
    format.sampleRate    = sampleRate;
    format.channels      = 1;  /* Mono */
    format.bitsPerSample = 16; /* 16-bit */

    if (!tinyaiAudioCreateFromSamples(samples, numSamples, &format, &audio)) {
        fprintf(stderr, "Failed to create audio data\n");
        free(samples);
        return false;
    }

    /* Save audio to file */
    bool result = tinyaiAudioSaveFile(filename, TINYAI_AUDIO_FORMAT_WAV, &audio);

    /* Clean up */
    tinyaiAudioDataFree(&audio);
    free(samples);

    return result;
}

/**
 * Test audio feature extraction
 * @return true on success, false on failure
 */
static bool testAudioFeatures()
{
    printf("Testing audio feature extraction...\n");

    /* Create a sample audio file */
    const char *sampleFile = "test_audio.wav";
    if (!createSampleAudio(sampleFile, 2000, 440.0f)) {
        fprintf(stderr, "Failed to create sample audio\n");
        return false;
    }

    /* Load audio file */
    TinyAIAudioData audio;
    if (!tinyaiAudioLoadFile(sampleFile, TINYAI_AUDIO_FORMAT_WAV, &audio)) {
        fprintf(stderr, "Failed to load audio file\n");
        return false;
    }

    /* Set up feature extraction configuration */
    TinyAIAudioFeaturesConfig config;
    memset(&config, 0, sizeof(config));
    config.type              = TINYAI_AUDIO_FEATURES_MFCC;
    config.frameLength       = 400; /* 25ms at 16kHz */
    config.frameShift        = 160; /* 10ms at 16kHz */
    config.numFilters        = 26;
    config.numCoefficients   = 13;
    config.includeDelta      = true;
    config.includeDeltaDelta = false;

    /* Set up advanced options */
    TinyAIAudioFeaturesAdvancedOptions options;
    tinyaiAudioFeaturesInitAdvancedOptions(&options);

    /* Extract features */
    TinyAIAudioFeatures features;
    if (!tinyaiAudioExtractMFCC(&audio, &config, &options, &features)) {
        fprintf(stderr, "Failed to extract MFCC features\n");
        tinyaiAudioDataFree(&audio);
        return false;
    }

    /* Print feature dimensions */
    printf("Extracted features: %d frames x %d coefficients\n", features.numFrames,
           features.numFeatures);

    /* Clean up */
    tinyaiAudioFeaturesFree(&features);
    tinyaiAudioDataFree(&audio);

    printf("Audio feature extraction test passed!\n");
    return true;
}

/**
 * Test audio model creation and processing
 * @return true on success, false on failure
 */
static bool testAudioModel()
{
    printf("Testing audio model creation and processing...\n");

    /* Create a sample audio file */
    const char *sampleFile = "test_audio.wav";
    if (!createSampleAudio(sampleFile, 2000, 440.0f)) {
        fprintf(stderr, "Failed to create sample audio\n");
        return false;
    }

    /* Load audio file */
    TinyAIAudioData audio;
    if (!tinyaiAudioLoadFile(sampleFile, TINYAI_AUDIO_FORMAT_WAV, &audio)) {
        fprintf(stderr, "Failed to load audio file\n");
        return false;
    }

    /* Set up model configuration */
    TinyAIAudioModelConfig config;
    memset(&config, 0, sizeof(config));

    /* Feature configuration */
    config.featuresConfig.type              = TINYAI_AUDIO_FEATURES_MFCC;
    config.featuresConfig.frameLength       = 400; /* 25ms at 16kHz */
    config.featuresConfig.frameShift        = 160; /* 10ms at 16kHz */
    config.featuresConfig.numFilters        = 26;
    config.featuresConfig.numCoefficients   = 13;
    config.featuresConfig.includeDelta      = true;
    config.featuresConfig.includeDeltaDelta = false;

    /* Model architecture */
    config.hiddenSize          = 64;
    config.numLayers           = 2;
    config.numClasses          = 10; /* Example: 10 speech commands */
    config.use4BitQuantization = true;
    config.useSIMD             = true;
    config.weightsFile         = NULL; /* Initialize with random weights */

    /* Create audio model */
    TinyAIAudioModel *model = tinyaiAudioModelCreate(&config);
    if (!model) {
        fprintf(stderr, "Failed to create audio model\n");
        tinyaiAudioDataFree(&audio);
        return false;
    }

    /* Initialize output structure */
    TinyAIAudioModelOutput output;
    if (!tinyaiAudioModelOutputInit(&output, config.numClasses)) {
        fprintf(stderr, "Failed to initialize audio model output\n");
        tinyaiAudioModelFree(model);
        tinyaiAudioDataFree(&audio);
        return false;
    }

    /* Process audio with model */
    if (!tinyaiAudioModelProcess(model, &audio, &output)) {
        fprintf(stderr, "Failed to process audio with model\n");
        tinyaiAudioModelOutputFree(&output);
        tinyaiAudioModelFree(model);
        tinyaiAudioDataFree(&audio);
        return false;
    }

    /* Print results */
    printf("Model prediction: class %d with confidence %.2f%%\n", output.predictedClass,
           output.confidence * 100.0f);

    /* Print top 3 probabilities */
    printf("Top probabilities:\n");
    for (int i = 0; i < 3 && i < config.numClasses; i++) {
        int   maxIdx  = 0;
        float maxProb = output.probabilities[0];

        /* Find max probability among remaining classes */
        for (int j = 1; j < config.numClasses; j++) {
            if (output.probabilities[j] > maxProb) {
                maxIdx  = j;
                maxProb = output.probabilities[j];
            }
        }

        printf("  Class %d: %.2f%%\n", maxIdx, maxProb * 100.0f);

        /* Set this probability to zero so we find the next highest */
        output.probabilities[maxIdx] = 0.0f;
    }

    /* Clean up */
    tinyaiAudioModelOutputFree(&output);
    tinyaiAudioModelFree(model);
    tinyaiAudioDataFree(&audio);

    printf("Audio model test passed!\n");
    return true;
}

/**
 * Main test function
 */
int main(int argc, char *argv[])
{
    printf("TinyAI Audio Model Tests\n");
    printf("=========================\n");

    /* Seed random number generator */
    srand((unsigned int)time(NULL));

    /* Run tests */
    bool featuresResult = testAudioFeatures();
    bool modelResult    = testAudioModel();

    /* Print overall result */
    printf("\nTest Results:\n");
    printf("  Audio Features: %s\n", featuresResult ? "PASSED" : "FAILED");
    printf("  Audio Model: %s\n", modelResult ? "PASSED" : "FAILED");

    return (featuresResult && modelResult) ? 0 : 1;
}
