/**
 * @file vad.c
 * @brief Voice Activity Detection implementation for TinyAI
 */

#include "vad.h"
#include "../../../core/memory.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Default configuration values */
#define DEFAULT_FRAME_SIZE 20          /* 20 ms */
#define DEFAULT_FRAME_SHIFT 10         /* 10 ms */
#define DEFAULT_SENSITIVITY 0.5f       /* Medium sensitivity */
#define DEFAULT_USE_ZCR true           /* Use zero-crossing rate */
#define DEFAULT_ZCR_WEIGHT 0.3f        /* Weight for ZCR */
#define DEFAULT_ENERGY_THRESHOLD 0.05f /* Energy threshold */
#define DEFAULT_ZCR_THRESHOLD 0.2f     /* ZCR threshold */
#define DEFAULT_SMOOTHING true         /* Apply smoothing */
#define DEFAULT_HANGOVER_FRAMES 3      /* Hangover frames */

/* Adaptive threshold parameters */
#define ADAPTATION_RATE 0.05f      /* Rate of adaptation */
#define MIN_ENERGY_THRESHOLD 0.01f /* Minimum energy threshold */
#define MAX_ENERGY_THRESHOLD 0.2f  /* Maximum energy threshold */
#define MIN_ZCR_THRESHOLD 0.05f    /* Minimum ZCR threshold */
#define MAX_ZCR_THRESHOLD 0.5f     /* Maximum ZCR threshold */

/**
 * Initialize VAD configuration with default values
 * @param config Configuration structure to initialize
 */
void tinyaiVADInitConfig(TinyAIVADConfig *config)
{
    if (!config) {
        return;
    }

    config->frameSize       = DEFAULT_FRAME_SIZE;
    config->frameShift      = DEFAULT_FRAME_SHIFT;
    config->sensitivity     = DEFAULT_SENSITIVITY;
    config->useZcr          = DEFAULT_USE_ZCR;
    config->zcrWeight       = DEFAULT_ZCR_WEIGHT;
    config->energyThreshold = DEFAULT_ENERGY_THRESHOLD;
    config->zcrThreshold    = DEFAULT_ZCR_THRESHOLD;
    config->smoothing       = DEFAULT_SMOOTHING;
    config->hangoverFrames  = DEFAULT_HANGOVER_FRAMES;
}

/**
 * Create a new VAD state
 * @param config VAD configuration
 * @param sampleRate Sample rate of audio in Hz
 * @return New VAD state, or NULL on failure
 */
TinyAIVADState *tinyaiVADCreate(const TinyAIVADConfig *config, int sampleRate)
{
    if (!config || sampleRate <= 0) {
        return NULL;
    }

    /* Allocate state */
    TinyAIVADState *state = (TinyAIVADState *)malloc(sizeof(TinyAIVADState));
    if (!state) {
        return NULL;
    }

    /* Initialize state */
    memset(state, 0, sizeof(TinyAIVADState));
    state->config     = *config;
    state->sampleRate = sampleRate;

    /* Calculate samples per frame and shift */
    state->samplesPerFrame = (state->sampleRate * state->config.frameSize) / 1000;
    state->samplesPerShift = (state->sampleRate * state->config.frameShift) / 1000;

    /* Allocate buffer */
    state->bufferSize = state->samplesPerFrame;
    state->buffer     = (float *)malloc(state->bufferSize * sizeof(float));
    if (!state->buffer) {
        free(state);
        return NULL;
    }

    /* Initialize buffer */
    memset(state->buffer, 0, state->bufferSize * sizeof(float));
    state->bufferIndex = 0;

    /* Initialize activity buffer */
    state->activitySize = 1000; /* Start with space for 1000 frames */
    state->activity     = (bool *)malloc(state->activitySize * sizeof(bool));
    if (!state->activity) {
        free(state->buffer);
        free(state);
        return NULL;
    }

    /* Initialize activity buffer */
    memset(state->activity, 0, state->activitySize * sizeof(bool));
    state->activityIndex = 0;

    /* Initialize running statistics */
    state->runningEnergy   = 0.0f;
    state->runningZcr      = 0.0f;
    state->frameCount      = 0;
    state->hangoverCounter = 0;
    state->lastActive      = false;

    return state;
}

/**
 * Free VAD state
 * @param state VAD state to free
 */
void tinyaiVADFree(TinyAIVADState *state)
{
    if (!state) {
        return;
    }

    /* Free buffers */
    if (state->buffer) {
        free(state->buffer);
        state->buffer = NULL;
    }

    if (state->activity) {
        free(state->activity);
        state->activity = NULL;
    }

    /* Free state */
    free(state);
}

/**
 * Reset VAD state
 * @param state VAD state to reset
 * @return true on success, false on failure
 */
bool tinyaiVADReset(TinyAIVADState *state)
{
    if (!state) {
        return false;
    }

    /* Reset buffer */
    memset(state->buffer, 0, state->bufferSize * sizeof(float));
    state->bufferIndex = 0;

    /* Reset activity */
    state->activityIndex = 0;

    /* Reset running statistics */
    state->runningEnergy   = 0.0f;
    state->runningZcr      = 0.0f;
    state->frameCount      = 0;
    state->hangoverCounter = 0;
    state->lastActive      = false;

    return true;
}

/**
 * Calculate energy of a frame
 * @param samples Audio samples
 * @param numSamples Number of samples
 * @return Energy of the frame
 */
static float calculateEnergy(const float *samples, int numSamples)
{
    if (!samples || numSamples <= 0) {
        return 0.0f;
    }

    float sum = 0.0f;
    for (int i = 0; i < numSamples; i++) {
        sum += samples[i] * samples[i];
    }

    return sum / numSamples;
}

/**
 * Calculate zero-crossing rate of a frame
 * @param samples Audio samples
 * @param numSamples Number of samples
 * @return Zero-crossing rate of the frame
 */
static float calculateZCR(const float *samples, int numSamples)
{
    if (!samples || numSamples <= 1) {
        return 0.0f;
    }

    int crossings = 0;
    for (int i = 1; i < numSamples; i++) {
        if ((samples[i - 1] >= 0.0f && samples[i] < 0.0f) ||
            (samples[i - 1] < 0.0f && samples[i] >= 0.0f)) {
            crossings++;
        }
    }

    return (float)crossings / (float)(numSamples - 1);
}

/**
 * Process a frame of audio samples
 * @param state VAD state
 * @param samples Audio samples
 * @param numSamples Number of samples
 * @param activity Output voice activity (true = voice, false = silence)
 * @return true on success, false on failure
 */
bool tinyaiVADProcessFrame(TinyAIVADState *state, const float *samples, int numSamples,
                           bool *activity)
{
    if (!state || !samples || !activity || numSamples <= 0) {
        return false;
    }

    /* Calculate energy and zero-crossing rate */
    float energy = calculateEnergy(samples, numSamples);
    float zcr    = state->config.useZcr ? calculateZCR(samples, numSamples) : 0.0f;

    /* Update running statistics */
    if (state->frameCount == 0) {
        state->runningEnergy = energy;
        state->runningZcr    = zcr;
    }
    else {
        /* Simple exponential moving average */
        state->runningEnergy = state->runningEnergy * 0.95f + energy * 0.05f;
        state->runningZcr    = state->runningZcr * 0.95f + zcr * 0.05f;
    }

    /* Determine voice activity */
    bool isActive = false;

    /* Energy-based decision */
    float scaledThreshold =
        state->config.energyThreshold * (1.0f - state->config.sensitivity * 0.5f);
    bool energyActive = energy > scaledThreshold;

    /* Zero-crossing rate decision */
    bool zcrActive = false;
    if (state->config.useZcr) {
        float zcrThreshold = state->config.zcrThreshold * (1.0f + state->config.sensitivity * 0.5f);

        /* For speech, ZCR is typically lower during voiced segments */
        zcrActive = zcr < zcrThreshold;
    }

    /* Combine decisions */
    if (state->config.useZcr) {
        /* Weighted combination */
        isActive = (energyActive && (1.0f - state->config.zcrWeight)) ||
                   (zcrActive && state->config.zcrWeight);
    }
    else {
        isActive = energyActive;
    }

    /* Apply hangover scheme for smoother detection */
    if (state->config.smoothing) {
        if (isActive) {
            state->hangoverCounter = state->config.hangoverFrames;
            state->lastActive      = true;
        }
        else if (state->hangoverCounter > 0) {
            state->hangoverCounter--;
            isActive          = true;
            state->lastActive = true;
        }
        else {
            state->lastActive = false;
        }
    }

    /* Store activity result */
    *activity = isActive;

    /* Store in activity buffer if there's space */
    if (state->activityIndex < state->activitySize) {
        state->activity[state->activityIndex++] = isActive;
    }

    /* Update frame count */
    state->frameCount++;

    return true;
}

/**
 * Process a full audio buffer
 * @param state VAD state
 * @param audio Audio data
 * @param activity Output array of voice activity (caller must free)
 * @param activitySize Output size of activity array
 * @return true on success, false on failure
 */
bool tinyaiVADProcessAudio(TinyAIVADState *state, const TinyAIAudioData *audio, bool **activity,
                           int *activitySize)
{
    if (!state || !audio || !activity || !activitySize || !audio->data) {
        return false;
    }

    /* Reset state */
    tinyaiVADReset(state);

    /* Calculate number of frames */
    int numSamples = audio->dataSize / sizeof(float);
    int numFrames  = (numSamples - state->samplesPerFrame) / state->samplesPerShift + 1;
    if (numFrames <= 0) {
        return false;
    }

    /* Allocate activity buffer */
    *activity = (bool *)malloc(numFrames * sizeof(bool));
    if (!*activity) {
        return false;
    }
    *activitySize = numFrames;

    /* Process frames */
    const float *samples = (const float *)audio->data;
    for (int i = 0; i < numFrames; i++) {
        int  offset        = i * state->samplesPerShift;
        bool frameActivity = false;

        /* Process frame */
        if (!tinyaiVADProcessFrame(state, samples + offset, state->samplesPerFrame,
                                   &frameActivity)) {
            free(*activity);
            *activity     = NULL;
            *activitySize = 0;
            return false;
        }

        /* Store result */
        (*activity)[i] = frameActivity;
    }

    return true;
}

/**
 * Visualize voice activity detection results on console
 * @param activity Voice activity array
 * @param activitySize Size of activity array
 * @param width Width of visualization in characters
 */
void tinyaiVADVisualize(const bool *activity, int activitySize, int width)
{
    if (!activity || activitySize <= 0 || width <= 0) {
        return;
    }

    /* Determine how many frames to display per character */
    int framesPerChar = (activitySize + width - 1) / width;
    if (framesPerChar < 1) {
        framesPerChar = 1;
    }

    /* Print header */
    printf("Voice Activity Detection Results:\n");
    printf("┌");
    for (int i = 0; i < width; i++) {
        printf("─");
    }
    printf("┐\n");

    /* Print visualization */
    printf("│");
    for (int i = 0; i < width; i++) {
        int startFrame = i * framesPerChar;
        int endFrame   = (i + 1) * framesPerChar;
        if (endFrame > activitySize) {
            endFrame = activitySize;
        }

        /* Count active frames */
        int activeCount = 0;
        for (int j = startFrame; j < endFrame; j++) {
            if (activity[j]) {
                activeCount++;
            }
        }

        /* Determine character to print */
        float activeRatio = (float)activeCount / (float)(endFrame - startFrame);
        if (activeRatio >= 0.5f) {
            printf("█"); /* Active */
        }
        else if (activeRatio > 0.0f) {
            printf("▒"); /* Partially active */
        }
        else {
            printf(" "); /* Inactive */
        }
    }
    printf("│\n");

    /* Print footer */
    printf("└");
    for (int i = 0; i < width; i++) {
        printf("─");
    }
    printf("┘\n");

    /* Print summary */
    int totalActive = 0;
    for (int i = 0; i < activitySize; i++) {
        if (activity[i]) {
            totalActive++;
        }
    }
    float activePercent = 100.0f * (float)totalActive / (float)activitySize;
    printf("Voice activity: %.1f%% (%d/%d frames)\n", activePercent, totalActive, activitySize);
}
