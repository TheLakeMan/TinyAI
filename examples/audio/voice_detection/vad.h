/**
 * @file vad.h
 * @brief Voice Activity Detection for TinyAI
 *
 * This header defines the voice activity detection functionality
 * for TinyAI, supporting energy-based and zero-crossing detection.
 */

#ifndef TINYAI_VAD_H
#define TINYAI_VAD_H

#include "../../../models/audio/audio_model.h"
#include "../../../models/audio/audio_utils.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Voice activity detection configuration
 */
typedef struct {
    int   frameSize;       /* Frame size in milliseconds */
    int   frameShift;      /* Frame shift in milliseconds */
    float sensitivity;     /* Sensitivity (0.0-1.0, higher = more sensitive) */
    bool  useZcr;          /* Whether to use zero-crossing rate */
    float zcrWeight;       /* Weight for zero-crossing rate (0.0-1.0) */
    float energyThreshold; /* Energy threshold for voice activity */
    float zcrThreshold;    /* Zero-crossing rate threshold */
    bool  smoothing;       /* Whether to apply smoothing */
    int   hangoverFrames;  /* Number of hangover frames */
} TinyAIVADConfig;

/**
 * Voice activity detection state
 */
typedef struct {
    TinyAIVADConfig config;          /* VAD configuration */
    int             sampleRate;      /* Sample rate of audio */
    int             samplesPerFrame; /* Number of samples per frame */
    int             samplesPerShift; /* Number of samples per shift */
    float          *buffer;          /* Buffer for current frame */
    int             bufferSize;      /* Size of buffer in samples */
    int             bufferIndex;     /* Current index in buffer */
    bool           *activity;        /* Voice activity results */
    int             activitySize;    /* Size of activity buffer */
    int             activityIndex;   /* Current index in activity buffer */
    int             hangoverCounter; /* Counter for hangover frames */
    bool            lastActive;      /* Whether last frame was active */
    float           runningEnergy;   /* Running average of energy */
    float           runningZcr;      /* Running average of ZCR */
    int             frameCount;      /* Number of frames processed */
} TinyAIVADState;

/**
 * Initialize VAD configuration with default values
 * @param config Configuration structure to initialize
 */
void tinyaiVADInitConfig(TinyAIVADConfig *config);

/**
 * Create a new VAD state
 * @param config VAD configuration
 * @param sampleRate Sample rate of audio in Hz
 * @return New VAD state, or NULL on failure
 */
TinyAIVADState *tinyaiVADCreate(const TinyAIVADConfig *config, int sampleRate);

/**
 * Free VAD state
 * @param state VAD state to free
 */
void tinyaiVADFree(TinyAIVADState *state);

/**
 * Reset VAD state
 * @param state VAD state to reset
 * @return true on success, false on failure
 */
bool tinyaiVADReset(TinyAIVADState *state);

/**
 * Process a frame of audio samples
 * @param state VAD state
 * @param samples Audio samples
 * @param numSamples Number of samples
 * @param activity Output voice activity (true = voice, false = silence)
 * @return true on success, false on failure
 */
bool tinyaiVADProcessFrame(TinyAIVADState *state, const float *samples, int numSamples,
                           bool *activity);

/**
 * Process a full audio buffer
 * @param state VAD state
 * @param audio Audio data
 * @param activity Output array of voice activity (caller must free)
 * @param activitySize Output size of activity array
 * @return true on success, false on failure
 */
bool tinyaiVADProcessAudio(TinyAIVADState *state, const TinyAIAudioData *audio, bool **activity,
                           int *activitySize);

/**
 * Visualize voice activity detection results
 * @param activity Voice activity array
 * @param activitySize Size of activity array
 * @param width Width of visualization in characters
 */
void tinyaiVADVisualize(const bool *activity, int activitySize, int width);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_VAD_H */
