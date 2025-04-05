/**
 * @file audio_model.h
 * @brief Audio model interface for TinyAI
 *
 * This header defines the interface for audio models in TinyAI,
 * including structure definitions and function declarations for
 * audio classification, keyword spotting, and feature extraction.
 */

#ifndef TINYAI_AUDIO_MODEL_H
#define TINYAI_AUDIO_MODEL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Audio format configuration
 */
typedef struct {
    int sampleRate;    /* Sample rate in Hz (e.g., 16000) */
    int channels;      /* Number of audio channels (1 for mono, 2 for stereo) */
    int bitsPerSample; /* Bits per sample (8, 16, 24, etc.) */
} TinyAIAudioFormat;

/**
 * Raw audio data
 */
typedef struct {
    void             *data;       /* Pointer to raw audio samples */
    size_t            dataSize;   /* Size of audio data in bytes */
    TinyAIAudioFormat format;     /* Audio format information */
    int               durationMs; /* Duration in milliseconds */
} TinyAIAudioData;

/**
 * Audio feature extraction type
 */
typedef enum {
    TINYAI_AUDIO_FEATURES_MFCC,        /* Mel-frequency cepstral coefficients */
    TINYAI_AUDIO_FEATURES_MEL,         /* Mel spectrogram */
    TINYAI_AUDIO_FEATURES_SPECTROGRAM, /* Regular spectrogram */
    TINYAI_AUDIO_FEATURES_RAW          /* Raw waveform (no feature extraction) */
} TinyAIAudioFeaturesType;

/**
 * Audio feature extraction configuration
 */
typedef struct {
    TinyAIAudioFeaturesType type;              /* Type of features to extract */
    int                     frameLength;       /* Frame length in samples */
    int                     frameShift;        /* Frame shift in samples */
    int                     numFilters;        /* Number of mel filters (for MFCC or MEL) */
    int                     numCoefficients;   /* Number of coefficients (for MFCC) */
    bool                    includeDelta;      /* Whether to include delta features */
    bool                    includeDeltaDelta; /* Whether to include delta-delta features */
    bool                    useLogMel;         /* Whether to use log-mel features */
    float                   preEmphasis;       /* Pre-emphasis coefficient (0.0 to disable) */
} TinyAIAudioFeaturesConfig;

/**
 * Extracted audio features
 */
typedef struct {
    float                  *data;        /* Pointer to feature data */
    size_t                  dataSize;    /* Size of feature data in bytes */
    int                     numFrames;   /* Number of frames */
    int                     numFeatures; /* Number of features per frame */
    TinyAIAudioFeaturesType type;        /* Type of features */
} TinyAIAudioFeatures;

/**
 * Audio model configuration
 */
typedef struct {
    TinyAIAudioFeaturesConfig featuresConfig;      /* Feature extraction configuration */
    int                       hiddenSize;          /* Size of hidden layers */
    int                       numLayers;           /* Number of model layers */
    int                       numClasses;          /* Number of output classes */
    bool                      use4BitQuantization; /* Whether to use 4-bit quantization */
    bool                      useSIMD;             /* Whether to use SIMD acceleration */
    const char               *weightsFile;         /* Path to weights file (NULL for random) */
} TinyAIAudioModelConfig;

/**
 * Audio model output
 */
typedef struct {
    float *logits;         /* Raw model output logits */
    float *probabilities;  /* Softmax probabilities */
    int    predictedClass; /* Index of highest probability class */
    float  confidence;     /* Confidence of prediction (0.0-1.0) */
} TinyAIAudioModelOutput;

/**
 * Audio model structure
 * Opaque pointer to hide implementation details
 */
typedef struct TinyAIAudioModel TinyAIAudioModel;

/**
 * Create an audio model
 * @param config Configuration for the audio model
 * @return New audio model, or NULL on failure
 */
TinyAIAudioModel *tinyaiAudioModelCreate(const TinyAIAudioModelConfig *config);

/**
 * Free an audio model
 * @param model The model to free
 */
void tinyaiAudioModelFree(TinyAIAudioModel *model);

/**
 * Process audio data with the model
 * @param model The audio model to use
 * @param audio The audio data to process
 * @param output The output structure to fill
 * @return true on success, false on failure
 */
bool tinyaiAudioModelProcess(TinyAIAudioModel *model, const TinyAIAudioData *audio,
                             TinyAIAudioModelOutput *output);

/**
 * Extract features from audio data
 * @param audio The audio data to process
 * @param config Feature extraction configuration
 * @param features Output structure to receive extracted features
 * @return true on success, false on failure
 */
bool tinyaiAudioFeaturesExtract(const TinyAIAudioData           *audio,
                                const TinyAIAudioFeaturesConfig *config,
                                TinyAIAudioFeatures             *features);

/**
 * Free audio features
 * @param features The features to free
 */
void tinyaiAudioFeaturesFree(TinyAIAudioFeatures *features);

/**
 * Load audio data from a file
 * @param path Path to the audio file
 * @param audio Output structure to receive audio data
 * @return true on success, false on failure
 */
bool tinyaiAudioDataLoad(const char *path, TinyAIAudioData *audio);

/**
 * Free audio data
 * @param audio The audio data to free
 */
void tinyaiAudioDataFree(TinyAIAudioData *audio);

/**
 * Initialize audio model output structure
 * @param output The output structure to initialize
 * @param numClasses Number of output classes
 * @return true on success, false on failure
 */
bool tinyaiAudioModelOutputInit(TinyAIAudioModelOutput *output, int numClasses);

/**
 * Free audio model output
 * @param output The output to free
 */
void tinyaiAudioModelOutputFree(TinyAIAudioModelOutput *output);

/**
 * Enable SIMD acceleration for audio model
 * @param model The model to configure
 * @param enable Whether to enable SIMD
 * @return true on success, false on failure
 */
bool tinyaiAudioModelEnableSIMD(TinyAIAudioModel *model, bool enable);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_AUDIO_MODEL_H */
