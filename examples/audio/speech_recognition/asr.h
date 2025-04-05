/**
 * @file asr.h
 * @brief Automatic Speech Recognition for TinyAI
 *
 * This header defines the speech recognition functionality
 * for TinyAI, providing a lightweight implementation for
 * converting speech to text.
 */

#ifndef TINYAI_ASR_H
#define TINYAI_ASR_H

#include "../../../models/audio/audio_features.h"
#include "../../../models/audio/audio_model.h"
#include "../../../models/audio/audio_utils.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Maximum number of characters in an output transcript
 */
#define TINYAI_ASR_MAX_TRANSCRIPT_LENGTH 4096

/**
 * Maximum token length in characters
 */
#define TINYAI_ASR_MAX_TOKEN_LENGTH 64

/**
 * Maximum number of tokens in a recognition result
 */
#define TINYAI_ASR_MAX_TOKENS 512

/**
 * Maximum beam width for decoding
 */
#define TINYAI_ASR_MAX_BEAM_WIDTH 32

/**
 * Recognition mode (tradeoff between speed and accuracy)
 */
typedef enum {
    TINYAI_ASR_MODE_FAST,     /* Prioritize speed over accuracy */
    TINYAI_ASR_MODE_BALANCED, /* Balance speed and accuracy */
    TINYAI_ASR_MODE_ACCURATE  /* Prioritize accuracy over speed */
} TinyAIASRMode;

/**
 * Token type
 */
typedef enum {
    TINYAI_ASR_TOKEN_WORD,        /* Regular word */
    TINYAI_ASR_TOKEN_PUNCTUATION, /* Punctuation mark */
    TINYAI_ASR_TOKEN_NOISE,       /* Non-speech sound */
    TINYAI_ASR_TOKEN_SILENCE      /* Silence */
} TinyAIASRTokenType;

/**
 * Token information structure
 */
typedef struct {
    char               text[TINYAI_ASR_MAX_TOKEN_LENGTH]; /* Token text */
    TinyAIASRTokenType type;                              /* Token type */
    float              confidence;                        /* Confidence score (0.0-1.0) */
    float              startTime;                         /* Start time in seconds */
    float              endTime;                           /* End time in seconds */
} TinyAIASRToken;

/**
 * Recognition result structure
 */
typedef struct {
    TinyAIASRToken tokens[TINYAI_ASR_MAX_TOKENS];                /* Array of recognized tokens */
    int            numTokens;                                    /* Number of tokens */
    char           transcript[TINYAI_ASR_MAX_TRANSCRIPT_LENGTH]; /* Full transcript text */
    float          confidence;                                   /* Overall confidence score */
} TinyAIASRResult;

/**
 * Language model type
 */
typedef enum {
    TINYAI_ASR_LM_NONE,    /* No language model */
    TINYAI_ASR_LM_UNIGRAM, /* Unigram probabilities only */
    TINYAI_ASR_LM_BIGRAM,  /* Bigram language model */
    TINYAI_ASR_LM_TRIGRAM  /* Trigram language model */
} TinyAIASRLanguageModelType;

/**
 * ASR configuration
 */
typedef struct {
    TinyAIASRMode              mode;                 /* Recognition mode */
    TinyAIASRLanguageModelType lmType;               /* Type of language model to use */
    float                      lmWeight;             /* Weight for language model (0.0-1.0) */
    int                        beamWidth;            /* Beam width for decoding */
    bool                       enablePunctuation;    /* Whether to infer punctuation */
    bool                       enableVerboseOutput;  /* Whether to print verbose output */
    bool                       enableWordTimestamps; /* Whether to compute word timestamps */
    bool                       filterProfanity;      /* Whether to filter profanity */
    float                      vadSensitivity; /* Voice activity detection sensitivity (0.0-1.0) */
    char                      *customVocabulary; /* Additional vocabulary (comma-separated) */
} TinyAIASRConfig;

/**
 * Decoder hypothesis
 */
typedef struct TinyAIASRHypothesis TinyAIASRHypothesis;

/**
 * Acoustic model
 */
typedef struct TinyAIASRAcousticModel TinyAIASRAcousticModel;

/**
 * Language model
 */
typedef struct TinyAIASRLanguageModel TinyAIASRLanguageModel;

/**
 * Speech recognition state
 */
typedef struct {
    TinyAIASRConfig         config;        /* Configuration */
    TinyAIASRAcousticModel *acousticModel; /* Acoustic model */
    TinyAIASRLanguageModel *languageModel; /* Language model */

    /* Feature extraction */
    TinyAIAudioFeaturesConfig featuresConfig; /* Feature extraction configuration */
    float                    *features;       /* Buffer for features */
    int                       featuresSize;   /* Size of features buffer */
    int                       featureIndex;   /* Current index in features buffer */

    /* Recognition state */
    TinyAIASRHypothesis *hypotheses;    /* Beam search hypotheses */
    int                  maxHypotheses; /* Maximum number of hypotheses */
    float               *phonemeProbs;  /* Phoneme probabilities buffer */
    int                  numPhonemes;   /* Number of phonemes */

    /* Results */
    TinyAIASRResult currentResult; /* Current recognition result */
    bool            resultReady;   /* Whether a result is ready */

    /* Audio processing */
    int   sampleRate; /* Sample rate of input audio */
    bool  useVAD;     /* Whether to use voice activity detection */
    void *vadState;   /* VAD state (if used) */
} TinyAIASRState;

/**
 * Initialize default ASR configuration
 * @param config Configuration structure to initialize
 */
void tinyaiASRInitConfig(TinyAIASRConfig *config);

/**
 * Create a new ASR state
 * @param config Configuration
 * @param acousticModelPath Path to acoustic model file
 * @param languageModelPath Path to language model file (can be NULL)
 * @return New ASR state, or NULL on failure
 */
TinyAIASRState *tinyaiASRCreate(const TinyAIASRConfig *config, const char *acousticModelPath,
                                const char *languageModelPath);

/**
 * Free ASR state
 * @param state ASR state to free
 */
void tinyaiASRFree(TinyAIASRState *state);

/**
 * Reset ASR state
 * @param state ASR state to reset
 * @return true on success, false on failure
 */
bool tinyaiASRReset(TinyAIASRState *state);

/**
 * Begin a new recognition session
 * @param state ASR state
 * @param sampleRate Sample rate of input audio
 * @return true on success, false on failure
 */
bool tinyaiASRBeginRecognition(TinyAIASRState *state, int sampleRate);

/**
 * End the current recognition session and finalize results
 * @param state ASR state
 * @return true on success, false on failure
 */
bool tinyaiASREndRecognition(TinyAIASRState *state);

/**
 * Process audio frame for speech recognition
 * @param state ASR state
 * @param frame Audio frame (float samples)
 * @param frameSize Number of samples in frame
 * @return true on success, false on failure
 */
bool tinyaiASRProcessFrame(TinyAIASRState *state, const float *frame, int frameSize);

/**
 * Process complete audio for speech recognition
 * @param state ASR state
 * @param audio Audio data
 * @param result Output recognition result
 * @return true on success, false on failure
 */
bool tinyaiASRProcessAudio(TinyAIASRState *state, const TinyAIAudioData *audio,
                           TinyAIASRResult *result);

/**
 * Get the current recognition result
 * @param state ASR state
 * @param result Output recognition result
 * @return true on success, false on failure
 */
bool tinyaiASRGetResult(TinyAIASRState *state, TinyAIASRResult *result);

/**
 * Get the partial recognition result during streaming
 * @param state ASR state
 * @param result Output recognition result
 * @return true on success, false on failure
 */
bool tinyaiASRGetPartialResult(TinyAIASRState *state, TinyAIASRResult *result);

/**
 * Calibrate acoustic model for current environment (adapt to noise, etc.)
 * @param state ASR state
 * @param calibrationAudio Audio data for calibration (ambient noise)
 * @return true on success, false on failure
 */
bool tinyaiASRCalibrateAcousticModel(TinyAIASRState        *state,
                                     const TinyAIAudioData *calibrationAudio);

/**
 * Add custom vocabulary words to improve recognition
 * @param state ASR state
 * @param vocabulary Array of vocabulary words
 * @param numWords Number of words
 * @param weight Weight to assign to custom vocabulary (0.0-1.0)
 * @return true on success, false on failure
 */
bool tinyaiASRAddCustomVocabulary(TinyAIASRState *state, const char **vocabulary, int numWords,
                                  float weight);

/**
 * Save recognition result to file
 * @param result Recognition result
 * @param filePath Path to output file
 * @param includeTimestamps Whether to include timestamps in output
 * @return true on success, false on failure
 */
bool tinyaiASRSaveResult(const TinyAIASRResult *result, const char *filePath,
                         bool includeTimestamps);

/**
 * Get word error rate between recognized text and reference text
 * @param result Recognition result
 * @param referenceText Reference text
 * @return Word error rate (0.0-1.0)
 */
float tinyaiASRCalculateWER(const TinyAIASRResult *result, const char *referenceText);

/**
 * Print information about available models
 */
void tinyaiASRPrintModelInfo(void);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_ASR_H */
