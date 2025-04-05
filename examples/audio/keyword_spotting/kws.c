/**
 * @file kws.c
 * @brief Keyword Spotting implementation for TinyAI
 */

#include "kws.h"
#include "../../../core/memory.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Default configuration values */
#define DEFAULT_DETECTION_THRESHOLD 0.5f   /* Medium sensitivity */
#define DEFAULT_FRAME_SIZE 25              /* 25 ms frames */
#define DEFAULT_FRAME_SHIFT 10             /* 10 ms shift */
#define DEFAULT_USE_DELTAS true            /* Use delta features */
#define DEFAULT_NUM_MFCC 13                /* 13 MFCC coefficients */
#define DEFAULT_NUM_CONTEXT_FRAMES 7       /* 7 frames of context (3 before, 3 after) */
#define DEFAULT_SMOOTH_DETECTIONS true     /* Apply smoothing */
#define DEFAULT_MIN_DETECTION_DURATION 50  /* Minimum 50 ms for a valid detection */
#define DEFAULT_NOISE_ADAPTATION_RATE 0.1f /* Noise adaptation rate */

/* Keyword spotting model definition
 * This is a simplified model for demonstration purposes
 * In a real implementation, this would be a neural network model
 */
struct TinyAIKWSModel {
    int     inputSize;   /* Size of input features */
    int     numKeywords; /* Number of keywords supported */
    char  **keywords;    /* Array of keyword strings */
    float **weights;     /* Simulated weights (for demonstration) */
    bool    initialized; /* Whether the model is initialized */
};

/**
 * Initialize the default keyword spotting configuration
 * @param config Configuration structure to initialize
 */
void tinyaiKWSInitConfig(TinyAIKWSConfig *config)
{
    if (!config) {
        return;
    }

    config->detectionThreshold   = DEFAULT_DETECTION_THRESHOLD;
    config->frameSize            = DEFAULT_FRAME_SIZE;
    config->frameShift           = DEFAULT_FRAME_SHIFT;
    config->useDeltas            = DEFAULT_USE_DELTAS;
    config->numMfccCoefficients  = DEFAULT_NUM_MFCC;
    config->numContextFrames     = DEFAULT_NUM_CONTEXT_FRAMES;
    config->smoothDetections     = DEFAULT_SMOOTH_DETECTIONS;
    config->minDetectionDuration = DEFAULT_MIN_DETECTION_DURATION;
    config->noiseAdaptationRate  = DEFAULT_NOISE_ADAPTATION_RATE;
}

/**
 * Initialize feature extraction configuration based on KWS config
 * @param kwsConfig KWS configuration
 * @param featuresConfig Output features configuration
 */
static void initFeatureConfig(const TinyAIKWSConfig     *kwsConfig,
                              TinyAIAudioFeaturesConfig *featuresConfig)
{
    featuresConfig->type        = TINYAI_AUDIO_FEATURES_MFCC;
    featuresConfig->frameLength = (16000 * kwsConfig->frameSize) / 1000; /* Assuming 16kHz audio */
    featuresConfig->frameShift  = (16000 * kwsConfig->frameShift) / 1000;
    featuresConfig->numFilters  = 26; /* Standard value for speech */
    featuresConfig->numCoefficients   = kwsConfig->numMfccCoefficients;
    featuresConfig->includeDelta      = kwsConfig->useDeltas;
    featuresConfig->includeDeltaDelta = kwsConfig->useDeltas;
    featuresConfig->useLogMel         = true;
    featuresConfig->preEmphasis       = 0.97f; /* Standard value for speech */
}

/**
 * Create a simple keyword spotting model
 * @param numKeywords Number of keywords to support
 * @param inputSize Size of input features
 * @return New model, or NULL on failure
 */
static TinyAIKWSModel *createModel(int numKeywords, int inputSize)
{
    TinyAIKWSModel *model = (TinyAIKWSModel *)malloc(sizeof(TinyAIKWSModel));
    if (!model) {
        return NULL;
    }

    /* Initialize model */
    model->inputSize   = inputSize;
    model->numKeywords = numKeywords;
    model->initialized = false;

    /* Allocate keywords array */
    model->keywords = (char **)malloc(numKeywords * sizeof(char *));
    if (!model->keywords) {
        free(model);
        return NULL;
    }

    for (int i = 0; i < numKeywords; i++) {
        model->keywords[i] = NULL;
    }

    /* Allocate weights "matrix" for simulation */
    model->weights = (float **)malloc(numKeywords * sizeof(float *));
    if (!model->weights) {
        free(model->keywords);
        free(model);
        return NULL;
    }

    for (int i = 0; i < numKeywords; i++) {
        model->weights[i] = (float *)malloc(inputSize * sizeof(float));
        if (!model->weights[i]) {
            for (int j = 0; j < i; j++) {
                free(model->weights[j]);
            }
            free(model->weights);
            free(model->keywords);
            free(model);
            return NULL;
        }

        /* Initialize with some random values for simulation */
        for (int j = 0; j < inputSize; j++) {
            model->weights[i][j] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 0.1f;
        }
    }

    model->initialized = true;
    return model;
}

/**
 * Free keyword spotting model
 * @param model Model to free
 */
static void freeModel(TinyAIKWSModel *model)
{
    if (!model) {
        return;
    }

    if (model->keywords) {
        for (int i = 0; i < model->numKeywords; i++) {
            if (model->keywords[i]) {
                free(model->keywords[i]);
            }
        }
        free(model->keywords);
    }

    if (model->weights) {
        for (int i = 0; i < model->numKeywords; i++) {
            if (model->weights[i]) {
                free(model->weights[i]);
            }
        }
        free(model->weights);
    }

    free(model);
}

/**
 * Add keyword to model
 * @param model Model to update
 * @param keyword Keyword to add
 * @param keywordIndex Index to add at
 * @return true on success, false on failure
 */
static bool addKeywordToModel(TinyAIKWSModel *model, const char *keyword, int keywordIndex)
{
    if (!model || !keyword || keywordIndex < 0 || keywordIndex >= model->numKeywords) {
        return false;
    }

    /* Free existing keyword if present */
    if (model->keywords[keywordIndex]) {
        free(model->keywords[keywordIndex]);
    }

    /* Allocate and copy new keyword */
    model->keywords[keywordIndex] = strdup(keyword);
    if (!model->keywords[keywordIndex]) {
        return false;
    }

    return true;
}

/**
 * Create a new keyword spotting state
 * @param config Configuration
 * @param modelPath Path to model file
 * @return New keyword spotting state, or NULL on failure
 */
TinyAIKWSState *tinyaiKWSCreate(const TinyAIKWSConfig *config, const char *modelPath)
{
    if (!config) {
        return NULL;
    }

    /* Allocate state */
    TinyAIKWSState *state = (TinyAIKWSState *)malloc(sizeof(TinyAIKWSState));
    if (!state) {
        return NULL;
    }

    /* Initialize state */
    memset(state, 0, sizeof(TinyAIKWSState));
    state->config      = *config;
    state->numKeywords = 0;

    /* Initialize features configuration */
    initFeatureConfig(&state->config, &state->featuresConfig);

    /* Create model (would load from file in real implementation) */
    int numMfccFeatures = state->config.numMfccCoefficients;
    if (state->config.useDeltas) {
        numMfccFeatures *= 3; /* MFCC + delta + delta-delta */
    }
    int contextFrames = 1 + 2 * state->config.numContextFrames; /* center frame + context */
    int inputSize     = numMfccFeatures * contextFrames;

    state->model = createModel(TINYAI_KWS_MAX_KEYWORDS, inputSize);
    if (!state->model) {
        free(state);
        return NULL;
    }

    /* Allocate feature buffer */
    state->featureBufferSize = contextFrames * numMfccFeatures;
    state->featureBuffer     = (float *)malloc(state->featureBufferSize * sizeof(float));
    if (!state->featureBuffer) {
        freeModel(state->model);
        free(state);
        return NULL;
    }
    memset(state->featureBuffer, 0, state->featureBufferSize * sizeof(float));
    state->featureBufferIndex = 0;

    /* Allocate detection state buffers */
    state->scoresSize = TINYAI_KWS_MAX_KEYWORDS;
    state->scores     = (float *)malloc(state->scoresSize * sizeof(float));
    if (!state->scores) {
        free(state->featureBuffer);
        freeModel(state->model);
        free(state);
        return NULL;
    }
    memset(state->scores, 0, state->scoresSize * sizeof(float));

    state->smoothedScores = (float *)malloc(state->scoresSize * sizeof(float));
    if (!state->smoothedScores) {
        free(state->scores);
        free(state->featureBuffer);
        freeModel(state->model);
        free(state);
        return NULL;
    }
    memset(state->smoothedScores, 0, state->scoresSize * sizeof(float));

    state->activeDetections = (int *)malloc(state->scoresSize * sizeof(int));
    if (!state->activeDetections) {
        free(state->smoothedScores);
        free(state->scores);
        free(state->featureBuffer);
        freeModel(state->model);
        free(state);
        return NULL;
    }
    memset(state->activeDetections, 0, state->scoresSize * sizeof(int));

    /* Allocate detection results buffer */
    state->detectionsSize = 100; /* Initial size */
    state->detections =
        (TinyAIKWSDetection *)malloc(state->detectionsSize * sizeof(TinyAIKWSDetection));
    if (!state->detections) {
        free(state->activeDetections);
        free(state->smoothedScores);
        free(state->scores);
        free(state->featureBuffer);
        freeModel(state->model);
        free(state);
        return NULL;
    }
    memset(state->detections, 0, state->detectionsSize * sizeof(TinyAIKWSDetection));
    state->numDetections = 0;

    /* Initialize noise level */
    state->noiseLevel = 0.1f; /* Initial estimate */

    return state;
}

/**
 * Free keyword spotting state
 * @param state State to free
 */
void tinyaiKWSFree(TinyAIKWSState *state)
{
    if (!state) {
        return;
    }

    /* Free model */
    if (state->model) {
        freeModel(state->model);
    }

    /* Free feature buffer */
    if (state->featureBuffer) {
        free(state->featureBuffer);
    }

    /* Free features */
    if (state->features) {
        tinyaiAudioFeaturesFree(state->features);
    }

    /* Free detection state */
    if (state->scores) {
        free(state->scores);
    }
    if (state->smoothedScores) {
        free(state->smoothedScores);
    }
    if (state->activeDetections) {
        free(state->activeDetections);
    }

    /* Free detections */
    if (state->detections) {
        free(state->detections);
    }

    /* Free state */
    free(state);
}

/**
 * Reset keyword spotting state
 * @param state State to reset
 * @return true on success, false on failure
 */
bool tinyaiKWSReset(TinyAIKWSState *state)
{
    if (!state) {
        return false;
    }

    /* Reset feature buffer */
    if (state->featureBuffer) {
        memset(state->featureBuffer, 0, state->featureBufferSize * sizeof(float));
    }
    state->featureBufferIndex = 0;

    /* Reset detection state */
    if (state->scores) {
        memset(state->scores, 0, state->scoresSize * sizeof(float));
    }
    if (state->smoothedScores) {
        memset(state->smoothedScores, 0, state->scoresSize * sizeof(float));
    }
    if (state->activeDetections) {
        memset(state->activeDetections, 0, state->scoresSize * sizeof(int));
    }

    /* Reset features */
    if (state->features) {
        tinyaiAudioFeaturesFree(state->features);
        state->features = NULL;
    }

    /* Reset detections */
    state->numDetections = 0;

    /* Reset noise level */
    state->noiseLevel = 0.1f;

    return true;
}

/**
 * Add a keyword to detect
 * @param state Keyword spotting state
 * @param keyword Keyword text
 * @param threshold Detection threshold (0.0-1.0), or negative to use default
 * @return true on success, false on failure
 */
bool tinyaiKWSAddKeyword(TinyAIKWSState *state, const char *keyword, float threshold)
{
    if (!state || !keyword || state->numKeywords >= TINYAI_KWS_MAX_KEYWORDS) {
        return false;
    }

    /* Set threshold */
    float actualThreshold = threshold;
    if (actualThreshold < 0.0f) {
        actualThreshold = TINYAI_KWS_DEFAULT_THRESHOLD;
    }
    if (actualThreshold > 1.0f) {
        actualThreshold = 1.0f;
    }

    /* Add keyword */
    int               index = state->numKeywords;
    TinyAIKWSKeyword *kw    = &state->keywords[index];

    /* Copy keyword text */
    strncpy(kw->word, keyword, TINYAI_KWS_MAX_KEYWORD_LENGTH - 1);
    kw->word[TINYAI_KWS_MAX_KEYWORD_LENGTH - 1] = '\0';

    /* Set properties */
    kw->threshold  = actualThreshold;
    kw->modelIndex = index;

    /* Add to model */
    if (!addKeywordToModel(state->model, keyword, index)) {
        return false;
    }

    /* Increment count */
    state->numKeywords++;

    return true;
}

/**
 * Perform keyword detection on features
 * @param state Keyword spotting state
 * @param features Frame features
 * @param numFeatures Number of features
 * @return true on success, false on failure
 */
static bool detectKeywords(TinyAIKWSState *state, const float *features, int numFeatures)
{
    if (!state || !features || numFeatures <= 0) {
        return false;
    }

    /* Run "model" (simulation) */
    for (int i = 0; i < state->numKeywords; i++) {
        float sum = 0.0f;

        /* Simple dot product with weights as a simulation */
        for (int j = 0; j < numFeatures && j < state->model->inputSize; j++) {
            sum += features[j] * state->model->weights[i][j];
        }

        /* Apply sigmoid activation */
        state->scores[i] = 1.0f / (1.0f + expf(-sum));

        /* Apply smoothing */
        if (state->config.smoothDetections) {
            state->smoothedScores[i] = state->smoothedScores[i] * 0.8f + state->scores[i] * 0.2f;
        }
        else {
            state->smoothedScores[i] = state->scores[i];
        }

        /* Check against threshold */
        float threshold = state->keywords[i].threshold * (1.0f + state->noiseLevel);
        bool  isActive  = state->smoothedScores[i] > threshold;

        /* Update active detections */
        if (isActive) {
            state->activeDetections[i]++;
        }
        else if (state->activeDetections[i] > 0) {
            /* End of detection */
            int duration          = state->activeDetections[i];
            int minDurationFrames = state->config.minDetectionDuration / state->config.frameShift;

            if (duration >= minDurationFrames) {
                /* Valid detection, add to results */
                if (state->numDetections < state->detectionsSize) {
                    int                 idx = state->numDetections++;
                    TinyAIKWSDetection *det = &state->detections[idx];

                    det->keywordIndex = i;
                    det->confidence   = state->smoothedScores[i];
                    det->startFrame   = 0; /* Would calculate from history in real impl */
                    det->endFrame     = 0; /* Would calculate from history in real impl */
                    det->startTime    = 0.0f;
                    det->endTime      = 0.0f;
                }
            }

            /* Reset active detection */
            state->activeDetections[i] = 0;
        }
    }

    return true;
}

/**
 * Process audio frame for keyword detection
 * @param state Keyword spotting state
 * @param frame Audio frame (float samples)
 * @param frameSize Number of samples in frame
 * @return true on success, false on failure
 */
bool tinyaiKWSProcessFrame(TinyAIKWSState *state, const float *frame, int frameSize)
{
    if (!state || !frame || frameSize <= 0) {
        return false;
    }

    /* This function would typically:
     * 1. Convert frame to MFCC features
     * 2. Store in feature buffer
     * 3. When enough context frames, perform detection
     *
     * This is a simplified implementation
     */

    /* Extract features (simplified, would use audio_features.h in real impl) */
    float features[13] = {0}; /* Simplified MFCC features */

    /* Simple energy-based features for simulation */
    float totalEnergy = 0.0f;
    for (int i = 0; i < frameSize; i++) {
        totalEnergy += frame[i] * frame[i];
    }

    /* Update noise level */
    float frameEnergy = totalEnergy / frameSize;
    if (frameEnergy < state->noiseLevel) {
        state->noiseLevel = state->noiseLevel * (1.0f - state->config.noiseAdaptationRate) +
                            frameEnergy * state->config.noiseAdaptationRate;
    }

    /* Generate some fake features for the simulation */
    for (int i = 0; i < 13; i++) {
        features[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * sqrtf(frameEnergy);
    }

    /* Would typically handle feature buffer with context here */

    /* Simulate feature vector for detection */
    float featureVector[39]; /* 13 MFCC + delta + delta-delta */
    for (int i = 0; i < 13; i++) {
        featureVector[i]      = features[i];
        featureVector[i + 13] = features[i] * 0.5f; /* Fake delta */
        featureVector[i + 26] = features[i] * 0.2f; /* Fake delta-delta */
    }

    /* Run detection */
    return detectKeywords(state, featureVector, 39);
}

/**
 * Process full audio buffer for keyword detection
 * @param state Keyword spotting state
 * @param audio Audio data
 * @param detections Output array of detections (caller must free)
 * @param numDetections Output number of detections
 * @return true on success, false on failure
 */
bool tinyaiKWSProcessAudio(TinyAIKWSState *state, const TinyAIAudioData *audio,
                           TinyAIKWSDetection **detections, int *numDetections)
{
    if (!state || !audio || !detections || !numDetections || !audio->data) {
        return false;
    }

    /* Reset state */
    tinyaiKWSReset(state);

    /* Process audio in frames */
    const float *samples    = (const float *)audio->data;
    int          numSamples = audio->dataSize / sizeof(float);

    int frameSizeSamples  = (audio->format.sampleRate * state->config.frameSize) / 1000;
    int frameShiftSamples = (audio->format.sampleRate * state->config.frameShift) / 1000;

    if (frameSizeSamples <= 0 || frameShiftSamples <= 0) {
        return false;
    }

    for (int i = 0; i + frameSizeSamples <= numSamples; i += frameShiftSamples) {
        if (!tinyaiKWSProcessFrame(state, samples + i, frameSizeSamples)) {
            return false;
        }
    }

    /* Return detections */
    if (state->numDetections > 0) {
        *detections =
            (TinyAIKWSDetection *)malloc(state->numDetections * sizeof(TinyAIKWSDetection));
        if (!*detections) {
            return false;
        }

        memcpy(*detections, state->detections, state->numDetections * sizeof(TinyAIKWSDetection));
        *numDetections = state->numDetections;

        /* Calculate times */
        float frameShiftSeconds = (float)state->config.frameShift / 1000.0f;
        for (int i = 0; i < *numDetections; i++) {
            (*detections)[i].startTime = (*detections)[i].startFrame * frameShiftSeconds;
            (*detections)[i].endTime   = (*detections)[i].endFrame * frameShiftSeconds;
        }
    }
    else {
        *detections    = NULL;
        *numDetections = 0;
    }

    return true;
}

/**
 * Get current detection results
 * @param state Keyword spotting state
 * @param detections Output array of detections (not to be freed by caller)
 * @param maxDetections Maximum number of detections to return
 * @param numDetections Output number of detections
 * @return true on success, false on failure
 */
bool tinyaiKWSGetDetections(TinyAIKWSState *state, TinyAIKWSDetection *detections,
                            int maxDetections, int *numDetections)
{
    if (!state || !detections || !numDetections || maxDetections <= 0) {
        return false;
    }

    /* Copy detections */
    int count = state->numDetections < maxDetections ? state->numDetections : maxDetections;
    memcpy(detections, state->detections, count * sizeof(TinyAIKWSDetection));
    *numDetections = count;

    return true;
}

/**
 * Get list of available keywords in the model
 * @param state Keyword spotting state
 * @param keywords Output array of keywords (caller allocates)
 * @param maxKeywords Maximum number of keywords to return
 * @param numKeywords Output number of keywords
 * @return true on success, false on failure
 */
bool tinyaiKWSGetAvailableKeywords(TinyAIKWSState *state, char **keywords, int maxKeywords,
                                   int *numKeywords)
{
    if (!state || !keywords || !numKeywords || maxKeywords <= 0) {
        return false;
    }

    /* Copy keywords */
    int count = state->numKeywords < maxKeywords ? state->numKeywords : maxKeywords;
    for (int i = 0; i < count; i++) {
        keywords[i] = state->keywords[i].word;
    }
    *numKeywords = count;

    return true;
}

/**
 * Visualize keyword detection results
 * @param state Keyword spotting state
 * @param audio Audio data
 * @param detections Detection results
 * @param numDetections Number of detections
 * @param width Width of visualization in characters
 */
void tinyaiKWSVisualizeDetections(TinyAIKWSState *state, const TinyAIAudioData *audio,
                                  const TinyAIKWSDetection *detections, int numDetections,
                                  int width)
{
    if (!state || !audio || !detections || numDetections <= 0 || width <= 0) {
        return;
    }

    /* Calculate audio duration */
    float duration = (float)audio->durationMs / 1000.0f;

    /* Print header */
    printf("Keyword Detection Results (%.2f seconds):\n", duration);
    printf("┌");
    for (int i = 0; i < width; i++) {
        printf("─");
    }
    printf("┐\n");

    /* Calculate time scale */
    float timePerChar = duration / width;

    /* Print visualization */
    printf("│");
    for (int i = 0; i < width; i++) {
        float startTime = i * timePerChar;
        float endTime   = (i + 1) * timePerChar;

        /* Check for detections in this time window */
        bool hasDetection = false;
        for (int j = 0; j < numDetections; j++) {
            if ((detections[j].startTime < endTime) && (detections[j].endTime > startTime)) {
                hasDetection = true;
                break;
            }
        }

        if (hasDetection) {
            printf("█"); /* Detection */
        }
        else {
            printf(" "); /* No detection */
        }
    }
    printf("│\n");

    /* Print footer */
    printf("└");
    for (int i = 0; i < width; i++) {
        printf("─");
    }
    printf("┘\n");

    /* Print time scale */
    printf("0");
    for (int i = 1; i < 10; i++) {
        int pos = (i * width) / 10;
        for (int j = 0; j < pos - (i > 1 ? ((i - 1) * width) / 10 + (i - 1) : 1); j++) {
            printf(" ");
        }
        printf("%.1f", i * duration / 10.0f);
    }
    printf(" (sec)\n\n");

    /* Print detection details */
    printf("Detected keywords:\n");
    for (int i = 0; i < numDetections; i++) {
        int keywordIndex = detections[i].keywordIndex;
        if (keywordIndex >= 0 && keywordIndex < state->numKeywords) {
            printf("%d. \"%s\" at %.2f-%.2f sec (confidence: %.2f)\n", i + 1,
                   state->keywords[keywordIndex].word, detections[i].startTime,
                   detections[i].endTime, detections[i].confidence);
        }
    }
}
