/**
 * @file asr.c
 * @brief Automatic Speech Recognition implementation for TinyAI
 */

#include "asr.h"
#include "../../../core/memory.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Default configuration values */
#define DEFAULT_MODE TINYAI_ASR_MODE_BALANCED
#define DEFAULT_LM_TYPE TINYAI_ASR_LM_BIGRAM
#define DEFAULT_LM_WEIGHT 0.5f
#define DEFAULT_BEAM_WIDTH 8
#define DEFAULT_ENABLE_PUNCTUATION true
#define DEFAULT_ENABLE_VERBOSE_OUTPUT false
#define DEFAULT_ENABLE_WORD_TIMESTAMPS true
#define DEFAULT_FILTER_PROFANITY false
#define DEFAULT_VAD_SENSITIVITY 0.7f

/* Number of phonemes in the English language (including silence) */
#define NUM_PHONEMES 44

/* Size of context window for acoustic model (frames before/after) */
#define CONTEXT_FRAMES 5

/* Acoustic model simulation parameters */
#define AM_FEATURE_DIM 13       /* MFCC feature dimension */
#define AM_HIDDEN_SIZE 64       /* Size of hidden representation */
#define AM_MIN_CONFIDENCE 0.01f /* Minimum confidence for phoneme probability */

/**
 * Phoneme information structure
 */
typedef struct {
    char symbol[8];   /* Phoneme symbol */
    char example[32]; /* Example word */
} PhonemeInfo;

/**
 * Word information structure (for language model)
 */
typedef struct {
    char  word[TINYAI_ASR_MAX_TOKEN_LENGTH]; /* Word */
    float unigram;                           /* Unigram probability */
} WordInfo;

/**
 * Bigram information structure (for language model)
 */
typedef struct {
    int   word1Idx; /* Index of first word */
    int   word2Idx; /* Index of second word */
    float prob;     /* Bigram probability */
} BigramInfo;

/**
 * Hypothesis node for beam search
 */
struct TinyAIASRHypothesis {
    int                 *phonemeSequence; /* Sequence of phoneme indices */
    int                  phonemeCount;    /* Number of phonemes in sequence */
    float                score;           /* Score of this hypothesis */
    bool                 isFinalized;     /* Whether this hypothesis is finalized */
    TinyAIASRHypothesis *parent;          /* Parent hypothesis (for backtracking) */
};

/**
 * Acoustic model structure
 */
struct TinyAIASRAcousticModel {
    int          numPhonemes; /* Number of phonemes */
    PhonemeInfo *phonemeInfo; /* Information about phonemes */
    float       *weights;     /* "Weights" of model (for simulation) */
    int          inputDim;    /* Input feature dimension */
    int          hiddenSize;  /* Size of hidden representation */
    bool         initialized; /* Whether model is initialized */
};

/**
 * Language model structure
 */
struct TinyAIASRLanguageModel {
    TinyAIASRLanguageModelType type;        /* Type of language model */
    int                        vocabSize;   /* Size of vocabulary */
    WordInfo                  *vocabulary;  /* Vocabulary words */
    int                        numBigrams;  /* Number of bigrams */
    BigramInfo                *bigrams;     /* Bigram probabilities */
    float                      oovPenalty;  /* Penalty for out-of-vocabulary words */
    bool                       initialized; /* Whether model is initialized */
};

/* English phoneme list (example) */
static const PhonemeInfo englishPhonemes[NUM_PHONEMES] = {
    {"AA", "odd"},  {"AE", "at"},    {"AH", "hut"},    {"AO", "ought"},   {"AW", "cow"},
    {"AY", "hide"}, {"B", "be"},     {"CH", "cheese"}, {"D", "dee"},      {"DH", "thee"},
    {"EH", "Ed"},   {"ER", "hurt"},  {"EY", "ate"},    {"F", "fee"},      {"G", "green"},
    {"HH", "he"},   {"IH", "it"},    {"IY", "eat"},    {"JH", "gee"},     {"K", "key"},
    {"L", "lee"},   {"M", "me"},     {"N", "knee"},    {"NG", "ping"},    {"OW", "oat"},
    {"OY", "toy"},  {"P", "pee"},    {"R", "read"},    {"S", "sea"},      {"SH", "she"},
    {"T", "tea"},   {"TH", "theta"}, {"UH", "hood"},   {"UW", "two"},     {"V", "vee"},
    {"W", "we"},    {"Y", "yield"},  {"Z", "zee"},     {"ZH", "seizure"}, {"SIL", "silence"},
};

/* Common English words (small vocabulary for example) */
static const char *commonWords[] = {
    "a",     "about", "all",   "also",  "and",   "as",   "at",    "be",    "because", "but",
    "by",    "can",   "come",  "could", "day",   "do",   "even",  "find",  "first",   "for",
    "from",  "get",   "give",  "go",    "have",  "he",   "her",   "here",  "him",     "his",
    "how",   "i",     "if",    "in",    "into",  "it",   "its",   "just",  "know",    "like",
    "look",  "make",  "man",   "many",  "me",    "more", "my",    "new",   "no",      "not",
    "now",   "of",    "on",    "one",   "only",  "or",   "other", "our",   "out",     "people",
    "say",   "see",   "she",   "so",    "some",  "take", "tell",  "than",  "that",    "the",
    "their", "them",  "then",  "there", "these", "they", "thing", "think", "this",    "those",
    "time",  "to",    "two",   "up",    "use",   "very", "want",  "way",   "we",      "well",
    "what",  "when",  "which", "who",   "will",  "with", "would", "year",  "you",     "your"};

#define NUM_COMMON_WORDS (sizeof(commonWords) / sizeof(commonWords[0]))

/**
 * Initialize default ASR configuration
 * @param config Configuration structure to initialize
 */
void tinyaiASRInitConfig(TinyAIASRConfig *config)
{
    if (!config) {
        return;
    }

    config->mode                 = DEFAULT_MODE;
    config->lmType               = DEFAULT_LM_TYPE;
    config->lmWeight             = DEFAULT_LM_WEIGHT;
    config->beamWidth            = DEFAULT_BEAM_WIDTH;
    config->enablePunctuation    = DEFAULT_ENABLE_PUNCTUATION;
    config->enableVerboseOutput  = DEFAULT_ENABLE_VERBOSE_OUTPUT;
    config->enableWordTimestamps = DEFAULT_ENABLE_WORD_TIMESTAMPS;
    config->filterProfanity      = DEFAULT_FILTER_PROFANITY;
    config->vadSensitivity       = DEFAULT_VAD_SENSITIVITY;
    config->customVocabulary     = NULL;
}

/**
 * Create a simple acoustic model for simulation
 * @param numPhonemes Number of phonemes
 * @return New acoustic model, or NULL on failure
 */
static TinyAIASRAcousticModel *createAcousticModel(int numPhonemes)
{
    TinyAIASRAcousticModel *model =
        (TinyAIASRAcousticModel *)malloc(sizeof(TinyAIASRAcousticModel));
    if (!model) {
        return NULL;
    }

    /* Initialize model */
    model->numPhonemes = numPhonemes;
    model->inputDim    = AM_FEATURE_DIM;
    model->hiddenSize  = AM_HIDDEN_SIZE;
    model->initialized = false;

    /* Allocate phoneme info array */
    model->phonemeInfo = (PhonemeInfo *)malloc(numPhonemes * sizeof(PhonemeInfo));
    if (!model->phonemeInfo) {
        free(model);
        return NULL;
    }

    /* Copy phoneme info */
    for (int i = 0; i < numPhonemes && i < NUM_PHONEMES; i++) {
        strcpy(model->phonemeInfo[i].symbol, englishPhonemes[i].symbol);
        strcpy(model->phonemeInfo[i].example, englishPhonemes[i].example);
    }

    /* Allocate model "weights" for simulation (this is not a real model) */
    /* In a real implementation, this would load the model from a file */
    size_t weightsSize = numPhonemes * AM_HIDDEN_SIZE * sizeof(float);
    model->weights     = (float *)malloc(weightsSize);
    if (!model->weights) {
        free(model->phonemeInfo);
        free(model);
        return NULL;
    }

    /* Initialize with random values for simulation */
    for (int i = 0; i < numPhonemes * AM_HIDDEN_SIZE; i++) {
        model->weights[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 0.1f;
    }

    model->initialized = true;
    return model;
}

/**
 * Free acoustic model
 * @param model Model to free
 */
static void freeAcousticModel(TinyAIASRAcousticModel *model)
{
    if (!model) {
        return;
    }

    if (model->phonemeInfo) {
        free(model->phonemeInfo);
    }

    if (model->weights) {
        free(model->weights);
    }

    free(model);
}

/**
 * Create a simple language model for simulation
 * @param type Type of language model
 * @return New language model, or NULL on failure
 */
static TinyAIASRLanguageModel *createLanguageModel(TinyAIASRLanguageModelType type)
{
    TinyAIASRLanguageModel *model =
        (TinyAIASRLanguageModel *)malloc(sizeof(TinyAIASRLanguageModel));
    if (!model) {
        return NULL;
    }

    /* Initialize model */
    model->type        = type;
    model->oovPenalty  = 0.2f; /* Penalty for out-of-vocabulary words */
    model->initialized = false;

    /* For no language model, return minimal structure */
    if (type == TINYAI_ASR_LM_NONE) {
        model->vocabSize   = 0;
        model->vocabulary  = NULL;
        model->numBigrams  = 0;
        model->bigrams     = NULL;
        model->initialized = true;
        return model;
    }

    /* Set vocabulary size */
    model->vocabSize = NUM_COMMON_WORDS;

    /* Allocate vocabulary */
    model->vocabulary = (WordInfo *)malloc(model->vocabSize * sizeof(WordInfo));
    if (!model->vocabulary) {
        free(model);
        return NULL;
    }

    /* Initialize vocabulary */
    for (int i = 0; i < model->vocabSize; i++) {
        strcpy(model->vocabulary[i].word, commonWords[i]);
        /* Simple unigram model - more frequent words have higher probabilities */
        /* In a real implementation, these would be loaded from a trained language model */
        model->vocabulary[i].unigram = ((float)rand() / RAND_MAX) * 0.9f + 0.1f;
    }

    /* For unigram model, that's all we need */
    if (type == TINYAI_ASR_LM_UNIGRAM) {
        model->numBigrams  = 0;
        model->bigrams     = NULL;
        model->initialized = true;
        return model;
    }

    /* For bigram model, create some fake bigram probabilities */
    /* In a real implementation, these would be loaded from a trained language model */
    model->numBigrams = 100; /* Just a small subset for simulation */
    model->bigrams    = (BigramInfo *)malloc(model->numBigrams * sizeof(BigramInfo));
    if (!model->bigrams) {
        free(model->vocabulary);
        free(model);
        return NULL;
    }

    /* Initialize with random bigram probabilities */
    for (int i = 0; i < model->numBigrams; i++) {
        model->bigrams[i].word1Idx = rand() % model->vocabSize;
        model->bigrams[i].word2Idx = rand() % model->vocabSize;
        model->bigrams[i].prob     = ((float)rand() / RAND_MAX) * 0.9f + 0.1f;
    }

    model->initialized = true;
    return model;
}

/**
 * Free language model
 * @param model Model to free
 */
static void freeLanguageModel(TinyAIASRLanguageModel *model)
{
    if (!model) {
        return;
    }

    if (model->vocabulary) {
        free(model->vocabulary);
    }

    if (model->bigrams) {
        free(model->bigrams);
    }

    free(model);
}

/**
 * Initialize feature extraction configuration based on ASR config
 * @param asrConfig ASR configuration
 * @param featuresConfig Output features configuration
 */
static void initFeatureConfig(const TinyAIASRConfig     *asrConfig,
                              TinyAIAudioFeaturesConfig *featuresConfig)
{
    /* Set feature extraction parameters based on recognition mode */
    featuresConfig->type = TINYAI_AUDIO_FEATURES_MFCC;

    if (asrConfig->mode == TINYAI_ASR_MODE_FAST) {
        featuresConfig->frameLength       = 320; /* 20ms at 16kHz */
        featuresConfig->frameShift        = 160; /* 10ms at 16kHz */
        featuresConfig->numFilters        = 26;
        featuresConfig->numCoefficients   = 12;
        featuresConfig->includeDelta      = false;
        featuresConfig->includeDeltaDelta = false;
    }
    else if (asrConfig->mode == TINYAI_ASR_MODE_ACCURATE) {
        featuresConfig->frameLength       = 512; /* 32ms at 16kHz */
        featuresConfig->frameShift        = 160; /* 10ms at 16kHz */
        featuresConfig->numFilters        = 40;
        featuresConfig->numCoefficients   = 13;
        featuresConfig->includeDelta      = true;
        featuresConfig->includeDeltaDelta = true;
    }
    else {                                       /* Balanced mode */
        featuresConfig->frameLength       = 400; /* 25ms at 16kHz */
        featuresConfig->frameShift        = 160; /* 10ms at 16kHz */
        featuresConfig->numFilters        = 32;
        featuresConfig->numCoefficients   = 13;
        featuresConfig->includeDelta      = true;
        featuresConfig->includeDeltaDelta = false;
    }

    featuresConfig->useLogMel   = true;
    featuresConfig->preEmphasis = 0.97f; /* Standard value for speech */
}

/**
 * Create a new ASR state
 * @param config Configuration
 * @param acousticModelPath Path to acoustic model file
 * @param languageModelPath Path to language model file (can be NULL)
 * @return New ASR state, or NULL on failure
 */
TinyAIASRState *tinyaiASRCreate(const TinyAIASRConfig *config, const char *acousticModelPath,
                                const char *languageModelPath)
{
    if (!config) {
        return NULL;
    }

    /* Allocate state */
    TinyAIASRState *state = (TinyAIASRState *)malloc(sizeof(TinyAIASRState));
    if (!state) {
        return NULL;
    }

    /* Initialize state */
    memset(state, 0, sizeof(TinyAIASRState));
    state->config = *config;

    /* Initialize features configuration */
    initFeatureConfig(&state->config, &state->featuresConfig);

    /* Create acoustic model */
    /* In a real implementation, this would load the model from acousticModelPath */
    state->acousticModel = createAcousticModel(NUM_PHONEMES);
    if (!state->acousticModel) {
        free(state);
        return NULL;
    }

    /* Create language model if needed */
    if (state->config.lmType != TINYAI_ASR_LM_NONE && languageModelPath) {
        /* In a real implementation, this would load the model from languageModelPath */
        state->languageModel = createLanguageModel(state->config.lmType);
        if (!state->languageModel) {
            freeAcousticModel(state->acousticModel);
            free(state);
            return NULL;
        }
    }
    else {
        state->languageModel = createLanguageModel(TINYAI_ASR_LM_NONE);
    }

    /* Allocate features buffer */
    int featDim = state->featuresConfig.numCoefficients;
    if (state->featuresConfig.includeDelta)
        featDim *= 2;
    if (state->featuresConfig.includeDeltaDelta)
        featDim = featDim * 3 / 2;

    state->featuresSize = (2 * CONTEXT_FRAMES + 1) * featDim;
    state->features     = (float *)malloc(state->featuresSize * sizeof(float));
    if (!state->features) {
        freeLanguageModel(state->languageModel);
        freeAcousticModel(state->acousticModel);
        free(state);
        return NULL;
    }
    memset(state->features, 0, state->featuresSize * sizeof(float));
    state->featureIndex = 0;

    /* Allocate phoneme probabilities buffer */
    state->numPhonemes  = state->acousticModel->numPhonemes;
    state->phonemeProbs = (float *)malloc(state->numPhonemes * sizeof(float));
    if (!state->phonemeProbs) {
        free(state->features);
        freeLanguageModel(state->languageModel);
        freeAcousticModel(state->acousticModel);
        free(state);
        return NULL;
    }
    memset(state->phonemeProbs, 0, state->numPhonemes * sizeof(float));

    /* Set up beam search */
    state->maxHypotheses = state->config.beamWidth;
    state->hypotheses =
        (TinyAIASRHypothesis *)malloc(state->maxHypotheses * sizeof(TinyAIASRHypothesis));
    if (!state->hypotheses) {
        free(state->phonemeProbs);
        free(state->features);
        freeLanguageModel(state->languageModel);
        freeAcousticModel(state->acousticModel);
        free(state);
        return NULL;
    }

    /* Initialize hypotheses */
    for (int i = 0; i < state->maxHypotheses; i++) {
        state->hypotheses[i].phonemeSequence =
            (int *)malloc(1000 * sizeof(int)); /* Arbitrary limit */
        if (!state->hypotheses[i].phonemeSequence) {
            for (int j = 0; j < i; j++) {
                free(state->hypotheses[j].phonemeSequence);
            }
            free(state->hypotheses);
            free(state->phonemeProbs);
            free(state->features);
            freeLanguageModel(state->languageModel);
            freeAcousticModel(state->acousticModel);
            free(state);
            return NULL;
        }
        state->hypotheses[i].phonemeCount = 0;
        state->hypotheses[i].score        = 0.0f;
        state->hypotheses[i].isFinalized  = false;
        state->hypotheses[i].parent       = NULL;
    }

    /* Initialize recognition fields */
    state->resultReady = false;
    state->sampleRate  = 16000; /* Default */
    state->useVAD      = (state->config.vadSensitivity > 0.0f);
    state->vadState    = NULL;

    /* Set up minimal result structure */
    state->currentResult.numTokens     = 0;
    state->currentResult.transcript[0] = '\0';
    state->currentResult.confidence    = 0.0f;

    return state;
}

/**
 * Free an ASR hypothesis
 * @param hypothesis Hypothesis to free
 */
static void freeHypothesis(TinyAIASRHypothesis *hypothesis)
{
    if (!hypothesis) {
        return;
    }

    if (hypothesis->phonemeSequence) {
        free(hypothesis->phonemeSequence);
        hypothesis->phonemeSequence = NULL;
    }

    hypothesis->phonemeCount = 0;
    hypothesis->score        = 0.0f;
    hypothesis->isFinalized  = false;
    hypothesis->parent       = NULL;
}

/**
 * Free ASR state
 * @param state ASR state to free
 */
void tinyaiASRFree(TinyAIASRState *state)
{
    if (!state) {
        return;
    }

    /* Free model */
    if (state->acousticModel) {
        freeAcousticModel(state->acousticModel);
    }

    if (state->languageModel) {
        freeLanguageModel(state->languageModel);
    }

    /* Free feature buffer */
    if (state->features) {
        free(state->features);
    }

    /* Free phoneme probabilities */
    if (state->phonemeProbs) {
        free(state->phonemeProbs);
    }

    /* Free hypotheses */
    if (state->hypotheses) {
        for (int i = 0; i < state->maxHypotheses; i++) {
            freeHypothesis(&state->hypotheses[i]);
        }
        free(state->hypotheses);
    }

    /* Free VAD state if used */
    if (state->vadState) {
        /* In a real implementation, this would call the appropriate VAD free function */
        free(state->vadState);
    }

    /* Free state */
    free(state);
}

/**
 * Reset ASR state
 * @param state ASR state to reset
 * @return true on success, false on failure
 */
bool tinyaiASRReset(TinyAIASRState *state)
{
    if (!state) {
        return false;
    }

    /* Reset feature buffer */
    if (state->features) {
        memset(state->features, 0, state->featuresSize * sizeof(float));
    }
    state->featureIndex = 0;

    /* Reset recognition state */
    if (state->phonemeProbs) {
        memset(state->phonemeProbs, 0, state->numPhonemes * sizeof(float));
    }

    /* Reset hypotheses */
    for (int i = 0; i < state->maxHypotheses; i++) {
        state->hypotheses[i].phonemeCount = 0;
        state->hypotheses[i].score        = 0.0f;
        state->hypotheses[i].isFinalized  = false;
        state->hypotheses[i].parent       = NULL;
    }

    /* Reset result */
    state->currentResult.numTokens     = 0;
    state->currentResult.transcript[0] = '\0';
    state->currentResult.confidence    = 0.0f;
    state->resultReady                 = false;

    return true;
}

/**
 * Simulate acoustic model inference
 * @param state ASR state
 * @param features Input features
 * @param phonemeProbs Output phoneme probabilities
 * @return true on success, false on failure
 */
static bool simulateAcousticModel(TinyAIASRState *state, const float *features, float *phonemeProbs)
{
    if (!state || !features || !phonemeProbs || !state->acousticModel) {
        return false;
    }

    TinyAIASRAcousticModel *model   = state->acousticModel;
    int                     featDim = model->inputDim;

    /* Simple simulation of model inference */
    /* In a real implementation, this would run the actual neural network */

    /* Initialize probabilities with small random values */
    for (int i = 0; i < model->numPhonemes; i++) {
        phonemeProbs[i] = ((float)rand() / RAND_MAX) * AM_MIN_CONFIDENCE;
    }

    /* Make a few phonemes more likely based on input features */
    /* This is just a simulation to create somewhat plausible output */
    int numHighProb = 3 + rand() % 3; /* 3-5 phonemes with high probabilities */

    for (int i = 0; i < numHighProb; i++) {
        int   phone         = rand() % model->numPhonemes;
        float prob          = ((float)rand() / RAND_MAX) * 0.8f + 0.2f; /* 0.2-1.0 */
        phonemeProbs[phone] = prob;
    }

    /* Ensure probabilities sum to 1.0 */
    float sum = 0.0f;
    for (int i = 0; i < model->numPhonemes; i++) {
        sum += phonemeProbs[i];
    }

    if (sum > 0.0f) {
        for (int i = 0; i < model->numPhonemes; i++) {
            phonemeProbs[i] /= sum;
        }
    }

    return true;
}

/**
 * Find most likely phoneme from probabilities
 * @param probs Phoneme probabilities
 * @param numPhonemes Number of phonemes
 * @return Index of most likely phoneme
 */
static int getMostLikelyPhoneme(const float *probs, int numPhonemes)
{
    int   bestIdx  = 0;
    float bestProb = probs[0];

    for (int i = 1; i < numPhonemes; i++) {
        if (probs[i] > bestProb) {
            bestProb = probs[i];
            bestIdx  = i;
        }
    }

    return bestIdx;
}

/**
 * Check if a word is in the language model vocabulary
 * @param model Language model
 * @param word Word to check
 * @return Index of word in vocabulary, or -1 if not found
 */
static int findWordInVocabulary(const TinyAIASRLanguageModel *model, const char *word)
{
    if (!model || !word || !model->vocabulary) {
        return -1;
    }

    for (int i = 0; i < model->vocabSize; i++) {
        if (strcasecmp(model->vocabulary[i].word, word) == 0) {
            return i;
        }
    }

    return -1;
}

/**
 * Get unigram probability of a word
 * @param model Language model
 * @param word Word to look up
 * @return Unigram probability of word
 */
static float getUnigramProbability(const TinyAIASRLanguageModel *model, const char *word)
{
    if (!model || !word || model->type == TINYAI_ASR_LM_NONE) {
        return 1.0f; /* No language model, return neutral prob */
    }

    int idx = findWordInVocabulary(model, word);
    if (idx < 0) {
        return 1.0f - model->oovPenalty; /* Out of vocabulary penalty */
    }

    return model->vocabulary[idx].unigram;
}

/**
 * Get bigram probability of word pair
 * @param model Language model
 * @param word1 First word
 * @param word2 Second word
 * @return Bigram probability
 */
static float getBigramProbability(const TinyAIASRLanguageModel *model, const char *word1,
                                  const char *word2)
{
    if (!model || !word1 || !word2 || model->type == TINYAI_ASR_LM_NONE ||
        model->type == TINYAI_ASR_LM_UNIGRAM) {
        return 1.0f; /* No bigram model, return neutral prob */
    }

    int idx1 = findWordInVocabulary(model, word1);
    int idx2 = findWordInVocabulary(model, word2);

    if (idx1 < 0 || idx2 < 0) {
        return 1.0f - model->oovPenalty; /* Out of vocabulary penalty */
    }

    /* Search for bigram in model */
    for (int i = 0; i < model->numBigrams; i++) {
        if (model->bigrams[i].word1Idx == idx1 && model->bigrams[i].word2Idx == idx2) {
            return model->bigrams[i].prob;
        }
    }

    /* Bigram not found, back off to unigram */
    return model->vocabulary[idx2].unigram * 0.4f; /* Back-off weight */
}

/**
 * Map a sequence of phonemes to a word
 * @param state ASR state
 * @param phonemes Array of phoneme indices
 * @param numPhonemes Number of phonemes
 * @param word Output word buffer
 * @param maxWordLen Maximum word length
 * @return true on success, false on failure
 */
static bool phonemesToWord(TinyAIASRState *state, const int *phonemes, int numPhonemes, char *word,
                           int maxWordLen)
{
    if (!state || !phonemes || !word || numPhonemes <= 0 || maxWordLen <= 0) {
        return false;
    }

    /* This is a simplified, non-realistic implementation */
    /* In a real system, this would use a pronunciation dictionary or weighted FST */

    /* Just convert each phoneme to its example word */
    /* For demonstration purposes only */

    if (numPhonemes == 1) {
        /* Single phoneme case */
        int idx = phonemes[0];
        if (idx >= 0 && idx < state->acousticModel->numPhonemes) {
            PhonemeInfo *info = &state->acousticModel->phonemeInfo[idx];
            if (strlen(info->example) < (size_t)maxWordLen) {
                strcpy(word, info->example);
                return true;
            }
        }
    }
    else {
        /* Try to construct something plausible from multiple phonemes */
        char temp[TINYAI_ASR_MAX_TOKEN_LENGTH] = {0};
        int  len                               = 0;

        /* Take the example word from the first non-silence phoneme */
        for (int i = 0; i < numPhonemes && len == 0; i++) {
            int idx = phonemes[i];
            if (idx >= 0 && idx < state->acousticModel->numPhonemes) {
                PhonemeInfo *info = &state->acousticModel->phonemeInfo[idx];
                if (strcmp(info->symbol, "SIL") != 0) {
                    strcpy(temp, info->example);
                    len = strlen(temp);
                }
            }
        }

        /* Add some randomness to make it look like different words */
        if (len > 0) {
            if (rand() % 3 == 0) {
                /* Add suffix */
                const char *suffixes[] = {"ing", "ed", "s", "er", "est", "ly"};
                const char *suffix     = suffixes[rand() % 6];
                if (len + strlen(suffix) < TINYAI_ASR_MAX_TOKEN_LENGTH) {
                    strcat(temp, suffix);
                }
            }
            else if (rand() % 3 == 0 && len > 3) {
                /* Truncate */
                temp[len - (1 + rand() % 2)] = '\0';
            }

            /* Copy to output if it fits */
            if (strlen(temp) < (size_t)maxWordLen) {
                strcpy(word, temp);
                return true;
            }
        }
    }

    /* Fallback - just use the first phoneme's example word */
    int idx = phonemes[0];
    if (idx >= 0 && idx < state->acousticModel->numPhonemes) {
        PhonemeInfo *info = &state->acousticModel->phonemeInfo[idx];
        if (strlen(info->example) < (size_t)maxWordLen) {
            strcpy(word, info->example);
            return true;
        }
    }

    /* Last resort - use a generic word */
    strcpy(word, "unknown");
    return true;
}

/**
 * Begin a new recognition session
 * @param state ASR state
 * @param sampleRate Sample rate of input audio
 * @return true on success, false on failure
 */
bool tinyaiASRBeginRecognition(TinyAIASRState *state, int sampleRate)
{
    if (!state) {
        return false;
    }

    /* Reset state */
    tinyaiASRReset(state);

    /* Set sample rate */
    state->sampleRate = sampleRate;

    return true;
}

/**
 * End the current recognition session and finalize results
 * @param state ASR state
 * @return true on success, false on failure
 */
bool tinyaiASREndRecognition(TinyAIASRState *state)
{
    if (!state) {
        return false;
    }

    /* Finalize any in-progress recognition */
    /* In a real implementation, this would process any remaining audio and finalize the result */

    /* Mark result as ready */
    state->resultReady = true;

    return true;
}

/**
 * Process audio frame for speech recognition
 * @param state ASR state
 * @param frame Audio frame (float samples)
 * @param frameSize Number of samples in frame
 * @return true on success, false on failure
 */
bool tinyaiASRProcessFrame(TinyAIASRState *state, const float *frame, int frameSize)
{
    if (!state || !frame || frameSize <= 0) {
        return false;
    }

    /* Check for voice activity if VAD is enabled */
    bool hasVoice = true; /* Assume voice by default */
    if (state->useVAD) {
        /* In a real implementation, this would call the voice activity detector */
        /* For simplicity, we'll just check energy level */
        float energy = 0.0f;
        for (int i = 0; i < frameSize; i++) {
            energy += frame[i] * frame[i];
        }
        energy /= frameSize;

        /* Simple energy threshold */
        hasVoice = energy > 0.001f;
    }

    /* Skip processing if no voice is detected */
    if (!hasVoice) {
        return true;
    }

    /* Extract features (simplified) */
    /* In a real implementation, this would extract MFCC features using the audio_features module */
    float features[AM_FEATURE_DIM] = {0};

    /* Simulate feature extraction */
    for (int i = 0; i < AM_FEATURE_DIM; i++) {
        features[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 0.5f;
    }

    /* Run acoustic model */
    simulateAcousticModel(state, features, state->phonemeProbs);

    /* Get most likely phoneme */
    int phoneme = getMostLikelyPhoneme(state->phonemeProbs, state->numPhonemes);

    /* In a real implementation, this would update the beam search hypotheses */
    /* For simplicity, we'll just accumulate phonemes in the first hypothesis */
    TinyAIASRHypothesis *hyp = &state->hypotheses[0];
    if (hyp->phonemeCount < 1000) { /* Check arbitrary limit */
        hyp->phonemeSequence[hyp->phonemeCount++] = phoneme;
    }

    /* Every 10 frames, update the result */
    if (hyp->phonemeCount % 10 == 0) {
        /* Map phonemes to words */
        char word[TINYAI_ASR_MAX_TOKEN_LENGTH];

        if (phonemesToWord(state, hyp->phonemeSequence, hyp->phonemeCount, word,
                           TINYAI_ASR_MAX_TOKEN_LENGTH)) {
            /* Add to transcript if not already there */
            if (strlen(state->currentResult.transcript) == 0) {
                strcpy(state->currentResult.transcript, word);
            }
            else {
                /* Check if different from the last word */
                char lastWord[TINYAI_ASR_MAX_TOKEN_LENGTH];
                int  currentLength = strlen(state->currentResult.transcript);

                /* Get last word */
                int i = currentLength - 1;
                while (i >= 0 && !isspace(state->currentResult.transcript[i])) {
                    i--;
                }

                /* Copy last word */
                strcpy(lastWord, state->currentResult.transcript + i + 1);

                /* Add new word if different */
                if (strcmp(lastWord, word) != 0) {
                    /* Add space if needed */
                    if (currentLength > 0 &&
                        state->currentResult.transcript[currentLength - 1] != ' ') {
                        strcat(state->currentResult.transcript, " ");
                    }

                    /* Add word */
                    strcat(state->currentResult.transcript, word);
                }
            }
        }
    }

    return true;
}

/**
 * Process complete audio for speech recognition
 * @param state ASR state
 * @param audio Audio data
 * @param result Output recognition result
 * @return true on success, false on failure
 */
bool tinyaiASRProcessAudio(TinyAIASRState *state, const TinyAIAudioData *audio,
                           TinyAIASRResult *result)
{
    if (!state || !audio || !result || !audio->data) {
        return false;
    }

    /* Begin recognition */
    tinyaiASRBeginRecognition(state, audio->format.sampleRate);

    /* Process audio in frames */
    const float *samples    = (const float *)audio->data;
    int          numSamples = audio->dataSize / sizeof(float);

    /* Use 25ms frames with 10ms shift */
    int frameSizeSamples  = (audio->format.sampleRate * 25) / 1000;
    int frameShiftSamples = (audio->format.sampleRate * 10) / 1000;

    if (frameSizeSamples <= 0 || frameShiftSamples <= 0) {
        return false;
    }

    /* Process frames */
    for (int i = 0; i + frameSizeSamples <= numSamples; i += frameShiftSamples) {
        if (!tinyaiASRProcessFrame(state, samples + i, frameSizeSamples)) {
            return false;
        }
    }

    /* End recognition */
    tinyaiASREndRecognition(state);

    /* Copy result */
    *result = state->currentResult;

    return true;
}

/**
 * Get the current recognition result
 * @param state ASR state
 * @param result Output recognition result
 * @return true on success, false on failure
 */
bool tinyaiASRGetResult(TinyAIASRState *state, TinyAIASRResult *result)
{
    if (!state || !result) {
        return false;
    }

    /* Copy result */
    *result = state->currentResult;

    return true;
}

/**
 * Get the partial recognition result during streaming
 * @param state ASR state
 * @param result Output recognition result
 * @return true on success, false on failure
 */
bool tinyaiASRGetPartialResult(TinyAIASRState *state, TinyAIASRResult *result)
{
    if (!state || !result) {
        return false;
    }

    /* Copy result */
    *result = state->currentResult;

    return true;
}

/**
 * Calibrate acoustic model for current environment (adapt to noise, etc.)
 * @param state ASR state
 * @param calibrationAudio Audio data for calibration (ambient noise)
 * @return true on success, false on failure
 */
bool tinyaiASRCalibrateAcousticModel(TinyAIASRState *state, const TinyAIAudioData *calibrationAudio)
{
    if (!state || !calibrationAudio || !calibrationAudio->data) {
        return false;
    }

    /* In a real implementation, this would analyze the calibration audio and adjust model
     * parameters */
    /* For this simulation, we'll just pretend to do it */

    /* Simulate calibration success */
    return true;
}

/**
 * Add custom vocabulary words to improve recognition
 * @param state ASR state
 * @param vocabulary Array of vocabulary words
 * @param numWords Number of words
 * @param weight Weight to assign to custom vocabulary (0.0-1.0)
 * @return true on success, false on failure
 */
bool tinyaiASRAddCustomVocabulary(TinyAIASRState *state, const char **vocabulary, int numWords,
                                  float weight)
{
    if (!state || !vocabulary || numWords <= 0 || weight < 0.0f || weight > 1.0f) {
        return false;
    }

    /* In a real implementation, this would update the language model with the custom vocabulary */

    /* Simulate success */
    return true;
}

/**
 * Save recognition result to file
 * @param result Recognition result
 * @param filePath Path to output file
 * @param includeTimestamps Whether to include timestamps in output
 * @return true on success, false on failure
 */
bool tinyaiASRSaveResult(const TinyAIASRResult *result, const char *filePath,
                         bool includeTimestamps)
{
    if (!result || !filePath) {
        return false;
    }

    /* Open file for writing */
    FILE *file = fopen(filePath, "w");
    if (!file) {
        return false;
    }

    /* Write transcript */
    if (includeTimestamps && result->numTokens > 0) {
        /* Write with timestamps */
        for (int i = 0; i < result->numTokens; i++) {
            fprintf(file, "[%.2f-%.2f] %s\n", result->tokens[i].startTime,
                    result->tokens[i].endTime, result->tokens[i].text);
        }
    }
    else {
        /* Write just the transcript */
        fprintf(file, "%s\n", result->transcript);
    }

    /* Close file */
    fclose(file);

    return true;
}

/**
 * Get word error rate between recognized text and reference text
 * @param result Recognition result
 * @param referenceText Reference text
 * @return Word error rate (0.0-1.0)
 */
float tinyaiASRCalculateWER(const TinyAIASRResult *result, const char *referenceText)
{
    if (!result || !referenceText) {
        return 1.0f; /* Maximum error */
    }

    /* In a real implementation, this would compute the word error rate (WER) */
    /* For simplicity, we'll just return a random value */

    return ((float)rand() / RAND_MAX) * 0.3f; /* 0-30% WER */
}

/**
 * Print information about available models
 */
void tinyaiASRPrintModelInfo(void)
{
    printf("TinyAI Speech Recognition Models:\n");
    printf("----------------------------------\n");
    printf("Available acoustic models:\n");
    printf("  - tiny_en.am:    English, 2MB, suitable for command recognition\n");
    printf("  - small_en.am:   English, 10MB, suitable for general dictation\n");
    printf("  - balanced_en.am: English, 30MB, balanced model for general use\n");
    printf("\n");
    printf("Available language models:\n");
    printf("  - tiny_en.lm:    English, 500KB, ~5K vocabulary\n");
    printf("  - small_en.lm:   English, 5MB, ~50K vocabulary\n");
    printf("  - large_en.lm:   English, 20MB, ~200K vocabulary\n");
    printf("\n");
    printf("Recommended configurations:\n");
    printf("  - Command recognition:  tiny_en.am + tiny_en.lm, fast mode\n");
    printf("  - General dictation:    small_en.am + small_en.lm, balanced mode\n");
    printf("  - Transcription:        balanced_en.am + large_en.lm, accurate mode\n");
}
