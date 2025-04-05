/**
 * @file audio_utils.c
 * @brief Audio utilities implementation for TinyAI
 */

#include "audio_utils.h"
#include "../../core/io.h"
#include "../../core/memory.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Constants for WAV file format */
#define RIFF_CHUNK_ID 0x46464952 /* "RIFF" */
#define WAVE_FORMAT 0x45564157   /* "WAVE" */
#define FMT_CHUNK_ID 0x20746D66  /* "fmt " */
#define DATA_CHUNK_ID 0x61746164 /* "data" */

/* Maximum number of supported channels */
#define MAX_CHANNELS 8

/* WAV file header structures */
typedef struct {
    uint32_t chunkId;   /* "RIFF" */
    uint32_t chunkSize; /* 4 + (8 + fmtSize) + (8 + dataSize) */
    uint32_t format;    /* "WAVE" */
} RiffHeader;

typedef struct {
    uint32_t chunkId;       /* "fmt " */
    uint32_t chunkSize;     /* 16 for PCM */
    uint16_t audioFormat;   /* 1 for PCM */
    uint16_t numChannels;   /* 1 for mono, 2 for stereo */
    uint32_t sampleRate;    /* 8000, 16000, 44100, etc. */
    uint32_t byteRate;      /* sampleRate * numChannels * bitsPerSample/8 */
    uint16_t blockAlign;    /* numChannels * bitsPerSample/8 */
    uint16_t bitsPerSample; /* 8, 16, 24, etc. */
} FmtHeader;

typedef struct {
    uint32_t chunkId;   /* "data" */
    uint32_t chunkSize; /* Number of bytes in data */
} DataHeader;

/**
 * Convert samples from integer to float format
 * @param input Input integer samples
 * @param output Output float samples
 * @param numSamples Number of samples
 * @param bitsPerSample Bits per sample (8, 16, 24, etc.)
 * @return true on success, false on failure
 */
static bool convertIntToFloat(const void *input, float *output, int numSamples, int bitsPerSample)
{
    if (!input || !output || numSamples <= 0) {
        return false;
    }

    switch (bitsPerSample) {
    case 8: {
        /* 8-bit samples are unsigned [0, 255] */
        const uint8_t *samples = (const uint8_t *)input;
        for (int i = 0; i < numSamples; i++) {
            /* Convert to [-1.0, 1.0] */
            output[i] = ((float)samples[i] - 128.0f) / 128.0f;
        }
        break;
    }
    case 16: {
        /* 16-bit samples are signed [-32768, 32767] */
        const int16_t *samples = (const int16_t *)input;
        for (int i = 0; i < numSamples; i++) {
            /* Convert to [-1.0, 1.0] */
            output[i] = (float)samples[i] / 32768.0f;
        }
        break;
    }
    case 24: {
        /* 24-bit samples are signed, stored as 3 bytes */
        const uint8_t *bytes = (const uint8_t *)input;
        for (int i = 0; i < numSamples; i++) {
            /* Extract 24-bit signed value */
            int value = (bytes[i * 3] << 8) | (bytes[i * 3 + 1] << 16) | (bytes[i * 3 + 2] << 24);
            value >>= 8; /* Shift back to get correct sign extension */
            /* Convert to [-1.0, 1.0] */
            output[i] = (float)value / 8388608.0f; /* 2^23 */
        }
        break;
    }
    case 32: {
        /* 32-bit samples could be signed int or float */
        /* For now, assume signed int */
        const int32_t *samples = (const int32_t *)input;
        for (int i = 0; i < numSamples; i++) {
            /* Convert to [-1.0, 1.0] */
            output[i] = (float)samples[i] / 2147483648.0f; /* 2^31 */
        }
        break;
    }
    default:
        return false;
    }

    return true;
}

/**
 * Convert multi-channel audio to mono
 * @param input Input multi-channel samples (interleaved)
 * @param output Output mono samples
 * @param numFrames Number of frames
 * @param numChannels Number of channels
 * @return true on success, false on failure
 */
static bool convertToMono(const float *input, float *output, int numFrames, int numChannels)
{
    if (!input || !output || numFrames <= 0 || numChannels <= 0) {
        return false;
    }

    /* Special case for mono */
    if (numChannels == 1) {
        if (input != output) {
            memcpy(output, input, numFrames * sizeof(float));
        }
        return true;
    }

    /* Average all channels */
    for (int i = 0; i < numFrames; i++) {
        float sum = 0.0f;
        for (int j = 0; j < numChannels; j++) {
            sum += input[i * numChannels + j];
        }
        output[i] = sum / numChannels;
    }

    return true;
}

/**
 * Load audio data from a WAV file
 * @param path Path to the WAV file
 * @param audio Output structure to receive audio data
 * @return true on success, false on failure
 */
static bool loadWavFile(const char *path, TinyAIAudioData *audio)
{
    if (!path || !audio) {
        return false;
    }

    /* Open file */
    FILE *file = fopen(path, "rb");
    if (!file) {
        return false;
    }

    /* Read RIFF header */
    RiffHeader riffHeader;
    if (fread(&riffHeader, sizeof(RiffHeader), 1, file) != 1 ||
        riffHeader.chunkId != RIFF_CHUNK_ID || riffHeader.format != WAVE_FORMAT) {
        fclose(file);
        return false;
    }

    /* Read format header */
    FmtHeader fmtHeader;
    if (fread(&fmtHeader, sizeof(FmtHeader), 1, file) != 1 || fmtHeader.chunkId != FMT_CHUNK_ID) {
        fclose(file);
        return false;
    }

    /* Skip extra format bytes if present */
    if (fmtHeader.chunkSize > 16) {
        fseek(file, fmtHeader.chunkSize - 16, SEEK_CUR);
    }

    /* Find data chunk */
    DataHeader dataHeader;
    while (1) {
        if (fread(&dataHeader, sizeof(DataHeader), 1, file) != 1) {
            fclose(file);
            return false;
        }
        if (dataHeader.chunkId == DATA_CHUNK_ID) {
            break;
        }
        fseek(file, dataHeader.chunkSize, SEEK_CUR);
    }

    /* Only PCM format is supported */
    if (fmtHeader.audioFormat != 1) {
        fclose(file);
        return false;
    }

    /* Check if we support this format */
    if (fmtHeader.numChannels > MAX_CHANNELS || fmtHeader.bitsPerSample > 32) {
        fclose(file);
        return false;
    }

    /* Set audio format */
    audio->format.sampleRate    = fmtHeader.sampleRate;
    audio->format.channels      = fmtHeader.numChannels;
    audio->format.bitsPerSample = fmtHeader.bitsPerSample;

    /* Calculate duration */
    int bytesPerSample = fmtHeader.bitsPerSample / 8;
    int numSamples     = dataHeader.chunkSize / (bytesPerSample * fmtHeader.numChannels);
    audio->durationMs  = (numSamples * 1000) / fmtHeader.sampleRate;

    /* Read audio data */
    size_t dataSize = dataHeader.chunkSize;
    void  *data     = malloc(dataSize);
    if (!data) {
        fclose(file);
        return false;
    }
    if (fread(data, 1, dataSize, file) != dataSize) {
        free(data);
        fclose(file);
        return false;
    }

    /* Convert to floating-point mono */
    int    totalSamples = numSamples * fmtHeader.numChannels;
    float *floatSamples = (float *)malloc(totalSamples * sizeof(float));
    if (!floatSamples) {
        free(data);
        fclose(file);
        return false;
    }
    if (!convertIntToFloat(data, floatSamples, totalSamples, fmtHeader.bitsPerSample)) {
        free(data);
        free(floatSamples);
        fclose(file);
        return false;
    }

    /* Convert to mono if necessary */
    if (fmtHeader.numChannels > 1) {
        float *monoSamples = (float *)malloc(numSamples * sizeof(float));
        if (!monoSamples) {
            free(data);
            free(floatSamples);
            fclose(file);
            return false;
        }
        if (!convertToMono(floatSamples, monoSamples, numSamples, fmtHeader.numChannels)) {
            free(data);
            free(floatSamples);
            free(monoSamples);
            fclose(file);
            return false;
        }
        free(floatSamples);
        floatSamples           = monoSamples;
        totalSamples           = numSamples;
        audio->format.channels = 1; /* Now mono */
    }

    /* Set audio data */
    audio->data     = floatSamples;
    audio->dataSize = totalSamples * sizeof(float);

    /* Clean up */
    free(data);
    fclose(file);

    return true;
}

/**
 * Load audio data from a file
 * @param path Path to the audio file
 * @param audio Output structure to receive audio data
 * @return true on success, false on failure
 */
bool tinyaiAudioDataLoad(const char *path, TinyAIAudioData *audio)
{
    if (!path || !audio) {
        return false;
    }

    /* Initialize audio structure */
    memset(audio, 0, sizeof(TinyAIAudioData));

    /* Check file extension */
    const char *extension = strrchr(path, '.');
    if (!extension) {
        return false;
    }

    /* Load based on file format */
    if (strcasecmp(extension, ".wav") == 0) {
        return loadWavFile(path, audio);
    }

    /* Unsupported format */
    return false;
}

/**
 * Free audio data
 * @param audio The audio data to free
 */
void tinyaiAudioDataFree(TinyAIAudioData *audio)
{
    if (!audio) {
        return;
    }

    /* Free data */
    if (audio->data) {
        free(audio->data);
        audio->data = NULL;
    }
    audio->dataSize = 0;
}

/**
 * Resample audio data
 * @param input Input audio data
 * @param output Output audio data (preallocated)
 * @param targetSampleRate Target sample rate in Hz
 * @return true on success, false on failure
 */
bool tinyaiAudioResample(const TinyAIAudioData *input, TinyAIAudioData *output,
                         int targetSampleRate)
{
    if (!input || !output || !input->data || targetSampleRate <= 0) {
        return false;
    }

    /* No resampling needed if already at target rate */
    if (input->format.sampleRate == targetSampleRate) {
        /* Make a copy */
        output->format     = input->format;
        output->durationMs = input->durationMs;
        output->dataSize   = input->dataSize;
        output->data       = malloc(input->dataSize);
        if (!output->data) {
            return false;
        }
        memcpy(output->data, input->data, input->dataSize);
        return true;
    }

    /* Simple linear resampling for now */
    int    numSamples    = input->dataSize / sizeof(float);
    double ratio         = (double)targetSampleRate / (double)input->format.sampleRate;
    int    outputSamples = (int)(numSamples * ratio);

    /* Allocate output buffer */
    float *outputData = (float *)malloc(outputSamples * sizeof(float));
    if (!outputData) {
        return false;
    }

    /* Resample */
    const float *inputData = (const float *)input->data;
    for (int i = 0; i < outputSamples; i++) {
        double inputIndex = i / ratio;
        int    index1     = (int)inputIndex;
        int    index2     = index1 + 1;
        float  t          = (float)(inputIndex - index1);

        if (index2 >= numSamples) {
            outputData[i] = inputData[numSamples - 1];
        }
        else {
            outputData[i] = (1.0f - t) * inputData[index1] + t * inputData[index2];
        }
    }

    /* Set output */
    output->format            = input->format;
    output->format.sampleRate = targetSampleRate;
    output->durationMs        = (outputSamples * 1000) / targetSampleRate;
    output->dataSize          = outputSamples * sizeof(float);
    output->data              = outputData;

    return true;
}

/**
 * Apply gain to audio data
 * @param audio Audio data to modify
 * @param gainDb Gain in decibels
 * @return true on success, false on failure
 */
bool tinyaiAudioApplyGain(TinyAIAudioData *audio, float gainDb)
{
    if (!audio || !audio->data) {
        return false;
    }

    /* Convert dB to linear */
    float gain = powf(10.0f, gainDb / 20.0f);

    /* Apply gain */
    float *samples    = (float *)audio->data;
    int    numSamples = audio->dataSize / sizeof(float);
    for (int i = 0; i < numSamples; i++) {
        samples[i] *= gain;
        /* Clamp to [-1, 1] */
        if (samples[i] > 1.0f) {
            samples[i] = 1.0f;
        }
        else if (samples[i] < -1.0f) {
            samples[i] = -1.0f;
        }
    }

    return true;
}

/**
 * Calculate RMS energy of audio data
 * @param audio Audio data
 * @param rms Output RMS energy
 * @return true on success, false on failure
 */
bool tinyaiAudioCalculateRMS(const TinyAIAudioData *audio, float *rms)
{
    if (!audio || !audio->data || !rms) {
        return false;
    }

    /* Calculate RMS */
    const float *samples    = (const float *)audio->data;
    int          numSamples = audio->dataSize / sizeof(float);
    float        sumSquares = 0.0f;
    for (int i = 0; i < numSamples; i++) {
        sumSquares += samples[i] * samples[i];
    }
    *rms = sqrtf(sumSquares / numSamples);

    return true;
}

/**
 * Normalize audio data
 * @param audio Audio data to modify
 * @param targetRMS Target RMS energy
 * @return true on success, false on failure
 */
bool tinyaiAudioNormalize(TinyAIAudioData *audio, float targetRMS)
{
    if (!audio || !audio->data || targetRMS <= 0.0f) {
        return false;
    }

    /* Calculate current RMS */
    float currentRMS;
    if (!tinyaiAudioCalculateRMS(audio, &currentRMS) || currentRMS <= 0.0f) {
        return false;
    }

    /* Calculate gain */
    float gain = targetRMS / currentRMS;

    /* Apply gain */
    float *samples    = (float *)audio->data;
    int    numSamples = audio->dataSize / sizeof(float);
    for (int i = 0; i < numSamples; i++) {
        samples[i] *= gain;
        /* Clamp to [-1, 1] */
        if (samples[i] > 1.0f) {
            samples[i] = 1.0f;
        }
        else if (samples[i] < -1.0f) {
            samples[i] = -1.0f;
        }
    }

    return true;
}

/**
 * Apply silence detection
 * @param audio Audio data
 * @param threshold Silence threshold (0.0-1.0)
 * @param isSilence Output array indicating if each frame is silence
 * @param frameSize Frame size in samples
 * @param frameShift Frame shift in samples
 * @param numFrames Output number of frames
 * @return true on success, false on failure
 */
bool tinyaiAudioDetectSilence(const TinyAIAudioData *audio, float threshold, bool **isSilence,
                              int frameSize, int frameShift, int *numFrames)
{
    if (!audio || !audio->data || !isSilence || !numFrames || threshold < 0.0f || frameSize <= 0 ||
        frameShift <= 0) {
        return false;
    }

    /* Calculate number of frames */
    int samples = audio->dataSize / sizeof(float);
    *numFrames  = (samples - frameSize) / frameShift + 1;
    if (*numFrames <= 0) {
        return false;
    }

    /* Allocate output array */
    *isSilence = (bool *)malloc(*numFrames * sizeof(bool));
    if (!*isSilence) {
        return false;
    }

    /* Check each frame */
    const float *audioData = (const float *)audio->data;
    for (int i = 0; i < *numFrames; i++) {
        int   offset = i * frameShift;
        float maxAbs = 0.0f;
        for (int j = 0; j < frameSize; j++) {
            float abs = fabsf(audioData[offset + j]);
            if (abs > maxAbs) {
                maxAbs = abs;
            }
        }
        (*isSilence)[i] = (maxAbs < threshold);
    }

    return true;
}

/**
 * Trim silence from audio
 * @param audio Audio data to modify
 * @param threshold Silence threshold (0.0-1.0)
 * @param frameSize Frame size in samples
 * @return true on success, false on failure
 */
bool tinyaiAudioTrimSilence(TinyAIAudioData *audio, float threshold, int frameSize)
{
    if (!audio || !audio->data || threshold < 0.0f || frameSize <= 0) {
        return false;
    }

    const float *samples    = (const float *)audio->data;
    int          numSamples = audio->dataSize / sizeof(float);

    /* Find start of audio */
    int start = 0;
    while (start + frameSize <= numSamples) {
        float maxAbs = 0.0f;
        for (int i = 0; i < frameSize; i++) {
            float abs = fabsf(samples[start + i]);
            if (abs > maxAbs) {
                maxAbs = abs;
            }
        }
        if (maxAbs >= threshold) {
            break;
        }
        start += frameSize;
    }

    /* Find end of audio */
    int end = numSamples - frameSize;
    while (end >= 0) {
        float maxAbs = 0.0f;
        for (int i = 0; i < frameSize; i++) {
            float abs = fabsf(samples[end + i]);
            if (abs > maxAbs) {
                maxAbs = abs;
            }
        }
        if (maxAbs >= threshold) {
            end += frameSize;
            break;
        }
        end -= frameSize;
    }

    /* Ensure valid range */
    if (start >= end || end <= start) {
        return false; /* No non-silent audio found */
    }

    /* Create new buffer */
    int    newSamples = end - start;
    float *newData    = (float *)malloc(newSamples * sizeof(float));
    if (!newData) {
        return false;
    }

    /* Copy trimmed audio */
    memcpy(newData, samples + start, newSamples * sizeof(float));

    /* Update audio struct */
    free(audio->data);
    audio->data       = newData;
    audio->dataSize   = newSamples * sizeof(float);
    audio->durationMs = (newSamples * 1000) / audio->format.sampleRate;

    return true;
}

/**
 * Create a new audio buffer
 * @param format Audio format
 * @param durationMs Duration in milliseconds
 * @return New audio data, or NULL on failure
 */
TinyAIAudioData *tinyaiAudioDataCreate(const TinyAIAudioFormat *format, int durationMs)
{
    if (!format || durationMs <= 0 || format->sampleRate <= 0 || format->channels <= 0 ||
        format->bitsPerSample <= 0) {
        return NULL;
    }

    /* Allocate struct */
    TinyAIAudioData *audio = (TinyAIAudioData *)malloc(sizeof(TinyAIAudioData));
    if (!audio) {
        return NULL;
    }

    /* Calculate buffer size */
    int    numSamples = (format->sampleRate * durationMs) / 1000;
    size_t dataSize   = numSamples * format->channels * sizeof(float);

    /* Allocate buffer */
    audio->data = malloc(dataSize);
    if (!audio->data) {
        free(audio);
        return NULL;
    }

    /* Initialize to silence */
    memset(audio->data, 0, dataSize);

    /* Set properties */
    audio->format     = *format;
    audio->durationMs = durationMs;
    audio->dataSize   = dataSize;

    return audio;
}

/**
 * Get audio sample rate
 * @param audio Audio data
 * @return Sample rate in Hz, or 0 on failure
 */
int tinyaiAudioGetSampleRate(const TinyAIAudioData *audio)
{
    if (!audio) {
        return 0;
    }
    return audio->format.sampleRate;
}

/**
 * Get audio duration
 * @param audio Audio data
 * @return Duration in milliseconds, or 0 on failure
 */
int tinyaiAudioGetDurationMs(const TinyAIAudioData *audio)
{
    if (!audio) {
        return 0;
    }
    return audio->durationMs;
}

/**
 * Get audio channels
 * @param audio Audio data
 * @return Number of channels, or 0 on failure
 */
int tinyaiAudioGetChannels(const TinyAIAudioData *audio)
{
    if (!audio) {
        return 0;
    }
    return audio->format.channels;
}

/**
 * Scale audio data to a different duration
 * @param audio Audio data to modify
 * @param scaleFactor Scale factor (1.0 = no change)
 * @return true on success, false on failure
 */
bool tinyaiAudioScale(TinyAIAudioData *audio, float scaleFactor)
{
    if (!audio || !audio->data || scaleFactor <= 0.0f) {
        return false;
    }

    int numSamples = audio->dataSize / sizeof(float);
    int newSamples = (int)(numSamples / scaleFactor);

    /* Allocate new buffer */
    float *newData = (float *)malloc(newSamples * sizeof(float));
    if (!newData) {
        return false;
    }

    /* Resample */
    const float *samples = (const float *)audio->data;
    for (int i = 0; i < newSamples; i++) {
        float srcIndex = i * scaleFactor;
        int   index    = (int)srcIndex;
        float frac     = srcIndex - index;

        if (index + 1 >= numSamples) {
            newData[i] = samples[numSamples - 1];
        }
        else {
            newData[i] = (1.0f - frac) * samples[index] + frac * samples[index + 1];
        }
    }

    /* Update audio struct */
    free(audio->data);
    audio->data       = newData;
    audio->dataSize   = newSamples * sizeof(float);
    audio->durationMs = (int)(audio->durationMs / scaleFactor);

    return true;
}

/**
 * Detect audio file format from file extension
 * @param filePath Path to audio file
 * @return Detected file format, or TINYAI_AUDIO_FORMAT_WAV on failure
 */
TinyAIAudioFileFormat tinyaiAudioDetectFormat(const char *filePath)
{
    if (!filePath) {
        return TINYAI_AUDIO_FORMAT_WAV;
    }

    /* Check file extension */
    const char *extension = strrchr(filePath, '.');
    if (!extension) {
        return TINYAI_AUDIO_FORMAT_WAV;
    }

    /* Match extension */
    if (strcasecmp(extension, ".wav") == 0) {
        return TINYAI_AUDIO_FORMAT_WAV;
    }
    else if (strcasecmp(extension, ".mp3") == 0) {
        return TINYAI_AUDIO_FORMAT_MP3;
    }
    else if (strcasecmp(extension, ".flac") == 0) {
        return TINYAI_AUDIO_FORMAT_FLAC;
    }
    else if (strcasecmp(extension, ".ogg") == 0) {
        return TINYAI_AUDIO_FORMAT_OGG;
    }
    else if (strcasecmp(extension, ".raw") == 0 || strcasecmp(extension, ".pcm") == 0) {
        return TINYAI_AUDIO_FORMAT_RAW;
    }

    /* Default to WAV */
    return TINYAI_AUDIO_FORMAT_WAV;
}

/**
 * Load audio data from file
 * @param filePath Path to audio file
 * @param format Format of audio file (or -1 to detect from extension)
 * @param audio Output structure to receive audio data
 * @return true on success, false on failure
 */
bool tinyaiAudioLoadFile(const char *filePath, TinyAIAudioFileFormat format, TinyAIAudioData *audio)
{
    if (!filePath || !audio) {
        return false;
    }

    /* Detect format if not specified */
    if (format == (TinyAIAudioFileFormat)(-1)) {
        format = tinyaiAudioDetectFormat(filePath);
    }

    /* Load based on format */
    switch (format) {
    case TINYAI_AUDIO_FORMAT_WAV:
        return loadWavFile(filePath, audio);
    case TINYAI_AUDIO_FORMAT_MP3:
    case TINYAI_AUDIO_FORMAT_FLAC:
    case TINYAI_AUDIO_FORMAT_OGG:
    case TINYAI_AUDIO_FORMAT_RAW:
        /* Not implemented yet */
        return false;
    default:
        return false;
    }
}

/**
 * Save audio data to file
 * @param filePath Path to output file
 * @param format Format of output file
 * @param audio Audio data to save
 * @return true on success, false on failure
 */
bool tinyaiAudioSaveFile(const char *filePath, TinyAIAudioFileFormat format,
                         const TinyAIAudioData *audio)
{
    /* TODO: Implement saving audio to different formats */
    /* For now, just indicate not implemented */
    return false;
}

/**
 * Convert audio data between formats
 * @param input Input audio data
 * @param output Output structure to receive converted audio
 * @param targetFormat Target audio format
 * @return true on success, false on failure
 */
bool tinyaiAudioConvertFormat(const TinyAIAudioData *input, TinyAIAudioData *output,
                              const TinyAIAudioFormat *targetFormat)
{
    /* TODO: Implement format conversion */
    /* For now, just make a copy */
    if (!input || !output || !targetFormat || !input->data) {
        return false;
    }

    /* Simple copy */
    output->format     = *targetFormat;
    output->dataSize   = input->dataSize;
    output->durationMs = input->durationMs;
    output->data       = malloc(input->dataSize);
    if (!output->data) {
        return false;
    }
    memcpy(output->data, input->data, input->dataSize);

    return true;
}

/**
 * Apply fade-in to audio
 * @param audio Audio data to modify
 * @param durationMs Duration of fade-in in milliseconds
 * @return true on success, false on failure
 */
bool tinyaiAudioFadeIn(TinyAIAudioData *audio, int durationMs)
{
    if (!audio || !audio->data || durationMs <= 0) {
        return false;
    }

    /* Calculate fade samples */
    int fadeSamples = (audio->format.sampleRate * durationMs) / 1000;
    if (fadeSamples <= 0) {
        return false;
    }

    /* Apply fade-in */
    float *samples    = (float *)audio->data;
    int    numSamples = audio->dataSize / sizeof(float);
    if (fadeSamples > numSamples) {
        fadeSamples = numSamples;
    }

    for (int i = 0; i < fadeSamples; i++) {
        float gain = (float)i / (float)fadeSamples;
        samples[i] *= gain;
    }

    return true;
}

/**
 * Apply fade-out to audio
 * @param audio Audio data to modify
 * @param durationMs Duration of fade-out in milliseconds
 * @return true on success, false on failure
 */
bool tinyaiAudioFadeOut(TinyAIAudioData *audio, int durationMs)
{
    if (!audio || !audio->data || durationMs <= 0) {
        return false;
    }

    /* Calculate fade samples */
    int fadeSamples = (audio->format.sampleRate * durationMs) / 1000;
    if (fadeSamples <= 0) {
        return false;
    }

    /* Apply fade-out */
    float *samples    = (float *)audio->data;
    int    numSamples = audio->dataSize / sizeof(float);
    if (fadeSamples > numSamples) {
        fadeSamples = numSamples;
    }

    int startIndex = numSamples - fadeSamples;
    for (int i = 0; i < fadeSamples; i++) {
        float gain = (float)(fadeSamples - i) / (float)fadeSamples;
        samples[startIndex + i] *= gain;
    }

    return true;
}

/**
 * Mix two audio streams
 * @param audio1 First audio stream
 * @param audio2 Second audio stream
 * @param output Output structure to receive mixed audio
 * @param mixRatio Mixing ratio (0.0 = only audio1, 1.0 = only audio2, 0.5 = equal mix)
 * @return true on success, false on failure
 */
bool tinyaiAudioMix(const TinyAIAudioData *audio1, const TinyAIAudioData *audio2,
                    TinyAIAudioData *output, float mixRatio)
{
    if (!audio1 || !audio2 || !output || !audio1->data || !audio2->data || mixRatio < 0.0f ||
        mixRatio > 1.0f) {
        return false;
    }

    /* Check if formats are compatible */
    if (audio1->format.sampleRate != audio2->format.sampleRate ||
        audio1->format.channels != audio2->format.channels) {
        return false;
    }

    /* Determine output length (the longer of the two) */
    int numSamples1   = audio1->dataSize / sizeof(float);
    int numSamples2   = audio2->dataSize / sizeof(float);
    int outputSamples = (numSamples1 > numSamples2) ? numSamples1 : numSamples2;

    /* Allocate output buffer */
    float *outputData = (float *)malloc(outputSamples * sizeof(float));
    if (!outputData) {
        return false;
    }

    /* Mix samples */
    const float *samples1 = (const float *)audio1->data;
    const float *samples2 = (const float *)audio2->data;
    float        ratio1   = 1.0f - mixRatio;
    float        ratio2   = mixRatio;

    for (int i = 0; i < outputSamples; i++) {
        float sample1 = (i < numSamples1) ? samples1[i] : 0.0f;
        float sample2 = (i < numSamples2) ? samples2[i] : 0.0f;
        outputData[i] = sample1 * ratio1 + sample2 * ratio2;

        /* Clamp to [-1, 1] */
        if (outputData[i] > 1.0f) {
            outputData[i] = 1.0f;
        }
        else if (outputData[i] < -1.0f) {
            outputData[i] = -1.0f;
        }
    }

    /* Set output */
    output->format     = audio1->format;
    output->dataSize   = outputSamples * sizeof(float);
    output->durationMs = (outputSamples * 1000) / audio1->format.sampleRate;
    output->data       = outputData;

    return true;
}

/**
 * Apply band-pass filter to audio
 * @param input Input audio data
 * @param output Output structure to receive filtered audio
 * @param lowFreqHz Low cutoff frequency in Hz
 * @param highFreqHz High cutoff frequency in Hz
 * @param order Filter order (1, 2, 4, 8, etc.)
 * @return true on success, false on failure
 */
bool tinyaiAudioBandpassFilter(const TinyAIAudioData *input, TinyAIAudioData *output,
                               float lowFreqHz, float highFreqHz, int order)
{
    /* TODO: Implement bandpass filter */
    /* For now, just make a copy */
    if (!input || !output || !input->data || lowFreqHz < 0.0f || highFreqHz <= lowFreqHz ||
        order <= 0) {
        return false;
    }

    /* Simple copy */
    output->format     = input->format;
    output->dataSize   = input->dataSize;
    output->durationMs = input->durationMs;
    output->data       = malloc(input->dataSize);
    if (!output->data) {
        return false;
    }
    memcpy(output->data, input->data, input->dataSize);

    return true;
}

/**
 * Apply noise reduction to audio
 * @param input Input audio data
 * @param output Output structure to receive noise-reduced audio
 * @param strengthDb Noise reduction strength in decibels
 * @return true on success, false on failure
 */
bool tinyaiAudioReduceNoise(const TinyAIAudioData *input, TinyAIAudioData *output, float strengthDb)
{
    /* TODO: Implement noise reduction */
    /* For now, just make a copy */
    if (!input || !output || !input->data || strengthDb < 0.0f) {
        return false;
    }

    /* Simple copy */
    output->format     = input->format;
    output->dataSize   = input->dataSize;
    output->durationMs = input->durationMs;
    output->data       = malloc(input->dataSize);
    if (!output->data) {
        return false;
    }
    memcpy(output->data, input->data, input->dataSize);

    return true;
}

/**
 * Create audio data from raw samples
 * @param samples Array of raw audio samples (float)
 * @param numSamples Number of samples
 * @param format Audio format information
 * @param audio Output structure to receive audio data
 * @return true on success, false on failure
 */
bool tinyaiAudioCreateFromSamples(const float *samples, int numSamples,
                                  const TinyAIAudioFormat *format, TinyAIAudioData *audio)
{
    if (!samples || !format || !audio || numSamples <= 0) {
        return false;
    }

    /* Initialize audio structure */
    audio->format     = *format;
    audio->dataSize   = numSamples * sizeof(float);
    audio->durationMs = (numSamples * 1000) / format->sampleRate;
    audio->data       = malloc(audio->dataSize);
    if (!audio->data) {
        return false;
    }

    /* Copy samples */
    memcpy(audio->data, samples, audio->dataSize);

    return true;
}

/**
 * Convert audio samples to float array
 * @param audio Input audio data
 * @param samples Output array of float samples
 * @param maxSamples Maximum number of samples to convert
 * @param numSamples Output number of samples converted
 * @return true on success, false on failure
 */
bool tinyaiAudioToFloatSamples(const TinyAIAudioData *audio, float *samples, int maxSamples,
                               int *numSamples)
{
    if (!audio || !audio->data || !samples || !numSamples || maxSamples <= 0) {
        return false;
    }

    /* Calculate number of samples to convert */
    int availableSamples = audio->dataSize / sizeof(float);
    *numSamples          = (availableSamples > maxSamples) ? maxSamples : availableSamples;

    /* Copy samples */
    memcpy(samples, audio->data, *numSamples * sizeof(float));

    return true;
}

/**
 * Calculate zero-crossing rate of audio
 * @param audio Audio data to analyze
 * @param windowMs Window size in milliseconds (0 for entire audio)
 * @param rates Output array of zero-crossing rates
 * @param maxRates Maximum number of rates to calculate
 * @param numRates Output number of rates calculated
 * @return true on success, false on failure
 */
bool tinyaiAudioCalculateZeroCrossingRate(const TinyAIAudioData *audio, int windowMs, float *rates,
                                          int maxRates, int *numRates)
{
    if (!audio || !audio->data || !rates || !numRates || maxRates <= 0) {
        return false;
    }

    /* Calculate window size */
    int samplesPerWindow = 0;
    if (windowMs <= 0) {
        /* Use entire audio */
        samplesPerWindow = audio->dataSize / sizeof(float);
        *numRates        = 1;
    }
    else {
        /* Calculate window size and number of windows */
        samplesPerWindow = (audio->format.sampleRate * windowMs) / 1000;
        if (samplesPerWindow <= 0) {
            return false;
        }
        int totalSamples = audio->dataSize / sizeof(float);
        int totalWindows = (totalSamples + samplesPerWindow - 1) / samplesPerWindow;
        *numRates        = (totalWindows > maxRates) ? maxRates : totalWindows;
    }

    /* Calculate zero-crossing rate for each window */
    const float *samples      = (const float *)audio->data;
    int          totalSamples = audio->dataSize / sizeof(float);

    for (int i = 0; i < *numRates; i++) {
        int startSample = i * samplesPerWindow;
        int endSample   = startSample + samplesPerWindow;
        if (endSample > totalSamples) {
            endSample = totalSamples;
        }

        /* Count zero crossings */
        int crossings = 0;
        for (int j = startSample + 1; j < endSample; j++) {
            if ((samples[j - 1] >= 0.0f && samples[j] < 0.0f) ||
                (samples[j - 1] < 0.0f && samples[j] >= 0.0f)) {
                crossings++;
            }
        }

        /* Calculate rate */
        int numSamplesInWindow = endSample - startSample;
        rates[i]               = (float)crossings / (float)numSamplesInWindow;
    }

    return true;
}

/**
 * Calculate peak level of audio
 * @param audio Audio data to analyze
 * @param windowMs Window size in milliseconds (0 for entire audio)
 * @param peakLevels Output array of peak levels
 * @param maxLevels Maximum number of levels to calculate
 * @param numLevels Output number of levels calculated
 * @return true on success, false on failure
 */
bool tinyaiAudioCalculatePeakLevel(const TinyAIAudioData *audio, int windowMs, float *peakLevels,
                                   int maxLevels, int *numLevels)
{
    if (!audio || !audio->data || !peakLevels || !numLevels || maxLevels <= 0) {
        return false;
    }

    /* Calculate window size */
    int samplesPerWindow = 0;
    if (windowMs <= 0) {
        /* Use entire audio */
        samplesPerWindow = audio->dataSize / sizeof(float);
        *numLevels       = 1;
    }
    else {
        /* Calculate window size and number of windows */
        samplesPerWindow = (audio->format.sampleRate * windowMs) / 1000;
        if (samplesPerWindow <= 0) {
            return false;
        }
        int totalSamples = audio->dataSize / sizeof(float);
        int totalWindows = (totalSamples + samplesPerWindow - 1) / samplesPerWindow;
        *numLevels       = (totalWindows > maxLevels) ? maxLevels : totalWindows;
    }

    /* Calculate peak level for each window */
    const float *samples      = (const float *)audio->data;
    int          totalSamples = audio->dataSize / sizeof(float);

    for (int i = 0; i < *numLevels; i++) {
        int startSample = i * samplesPerWindow;
        int endSample   = startSample + samplesPerWindow;
        if (endSample > totalSamples) {
            endSample = totalSamples;
        }

        /* Find peak */
        float peak = 0.0f;
        for (int j = startSample; j < endSample; j++) {
            float abs = fabsf(samples[j]);
            if (abs > peak) {
                peak = abs;
            }
        }

        peakLevels[i] = peak;
    }

    return true;
}

/**
 * Calculate root mean square (RMS) level of audio
 * @param audio Audio data to analyze
 * @param windowMs Window size in milliseconds (0 for entire audio)
 * @param rmsLevels Output array of RMS levels
 * @param maxLevels Maximum number of levels to calculate
 * @param numLevels Output number of levels calculated
 * @return true on success, false on failure
 */
bool tinyaiAudioCalculateRMSLevel(const TinyAIAudioData *audio, int windowMs, float *rmsLevels,
                                  int maxLevels, int *numLevels)
{
    if (!audio || !audio->data || !rmsLevels || !numLevels || maxLevels <= 0) {
        return false;
    }

    /* Calculate window size */
    int samplesPerWindow = 0;
    if (windowMs <= 0) {
        /* Use entire audio */
        samplesPerWindow = audio->dataSize / sizeof(float);
        *numLevels       = 1;
    }
    else {
        /* Calculate window size and number of windows */
        samplesPerWindow = (audio->format.sampleRate * windowMs) / 1000;
        if (samplesPerWindow <= 0) {
            return false;
        }
        int totalSamples = audio->dataSize / sizeof(float);
        int totalWindows = (totalSamples + samplesPerWindow - 1) / samplesPerWindow;
        *numLevels       = (totalWindows > maxLevels) ? maxLevels : totalWindows;
    }

    /* Calculate RMS level for each window */
    const float *samples      = (const float *)audio->data;
    int          totalSamples = audio->dataSize / sizeof(float);

    for (int i = 0; i < *numLevels; i++) {
        int startSample = i * samplesPerWindow;
        int endSample   = startSample + samplesPerWindow;
        if (endSample > totalSamples) {
            endSample = totalSamples;
        }

        /* Calculate RMS */
        float sumSquares = 0.0f;
        for (int j = startSample; j < endSample; j++) {
            sumSquares += samples[j] * samples[j];
        }
        int numSamplesInWindow = endSample - startSample;
        rmsLevels[i]           = sqrtf(sumSquares / numSamplesInWindow);
    }

    return true;
}

/**
 * Detect voice activity in audio
 * @param audio Audio data to analyze
 * @param windowMs Window size in milliseconds
 * @param activity Output array of voice activity (1.0 = voice, 0.0 = silence)
 * @param maxFrames Maximum number of frames to analyze
 * @param numFrames Output number of frames analyzed
 * @param sensitivity Sensitivity (0.0-1.0, higher = more sensitive)
 * @return true on success, false on failure
 */
bool tinyaiAudioDetectVoiceActivity(const TinyAIAudioData *audio, int windowMs, float *activity,
                                    int maxFrames, int *numFrames, float sensitivity)
{
    if (!audio || !audio->data || !activity || !numFrames || maxFrames <= 0 || windowMs <= 0 ||
        sensitivity < 0.0f || sensitivity > 1.0f) {
        return false;
    }

    /* Calculate window size */
    int samplesPerWindow = (audio->format.sampleRate * windowMs) / 1000;
    if (samplesPerWindow <= 0) {
        return false;
    }

    /* Calculate number of frames */
    int totalSamples = audio->dataSize / sizeof(float);
    int totalFrames  = (totalSamples + samplesPerWindow - 1) / samplesPerWindow;
    *numFrames       = (totalFrames > maxFrames) ? maxFrames : totalFrames;

    /* Calculate threshold based on sensitivity */
    float threshold = 0.03f * (1.0f - sensitivity);

    /* Compute energy for each frame */
    const float *samples = (const float *)audio->data;
    for (int i = 0; i < *numFrames; i++) {
        int startSample = i * samplesPerWindow;
        int endSample   = startSample + samplesPerWindow;
        if (endSample > totalSamples) {
            endSample = totalSamples;
        }

        /* Calculate energy */
        float energy = 0.0f;
        for (int j = startSample; j < endSample; j++) {
            energy += samples[j] * samples[j];
        }
        int numSamplesInWindow = endSample - startSample;
        energy /= numSamplesInWindow;

        /* Classify as voice or silence */
        activity[i] = (energy > threshold) ? 1.0f : 0.0f;
    }

    return true;
}
