/**
 * @file audio_features.c
 * @brief Implementation of audio feature extraction for TinyAI
 */

#include "audio_features.h"
#include "../../core/memory.h"
#include "../../utils/simd_ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Constants for feature extraction */
#define PI 3.14159265358979323846f
#define PI2 6.28318530717958647692f

/* Default settings for MFCC extraction */
#define DEFAULT_FFT_SIZE 512
#define DEFAULT_NUM_FILTERS 26
#define DEFAULT_NUM_COEFFICIENTS 13
#define DEFAULT_WINDOW TINYAI_WINDOW_HAMMING
#define DEFAULT_FMIN 0.0f
#define DEFAULT_FMAX 8000.0f
#define DEFAULT_PREEMPHASIS 0.97f

/**
 * Initialize default advanced options for feature extraction
 * @param options Options structure to initialize
 * @return true on success, false on failure
 */
bool tinyaiAudioFeaturesInitAdvancedOptions(TinyAIAudioFeaturesAdvancedOptions *options)
{
    if (!options) {
        return false;
    }

    options->windowType      = DEFAULT_WINDOW;
    options->windowParam     = 0.0f; /* Not used for Hamming/Hann windows */
    options->removeDC        = true;
    options->energyNormalize = true;
    options->fftSize         = DEFAULT_FFT_SIZE;
    options->fMin            = DEFAULT_FMIN;
    options->fMax            = DEFAULT_FMAX;
    options->usePower        = true;
    options->lifterCoeff     = 22.0f; /* Standard liftering coefficient */
    options->dithering       = false;
    options->ditheringCoeff  = 0.0f;

    return true;
}

/**
 * Apply window function to a frame of audio samples
 * @param samples Array of audio samples
 * @param numSamples Number of samples
 * @param windowType Type of window function to apply
 * @param param Parameter for parameterized windows (if applicable)
 * @param output Output buffer (can be same as input)
 * @return true on success, false on failure
 */
bool tinyaiAudioApplyWindow(const float *samples, int numSamples, TinyAIWindowType windowType,
                            float param, float *output)
{
    if (!samples || !output || numSamples <= 0) {
        return false;
    }

    /* Apply window function */
    switch (windowType) {
    case TINYAI_WINDOW_HAMMING:
        /* Hamming window: w(n) = 0.54 - 0.46 * cos(2*PI*n/(N-1)) */
        for (int i = 0; i < numSamples; i++) {
            float n      = (float)i;
            float N      = (float)numSamples;
            float window = 0.54f - 0.46f * cosf(PI2 * n / (N - 1));
            output[i]    = samples[i] * window;
        }
        break;

    case TINYAI_WINDOW_HANN:
        /* Hann window: w(n) = 0.5 * (1 - cos(2*PI*n/(N-1))) */
        for (int i = 0; i < numSamples; i++) {
            float n      = (float)i;
            float N      = (float)numSamples;
            float window = 0.5f * (1.0f - cosf(PI2 * n / (N - 1)));
            output[i]    = samples[i] * window;
        }
        break;

    case TINYAI_WINDOW_BLACKMAN:
        /* Blackman window: w(n) = 0.42 - 0.5*cos(2*PI*n/(N-1)) + 0.08*cos(4*PI*n/(N-1)) */
        for (int i = 0; i < numSamples; i++) {
            float n = (float)i;
            float N = (float)numSamples;
            float window =
                0.42f - 0.5f * cosf(PI2 * n / (N - 1)) + 0.08f * cosf(2.0f * PI2 * n / (N - 1));
            output[i] = samples[i] * window;
        }
        break;

    case TINYAI_WINDOW_RECTANGULAR:
        /* Rectangular window: no windowing */
        if (samples != output) {
            memcpy(output, samples, numSamples * sizeof(float));
        }
        break;

    default:
        return false;
    }

    return true;
}

/**
 * Compute bit reversal for FFT
 * @param i Index to reverse
 * @param bits Number of bits
 * @return Bit-reversed index
 */
static int bitReverse(int i, int bits)
{
    int j = 0;
    for (int b = 0; b < bits; b++) {
        j = (j << 1) | (i & 1);
        i >>= 1;
    }
    return j;
}

/**
 * Compute FFT using Cooley-Tukey algorithm
 * @param real Real part of input/output
 * @param imag Imaginary part of input/output
 * @param size Size of FFT (must be power of 2)
 */
static void fft(float *real, float *imag, int size)
{
    /* Check if size is power of 2 */
    int bits = 0;
    int temp = size;
    while (temp > 1) {
        temp >>= 1;
        bits++;
    }

    /* Bit reversal */
    for (int i = 0; i < size; i++) {
        int j = bitReverse(i, bits);
        if (j > i) {
            float temp_real = real[i];
            float temp_imag = imag[i];
            real[i]         = real[j];
            imag[i]         = imag[j];
            real[j]         = temp_real;
            imag[j]         = temp_imag;
        }
    }

    /* FFT computation */
    for (int stage = 1; stage <= bits; stage++) {
        int   m     = 1 << stage;
        int   m2    = m >> 1;
        float wr    = 1.0f;
        float wi    = 0.0f;
        float theta = -PI / (float)m2;
        float wpr   = cosf(theta);
        float wpi   = sinf(theta);

        for (int j = 0; j < m2; j++) {
            for (int i = j; i < size; i += m) {
                int   k  = i + m2;
                float tr = wr * real[k] - wi * imag[k];
                float ti = wr * imag[k] + wi * real[k];
                real[k]  = real[i] - tr;
                imag[k]  = imag[i] - ti;
                real[i] += tr;
                imag[i] += ti;
            }
            float wtemp = wr;
            wr          = wr * wpr - wi * wpi;
            wi          = wi * wpr + wtemp * wpi;
        }
    }
}

/**
 * Compute FFT of a frame of audio samples
 * @param samples Array of audio samples (windowed)
 * @param numSamples Number of samples
 * @param fftSize Size of FFT (power of 2, >= numSamples)
 * @param real Output real part of FFT
 * @param imag Output imaginary part of FFT
 * @return true on success, false on failure
 */
bool tinyaiAudioComputeFFT(const float *samples, int numSamples, int fftSize, float *real,
                           float *imag)
{
    if (!samples || !real || !imag || numSamples <= 0 || fftSize <= 0 || fftSize < numSamples) {
        return false;
    }

    /* Check if fftSize is power of 2 */
    int temp = fftSize;
    while (temp > 1) {
        if (temp & 1) {
            return false; /* Not a power of 2 */
        }
        temp >>= 1;
    }

    /* Initialize real and imaginary parts */
    for (int i = 0; i < fftSize; i++) {
        if (i < numSamples) {
            real[i] = samples[i];
        }
        else {
            real[i] = 0.0f; /* Zero padding */
        }
        imag[i] = 0.0f;
    }

    /* Compute FFT */
    fft(real, imag, fftSize);

    return true;
}

/**
 * Compute power spectrum from FFT results
 * @param real Real part of FFT
 * @param imag Imaginary part of FFT
 * @param size Size of FFT
 * @param output Output power/magnitude spectrum
 * @param usePower Whether to compute power (otherwise magnitude)
 * @return true on success, false on failure
 */
bool tinyaiAudioComputeSpectrum(const float *real, const float *imag, int size, float *output,
                                bool usePower)
{
    if (!real || !imag || !output || size <= 0) {
        return false;
    }

    /* Compute power/magnitude spectrum */
    int halfSize = size / 2 + 1;
    for (int i = 0; i < halfSize; i++) {
        if (usePower) {
            /* Power spectrum */
            output[i] = real[i] * real[i] + imag[i] * imag[i];
        }
        else {
            /* Magnitude spectrum */
            output[i] = sqrtf(real[i] * real[i] + imag[i] * imag[i]);
        }
    }

    return true;
}

/**
 * Convert frequency to mel scale
 * @param f Frequency in Hz
 * @return Mel scale value
 */
static float freqToMel(float f) { return 1127.0f * logf(1.0f + f / 700.0f); }

/**
 * Convert mel scale to frequency
 * @param m Mel scale value
 * @return Frequency in Hz
 */
static float melToFreq(float m) { return 700.0f * (expf(m / 1127.0f) - 1.0f); }

/**
 * Create mel filter bank
 * @param numFilters Number of mel filters
 * @param fftSize Size of FFT
 * @param sampleRate Sample rate in Hz
 * @param fMin Minimum frequency in Hz
 * @param fMax Maximum frequency in Hz
 * @param filterBank Output filter bank coefficients (numFilters x (fftSize/2+1))
 * @return true on success, false on failure
 */
bool tinyaiAudioCreateMelFilterBank(int numFilters, int fftSize, int sampleRate, float fMin,
                                    float fMax, float *filterBank)
{
    if (!filterBank || numFilters <= 0 || fftSize <= 0 || sampleRate <= 0 || fMin >= fMax) {
        return false;
    }

    /* Convert frequencies to mel scale */
    float melMin = freqToMel(fMin);
    float melMax = freqToMel(fMax);

    /* Create mel-spaced points */
    float *melPoints = (float *)malloc((numFilters + 2) * sizeof(float));
    if (!melPoints) {
        return false;
    }

    /* Initialize mel points */
    for (int i = 0; i < numFilters + 2; i++) {
        melPoints[i] = melMin + (melMax - melMin) * i / (numFilters + 1);
    }

    /* Convert mel points to frequency */
    float *freqPoints = (float *)malloc((numFilters + 2) * sizeof(float));
    if (!freqPoints) {
        free(melPoints);
        return false;
    }

    for (int i = 0; i < numFilters + 2; i++) {
        freqPoints[i] = melToFreq(melPoints[i]);
    }

    /* Create filter bank */
    int   halfSize    = fftSize / 2 + 1;
    float fftBinWidth = (float)sampleRate / (float)fftSize;

    /* Initialize filter bank to zeros */
    memset(filterBank, 0, numFilters * halfSize * sizeof(float));

    /* Compute filter bank coefficients */
    for (int i = 0; i < numFilters; i++) {
        /* Convert frequencies to FFT bin indices */
        float f_m_minus = freqPoints[i];
        float f_m       = freqPoints[i + 1];
        float f_m_plus  = freqPoints[i + 2];

        int bin_m_minus = (int)floorf(f_m_minus / fftBinWidth);
        int bin_m       = (int)floorf(f_m / fftBinWidth);
        int bin_m_plus  = (int)ceilf(f_m_plus / fftBinWidth);

        /* Compute filter coefficients */
        for (int j = bin_m_minus; j < bin_m; j++) {
            if (j >= 0 && j < halfSize) {
                float freq                   = j * fftBinWidth;
                filterBank[i * halfSize + j] = (freq - f_m_minus) / (f_m - f_m_minus);
            }
        }

        for (int j = bin_m; j < bin_m_plus; j++) {
            if (j >= 0 && j < halfSize) {
                float freq                   = j * fftBinWidth;
                filterBank[i * halfSize + j] = (f_m_plus - freq) / (f_m_plus - f_m);
            }
        }
    }

    /* Free temporary arrays */
    free(melPoints);
    free(freqPoints);

    return true;
}

/**
 * Apply mel filter bank to spectrum
 * @param spectrum Input power/magnitude spectrum
 * @param spectrumSize Size of spectrum (fftSize/2+1)
 * @param filterBank Mel filter bank
 * @param numFilters Number of mel filters
 * @param melEnergies Output mel filter energies
 * @return true on success, false on failure
 */
bool tinyaiAudioApplyMelFilterBank(const float *spectrum, int spectrumSize, const float *filterBank,
                                   int numFilters, float *melEnergies)
{
    if (!spectrum || !filterBank || !melEnergies || spectrumSize <= 0 || numFilters <= 0) {
        return false;
    }

    /* Initialize mel energies to zeros */
    memset(melEnergies, 0, numFilters * sizeof(float));

    /* Apply filter bank */
    for (int i = 0; i < numFilters; i++) {
        /* Compute weighted sum */
        for (int j = 0; j < spectrumSize; j++) {
            melEnergies[i] += spectrum[j] * filterBank[i * spectrumSize + j];
        }

        /* Avoid negative energies (due to floating-point errors) */
        if (melEnergies[i] < 1e-10f) {
            melEnergies[i] = 1e-10f;
        }
    }

    return true;
}

/**
 * Apply liftering to MFCCs
 * @param coefficients MFCC coefficients
 * @param numCoefficients Number of coefficients
 * @param lifterCoeff Liftering coefficient
 */
static void applyCepstralLifter(float *coefficients, int numCoefficients, float lifterCoeff)
{
    if (lifterCoeff <= 0.0f) {
        return; /* No liftering */
    }

    for (int i = 0; i < numCoefficients; i++) {
        float lifter = 1.0f + (lifterCoeff / 2.0f) * sinf(PI * i / lifterCoeff);
        coefficients[i] *= lifter;
    }
}

/**
 * Compute DCT-II for MFCC calculation
 * @param input Input array
 * @param output Output array
 * @param inputSize Input size
 * @param outputSize Output size
 */
static void dct(const float *input, float *output, int inputSize, int outputSize)
{
    float inputSizeInv = 1.0f / (float)inputSize;

    for (int k = 0; k < outputSize; k++) {
        float sum = 0.0f;
        for (int n = 0; n < inputSize; n++) {
            sum += input[n] * cosf(PI * k * (2 * n + 1) * inputSizeInv);
        }
        output[k] = sum * 2.0f;
    }
}

/**
 * Compute MFCCs from mel energies
 * @param melEnergies Input mel filter energies
 * @param numFilters Number of mel filters
 * @param numCoefficients Number of cepstral coefficients to compute
 * @param coefficients Output cepstral coefficients
 * @param lifterCoeff Liftering coefficient (0 to disable)
 * @return true on success, false on failure
 */
bool tinyaiAudioComputeMFCC(const float *melEnergies, int numFilters, int numCoefficients,
                            float *coefficients, float lifterCoeff)
{
    if (!melEnergies || !coefficients || numFilters <= 0 || numCoefficients <= 0 ||
        numCoefficients > numFilters) {
        return false;
    }

    /* Apply log to mel energies */
    float *logMelEnergies = (float *)malloc(numFilters * sizeof(float));
    if (!logMelEnergies) {
        return false;
    }

    for (int i = 0; i < numFilters; i++) {
        logMelEnergies[i] = logf(melEnergies[i]);
    }

    /* Apply DCT-II to get MFCCs */
    dct(logMelEnergies, coefficients, numFilters, numCoefficients);

    /* Apply liftering */
    applyCepstralLifter(coefficients, numCoefficients, lifterCoeff);

    /* Free temporary array */
    free(logMelEnergies);

    return true;
}

/**
 * Apply pre-emphasis to audio samples
 * @param input Input samples
 * @param output Output samples
 * @param numSamples Number of samples
 * @param coeff Pre-emphasis coefficient
 */
static void applyPreEmphasis(const float *input, float *output, int numSamples, float coeff)
{
    if (coeff <= 0.0f) {
        /* No pre-emphasis */
        if (input != output) {
            memcpy(output, input, numSamples * sizeof(float));
        }
        return;
    }

    /* Apply pre-emphasis: y[n] = x[n] - coeff * x[n-1] */
    output[0] = input[0];
    for (int i = 1; i < numSamples; i++) {
        output[i] = input[i] - coeff * input[i - 1];
    }
}

/**
 * Remove DC offset from audio samples
 * @param samples Audio samples
 * @param numSamples Number of samples
 */
static void removeDC(float *samples, int numSamples)
{
    /* Compute mean */
    float mean = 0.0f;
    for (int i = 0; i < numSamples; i++) {
        mean += samples[i];
    }
    mean /= numSamples;

    /* Subtract mean */
    for (int i = 0; i < numSamples; i++) {
        samples[i] -= mean;
    }
}

/**
 * Extract MFCC features from audio frame
 * @param frame Input audio frame
 * @param frameLength Frame length in samples
 * @param config Feature extraction configuration
 * @param advancedOptions Advanced options (NULL for defaults)
 * @param coefficients Output MFCC coefficients
 * @return true on success, false on failure
 */
static bool extractMFCCFrame(const float *frame, int frameLength,
                             const TinyAIAudioFeaturesConfig          *config,
                             const TinyAIAudioFeaturesAdvancedOptions *advancedOptions,
                             float                                    *coefficients)
{
    if (!frame || !config || !coefficients || frameLength <= 0) {
        return false;
    }

    /* Use default options if not provided */
    TinyAIAudioFeaturesAdvancedOptions defaultOptions;
    if (!advancedOptions) {
        tinyaiAudioFeaturesInitAdvancedOptions(&defaultOptions);
        advancedOptions = &defaultOptions;
    }

    /* Allocate temporary buffers */
    float *preemphasizedFrame = (float *)malloc(frameLength * sizeof(float));
    if (!preemphasizedFrame) {
        return false;
    }

    int    fftSize       = advancedOptions->fftSize;
    float *windowedFrame = (float *)malloc(frameLength * sizeof(float));
    float *real          = (float *)malloc(fftSize * sizeof(float));
    float *imag          = (float *)malloc(fftSize * sizeof(float));
    float *spectrum      = (float *)malloc((fftSize / 2 + 1) * sizeof(float));
    float *filterBank    = (float *)malloc(config->numFilters * (fftSize / 2 + 1) * sizeof(float));
    float *melEnergies   = (float *)malloc(config->numFilters * sizeof(float));

    if (!windowedFrame || !real || !imag || !spectrum || !filterBank || !melEnergies) {
        /* Free allocated buffers */
        free(preemphasizedFrame);
        if (windowedFrame)
            free(windowedFrame);
        if (real)
            free(real);
        if (imag)
            free(imag);
        if (spectrum)
            free(spectrum);
        if (filterBank)
            free(filterBank);
        if (melEnergies)
            free(melEnergies);
        return false;
    }

    /* Apply pre-emphasis */
    applyPreEmphasis(frame, preemphasizedFrame, frameLength, config->preEmphasis);

    /* Remove DC offset if requested */
    if (advancedOptions->removeDC) {
        removeDC(preemphasizedFrame, frameLength);
    }

    /* Apply window function */
    tinyaiAudioApplyWindow(preemphasizedFrame, frameLength, advancedOptions->windowType,
                           advancedOptions->windowParam, windowedFrame);

    /* Compute FFT */
    tinyaiAudioComputeFFT(windowedFrame, frameLength, fftSize, real, imag);

    /* Compute power/magnitude spectrum */
    tinyaiAudioComputeSpectrum(real, imag, fftSize, spectrum, advancedOptions->usePower);

    /* Create mel filter bank - use default 16kHz sample rate */
    int sampleRate = 16000; /* Standard sample rate for speech processing */
    tinyaiAudioCreateMelFilterBank(config->numFilters, fftSize, sampleRate, advancedOptions->fMin,
                                   advancedOptions->fMax, filterBank);

    /* Apply mel filter bank */
    tinyaiAudioApplyMelFilterBank(spectrum, fftSize / 2 + 1, filterBank, config->numFilters,
                                  melEnergies);

    /* Compute MFCCs */
    tinyaiAudioComputeMFCC(melEnergies, config->numFilters, config->numCoefficients, coefficients,
                           advancedOptions->lifterCoeff);

    /* Free temporary buffers */
    free(preemphasizedFrame);
    free(windowedFrame);
    free(real);
    free(imag);
    free(spectrum);
    free(filterBank);
    free(melEnergies);

    return true;
}

/**
 * Extract features from audio data
 * @param audio The audio data to process
 * @param config Feature extraction configuration
 * @param advancedOptions Advanced options (NULL for defaults)
 * @param featureType Type of features to extract
 * @param features Output structure to receive extracted features
 * @return true on success, false on failure
 */
static bool extractAudioFeatures(const TinyAIAudioData                    *audio,
                                 const TinyAIAudioFeaturesConfig          *config,
                                 const TinyAIAudioFeaturesAdvancedOptions *advancedOptions,
                                 TinyAIAudioFeaturesType featureType, TinyAIAudioFeatures *features)
{
    if (!audio || !config || !features || !audio->data) {
        return false;
    }

    /* Use default options if not provided */
    TinyAIAudioFeaturesAdvancedOptions defaultOptions;
    if (!advancedOptions) {
        tinyaiAudioFeaturesInitAdvancedOptions(&defaultOptions);
        advancedOptions = &defaultOptions;
    }

    /* Convert audio data to float samples */
    int    numSamples = (int)(audio->dataSize / (audio->format.bitsPerSample / 8));
    float *samples    = (float *)malloc(numSamples * sizeof(float));
    if (!samples) {
        return false;
    }

    /* TODO: Convert audio samples based on format */
    /* For now, assume the data is already in float format */
    memcpy(samples, audio->data, numSamples * sizeof(float));

    /* Calculate number of frames */
    int frameLength = config->frameLength;
    int frameShift  = config->frameShift;
    int numFrames   = (numSamples - frameLength) / frameShift + 1;
    if (numFrames <= 0) {
        free(samples);
        return false;
    }

    /* Determine feature dimensions */
    int featuresPerFrame = 0;
    switch (featureType) {
    case TINYAI_AUDIO_FEATURES_MFCC:
        featuresPerFrame = config->numCoefficients;
        break;
    case TINYAI_AUDIO_FEATURES_MEL:
        featuresPerFrame = config->numFilters;
        break;
    case TINYAI_AUDIO_FEATURES_SPECTROGRAM:
        featuresPerFrame = advancedOptions->fftSize / 2 + 1;
        break;
    case TINYAI_AUDIO_FEATURES_RAW:
        featuresPerFrame = frameLength;
        break;
    default:
        free(samples);
        return false;
    }

    /* Add delta features if requested */
    if (config->includeDelta) {
        featuresPerFrame *= 2;
    }
    if (config->includeDeltaDelta) {
        featuresPerFrame += config->numCoefficients;
    }

    /* Allocate features */
    features->data = (float *)malloc(numFrames * featuresPerFrame * sizeof(float));
    if (!features->data) {
        free(samples);
        return false;
    }

    features->dataSize    = numFrames * featuresPerFrame * sizeof(float);
    features->numFrames   = numFrames;
    features->numFeatures = featuresPerFrame;
    features->type        = featureType;

    /* Extract features for each frame */
    for (int i = 0; i < numFrames; i++) {
        int    offset        = i * frameShift;
        float *frame         = samples + offset;
        float *featureVector = features->data + i * featuresPerFrame;

        switch (featureType) {
        case TINYAI_AUDIO_FEATURES_MFCC:
            if (!extractMFCCFrame(frame, frameLength, config, advancedOptions, featureVector)) {
                free(samples);
                free(features->data);
                features->data = NULL;
                return false;
            }
            break;

        case TINYAI_AUDIO_FEATURES_MEL:
            /* TODO: Implement Mel spectrogram extraction */
            break;

        case TINYAI_AUDIO_FEATURES_SPECTROGRAM:
            /* TODO: Implement spectrogram extraction */
            break;

        case TINYAI_AUDIO_FEATURES_RAW:
            /* Just copy the raw samples */
            memcpy(featureVector, frame, frameLength * sizeof(float));
            break;

        default:
            free(samples);
            free(features->data);
            features->data = NULL;
            return false;
        }
    }

    /* Compute delta features if requested */
    if (config->includeDelta) {
        /* TODO: Implement delta feature computation */
    }

    /* Free temporary buffers */
    free(samples);

    return true;
}

/**
 * Extract MFCC features from raw audio
 * @param audio Input audio data
 * @param config Feature extraction configuration
 * @param advancedOptions Advanced options (NULL for defaults)
 * @param features Output structure to receive extracted features
 * @return true on success, false on failure
 */
bool tinyaiAudioExtractMFCC(const TinyAIAudioData *audio, const TinyAIAudioFeaturesConfig *config,
                            const TinyAIAudioFeaturesAdvancedOptions *advancedOptions,
                            TinyAIAudioFeatures                      *features)
{
    return extractAudioFeatures(audio, config, advancedOptions, TINYAI_AUDIO_FEATURES_MFCC,
                                features);
}

/**
 * Extract Mel spectrogram features from raw audio
 * @param audio Input audio data
 * @param config Feature extraction configuration
 * @param advancedOptions Advanced options (NULL for defaults)
 * @param features Output structure to receive extracted features
 * @return true on success, false on failure
 */
bool tinyaiAudioExtractMelSpectrogram(const TinyAIAudioData                    *audio,
                                      const TinyAIAudioFeaturesConfig          *config,
                                      const TinyAIAudioFeaturesAdvancedOptions *advancedOptions,
                                      TinyAIAudioFeatures                      *features)
{
    return extractAudioFeatures(audio, config, advancedOptions, TINYAI_AUDIO_FEATURES_MEL,
                                features);
}

/**
 * Extract spectrogram features from raw audio
 * @param audio Input audio data
 * @param config Feature extraction configuration
 * @param advancedOptions Advanced options (NULL for defaults)
 * @param features Output structure to receive extracted features
 * @return true on success, false on failure
 */
bool tinyaiAudioExtractSpectrogram(const TinyAIAudioData                    *audio,
                                   const TinyAIAudioFeaturesConfig          *config,
                                   const TinyAIAudioFeaturesAdvancedOptions *advancedOptions,
                                   TinyAIAudioFeatures                      *features)
{
    return extractAudioFeatures(audio, config, advancedOptions, TINYAI_AUDIO_FEATURES_SPECTROGRAM,
                                features);
}
