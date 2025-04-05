/**
 * @file audio_features.h
 * @brief Audio feature extraction for TinyAI
 *
 * This header defines functions for extracting audio features such as
 * MFCCs, Mel spectrograms, and regular spectrograms from raw audio data.
 * These features are commonly used in audio machine learning models.
 */

#ifndef TINYAI_AUDIO_FEATURES_H
#define TINYAI_AUDIO_FEATURES_H

#include "audio_model.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Window function type for frame analysis
 */
typedef enum {
    TINYAI_WINDOW_HANN,       /* Hann window (cosine-sum window) */
    TINYAI_WINDOW_HAMMING,    /* Hamming window (similar to Hann but doesn't reach zero) */
    TINYAI_WINDOW_BLACKMAN,   /* Blackman window (three-term cosine-sum window) */
    TINYAI_WINDOW_RECTANGULAR /* Rectangular window (no windowing) */
} TinyAIWindowType;

/**
 * Advanced audio feature extraction options
 */
typedef struct {
    TinyAIWindowType windowType;      /* Window function for frame analysis */
    float            windowParam;     /* Parameter for parameterized windows (if applicable) */
    bool             removeDC;        /* Whether to remove DC offset */
    bool             energyNormalize; /* Whether to normalize energy */
    int              fftSize;         /* Size of FFT (power of 2, >= frameLength) */
    float            fMin;            /* Minimum frequency for mel filters (Hz) */
    float            fMax;            /* Maximum frequency for mel filters (Hz) */
    bool             usePower;        /* Whether to use power spectrum (otherwise magnitude) */
    float            lifterCoeff;     /* Liftering coefficient for MFCCs (0 to disable) */
    bool             dithering;       /* Whether to apply dithering */
    float            ditheringCoeff;  /* Coefficient for dithering */
} TinyAIAudioFeaturesAdvancedOptions;

/**
 * Initialize default advanced options for feature extraction
 * @param options Options structure to initialize
 * @return true on success, false on failure
 */
bool tinyaiAudioFeaturesInitAdvancedOptions(TinyAIAudioFeaturesAdvancedOptions *options);

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
                            float param, float *output);

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
                           float *imag);

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
                                bool usePower);

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
                                    float fMax, float *filterBank);

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
                                   int numFilters, float *melEnergies);

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
                            float *coefficients, float lifterCoeff);

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
                            TinyAIAudioFeatures                      *features);

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
                                      TinyAIAudioFeatures                      *features);

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
                                   TinyAIAudioFeatures                      *features);

/**
 * Compute delta features from base features
 * @param features Base features
 * @param numFrames Number of frames
 * @param numFeatures Number of features per frame
 * @param delta Output delta features
 * @param windowSize Window size for delta computation (typically 2)
 * @return true on success, false on failure
 */
bool tinyaiAudioComputeDelta(const float *features, int numFrames, int numFeatures, float *delta,
                             int windowSize);

/**
 * Compute histogram of onset strength (for onset detection)
 * @param features Input audio features (MFCC, Mel, or Spectrogram)
 * @param onsetStrength Output onset strength (numFrames)
 * @return true on success, false on failure
 */
bool tinyaiAudioComputeOnsetStrength(const TinyAIAudioFeatures *features, float *onsetStrength);

/**
 * Detect peaks in onset strength (for beat tracking)
 * @param onsetStrength Input onset strength
 * @param numFrames Number of frames
 * @param peaks Output array of peak locations (indexes)
 * @param maxPeaks Maximum number of peaks to detect
 * @param numPeaks Output number of detected peaks
 * @param threshold Minimum threshold for peak detection
 * @return true on success, false on failure
 */
bool tinyaiAudioDetectPeaks(const float *onsetStrength, int numFrames, int *peaks, int maxPeaks,
                            int *numPeaks, float threshold);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_AUDIO_FEATURES_H */
