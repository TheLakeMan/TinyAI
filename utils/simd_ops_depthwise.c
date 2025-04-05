/**
 * @file simd_ops_depthwise.c
 * @brief SIMD-accelerated depthwise convolution operations for TinyAI
 *
 * This file implements SIMD-optimized versions of depthwise convolution operations
 * used in neural networks, with support for 4-bit quantized weights.
 */

#include "simd_ops.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* SIMD detection and platform-specific includes */
#if defined(_MSC_VER)
/* Windows/MSVC */
#include <intrin.h>
#define HAS_SSE2_SUPPORT 1
#if (_MSC_VER >= 1600) /* Visual Studio 2010 and later */
#define HAS_AVX_SUPPORT 1
#endif
#if (_MSC_VER >= 1700) /* Visual Studio 2012 and later */
#define HAS_AVX2_SUPPORT 1
#endif
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
/* GCC/Clang on x86 */
#include <cpuid.h>
#if defined(__SSE2__)
#include <emmintrin.h>
#define HAS_SSE2_SUPPORT 1
#endif
#if defined(__AVX__)
#include <immintrin.h>
#define HAS_AVX_SUPPORT 1
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#define HAS_AVX2_SUPPORT 1
#endif
#endif

/**
 * Reference implementation of depthwise convolution with 4-bit quantized weights
 * (This is the same as in simd_ops_conv.c, included here for completeness)
 */
void tinyaiDepthwiseConv2d4BitReference(float *output, const float *input, const uint8_t *weights,
                                        const float *biases, const float *scaleFactors, int inWidth,
                                        int inHeight, int inChannels, int outWidth, int outHeight,
                                        int multiplier, int kernelSize, int stride, int padding)
{
    int outChannels = inChannels * multiplier;

    /* Compute each output element */
    for (int ic = 0; ic < inChannels; ic++) {
        for (int m = 0; m < multiplier; m++) {
            int oc = ic * multiplier + m;

            /* Get bias for this output channel */
            float bias = biases ? biases[oc] : 0.0f;

            /* Process each output position */
            for (int oh = 0; oh < outHeight; oh++) {
                for (int ow = 0; ow < outWidth; ow++) {
                    float sum = bias;

                    /* Calculate input position corresponding to this output position */
                    int inStartH = oh * stride - padding;
                    int inStartW = ow * stride - padding;

                    /* Process each element of the convolution kernel */
                    for (int kh = 0; kh < kernelSize; kh++) {
                        int inH = inStartH + kh;

                        /* Skip if outside input height */
                        if (inH < 0 || inH >= inHeight)
                            continue;

                        for (int kw = 0; kw < kernelSize; kw++) {
                            int inW = inStartW + kw;

                            /* Skip if outside input width */
                            if (inW < 0 || inW >= inWidth)
                                continue;

                            /* Get input value */
                            float inVal = input[(inH * inWidth + inW) * inChannels + ic];

                            /* Get weight index */
                            int weightIdx =
                                ((kh * kernelSize + kw) * inChannels + ic) * multiplier + m;

                            /* Convert from byte index to 4-bit index and extract the 4-bit value */
                            int byteIndex = weightIdx / 2;
                            int nibblePos = weightIdx % 2;

                            uint8_t packed = weights[byteIndex];
                            uint8_t nibble =
                                nibblePos == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);

                            /* Convert from 4-bit unsigned (0-15) to signed (-8 to 7) range */
                            int8_t quantized = (int8_t)(nibble)-8;

                            /* Dequantize weight */
                            float weight = quantized * scaleFactors[oc];

                            /* Accumulate */
                            sum += inVal * weight;
                        }
                    }

                    /* Store result */
                    output[(oh * outWidth + ow) * outChannels + oc] = sum;
                }
            }
        }
    }
}

#if defined(HAS_SSE2_SUPPORT)
/**
 * SSE2-optimized implementation of depthwise convolution with 4-bit quantized weights
 */
void tinyaiDepthwiseConv2d4BitSSE2(float *output, const float *input, const uint8_t *weights,
                                   const float *biases, const float *scaleFactors, int inWidth,
                                   int inHeight, int inChannels, int outWidth, int outHeight,
                                   int multiplier, int kernelSize, int stride, int padding)
{
    int outChannels = inChannels * multiplier;

    /* Process each input channel and multiplier combination */
    for (int ic = 0; ic < inChannels; ic++) {
        for (int m = 0; m < multiplier; m++) {
            int oc = ic * multiplier + m;

            /* Get bias for this output channel */
            float  bias    = biases ? biases[oc] : 0.0f;
            __m128 biasVec = _mm_set1_ps(bias);

            /* Get scale factor for this output channel */
            float  scale    = scaleFactors[oc];
            __m128 scaleVec = _mm_set1_ps(scale);

            /* Calculate number of output elements processed per iteration */
            int vectorWidth = 4; /* SSE processes 4 floats at a time */

            /* Process output height */
            for (int oh = 0; oh < outHeight; oh++) {
                /* Calculate input row position */
                int inStartH = oh * stride - padding;

                /* Process output width in vector chunks */
                for (int ow = 0; ow < outWidth; ow += vectorWidth) {
                    /* Clamp vector width if near the end */
                    int currentVectorWidth =
                        (ow + vectorWidth <= outWidth) ? vectorWidth : (outWidth - ow);

                    /* Initialize accumulator with bias */
                    __m128 sumVec = biasVec;

                    /* Process each element of the kernel */
                    for (int kh = 0; kh < kernelSize; kh++) {
                        int inH = inStartH + kh;

                        /* Skip if outside input height */
                        if (inH < 0 || inH >= inHeight)
                            continue;

                        for (int kw = 0; kw < kernelSize; kw++) {
                            /* Get weight index base for this kernel position and input channel */
                            int weightBase =
                                ((kh * kernelSize + kw) * inChannels + ic) * multiplier + m;

                            /* Extract and dequantize weight */
                            int     byteIndex = weightBase / 2;
                            int     nibblePos = weightBase % 2;
                            uint8_t packed    = weights[byteIndex];
                            uint8_t nibble =
                                nibblePos == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
                            int8_t quantized = (int8_t)(nibble)-8;
                            float  weight    = quantized * scale;
                            __m128 weightVec = _mm_set1_ps(weight);

                            /* Load input values for the 4 output positions */
                            float inVals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                            for (int i = 0; i < currentVectorWidth; i++) {
                                int inW = (ow + i) * stride - padding + kw;
                                if (inW >= 0 && inW < inWidth) {
                                    inVals[i] = input[(inH * inWidth + inW) * inChannels + ic];
                                }
                            }

                            /* Load input values and multiply with weight */
                            __m128 inVec   = _mm_loadu_ps(inVals);
                            __m128 prodVec = _mm_mul_ps(inVec, weightVec);

                            /* Accumulate */
                            sumVec = _mm_add_ps(sumVec, prodVec);
                        }
                    }

                    /* Store results */
                    float sums[4];
                    _mm_storeu_ps(sums, sumVec);

                    for (int i = 0; i < currentVectorWidth; i++) {
                        output[((oh * outWidth) + (ow + i)) * outChannels + oc] = sums[i];
                    }
                }
            }
        }
    }
}
#endif

#if defined(HAS_AVX2_SUPPORT)
/**
 * AVX2-optimized implementation of depthwise convolution with 4-bit quantized weights
 */
void tinyaiDepthwiseConv2d4BitAVX2(float *output, const float *input, const uint8_t *weights,
                                   const float *biases, const float *scaleFactors, int inWidth,
                                   int inHeight, int inChannels, int outWidth, int outHeight,
                                   int multiplier, int kernelSize, int stride, int padding)
{
    int outChannels = inChannels * multiplier;

    /* Process each input channel and multiplier combination */
    for (int ic = 0; ic < inChannels; ic++) {
        for (int m = 0; m < multiplier; m++) {
            int oc = ic * multiplier + m;

            /* Get bias for this output channel */
            float  bias    = biases ? biases[oc] : 0.0f;
            __m256 biasVec = _mm256_set1_ps(bias);

            /* Get scale factor for this output channel */
            float  scale    = scaleFactors[oc];
            __m256 scaleVec = _mm256_set1_ps(scale);

            /* Calculate number of output elements processed per iteration */
            int vectorWidth = 8; /* AVX2 processes 8 floats at a time */

            /* Process output height */
            for (int oh = 0; oh < outHeight; oh++) {
                /* Calculate input row position */
                int inStartH = oh * stride - padding;

                /* Process output width in vector chunks */
                for (int ow = 0; ow < outWidth; ow += vectorWidth) {
                    /* Clamp vector width if near the end */
                    int currentVectorWidth =
                        (ow + vectorWidth <= outWidth) ? vectorWidth : (outWidth - ow);

                    /* Initialize accumulator with bias */
                    __m256 sumVec = biasVec;

                    /* Process each element of the kernel */
                    for (int kh = 0; kh < kernelSize; kh++) {
                        int inH = inStartH + kh;

                        /* Skip if outside input height */
                        if (inH < 0 || inH >= inHeight)
                            continue;

                        for (int kw = 0; kw < kernelSize; kw++) {
                            /* Get weight index base for this kernel position and input channel */
                            int weightBase =
                                ((kh * kernelSize + kw) * inChannels + ic) * multiplier + m;

                            /* Extract and dequantize weight */
                            int     byteIndex = weightBase / 2;
                            int     nibblePos = weightBase % 2;
                            uint8_t packed    = weights[byteIndex];
                            uint8_t nibble =
                                nibblePos == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
                            int8_t quantized = (int8_t)(nibble)-8;
                            float  weight    = quantized * scale;
                            __m256 weightVec = _mm256_set1_ps(weight);

                            /* Load input values for the 8 output positions */
                            float inVals[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                            for (int i = 0; i < currentVectorWidth; i++) {
                                int inW = (ow + i) * stride - padding + kw;
                                if (inW >= 0 && inW < inWidth) {
                                    inVals[i] = input[(inH * inWidth + inW) * inChannels + ic];
                                }
                            }

                            /* Load input values and multiply with weight */
                            __m256 inVec = _mm256_loadu_ps(inVals);

                            /* Multiply and accumulate */
#if defined(__FMA__)
                            /* Use FMA if available */
                            sumVec = _mm256_fmadd_ps(inVec, weightVec, sumVec);
#else
                            /* Regular multiply and add */
                            __m256 prodVec = _mm256_mul_ps(inVec, weightVec);
                            sumVec         = _mm256_add_ps(sumVec, prodVec);
#endif
                        }
                    }

                    /* Store results */
                    float sums[8];
                    _mm256_storeu_ps(sums, sumVec);

                    for (int i = 0; i < currentVectorWidth; i++) {
                        output[((oh * outWidth) + (ow + i)) * outChannels + oc] = sums[i];
                    }
                }
            }
        }
    }
}
#endif

/**
 * Update the main depthwise convolution function to use the appropriate
 * SIMD implementation based on the available hardware
 */
void tinyaiSimdDepthwiseConv2d4Bit(float *output, const float *input, const uint8_t *weights,
                                   const float *biases, const float *scaleFactors, int inWidth,
                                   int inHeight, int inChannels, int outWidth, int outHeight,
                                   int multiplier, int kernelSize, int stride, int padding)
{
    /* Check if SIMD is available and which version */
    extern bool g_simdInitialized;
    extern bool g_hasSSE2;
    extern bool g_hasAVX;
    extern bool g_hasAVX2;

    /* Use the most advanced SIMD version available */
#if defined(HAS_AVX2_SUPPORT)
    if (g_hasAVX2) {
        tinyaiDepthwiseConv2d4BitAVX2(output, input, weights, biases, scaleFactors, inWidth,
                                      inHeight, inChannels, outWidth, outHeight, multiplier,
                                      kernelSize, stride, padding);
        return;
    }
#endif

#if defined(HAS_SSE2_SUPPORT)
    if (g_hasSSE2) {
        tinyaiDepthwiseConv2d4BitSSE2(output, input, weights, biases, scaleFactors, inWidth,
                                      inHeight, inChannels, outWidth, outHeight, multiplier,
                                      kernelSize, stride, padding);
        return;
    }
#endif

    /* Fallback to reference implementation */
    tinyaiDepthwiseConv2d4BitReference(output, input, weights, biases, scaleFactors, inWidth,
                                       inHeight, inChannels, outWidth, outHeight, multiplier,
                                       kernelSize, stride, padding);
}
