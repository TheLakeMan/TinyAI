/**
 * @file simd_ops_conv.c
 * @brief SIMD-accelerated convolution operations for TinyAI image models
 *
 * This file implements SIMD-optimized versions of 2D convolution operations
 * used in image models, with support for 4-bit quantized weights.
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
 * Reference implementation of 2D convolution with 4-bit quantized weights
 *
 * @param output Output feature map (outHeight x outWidth x outChannels)
 * @param input Input feature map (inHeight x inWidth x inChannels)
 * @param weights 4-bit quantized weights (kernelSize x kernelSize x inChannels x outChannels)
 * @param biases Bias values for each output channel
 * @param scaleFactors Scale factors for dequantizing weights
 * @param inWidth Width of input feature map
 * @param inHeight Height of input feature map
 * @param inChannels Number of input channels
 * @param outWidth Width of output feature map
 * @param outHeight Height of output feature map
 * @param outChannels Number of output channels
 * @param kernelSize Size of the convolution kernel (assuming square kernel)
 * @param stride Stride of the convolution
 * @param padding Padding size
 */
void tinyaiConv2d4BitReference(float *output, const float *input, const uint8_t *weights,
                               const float *biases, const float *scaleFactors, int inWidth,
                               int inHeight, int inChannels, int outWidth, int outHeight,
                               int outChannels, int kernelSize, int stride, int padding)
{
    /* Compute each output element */
    for (int oc = 0; oc < outChannels; oc++) {
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

                        /* Process each input channel */
                        for (int ic = 0; ic < inChannels; ic++) {
                            /* Get input value */
                            float inVal = input[(inH * inWidth + inW) * inChannels + ic];

                            /* Get weight index */
                            int weightIdx =
                                ((kh * kernelSize + kw) * inChannels + ic) * outChannels + oc;

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
                }

                /* Store result */
                output[(oh * outWidth + ow) * outChannels + oc] = sum;
            }
        }
    }
}

#if defined(HAS_SSE2_SUPPORT)
/**
 * SSE2-optimized implementation of 2D convolution with 4-bit quantized weights
 */
void tinyaiConv2d4BitSSE2(float *output, const float *input, const uint8_t *weights,
                          const float *biases, const float *scaleFactors, int inWidth, int inHeight,
                          int inChannels, int outWidth, int outHeight, int outChannels,
                          int kernelSize, int stride, int padding)
{
    /* Process each output channel */
    for (int oc = 0; oc < outChannels; oc++) {
        /* Get bias for this output channel */
        float  bias    = biases ? biases[oc] : 0.0f;
        __m128 biasVec = _mm_set1_ps(bias);

        /* Get scale factor for this output channel */
        __m128 scaleVec = _mm_set1_ps(scaleFactors[oc]);

        /* Calculate number of output elements processed per iteration */
        int vectorWidth = 4; /* SSE processes 4 floats at a time */
        int outVectors  = outWidth / vectorWidth;

        /* Process output rows */
        for (int oh = 0; oh < outHeight; oh++) {
            /* Calculate input position corresponding to this output row */
            int inStartH = oh * stride - padding;

            /* Process output columns in vector chunks */
            for (int owv = 0; owv < outVectors; owv++) {
                int ow = owv * vectorWidth;

                /* Initialize accumulators for 4 outputs with bias */
                __m128 sumVec = biasVec;

                /* Process each element of the convolution kernel */
                for (int kh = 0; kh < kernelSize; kh++) {
                    int inH = inStartH + kh;

                    /* Skip if outside input height */
                    if (inH < 0 || inH >= inHeight)
                        continue;

                    for (int kw = 0; kw < kernelSize; kw++) {
                        /* Calculate starting input column positions for the 4 outputs */
                        int inW0 = ow * stride - padding + kw;
                        int inW1 = (ow + 1) * stride - padding + kw;
                        int inW2 = (ow + 2) * stride - padding + kw;
                        int inW3 = (ow + 3) * stride - padding + kw;

                        /* Process each input channel */
                        for (int ic = 0; ic < inChannels; ic++) {
                            /* Get weight index base for this kernel position and input channel */
                            int weightBase =
                                ((kh * kernelSize + kw) * inChannels + ic) * outChannels + oc;

                            /* Convert from byte index to 4-bit index and extract the 4-bit value */
                            int byteIndex = weightBase / 2;
                            int nibblePos = weightBase % 2;

                            uint8_t packed = weights[byteIndex];
                            uint8_t nibble =
                                nibblePos == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);

                            /* Convert from 4-bit unsigned (0-15) to signed (-8 to 7) range */
                            int8_t quantized = (int8_t)(nibble)-8;

                            /* Dequantize weight */
                            float  weight    = quantized * scaleFactors[oc];
                            __m128 weightVec = _mm_set1_ps(weight);

                            /* Get the 4 input values (with bounds checking) */
                            float inVals[4];
                            for (int i = 0; i < 4; i++) {
                                int inW = (ow + i) * stride - padding + kw;
                                if (inW >= 0 && inW < inWidth) {
                                    inVals[i] = input[(inH * inWidth + inW) * inChannels + ic];
                                }
                                else {
                                    inVals[i] = 0.0f; /* Zero padding */
                                }
                            }

                            /* Load input values as vector */
                            __m128 inVec = _mm_loadu_ps(inVals);

                            /* Multiply and accumulate */
                            sumVec = _mm_add_ps(sumVec, _mm_mul_ps(inVec, weightVec));
                        }
                    }
                }

                /* Store the results */
                _mm_storeu_ps(&output[(oh * outWidth + ow) * outChannels + oc], sumVec);
            }

            /* Handle remaining columns */
            for (int ow = outVectors * vectorWidth; ow < outWidth; ow++) {
                float sum = bias;

                /* Calculate input position corresponding to this output position */
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

                        /* Process each input channel */
                        for (int ic = 0; ic < inChannels; ic++) {
                            /* Get input value */
                            float inVal = input[(inH * inWidth + inW) * inChannels + ic];

                            /* Get weight index */
                            int weightIdx =
                                ((kh * kernelSize + kw) * inChannels + ic) * outChannels + oc;

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
                }

                /* Store result */
                output[(oh * outWidth + ow) * outChannels + oc] = sum;
            }
        }
    }
}
#endif

#if defined(HAS_AVX2_SUPPORT)
/**
 * AVX2-optimized implementation of 2D convolution with 4-bit quantized weights
 */
void tinyaiConv2d4BitAVX2(float *output, const float *input, const uint8_t *weights,
                          const float *biases, const float *scaleFactors, int inWidth, int inHeight,
                          int inChannels, int outWidth, int outHeight, int outChannels,
                          int kernelSize, int stride, int padding)
{
    /* Process each output channel */
    for (int oc = 0; oc < outChannels; oc++) {
        /* Get bias for this output channel */
        float  bias    = biases ? biases[oc] : 0.0f;
        __m256 biasVec = _mm256_set1_ps(bias);

        /* Get scale factor for this output channel */
        __m256 scaleVec = _mm256_set1_ps(scaleFactors[oc]);

        /* Calculate number of output elements processed per iteration */
        int vectorWidth = 8; /* AVX2 processes 8 floats at a time */
        int outVectors  = outWidth / vectorWidth;

        /* Process output rows */
        for (int oh = 0; oh < outHeight; oh++) {
            /* Calculate input position corresponding to this output row */
            int inStartH = oh * stride - padding;

            /* Process output columns in vector chunks */
            for (int owv = 0; owv < outVectors; owv++) {
                int ow = owv * vectorWidth;

                /* Initialize accumulators for 8 outputs with bias */
                __m256 sumVec = biasVec;

                /* Process each element of the convolution kernel */
                for (int kh = 0; kh < kernelSize; kh++) {
                    int inH = inStartH + kh;

                    /* Skip if outside input height */
                    if (inH < 0 || inH >= inHeight)
                        continue;

                    for (int kw = 0; kw < kernelSize; kw++) {
                        /* Process each input channel */
                        for (int ic = 0; ic < inChannels; ic++) {
                            /* Get weight index base for this kernel position and input channel */
                            int weightBase =
                                ((kh * kernelSize + kw) * inChannels + ic) * outChannels + oc;

                            /* Convert from byte index to 4-bit index and extract the 4-bit value */
                            int byteIndex = weightBase / 2;
                            int nibblePos = weightBase % 2;

                            uint8_t packed = weights[byteIndex];
                            uint8_t nibble =
                                nibblePos == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);

                            /* Convert from 4-bit unsigned (0-15) to signed (-8 to 7) range */
                            int8_t quantized = (int8_t)(nibble)-8;

                            /* Dequantize weight */
                            float  weight    = quantized * scaleFactors[oc];
                            __m256 weightVec = _mm256_set1_ps(weight);

                            /* Get the 8 input values (with bounds checking) */
                            float inVals[8];
                            for (int i = 0; i < 8; i++) {
                                int inW = (ow + i) * stride - padding + kw;
                                if (inW >= 0 && inW < inWidth && inH >= 0 && inH < inHeight) {
                                    inVals[i] = input[(inH * inWidth + inW) * inChannels + ic];
                                }
                                else {
                                    inVals[i] = 0.0f; /* Zero padding */
                                }
                            }

                            /* Load input values as vector */
                            __m256 inVec = _mm256_loadu_ps(inVals);

/* Multiply and accumulate */
#if defined(__FMA__)
                            /* Use FMA if available */
                            sumVec = _mm256_fmadd_ps(inVec, weightVec, sumVec);
#else
                            /* Regular multiply and add */
                            sumVec = _mm256_add_ps(sumVec, _mm256_mul_ps(inVec, weightVec));
#endif
                        }
                    }
                }

                /* Store the results */
                _mm256_storeu_ps(&output[(oh * outWidth + ow) * outChannels + oc], sumVec);
            }

            /* Handle remaining columns */
            for (int ow = outVectors * vectorWidth; ow < outWidth; ow++) {
                float sum = bias;

                /* Calculate input position corresponding to this output position */
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

                        /* Process each input channel */
                        for (int ic = 0; ic < inChannels; ic++) {
                            /* Get input value */
                            float inVal = input[(inH * inWidth + inW) * inChannels + ic];

                            /* Get weight index */
                            int weightIdx =
                                ((kh * kernelSize + kw) * inChannels + ic) * outChannels + oc;

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
                }

                /* Store result */
                output[(oh * outWidth + ow) * outChannels + oc] = sum;
            }
        }
    }
}
#endif

/**
 * Public API for 2D convolution with 4-bit quantized weights
 * Automatically selects the most optimized implementation available
 */
void tinyaiSimdConv2d4Bit(float *output, const float *input, const uint8_t *weights,
                          const float *biases, const float *scaleFactors, int inWidth, int inHeight,
                          int inChannels, int outWidth, int outHeight, int outChannels,
                          int kernelSize, int stride, int padding)
{
    /* Check if SIMD is available and which version */
    extern bool g_simdInitialized;
    extern bool g_hasSSE2;
    extern bool g_hasAVX;
    extern bool g_hasAVX2;

    /* Use the most advanced SIMD version available */
#if defined(HAS_AVX2_SUPPORT)
    if (g_hasAVX2) {
        tinyaiConv2d4BitAVX2(output, input, weights, biases, scaleFactors, inWidth, inHeight,
                             inChannels, outWidth, outHeight, outChannels, kernelSize, stride,
                             padding);
        return;
    }
#endif

#if defined(HAS_SSE2_SUPPORT)
    if (g_hasSSE2) {
        tinyaiConv2d4BitSSE2(output, input, weights, biases, scaleFactors, inWidth, inHeight,
                             inChannels, outWidth, outHeight, outChannels, kernelSize, stride,
                             padding);
        return;
    }
#endif

    /* Fallback to reference implementation */
    tinyaiConv2d4BitReference(output, input, weights, biases, scaleFactors, inWidth, inHeight,
                              inChannels, outWidth, outHeight, outChannels, kernelSize, stride,
                              padding);
}

/**
 * Depthwise convolution with 4-bit quantized weights (reference implementation)
 *
 * @param output Output feature map (outHeight x outWidth x outChannels)
 * @param input Input feature map (inHeight x inWidth x inChannels)
 * @param weights 4-bit quantized weights (kernelSize x kernelSize x inChannels x multiplier)
 * @param biases Bias values for each output channel
 * @param scaleFactors Scale factors for dequantizing weights
 * @param inWidth Width of input feature map
 * @param inHeight Height of input feature map
 * @param inChannels Number of input channels
 * @param outWidth Width of output feature map
 * @param outHeight Height of output feature map
 * @param multiplier Channel multiplier
 * @param kernelSize Size of the convolution kernel (assuming square kernel)
 * @param stride Stride of the convolution
 * @param padding Padding size
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

/**
 * Public API for depthwise convolution with 4-bit quantized weights
 *
 * The implementation of this function has been moved to simd_ops_depthwise.c
 * which contains optimized SSE2 and AVX2 implementations.
 */
void tinyaiSimdDepthwiseConv2d4Bit(float *output, const float *input, const uint8_t *weights,
                                   const float *biases, const float *scaleFactors, int inWidth,
                                   int inHeight, int inChannels, int outWidth, int outHeight,
                                   int multiplier, int kernelSize, int stride, int padding);
