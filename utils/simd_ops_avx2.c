/**
 * @file simd_ops_avx2.c
 * @brief AVX2-specific optimized SIMD operations
 *
 * This implementation contains the AVX2-optimized versions of key operations
 * for matrix multiplication and other performance-critical functions.
 */

#include "simd_ops.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Check for AVX2 support */
#if defined(_MSC_VER)
/* Windows/MSVC */
#include <intrin.h>
#if (_MSC_VER >= 1700) /* Visual Studio 2012 and later */
#define HAS_AVX2_SUPPORT 1
#endif
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
/* GCC/Clang on x86 */
#include <cpuid.h>
#include <immintrin.h>
#if defined(__AVX2__)
#define HAS_AVX2_SUPPORT 1
#endif
#endif

#if defined(HAS_AVX2_SUPPORT)

/**
 * Optimized 4-bit matrix-vector multiplication using AVX2 instructions
 *
 * This implementation uses several advanced optimization techniques:
 * 1. Processing 32 4-bit values (16 bytes) at a time with 256-bit AVX2 registers
 * 2. Using bit manipulation to unpack 4-bit values efficiently
 * 3. Minimizing memory accesses by processing larger chunks
 * 4. Using FMA instructions when available for better throughput
 *
 * @param out Output vector (rows elements)
 * @param weights 4-bit quantized weight matrix (packed)
 * @param input Input vector (cols elements)
 * @param rows Number of rows in weight matrix
 * @param cols Number of columns in weight matrix
 * @param scaleFactors Scale factors for dequantizing weights (one per row)
 */
void matMul4BitAVX2(float *out, const uint8_t *weights, const float *input, int rows, int cols,
                    const float *scaleFactors)
{
    /* Calculate how many complete 32-element chunks we have */
    int bytesPerRow = (cols + 1) / 2;
    int chunkSize   = 32; /* Process 32 4-bit values (16 bytes) at a time */
    int chunks      = cols / chunkSize;

    /* Process each row */
    for (int row = 0; row < rows; row++) {
        const uint8_t *rowData     = weights + row * bytesPerRow;
        float          scaleFactor = scaleFactors[row];

        /* Use AVX2 for accumulation */
        __m256 sum1 = _mm256_setzero_ps(); /* Accumulator 1 */
        __m256 sum2 = _mm256_setzero_ps(); /* Accumulator 2 */
        __m256 sum3 = _mm256_setzero_ps(); /* Accumulator 3 */
        __m256 sum4 = _mm256_setzero_ps(); /* Accumulator 4 */

        /* Process chunks of 32 elements at a time */
        for (int chunk = 0; chunk < chunks; chunk++) {
            int            colOffset = chunk * chunkSize;
            const uint8_t *chunkData = rowData + (colOffset / 2);

            /* Load 16 bytes containing 32 4-bit values */
            __m128i packed = _mm_loadu_si128((const __m128i *)chunkData);

            /* Extract the 4-bit values using bit manipulation */
            /* First, expand the 4-bit values to 8-bit values */
            __m256i values_8bit = _mm256_cvtepu8_epi16(packed);

            /* Extract lower 4 bits from each byte for even elements */
            __m256i lower_mask  = _mm256_set1_epi16(0x000F);
            __m256i even_values = _mm256_and_si256(values_8bit, lower_mask);

            /* Extract upper 4 bits from each byte for odd elements */
            __m256i odd_values = _mm256_srli_epi16(values_8bit, 4);
            odd_values         = _mm256_and_si256(odd_values, lower_mask);

            /* Combine even and odd values by shuffling */
            __m256i values_mask =
                _mm256_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 0, 8, 1, 9,
                                 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
            __m256i values_combined = _mm256_shuffle_epi8(odd_values, values_mask);

            /* Convert from 4-bit unsigned (0-15) to signed (-8 to 7) range */
            __m256i offset  = _mm256_set1_epi8(8);
            values_combined = _mm256_sub_epi8(values_combined, offset);

            /* Convert the first 8 values to floats */
            __m128i values_lo  = _mm256_extracti128_si256(values_combined, 0);
            __m256  weights_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(values_lo));
            values_lo          = _mm_srli_si128(values_lo, 4);
            __m256 weights_f2  = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(values_lo));

            /* Convert the second 8 values to floats */
            __m128i values_hi  = _mm256_extracti128_si256(values_combined, 1);
            __m256  weights_f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(values_hi));
            values_hi          = _mm_srli_si128(values_hi, 4);
            __m256 weights_f4  = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(values_hi));

            /* Scale the weights */
            __m256 scale = _mm256_set1_ps(scaleFactor);
            weights_f1   = _mm256_mul_ps(weights_f1, scale);
            weights_f2   = _mm256_mul_ps(weights_f2, scale);
            weights_f3   = _mm256_mul_ps(weights_f3, scale);
            weights_f4   = _mm256_mul_ps(weights_f4, scale);

            /* Load input chunks */
            __m256 input1 = _mm256_loadu_ps(input + colOffset);
            __m256 input2 = _mm256_loadu_ps(input + colOffset + 8);
            __m256 input3 = _mm256_loadu_ps(input + colOffset + 16);
            __m256 input4 = _mm256_loadu_ps(input + colOffset + 24);

            /* Multiply and accumulate */
#ifdef __FMA__
            /* Use FMA instructions if available for better performance */
            sum1 = _mm256_fmadd_ps(weights_f1, input1, sum1);
            sum2 = _mm256_fmadd_ps(weights_f2, input2, sum2);
            sum3 = _mm256_fmadd_ps(weights_f3, input3, sum3);
            sum4 = _mm256_fmadd_ps(weights_f4, input4, sum4);
#else
            /* Regular multiply + add */
            __m256 prod1 = _mm256_mul_ps(weights_f1, input1);
            __m256 prod2 = _mm256_mul_ps(weights_f2, input2);
            __m256 prod3 = _mm256_mul_ps(weights_f3, input3);
            __m256 prod4 = _mm256_mul_ps(weights_f4, input4);

            sum1 = _mm256_add_ps(sum1, prod1);
            sum2 = _mm256_add_ps(sum2, prod2);
            sum3 = _mm256_add_ps(sum3, prod3);
            sum4 = _mm256_add_ps(sum4, prod4);
#endif
        }

        /* Combine accumulators */
        __m256 sum_12  = _mm256_add_ps(sum1, sum2);
        __m256 sum_34  = _mm256_add_ps(sum3, sum4);
        __m256 sum_all = _mm256_add_ps(sum_12, sum_34);

        /* Horizontal sum of the 8 floats in sum_all */
        /* Method 1: hadd */
        __m256 sum_hadd1 = _mm256_hadd_ps(sum_all, sum_all);
        __m256 sum_hadd2 = _mm256_hadd_ps(sum_hadd1, sum_hadd1);

        /* Extract the result (both low and high 128-bits) */
        __m128 sum_high  = _mm256_extractf128_ps(sum_hadd2, 1);
        __m128 sum_low   = _mm256_castps256_ps128(sum_hadd2);
        __m128 sum_final = _mm_add_ps(sum_low, sum_high);
        float  totalSum  = _mm_cvtss_f32(sum_final);

        /* Handle remaining columns */
        for (int col = chunks * chunkSize; col < cols; col++) {
            int byteIndex = col / 2;
            int nibblePos = col % 2;

            uint8_t packed = rowData[byteIndex];
            uint8_t nibble = nibblePos == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);

            int8_t quantized = (int8_t)(nibble)-8;
            float  weight    = quantized * scaleFactor;

            totalSum += weight * input[col];
        }

        out[row] = totalSum;
    }
}

/**
 * Improved 4-bit quantization using AVX2
 *
 * @param out Output 4-bit packed array
 * @param in Input float array
 * @param size Number of elements (before packing)
 * @param scaleFactors Scale factors for quantization (output)
 * @param blockSize Number of elements per scale factor
 */
void quantize4BitAVX2(uint8_t *out, const float *in, int size, float *scaleFactors, int blockSize)
{
    int blocks = (size + blockSize - 1) / blockSize;

    /* Process each block */
    for (int block = 0; block < blocks; block++) {
        int blockStart = block * blockSize;
        int blockEnd   = blockStart + blockSize;
        if (blockEnd > size)
            blockEnd = size;

        /* Find max absolute value in the block using AVX2 */
        __m256 maxAbsVec = _mm256_setzero_ps();
        int    i;

        /* Process 8 elements at a time */
        for (i = blockStart; i <= blockEnd - 8; i += 8) {
            __m256 values = _mm256_loadu_ps(in + i);
            /* Get absolute values */
            __m256 absValues = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), values);
            /* Update maximum */
            maxAbsVec = _mm256_max_ps(maxAbsVec, absValues);
        }

        /* Horizontal maximum */
        __m128 max_lo = _mm256_castps256_ps128(maxAbsVec);
        __m128 max_hi = _mm256_extractf128_ps(maxAbsVec, 1);
        __m128 max_2  = _mm_max_ps(max_lo, max_hi);
        __m128 max_3  = _mm_max_ps(max_2, _mm_movehl_ps(max_2, max_2));
        __m128 max_4  = _mm_max_ss(max_3, _mm_shuffle_ps(max_3, max_3, 1));
        float  maxAbs = _mm_cvtss_f32(max_4);

        /* Process remaining elements */
        for (; i < blockEnd; i++) {
            float absVal = fabsf(in[i]);
            if (absVal > maxAbs)
                maxAbs = absVal;
        }

        /* Calculate scale factor */
        float scale         = maxAbs / 7.0f; /* -7..7 range for 4-bit signed */
        scaleFactors[block] = scale;

        /* Inverse scale for quantization */
        float  invScale    = scale > 0.0f ? 1.0f / scale : 0.0f;
        __m256 invScaleVec = _mm256_set1_ps(invScale);

        /* Quantize values in block using AVX2 */
        for (i = blockStart; i <= blockEnd - 8; i += 8) {
            /* Load 8 floats */
            __m256 values = _mm256_loadu_ps(in + i);

            /* Scale */
            __m256 scaled = _mm256_mul_ps(values, invScaleVec);

            /* Round to nearest integer (-8..7 range) */
            __m256 rounded;
/* Rounding implementation depends on compiler/library */
#if defined(__FMA__)
            rounded = _mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
            /* Fallback rounding method */
            __m256 half    = _mm256_set1_ps(0.5f);
            __m256 negHalf = _mm256_set1_ps(-0.5f);
            __m256 add     = _mm256_blendv_ps(half, negHalf,
                                              _mm256_cmp_ps(scaled, _mm256_setzero_ps(), _CMP_LT_OQ));
            rounded        = _mm256_add_ps(scaled, add);
            /* truncate */
            rounded = _mm256_cvtepi32_ps(_mm256_cvtps_epi32(rounded));
#endif

            /* Clamp to -8..7 range */
            __m256 minVal = _mm256_set1_ps(-8.0f);
            __m256 maxVal = _mm256_set1_ps(7.0f);
            rounded       = _mm256_max_ps(rounded, minVal);
            rounded       = _mm256_min_ps(rounded, maxVal);

            /* Convert to integers */
            __m256i intValues = _mm256_cvtps_epi32(rounded);

            /* Convert to unsigned 0..15 range */
            __m256i offset  = _mm256_set1_epi32(8);
            __m256i nibbles = _mm256_add_epi32(intValues, offset);

            /* Pack to 16-bit integers */
            __m128i nibbles_lo = _mm_packs_epi32(_mm256_castsi256_si128(nibbles),
                                                 _mm256_extractf128_si256(nibbles, 1));

            /* Further pack to 8-bit integers */
            __m128i packed_hi    = _mm_shuffle_epi32(nibbles_lo, _MM_SHUFFLE(3, 2, 3, 2));
            __m128i packed_lo    = _mm_shuffle_epi32(nibbles_lo, _MM_SHUFFLE(1, 0, 1, 0));
            __m128i packed_bytes = _mm_packus_epi16(packed_lo, packed_hi);

            /* Compute output positions and nibble positions */
            int outOffset = (i - blockStart) / 2;
            int remain    = (blockEnd - i) / 2;
            int toCopy    = remain < 4 ? remain : 4;

            /* Store bytes (4 or fewer) */
            if (toCopy == 4) {
                /* Store all 4 bytes */
                _mm_storeu_si32(out + outOffset, packed_bytes);
            }
            else {
                /* Store individual bytes */
                for (int b = 0; b < toCopy; b++) {
                    out[outOffset + b] = (uint8_t)_mm_extract_epi8(packed_bytes, b);
                }
            }
        }

        /* Process remaining elements */
        for (; i < blockEnd; i++) {
            /* Scale and round to nearest integer */
            float value  = in[i];
            int   scaled = (int)(value * invScale + (value >= 0.0f ? 0.5f : -0.5f));

            /* Clamp to -8..7 range */
            if (scaled < -8)
                scaled = -8;
            if (scaled > 7)
                scaled = 7;

            /* Convert to unsigned 0..15 range */
            uint8_t nibble = (uint8_t)(scaled + 8);

            /* Pack into bytes */
            int outIndex  = (i - blockStart) / 2;
            int nibblePos = (i - blockStart) % 2;

            if (nibblePos == 0) {
                /* Lower nibble */
                out[outIndex] = (out[outIndex] & 0xF0) | nibble;
            }
            else {
                /* Upper nibble */
                out[outIndex] = (out[outIndex] & 0x0F) | (nibble << 4);
            }
        }
    }
}

/**
 * Optimized GELU activation using AVX2
 *
 * Approximates GELU using the formula:
 * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
 *
 * @param inout Input/output vector
 * @param size Vector size
 */
void geluActivateAVX2(float *inout, int size)
{
    const float  sqrt2_over_pi     = 0.7978845608f;
    const __m256 sqrt2_over_pi_vec = _mm256_set1_ps(sqrt2_over_pi);
    const __m256 coef_vec          = _mm256_set1_ps(0.044715f);
    const __m256 half_vec          = _mm256_set1_ps(0.5f);
    const __m256 one_vec           = _mm256_set1_ps(1.0f);

    int i = 0;

    /* Process 8 elements at a time */
    for (; i <= size - 8; i += 8) {
        /* Load 8 input values */
        __m256 x = _mm256_loadu_ps(inout + i);

        /* Calculate x^3 */
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x3 = _mm256_mul_ps(x2, x);

        /* Calculate sqrt(2/π) * (x + 0.044715 * x^3) */
        __m256 x3_scaled  = _mm256_mul_ps(x3, coef_vec);
        __m256 inner_term = _mm256_add_ps(x, x3_scaled);
        __m256 inner      = _mm256_mul_ps(sqrt2_over_pi_vec, inner_term);

        /* Calculate tanh(inner) */
        /* Approximate tanh using exp */
        __m256 exp_pos = exp256_ps(inner);
        __m256 exp_neg = exp256_ps(_mm256_xor_ps(inner, _mm256_set1_ps(-0.0f))); // -inner
        __m256 tanh =
            _mm256_div_ps(_mm256_sub_ps(exp_pos, exp_neg), _mm256_add_ps(exp_pos, exp_neg));

        /* Calculate 0.5 * x * (1 + tanh(...)) */
        __m256 one_plus_tanh = _mm256_add_ps(one_vec, tanh);
        __m256 result        = _mm256_mul_ps(x, one_plus_tanh);
        result               = _mm256_mul_ps(half_vec, result);

        /* Store the result */
        _mm256_storeu_ps(inout + i, result);
    }

    /* Process remaining elements */
    for (; i < size; i++) {
        float x        = inout[i];
        float x3       = x * x * x;
        float inner    = sqrt2_over_pi * (x + 0.044715f * x3);
        float tanh_val = tanhf(inner);
        inout[i]       = 0.5f * x * (1.0f + tanh_val);
    }
}

/* Approximation of exp(x) using AVX2 */
__m256 exp256_ps(__m256 x)
{
    /* Clamp input to avoid over/underflow */
    __m256 max_x = _mm256_set1_ps(88.3762626647949f);
    __m256 min_x = _mm256_set1_ps(-88.3762626647949f);
    x            = _mm256_max_ps(_mm256_min_ps(x, max_x), min_x);

    /* Constants for polynomial approximation */
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    const __m256 c0    = _mm256_set1_ps(1.0f);
    const __m256 c1    = _mm256_set1_ps(0.9999998807907104f);
    const __m256 c2    = _mm256_set1_ps(0.4999999650754042f);
    const __m256 c3    = _mm256_set1_ps(0.1666653019750762f);
    const __m256 c4    = _mm256_set1_ps(0.0416573311526364f);
    const __m256 c5    = _mm256_set1_ps(0.0083013860616588f);

    /* exp(x) = 2^(log2(e) * x) */
    /* First, calculate log2(e) * x */
    __m256 tx = _mm256_mul_ps(x, log2e);

    /* Split into integer and fractional parts */
    __m256 tx_floor = _mm256_floor_ps(tx);
    __m256 fx       = _mm256_sub_ps(tx, tx_floor);

    /* Polynomial approximation of 2^fx */
    __m256 fx2 = _mm256_mul_ps(fx, fx);
    __m256 fx3 = _mm256_mul_ps(fx2, fx);
    __m256 fx4 = _mm256_mul_ps(fx2, fx2);
    __m256 fx5 = _mm256_mul_ps(fx3, fx2);

    __m256 poly = _mm256_add_ps(c0, _mm256_mul_ps(c1, fx));
    poly        = _mm256_add_ps(poly, _mm256_mul_ps(c2, fx2));
    poly        = _mm256_add_ps(poly, _mm256_mul_ps(c3, fx3));
    poly        = _mm256_add_ps(poly, _mm256_mul_ps(c4, fx4));
    poly        = _mm256_add_ps(poly, _mm256_mul_ps(c5, fx5));

    /* Scale by 2^tx_floor (integer power of 2) */
    __m256i emm0 = _mm256_cvtps_epi32(tx_floor);
    emm0         = _mm256_add_epi32(emm0, _mm256_set1_epi32(127));
    emm0         = _mm256_slli_epi32(emm0, 23);
    __m256 pow2n = _mm256_castsi256_ps(emm0);

    /* Final result: 2^(log2(e) * x) = 2^(tx_floor) * 2^fx */
    return _mm256_mul_ps(pow2n, poly);
}

/**
 * Optimized Sigmoid activation using AVX2
 *
 * Sigmoid(x) = 1 / (1 + exp(-x))
 *
 * @param inout Input/output vector
 * @param size Vector size
 */
void sigmoidActivateAVX2(float *inout, int size)
{
    const __m256 one_vec = _mm256_set1_ps(1.0f);

    int i = 0;

    /* Process 8 elements at a time */
    for (; i <= size - 8; i += 8) {
        /* Load 8 input values */
        __m256 x = _mm256_loadu_ps(inout + i);

        /* Calculate -x */
        __m256 neg_x = _mm256_xor_ps(x, _mm256_set1_ps(-0.0f));

        /* Calculate exp(-x) */
        __m256 exp_neg_x = exp256_ps(neg_x);

        /* Calculate 1 / (1 + exp(-x)) */
        __m256 result = _mm256_div_ps(one_vec, _mm256_add_ps(one_vec, exp_neg_x));

        /* Store the result */
        _mm256_storeu_ps(inout + i, result);
    }

    /* Process remaining elements */
    for (; i < size; i++) {
        float x  = inout[i];
        inout[i] = 1.0f / (1.0f + expf(-x));
    }
}

#endif /* HAS_AVX2_SUPPORT */
