/**
 * @file simd_ops.c
 * @brief Implementation of SIMD-accelerated operations for TinyAI
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
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
/* GCC/Clang on x86 */
#include <cpuid.h>
#include <x86intrin.h>
#define HAS_SSE2_SUPPORT 1
#if defined(__AVX__)
#define HAS_AVX_SUPPORT 1
#endif
#endif

/* SIMD capability flags */
static bool g_simdInitialized = false;
static bool g_hasSSE2         = false;
static bool g_hasAVX          = false;
static bool g_hasAVX2         = false;

/* Detect CPU SIMD capabilities */
static void detectSimdCapabilities()
{
    if (g_simdInitialized)
        return;

#if defined(_MSC_VER)
    int cpuInfo[4] = {0};

    /* Check SSE2 support */
    __cpuid(cpuInfo, 1);
    g_hasSSE2 = (cpuInfo[3] & (1 << 26)) != 0;

    /* Check AVX support */
    g_hasAVX = (cpuInfo[2] & (1 << 28)) != 0;

    /* Check AVX2 support */
    __cpuid(cpuInfo, 7);
    g_hasAVX2 = (cpuInfo[1] & (1 << 5)) != 0;

#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    unsigned int eax, ebx, ecx, edx;

    /* Check SSE2 support */
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    g_hasSSE2 = (edx & (1 << 26)) != 0;

    /* Check AVX support */
    g_hasAVX = (ecx & (1 << 28)) != 0;

    /* Check AVX2 support */
    if (__get_cpuid_max(0, NULL) >= 7) {
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        g_hasAVX2 = (ebx & (1 << 5)) != 0;
    }
#endif

    g_simdInitialized = true;
}

bool tinyaiSimdAvailable(void)
{
    if (!g_simdInitialized) {
        detectSimdCapabilities();
    }
    return g_hasSSE2 || g_hasAVX || g_hasAVX2;
}

/* Reference implementation for 4-bit matrix-vector multiplication */
static void matMul4BitReference(float *out, const uint8_t *weights, const float *input, int rows,
                                int cols, const float *scaleFactors)
{
    /* Each 4-bit weight takes up half a byte, so we need cols/2 bytes per row */
    int bytesPerRow = (cols + 1) / 2;

    for (int row = 0; row < rows; row++) {
        const uint8_t *rowData     = weights + row * bytesPerRow;
        float          sum         = 0.0f;
        float          scaleFactor = scaleFactors[row];

        for (int col = 0; col < cols; col++) {
            int byteIndex = col / 2;
            int nibblePos = col % 2;

            /* Extract the 4-bit value */
            uint8_t packed = rowData[byteIndex];
            uint8_t nibble = nibblePos == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);

            /* Convert from 4-bit unsigned (0-15) to signed (-8 to 7) range */
            int8_t quantized = (int8_t)(nibble)-8;

            /* Dequantize and multiply */
            float weight = quantized * scaleFactor;
            sum += weight * input[col];
        }

        out[row] = sum;
    }
}

#if defined(HAS_SSE2_SUPPORT)
/* SSE2 implementation for 4-bit matrix-vector multiplication */
static void matMul4BitSSE2(float *out, const uint8_t *weights, const float *input, int rows,
                           int cols, const float *scaleFactors)
{
    /* Calculate how many complete 8-element chunks we have */
    int bytesPerRow = (cols + 1) / 2;
    int chunkSize   = 8; /* Process 8 4-bit values (4 bytes) at a time */
    int chunks      = cols / chunkSize;

    for (int row = 0; row < rows; row++) {
        const uint8_t *rowData     = weights + row * bytesPerRow;
        float          scaleFactor = scaleFactors[row];

        /* Use __m128 for 4 floats at a time */
        __m128 sum = _mm_setzero_ps();

        /* Process 8 elements (4 bytes) at a time */
        for (int chunk = 0; chunk < chunks; chunk++) {
            int            colOffset = chunk * chunkSize;
            const uint8_t *chunkData = rowData + (colOffset / 2);

            /* Load 4 bytes containing 8 4-bit values */
            uint32_t packed = *(uint32_t *)chunkData;

            /* Extract 8 4-bit values and convert to signed integers */
            int16_t values[8];
            for (int i = 0; i < 8; i++) {
                uint8_t nibble = (packed >> (i * 4)) & 0x0F;
                values[i]      = (int16_t)nibble - 8; /* Convert to -8 to 7 range */
            }

            /* Load inputs */
            __m128 inputChunk1 = _mm_loadu_ps(input + colOffset);
            __m128 inputChunk2 = _mm_loadu_ps(input + colOffset + 4);

            /* Convert values to floats and scale */
            __m128 scale = _mm_set1_ps(scaleFactor);
            __m128 weights1 =
                _mm_set_ps((float)values[3], (float)values[2], (float)values[1], (float)values[0]);
            __m128 weights2 =
                _mm_set_ps((float)values[7], (float)values[6], (float)values[5], (float)values[4]);

            weights1 = _mm_mul_ps(weights1, scale);
            weights2 = _mm_mul_ps(weights2, scale);

            /* Multiply and accumulate */
            __m128 prod1 = _mm_mul_ps(weights1, inputChunk1);
            __m128 prod2 = _mm_mul_ps(weights2, inputChunk2);

            sum = _mm_add_ps(sum, prod1);
            sum = _mm_add_ps(sum, prod2);
        }

        /* Horizontal sum of the 4 floats in sum */
        __m128 shuf    = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 sums    = _mm_add_ps(sum, shuf);
        shuf           = _mm_movehl_ps(shuf, sums);
        sums           = _mm_add_ss(sums, shuf);
        float totalSum = _mm_cvtss_f32(sums);

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
#endif

#if defined(HAS_AVX_SUPPORT)
/* AVX implementation for 4-bit matrix-vector multiplication */
static void matMul4BitAVX(float *out, const uint8_t *weights, const float *input, int rows,
                          int cols, const float *scaleFactors)
{
    /* Calculate how many complete 16-element chunks we have */
    int bytesPerRow = (cols + 1) / 2;
    int chunkSize   = 16; /* Process 16 4-bit values (8 bytes) at a time */
    int chunks      = cols / chunkSize;

    for (int row = 0; row < rows; row++) {
        const uint8_t *rowData     = weights + row * bytesPerRow;
        float          scaleFactor = scaleFactors[row];

        /* Use __m256 for 8 floats at a time */
        __m256 sum = _mm256_setzero_ps();

        /* Process 16 elements (8 bytes) at a time */
        for (int chunk = 0; chunk < chunks; chunk++) {
            int            colOffset = chunk * chunkSize;
            const uint8_t *chunkData = rowData + (colOffset / 2);

            /* Load 8 bytes containing 16 4-bit values */
            uint64_t packed = *(uint64_t *)chunkData;

            /* Extract 16 4-bit values and convert to signed integers */
            int16_t values[16];
            for (int i = 0; i < 16; i++) {
                uint8_t nibble = (packed >> (i * 4)) & 0x0F;
                values[i]      = (int16_t)nibble - 8; /* Convert to -8 to 7 range */
            }

            /* Load inputs */
            __m256 inputChunk1 = _mm256_loadu_ps(input + colOffset);
            __m256 inputChunk2 = _mm256_loadu_ps(input + colOffset + 8);

            /* Convert values to floats and scale */
            __m256 scale    = _mm256_set1_ps(scaleFactor);
            __m256 weights1 = _mm256_set_ps((float)values[7], (float)values[6], (float)values[5],
                                            (float)values[4], (float)values[3], (float)values[2],
                                            (float)values[1], (float)values[0]);
            __m256 weights2 = _mm256_set_ps((float)values[15], (float)values[14], (float)values[13],
                                            (float)values[12], (float)values[11], (float)values[10],
                                            (float)values[9], (float)values[8]);

            weights1 = _mm256_mul_ps(weights1, scale);
            weights2 = _mm256_mul_ps(weights2, scale);

            /* Multiply and accumulate */
            __m256 prod1 = _mm256_mul_ps(weights1, inputChunk1);
            __m256 prod2 = _mm256_mul_ps(weights2, inputChunk2);

            sum = _mm256_add_ps(sum, prod1);
            sum = _mm256_add_ps(sum, prod2);
        }

        /* Horizontal sum of the 8 floats in sum */
        __m128 sum128  = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
        __m128 shuf    = _mm_movehdup_ps(sum128);
        __m128 sums    = _mm_add_ps(sum128, shuf);
        shuf           = _mm_movehl_ps(shuf, sums);
        sums           = _mm_add_ss(sums, shuf);
        float totalSum = _mm_cvtss_f32(sums);

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
#endif

/* Public API implementation that selects the appropriate SIMD version */
void tinyaiSimdMatMul4Bit(float *out, const uint8_t *weights, const float *input, int rows,
                          int cols, const float *scaleFactors)
{
    if (!g_simdInitialized) {
        detectSimdCapabilities();
    }

    /* Use the most advanced SIMD version available */
#if defined(HAS_AVX2_SUPPORT)
    if (g_hasAVX2) {
        matMul4BitAVX2(out, weights, input, rows, cols, scaleFactors);
        return;
    }
#endif

#if defined(HAS_AVX_SUPPORT)
    if (g_hasAVX) {
        matMul4BitAVX(out, weights, input, rows, cols, scaleFactors);
        return;
    }
#endif

#if defined(HAS_SSE2_SUPPORT)
    if (g_hasSSE2) {
        matMul4BitSSE2(out, weights, input, rows, cols, scaleFactors);
        return;
    }
#endif

    /* Fallback to reference implementation if SIMD not available */
    matMul4BitReference(out, weights, input, rows, cols, scaleFactors);
}

/* Reference implementation for vector addition */
static void vecAddReference(float *out, const float *a, const float *b, int size)
{
    for (int i = 0; i < size; i++) {
        out[i] = a[i] + b[i];
    }
}

#if defined(HAS_SSE2_SUPPORT)
/* SSE2 implementation for vector addition */
static void vecAddSSE2(float *out, const float *a, const float *b, int size)
{
    int chunks = size / 4;

    /* Process 4 elements at a time */
    for (int i = 0; i < chunks; i++) {
        __m128 va  = _mm_loadu_ps(a + i * 4);
        __m128 vb  = _mm_loadu_ps(b + i * 4);
        __m128 sum = _mm_add_ps(va, vb);
        _mm_storeu_ps(out + i * 4, sum);
    }

    /* Handle remaining elements */
    for (int i = chunks * 4; i < size; i++) {
        out[i] = a[i] + b[i];
    }
}
#endif

#if defined(HAS_AVX_SUPPORT)
/* AVX implementation for vector addition */
static void vecAddAVX(float *out, const float *a, const float *b, int size)
{
    int chunks = size / 8;

    /* Process 8 elements at a time */
    for (int i = 0; i < chunks; i++) {
        __m256 va  = _mm256_loadu_ps(a + i * 8);
        __m256 vb  = _mm256_loadu_ps(b + i * 8);
        __m256 sum = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(out + i * 8, sum);
    }

    /* Handle remaining elements */
    for (int i = chunks * 8; i < size; i++) {
        out[i] = a[i] + b[i];
    }
}
#endif

/* AVX2 implementation forward declarations */
#if defined(HAS_AVX2_SUPPORT)
/* Forward declarations from simd_ops_avx2.c */
void matMul4BitAVX2(float *out, const uint8_t *weights, const float *input, int rows, int cols,
                    const float *scaleFactors);
void quantize4BitAVX2(uint8_t *out, const float *in, int size, float *scaleFactors, int blockSize);
void geluActivateAVX2(float *inout, int size);
void sigmoidActivateAVX2(float *inout, int size);
#endif

/* Public API for vector addition */
void tinyaiSimdVecAdd(float *out, const float *a, const float *b, int size)
{
    if (!g_simdInitialized) {
        detectSimdCapabilities();
    }

#if defined(HAS_AVX_SUPPORT)
    if (g_hasAVX) {
        vecAddAVX(out, a, b, size);
        return;
    }
#endif

#if defined(HAS_SSE2_SUPPORT)
    if (g_hasSSE2) {
        vecAddSSE2(out, a, b, size);
        return;
    }
#endif

    vecAddReference(out, a, b, size);
}

/* Reference implementations for activation functions */
static float reluReference(float x) { return x > 0.0f ? x : 0.0f; }

static float geluReference(float x)
{
    /* Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3))) */
    const float sqrt2_over_pi = 0.7978845608f;
    float       x3            = x * x * x;
    float       inner         = sqrt2_over_pi * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

static float sigmoidReference(float x) { return 1.0f / (1.0f + expf(-x)); }

static void activateReference(float *inout, int size, int activationType)
{
    for (int i = 0; i < size; i++) {
        float x = inout[i];
        switch (activationType) {
        case 0: /* ReLU */
            inout[i] = reluReference(x);
            break;
        case 1: /* GELU */
            inout[i] = geluReference(x);
            break;
        case 2: /* Sigmoid */
            inout[i] = sigmoidReference(x);
            break;
        default:
            /* No change for unknown activation type */
            break;
        }
    }
}

#if defined(HAS_SSE2_SUPPORT)
/* SSE2 implementation for activation functions */
static void activateSSE2(float *inout, int size, int activationType)
{
    int chunks = size / 4;

    switch (activationType) {
    case 0: { /* ReLU */
        __m128 zeros = _mm_setzero_ps();
        for (int i = 0; i < chunks; i++) {
            __m128 x      = _mm_loadu_ps(inout + i * 4);
            __m128 result = _mm_max_ps(x, zeros);
            _mm_storeu_ps(inout + i * 4, result);
        }
        break;
    }
    case 1: /* GELU - fallback to reference for complex functions */
    case 2: /* Sigmoid - fallback to reference for complex functions */
        activateReference(inout, size, activationType);
        return;
    }

    /* Handle remaining elements */
    for (int i = chunks * 4; i < size; i++) {
        float x = inout[i];
        switch (activationType) {
        case 0: /* ReLU */
            inout[i] = reluReference(x);
            break;
        case 1: /* GELU */
            inout[i] = geluReference(x);
            break;
        case 2: /* Sigmoid */
            inout[i] = sigmoidReference(x);
            break;
        }
    }
}
#endif

#if defined(HAS_AVX_SUPPORT)
/* AVX implementation for activation functions */
static void activateAVX(float *inout, int size, int activationType)
{
    int chunks = size / 8;

    switch (activationType) {
    case 0: { /* ReLU */
        __m256 zeros = _mm256_setzero_ps();
        for (int i = 0; i < chunks; i++) {
            __m256 x      = _mm256_loadu_ps(inout + i * 8);
            __m256 result = _mm256_max_ps(x, zeros);
            _mm256_storeu_ps(inout + i * 8, result);
        }
        break;
    }
    case 1: /* GELU - fallback to reference for complex functions */
    case 2: /* Sigmoid - fallback to reference for complex functions */
        activateReference(inout, size, activationType);
        return;
    }

    /* Handle remaining elements */
    for (int i = chunks * 8; i < size; i++) {
        float x = inout[i];
        switch (activationType) {
        case 0: /* ReLU */
            inout[i] = reluReference(x);
            break;
        case 1: /* GELU */
            inout[i] = geluReference(x);
            break;
        case 2: /* Sigmoid */
            inout[i] = sigmoidReference(x);
            break;
        }
    }
}
#endif

/* Public API for vector activation */
void tinyaiSimdActivate(float *inout, int size, int activationType)
{
    if (!g_simdInitialized) {
        detectSimdCapabilities();
    }

#if defined(HAS_AVX2_SUPPORT)
    if (g_hasAVX2) {
        /* For AVX2, we have optimized implementations for all activation types */
        switch (activationType) {
        case 1: /* GELU */
            geluActivateAVX2(inout, size);
            break;
        case 2: /* Sigmoid */
            sigmoidActivateAVX2(inout, size);
            break;
        default: /* ReLU and others */
            /* For ReLU, the AVX implementation is sufficient */
            activateAVX(inout, size, activationType);
            break;
        }
        return;
    }
#endif

#if defined(HAS_AVX_SUPPORT)
    if (g_hasAVX) {
        activateAVX(inout, size, activationType);
        return;
    }
#endif

#if defined(HAS_SSE2_SUPPORT)
    if (g_hasSSE2) {
        activateSSE2(inout, size, activationType);
        return;
    }
#endif

    activateReference(inout, size, activationType);
}

/* Implement simplified versions of remaining functions */

void tinyaiSimdMatMul4BitMM(float *out, const uint8_t *a, const float *b, int rowsA, int colsA,
                            int colsB, const float *scaleFactors)
{
    /* Simplified implementation that calls matrix-vector multiply for each column of B */
    float *temp   = (float *)malloc(rowsA * sizeof(float));
    float *column = (float *)malloc(colsA * sizeof(float));

    if (temp && column) {
        for (int colB = 0; colB < colsB; colB++) {
            /* Extract column from B */
            for (int i = 0; i < colsA; i++) {
                column[i] = b[i * colsB + colB];
            }

            /* Multiply A by this column */
            tinyaiSimdMatMul4Bit(temp, a, column, rowsA, colsA, scaleFactors);

            /* Store results in output */
            for (int rowA = 0; rowA < rowsA; rowA++) {
                out[rowA * colsB + colB] = temp[rowA];
            }
        }
    }

    /* Free dynamically allocated memory */
    if (column)
        free(column);
    if (temp)
        free(temp);
}

void tinyaiSimdDequantize4Bit(float *out, const uint8_t *in, int size, const float *scaleFactors)
{
    int bytesPerBlock = (size + 1) / 2;

    for (int i = 0; i < size; i++) {
        int byteIndex = i / 2;
        int nibblePos = i % 2;

        /* Extract the 4-bit value */
        uint8_t packed = in[byteIndex];
        uint8_t nibble = nibblePos == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);

        /* Convert from 4-bit unsigned (0-15) to signed (-8 to 7) range */
        int8_t quantized = (int8_t)(nibble)-8;

        /* Dequantize */
        out[i] = quantized * scaleFactors[i / 256]; /* Assuming 256 values per scale factor */
    }
}

void tinyaiSimdQuantize4Bit(uint8_t *out, const float *in, int size, float *scaleFactors,
                            int blockSize)
{
    if (!g_simdInitialized) {
        detectSimdCapabilities();
    }

    /* Use the most advanced SIMD version available */
#if defined(HAS_AVX2_SUPPORT)
    if (g_hasAVX2) {
        quantize4BitAVX2(out, in, size, scaleFactors, blockSize);
        return;
    }
#endif

    /* Fallback to reference implementation */
    int blocks = (size + blockSize - 1) / blockSize;

    for (int block = 0; block < blocks; block++) {
        int blockStart = block * blockSize;
        int blockEnd   = blockStart + blockSize;
        if (blockEnd > size)
            blockEnd = size;

        /* Find absolute maximum value in block for scaling */
        float maxAbs = 0.0f;
        for (int i = blockStart; i < blockEnd; i++) {
            float absVal = fabsf(in[i]);
            if (absVal > maxAbs)
                maxAbs = absVal;
        }

        /* Calculate scale factor */
        float scale         = maxAbs / 7.0f; /* -7..7 range for 4-bit signed */
        scaleFactors[block] = scale;

        /* Inverse scale for quantization */
        float invScale = scale > 0.0f ? 1.0f / scale : 0.0f;

        /* Quantize values in block */
        for (int i = blockStart; i < blockEnd; i++) {
            /* Scale and round to nearest integer in -8..7 range */
            int scaled = (int)(in[i] * invScale + (in[i] >= 0.0f ? 0.5f : -0.5f));

            /* Clamp to -8..7 range */
            if (scaled < -8)
                scaled = -8;
            if (scaled > 7)
                scaled = 7;

            /* Convert to unsigned 0..15 range */
            uint8_t nibble = (uint8_t)(scaled + 8);

            /* Pack into bytes */
            int outIndex  = i / 2;
            int nibblePos = i % 2;

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
