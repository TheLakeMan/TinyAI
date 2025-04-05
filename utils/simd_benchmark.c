/**
 * @file simd_benchmark.c
 * @brief Implementation of benchmarking utilities specifically for SIMD operations in TinyAI
 */

#include "simd_benchmark.h"
#include "../core/memory.h"
#include "../models/text/attention.h"
#include "../utils/simd_ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <intrin.h>
#include <windows.h>
#else
#include <cpuid.h>
#include <sys/time.h>
#endif

// Define error threshold for floating point comparisons
#define EPSILON 1e-4f

/**
 * Check if AVX2 instructions are supported
 */
static bool is_avx2_supported()
{
#ifdef _WIN32
    int cpuInfo[4];
    __cpuid(cpuInfo, 0);
    int nIds = cpuInfo[0];

    if (nIds >= 7) {
        __cpuidex(cpuInfo, 7, 0);
        return (cpuInfo[1] & (1 << 5)) != 0; // AVX2 bit
    }
    return false;
#else
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(0, &eax, &ebx, &ecx, &edx)) {
        if (eax >= 7) {
            __cpuid_count(7, 0, eax, ebx, ecx, edx);
            return (ebx & (1 << 5)) != 0; // AVX2 bit
        }
    }
    return false;
#endif
}

/**
 * Check if AVX instructions are supported
 */
static bool is_avx_supported()
{
#ifdef _WIN32
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);
    return (cpuInfo[2] & (1 << 28)) != 0; // AVX bit
#else
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return (ecx & (1 << 28)) != 0; // AVX bit
    }
    return false;
#endif
}

/**
 * Check if SSE2 instructions are supported
 */
static bool is_sse2_supported()
{
#ifdef _WIN32
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);
    return (cpuInfo[3] & (1 << 26)) != 0; // SSE2 bit
#else
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return (edx & (1 << 26)) != 0; // SSE2 bit
    }
    return false;
#endif
}

/**
 * Get CPU model name
 */
static char *get_cpu_model()
{
    char *model = (char *)malloc(256);
    if (!model)
        return NULL;

#ifdef _WIN32
    int  cpuInfo[4] = {-1};
    char temp[256]  = {0};

    __cpuid(cpuInfo, 0x80000000);
    unsigned int nExIds = cpuInfo[0];

    memset(model, 0, 256);

    // Get the CPU brand string (if available)
    if (nExIds >= 0x80000004) {
        for (unsigned int i = 0x80000002; i <= 0x80000004; ++i) {
            __cpuid(cpuInfo, i);
            memcpy(temp + (i - 0x80000002) * 16, cpuInfo, sizeof(cpuInfo));
        }
        strcpy(model, temp);
    }
    else {
        strcpy(model, "Unknown CPU");
    }
#else
    unsigned int eax, ebx, ecx, edx;
    char         temp[256] = {0};

    if (__get_cpuid(0x80000000, &eax, &ebx, &ecx, &edx)) {
        unsigned int nExIds = eax;

        memset(model, 0, 256);

        // Get the CPU brand string (if available)
        if (nExIds >= 0x80000004) {
            for (unsigned int i = 0x80000002; i <= 0x80000004; ++i) {
                __get_cpuid(i, &eax, &ebx, &ecx, &edx);

                // Each call gives us 16 bytes (4 32-bit registers)
                uint32_t *p = (uint32_t *)(temp + (i - 0x80000002) * 16);
                p[0]        = eax;
                p[1]        = ebx;
                p[2]        = ecx;
                p[3]        = edx;
            }
            strcpy(model, temp);
        }
        else {
            strcpy(model, "Unknown CPU");
        }
    }
    else {
        strcpy(model, "Unknown CPU");
    }
#endif

    // Trim leading spaces
    char *p = model;
    while (*p == ' ')
        p++;

    if (p != model) {
        memmove(model, p, strlen(p) + 1);
    }

    return model;
}

/**
 * Initialize the SIMD benchmarking system
 * Detects available SIMD features and prepares the benchmarking environment
 *
 * @return Descriptive string of available SIMD features (caller must free)
 */
char *simd_benchmark_init()
{
    char *cpuModel = get_cpu_model();
    bool  hasAVX2  = is_avx2_supported();
    bool  hasAVX   = is_avx_supported();
    bool  hasSSE2  = is_sse2_supported();

    char *result = (char *)malloc(512);
    if (!result) {
        free(cpuModel);
        return NULL;
    }

    snprintf(result, 512, "CPU: %s\nSIMD Support: %s%s%s", cpuModel, hasAVX2 ? "AVX2 " : "",
             hasAVX ? "AVX " : "", hasSSE2 ? "SSE2" : "None");

    free(cpuModel);
    return result;
}

/**
 * Start a timer for benchmark measurement
 *
 * @return Start timestamp in microseconds
 */
uint64_t benchmark_start()
{
#ifdef _WIN32
    static LARGE_INTEGER frequency;
    static int           initialized = 0;

    if (!initialized) {
        QueryPerformanceFrequency(&frequency);
        initialized = 1;
    }

    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);

    return (uint64_t)(counter.QuadPart * 1000000.0 / (double)frequency.QuadPart);
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000 + (uint64_t)tv.tv_usec;
#endif
}

/**
 * End a timer for benchmark measurement
 *
 * @param start Start timestamp from benchmark_start
 * @return Duration in microseconds
 */
uint64_t benchmark_end(uint64_t start)
{
    uint64_t end = benchmark_start();
    return end - start;
}

/**
 * Initialize a benchmark result with common information
 */
static SimdBenchmarkResult initialize_benchmark_result(const char *operationName, size_t dataSize)
{
    SimdBenchmarkResult result;
    memset(&result, 0, sizeof(SimdBenchmarkResult));

    result.operationName   = operationName;
    result.dataSize        = dataSize;
    result.hardwareDetails = simd_benchmark_init();
    result.correctness     = true;
    result.maxError        = 0.0f;

    return result;
}

/**
 * Helper function to compare results and calculate max error
 */
static bool compare_results(const float *simdResult, const float *refResult, size_t size,
                            float *maxError)
{
    bool  correct = true;
    float maxDiff = 0.0f;

    for (size_t i = 0; i < size; i++) {
        float diff = fabsf(simdResult[i] - refResult[i]);
        if (diff > maxDiff) {
            maxDiff = diff;
        }
        if (diff > EPSILON) {
            correct = false;
        }
    }

    *maxError = maxDiff;
    return correct;
}

/**
 * Benchmark matrix multiplication operations (SIMD vs reference implementation)
 *
 * @param rows Number of rows in first matrix
 * @param cols Number of columns in first matrix / rows in second matrix
 * @param innerDim Number of columns in second matrix
 * @param iterations Number of iterations for timing
 * @return Benchmark results
 */
SimdBenchmarkResult benchmark_matrix_multiply(int rows, int cols, int innerDim, int iterations)
{
    size_t              dataSize = rows * cols * innerDim * sizeof(float);
    SimdBenchmarkResult result   = initialize_benchmark_result("Matrix Multiplication", dataSize);

    // Allocate memory for matrices
    float *matrixA    = (float *)malloc(rows * cols * sizeof(float));
    float *matrixB    = (float *)malloc(cols * innerDim * sizeof(float));
    float *outputSimd = (float *)malloc(rows * innerDim * sizeof(float));
    float *outputRef  = (float *)malloc(rows * innerDim * sizeof(float));

    if (!matrixA || !matrixB || !outputSimd || !outputRef) {
        result.correctness = false;

        // Clean up any allocated memory
        if (matrixA)
            free(matrixA);
        if (matrixB)
            free(matrixB);
        if (outputSimd)
            free(outputSimd);
        if (outputRef)
            free(outputRef);

        return result;
    }

    // Initialize matrices with random data
    for (int i = 0; i < rows * cols; i++) {
        matrixA[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    for (int i = 0; i < cols * innerDim; i++) {
        matrixB[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    // Benchmark SIMD implementation
    uint64_t startSimd = benchmark_start();

    for (int i = 0; i < iterations; i++) {
        // Call SIMD matrix multiplication
        // Note: This is a placeholder. The actual function depends on your SIMD implementation
        // simd_matrix_multiply(matrixA, matrixB, outputSimd, rows, cols, innerDim);

        // Instead, we'll use a reference implementation for demonstration
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < innerDim; c++) {
                float sum = 0.0f;
                for (int k = 0; k < cols; k++) {
                    sum += matrixA[r * cols + k] * matrixB[k * innerDim + c];
                }
                outputSimd[r * innerDim + c] = sum;
            }
        }
    }

    result.simdTime = benchmark_end(startSimd);

    // Benchmark reference implementation
    uint64_t startRef = benchmark_start();

    for (int i = 0; i < iterations; i++) {
        // Reference implementation
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < innerDim; c++) {
                float sum = 0.0f;
                for (int k = 0; k < cols; k++) {
                    sum += matrixA[r * cols + k] * matrixB[k * innerDim + c];
                }
                outputRef[r * innerDim + c] = sum;
            }
        }
    }

    result.referenceTime = benchmark_end(startRef);

    // Calculate speedup factor
    result.speedupFactor = (float)result.referenceTime / result.simdTime;

    // Check correctness
    result.correctness = compare_results(outputSimd, outputRef, rows * innerDim, &result.maxError);

    // Clean up
    free(matrixA);
    free(matrixB);
    free(outputSimd);
    free(outputRef);

    return result;
}

/**
 * Benchmark activation functions (SIMD vs reference implementation)
 *
 * @param size Size of the input vector
 * @param activationType Type of activation function (0=ReLU, 1=GELU, 2=Sigmoid)
 * @param iterations Number of iterations for timing
 * @return Benchmark results
 */
SimdBenchmarkResult benchmark_activation(int size, int activationType, int iterations)
{
    const char *activationNames[] = {"ReLU", "GELU", "Sigmoid"};
    char        operationName[64];
    snprintf(operationName, sizeof(operationName), "%s Activation",
             activationType < 3 ? activationNames[activationType] : "Unknown");

    size_t              dataSize = size * sizeof(float);
    SimdBenchmarkResult result   = initialize_benchmark_result(operationName, dataSize);

    // Allocate memory for input/output vectors
    float *input      = (float *)malloc(size * sizeof(float));
    float *outputSimd = (float *)malloc(size * sizeof(float));
    float *outputRef  = (float *)malloc(size * sizeof(float));

    if (!input || !outputSimd || !outputRef) {
        result.correctness = false;

        // Clean up any allocated memory
        if (input)
            free(input);
        if (outputSimd)
            free(outputSimd);
        if (outputRef)
            free(outputRef);

        return result;
    }

    // Initialize input with random data
    for (int i = 0; i < size; i++) {
        input[i] = ((float)rand() / RAND_MAX) * 8.0f - 4.0f; // Range [-4, 4]
    }

    // Function pointers for different activation types
    void (*simd_activation)(const float *, float *, int);
    void (*ref_activation)(const float *, float *, int);

    // Set function pointers based on activation type
    // Note: These are placeholders. Replace with your actual function implementations
    switch (activationType) {
    case 0:                     // ReLU
        simd_activation = NULL; // simd_relu_activation
        ref_activation  = NULL; // reference_relu_activation
        break;
    case 1:                     // GELU
        simd_activation = NULL; // simd_gelu_activation
        ref_activation  = NULL; // reference_gelu_activation
        break;
    case 2:                     // Sigmoid
        simd_activation = NULL; // simd_sigmoid_activation
        ref_activation  = NULL; // reference_sigmoid_activation
        break;
    default:
        result.correctness = false;
        free(input);
        free(outputSimd);
        free(outputRef);
        return result;
    }

    // For demonstration, we'll implement simple versions
    simd_activation = NULL; // Placeholder

    // Benchmark SIMD implementation
    uint64_t startSimd = benchmark_start();

    for (int i = 0; i < iterations; i++) {
        // Call SIMD activation function
        // If we had a real implementation:
        // simd_activation(input, outputSimd, size);

        // For demonstration, use reference implementation
        for (int j = 0; j < size; j++) {
            float x = input[j];
            switch (activationType) {
            case 0: // ReLU
                outputSimd[j] = x > 0.0f ? x : 0.0f;
                break;
            case 1: // GELU
                outputSimd[j] =
                    0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
                break;
            case 2: // Sigmoid
                outputSimd[j] = 1.0f / (1.0f + expf(-x));
                break;
            }
        }
    }

    result.simdTime = benchmark_end(startSimd);

    // Benchmark reference implementation
    uint64_t startRef = benchmark_start();

    for (int i = 0; i < iterations; i++) {
        // Call reference activation function
        // If we had a real implementation:
        // ref_activation(input, outputRef, size);

        // Reference implementation
        for (int j = 0; j < size; j++) {
            float x = input[j];
            switch (activationType) {
            case 0: // ReLU
                outputRef[j] = x > 0.0f ? x : 0.0f;
                break;
            case 1: // GELU
                outputRef[j] =
                    0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
                break;
            case 2: // Sigmoid
                outputRef[j] = 1.0f / (1.0f + expf(-x));
                break;
            }
        }
    }

    result.referenceTime = benchmark_end(startRef);

    // Calculate speedup factor
    result.speedupFactor = (float)result.referenceTime / result.simdTime;

    // Check correctness
    result.correctness = compare_results(outputSimd, outputRef, size, &result.maxError);

    // Clean up
    free(input);
    free(outputSimd);
    free(outputRef);

    return result;
}

/**
 * Benchmark attention mechanisms (SIMD vs reference implementation)
 *
 * @param batchSize Batch size
 * @param seqLength Sequence length
 * @param headDim Dimension of attention heads
 * @param attentionOp Type of attention operation (0=scores, 1=softmax, 2=weighted sum, 3=full)
 * @param iterations Number of iterations for timing
 * @return Benchmark results
 */
SimdBenchmarkResult benchmark_attention(int batchSize, int seqLength, int headDim, int attentionOp,
                                        int iterations)
{
    const char *opNames[] = {"Attention Scores", "Attention Softmax", "Attention Weighted Sum",
                             "Full Attention"};

    char operationName[64];
    snprintf(operationName, sizeof(operationName), "%s",
             attentionOp < 4 ? opNames[attentionOp] : "Unknown Attention Operation");

    size_t dataSize;
    switch (attentionOp) {
    case 0: // scores
        dataSize = batchSize * seqLength * seqLength * sizeof(float);
        break;
    case 1: // softmax
        dataSize = batchSize * seqLength * seqLength * sizeof(float);
        break;
    case 2: // weighted sum
        dataSize = batchSize * seqLength * headDim * sizeof(float);
        break;
    case 3:                                                             // full attention
        dataSize = batchSize * seqLength * headDim * 3 * sizeof(float); // Q, K, V
        break;
    default:
        dataSize = 0;
    }

    SimdBenchmarkResult result = initialize_benchmark_result(operationName, dataSize);

    // Allocate memory based on operation type
    float *query = NULL, *key = NULL, *value = NULL;
    float *scores = NULL, *probsSimd = NULL, *probsRef = NULL;
    float *outputSimd = NULL, *outputRef = NULL;

    switch (attentionOp) {
    case 0: // scores
        query      = (float *)malloc(batchSize * seqLength * headDim * sizeof(float));
        key        = (float *)malloc(batchSize * seqLength * headDim * sizeof(float));
        scores     = (float *)malloc(batchSize * seqLength * seqLength * sizeof(float));
        outputSimd = (float *)malloc(batchSize * seqLength * seqLength * sizeof(float));
        outputRef  = (float *)malloc(batchSize * seqLength * seqLength * sizeof(float));

        if (!query || !key || !scores || !outputSimd || !outputRef) {
            result.correctness = false;
            goto cleanup;
        }

        // Initialize with random data
        for (int i = 0; i < batchSize * seqLength * headDim; i++) {
            query[i] = ((float)rand() / RAND_MAX) * 0.1f;
            key[i]   = ((float)rand() / RAND_MAX) * 0.1f;
        }
        break;

    case 1: // softmax
        scores    = (float *)malloc(batchSize * seqLength * seqLength * sizeof(float));
        probsSimd = (float *)malloc(batchSize * seqLength * seqLength * sizeof(float));
        probsRef  = (float *)malloc(batchSize * seqLength * seqLength * sizeof(float));

        if (!scores || !probsSimd || !probsRef) {
            result.correctness = false;
            goto cleanup;
        }

        // Initialize with random values
        for (int i = 0; i < batchSize * seqLength * seqLength; i++) {
            scores[i] = ((float)rand() / RAND_MAX) * 4.0f - 2.0f; // Range [-2, 2]
        }
        break;

    case 2: // weighted sum
        probsSimd  = (float *)malloc(batchSize * seqLength * seqLength * sizeof(float));
        value      = (float *)malloc(batchSize * seqLength * headDim * sizeof(float));
        outputSimd = (float *)malloc(batchSize * seqLength * headDim * sizeof(float));
        outputRef  = (float *)malloc(batchSize * seqLength * headDim * sizeof(float));

        if (!probsSimd || !value || !outputSimd || !outputRef) {
            result.correctness = false;
            goto cleanup;
        }

        // Initialize with random values
        for (int i = 0; i < batchSize * seqLength * seqLength; i++) {
            probsSimd[i] = ((float)rand() / RAND_MAX) / seqLength; // Ensure row sum close to 1
        }

        // Normalize rows to ensure they sum to 1
        for (int b = 0; b < batchSize; b++) {
            for (int i = 0; i < seqLength; i++) {
                float sum = 0.0f;
                for (int j = 0; j < seqLength; j++) {
                    sum += probsSimd[b * seqLength * seqLength + i * seqLength + j];
                }
                for (int j = 0; j < seqLength; j++) {
                    probsSimd[b * seqLength * seqLength + i * seqLength + j] /= sum;
                }
            }
        }

        for (int i = 0; i < batchSize * seqLength * headDim; i++) {
            value[i] = ((float)rand() / RAND_MAX) * 0.1f;
        }
        break;

    case 3: // full attention
        query      = (float *)malloc(batchSize * seqLength * headDim * sizeof(float));
        key        = (float *)malloc(batchSize * seqLength * headDim * sizeof(float));
        value      = (float *)malloc(batchSize * seqLength * headDim * sizeof(float));
        outputSimd = (float *)malloc(batchSize * seqLength * headDim * sizeof(float));
        outputRef  = (float *)malloc(batchSize * seqLength * headDim * sizeof(float));

        if (!query || !key || !value || !outputSimd || !outputRef) {
            result.correctness = false;
            goto cleanup;
        }

        // Initialize with random values
        for (int i = 0; i < batchSize * seqLength * headDim; i++) {
            query[i] = ((float)rand() / RAND_MAX) * 0.1f;
            key[i]   = ((float)rand() / RAND_MAX) * 0.1f;
            value[i] = ((float)rand() / RAND_MAX) * 0.1f;
        }
        break;

    default:
        result.correctness = false;
        goto cleanup;
    }

    // Benchmark SIMD implementation
    uint64_t startSimd = benchmark_start();

    for (int i = 0; i < iterations; i++) {
        switch (attentionOp) {
        case 0: // scores
            attention_compute_scores_simd(query, key, outputSimd, batchSize, seqLength, headDim);
            break;
        case 1: // softmax
            attention_softmax_simd(scores, probsSimd, batchSize, seqLength);
            break;
        case 2: // weighted sum
            attention_weighted_sum_simd(probsSimd, value, outputSimd, batchSize, seqLength,
                                        headDim);
            break;
        case 3: // full attention
            attention_forward_simd(query, key, value, outputSimd, batchSize, seqLength, headDim);
            break;
        }
    }

    result.simdTime = benchmark_end(startSimd);

    // Benchmark reference implementation
    uint64_t startRef = benchmark_start();

    for (int i = 0; i < iterations; i++) {
        switch (attentionOp) {
        case 0: // scores
            attention_compute_scores_reference(query, key, outputRef, batchSize, seqLength,
                                               headDim);
            break;
        case 1: // softmax
            attention_softmax_reference(scores, probsRef, batchSize, seqLength);
            break;
        case 2: // weighted sum
            attention_weighted_sum_reference(probsSimd, value, outputRef, batchSize, seqLength,
                                             headDim);
            break;
        case 3: // full attention
            attention_forward_reference(query, key, value, outputRef, batchSize, seqLength,
                                        headDim);
            break;
        }
    }

    result.referenceTime = benchmark_end(startRef);

    // Calculate speedup factor
    result.speedupFactor = (float)result.referenceTime / result.simdTime;

    // Check correctness
    switch (attentionOp) {
    case 0: // scores
    case 3: // full attention
        result.correctness = compare_results(
            outputSimd, outputRef, batchSize * seqLength * (attentionOp == 0 ? seqLength : headDim),
            &result.maxError);
        break;
    case 1: // softmax
        result.correctness = compare_results(probsSimd, probsRef, batchSize * seqLength * seqLength,
                                             &result.maxError);
        break;
    case 2: // weighted sum
        result.correctness = compare_results(outputSimd, outputRef, batchSize * seqLength * headDim,
                                             &result.maxError);
        break;
    }

cleanup:
    // Clean up
    if (query)
        free(query);
    if (key)
        free(key);
    if (value)
        free(value);
    if (scores)
        free(scores);
    if (probsSimd)
        free(probsSimd);
    if (probsRef)
        free(probsRef);
    if (outputSimd)
        free(outputSimd);
    if (outputRef)
        free(outputRef);

    return result;
}

/**
 * Benchmark convolution operations (SIMD vs reference implementation)
 *
 * @param inputWidth Input width
 * @param inputHeight Input height
 * @param inputChannels Input channels
 * @param outputChannels Output channels
 * @param kernelSize Kernel size (assumes square kernel)
 * @param convType Type of convolution (0=standard, 1=depthwise)
 * @param iterations Number of iterations for timing
 * @return Benchmark results
 */
SimdBenchmarkResult benchmark_convolution(int inputWidth, int inputHeight, int inputChannels,
                                          int outputChannels, int kernelSize, int convType,
                                          int iterations)
{
    const char *convNames[] = {"Standard Convolution", "Depthwise Convolution"};

    char operationName[64];
    snprintf(operationName, sizeof(operationName), "%s (%dx%d)",
             convType < 2 ? convNames[convType] : "Unknown Convolution", kernelSize, kernelSize);

    size_t dataSize = inputWidth * inputHeight * inputChannels * sizeof(float);

    // Add kernel size to data size calculation
    if (convType == 0) { // standard conv
        dataSize += kernelSize * kernelSize * inputChannels * outputChannels * sizeof(float);
    }
    else { // depthwise conv
        dataSize += kernelSize * kernelSize * inputChannels * sizeof(float);
    }

    SimdBenchmarkResult result = initialize_benchmark_result(operationName, dataSize);

    // Output dimensions
    int outputWidth  = inputWidth - kernelSize + 1;
    int outputHeight = inputHeight - kernelSize + 1;

    // Allocate memory
    float *input = (float *)malloc(inputWidth * inputHeight * inputChannels * sizeof(float));
    float *kernel;

    if (convType == 0) { // standard conv
        kernel = (float *)malloc(kernelSize * kernelSize * inputChannels * outputChannels *
                                 sizeof(float));
    }
    else { // depthwise conv
        kernel = (float *)malloc(kernelSize * kernelSize * inputChannels * sizeof(float));
    }

    float *outputSimd;
    float *outputRef;

    if (convType == 0) { // standard conv
        outputSimd = (float *)malloc(outputWidth * outputHeight * outputChannels * sizeof(float));
        outputRef  = (float *)malloc(outputWidth * outputHeight * outputChannels * sizeof(float));
    }
    else { // depthwise conv
        outputSimd = (float *)malloc(outputWidth * outputHeight * inputChannels * sizeof(float));
        outputRef  = (float *)malloc(outputWidth * outputHeight * inputChannels * sizeof(float));
    }

    if (!input || !kernel || !outputSimd || !outputRef) {
        result.correctness = false;

        // Clean up any allocated memory
        if (input)
            free(input);
        if (kernel)
            free(kernel);
        if (outputSimd)
            free(outputSimd);
        if (outputRef)
            free(outputRef);

        return result;
    }

    // Initialize input with random data
    for (int i = 0; i < inputWidth * inputHeight * inputChannels; i++) {
        input[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    // Initialize kernel with random data
    if (convType == 0) { // standard conv
        for (int i = 0; i < kernelSize * kernelSize * inputChannels * outputChannels; i++) {
            kernel[i] = ((float)rand() / RAND_MAX) * 0.1f;
        }
    }
    else { // depthwise conv
        for (int i = 0; i < kernelSize * kernelSize * inputChannels; i++) {
            kernel[i] = ((float)rand() / RAND_MAX) * 0.1f;
        }
    }

    // Benchmark SIMD implementation
    uint64_t startSimd = benchmark_start();

    for (int i = 0; i < iterations; i++) {
        if (convType == 0) { // standard conv
            // simd_conv2d(input, kernel, outputSimd, inputWidth, inputHeight, inputChannels,
            //             outputChannels, kernelSize);

            // Instead, use a reference implementation for demonstration
            for (int oc = 0; oc < outputChannels; oc++) {
                for (int oh = 0; oh < outputHeight; oh++) {
                    for (int ow = 0; ow < outputWidth; ow++) {
                        float sum = 0.0f;
                        for (int ic = 0; ic < inputChannels; ic++) {
                            for (int kh = 0; kh < kernelSize; kh++) {
                                for (int kw = 0; kw < kernelSize; kw++) {
                                    sum += input[(oh + kh) * inputWidth * inputChannels +
                                                 (ow + kw) * inputChannels + ic] *
                                           kernel[oc * kernelSize * kernelSize * inputChannels +
                                                  kh * kernelSize * inputChannels +
                                                  kw * inputChannels + ic];
                                }
                            }
                        }
                        outputSimd[oh * outputWidth * outputChannels + ow * outputChannels + oc] =
                            sum;
                    }
                }
            }
        }
        else { // depthwise conv
            // simd_depthwise_conv2d(input, kernel, outputSimd, inputWidth, inputHeight,
            //                        inputChannels, kernelSize);

            // Instead, use a reference implementation for demonstration
            for (int ic = 0; ic < inputChannels; ic++) {
                for (int oh = 0; oh < outputHeight; oh++) {
                    for (int ow = 0; ow < outputWidth; ow++) {
                        float sum = 0.0f;
                        for (int kh = 0; kh < kernelSize; kh++) {
                            for (int kw = 0; kw < kernelSize; kw++) {
                                sum += input[(oh + kh) * inputWidth * inputChannels +
                                             (ow + kw) * inputChannels + ic] *
                                       kernel[ic * kernelSize * kernelSize + kh * kernelSize + kw];
                            }
                        }
                        outputSimd[oh * outputWidth * inputChannels + ow * inputChannels + ic] =
                            sum;
                    }
                }
            }
        }
    }

    result.simdTime = benchmark_end(startSimd);

    // Benchmark reference implementation
    uint64_t startRef = benchmark_start();

    for (int i = 0; i < iterations; i++) {
        if (convType == 0) { // standard conv
            // Reference implementation for standard convolution
            for (int oc = 0; oc < outputChannels; oc++) {
                for (int oh = 0; oh < outputHeight; oh++) {
                    for (int ow = 0; ow < outputWidth; ow++) {
                        float sum = 0.0f;
                        for (int ic = 0; ic < inputChannels; ic++) {
                            for (int kh = 0; kh < kernelSize; kh++) {
                                for (int kw = 0; kw < kernelSize; kw++) {
                                    sum += input[(oh + kh) * inputWidth * inputChannels +
                                                 (ow + kw) * inputChannels + ic] *
                                           kernel[oc * kernelSize * kernelSize * inputChannels +
                                                  kh * kernelSize * inputChannels +
                                                  kw * inputChannels + ic];
                                }
                            }
                        }
                        outputRef[oh * outputWidth * outputChannels + ow * outputChannels + oc] =
                            sum;
                    }
                }
            }
        }
        else { // depthwise conv
            // Reference implementation for depthwise convolution
            for (int ic = 0; ic < inputChannels; ic++) {
                for (int oh = 0; oh < outputHeight; oh++) {
                    for (int ow = 0; ow < outputWidth; ow++) {
                        float sum = 0.0f;
                        for (int kh = 0; kh < kernelSize; kh++) {
                            for (int kw = 0; kw < kernelSize; kw++) {
                                sum += input[(oh + kh) * inputWidth * inputChannels +
                                             (ow + kw) * inputChannels + ic] *
                                       kernel[ic * kernelSize * kernelSize + kh * kernelSize + kw];
                            }
                        }
                        outputRef[oh * outputWidth * inputChannels + ow * inputChannels + ic] = sum;
                    }
                }
            }
        }
    }

    result.referenceTime = benchmark_end(startRef);

    // Calculate speedup factor
    result.speedupFactor = (float)result.referenceTime / result.simdTime;

    // Check correctness
    size_t outputSize =
        outputWidth * outputHeight * (convType == 0 ? outputChannels : inputChannels);
    result.correctness = compare_results(outputSimd, outputRef, outputSize, &result.maxError);

    // Clean up
    free(input);
    free(kernel);
    free(outputSimd);
    free(outputRef);

    return result;
}

/**
 * Run comprehensive benchmark suite for all SIMD operations
 * Creates a detailed performance profile for current hardware
 *
 * @param outputFile Path to save benchmark results as JSON (NULL for no file output)
 * @return Array of benchmark results (caller must free)
 */
SimdBenchmarkResult *run_comprehensive_simd_benchmark(const char *outputFile)
{
    // Initialize random seed
    srand((unsigned int)time(NULL));

    // Allocate memory for results (total of 14 different benchmarks)
    SimdBenchmarkResult *results = (SimdBenchmarkResult *)malloc(14 * sizeof(SimdBenchmarkResult));
    if (!results) {
        return NULL;
    }

    int resultIndex = 0;

    // Matrix multiplication benchmarks
    results[resultIndex++] = benchmark_matrix_multiply(32, 32, 32, 10);    // Small matrices
    results[resultIndex++] = benchmark_matrix_multiply(128, 128, 128, 10); // Medium matrices
    results[resultIndex++] = benchmark_matrix_multiply(512, 512, 512, 3);  // Large matrices

    // Activation function benchmarks
    results[resultIndex++] = benchmark_activation(10000, 0, 100); // ReLU
    results[resultIndex++] = benchmark_activation(10000, 1, 100); // GELU
    results[resultIndex++] = benchmark_activation(10000, 2, 100); // Sigmoid

    // Attention mechanism benchmarks
    results[resultIndex++] = benchmark_attention(1, 32, 64, 0, 20); // Attention scores
    results[resultIndex++] = benchmark_attention(1, 32, 64, 1, 20); // Softmax
    results[resultIndex++] = benchmark_attention(1, 32, 64, 2, 20); // Weighted sum
    results[resultIndex++] = benchmark_attention(1, 128, 64, 3, 5); // Full attention

    // Convolution benchmarks
    results[resultIndex++] =
        benchmark_convolution(32, 32, 3, 16, 3, 0, 10); // Standard conv (small)
    results[resultIndex++] =
        benchmark_convolution(64, 64, 16, 32, 3, 0, 5); // Standard conv (medium)
    results[resultIndex++] =
        benchmark_convolution(32, 32, 16, 16, 3, 1, 10); // Depthwise conv (small)
    results[resultIndex++] =
        benchmark_convolution(64, 64, 32, 32, 3, 1, 5); // Depthwise conv (medium)

    // Save results to file if requested
    if (outputFile) {
        FILE *file = fopen(outputFile, "w");
        if (file) {
            fprintf(file, "{\n");
            fprintf(file, "  \"hardware\": \"%s\",\n", results[0].hardwareDetails);
            fprintf(file, "  \"timestamp\": %lld,\n", (long long)time(NULL));
            fprintf(file, "  \"results\": [\n");

            for (int i = 0; i < resultIndex; i++) {
                fprintf(file, "    {\n");
                fprintf(file, "      \"operation\": \"%s\",\n", results[i].operationName);
                fprintf(file, "      \"simd_time_us\": %llu,\n", results[i].simdTime);
                fprintf(file, "      \"reference_time_us\": %llu,\n", results[i].referenceTime);
                fprintf(file, "      \"speedup\": %.2f,\n", results[i].speedupFactor);
                fprintf(file, "      \"correct\": %s,\n",
                        results[i].correctness ? "true" : "false");
                fprintf(file, "      \"max_error\": %.8f,\n", results[i].maxError);
                fprintf(file, "      \"data_size_bytes\": %zu\n", results[i].dataSize);
                fprintf(file, "    }%s\n", (i < resultIndex - 1) ? "," : "");
            }

            fprintf(file, "  ]\n");
            fprintf(file, "}\n");
            fclose(file);
        }
    }

    return results;
}

/**
 * Print a SIMD benchmark result
 *
 * @param result The benchmark result to print
 */
void print_simd_benchmark_result(const SimdBenchmarkResult *result)
{
    printf("Benchmark: %s\n", result->operationName);
    printf("---------------------------------------------\n");
    printf("SIMD Time:      %llu us\n", result->simdTime);
    printf("Reference Time: %llu us\n", result->referenceTime);
    printf("Speedup:        %.2fx\n", result->speedupFactor);
    printf("Correctness:    %s\n", result->correctness ? "PASSED" : "FAILED");
    printf("Max Error:      %.8f\n", result->maxError);
    printf("Data Size:      %.2f KB\n", result->dataSize / 1024.0);
    printf("Hardware:       %s\n", result->hardwareDetails);
    printf("---------------------------------------------\n");
}

/**
 * Create a CSV report from SIMD benchmark results
 *
 * @param results Array of benchmark results
 * @param numResults Number of results in the array
 * @param filepath Path to save CSV file
 * @return true on success, false on failure
 */
bool create_simd_benchmark_report(const SimdBenchmarkResult *results, int numResults,
                                  const char *filepath)
{
    FILE *file = fopen(filepath, "w");
    if (!file) {
        return false;
    }

    // Write CSV header
    fprintf(file, "Operation,SIMD Time (us),Reference Time (us),Speedup Factor,Correctness,Max "
                  "Error,Data Size (KB)\n");

    // Write each result
    for (int i = 0; i < numResults; i++) {
        const SimdBenchmarkResult *result = &results[i];
        fprintf(file, "\"%s\",%llu,%llu,%.2f,%s,%.8f,%.2f\n", result->operationName,
                result->simdTime, result->referenceTime, result->speedupFactor,
                result->correctness ? "PASSED" : "FAILED", result->maxError,
                result->dataSize / 1024.0);
    }

    fclose(file);
    return true;
}

/**
 * Analyze cache usage and memory access patterns during SIMD operations
 *
 * @param operationType Type of operation to analyze (0=matmul, 1=conv, 2=attention)
 * @param dataSize Size of the test data
 * @param reportFile Path to save analysis report (NULL for no file output)
 * @return Performance recommendations string (caller must free)
 */
char *analyze_memory_access_patterns(int operationType, int dataSize, const char *reportFile)
{
    const char *opNames[] = {"Matrix Multiplication", "Convolution", "Attention Mechanism"};

    // Allocate memory for recommendations
    char *recommendations = (char *)malloc(2048);
    if (!recommendations) {
        return NULL;
    }

    // Ensure operation type is valid
    if (operationType < 0 || operationType > 2) {
        snprintf(recommendations, 2048, "Invalid operation type: %d", operationType);
        return recommendations;
    }

    // Prepare test data based on operation type
    int rows, cols, channels, kernelSize, seqLen, headDim;
    switch (operationType) {
    case 0: // Matrix multiplication
        rows = cols = (int)sqrt(dataSize / sizeof(float) / 2);
        break;
    case 1: // Convolution
        channels   = 3;
        kernelSize = 3;
        rows = cols = (int)sqrt(dataSize / sizeof(float) / channels);
        break;
    case 2: // Attention
        seqLen  = (int)cbrt(dataSize / sizeof(float));
        headDim = seqLen * 2;
        break;
    }

    // Run test with instrumentation to detect cache patterns
    // This is a simplified placeholder - real implementation would use hardware counters
    // or detailed profiling tools

    // Create recommendations based on operation type
    snprintf(recommendations, 2048,
             "Memory Access Pattern Analysis for %s\n"
             "===================================================\n\n"
             "Cache Optimization Recommendations:\n\n",
             opNames[operationType]);

    switch (operationType) {
    case 0: // Matrix multiplication
        snprintf(recommendations + strlen(recommendations), 2048 - strlen(recommendations),
                 "1. Use blocking/tiling technique with %dx%d blocks to fit in L1 cache\n"
                 "2. Ensure matrix B is transposed for better memory access patterns\n"
                 "3. Consider loop unrolling and SIMD vectorization for inner loops\n"
                 "4. Pre-fetch data for next block while computing current block\n"
                 "5. Use memory alignment to optimize SIMD loads and stores\n"
                 "6. Consider using Strassen's algorithm for very large matrices\n",
                 32, 32);
        break;
    case 1: // Convolution
        snprintf(recommendations + strlen(recommendations), 2048 - strlen(recommendations),
                 "1. Use im2col transformation for large kernels\n"
                 "2. For smaller kernels, use direct convolution with register blocking\n"
                 "3. Apply input feature map tiling to maximize cache reuse\n"
                 "4. Consider Winograd algorithm for 3x3 kernels to reduce multiplications\n"
                 "5. For depthwise convolution, optimize for spatial locality\n"
                 "6. Fuse ReLU or other activation functions directly into convolution\n");
        break;
    case 2: // Attention
        snprintf(recommendations + strlen(recommendations), 2048 - strlen(recommendations),
                 "1. Pre-compute and cache key-query products for repeated access\n"
                 "2. Use attention masks as early as possible to avoid unnecessary computation\n"
                 "3. Split attention computation into blocks to maximize cache usage\n"
                 "4. Optimize softmax implementation with lookup tables for common cases\n"
                 "5. Consider packed matrices for small sequence lengths\n"
                 "6. Use low-precision arithmetic (FP16) for attention scores when possible\n");
        break;
    }

    // Add general SIMD optimization suggestions
    snprintf(recommendations + strlen(recommendations), 2048 - strlen(recommendations),
             "\nGeneral SIMD and Cache Optimization Suggestions:\n\n"
             "1. Ensure data is aligned to %d-byte boundaries for optimal SIMD performance\n"
             "2. Minimize memory allocation inside critical loops\n"
             "3. Consider data layout transformation (AoS->SoA) for better vectorization\n"
             "4. Avoid pointer aliasing by using const and restrict qualifiers\n"
             "5. Profile cache miss rates and adjust data access patterns accordingly\n"
             "6. Use non-temporal stores for streaming operations (large data, written once)\n"
             "7. Consider using explicit prefetch instructions for predictable access patterns\n",
             32); // Assuming AVX 256-bit registers (32 bytes)

    // Save recommendations to file if requested
    if (reportFile) {
        FILE *file = fopen(reportFile, "w");
        if (file) {
            fprintf(file, "%s", recommendations);
            fclose(file);
        }
    }

    return recommendations;
}
