/**
 * @file benchmark.c
 * @brief Benchmarking utilities for TinyAI to compare model performance
 */

#include "../core/memory.h"
#include "../models/image/image_model.h"
#include "../utils/cache_opt.h"
#include "../utils/simd_ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

/**
 * Enumeration for different types of operations to benchmark
 */
typedef enum {
    TINYAI_BENCHMARK_MATMUL,         /* Matrix multiplication */
    TINYAI_BENCHMARK_CONV,           /* 2D Convolution */
    TINYAI_BENCHMARK_DEPTHWISE_CONV, /* Depthwise convolution */
    TINYAI_BENCHMARK_ACTIVATION,     /* Activation functions */
    TINYAI_BENCHMARK_POOLING,        /* Pooling operations */
    TINYAI_BENCHMARK_ATTENTION,      /* Attention mechanism */
    TINYAI_BENCHMARK_QUANTIZE,       /* Quantization */
    TINYAI_BENCHMARK_CUSTOM          /* Custom operation */
} TinyAIBenchmarkType;

/**
 * Structure for SIMD benchmark results
 */
typedef struct {
    const char         *operationName; /* Name of the operation */
    TinyAIBenchmarkType type;          /* Type of operation */
    double              simdTime;      /* Time taken with SIMD (milliseconds) */
    double              nonSimdTime;   /* Time taken without SIMD (milliseconds) */
    double              speedup;       /* SIMD speedup factor */
    int                 dimensions[4]; /* Size dimensions of the operation (depends on type) */
    float               accuracy;      /* Accuracy of SIMD vs non-SIMD (1.0 = identical) */
    const char         *simdType;      /* Type of SIMD used (AVX2, AVX, SSE2, etc.) */
    double              cacheOptTime;  /* Time with cache optimization (milliseconds) */
    double              cacheSpeedup;  /* Cache optimization speedup factor */
} TinyAISimdBenchmarkResult;

/**
 * Structure to hold benchmark results
 */
typedef struct {
    const char *modelName;
    size_t      modelSize;
    size_t      activationSize;
    double      totalTime;
    double      avgInferenceTime;
    double      memoryUsage;
    float       accuracy;
    int         numIterations;
} BenchmarkResult;

/**
 * Get current time in milliseconds
 */
static double getCurrentTimeMs()
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

    return (double)counter.QuadPart * 1000.0 / (double)frequency.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
#endif
}

/**
 * Run a benchmark on an image model
 * @param model The image model to benchmark
 * @param images Array of test images
 * @param labels Array of ground truth labels
 * @param numImages Number of images in the test set
 * @param numIterations Number of iterations to run for each image (for averaging)
 * @return BenchmarkResult structure with benchmark results
 */
BenchmarkResult benchmarkImageModel(TinyAIImageModel *model, TinyAIImage **images, int *labels,
                                    int numImages, int numIterations)
{
    BenchmarkResult result;
    memset(&result, 0, sizeof(BenchmarkResult));

    /* Initialize result structure */
    result.modelName     = "Unknown";
    result.numIterations = numIterations;

    /* Get model size and activation memory information */
    tinyaiImageModelGetMemoryUsage(model, &result.modelSize, &result.activationSize);

    /* Prepare for inference */
    TinyAIImageClassResult classResults[5]; /* Top 5 results for each inference */
    double                 startTime, endTime;
    int                    totalCorrect = 0;

    /* Run benchmark */
    startTime = getCurrentTimeMs();

    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < numIterations; j++) {
            /* Run inference */
            tinyaiImageModelClassify(model, images[i], 5, classResults);

            /* Check if top prediction matches ground truth (for first iteration only) */
            if (j == 0 && classResults[0].classId == labels[i]) {
                totalCorrect++;
            }
        }
    }

    endTime = getCurrentTimeMs();

    /* Calculate results */
    result.totalTime = (endTime - startTime) / 1000.0; /* Convert to seconds */
    result.avgInferenceTime =
        result.totalTime / (numImages * numIterations) * 1000.0; /* ms per inference */
    result.accuracy    = (float)totalCorrect / numImages;
    result.memoryUsage = (result.modelSize + result.activationSize) / (1024.0 * 1024.0); /* MB */

    return result;
}

/**
 * Print a benchmark result
 * @param result The benchmark result to print
 */
void printBenchmarkResult(const BenchmarkResult *result)
{
    printf("Benchmark Results for %s:\n", result->modelName);
    printf("-------------------------------------------\n");
    printf("Model Size: %.2f MB\n", result->modelSize / (1024.0 * 1024.0));
    printf("Activation Memory: %.2f MB\n", result->activationSize / (1024.0 * 1024.0));
    printf("Total Memory: %.2f MB\n", result->memoryUsage);
    printf("Average Inference Time: %.3f ms\n", result->avgInferenceTime);
    printf("Total Benchmark Time: %.3f seconds\n", result->totalTime);
    printf("Accuracy: %.2f%%\n", result->accuracy * 100.0);
    printf("-------------------------------------------\n");
}

/**
 * Compare two models side by side (4-bit quantized vs full precision)
 * @param quantizedModel The 4-bit quantized model
 * @param fullModel The full precision model
 * @param images Test images
 * @param labels Ground truth labels
 * @param numImages Number of images
 * @param numIterations Number of iterations for each model
 */
void compareModels(TinyAIImageModel *quantizedModel, TinyAIImageModel *fullModel,
                   TinyAIImage **images, int *labels, int numImages, int numIterations)
{
    /* Run benchmarks */
    BenchmarkResult quantizedResult =
        benchmarkImageModel(quantizedModel, images, labels, numImages, numIterations);
    BenchmarkResult fullResult =
        benchmarkImageModel(fullModel, images, labels, numImages, numIterations);

    /* Set model names */
    quantizedResult.modelName = "4-bit Quantized Model";
    fullResult.modelName      = "Full Precision Model";

    /* Print individual results */
    printf("\n");
    printBenchmarkResult(&quantizedResult);
    printf("\n");
    printBenchmarkResult(&fullResult);

    /* Print comparison */
    printf("\nComparison (Quantized vs Full Precision):\n");
    printf("-------------------------------------------\n");
    printf("Memory Reduction: %.2fx\n", fullResult.memoryUsage / quantizedResult.memoryUsage);
    printf("Speed Difference: %.2fx\n",
           fullResult.avgInferenceTime / quantizedResult.avgInferenceTime);
    printf("Accuracy Difference: %.2f percentage points\n",
           (quantizedResult.accuracy - fullResult.accuracy) * 100.0);
    printf("-------------------------------------------\n");
}

/**
 * Benchmark multiple models on the same dataset
 * @param models Array of models to benchmark
 * @param modelNames Array of model names
 * @param numModels Number of models
 * @param images Test images
 * @param labels Ground truth labels
 * @param numImages Number of images
 * @param numIterations Number of iterations for each model
 * @return Array of benchmark results
 */
BenchmarkResult *benchmarkMultipleModels(TinyAIImageModel **models, const char **modelNames,
                                         int numModels, TinyAIImage **images, int *labels,
                                         int numImages, int numIterations)
{
    BenchmarkResult *results = (BenchmarkResult *)malloc(numModels * sizeof(BenchmarkResult));
    if (!results) {
        return NULL;
    }

    /* Run benchmarks for each model */
    for (int i = 0; i < numModels; i++) {
        results[i] = benchmarkImageModel(models[i], images, labels, numImages, numIterations);
        results[i].modelName = modelNames[i];
    }

    /* Print each result */
    for (int i = 0; i < numModels; i++) {
        printf("\n");
        printBenchmarkResult(&results[i]);
    }

    return results;
}

/**
 * Create a CSV report from benchmark results
 * @param results Array of benchmark results
 * @param numResults Number of results in the array
 * @param filepath Path to save CSV file
 * @return true on success, false on failure
 */
bool createBenchmarkReport(const BenchmarkResult *results, int numResults, const char *filepath)
{
    FILE *file = fopen(filepath, "w");
    if (!file) {
        return false;
    }

    /* Write CSV header */
    fprintf(file, "Model,Size (MB),Activation Memory (MB),Total Memory (MB),Inference Time "
                  "(ms),Total Time (s),Accuracy (%%)\n");

    /* Write each result */
    for (int i = 0; i < numResults; i++) {
        const BenchmarkResult *result = &results[i];
        fprintf(file, "%s,%.2f,%.2f,%.2f,%.3f,%.3f,%.2f\n", result->modelName,
                result->modelSize / (1024.0 * 1024.0), result->activationSize / (1024.0 * 1024.0),
                result->memoryUsage, result->avgInferenceTime, result->totalTime,
                result->accuracy * 100.0);
    }

    fclose(file);
    return true;
}

/* -------------- SIMD Benchmarking Functions -------------- */

/**
 * Generate random float test data
 * @param size Size of the array to generate
 * @return Pointer to allocated float array with random values
 */
static float *generateRandomData(int size)
{
    float *data = (float *)malloc(size * sizeof(float));
    if (!data)
        return NULL;

    /* Initialize with random values between -1 and 1 */
    for (int i = 0; i < size; i++) {
        data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }

    return data;
}

/**
 * Calculate max difference between two float arrays
 * @param a First array
 * @param b Second array
 * @param size Size of the arrays
 * @return Maximum absolute difference
 */
static float calculateMaxDifference(const float *a, const float *b, int size)
{
    float maxDiff = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > maxDiff) {
            maxDiff = diff;
        }
    }
    return maxDiff;
}

/**
 * Calculate accuracy score (1.0 = perfect match, lower = more divergence)
 * @param a First array
 * @param b Second array
 * @param size Size of the arrays
 * @return Accuracy score between 0 and 1
 */
static float calculateAccuracy(const float *a, const float *b, int size)
{
    float maxDiff = calculateMaxDifference(a, b, size);

    /* Convert max difference to an accuracy score (1.0 = perfect) */
    if (maxDiff < 1e-6f) {
        return 1.0f; /* Perfect match */
    }
    else if (maxDiff > 1.0f) {
        return 0.0f; /* Completely different */
    }
    else {
        return 1.0f - maxDiff;
    }
}

/**
 * Benchmark matrix multiplication with and without SIMD
 * @param rows Number of rows in first matrix
 * @param inner Inner dimension (cols of first, rows of second)
 * @param cols Number of columns in second matrix
 * @param numIterations Number of iterations for benchmark
 * @return Benchmark result
 */
TinyAISimdBenchmarkResult benchmarkMatrixMultiply(int rows, int inner, int cols, int numIterations)
{
    TinyAISimdBenchmarkResult result;
    memset(&result, 0, sizeof(TinyAISimdBenchmarkResult));

    /* Set operation name and type */
    result.operationName = "Matrix Multiplication";
    result.type          = TINYAI_BENCHMARK_MATMUL;

    /* Store dimensions */
    result.dimensions[0] = rows;
    result.dimensions[1] = inner;
    result.dimensions[2] = cols;

    /* Detect available SIMD capabilities */
    int simdCaps = tinyai_simd_detect_capabilities();
    if (simdCaps >= 3) {
        result.simdType = "AVX2";
    }
    else if (simdCaps >= 2) {
        result.simdType = "AVX";
    }
    else if (simdCaps >= 1) {
        result.simdType = "SSE2";
    }
    else {
        result.simdType = "None";
    }

    /* Generate test data */
    float *a       = generateRandomData(rows * inner);
    float *b       = generateRandomData(inner * cols);
    float *c_simd  = (float *)malloc(rows * cols * sizeof(float));
    float *c_ref   = (float *)malloc(rows * cols * sizeof(float));
    float *c_cache = (float *)malloc(rows * cols * sizeof(float));

    if (!a || !b || !c_simd || !c_ref || !c_cache) {
        /* Clean up on allocation failure */
        if (a)
            free(a);
        if (b)
            free(b);
        if (c_simd)
            free(c_simd);
        if (c_ref)
            free(c_ref);
        if (c_cache)
            free(c_cache);

        result.simdTime    = -1;
        result.nonSimdTime = -1;
        result.speedup     = 0;
        return result;
    }

    /* Benchmark SIMD implementation */
    double start = getCurrentTimeMs();
    for (int iter = 0; iter < numIterations; iter++) {
        /* Reset output */
        memset(c_simd, 0, rows * cols * sizeof(float));

        /* Use SIMD matrix multiplication */
        tinyai_simd_matmul_f32(a, b, c_simd, rows, inner, cols);
    }
    double end      = getCurrentTimeMs();
    result.simdTime = (end - start) / numIterations;

    /* Benchmark reference implementation */
    start = getCurrentTimeMs();
    for (int iter = 0; iter < numIterations; iter++) {
        /* Reset output */
        memset(c_ref, 0, rows * cols * sizeof(float));

        /* Simple triple-loop matrix multiplication */
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float sum = 0;
                for (int k = 0; k < inner; k++) {
                    sum += a[i * inner + k] * b[k * cols + j];
                }
                c_ref[i * cols + j] = sum;
            }
        }
    }
    end                = getCurrentTimeMs();
    result.nonSimdTime = (end - start) / numIterations;

    /* Benchmark cache-optimized implementation */
    TinyAICacheOptConfig config = tinyai_cache_opt_init_default();
    tinyai_cache_opt_matrix_multiply(rows, cols, inner, &config);

    start = getCurrentTimeMs();
    for (int iter = 0; iter < numIterations; iter++) {
        /* Reset output */
        memset(c_cache, 0, rows * cols * sizeof(float));

        /* Cache-optimized matrix multiplication with blocking */
        size_t i, j, k;
        TINYAI_LOOP_TILING_2D(i, 0, rows, k, 0, inner, config.blockSizeX, config.blockSizeY, {
            for (j = 0; j < cols; j++) {
                c_cache[i * cols + j] += a[i * inner + k] * b[k * cols + j];
            }
        });
    }
    end                 = getCurrentTimeMs();
    result.cacheOptTime = (end - start) / numIterations;

    /* Calculate speedups */
    result.speedup      = (result.nonSimdTime > 0) ? result.nonSimdTime / result.simdTime : 0;
    result.cacheSpeedup = (result.nonSimdTime > 0) ? result.nonSimdTime / result.cacheOptTime : 0;

    /* Calculate accuracy */
    result.accuracy = calculateAccuracy(c_simd, c_ref, rows * cols);

    /* Clean up */
    free(a);
    free(b);
    free(c_simd);
    free(c_ref);
    free(c_cache);

    return result;
}

/**
 * Benchmark activation functions with and without SIMD
 * @param size Size of the data to process
 * @param activationType Type of activation (0=ReLU, 1=GELU, 2=Sigmoid)
 * @param numIterations Number of iterations for benchmark
 * @return Benchmark result
 */
TinyAISimdBenchmarkResult benchmarkActivation(int size, int activationType, int numIterations)
{
    TinyAISimdBenchmarkResult result;
    memset(&result, 0, sizeof(TinyAISimdBenchmarkResult));

    /* Set operation name and type */
    const char *actNames[] = {"ReLU", "GELU", "Sigmoid"};
    char        nameBuf[64];
    snprintf(nameBuf, sizeof(nameBuf), "%s Activation",
             (activationType >= 0 && activationType <= 2) ? actNames[activationType] : "Unknown");

    result.operationName = strdup(nameBuf);
    result.type          = TINYAI_BENCHMARK_ACTIVATION;

    /* Store dimensions */
    result.dimensions[0] = size;

    /* Detect available SIMD capabilities */
    int simdCaps = tinyai_simd_detect_capabilities();
    if (simdCaps >= 3) {
        result.simdType = "AVX2";
    }
    else if (simdCaps >= 2) {
        result.simdType = "AVX";
    }
    else if (simdCaps >= 1) {
        result.simdType = "SSE2";
    }
    else {
        result.simdType = "None";
    }

    /* Generate test data */
    float *data      = generateRandomData(size);
    float *data_simd = (float *)malloc(size * sizeof(float));
    float *data_ref  = (float *)malloc(size * sizeof(float));

    if (!data || !data_simd || !data_ref) {
        /* Clean up on allocation failure */
        if (data)
            free(data);
        if (data_simd)
            free(data_simd);
        if (data_ref)
            free(data_ref);

        result.simdTime    = -1;
        result.nonSimdTime = -1;
        result.speedup     = 0;
        return result;
    }

    /* Benchmark SIMD implementation */
    double start = getCurrentTimeMs();
    for (int iter = 0; iter < numIterations; iter++) {
        /* Copy input data */
        memcpy(data_simd, data, size * sizeof(float));

        /* Apply activation function using SIMD */
        tinyaiSimdActivate(data_simd, size, activationType);
    }
    double end      = getCurrentTimeMs();
    result.simdTime = (end - start) / numIterations;

    /* Benchmark reference implementation */
    start = getCurrentTimeMs();
    for (int iter = 0; iter < numIterations; iter++) {
        /* Copy input data */
        memcpy(data_ref, data, size * sizeof(float));

        /* Apply activation function without SIMD */
        for (int i = 0; i < size; i++) {
            switch (activationType) {
            case 0: /* ReLU */
                data_ref[i] = (data_ref[i] > 0) ? data_ref[i] : 0;
                break;
            case 1: /* GELU */
                data_ref[i] =
                    0.5f * data_ref[i] *
                    (1.0f + tanhf(0.7978845608f * (data_ref[i] + 0.044715f * data_ref[i] *
                                                                     data_ref[i] * data_ref[i])));
                break;
            case 2: /* Sigmoid */
                data_ref[i] = 1.0f / (1.0f + expf(-data_ref[i]));
                break;
            default:
                /* Do nothing */
                break;
            }
        }
    }
    end                = getCurrentTimeMs();
    result.nonSimdTime = (end - start) / numIterations;

    /* Calculate speedup */
    result.speedup = (result.nonSimdTime > 0) ? result.nonSimdTime / result.simdTime : 0;

    /* Calculate accuracy */
    result.accuracy = calculateAccuracy(data_simd, data_ref, size);

    /* Clean up */
    free(data);
    free(data_simd);
    free(data_ref);

    return result;
}

/**
 * Print detailed SIMD benchmark result
 * @param result Benchmark result to print
 */
void printSimdBenchmarkResult(const TinyAISimdBenchmarkResult *result)
{
    printf("\n=== %s Benchmark ===\n", result->operationName);

    /* Print dimensions based on operation type */
    switch (result->type) {
    case TINYAI_BENCHMARK_MATMUL:
        printf("Dimensions: %d x %d x %d\n", result->dimensions[0], result->dimensions[1],
               result->dimensions[2]);
        break;

    case TINYAI_BENCHMARK_CONV:
    case TINYAI_BENCHMARK_DEPTHWISE_CONV:
        printf("Input: %dx%dx%d, Kernel: %dx%d\n", result->dimensions[0], result->dimensions[1],
               result->dimensions[2], result->dimensions[3], result->dimensions[3]);
        break;

    default:
        printf("Data Size: %d elements\n", result->dimensions[0]);
        break;
    }

    printf("SIMD Type: %s\n", result->simdType);
    printf("Standard Implementation: %.4f ms\n", result->nonSimdTime);
    printf("SIMD Implementation: %.4f ms\n", result->simdTime);

    if (result->cacheOptTime > 0) {
        printf("Cache-Optimized Implementation: %.4f ms\n", result->cacheOptTime);
        printf("Cache-Optimized Speedup: %.2fx\n", result->cacheSpeedup);
    }

    printf("SIMD Speedup: %.2fx\n", result->speedup);
    printf("Numerical Accuracy: %.6f\n", result->accuracy);

    /* Interpretation of results */
    printf("\nInterpretation:\n");

    if (result->speedup >= 4.0) {
        printf("- Outstanding performance improvement with SIMD!\n");
    }
    else if (result->speedup >= 2.0) {
        printf("- Good performance improvement with SIMD.\n");
    }
    else if (result->speedup >= 1.5) {
        printf("- Moderate performance improvement with SIMD.\n");
    }
    else if (result->speedup >= 1.0) {
        printf("- Minor performance improvement with SIMD.\n");
    }
    else {
        printf("- No improvement or slower with SIMD implementation.\n");
    }

    if (result->cacheOptTime > 0) {
        if (result->cacheSpeedup > result->speedup) {
            printf("- Cache optimization provides better speedup than SIMD alone.\n");
        }
        else if (result->cacheSpeedup > 1.0) {
            printf("- Cache optimization provides additional performance benefits.\n");
        }
    }

    if (result->accuracy > 0.999) {
        printf("- Results are numerically equivalent.\n");
    }
    else if (result->accuracy > 0.99) {
        printf("- Results have minor numerical differences, but acceptable.\n");
    }
    else {
        printf("- Warning: Significant numerical differences between implementations.\n");
    }

    printf("===============================\n");
}

/**
 * Create a CSV report for SIMD benchmark results
 * @param results Array of SIMD benchmark results
 * @param numResults Number of results
 * @param filepath Path to save CSV file
 * @return true on success, false on failure
 */
bool createSimdBenchmarkReport(const TinyAISimdBenchmarkResult *results, int numResults,
                               const char *filepath)
{
    FILE *file = fopen(filepath, "w");
    if (!file) {
        return false;
    }

    /* Write CSV header */
    fprintf(file, "Operation,SIMD Type,Standard Time (ms),SIMD Time (ms),SIMD Speedup,"
                  "Cache-Opt Time (ms),Cache Speedup,Accuracy\n");

    /* Write each result */
    for (int i = 0; i < numResults; i++) {
        const TinyAISimdBenchmarkResult *result = &results[i];
        fprintf(file, "%s,%s,%.4f,%.4f,%.2f,%.4f,%.2f,%.6f\n", result->operationName,
                result->simdType, result->nonSimdTime, result->simdTime, result->speedup,
                result->cacheOptTime, result->cacheSpeedup, result->accuracy);
    }

    fclose(file);
    return true;
}

/**
 * Run a comprehensive benchmark suite to compare SIMD and cache optimization performance
 * across various operations with different sizes
 * @param outputFilepath Path to save CSV report (NULL to skip)
 * @return Number of successful benchmarks run
 */
int runComprehensiveBenchmarkSuite(const char *outputFilepath)
{
    /* Number of iterations for consistent results */
    const int numIterations = 10;

/* Array to store results */
#define MAX_BENCHMARK_RESULTS 20
    TinyAISimdBenchmarkResult results[MAX_BENCHMARK_RESULTS];
    int                       resultCount = 0;

    printf("Running comprehensive SIMD and cache optimization benchmark suite...\n");

    /* Matrix multiplication benchmarks with different sizes */
    int sizes[] = {128, 256, 512, 1024};

    for (int i = 0; i < sizeof(sizes) / sizeof(sizes[0]) && resultCount < MAX_BENCHMARK_RESULTS;
         i++) {
        int size             = sizes[i];
        results[resultCount] = benchmarkMatrixMultiply(size, size, size, numIterations);
        printSimdBenchmarkResult(&results[resultCount]);
        resultCount++;
    }

    /* Activation function benchmarks */
    int vectorSizes[]     = {10000, 100000, 1000000};
    int activationTypes[] = {0, 1, 2}; /* ReLU, GELU, Sigmoid */

    for (int i = 0;
         i < sizeof(vectorSizes) / sizeof(vectorSizes[0]) && resultCount < MAX_BENCHMARK_RESULTS;
         i++) {
        for (int j = 0; j < sizeof(activationTypes) / sizeof(activationTypes[0]) &&
                        resultCount < MAX_BENCHMARK_RESULTS;
             j++) {
            results[resultCount] =
                benchmarkActivation(vectorSizes[i], activationTypes[j], numIterations);
            printSimdBenchmarkResult(&results[resultCount]);
            resultCount++;
        }
    }

    /* Save results to CSV if requested */
    if (outputFilepath && resultCount > 0) {
        if (createSimdBenchmarkReport(results, resultCount, outputFilepath)) {
            printf("Benchmark results saved to %s\n", outputFilepath);
        }
        else {
            printf("Failed to save benchmark results to %s\n", outputFilepath);
        }
    }

    /* Free any dynamically allocated memory in results */
    for (int i = 0; i < resultCount; i++) {
        if (results[i].operationName && strstr(results[i].operationName, "Activation")) {
            free((void *)results[i].operationName);
        }
    }

    return resultCount;
}
