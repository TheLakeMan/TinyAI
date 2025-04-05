/**
 * @file test_cache_opt.c
 * @brief Test for cache optimization utilities
 */

#include "../utils/cache_opt.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

/* Matrix dimensions for testing */
#define MATRIX_SIZE_SMALL 128
#define MATRIX_SIZE_MEDIUM 512
#define MATRIX_SIZE_LARGE 1024

/* Number of iterations for benchmarking */
#define NUM_ITERATIONS 10

/* Buffer for printing results */
char result_buffer[1024];

/**
 * Get current time in milliseconds
 */
static double get_time_ms(void)
{
#ifdef _WIN32
    static LARGE_INTEGER frequency;
    static int           initialized = 0;
    LARGE_INTEGER        counter;

    if (!initialized) {
        QueryPerformanceFrequency(&frequency);
        initialized = 1;
    }

    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)frequency.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
#endif
}

/**
 * Test matrix multiplication with and without cache optimization
 */
void test_matrix_multiply(void)
{
    printf("\n=== Testing Cache-Optimized Matrix Multiplication ===\n");

    /* Initialize random seed */
    srand((unsigned int)time(NULL));

    /* Test sizes */
    int         sizes[]      = {MATRIX_SIZE_SMALL, MATRIX_SIZE_MEDIUM, MATRIX_SIZE_LARGE};
    const char *size_names[] = {"SMALL", "MEDIUM", "LARGE"};

    for (int s = 0; s < 3; s++) {
        size_t size = sizes[s];
        printf("\nTesting with %s matrix (%zux%zu)\n", size_names[s], size, size);

        /* Allocate matrices */
        float *a          = (float *)malloc(size * size * sizeof(float));
        float *b          = (float *)malloc(size * size * sizeof(float));
        float *c_standard = (float *)malloc(size * size * sizeof(float));
        float *c_blocked  = (float *)malloc(size * size * sizeof(float));

        if (!a || !b || !c_standard || !c_blocked) {
            printf("Error: Memory allocation failed\n");
            if (a)
                free(a);
            if (b)
                free(b);
            if (c_standard)
                free(c_standard);
            if (c_blocked)
                free(c_blocked);
            continue;
        }

        /* Initialize matrices with random values */
        for (size_t i = 0; i < size * size; i++) {
            a[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            b[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }

        /* Initialize result matrices */
        memset(c_standard, 0, size * size * sizeof(float));
        memset(c_blocked, 0, size * size * sizeof(float));

        /* Configure cache optimization */
        TinyAICacheOptConfig config = tinyai_cache_opt_init_default();
        tinyai_cache_opt_matrix_multiply(size, size, size, &config);

        printf("Cache optimization config:\n");
        printf("  Block size X: %zu\n", config.blockSizeX);
        printf("  Block size Y: %zu\n", config.blockSizeY);
        printf("  Prefetch distance: %d\n", config.prefetchDistance);
        printf("  Enable tiling: %s\n", config.enableTiling ? "true" : "false");
        printf("  Enable prefetch: %s\n", config.enablePrefetch ? "true" : "false");

        /* Standard matrix multiplication */
        double start_time = get_time_ms();

        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            for (size_t i = 0; i < size; i++) {
                for (size_t j = 0; j < size; j++) {
                    float sum = 0;
                    for (size_t k = 0; k < size; k++) {
                        sum += a[i * size + k] * b[k * size + j];
                    }
                    c_standard[i * size + j] = sum;
                }
            }
        }

        double end_time      = get_time_ms();
        double standard_time = (end_time - start_time) / NUM_ITERATIONS;
        printf("Standard matrix multiply: %.4f ms per iteration\n", standard_time);

        /* Blocked matrix multiplication using cache optimization */
        start_time = get_time_ms();

        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            size_t blockSize = config.blockSizeX;

            for (size_t i_block = 0; i_block < size; i_block += blockSize) {
                for (size_t j_block = 0; j_block < size; j_block += blockSize) {
                    for (size_t k_block = 0; k_block < size; k_block += blockSize) {
                        /* Process current block */
                        size_t i_end = (i_block + blockSize < size) ? i_block + blockSize : size;
                        size_t j_end = (j_block + blockSize < size) ? j_block + blockSize : size;
                        size_t k_end = (k_block + blockSize < size) ? k_block + blockSize : size;

                        for (size_t i = i_block; i < i_end; i++) {
                            for (size_t j = j_block; j < j_end; j++) {
                                /* Prefetch */
                                if (config.enablePrefetch && i + config.prefetchDistance < size) {
                                    tinyai_prefetch(&a[(i + config.prefetchDistance) * size], 0, 2);
                                }

                                float sum = c_blocked[i * size + j];
                                for (size_t k = k_block; k < k_end; k++) {
                                    sum += a[i * size + k] * b[k * size + j];
                                }
                                c_blocked[i * size + j] = sum;
                            }
                        }
                    }
                }
            }
        }

        end_time            = get_time_ms();
        double blocked_time = (end_time - start_time) / NUM_ITERATIONS;
        printf("Blocked matrix multiply: %.4f ms per iteration\n", blocked_time);
        printf("Speedup: %.2fx\n", standard_time / blocked_time);

        /* Verify results */
        float max_diff = 0.0f;
        for (size_t i = 0; i < size * size; i++) {
            float diff = fabsf(c_standard[i] - c_blocked[i]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }

        printf("Maximum difference: %.6f\n", max_diff);
        if (max_diff < 1e-4) {
            printf("PASS: Results match within tolerance\n");
        }
        else {
            printf("FAIL: Results differ by more than tolerance\n");
        }

        /* Free memory */
        free(a);
        free(b);
        free(c_standard);
        free(c_blocked);
    }
}

/**
 * Test transpose with and without blocking
 */
void test_transpose(void)
{
    printf("\n=== Testing Cache-Optimized Matrix Transpose ===\n");

    /* Initialize random seed */
    srand((unsigned int)time(NULL));

    /* Test sizes */
    int         sizes[]      = {MATRIX_SIZE_SMALL, MATRIX_SIZE_MEDIUM, MATRIX_SIZE_LARGE};
    const char *size_names[] = {"SMALL", "MEDIUM", "LARGE"};

    for (int s = 0; s < 3; s++) {
        size_t size = sizes[s];
        printf("\nTesting with %s matrix (%zux%zu)\n", size_names[s], size, size);

        /* Allocate matrices */
        float *src          = (float *)malloc(size * size * sizeof(float));
        float *dst_standard = (float *)malloc(size * size * sizeof(float));
        float *dst_blocked  = (float *)malloc(size * size * sizeof(float));

        if (!src || !dst_standard || !dst_blocked) {
            printf("Error: Memory allocation failed\n");
            if (src)
                free(src);
            if (dst_standard)
                free(dst_standard);
            if (dst_blocked)
                free(dst_blocked);
            continue;
        }

        /* Initialize source matrix with random values */
        for (size_t i = 0; i < size * size; i++) {
            src[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }

        /* Standard transpose */
        double start_time = get_time_ms();

        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            for (size_t i = 0; i < size; i++) {
                for (size_t j = 0; j < size; j++) {
                    dst_standard[j * size + i] = src[i * size + j];
                }
            }
        }

        double end_time      = get_time_ms();
        double standard_time = (end_time - start_time) / NUM_ITERATIONS;
        printf("Standard transpose: %.4f ms per iteration\n", standard_time);

        /* Blocked transpose */
        start_time = get_time_ms();

        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            tinyai_transpose_blocked(dst_blocked, src, size, size, sizeof(float));
        }

        end_time            = get_time_ms();
        double blocked_time = (end_time - start_time) / NUM_ITERATIONS;
        printf("Blocked transpose: %.4f ms per iteration\n", blocked_time);
        printf("Speedup: %.2fx\n", standard_time / blocked_time);

        /* Verify results */
        float max_diff = 0.0f;
        for (size_t i = 0; i < size * size; i++) {
            float diff = fabsf(dst_standard[i] - dst_blocked[i]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }

        printf("Maximum difference: %.6f\n", max_diff);
        if (max_diff < 1e-6) {
            printf("PASS: Results match within tolerance\n");
        }
        else {
            printf("FAIL: Results differ by more than tolerance\n");
        }

        /* Free memory */
        free(src);
        free(dst_standard);
        free(dst_blocked);
    }
}

/**
 * Test cache info detection
 */
void test_cache_info(void)
{
    printf("\n=== Testing Cache Information Detection ===\n");

    TinyAICacheInfo cacheInfo = tinyai_get_cache_info();

    printf("L1 Data Cache Size: %zu bytes (%.2f KB)\n", cacheInfo.l1dCacheSize,
           cacheInfo.l1dCacheSize / 1024.0);
    printf("L2 Cache Size: %zu bytes (%.2f KB)\n", cacheInfo.l2CacheSize,
           cacheInfo.l2CacheSize / 1024.0);
    printf("L3 Cache Size: %zu bytes (%.2f MB)\n", cacheInfo.l3CacheSize,
           cacheInfo.l3CacheSize / (1024.0 * 1024.0));
    printf("Cache Line Size: %zu bytes\n", cacheInfo.cacheLineSize);
    printf("L1 Associativity: %d-way\n", cacheInfo.l1Associativity);
    printf("L2 Associativity: %d-way\n", cacheInfo.l2Associativity);
    printf("L3 Associativity: %d-way\n", cacheInfo.l3Associativity);

    /* Check that cache line size is reasonable */
    if (cacheInfo.cacheLineSize >= 32 && cacheInfo.cacheLineSize <= 128) {
        printf("PASS: Cache line size is reasonable\n");
    }
    else {
        printf("WARN: Cache line size may be incorrect\n");
    }

    /* Check that L1 cache is at least 8KB */
    if (cacheInfo.l1dCacheSize >= 8 * 1024) {
        printf("PASS: L1 cache size is reasonable\n");
    }
    else {
        printf("WARN: L1 cache size may be incorrect\n");
    }
}

/**
 * Main function
 */
int main(void)
{
    printf("=== TinyAI Cache Optimization Tests ===\n");

    /* Test cache info detection */
    test_cache_info();

    /* Test matrix multiplication */
    test_matrix_multiply();

    /* Test transpose */
    test_transpose();

    printf("\nAll cache optimization tests completed.\n");

    return 0;
}
