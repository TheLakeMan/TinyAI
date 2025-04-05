/**
 * @file test_depthwise_conv.c
 * @brief Test for depthwise convolution operations with SIMD acceleration
 */

#include "../utils/simd_ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TEST_WIDTH 28
#define TEST_HEIGHT 28
#define TEST_IN_CHANNELS 16
#define TEST_MULTIPLIER 2
#define TEST_OUT_CHANNELS (TEST_IN_CHANNELS * TEST_MULTIPLIER)
#define TEST_KERNEL_SIZE 3
#define TEST_STRIDE 1
#define TEST_PADDING 1

/**
 * Test routine for depthwise convolution operations
 * Tests correctness by comparing reference implementation with SIMD version
 */
void run_depthwise_conv_tests()
{
    printf("--- Testing Depthwise Convolution SIMD Operations ---\n");

    srand(42); // Fixed seed for reproducibility

    // Calculate output dimensions
    int outWidth  = (TEST_WIDTH + 2 * TEST_PADDING - TEST_KERNEL_SIZE) / TEST_STRIDE + 1;
    int outHeight = (TEST_HEIGHT + 2 * TEST_PADDING - TEST_KERNEL_SIZE) / TEST_STRIDE + 1;

    // Allocate input, weights, bias, scale factors, and output buffers
    float   *input   = (float *)malloc(TEST_WIDTH * TEST_HEIGHT * TEST_IN_CHANNELS * sizeof(float));
    uint8_t *weights = (uint8_t *)malloc(TEST_KERNEL_SIZE * TEST_KERNEL_SIZE * TEST_IN_CHANNELS *
                                         TEST_MULTIPLIER / 2 * sizeof(uint8_t));
    float   *biases  = (float *)malloc(TEST_OUT_CHANNELS * sizeof(float));
    float   *scaleFactors = (float *)malloc(TEST_OUT_CHANNELS * sizeof(float));

    float *outputRef  = (float *)malloc(outWidth * outHeight * TEST_OUT_CHANNELS * sizeof(float));
    float *outputSimd = (float *)malloc(outWidth * outHeight * TEST_OUT_CHANNELS * sizeof(float));

    // Initialize input with random values
    for (int i = 0; i < TEST_WIDTH * TEST_HEIGHT * TEST_IN_CHANNELS; i++) {
        input[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Range: [-1, 1]
    }

    // Initialize weights (4-bit values packed two per byte)
    for (int i = 0;
         i < TEST_KERNEL_SIZE * TEST_KERNEL_SIZE * TEST_IN_CHANNELS * TEST_MULTIPLIER / 2; i++) {
        // Pack two 4-bit values (0-15) into each byte
        uint8_t val1 = rand() % 16;
        uint8_t val2 = rand() % 16;
        weights[i]   = (val2 << 4) | val1;
    }

    // Initialize biases and scale factors
    for (int i = 0; i < TEST_OUT_CHANNELS; i++) {
        biases[i]       = ((float)rand() / RAND_MAX) * 0.1f;  // Small bias values
        scaleFactors[i] = ((float)rand() / RAND_MAX) * 0.01f; // Small scale factors
    }

    // Reference implementation
    printf("Running reference implementation...\n");
    clock_t start = clock();
    tinyaiDepthwiseConv2d4BitReference(
        outputRef, input, weights, biases, scaleFactors, TEST_WIDTH, TEST_HEIGHT, TEST_IN_CHANNELS,
        outWidth, outHeight, TEST_MULTIPLIER, TEST_KERNEL_SIZE, TEST_STRIDE, TEST_PADDING);
    clock_t end      = clock();
    double  ref_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0; // ms
    printf("Reference implementation time: %.3f ms\n", ref_time);

    // SIMD implementation
    printf("Running SIMD implementation...\n");
    start = clock();
    tinyaiSimdDepthwiseConv2d4Bit(outputSimd, input, weights, biases, scaleFactors, TEST_WIDTH,
                                  TEST_HEIGHT, TEST_IN_CHANNELS, outWidth, outHeight,
                                  TEST_MULTIPLIER, TEST_KERNEL_SIZE, TEST_STRIDE, TEST_PADDING);
    end              = clock();
    double simd_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0; // ms
    printf("SIMD implementation time: %.3f ms\n", simd_time);

    // Compute speedup
    double speedup = ref_time / simd_time;
    printf("SIMD Speedup: %.2fx\n", speedup);

    // Compare outputs and compute max difference
    double maxDiff = 0.0;
    for (int i = 0; i < outWidth * outHeight * TEST_OUT_CHANNELS; i++) {
        double diff = fabs(outputRef[i] - outputSimd[i]);
        if (diff > maxDiff) {
            maxDiff = diff;
        }
    }
    printf("Maximum difference between reference and SIMD: %.10f\n", maxDiff);

    // Check if results are accurate enough (considering floating point precision)
    bool success = maxDiff < 1e-5;
    printf("Depthwise convolution test %s!\n", success ? "PASSED" : "FAILED");

    // Free allocated memory
    free(input);
    free(weights);
    free(biases);
    free(scaleFactors);
    free(outputRef);
    free(outputSimd);
}

// Main function if compiled standalone
#ifdef TEST_STANDALONE
int main()
{
    run_depthwise_conv_tests();
    return 0;
}
#endif
