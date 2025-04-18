/**
 * @file test_simd_acceleration.c
 * @brief SIMD acceleration tests for TinyAI
 */

#include "../utils/simd_ops.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ASSERT(condition, message)                                                                 \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            fprintf(stderr, "Assertion Failed: %s (%s:%d)\n", message, __FILE__, __LINE__);        \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

/**
 * Create a test array with random values
 */
static float *create_test_array(int size)
{
    float *array = (float *)malloc(size * sizeof(float));
    ASSERT(array != NULL, "Failed to allocate test array");

    for (int i = 0; i < size; i++) {
        array[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // -1.0 to 1.0
    }

    return array;
}

/**
 * Test vector addition with different implementations
 */
static void test_vector_addition()
{
    printf("Testing vector addition implementations...\n");

    // Create large arrays to better measure performance differences
    const int size     = 10000;
    float    *a        = create_test_array(size);
    float    *b        = create_test_array(size);
    float    *c_scalar = (float *)malloc(size * sizeof(float));
    float    *c_simd   = (float *)malloc(size * sizeof(float));

    ASSERT(c_scalar != NULL && c_simd != NULL, "Failed to allocate result arrays");

    // Scalar implementation
    printf("  Running scalar implementation...\n");
    clock_t start_scalar = clock();

    for (int i = 0; i < size; i++) {
        c_scalar[i] = a[i] + b[i];
    }

    clock_t end_scalar  = clock();
    double  time_scalar = (double)(end_scalar - start_scalar) / CLOCKS_PER_SEC;

    // SIMD implementation
    printf("  Running SIMD implementation...\n");
    clock_t start_simd = clock();

    bool success = tinyaiVectorAdd(a, b, c_simd, size);
    ASSERT(success, "SIMD vector addition failed");

    clock_t end_simd  = clock();
    double  time_simd = (double)(end_simd - start_simd) / CLOCKS_PER_SEC;

    // Verify results
    for (int i = 0; i < size; i++) {
        ASSERT(fabsf(c_scalar[i] - c_simd[i]) < 0.000001f,
               "SIMD vector addition produced incorrect results");
    }

    // Check SIMD acceleration
    printf("  Scalar time: %.6f seconds\n", time_scalar);
    printf("  SIMD time: %.6f seconds\n", time_simd);
    printf("  Speedup: %.2fx\n", time_scalar / time_simd);

    // Cleanup
    free(a);
    free(b);
    free(c_scalar);
    free(c_simd);

    printf("  PASS: Vector addition tests\n");
}

/**
 * Test matrix multiplication with different implementations
 */
static void test_matrix_multiplication()
{
    printf("Testing matrix multiplication implementations...\n");

    // Matrix dimensions
    const int m = 128; // A rows
    const int k = 256; // A cols / B rows
    const int n = 128; // B cols

    // Create matrices
    float *a        = create_test_array(m * k);
    float *b        = create_test_array(k * n);
    float *c_scalar = (float *)malloc(m * n * sizeof(float));
    float *c_simd   = (float *)malloc(m * n * sizeof(float));

    ASSERT(c_scalar != NULL && c_simd != NULL, "Failed to allocate result matrices");

    // Scalar implementation
    printf("  Running scalar implementation...\n");
    clock_t start_scalar = clock();

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            c_scalar[i * n + j] = 0.0f;
            for (int l = 0; l < k; l++) {
                c_scalar[i * n + j] += a[i * k + l] * b[l * n + j];
            }
        }
    }

    clock_t end_scalar  = clock();
    double  time_scalar = (double)(end_scalar - start_scalar) / CLOCKS_PER_SEC;

    // SIMD implementation
    printf("  Running SIMD implementation...\n");
    clock_t start_simd = clock();

    bool success = tinyaiMatrixMultiply(a, b, c_simd, m, k, n);
    ASSERT(success, "SIMD matrix multiplication failed");

    clock_t end_simd  = clock();
    double  time_simd = (double)(end_simd - start_simd) / CLOCKS_PER_SEC;

    // Verify results with tolerance (due to potential differences in floating point arithmetic)
    float max_diff = 0.0f;
    for (int i = 0; i < m * n; i++) {
        float diff = fabsf(c_scalar[i] - c_simd[i]);
        max_diff   = fmaxf(max_diff, diff);
    }

    printf("  Maximum result difference: %.6f\n", max_diff);
    ASSERT(max_diff < 0.001f, "SIMD matrix multiplication produced incorrect results");

    // Check SIMD acceleration
    printf("  Scalar time: %.6f seconds\n", time_scalar);
    printf("  SIMD time: %.6f seconds\n", time_simd);
    printf("  Speedup: %.2fx\n", time_scalar / time_simd);

    // Cleanup
    free(a);
    free(b);
    free(c_scalar);
    free(c_simd);

    printf("  PASS: Matrix multiplication tests\n");
}

/**
 * Test SIMD convolution operations
 */
static void test_convolution()
{
    printf("Testing convolution implementations...\n");

    // Image dimensions
    const int height   = 32;
    const int width    = 32;
    const int channels = 3;

    // Kernel dimensions
    const int kernel_size = 3;
    const int num_filters = 16;

    // Create data
    float *input   = create_test_array(height * width * channels);
    float *kernels = create_test_array(kernel_size * kernel_size * channels * num_filters);
    float *bias    = create_test_array(num_filters);

    // Output dimensions
    const int out_height = height - kernel_size + 1;
    const int out_width  = width - kernel_size + 1;

    float *output_scalar = (float *)malloc(out_height * out_width * num_filters * sizeof(float));
    float *output_simd   = (float *)malloc(out_height * out_width * num_filters * sizeof(float));

    ASSERT(output_scalar != NULL && output_simd != NULL, "Failed to allocate output arrays");

    // Scalar implementation
    printf("  Running scalar convolution...\n");
    clock_t start_scalar = clock();

    // Simple scalar convolution implementation
    for (int f = 0; f < num_filters; f++) {
        for (int h = 0; h < out_height; h++) {
            for (int w = 0; w < out_width; w++) {
                float sum = bias[f];

                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        for (int c = 0; c < channels; c++) {
                            int input_idx = ((h + kh) * width + (w + kw)) * channels + c;
                            int kernel_idx =
                                ((f * kernel_size + kh) * kernel_size + kw) * channels + c;

                            sum += input[input_idx] * kernels[kernel_idx];
                        }
                    }
                }

                output_scalar[(f * out_height + h) * out_width + w] = sum;
            }
        }
    }

    clock_t end_scalar  = clock();
    double  time_scalar = (double)(end_scalar - start_scalar) / CLOCKS_PER_SEC;

    // SIMD implementation
    printf("  Running SIMD convolution...\n");
    clock_t start_simd = clock();

    bool success = tinyaiConv2D(input, kernels, bias, output_simd, height, width, channels,
                                kernel_size, num_filters);
    ASSERT(success, "SIMD convolution failed");

    clock_t end_simd  = clock();
    double  time_simd = (double)(end_simd - start_simd) / CLOCKS_PER_SEC;

    // Verify results with tolerance
    float max_diff = 0.0f;
    for (int i = 0; i < out_height * out_width * num_filters; i++) {
        float diff = fabsf(output_scalar[i] - output_simd[i]);
        max_diff   = fmaxf(max_diff, diff);
    }

    printf("  Maximum result difference: %.6f\n", max_diff);
    ASSERT(max_diff < 0.01f, "SIMD convolution produced incorrect results");

    // Check SIMD acceleration
    printf("  Scalar time: %.6f seconds\n", time_scalar);
    printf("  SIMD time: %.6f seconds\n", time_simd);
    printf("  Speedup: %.2fx\n", time_scalar / time_simd);

    // Cleanup
    free(input);
    free(kernels);
    free(bias);
    free(output_scalar);
    free(output_simd);

    printf("  PASS: Convolution tests\n");
}

/**
 * Test SIMD activation functions (ReLU, Sigmoid, Tanh)
 */
static void test_activation_functions()
{
    printf("Testing SIMD activation functions...\n");

    const int size  = 10000;
    float    *input = create_test_array(size);

    // ReLU
    printf("  Testing ReLU activation...\n");
    float *output_relu_scalar = (float *)malloc(size * sizeof(float));
    float *output_relu_simd   = (float *)malloc(size * sizeof(float));

    // Scalar ReLU
    clock_t start_scalar = clock();
    for (int i = 0; i < size; i++) {
        output_relu_scalar[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
    clock_t end_scalar       = clock();
    double  time_relu_scalar = (double)(end_scalar - start_scalar) / CLOCKS_PER_SEC;

    // SIMD ReLU
    clock_t start_simd   = clock();
    bool    relu_success = tinyaiApplyReLU(input, output_relu_simd, size);
    ASSERT(relu_success, "SIMD ReLU failed");
    clock_t end_simd       = clock();
    double  time_relu_simd = (double)(end_simd - start_simd) / CLOCKS_PER_SEC;

    // Verify ReLU results
    for (int i = 0; i < size; i++) {
        ASSERT(fabsf(output_relu_scalar[i] - output_relu_simd[i]) < 0.000001f,
               "SIMD ReLU produced incorrect results");
    }

    printf("    ReLU Scalar time: %.6f seconds\n", time_relu_scalar);
    printf("    ReLU SIMD time: %.6f seconds\n", time_relu_simd);
    printf("    ReLU Speedup: %.2fx\n", time_relu_scalar / time_relu_simd);

    // Sigmoid
    printf("  Testing Sigmoid activation...\n");
    float *output_sigmoid_scalar = (float *)malloc(size * sizeof(float));
    float *output_sigmoid_simd   = (float *)malloc(size * sizeof(float));

    // Scalar Sigmoid
    start_scalar = clock();
    for (int i = 0; i < size; i++) {
        output_sigmoid_scalar[i] = 1.0f / (1.0f + expf(-input[i]));
    }
    end_scalar                 = clock();
    double time_sigmoid_scalar = (double)(end_scalar - start_scalar) / CLOCKS_PER_SEC;

    // SIMD Sigmoid
    start_simd           = clock();
    bool sigmoid_success = tinyaiApplySigmoid(input, output_sigmoid_simd, size);
    ASSERT(sigmoid_success, "SIMD Sigmoid failed");
    end_simd                 = clock();
    double time_sigmoid_simd = (double)(end_simd - start_simd) / CLOCKS_PER_SEC;

    // Verify Sigmoid results with tolerance
    float max_sigmoid_diff = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff       = fabsf(output_sigmoid_scalar[i] - output_sigmoid_simd[i]);
        max_sigmoid_diff = fmaxf(max_sigmoid_diff, diff);
    }

    printf("    Maximum Sigmoid difference: %.6f\n", max_sigmoid_diff);
    ASSERT(max_sigmoid_diff < 0.01f, "SIMD Sigmoid produced incorrect results");

    printf("    Sigmoid Scalar time: %.6f seconds\n", time_sigmoid_scalar);
    printf("    Sigmoid SIMD time: %.6f seconds\n", time_sigmoid_simd);
    printf("    Sigmoid Speedup: %.2fx\n", time_sigmoid_scalar / time_sigmoid_simd);

    // Tanh
    printf("  Testing Tanh activation...\n");
    float *output_tanh_scalar = (float *)malloc(size * sizeof(float));
    float *output_tanh_simd   = (float *)malloc(size * sizeof(float));

    // Scalar Tanh
    start_scalar = clock();
    for (int i = 0; i < size; i++) {
        output_tanh_scalar[i] = tanhf(input[i]);
    }
    end_scalar              = clock();
    double time_tanh_scalar = (double)(end_scalar - start_scalar) / CLOCKS_PER_SEC;

    // SIMD Tanh
    start_simd        = clock();
    bool tanh_success = tinyaiApplyTanh(input, output_tanh_simd, size);
    ASSERT(tanh_success, "SIMD Tanh failed");
    end_simd              = clock();
    double time_tanh_simd = (double)(end_simd - start_simd) / CLOCKS_PER_SEC;

    // Verify Tanh results with tolerance
    float max_tanh_diff = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff    = fabsf(output_tanh_scalar[i] - output_tanh_simd[i]);
        max_tanh_diff = fmaxf(max_tanh_diff, diff);
    }

    printf("    Maximum Tanh difference: %.6f\n", max_tanh_diff);
    ASSERT(max_tanh_diff < 0.01f, "SIMD Tanh produced incorrect results");

    printf("    Tanh Scalar time: %.6f seconds\n", time_tanh_scalar);
    printf("    Tanh SIMD time: %.6f seconds\n", time_tanh_simd);
    printf("    Tanh Speedup: %.2fx\n", time_tanh_scalar / time_tanh_simd);

    // Cleanup
    free(input);
    free(output_relu_scalar);
    free(output_relu_simd);
    free(output_sigmoid_scalar);
    free(output_sigmoid_simd);
    free(output_tanh_scalar);
    free(output_tanh_simd);

    printf("  PASS: Activation function tests\n");
}

/**
 * Detect available SIMD instruction sets
 */
static void test_simd_detection()
{
    printf("Testing SIMD detection...\n");

    // Detect supported instruction sets
    bool has_sse2 = tinyaiHasSSE2();
    bool has_avx  = tinyaiHasAVX();
    bool has_avx2 = tinyaiHasAVX2();

    printf("  SSE2 support: %s\n", has_sse2 ? "Yes" : "No");
    printf("  AVX support: %s\n", has_avx ? "Yes" : "No");
    printf("  AVX2 support: %s\n", has_avx2 ? "Yes" : "No");

    // Verify that the right implementation is selected based on CPU capabilities
    const char *simd_impl = tinyaiGetSIMDImplementation();
    printf("  Selected SIMD implementation: %s\n", simd_impl);

    if (has_avx2) {
        ASSERT(strcmp(simd_impl, "AVX2") == 0, "AVX2 should be selected when available");
    }
    else if (has_avx) {
        ASSERT(strcmp(simd_impl, "AVX") == 0,
               "AVX should be selected when available but AVX2 is not");
    }
    else if (has_sse2) {
        ASSERT(strcmp(simd_impl, "SSE2") == 0,
               "SSE2 should be selected when available but AVX/AVX2 are not");
    }
    else {
        ASSERT(strcmp(simd_impl, "Scalar") == 0,
               "Scalar should be selected when no SIMD is available");
    }

    printf("  PASS: SIMD detection tests\n");
}

/**
 * Run all SIMD acceleration tests
 */
int main()
{
    printf("Running SIMD Acceleration Tests...\n");

    // Seed random number generator
    srand((unsigned)time(NULL));

    // Run tests
    test_simd_detection();
    test_vector_addition();
    test_matrix_multiplication();
    test_convolution();
    test_activation_functions();

    printf("All SIMD Acceleration Tests PASSED\n");
    return 0;
}