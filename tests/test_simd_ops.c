/**
 * TinyAI SIMD Operations Tests
 */

#include "../utils/simd_ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Basic assertion helper (consistent with other test files)
#define ASSERT(condition, message)                                                                 \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            fprintf(stderr, "Assertion Failed: %s (%s:%d)\n", message, __FILE__, __LINE__);        \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

// Test SIMD availability detection
void test_simd_availability()
{
    printf("  Testing SIMD availability detection...\n");

    bool available = tinyaiSimdAvailable();
    printf("    SIMD available: %s\n", available ? "YES" : "NO");

    // We don't actually fail the test if SIMD is not available,
    // as the code should work on any platform
    printf("    PASS\n");
}

// Helper function to initialize a matrix with random values
void init_random_matrix(float *matrix, int size)
{
    for (int i = 0; i < size; i++) {
        matrix[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // -1.0 to 1.0
    }
}

// Helper function to quantize a float matrix to 4-bit
void quantize_matrix(const float *input, uint8_t *output, int size, float *scale_factors)
{
    tinyaiSimdQuantize4Bit(output, input, size, scale_factors, 256);
}

// Helper function to compare two float arrays with epsilon
bool compare_float_arrays(const float *a, const float *b, int size, float epsilon)
{
    for (int i = 0; i < size; i++) {
        if (fabsf(a[i] - b[i]) > epsilon) {
            printf("    Mismatch at index %d: %f vs %f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

// Test matrix-vector multiplication
void test_matrix_vector_multiplication()
{
    printf("  Testing matrix-vector multiplication with 4-bit weights...\n");

    // Test dimensions
    const int rows = 128;
    const int cols = 256;

    // Allocate matrices and vectors
    float *matrix      = (float *)malloc(rows * cols * sizeof(float));
    float *vector      = (float *)malloc(cols * sizeof(float));
    float *result_ref  = (float *)malloc(rows * sizeof(float));
    float *result_simd = (float *)malloc(rows * sizeof(float));

    // Calculate quantized matrix size (4-bit per value, packed)
    int      quantized_size   = (rows * cols + 1) / 2; // Each byte holds 2 values
    uint8_t *quantized_matrix = (uint8_t *)malloc(quantized_size);
    float   *scale_factors    = (float *)malloc((rows * cols / 256 + 1) * sizeof(float));

    // Initialize matrices with random values
    init_random_matrix(matrix, rows * cols);
    init_random_matrix(vector, cols);

    // Quantize the matrix
    quantize_matrix(matrix, quantized_matrix, rows * cols, scale_factors);

    // Compute reference result (naive implementation)
    for (int r = 0; r < rows; r++) {
        result_ref[r] = 0.0f;
        for (int c = 0; c < cols; c++) {
            result_ref[r] += matrix[r * cols + c] * vector[c];
        }
    }

    // Compute result using SIMD operations
    tinyaiSimdMatMul4Bit(result_simd, quantized_matrix, vector, rows, cols, scale_factors);

    // Compare results - allow some difference due to quantization
    bool match = compare_float_arrays(result_ref, result_simd, rows, 0.1f);

    // Free memory
    free(matrix);
    free(vector);
    free(result_ref);
    free(result_simd);
    free(quantized_matrix);
    free(scale_factors);

    ASSERT(
        match,
        "SIMD matrix-vector multiplication should match reference implementation within tolerance");
    printf("    PASS\n");
}

// Test vector addition
void test_vector_addition()
{
    printf("  Testing vector addition...\n");

    const int size = 1024;

    // Allocate vectors
    float *vec_a       = (float *)malloc(size * sizeof(float));
    float *vec_b       = (float *)malloc(size * sizeof(float));
    float *result_ref  = (float *)malloc(size * sizeof(float));
    float *result_simd = (float *)malloc(size * sizeof(float));

    // Initialize with random values
    init_random_matrix(vec_a, size);
    init_random_matrix(vec_b, size);

    // Compute reference result
    for (int i = 0; i < size; i++) {
        result_ref[i] = vec_a[i] + vec_b[i];
    }

    // Compute SIMD result
    tinyaiSimdVecAdd(result_simd, vec_a, vec_b, size);

    // Compare results
    bool match = compare_float_arrays(result_ref, result_simd, size, 0.00001f);

    // Free memory
    free(vec_a);
    free(vec_b);
    free(result_ref);
    free(result_simd);

    ASSERT(match, "SIMD vector addition should match reference implementation");
    printf("    PASS\n");
}

// Test activation functions
void test_activation_functions()
{
    printf("  Testing activation functions...\n");

    const int size = 1024;

    // Allocate vectors
    float *input_relu         = (float *)malloc(size * sizeof(float));
    float *result_relu_ref    = (float *)malloc(size * sizeof(float));
    float *input_gelu         = (float *)malloc(size * sizeof(float));
    float *result_gelu_ref    = (float *)malloc(size * sizeof(float));
    float *input_sigmoid      = (float *)malloc(size * sizeof(float));
    float *result_sigmoid_ref = (float *)malloc(size * sizeof(float));

    // Initialize with random values in a reasonable range
    for (int i = 0; i < size; i++) {
        float val        = ((float)rand() / RAND_MAX) * 10.0f - 5.0f; // -5.0 to 5.0
        input_relu[i]    = val;
        input_gelu[i]    = val;
        input_sigmoid[i] = val;
    }

    // Copy for reference calculation
    memcpy(result_relu_ref, input_relu, size * sizeof(float));
    memcpy(result_gelu_ref, input_gelu, size * sizeof(float));
    memcpy(result_sigmoid_ref, input_sigmoid, size * sizeof(float));

    // Compute reference results
    for (int i = 0; i < size; i++) {
        // ReLU
        result_relu_ref[i] = result_relu_ref[i] < 0.0f ? 0.0f : result_relu_ref[i];

        // GELU approximate implementation
        float       x             = result_gelu_ref[i];
        const float sqrt2_over_pi = 0.7978845608f;
        float       x3            = x * x * x;
        float       inner         = sqrt2_over_pi * (x + 0.044715f * x3);
        result_gelu_ref[i]        = 0.5f * x * (1.0f + tanhf(inner));

        // Sigmoid
        result_sigmoid_ref[i] = 1.0f / (1.0f + expf(-result_sigmoid_ref[i]));
    }

    // Compute SIMD results
    tinyaiSimdActivate(input_relu, size, 0);    // ReLU
    tinyaiSimdActivate(input_gelu, size, 1);    // GELU
    tinyaiSimdActivate(input_sigmoid, size, 2); // Sigmoid

    // Compare results
    bool relu_match    = compare_float_arrays(result_relu_ref, input_relu, size, 0.00001f);
    bool gelu_match    = compare_float_arrays(result_gelu_ref, input_gelu, size,
                                              0.001f); // Higher tolerance for GELU
    bool sigmoid_match = compare_float_arrays(result_sigmoid_ref, input_sigmoid, size,
                                              0.001f); // Higher tolerance for Sigmoid

    // Free memory
    free(input_relu);
    free(result_relu_ref);
    free(input_gelu);
    free(result_gelu_ref);
    free(input_sigmoid);
    free(result_sigmoid_ref);

    ASSERT(relu_match, "SIMD ReLU activation should match reference implementation");
    ASSERT(gelu_match, "SIMD GELU activation should match reference implementation");
    ASSERT(sigmoid_match, "SIMD Sigmoid activation should match reference implementation");
    printf("    PASS\n");
}

// Test quantization and dequantization
void test_quantization()
{
    printf("  Testing 4-bit quantization and dequantization...\n");

    const int size = 1024;

    // Allocate memory
    float   *original      = (float *)malloc(size * sizeof(float));
    float   *reconstructed = (float *)malloc(size * sizeof(float));
    uint8_t *quantized     = (uint8_t *)malloc((size + 1) / 2); // 4-bit packed
    float   *scale_factors = (float *)malloc((size / 256 + 1) * sizeof(float));

    // Initialize with random values
    init_random_matrix(original, size);

    // Quantize
    tinyaiSimdQuantize4Bit(quantized, original, size, scale_factors, 256);

    // Dequantize
    tinyaiSimdDequantize4Bit(reconstructed, quantized, size, scale_factors);

    // Check reconstruction quality
    float max_error = 0.0f;
    float avg_error = 0.0f;

    for (int i = 0; i < size; i++) {
        float error = fabsf(original[i] - reconstructed[i]);
        max_error   = fmaxf(max_error, error);
        avg_error += error;
    }
    avg_error /= size;

    printf("    Max error: %f, Avg error: %f\n", max_error, avg_error);

    // Free memory
    free(original);
    free(reconstructed);
    free(quantized);
    free(scale_factors);

    // Quantization will introduce errors, we just want to make sure they're reasonable
    ASSERT(max_error < 0.2f, "Maximum quantization error should be bounded");
    ASSERT(avg_error < 0.05f, "Average quantization error should be small");
    printf("    PASS\n");
}

// Test performance benchmarking
void test_performance_benchmarking()
{
    printf("  Performance benchmarking of SIMD operations...\n");

    // Test dimensions for performance measurement
    const int rows = 1024;
    const int cols = 1024;

    // Allocate matrices and vectors
    float *matrix = (float *)malloc(rows * cols * sizeof(float));
    float *vector = (float *)malloc(cols * sizeof(float));
    float *result = (float *)malloc(rows * sizeof(float));

    // Calculate quantized matrix size
    int      quantized_size   = (rows * cols + 1) / 2; // 4-bit packed
    uint8_t *quantized_matrix = (uint8_t *)malloc(quantized_size);
    float   *scale_factors    = (float *)malloc((rows * cols / 256 + 1) * sizeof(float));

    // Initialize matrices with random values
    init_random_matrix(matrix, rows * cols);
    init_random_matrix(vector, cols);

    // Quantize the matrix
    clock_t start = clock();
    quantize_matrix(matrix, quantized_matrix, rows * cols, scale_factors);
    clock_t end           = clock();
    float   quantize_time = (float)(end - start) / CLOCKS_PER_SEC * 1000.0f; // milliseconds

    // Benchmark matrix-vector multiplication
    start = clock();
    tinyaiSimdMatMul4Bit(result, quantized_matrix, vector, rows, cols, scale_factors);
    end               = clock();
    float matmul_time = (float)(end - start) / CLOCKS_PER_SEC * 1000.0f; // milliseconds

    // Benchmark vector addition (use result and vector to create a new vector)
    start = clock();
    tinyaiSimdVecAdd(result, result, vector, cols); // Only use the first cols elements
    end               = clock();
    float vecadd_time = (float)(end - start) / CLOCKS_PER_SEC * 1000.0f; // milliseconds

    // Benchmark ReLU activation
    start = clock();
    tinyaiSimdActivate(result, rows, 0); // ReLU
    end             = clock();
    float relu_time = (float)(end - start) / CLOCKS_PER_SEC * 1000.0f; // milliseconds

    // Print performance results
    printf("    Performance Results:\n");
    printf("    - Quantization (%d elements): %.3f ms\n", rows * cols, quantize_time);
    printf("    - Matrix-Vector Multiply (%dx%d): %.3f ms\n", rows, cols, matmul_time);
    printf("    - Vector Addition (%d elements): %.3f ms\n", cols, vecadd_time);
    printf("    - ReLU Activation (%d elements): %.3f ms\n", rows, relu_time);

    // Calculate theoretical FLOPS (Floating Point Operations Per Second)
    float matmul_flops  = (float)rows * cols * 2; // Multiply + Add
    float matmul_gflops = matmul_flops / (matmul_time / 1000.0f) / 1e9f;
    printf("    - Matrix-Vector Multiply Performance: %.2f GFLOPS\n", matmul_gflops);

    // Free memory
    free(matrix);
    free(vector);
    free(result);
    free(quantized_matrix);
    free(scale_factors);

    printf("    PASS\n");
}

// Function to be called by test_main.c
void run_simd_ops_tests()
{
    printf("--- Running SIMD Operations Tests ---\n");

    // Set random seed for reproducibility
    srand(42);

    test_simd_availability();
    test_matrix_vector_multiplication();
    test_vector_addition();
    test_activation_functions();
    test_quantization();
    test_performance_benchmarking();

    printf("--- SIMD Operations Tests Finished ---\n");
}
