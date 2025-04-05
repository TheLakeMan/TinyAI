/**
 * @file test_sparse_ops.c
 * @brief Test for sparse matrix operations
 */

#include "../utils/sparse_ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TEST_ROWS 100
#define TEST_COLS 100
#define SPARSITY 0.9f /* 90% of elements will be zero */
#define THRESHOLD 1e-6f

/* Generate a random sparse matrix in dense format */
static void generateRandomSparseMatrix(float *matrix, int rows, int cols, float sparsity)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            /* Generate random value */
            float val = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

            /* Make most values zero based on sparsity */
            if ((float)rand() / RAND_MAX < sparsity) {
                val = 0.0f;
            }

            matrix[i * cols + j] = val;
        }
    }
}

/* Generate a random vector */
static void generateRandomVector(float *vector, int size)
{
    for (int i = 0; i < size; i++) {
        vector[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

/* Compare two float values with epsilon */
static bool floatEqual(float a, float b, float epsilon) { return fabsf(a - b) < epsilon; }

/* Test CSR matrix conversion */
static bool testCSRMatrixConversion()
{
    printf("Testing CSR matrix conversion...\n");

    /* Create dense matrix */
    float *dense = (float *)malloc(TEST_ROWS * TEST_COLS * sizeof(float));
    if (!dense) {
        printf("Memory allocation failed\n");
        return false;
    }

    /* Generate random sparse matrix */
    generateRandomSparseMatrix(dense, TEST_ROWS, TEST_COLS, SPARSITY);

    /* Convert to CSR format */
    TinyAICSRMatrix *csr = tinyaiCreateCSRMatrixFromDense(dense, TEST_ROWS, TEST_COLS, THRESHOLD);
    if (!csr) {
        printf("Failed to create CSR matrix\n");
        free(dense);
        return false;
    }

    /* Convert back to dense format */
    float *denseCopy = (float *)malloc(TEST_ROWS * TEST_COLS * sizeof(float));
    if (!denseCopy) {
        printf("Memory allocation failed\n");
        tinyaiCSRMatrixFree(csr);
        free(dense);
        return false;
    }

    if (!tinyaiCSRMatrixToDense(csr, denseCopy)) {
        printf("Failed to convert CSR matrix back to dense format\n");
        free(denseCopy);
        tinyaiCSRMatrixFree(csr);
        free(dense);
        return false;
    }

    /* Verify conversion */
    bool success = true;
    for (int i = 0; i < TEST_ROWS; i++) {
        for (int j = 0; j < TEST_COLS; j++) {
            int idx = i * TEST_COLS + j;

            /* Values below threshold should be zero */
            if (fabsf(dense[idx]) < THRESHOLD) {
                if (denseCopy[idx] != 0.0f) {
                    printf("Mismatch at (%d, %d): original = %f, copy = %f\n", i, j, dense[idx],
                           denseCopy[idx]);
                    success = false;
                    break;
                }
            }
            /* Values above threshold should be preserved */
            else if (!floatEqual(dense[idx], denseCopy[idx], THRESHOLD)) {
                printf("Mismatch at (%d, %d): original = %f, copy = %f\n", i, j, dense[idx],
                       denseCopy[idx]);
                success = false;
                break;
            }
        }
        if (!success)
            break;
    }

    /* Calculate compression ratio */
    float compressionRatio = tinyaiCSRMatrixCompressionRatio(csr);
    printf("CSR compression ratio: %.2f\n", compressionRatio);

    /* Clean up */
    free(denseCopy);
    tinyaiCSRMatrixFree(csr);
    free(dense);

    return success;
}

/* Test 4-bit quantized CSR matrix conversion */
static bool test4BitCSRMatrixConversion()
{
    printf("Testing 4-bit quantized CSR matrix conversion...\n");

    /* Create dense matrix */
    float *dense = (float *)malloc(TEST_ROWS * TEST_COLS * sizeof(float));
    if (!dense) {
        printf("Memory allocation failed\n");
        return false;
    }

    /* Generate random sparse matrix */
    generateRandomSparseMatrix(dense, TEST_ROWS, TEST_COLS, SPARSITY);

    /* Convert to 4-bit quantized CSR format */
    TinyAICSRMatrix4Bit *csr =
        tinyaiCreateCSRMatrix4BitFromDense(dense, TEST_ROWS, TEST_COLS, THRESHOLD);
    if (!csr) {
        printf("Failed to create 4-bit quantized CSR matrix\n");
        free(dense);
        return false;
    }

    /* Convert back to dense format */
    float *denseCopy = (float *)malloc(TEST_ROWS * TEST_COLS * sizeof(float));
    if (!denseCopy) {
        printf("Memory allocation failed\n");
        tinyaiCSRMatrix4BitFree(csr);
        free(dense);
        return false;
    }

    if (!tinyaiCSRMatrix4BitToDense(csr, denseCopy)) {
        printf("Failed to convert 4-bit quantized CSR matrix back to dense format\n");
        free(denseCopy);
        tinyaiCSRMatrix4BitFree(csr);
        free(dense);
        return false;
    }

    /* Verify conversion (with higher error tolerance due to quantization) */
    float quantizationError = 0.0f;
    int   nonZeroCount      = 0;
    bool  success           = true;

    for (int i = 0; i < TEST_ROWS; i++) {
        for (int j = 0; j < TEST_COLS; j++) {
            int idx = i * TEST_COLS + j;

            /* Values below threshold should be zero */
            if (fabsf(dense[idx]) < THRESHOLD) {
                if (denseCopy[idx] != 0.0f) {
                    printf("Mismatch at (%d, %d): original = %f, copy = %f\n", i, j, dense[idx],
                           denseCopy[idx]);
                    success = false;
                    break;
                }
            }
            /* Values above threshold should be approximately equal (within quantization error) */
            else {
                float error = fabsf(dense[idx] - denseCopy[idx]);
                quantizationError += error;
                nonZeroCount++;

                /* Check if error is reasonable for 4-bit quantization */
                if (error > csr->scale) { /* Scale represents the maximum quantization step */
                    printf("Large error at (%d, %d): original = %f, copy = %f, error = %f, scale = "
                           "%f\n",
                           i, j, dense[idx], denseCopy[idx], error, csr->scale);
                    success = false;
                    break;
                }
            }
        }
        if (!success)
            break;
    }

    /* Calculate average quantization error */
    if (nonZeroCount > 0) {
        float avgError = quantizationError / nonZeroCount;
        printf("Average quantization error: %f\n", avgError);
    }

    /* Calculate compression ratio */
    float compressionRatio = tinyaiCSRMatrix4BitCompressionRatio(csr);
    printf("4-bit quantized CSR compression ratio: %.2f\n", compressionRatio);

    /* Clean up */
    free(denseCopy);
    tinyaiCSRMatrix4BitFree(csr);
    free(dense);

    return success;
}

/* Test CSR matrix-vector multiplication */
static bool testCSRMatrixVectorMul()
{
    printf("Testing CSR matrix-vector multiplication...\n");

    /* Create dense matrix */
    float *dense = (float *)malloc(TEST_ROWS * TEST_COLS * sizeof(float));
    if (!dense) {
        printf("Memory allocation failed\n");
        return false;
    }

    /* Generate random sparse matrix */
    generateRandomSparseMatrix(dense, TEST_ROWS, TEST_COLS, SPARSITY);

    /* Create input vector */
    float *x = (float *)malloc(TEST_COLS * sizeof(float));
    if (!x) {
        printf("Memory allocation failed\n");
        free(dense);
        return false;
    }

    /* Generate random input vector */
    generateRandomVector(x, TEST_COLS);

    /* Convert to CSR format */
    TinyAICSRMatrix *csr = tinyaiCreateCSRMatrixFromDense(dense, TEST_ROWS, TEST_COLS, THRESHOLD);
    if (!csr) {
        printf("Failed to create CSR matrix\n");
        free(x);
        free(dense);
        return false;
    }

    /* Create output vectors */
    float *y1 = (float *)malloc(TEST_ROWS * sizeof(float));
    float *y2 = (float *)malloc(TEST_ROWS * sizeof(float));
    if (!y1 || !y2) {
        printf("Memory allocation failed\n");
        tinyaiCSRMatrixFree(csr);
        free(x);
        free(dense);
        if (y1)
            free(y1);
        return false;
    }

    /* Calculate reference result using dense matrix multiplication */
    for (int i = 0; i < TEST_ROWS; i++) {
        y1[i] = 0.0f;
        for (int j = 0; j < TEST_COLS; j++) {
            if (fabsf(dense[i * TEST_COLS + j]) >= THRESHOLD) {
                y1[i] += dense[i * TEST_COLS + j] * x[j];
            }
        }
    }

    /* Calculate result using CSR matrix-vector multiplication */
    if (!tinyaiCSRMatrixVectorMul(csr, x, y2)) {
        printf("Failed to perform CSR matrix-vector multiplication\n");
        free(y2);
        free(y1);
        tinyaiCSRMatrixFree(csr);
        free(x);
        free(dense);
        return false;
    }

    /* Calculate mean squared error */
    float mse = 0.0f;
    for (int i = 0; i < TEST_ROWS; i++) {
        float error = y1[i] - y2[i];
        mse += error * error;
    }
    mse /= TEST_ROWS;

    printf("Mean squared error: %g\n", mse);

    /* Verify results */
    bool success = true;
    for (int i = 0; i < TEST_ROWS; i++) {
        if (!floatEqual(y1[i], y2[i], THRESHOLD * 10)) {
            printf("Mismatch at index %d: reference = %f, CSR = %f\n", i, y1[i], y2[i]);
            success = false;
            break;
        }
    }

    /* Test SIMD version if available */
    float *y3 = (float *)malloc(TEST_ROWS * sizeof(float));
    if (y3) {
        /* Calculate result using SIMD-accelerated CSR matrix-vector multiplication */
        if (tinyaiCSRMatrixVectorMulSIMD(csr, x, y3)) {
            /* Calculate mean squared error */
            float mse_simd = 0.0f;
            for (int i = 0; i < TEST_ROWS; i++) {
                float error = y1[i] - y3[i];
                mse_simd += error * error;
            }
            mse_simd /= TEST_ROWS;

            printf("Mean squared error (SIMD): %g\n", mse_simd);

            /* Verify results */
            for (int i = 0; i < TEST_ROWS; i++) {
                if (!floatEqual(y1[i], y3[i], THRESHOLD * 10)) {
                    printf("SIMD mismatch at index %d: reference = %f, CSR SIMD = %f\n", i, y1[i],
                           y3[i]);
                    success = false;
                    break;
                }
            }
        }
        free(y3);
    }

    /* Clean up */
    free(y2);
    free(y1);
    tinyaiCSRMatrixFree(csr);
    free(x);
    free(dense);

    return success;
}

/* Test 4-bit quantized CSR matrix-vector multiplication */
static bool test4BitCSRMatrixVectorMul()
{
    printf("Testing 4-bit quantized CSR matrix-vector multiplication...\n");

    /* Create dense matrix */
    float *dense = (float *)malloc(TEST_ROWS * TEST_COLS * sizeof(float));
    if (!dense) {
        printf("Memory allocation failed\n");
        return false;
    }

    /* Generate random sparse matrix */
    generateRandomSparseMatrix(dense, TEST_ROWS, TEST_COLS, SPARSITY);

    /* Create input vector */
    float *x = (float *)malloc(TEST_COLS * sizeof(float));
    if (!x) {
        printf("Memory allocation failed\n");
        free(dense);
        return false;
    }

    /* Generate random input vector */
    generateRandomVector(x, TEST_COLS);

    /* Convert to 4-bit quantized CSR format */
    TinyAICSRMatrix4Bit *csr =
        tinyaiCreateCSRMatrix4BitFromDense(dense, TEST_ROWS, TEST_COLS, THRESHOLD);
    if (!csr) {
        printf("Failed to create 4-bit quantized CSR matrix\n");
        free(x);
        free(dense);
        return false;
    }

    /* Create output vectors */
    float *y1 = (float *)malloc(TEST_ROWS * sizeof(float));
    float *y2 = (float *)malloc(TEST_ROWS * sizeof(float));
    if (!y1 || !y2) {
        printf("Memory allocation failed\n");
        tinyaiCSRMatrix4BitFree(csr);
        free(x);
        free(dense);
        if (y1)
            free(y1);
        return false;
    }

    /* Calculate reference result using dense matrix multiplication */
    for (int i = 0; i < TEST_ROWS; i++) {
        y1[i] = 0.0f;
        for (int j = 0; j < TEST_COLS; j++) {
            if (fabsf(dense[i * TEST_COLS + j]) >= THRESHOLD) {
                y1[i] += dense[i * TEST_COLS + j] * x[j];
            }
        }
    }

    /* Calculate result using 4-bit quantized CSR matrix-vector multiplication */
    if (!tinyaiCSRMatrix4BitVectorMul(csr, x, y2)) {
        printf("Failed to perform 4-bit quantized CSR matrix-vector multiplication\n");
        free(y2);
        free(y1);
        tinyaiCSRMatrix4BitFree(csr);
        free(x);
        free(dense);
        return false;
    }

    /* Calculate mean squared error */
    float mse = 0.0f;
    for (int i = 0; i < TEST_ROWS; i++) {
        float error = y1[i] - y2[i];
        mse += error * error;
    }
    mse /= TEST_ROWS;

    printf("Mean squared error (4-bit quantized): %g\n", mse);

    /* Verify results (with higher tolerance due to quantization) */
    float errorTolerance = csr->scale * 2.0f; /* Allow for quantization error */
    bool  success        = true;
    for (int i = 0; i < TEST_ROWS; i++) {
        if (fabsf(y1[i] - y2[i]) > errorTolerance) {
            printf("Mismatch at index %d: reference = %f, 4-bit quantized CSR = %f\n", i, y1[i],
                   y2[i]);
            success = false;
            break;
        }
    }

    /* Test SIMD version if available */
    float *y3 = (float *)malloc(TEST_ROWS * sizeof(float));
    if (y3) {
        /* Calculate result using SIMD-accelerated 4-bit quantized CSR matrix-vector multiplication
         */
        if (tinyaiCSRMatrix4BitVectorMulSIMD(csr, x, y3)) {
            /* Calculate mean squared error */
            float mse_simd = 0.0f;
            for (int i = 0; i < TEST_ROWS; i++) {
                float error = y1[i] - y3[i];
                mse_simd += error * error;
            }
            mse_simd /= TEST_ROWS;

            printf("Mean squared error (4-bit quantized SIMD): %g\n", mse_simd);

            /* Verify results */
            for (int i = 0; i < TEST_ROWS; i++) {
                if (fabsf(y1[i] - y3[i]) > errorTolerance) {
                    printf("SIMD mismatch at index %d: reference = %f, 4-bit quantized CSR SIMD = "
                           "%f\n",
                           i, y1[i], y3[i]);
                    success = false;
                    break;
                }
            }
        }
        free(y3);
    }

    /* Clean up */
    free(y2);
    free(y1);
    tinyaiCSRMatrix4BitFree(csr);
    free(x);
    free(dense);

    return success;
}

int main(int argc, char **argv)
{
    /* Seed random number generator */
    srand(time(NULL));

    /* Run tests */
    bool success = true;

    printf("=== Sparse Matrix Operations Tests ===\n");

    if (!testCSRMatrixConversion()) {
        printf("CSR matrix conversion test failed\n");
        success = false;
    }
    else {
        printf("CSR matrix conversion test passed\n");
    }

    if (!test4BitCSRMatrixConversion()) {
        printf("4-bit quantized CSR matrix conversion test failed\n");
        success = false;
    }
    else {
        printf("4-bit quantized CSR matrix conversion test passed\n");
    }

    if (!testCSRMatrixVectorMul()) {
        printf("CSR matrix-vector multiplication test failed\n");
        success = false;
    }
    else {
        printf("CSR matrix-vector multiplication test passed\n");
    }

    if (!test4BitCSRMatrixVectorMul()) {
        printf("4-bit quantized CSR matrix-vector multiplication test failed\n");
        success = false;
    }
    else {
        printf("4-bit quantized CSR matrix-vector multiplication test passed\n");
    }

    if (success) {
        printf("All sparse matrix operation tests passed!\n");
        return 0;
    }
    else {
        printf("Some sparse matrix operation tests failed\n");
        return 1;
    }
}
