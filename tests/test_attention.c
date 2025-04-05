.

'ef'f'f//**
 * TinyAI - SIMD-accelerated Attention Mechanism Tests
 *
 * This file contains tests for the SIMD-accelerated attention mechanism
 * implementation in models/text/attention.c, verifying both correctness
 * and performance improvements.
 */

#include "../core/memory.h"
#include "../models/text/attention.h"
#include "../utils/benchmark.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Define error threshold for floating point comparisons
#define EPSILON 1e-4f

// Test attention score computation
static int test_attention_scores()
{
    printf("    Testing attention score computation...\n");

    // Create test data for query, key matrices
    const int batch_size = 2;
    const int seq_len    = 16;
    const int head_dim   = 64;

    // Allocate and initialize data
    float *query       = (float *)malloc(batch_size * seq_len * head_dim * sizeof(float));
    float *key         = (float *)malloc(batch_size * seq_len * head_dim * sizeof(float));
    float *scores_simd = (float *)malloc(batch_size * seq_len * seq_len * sizeof(float));
    float *scores_ref  = (float *)malloc(batch_size * seq_len * seq_len * sizeof(float));

    // Initialize with some test data
    for (int i = 0; i < batch_size * seq_len * head_dim; i++) {
        query[i] = ((float)rand() / RAND_MAX) * 0.1f;
        key[i]   = ((float)rand() / RAND_MAX) * 0.1f;
    }

    // Compute scores using SIMD implementation
    attention_compute_scores_simd(query, key, scores_simd, batch_size, seq_len, head_dim);

    // Compute scores using reference implementation
    attention_compute_scores_reference(query, key, scores_ref, batch_size, seq_len, head_dim);

    // Verify results
    int errors = 0;
    for (int i = 0; i < batch_size * seq_len * seq_len; i++) {
        if (fabsf(scores_simd[i] - scores_ref[i]) > EPSILON) {
            errors++;
            if (errors < 10) { // Print first few errors
                printf("      Error at index %d: SIMD = %f, Reference = %f\n", i, scores_simd[i],
                       scores_ref[i]);
            }
        }
    }

    if (errors > 0) {
        printf("      FAILED: %d error(s) found in attention score computation\n", errors);
    }
    else {
        printf("      PASSED: Attention score computation verified\n");
    }

    // Cleanup
    free(query);
    free(key);
    free(scores_simd);
    free(scores_ref);

    return (errors == 0) ? 1 : 0;
}

// Test softmax computation
static int test_softmax()
{
    printf("    Testing softmax computation...\n");

    // Create test data
    const int batch_size = 2;
    const int seq_len    = 32;

    // Allocate memory
    float *scores     = (float *)malloc(batch_size * seq_len * seq_len * sizeof(float));
    float *probs_simd = (float *)malloc(batch_size * seq_len * seq_len * sizeof(float));
    float *probs_ref  = (float *)malloc(batch_size * seq_len * seq_len * sizeof(float));

    // Initialize scores with random values
    for (int i = 0; i < batch_size * seq_len * seq_len; i++) {
        scores[i] = ((float)rand() / RAND_MAX) * 4.0f - 2.0f; // Range [-2, 2]
    }

    // Apply softmax using SIMD
    attention_softmax_simd(scores, probs_simd, batch_size, seq_len);

    // Apply softmax using reference implementation
    attention_softmax_reference(scores, probs_ref, batch_size, seq_len);

    // Verify results
    int errors = 0;
    for (int i = 0; i < batch_size * seq_len * seq_len; i++) {
        if (fabsf(probs_simd[i] - probs_ref[i]) > EPSILON) {
            errors++;
            if (errors < 10) {
                printf("      Error at index %d: SIMD = %f, Reference = %f\n", i, probs_simd[i],
                       probs_ref[i]);
            }
        }
    }

    if (errors > 0) {
        printf("      FAILED: %d error(s) found in softmax computation\n", errors);
    }
    else {
        printf("      PASSED: Softmax computation verified\n");
    }

    // Verify row sum = 1.0
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < seq_len; i++) {
            float sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                sum += probs_simd[b * seq_len * seq_len + i * seq_len + j];
            }
            if (fabsf(sum - 1.0f) > EPSILON) {
                errors++;
                printf("      Error: Row sum = %f (should be 1.0)\n", sum);
            }
        }
    }

    if (errors > 0) {
        printf("      FAILED: Row sums are not normalized to 1.0\n");
    }
    else {
        printf("      PASSED: All row sums correctly normalize to 1.0\n");
    }

    // Cleanup
    free(scores);
    free(probs_simd);
    free(probs_ref);

    return (errors == 0) ? 1 : 0;
}

// Test weighted sum computation
static int test_weighted_sum()
{
    printf("    Testing weighted sum computation...\n");

    // Create test data
    const int batch_size = 2;
    const int seq_len    = 16;
    const int head_dim   = 64;

    // Allocate memory
    float *probs       = (float *)malloc(batch_size * seq_len * seq_len * sizeof(float));
    float *value       = (float *)malloc(batch_size * seq_len * head_dim * sizeof(float));
    float *output_simd = (float *)malloc(batch_size * seq_len * head_dim * sizeof(float));
    float *output_ref  = (float *)malloc(batch_size * seq_len * head_dim * sizeof(float));

    // Initialize with test data
    for (int i = 0; i < batch_size * seq_len * seq_len; i++) {
        probs[i] = ((float)rand() / RAND_MAX) / seq_len; // Ensure row sum close to 1
    }

    // Normalize rows to ensure they sum to 1
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < seq_len; i++) {
            float sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                sum += probs[b * seq_len * seq_len + i * seq_len + j];
            }
            for (int j = 0; j < seq_len; j++) {
                probs[b * seq_len * seq_len + i * seq_len + j] /= sum;
            }
        }
    }

    for (int i = 0; i < batch_size * seq_len * head_dim; i++) {
        value[i] = ((float)rand() / RAND_MAX) * 0.1f;
    }

    // Compute weighted sum using SIMD
    attention_weighted_sum_simd(probs, value, output_simd, batch_size, seq_len, head_dim);

    // Compute weighted sum using reference implementation
    attention_weighted_sum_reference(probs, value, output_ref, batch_size, seq_len, head_dim);

    // Verify results
    int errors = 0;
    for (int i = 0; i < batch_size * seq_len * head_dim; i++) {
        if (fabsf(output_simd[i] - output_ref[i]) > EPSILON) {
            errors++;
            if (errors < 10) {
                printf("      Error at index %d: SIMD = %f, Reference = %f\n", i, output_simd[i],
                       output_ref[i]);
            }
        }
    }

    if (errors > 0) {
        printf("      FAILED: %d error(s) found in weighted sum computation\n", errors);
    }
    else {
        printf("      PASSED: Weighted sum computation verified\n");
    }

    // Cleanup
    free(probs);
    free(value);
    free(output_simd);
    free(output_ref);

    return (errors == 0) ? 1 : 0;
}

// Test full attention mechanism
static int test_full_attention()
{
    printf("    Testing full attention mechanism...\n");

    // Create test data
    const int batch_size = 1;
    const int seq_len    = 32;
    const int head_dim   = 64;

    // Allocate memory
    float *query       = (float *)malloc(batch_size * seq_len * head_dim * sizeof(float));
    float *key         = (float *)malloc(batch_size * seq_len * head_dim * sizeof(float));
    float *value       = (float *)malloc(batch_size * seq_len * head_dim * sizeof(float));
    float *output_simd = (float *)malloc(batch_size * seq_len * head_dim * sizeof(float));
    float *output_ref  = (float *)malloc(batch_size * seq_len * head_dim * sizeof(float));

    // Initialize with test data
    for (int i = 0; i < batch_size * seq_len * head_dim; i++) {
        query[i] = ((float)rand() / RAND_MAX) * 0.1f;
        key[i]   = ((float)rand() / RAND_MAX) * 0.1f;
        value[i] = ((float)rand() / RAND_MAX) * 0.1f;
    }

    // Perform full attention using SIMD
    attention_forward_simd(query, key, value, output_simd, batch_size, seq_len, head_dim);

    // Perform full attention using reference implementation
    attention_forward_reference(query, key, value, output_ref, batch_size, seq_len, head_dim);

    // Verify results
    int errors = 0;
    for (int i = 0; i < batch_size * seq_len * head_dim; i++) {
        if (fabsf(output_simd[i] - output_ref[i]) > EPSILON) {
            errors++;
            if (errors < 10) {
                printf("      Error at index %d: SIMD = %f, Reference = %f\n", i, output_simd[i],
                       output_ref[i]);
            }
        }
    }

    if (errors > 0) {
        printf("      FAILED: %d error(s) found in full attention computation\n", errors);
    }
    else {
        printf("      PASSED: Full attention computation verified\n");
    }

    // Cleanup
    free(query);
    free(key);
    free(value);
    free(output_simd);
    free(output_ref);

    return (errors == 0) ? 1 : 0;
}

// Test performance of SIMD vs reference implementation
static void test_performance()
{
    printf("    Testing attention performance...\n");

    // Configure test
    const int batch_size = 1;
    const int seq_len    = 128;
    const int head_dim   = 64;

    // Allocate memory
    float *query       = (float *)malloc(batch_size * seq_len * head_dim * sizeof(float));
    float *key         = (float *)malloc(batch_size * seq_len * head_dim * sizeof(float));
    float *value       = (float *)malloc(batch_size * seq_len * head_dim * sizeof(float));
    float *output_simd = (float *)malloc(batch_size * seq_len * head_dim * sizeof(float));
    float *output_ref  = (float *)malloc(batch_size * seq_len * head_dim * sizeof(float));

    // Initialize with test data
    for (int i = 0; i < batch_size * seq_len * head_dim; i++) {
        query[i] = ((float)rand() / RAND_MAX) * 0.1f;
        key[i]   = ((float)rand() / RAND_MAX) * 0.1f;
        value[i] = ((float)rand() / RAND_MAX) * 0.1f;
    }

    // Benchmark SIMD implementation
    uint64_t start_simd = benchmark_start();

    // Run multiple iterations for more accurate timing
    for (int i = 0; i < 10; i++) {
        attention_forward_simd(query, key, value, output_simd, batch_size, seq_len, head_dim);
    }

    uint64_t duration_simd = benchmark_end(start_simd);

    // Benchmark reference implementation
    uint64_t start_ref = benchmark_start();

    // Run multiple iterations
    for (int i = 0; i < 10; i++) {
        attention_forward_reference(query, key, value, output_ref, batch_size, seq_len, head_dim);
    }

    uint64_t duration_ref = benchmark_end(start_ref);

    // Report results
    printf("      SIMD implementation: %llu microseconds (10 iterations)\n", duration_simd);
    printf("      Reference implementation: %llu microseconds (10 iterations)\n", duration_ref);
    printf("      Speed improvement: %.2fx\n", (float)duration_ref / duration_simd);

    // Cleanup
    free(query);
    free(key);
    free(value);
    free(output_simd);
    free(output_ref);
}

// Run all attention tests
void run_attention_tests()
{
    printf("Running SIMD-accelerated Attention Mechanism Tests...\n");

    // Initialize random seed
    srand((unsigned int)time(NULL));

    // Run tests
    int score_test          = test_attention_scores();
    int softmax_test        = test_softmax();
    int weighted_sum_test   = test_weighted_sum();
    int full_attention_test = test_full_attention();

    // Run performance test
    test_performance();

    // Summarize results
    printf("\n  Attention Test Results:\n");
    printf("    Attention Score Computation: %s\n", score_test ? "PASSED" : "FAILED");
    printf("    Softmax Computation: %s\n", softmax_test ? "PASSED" : "FAILED");
    printf("    Weighted Sum Computation: %s\n", weighted_sum_test ? "PASSED" : "FAILED");
    printf("    Full Attention Mechanism: %s\n", full_attention_test ? "PASSED" : "FAILED");

    int total_tests  = 4;
    int passed_tests = score_test + softmax_test + weighted_sum_test + full_attention_test;

    printf("\n  Summary: %d/%d tests passed\n", passed_tests, total_tests);
}

// Standalone test entry point (when compiled separately)
#ifdef TEST_ATTENTION_STANDALONE
int main()
{
    run_attention_tests();
    return 0;
}
#endif
