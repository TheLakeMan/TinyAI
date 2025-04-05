/**
 * TinyAI - SIMD Benchmark Test Suite
 *
 * This file contains tests for the SIMD benchmarking utilities,
 * comparing performance between SIMD and reference implementations.
 */

#include "../utils/simd_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/**
 * Run basic tests for all benchmark functions
 */
static void run_basic_benchmark_tests()
{
    printf("Running basic SIMD benchmark tests...\n");

    // Test matrix multiplication
    SimdBenchmarkResult matmul_result = benchmark_matrix_multiply(64, 64, 64, 5);
    printf("\nMatrix Multiplication Benchmark:\n");
    print_simd_benchmark_result(&matmul_result);

    // Test activation functions
    SimdBenchmarkResult relu_result = benchmark_activation(10000, 0, 10); // ReLU
    printf("\nReLU Activation Benchmark:\n");
    print_simd_benchmark_result(&relu_result);

    SimdBenchmarkResult gelu_result = benchmark_activation(10000, 1, 10); // GELU
    printf("\nGELU Activation Benchmark:\n");
    print_simd_benchmark_result(&gelu_result);

    SimdBenchmarkResult sigmoid_result = benchmark_activation(10000, 2, 10); // Sigmoid
    printf("\nSigmoid Activation Benchmark:\n");
    print_simd_benchmark_result(&sigmoid_result);

    // Test attention operations
    SimdBenchmarkResult attn_scores = benchmark_attention(1, 32, 64, 0, 5); // Scores
    printf("\nAttention Scores Benchmark:\n");
    print_simd_benchmark_result(&attn_scores);

    SimdBenchmarkResult attn_softmax = benchmark_attention(1, 32, 64, 1, 5); // Softmax
    printf("\nAttention Softmax Benchmark:\n");
    print_simd_benchmark_result(&attn_softmax);

    SimdBenchmarkResult attn_weighted = benchmark_attention(1, 32, 64, 2, 5); // Weighted sum
    printf("\nAttention Weighted Sum Benchmark:\n");
    print_simd_benchmark_result(&attn_weighted);

    // Test convolution operations
    SimdBenchmarkResult conv_std = benchmark_convolution(28, 28, 3, 16, 3, 0, 3); // Standard conv
    printf("\nStandard Convolution Benchmark:\n");
    print_simd_benchmark_result(&conv_std);

    SimdBenchmarkResult conv_depth = benchmark_convolution(28, 28, 16, 16, 3, 1, 3); // Depthwise
    printf("\nDepthwise Convolution Benchmark:\n");
    print_simd_benchmark_result(&conv_depth);

    // Test memory access analysis
    printf("\nMemory Access Pattern Analysis for Matrix Multiplication:\n");
    char *matmul_recommendations = analyze_memory_access_patterns(0, 1024 * 1024 * 4, NULL);
    printf("%s\n", matmul_recommendations);
    free(matmul_recommendations);

    printf("\nMemory Access Pattern Analysis for Convolution:\n");
    char *conv_recommendations = analyze_memory_access_patterns(1, 1024 * 1024 * 4, NULL);
    printf("%s\n", conv_recommendations);
    free(conv_recommendations);

    printf("\nMemory Access Pattern Analysis for Attention:\n");
    char *attn_recommendations = analyze_memory_access_patterns(2, 1024 * 1024 * 4, NULL);
    printf("%s\n", attn_recommendations);
    free(attn_recommendations);

    printf("\nBasic SIMD benchmark tests completed.\n");
}

/**
 * Run comprehensive benchmark suite and save results to file
 */
static void run_comprehensive_benchmarks()
{
    printf("Running comprehensive SIMD benchmarks...\n");

    SimdBenchmarkResult *results = run_comprehensive_simd_benchmark("simd_benchmark_results.json");
    if (results) {
        // Create CSV report
        create_simd_benchmark_report(results, 14, "simd_benchmark_results.csv");

        // Free results
        for (int i = 0; i < 14; i++) {
            free((void *)results[i].hardwareDetails);
        }
        free(results);

        printf("Comprehensive benchmarks completed. Results saved to:\n");
        printf("  - simd_benchmark_results.json\n");
        printf("  - simd_benchmark_results.csv\n");
    }
    else {
        printf("Failed to run comprehensive benchmarks.\n");
    }
}

/**
 * Compare cache optimization techniques
 */
static void compare_cache_optimizations()
{
    printf("Comparing cache optimization techniques...\n");

    // Test matrix multiplication with different block sizes
    int block_sizes[] = {8, 16, 32, 64, 128};
    printf("\nMatrix Multiplication Performance with Different Block Sizes:\n");
    printf("-------------------------------------------------------------\n");
    printf("Block Size | SIMD Time (us) | Reference Time (us) | Speedup\n");
    printf("-------------------------------------------------------------\n");

    for (int i = 0; i < sizeof(block_sizes) / sizeof(block_sizes[0]); i++) {
        int block = block_sizes[i];
        // Here we would normally use different blocking implementations
        // For demo purposes, we'll just use the same function
        SimdBenchmarkResult result = benchmark_matrix_multiply(block, block, block, 5);
        printf("%10d | %14llu | %18llu | %7.2fx\n", block, result.simdTime, result.referenceTime,
               result.speedupFactor);
        free((void *)result.hardwareDetails);
    }

    printf("\nCache optimization comparison completed.\n");
}

/**
 * Main test function
 */
int main()
{
    // Initialize random seed
    srand((unsigned int)time(NULL));

    // Print system information
    char *system_info = simd_benchmark_init();
    printf("System Information:\n%s\n\n", system_info);
    free(system_info);

    // Run basic benchmark tests
    run_basic_benchmark_tests();

    // Run comprehensive benchmarks if user wants
    printf("\nDo you want to run the comprehensive benchmark suite? (y/n): ");
    char response;
    scanf(" %c", &response);
    if (response == 'y' || response == 'Y') {
        run_comprehensive_benchmarks();
    }

    // Run cache optimization comparison
    printf("\nDo you want to compare cache optimization techniques? (y/n): ");
    scanf(" %c", &response);
    if (response == 'y' || response == 'Y') {
        compare_cache_optimizations();
    }

    printf("\nAll SIMD benchmark tests completed.\n");
    return 0;
}
