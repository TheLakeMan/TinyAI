/**
 * @file simd_benchmark.h
 * @brief Benchmarking utilities specifically for SIMD operations in TinyAI
 */

#ifndef TINYAI_SIMD_BENCHMARK_H
#define TINYAI_SIMD_BENCHMARK_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Structure to hold SIMD vs non-SIMD benchmark comparison results
 */
typedef struct {
    const char *operationName;   /* Name of the operation being benchmarked */
    uint64_t    simdTime;        /* Time for SIMD implementation (microseconds) */
    uint64_t    referenceTime;   /* Time for reference implementation (microseconds) */
    float       speedupFactor;   /* Performance improvement factor (reference/SIMD) */
    bool        correctness;     /* Whether SIMD implementation matches reference results */
    float       maxError;        /* Maximum error between SIMD and reference results */
    const char *hardwareDetails; /* CPU model, SIMD features available */
    size_t      dataSize;        /* Size of data processed during benchmark */
} SimdBenchmarkResult;

/**
 * Initialize the SIMD benchmarking system
 * Detects available SIMD features and prepares the benchmarking environment
 *
 * @return Descriptive string of available SIMD features (caller must free)
 */
char *simd_benchmark_init();

/**
 * Start a timer for benchmark measurement
 *
 * @return Start timestamp in microseconds
 */
uint64_t benchmark_start();

/**
 * End a timer for benchmark measurement
 *
 * @param start Start timestamp from benchmark_start
 * @return Duration in microseconds
 */
uint64_t benchmark_end(uint64_t start);

/**
 * Benchmark matrix multiplication operations (SIMD vs reference implementation)
 *
 * @param rows Number of rows in first matrix
 * @param cols Number of columns in first matrix / rows in second matrix
 * @param innerDim Number of columns in second matrix
 * @param iterations Number of iterations for timing
 * @return Benchmark results
 */
SimdBenchmarkResult benchmark_matrix_multiply(int rows, int cols, int innerDim, int iterations);

/**
 * Benchmark activation functions (SIMD vs reference implementation)
 *
 * @param size Size of the input vector
 * @param activationType Type of activation function (0=ReLU, 1=GELU, 2=Sigmoid)
 * @param iterations Number of iterations for timing
 * @return Benchmark results
 */
SimdBenchmarkResult benchmark_activation(int size, int activationType, int iterations);

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
                                        int iterations);

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
                                          int iterations);

/**
 * Run comprehensive benchmark suite for all SIMD operations
 * Creates a detailed performance profile for current hardware
 *
 * @param outputFile Path to save benchmark results as JSON (NULL for no file output)
 * @return Array of benchmark results (caller must free)
 */
SimdBenchmarkResult *run_comprehensive_simd_benchmark(const char *outputFile);

/**
 * Print a SIMD benchmark result
 *
 * @param result The benchmark result to print
 */
void print_simd_benchmark_result(const SimdBenchmarkResult *result);

/**
 * Create a CSV report from SIMD benchmark results
 *
 * @param results Array of benchmark results
 * @param numResults Number of results in the array
 * @param filepath Path to save CSV file
 * @return true on success, false on failure
 */
bool create_simd_benchmark_report(const SimdBenchmarkResult *results, int numResults,
                                  const char *filepath);

/**
 * Analyze cache usage and memory access patterns during SIMD operations
 *
 * @param operationType Type of operation to analyze (0=matmul, 1=conv, 2=attention)
 * @param dataSize Size of the test data
 * @param reportFile Path to save analysis report (NULL for no file output)
 * @return Performance recommendations string (caller must free)
 */
char *analyze_memory_access_patterns(int operationType, int dataSize, const char *reportFile);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_SIMD_BENCHMARK_H */
