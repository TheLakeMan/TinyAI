/**
 * @file benchmark.h
 * @brief Header file for benchmarking utilities for TinyAI to compare model performance
 */

#ifndef TINYAI_BENCHMARK_H
#define TINYAI_BENCHMARK_H

#include "../models/image/image_model.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

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
 * Run a benchmark on an image model
 * @param model The image model to benchmark
 * @param images Array of test images
 * @param labels Array of ground truth labels
 * @param numImages Number of images in the test set
 * @param numIterations Number of iterations to run for each image (for averaging)
 * @return BenchmarkResult structure with benchmark results
 */
BenchmarkResult benchmarkImageModel(TinyAIImageModel *model, TinyAIImage **images, int *labels,
                                    int numImages, int numIterations);

/**
 * Print a benchmark result
 * @param result The benchmark result to print
 */
void printBenchmarkResult(const BenchmarkResult *result);

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
                   TinyAIImage **images, int *labels, int numImages, int numIterations);

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
                                         int numImages, int numIterations);

/**
 * Create a CSV report from benchmark results
 * @param results Array of benchmark results
 * @param numResults Number of results in the array
 * @param filepath Path to save CSV file
 * @return true on success, false on failure
 */
bool createBenchmarkReport(const BenchmarkResult *results, int numResults, const char *filepath);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_BENCHMARK_H */
