/**
 * @file prune.c
 * @brief Implementation of model pruning utilities for TinyAI
 */

#include "prune.h"
#include "sparse_ops.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Internal helper functions */

/* Compare function for qsort - sorts floats by absolute value in descending order */
static int compareAbsFloat(const void *a, const void *b)
{
    float fa = fabsf(*(const float *)a);
    float fb = fabsf(*(const float *)b);
    if (fa > fb)
        return -1; /* Sort in descending order */
    if (fa < fb)
        return 1;
    return 0;
}

/* Calculate L2 norm of a filter/channel for structured pruning */
static float calculateL2Norm(const float *weights, int size)
{
    float norm = 0.0f;
    for (int i = 0; i < size; i++) {
        norm += weights[i] * weights[i];
    }
    return sqrtf(norm);
}

/* Apply pruning mask to weights (in-place) */
static void applyPruningMask(float *weights, const bool *mask, int size)
{
    for (int i = 0; i < size; i++) {
        if (!mask[i]) {
            weights[i] = 0.0f;
        }
    }
}

/* Implementation of public API */

bool tinyaiPruneModel(const char *srcModelPath, const char *dstModelPath,
                      const TinyAIPruneConfig *config)
{
    /* This would be a complex implementation that:
     * 1. Loads the model from srcModelPath
     * 2. Applies the appropriate pruning method to each layer
     * 3. Optionally retrains the model if config->retrainAfterPrune is true
     * 4. Saves the pruned model to dstModelPath
     */

    /* Basic parameter validation */
    if (!srcModelPath || !dstModelPath || !config) {
        return false;
    }

    /* Simplified placeholder implementation */
    /* In a real implementation, this would parse the model file format,
     * extract each layer's weights, apply pruning, and save the model */

    /* Return success for now - this would need to be properly implemented
     * with support for the specific model format being used */
    return true;
}

bool tinyaiPruneMatrixByMagnitude(float *weights, int rows, int cols, float pruneRate)
{
    if (!weights || rows <= 0 || cols <= 0 || pruneRate < 0.0f || pruneRate > 1.0f) {
        return false;
    }

    int size = rows * cols;

    /* Handle edge cases */
    if (size == 0) {
        return true;
    }

    if (pruneRate >= 1.0f) {
        /* Prune everything */
        memset(weights, 0, size * sizeof(float));
        return true;
    }

    if (pruneRate <= 0.0f) {
        /* No pruning */
        return true;
    }

    /* Create a copy of weights for sorting */
    float *weightsCopy = (float *)malloc(size * sizeof(float));
    if (!weightsCopy) {
        return false;
    }

    /* Copy weights to apply sorting */
    memcpy(weightsCopy, weights, size * sizeof(float));

    /* Sort weights by absolute magnitude */
    qsort(weightsCopy, size, sizeof(float), compareAbsFloat);

    /* Determine threshold index */
    int thresholdIdx = (int)(pruneRate * size);

    /* Get magnitude threshold */
    float threshold = fabsf(weightsCopy[thresholdIdx]);

    /* Apply pruning - set all weights with magnitude less than threshold to zero */
    for (int i = 0; i < size; i++) {
        if (fabsf(weights[i]) <= threshold) {
            weights[i] = 0.0f;
        }
    }

    free(weightsCopy);
    return true;
}

bool tinyaiPruneMatrixByThreshold(float *weights, int rows, int cols, float threshold)
{
    if (!weights || rows <= 0 || cols <= 0 || threshold < 0.0f) {
        return false;
    }

    int size = rows * cols;

    /* Set all weights with magnitude less than threshold to zero */
    for (int i = 0; i < size; i++) {
        if (fabsf(weights[i]) < threshold) {
            weights[i] = 0.0f;
        }
    }

    return true;
}

bool tinyaiPruneMatrixStructured(float *weights, int rows, int cols, float pruneRate,
                                 bool isConvFilter, const int *filterShape)
{
    if (!weights || rows <= 0 || cols <= 0 || pruneRate < 0.0f || pruneRate > 1.0f) {
        return false;
    }

    /* For structured pruning of conv filters, we need valid filter shape */
    if (isConvFilter && (!filterShape || filterShape[0] <= 0 || filterShape[1] <= 0 ||
                         filterShape[2] <= 0 || filterShape[3] <= 0)) {
        return false;
    }

    /* Handle edge cases */
    if (pruneRate >= 1.0f) {
        /* Prune everything */
        memset(weights, 0, rows * cols * sizeof(float));
        return true;
    }

    if (pruneRate <= 0.0f) {
        /* No pruning */
        return true;
    }

    if (isConvFilter) {
        /* Structured pruning for convolution filters */
        int depth   = filterShape[0];
        int height  = filterShape[1];
        int width   = filterShape[2];
        int filters = filterShape[3];

        /* Verify that dimensions match */
        if (rows != filters || cols != depth * height * width) {
            return false;
        }

        /* Calculate L2 norm for each filter */
        float *filterNorms = (float *)malloc(filters * sizeof(float));
        if (!filterNorms) {
            return false;
        }

        for (int f = 0; f < filters; f++) {
            filterNorms[f] = calculateL2Norm(&weights[f * cols], cols);
        }

        /* Create index array for sorting */
        int *filterIndices = (int *)malloc(filters * sizeof(int));
        if (!filterIndices) {
            free(filterNorms);
            return false;
        }

        /* Initialize indices */
        for (int i = 0; i < filters; i++) {
            filterIndices[i] = i;
        }

        /* Sort indices by filter norms (non-increasing) */
        for (int i = 0; i < filters; i++) {
            for (int j = i + 1; j < filters; j++) {
                if (filterNorms[filterIndices[i]] < filterNorms[filterIndices[j]]) {
                    int temp         = filterIndices[i];
                    filterIndices[i] = filterIndices[j];
                    filterIndices[j] = temp;
                }
            }
        }

        /* Determine number of filters to prune */
        int numPruneFilters = (int)(pruneRate * filters);

        /* Create pruning mask */
        bool *filterMask = (bool *)calloc(filters, sizeof(bool));
        if (!filterMask) {
            free(filterIndices);
            free(filterNorms);
            return false;
        }

        /* Mark filters to keep (true in mask) */
        for (int i = 0; i < filters - numPruneFilters; i++) {
            filterMask[filterIndices[i]] = true;
        }

        /* Apply pruning to filters */
        for (int f = 0; f < filters; f++) {
            if (!filterMask[f]) {
                /* Prune entire filter */
                memset(&weights[f * cols], 0, cols * sizeof(float));
            }
        }

        free(filterMask);
        free(filterIndices);
        free(filterNorms);
    }
    else {
        /* Structured pruning for fully connected layers */
        /* For FC layers, we prune entire rows (output neurons) */

        /* Calculate L2 norm for each row */
        float *rowNorms = (float *)malloc(rows * sizeof(float));
        if (!rowNorms) {
            return false;
        }

        for (int r = 0; r < rows; r++) {
            rowNorms[r] = calculateL2Norm(&weights[r * cols], cols);
        }

        /* Create index array for sorting */
        int *rowIndices = (int *)malloc(rows * sizeof(int));
        if (!rowIndices) {
            free(rowNorms);
            return false;
        }

        /* Initialize indices */
        for (int i = 0; i < rows; i++) {
            rowIndices[i] = i;
        }

        /* Sort indices by row norms (non-increasing) */
        for (int i = 0; i < rows; i++) {
            for (int j = i + 1; j < rows; j++) {
                if (rowNorms[rowIndices[i]] < rowNorms[rowIndices[j]]) {
                    int temp      = rowIndices[i];
                    rowIndices[i] = rowIndices[j];
                    rowIndices[j] = temp;
                }
            }
        }

        /* Determine number of rows to prune */
        int numPruneRows = (int)(pruneRate * rows);

        /* Create pruning mask */
        bool *rowMask = (bool *)calloc(rows, sizeof(bool));
        if (!rowMask) {
            free(rowIndices);
            free(rowNorms);
            return false;
        }

        /* Mark rows to keep (true in mask) */
        for (int i = 0; i < rows - numPruneRows; i++) {
            rowMask[rowIndices[i]] = true;
        }

        /* Apply pruning to rows */
        for (int r = 0; r < rows; r++) {
            if (!rowMask[r]) {
                /* Prune entire row */
                memset(&weights[r * cols], 0, cols * sizeof(float));
            }
        }

        free(rowMask);
        free(rowIndices);
        free(rowNorms);
    }

    return true;
}

bool tinyaiPruneMatrixRandom(float *weights, int rows, int cols, float pruneRate)
{
    if (!weights || rows <= 0 || cols <= 0 || pruneRate < 0.0f || pruneRate > 1.0f) {
        return false;
    }

    int size = rows * cols;

    /* Handle edge cases */
    if (size == 0) {
        return true;
    }

    if (pruneRate >= 1.0f) {
        /* Prune everything */
        memset(weights, 0, size * sizeof(float));
        return true;
    }

    if (pruneRate <= 0.0f) {
        /* No pruning */
        return true;
    }

    /* Seed random number generator */
    srand((unsigned int)time(NULL));

    /* Determine number of elements to prune */
    int numPruneElements = (int)(pruneRate * size);

    /* Create pruning mask (true = keep, false = prune) */
    bool *mask = (bool *)malloc(size * sizeof(bool));
    if (!mask) {
        return false;
    }

    /* Initialize mask to keep all elements */
    for (int i = 0; i < size; i++) {
        mask[i] = true;
    }

    /* Randomly select elements to prune */
    for (int i = 0; i < numPruneElements; i++) {
        int idx;
        do {
            idx = rand() % size;
        } while (!mask[idx]); /* Ensure we don't select already pruned elements */

        mask[idx] = false;
    }

    /* Apply pruning mask */
    applyPruningMask(weights, mask, size);

    free(mask);
    return true;
}

float tinyaiCalculateSparsity(const float *weights, int rows, int cols, float threshold)
{
    if (!weights || rows <= 0 || cols <= 0) {
        return 0.0f;
    }

    int size = rows * cols;
    if (size == 0) {
        return 0.0f;
    }

    int zeroCount = 0;
    for (int i = 0; i < size; i++) {
        if (fabsf(weights[i]) <= threshold) {
            zeroCount++;
        }
    }

    return (float)zeroCount / size;
}

TinyAIPruneConfig *tinyaiCreateDefaultPruneConfig(void)
{
    TinyAIPruneConfig *config = (TinyAIPruneConfig *)malloc(sizeof(TinyAIPruneConfig));
    if (!config) {
        return NULL;
    }

    /* Initialize with default values */
    config->method              = TINYAI_PRUNE_MAGNITUDE;
    config->pruneRate           = 0.5f;  /* 50% sparsity by default */
    config->threshold           = 1e-4f; /* Small threshold for zeroing out weights */
    config->layerWisePruning    = true;  /* Prune each layer separately */
    config->retrainAfterPrune   = true;  /* Do fine-tuning after pruning */
    config->numRetrainSteps     = 1000;  /* Number of fine-tuning steps */
    config->retrainLearningRate = 1e-5f; /* Small learning rate for fine-tuning */

    return config;
}

bool tinyaiEstimatePrunedSize(const char *modelPath, float pruneRate, bool useCSR,
                              size_t *originalSize, size_t *prunedSize)
{
    /* This would analyze the model file and estimate the size savings from pruning */
    /* For this implementation, we'll do a simple calculation based on the pruning rate */

    if (!modelPath || !originalSize || !prunedSize || pruneRate < 0.0f || pruneRate > 1.0f) {
        return false;
    }

    /* Placeholder implementation - would need to actually parse the model file */
    /* In a real implementation, this would:
     * 1. Load the model's metadata to determine its structure
     * 2. Calculate the original size of all weight matrices
     * 3. Calculate the expected pruned size based on pruning rate and storage format
     */

    /* Assume model file exists and is valid for this simplified implementation */
    FILE *modelFile = fopen(modelPath, "rb");
    if (!modelFile) {
        return false;
    }

    /* Get file size */
    fseek(modelFile, 0, SEEK_END);
    size_t fileSize = ftell(modelFile);
    fclose(modelFile);

    /* Assume 80% of file size is weights */
    size_t weightsSize  = (size_t)(fileSize * 0.8);
    size_t metadataSize = fileSize - weightsSize;

    /* Calculate original size */
    *originalSize = fileSize;

    /* Calculate pruned size */
    if (useCSR) {
        /* For CSR format, assuming 4-byte indices and 4-byte values */
        /* For sparsity s, we need (1-s)*nnz values, nnz+rows indices */
        size_t nnz       = (size_t)(weightsSize / sizeof(float));
        size_t prunedNnz = (size_t)(nnz * (1.0f - pruneRate));

        /* Estimate rows as sqrt(nnz) for simplistic model */
        size_t rows = (size_t)sqrtf((float)nnz);

        /* CSR format: values (4 bytes each) + row_ptr (4 bytes * (rows+1)) + col_indices (4 bytes
         * each) */
        size_t csrSize =
            prunedNnz * sizeof(float) + (rows + 1) * sizeof(int) + prunedNnz * sizeof(int);

        *prunedSize = metadataSize + csrSize;
    }
    else {
        /* For dense format with zeros, size reduction comes from storing quantized zeros */
        /* Simple approximation - weights size is reduced proportionally to pruning rate */
        *prunedSize = metadataSize + (size_t)(weightsSize * (1.0f - 0.75f * pruneRate));
    }

    return true;
}

bool tinyaiRetrainPrunedModel(const char *modelPath, const char *dataPath,
                              const TinyAIPruneConfig *config)
{
    /* This would implement fine-tuning of a pruned model to recover accuracy */
    /* A real implementation would:
     * 1. Load the pruned model
     * 2. Load calibration data
     * 3. Run a few epochs of training with a small learning rate
     * 4. Ensure zeros remain zeros during training (masked SGD)
     * 5. Save the fine-tuned model
     */

    if (!modelPath || !dataPath || !config) {
        return false;
    }

    /* Simplified placeholder - just return success for now */
    /* This is a placeholder for a complex implementation that would depend
     * on the training framework being used */
    return true;
}
