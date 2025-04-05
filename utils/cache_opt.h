/**
 * @file cache_opt.h
 * @brief Cache optimization utilities for TinyAI
 *
 * This file contains functions and structures for optimizing memory access patterns
 * and maximizing cache utilization through techniques like blocking/tiling,
 * prefetching, and cache-aware data layouts.
 */

#ifndef TINYAI_CACHE_OPT_H
#define TINYAI_CACHE_OPT_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Cache optimization configuration structure
 */
typedef struct {
    size_t blockSizeX;       /* Block size for the first dimension */
    size_t blockSizeY;       /* Block size for the second dimension */
    int    prefetchDistance; /* Prefetch distance in elements */
    bool   enablePrefetch;   /* Enable software prefetching */
    bool   enableTiling;     /* Enable loop tiling/blocking */
} TinyAICacheOptConfig;

/**
 * Cache hierarchy information structure
 */
typedef struct {
    size_t l1dCacheSize;    /* L1 data cache size in bytes */
    size_t l2CacheSize;     /* L2 cache size in bytes */
    size_t l3CacheSize;     /* L3 cache size in bytes */
    size_t cacheLineSize;   /* Cache line size in bytes */
    int    l1Associativity; /* L1 cache associativity */
    int    l2Associativity; /* L2 cache associativity */
    int    l3Associativity; /* L3 cache associativity */
} TinyAICacheInfo;

/**
 * Initialize default cache optimization configuration
 * @return Default configuration structure
 */
TinyAICacheOptConfig tinyai_cache_opt_init_default(void);

/**
 * Get cache hierarchy information for the current CPU
 * @return Cache hierarchy information structure
 */
TinyAICacheInfo tinyai_get_cache_info(void);

/**
 * Optimize configuration for matrix multiplication
 * @param rows Number of rows in the first matrix
 * @param cols Number of columns in the second matrix
 * @param inner Inner dimension (columns of first, rows of second)
 * @param config Pointer to configuration structure to optimize
 */
void tinyai_cache_opt_matrix_multiply(size_t rows, size_t cols, size_t inner,
                                      TinyAICacheOptConfig *config);

/**
 * Optimize configuration for convolution operations
 * @param inputWidth Width of input feature map
 * @param inputHeight Height of input feature map
 * @param inputChannels Number of input channels
 * @param kernelSize Size of convolution kernel (assume square)
 * @param outputChannels Number of output channels
 * @param config Pointer to configuration structure to optimize
 */
void tinyai_cache_opt_convolution(size_t inputWidth, size_t inputHeight, size_t inputChannels,
                                  size_t kernelSize, size_t outputChannels,
                                  TinyAICacheOptConfig *config);

/**
 * Perform software prefetch of memory address
 * @param addr Memory address to prefetch
 * @param readWrite 0 for read, 1 for write
 * @param locality Temporal locality hint (0-3, where 3 is highest locality)
 */
void tinyai_prefetch(const void *addr, int readWrite, int locality);

/**
 * Transpose matrix in a cache-friendly way
 * @param dest Destination matrix
 * @param src Source matrix
 * @param rows Number of rows
 * @param cols Number of columns
 * @param elemSize Size of each element in bytes
 */
void tinyai_transpose_blocked(void *dest, const void *src, size_t rows, size_t cols,
                              size_t elemSize);

/**
 * Loop tiling macro for 2D loops
 * Iterates over the specified ranges in tiles/blocks for better cache utilization
 */
#define TINYAI_LOOP_TILING_2D(i, i_start, i_end, j, j_start, j_end, block_i, block_j, body)        \
    for (size_t i##_block = (i_start); i##_block < (i_end); i##_block += (block_i)) {              \
        for (size_t j##_block = (j_start); j##_block < (j_end); j##_block += (block_j)) {          \
            for (size_t i = i##_block; i < i##_block + (block_i) && i < (i_end); i++) {            \
                for (size_t j = j##_block; j < j##_block + (block_j) && j < (j_end); j++) {        \
                    body                                                                           \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_CACHE_OPT_H */
