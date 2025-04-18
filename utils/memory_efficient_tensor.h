/**
 * @file memory_efficient_tensor.h
 * @brief Memory-efficient tensor operations for TinyAI
 *
 * This header provides utilities for performing tensor operations with minimal
 * memory overhead, including in-place operations, memory pooling, and
 * streaming operations.
 */

#ifndef TINYAI_MEMORY_EFFICIENT_TENSOR_H
#define TINYAI_MEMORY_EFFICIENT_TENSOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Tensor data type
 */
typedef enum {
    TINYAI_TENSOR_FLOAT32 = 0,
    TINYAI_TENSOR_FLOAT16 = 1,
    TINYAI_TENSOR_INT8    = 2,
    TINYAI_TENSOR_INT16   = 3,
    TINYAI_TENSOR_INT32   = 4
} TinyAITensorType;

/**
 * @brief Memory allocation strategy
 */
typedef enum {
    TINYAI_MEMORY_STATIC = 0, // Pre-allocated memory
    TINYAI_MEMORY_POOLED = 1, // Memory pooling
    TINYAI_MEMORY_STREAM = 2  // Streaming memory
} TinyAIMemoryStrategy;

/**
 * @brief Tensor shape information
 */
typedef struct {
    size_t *dims;       // Array of dimension sizes
    size_t  num_dims;   // Number of dimensions
    size_t  total_size; // Total number of elements
} TinyAITensorShape;

/**
 * @brief Memory-efficient tensor
 */
typedef struct {
    void                *data;          // Pointer to tensor data
    TinyAITensorType     type;          // Data type
    TinyAITensorShape    shape;         // Shape information
    TinyAIMemoryStrategy strategy;      // Memory allocation strategy
    size_t               memory_usage;  // Current memory usage
    bool                 is_contiguous; // Whether data is contiguous
    void                *memory_pool;   // Memory pool for pooled strategy
    size_t               pool_size;     // Size of memory pool
} TinyAIMemoryEfficientTensor;

/**
 * @brief Create a memory-efficient tensor
 *
 * @param shape Tensor shape
 * @param type Data type
 * @param strategy Memory allocation strategy
 * @return Pointer to created tensor or NULL on failure
 */
TinyAIMemoryEfficientTensor *tinyaiCreateMemoryEfficientTensor(const TinyAITensorShape *shape,
                                                               TinyAITensorType         type,
                                                               TinyAIMemoryStrategy     strategy);

/**
 * @brief Free a memory-efficient tensor
 *
 * @param tensor Tensor to free
 */
void tinyaiFreeMemoryEfficientTensor(TinyAIMemoryEfficientTensor *tensor);

/**
 * @brief Perform in-place tensor addition
 *
 * @param dest Destination tensor
 * @param src Source tensor
 * @return true if successful, false on failure
 */
bool tinyaiTensorAddInPlace(TinyAIMemoryEfficientTensor       *dest,
                            const TinyAIMemoryEfficientTensor *src);

/**
 * @brief Perform in-place tensor multiplication
 *
 * @param dest Destination tensor
 * @param src Source tensor
 * @return true if successful, false on failure
 */
bool tinyaiTensorMulInPlace(TinyAIMemoryEfficientTensor       *dest,
                            const TinyAIMemoryEfficientTensor *src);

/**
 * @brief Perform streaming tensor operation
 *
 * @param dest Destination tensor
 * @param src Source tensor
 * @param operation Operation to perform
 * @param chunk_size Size of chunks for streaming
 * @return true if successful, false on failure
 */
bool tinyaiTensorStreamOperation(TinyAIMemoryEfficientTensor       *dest,
                                 const TinyAIMemoryEfficientTensor *src,
                                 void (*operation)(void *, const void *, size_t),
                                 size_t chunk_size);

/**
 * @brief Allocate memory from pool
 *
 * @param tensor Tensor using memory pool
 * @param size Size to allocate
 * @return Pointer to allocated memory or NULL on failure
 */
void *tinyaiTensorPoolAlloc(TinyAIMemoryEfficientTensor *tensor, size_t size);

/**
 * @brief Free memory to pool
 *
 * @param tensor Tensor using memory pool
 * @param ptr Pointer to memory to free
 */
void tinyaiTensorPoolFree(TinyAIMemoryEfficientTensor *tensor, void *ptr);

/**
 * @brief Get tensor memory usage
 *
 * @param tensor Tensor to check
 * @return Current memory usage in bytes
 */
size_t tinyaiGetTensorMemoryUsage(const TinyAIMemoryEfficientTensor *tensor);

/**
 * @brief Optimize tensor memory layout
 *
 * @param tensor Tensor to optimize
 * @return true if successful, false on failure
 */
bool tinyaiOptimizeTensorMemory(TinyAIMemoryEfficientTensor *tensor);

/**
 * @brief Convert tensor to contiguous memory layout
 *
 * @param tensor Tensor to convert
 * @return true if successful, false on failure
 */
bool tinyaiMakeTensorContiguous(TinyAIMemoryEfficientTensor *tensor);

/**
 * @brief Resize tensor memory pool
 *
 * @param tensor Tensor using memory pool
 * @param new_size New pool size
 * @return true if successful, false on failure
 */
bool tinyaiResizeTensorPool(TinyAIMemoryEfficientTensor *tensor, size_t new_size);

/**
 * @brief Get tensor data pointer
 *
 * @param tensor Tensor to get data from
 * @return Pointer to tensor data
 */
void *tinyaiGetTensorData(const TinyAIMemoryEfficientTensor *tensor);

/**
 * @brief Get tensor shape
 *
 * @param tensor Tensor to get shape from
 * @return Tensor shape
 */
TinyAITensorShape tinyaiGetTensorShape(const TinyAIMemoryEfficientTensor *tensor);

/**
 * @brief Set tensor data
 *
 * @param tensor Tensor to set data for
 * @param data Data to set
 * @param size Size of data in bytes
 * @return true if successful, false on failure
 */
bool tinyaiSetTensorData(TinyAIMemoryEfficientTensor *tensor, const void *data, size_t size);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_MEMORY_EFFICIENT_TENSOR_H */