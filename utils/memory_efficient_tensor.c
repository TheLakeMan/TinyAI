#include "memory_efficient_tensor.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Get size of tensor type in bytes
static size_t get_type_size(TinyAITensorType type)
{
    switch (type) {
    case TINYAI_TENSOR_FLOAT32:
        return sizeof(float);
    case TINYAI_TENSOR_FLOAT16:
        return sizeof(uint16_t);
    case TINYAI_TENSOR_INT8:
        return sizeof(int8_t);
    case TINYAI_TENSOR_INT16:
        return sizeof(int16_t);
    case TINYAI_TENSOR_INT32:
        return sizeof(int32_t);
    default:
        return 0;
    }
}

// Create a memory-efficient tensor
TinyAIMemoryEfficientTensor *tinyaiCreateMemoryEfficientTensor(const TinyAITensorShape *shape,
                                                               TinyAITensorType         type,
                                                               TinyAIMemoryStrategy     strategy)
{

    if (!shape || !shape->dims || shape->num_dims == 0) {
        return NULL;
    }

    // Calculate total size
    size_t type_size = get_type_size(type);
    if (type_size == 0) {
        return NULL;
    }

    // Allocate tensor structure
    TinyAIMemoryEfficientTensor *tensor = malloc(sizeof(TinyAIMemoryEfficientTensor));
    if (!tensor) {
        return NULL;
    }

    // Initialize tensor shape
    tensor->shape.dims = malloc(shape->num_dims * sizeof(size_t));
    if (!tensor->shape.dims) {
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape.dims, shape->dims, shape->num_dims * sizeof(size_t));
    tensor->shape.num_dims   = shape->num_dims;
    tensor->shape.total_size = shape->total_size;

    // Initialize other fields
    tensor->type          = type;
    tensor->strategy      = strategy;
    tensor->is_contiguous = true;
    tensor->memory_pool   = NULL;
    tensor->pool_size     = 0;

    // Allocate memory based on strategy
    size_t total_bytes   = shape->total_size * type_size;
    tensor->memory_usage = total_bytes;

    switch (strategy) {
    case TINYAI_MEMORY_STATIC:
        tensor->data = malloc(total_bytes);
        if (!tensor->data) {
            free(tensor->shape.dims);
            free(tensor);
            return NULL;
        }
        break;

    case TINYAI_MEMORY_POOLED:
        tensor->memory_pool = malloc(total_bytes);
        if (!tensor->memory_pool) {
            free(tensor->shape.dims);
            free(tensor);
            return NULL;
        }
        tensor->pool_size = total_bytes;
        tensor->data      = tensor->memory_pool;
        break;

    case TINYAI_MEMORY_STREAM:
        // For streaming, we'll allocate memory in chunks as needed
        tensor->data = NULL;
        break;

    default:
        free(tensor->shape.dims);
        free(tensor);
        return NULL;
    }

    return tensor;
}

// Free a memory-efficient tensor
void tinyaiFreeMemoryEfficientTensor(TinyAIMemoryEfficientTensor *tensor)
{
    if (!tensor) {
        return;
    }

    if (tensor->shape.dims) {
        free(tensor->shape.dims);
    }

    switch (tensor->strategy) {
    case TINYAI_MEMORY_STATIC:
        if (tensor->data) {
            free(tensor->data);
        }
        break;

    case TINYAI_MEMORY_POOLED:
        if (tensor->memory_pool) {
            free(tensor->memory_pool);
        }
        break;

    case TINYAI_MEMORY_STREAM:
        if (tensor->data) {
            free(tensor->data);
        }
        break;
    }

    free(tensor);
}

// Perform in-place tensor addition
bool tinyaiTensorAddInPlace(TinyAIMemoryEfficientTensor       *dest,
                            const TinyAIMemoryEfficientTensor *src)
{
    if (!dest || !src || dest->type != src->type ||
        dest->shape.total_size != src->shape.total_size) {
        return false;
    }

    size_t type_size   = get_type_size(dest->type);
    size_t total_bytes = dest->shape.total_size * type_size;

    // Ensure tensors are contiguous
    if (!dest->is_contiguous) {
        if (!tinyaiMakeTensorContiguous(dest)) {
            return false;
        }
    }

    if (!src->is_contiguous) {
        TinyAIMemoryEfficientTensor *temp =
            tinyaiCreateMemoryEfficientTensor(&src->shape, src->type, TINYAI_MEMORY_STATIC);
        if (!temp) {
            return false;
        }
        memcpy(temp->data, src->data, total_bytes);
        tinyaiMakeTensorContiguous(temp);

        // Perform addition
        for (size_t i = 0; i < dest->shape.total_size; i++) {
            switch (dest->type) {
            case TINYAI_TENSOR_FLOAT32:
                ((float *)dest->data)[i] += ((float *)temp->data)[i];
                break;
            case TINYAI_TENSOR_FLOAT16:
                ((uint16_t *)dest->data)[i] += ((uint16_t *)temp->data)[i];
                break;
            case TINYAI_TENSOR_INT8:
                ((int8_t *)dest->data)[i] += ((int8_t *)temp->data)[i];
                break;
            case TINYAI_TENSOR_INT16:
                ((int16_t *)dest->data)[i] += ((int16_t *)temp->data)[i];
                break;
            case TINYAI_TENSOR_INT32:
                ((int32_t *)dest->data)[i] += ((int32_t *)temp->data)[i];
                break;
            }
        }

        tinyaiFreeMemoryEfficientTensor(temp);
    }
    else {
        // Perform addition directly
        for (size_t i = 0; i < dest->shape.total_size; i++) {
            switch (dest->type) {
            case TINYAI_TENSOR_FLOAT32:
                ((float *)dest->data)[i] += ((float *)src->data)[i];
                break;
            case TINYAI_TENSOR_FLOAT16:
                ((uint16_t *)dest->data)[i] += ((uint16_t *)src->data)[i];
                break;
            case TINYAI_TENSOR_INT8:
                ((int8_t *)dest->data)[i] += ((int8_t *)src->data)[i];
                break;
            case TINYAI_TENSOR_INT16:
                ((int16_t *)dest->data)[i] += ((int16_t *)src->data)[i];
                break;
            case TINYAI_TENSOR_INT32:
                ((int32_t *)dest->data)[i] += ((int32_t *)src->data)[i];
                break;
            }
        }
    }

    return true;
}

// Perform in-place tensor multiplication
bool tinyaiTensorMulInPlace(TinyAIMemoryEfficientTensor       *dest,
                            const TinyAIMemoryEfficientTensor *src)
{
    if (!dest || !src || dest->type != src->type ||
        dest->shape.total_size != src->shape.total_size) {
        return false;
    }

    // Ensure tensors are contiguous
    if (!dest->is_contiguous) {
        if (!tinyaiMakeTensorContiguous(dest)) {
            return false;
        }
    }

    if (!src->is_contiguous) {
        TinyAIMemoryEfficientTensor *temp =
            tinyaiCreateMemoryEfficientTensor(&src->shape, src->type, TINYAI_MEMORY_STATIC);
        if (!temp) {
            return false;
        }
        memcpy(temp->data, src->data, src->shape.total_size * get_type_size(src->type));
        tinyaiMakeTensorContiguous(temp);

        // Perform multiplication
        for (size_t i = 0; i < dest->shape.total_size; i++) {
            switch (dest->type) {
            case TINYAI_TENSOR_FLOAT32:
                ((float *)dest->data)[i] *= ((float *)temp->data)[i];
                break;
            case TINYAI_TENSOR_FLOAT16:
                ((uint16_t *)dest->data)[i] *= ((uint16_t *)temp->data)[i];
                break;
            case TINYAI_TENSOR_INT8:
                ((int8_t *)dest->data)[i] *= ((int8_t *)temp->data)[i];
                break;
            case TINYAI_TENSOR_INT16:
                ((int16_t *)dest->data)[i] *= ((int16_t *)temp->data)[i];
                break;
            case TINYAI_TENSOR_INT32:
                ((int32_t *)dest->data)[i] *= ((int32_t *)temp->data)[i];
                break;
            }
        }

        tinyaiFreeMemoryEfficientTensor(temp);
    }
    else {
        // Perform multiplication directly
        for (size_t i = 0; i < dest->shape.total_size; i++) {
            switch (dest->type) {
            case TINYAI_TENSOR_FLOAT32:
                ((float *)dest->data)[i] *= ((float *)src->data)[i];
                break;
            case TINYAI_TENSOR_FLOAT16:
                ((uint16_t *)dest->data)[i] *= ((uint16_t *)src->data)[i];
                break;
            case TINYAI_TENSOR_INT8:
                ((int8_t *)dest->data)[i] *= ((int8_t *)src->data)[i];
                break;
            case TINYAI_TENSOR_INT16:
                ((int16_t *)dest->data)[i] *= ((int16_t *)src->data)[i];
                break;
            case TINYAI_TENSOR_INT32:
                ((int32_t *)dest->data)[i] *= ((int32_t *)src->data)[i];
                break;
            }
        }
    }

    return true;
}

// Perform streaming tensor operation
bool tinyaiTensorStreamOperation(TinyAIMemoryEfficientTensor       *dest,
                                 const TinyAIMemoryEfficientTensor *src,
                                 void (*operation)(void *, const void *, size_t), size_t chunk_size)
{
    if (!dest || !src || !operation || chunk_size == 0 || dest->type != src->type ||
        dest->shape.total_size != src->shape.total_size) {
        return false;
    }

    size_t type_size   = get_type_size(dest->type);
    size_t total_bytes = dest->shape.total_size * type_size;
    size_t chunk_bytes = chunk_size * type_size;

    // Allocate temporary buffers for streaming
    void *dest_chunk = malloc(chunk_bytes);
    void *src_chunk  = malloc(chunk_bytes);
    if (!dest_chunk || !src_chunk) {
        free(dest_chunk);
        free(src_chunk);
        return false;
    }

    // Process data in chunks
    for (size_t offset = 0; offset < total_bytes; offset += chunk_bytes) {
        size_t current_chunk =
            (total_bytes - offset < chunk_bytes) ? (total_bytes - offset) : chunk_bytes;

        // Copy chunks
        memcpy(dest_chunk, (char *)dest->data + offset, current_chunk);
        memcpy(src_chunk, (char *)src->data + offset, current_chunk);

        // Perform operation on chunks
        operation(dest_chunk, src_chunk, current_chunk / type_size);

        // Copy result back
        memcpy((char *)dest->data + offset, dest_chunk, current_chunk);
    }

    free(dest_chunk);
    free(src_chunk);
    return true;
}

// Allocate memory from pool
void *tinyaiTensorPoolAlloc(TinyAIMemoryEfficientTensor *tensor, size_t size)
{
    if (!tensor || tensor->strategy != TINYAI_MEMORY_POOLED || !tensor->memory_pool) {
        return NULL;
    }

    // Simple first-fit allocation
    size_t offset = 0;
    while (offset + size <= tensor->pool_size) {
        bool is_free = true;
        for (size_t i = 0; i < size; i++) {
            if (((char *)tensor->memory_pool)[offset + i] != 0) {
                is_free = false;
                break;
            }
        }
        if (is_free) {
            return (char *)tensor->memory_pool + offset;
        }
        offset++;
    }

    return NULL;
}

// Free memory to pool
void tinyaiTensorPoolFree(TinyAIMemoryEfficientTensor *tensor, void *ptr)
{
    if (!tensor || !ptr || tensor->strategy != TINYAI_MEMORY_POOLED || ptr < tensor->memory_pool ||
        ptr >= (char *)tensor->memory_pool + tensor->pool_size) {
        return;
    }

    // Mark memory as free
    memset(ptr, 0, tensor->pool_size - ((char *)ptr - (char *)tensor->memory_pool));
}

// Get tensor memory usage
size_t tinyaiGetTensorMemoryUsage(const TinyAIMemoryEfficientTensor *tensor)
{
    return tensor ? tensor->memory_usage : 0;
}

// Optimize tensor memory layout
bool tinyaiOptimizeTensorMemory(TinyAIMemoryEfficientTensor *tensor)
{
    if (!tensor) {
        return false;
    }

    // Make tensor contiguous if not already
    if (!tensor->is_contiguous) {
        return tinyaiMakeTensorContiguous(tensor);
    }

    // For pooled memory, try to compact the pool
    if (tensor->strategy == TINYAI_MEMORY_POOLED && tensor->memory_pool) {
        void *new_pool = malloc(tensor->pool_size);
        if (!new_pool) {
            return false;
        }

        // Copy used memory to new pool
        memcpy(new_pool, tensor->memory_pool, tensor->pool_size);
        free(tensor->memory_pool);
        tensor->memory_pool = new_pool;
        tensor->data        = new_pool;
    }

    return true;
}

// Convert tensor to contiguous memory layout
bool tinyaiMakeTensorContiguous(TinyAIMemoryEfficientTensor *tensor)
{
    if (!tensor || tensor->is_contiguous) {
        return true;
    }

    size_t type_size   = get_type_size(tensor->type);
    size_t total_bytes = tensor->shape.total_size * type_size;

    // Allocate new contiguous memory
    void *new_data = malloc(total_bytes);
    if (!new_data) {
        return false;
    }

    // Copy data to new memory
    memcpy(new_data, tensor->data, total_bytes);

    // Update tensor
    if (tensor->strategy == TINYAI_MEMORY_STATIC) {
        free(tensor->data);
    }
    tensor->data          = new_data;
    tensor->is_contiguous = true;

    return true;
}

// Resize tensor memory pool
bool tinyaiResizeTensorPool(TinyAIMemoryEfficientTensor *tensor, size_t new_size)
{
    if (!tensor || tensor->strategy != TINYAI_MEMORY_POOLED) {
        return false;
    }

    void *new_pool = realloc(tensor->memory_pool, new_size);
    if (!new_pool) {
        return false;
    }

    tensor->memory_pool  = new_pool;
    tensor->pool_size    = new_size;
    tensor->data         = new_pool;
    tensor->memory_usage = new_size;

    return true;
}

// Get tensor data pointer
void *tinyaiGetTensorData(const TinyAIMemoryEfficientTensor *tensor)
{
    return tensor ? tensor->data : NULL;
}

// Get tensor shape
TinyAITensorShape tinyaiGetTensorShape(const TinyAIMemoryEfficientTensor *tensor)
{
    TinyAITensorShape shape = {0};
    if (tensor) {
        shape = tensor->shape;
    }
    return shape;
}

// Set tensor data
bool tinyaiSetTensorData(TinyAIMemoryEfficientTensor *tensor, const void *data, size_t size)
{
    if (!tensor || !data || size == 0) {
        return false;
    }

    size_t type_size   = get_type_size(tensor->type);
    size_t total_bytes = tensor->shape.total_size * type_size;

    if (size > total_bytes) {
        return false;
    }

    // Ensure tensor is contiguous
    if (!tensor->is_contiguous) {
        if (!tinyaiMakeTensorContiguous(tensor)) {
            return false;
        }
    }

    // Copy data
    memcpy(tensor->data, data, size);
    return true;
}