#include "memory_efficient_tensor.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Test tensor creation and initialization
static void test_tensor_creation()
{
    // Create a 2x3 tensor shape
    size_t            dims[] = {2, 3};
    TinyAITensorShape shape  = {.dims = dims, .num_dims = 2, .total_size = 6};

    // Test static memory allocation
    TinyAIMemoryEfficientTensor *tensor =
        tinyaiCreateMemoryEfficientTensor(&shape, TINYAI_TENSOR_FLOAT32, TINYAI_MEMORY_STATIC);
    assert(tensor != NULL);
    assert(tensor->type == TINYAI_TENSOR_FLOAT32);
    assert(tensor->strategy == TINYAI_MEMORY_STATIC);
    assert(tensor->shape.num_dims == 2);
    assert(tensor->shape.total_size == 6);
    assert(tensor->memory_usage == 6 * sizeof(float));
    assert(tensor->is_contiguous == true);
    tinyaiFreeMemoryEfficientTensor(tensor);

    // Test pooled memory allocation
    tensor = tinyaiCreateMemoryEfficientTensor(&shape, TINYAI_TENSOR_FLOAT32, TINYAI_MEMORY_POOLED);
    assert(tensor != NULL);
    assert(tensor->strategy == TINYAI_MEMORY_POOLED);
    assert(tensor->memory_pool != NULL);
    assert(tensor->pool_size == 6 * sizeof(float));
    tinyaiFreeMemoryEfficientTensor(tensor);

    // Test streaming memory allocation
    tensor = tinyaiCreateMemoryEfficientTensor(&shape, TINYAI_TENSOR_FLOAT32, TINYAI_MEMORY_STREAM);
    assert(tensor != NULL);
    assert(tensor->strategy == TINYAI_MEMORY_STREAM);
    assert(tensor->data == NULL);
    tinyaiFreeMemoryEfficientTensor(tensor);
}

// Test in-place tensor operations
static void test_inplace_operations()
{
    // Create two 2x2 tensors
    size_t            dims[] = {2, 2};
    TinyAITensorShape shape  = {.dims = dims, .num_dims = 2, .total_size = 4};

    // Create tensors
    TinyAIMemoryEfficientTensor *tensor1 =
        tinyaiCreateMemoryEfficientTensor(&shape, TINYAI_TENSOR_FLOAT32, TINYAI_MEMORY_STATIC);
    TinyAIMemoryEfficientTensor *tensor2 =
        tinyaiCreateMemoryEfficientTensor(&shape, TINYAI_TENSOR_FLOAT32, TINYAI_MEMORY_STATIC);
    assert(tensor1 != NULL && tensor2 != NULL);

    // Initialize data
    float data1[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float data2[] = {0.5f, 1.5f, 2.5f, 3.5f};
    assert(tinyaiSetTensorData(tensor1, data1, sizeof(data1)));
    assert(tinyaiSetTensorData(tensor2, data2, sizeof(data2)));

    // Test addition
    assert(tinyaiTensorAddInPlace(tensor1, tensor2));
    float *result = (float *)tinyaiGetTensorData(tensor1);
    assert(result[0] == 1.5f);
    assert(result[1] == 3.5f);
    assert(result[2] == 5.5f);
    assert(result[3] == 7.5f);

    // Test multiplication
    assert(tinyaiTensorMulInPlace(tensor1, tensor2));
    result = (float *)tinyaiGetTensorData(tensor1);
    assert(result[0] == 0.75f);
    assert(result[1] == 5.25f);
    assert(result[2] == 13.75f);
    assert(result[3] == 26.25f);

    tinyaiFreeMemoryEfficientTensor(tensor1);
    tinyaiFreeMemoryEfficientTensor(tensor2);
}

// Test streaming operations
static void test_streaming_operations()
{
    // Create a 4x4 tensor
    size_t            dims[] = {4, 4};
    TinyAITensorShape shape  = {.dims = dims, .num_dims = 2, .total_size = 16};

    // Create tensors
    TinyAIMemoryEfficientTensor *tensor1 =
        tinyaiCreateMemoryEfficientTensor(&shape, TINYAI_TENSOR_FLOAT32, TINYAI_MEMORY_STATIC);
    TinyAIMemoryEfficientTensor *tensor2 =
        tinyaiCreateMemoryEfficientTensor(&shape, TINYAI_TENSOR_FLOAT32, TINYAI_MEMORY_STATIC);
    assert(tensor1 != NULL && tensor2 != NULL);

    // Initialize data
    float data1[16], data2[16];
    for (int i = 0; i < 16; i++) {
        data1[i] = (float)i;
        data2[i] = (float)(i + 1);
    }
    assert(tinyaiSetTensorData(tensor1, data1, sizeof(data1)));
    assert(tinyaiSetTensorData(tensor2, data2, sizeof(data2)));

    // Define a streaming operation (addition)
    void add_operation(void *dest, const void *src, size_t count)
    {
        float       *d = (float *)dest;
        const float *s = (const float *)src;
        for (size_t i = 0; i < count; i++) {
            d[i] += s[i];
        }
    }

    // Test streaming operation with chunk size 4
    assert(tinyaiTensorStreamOperation(tensor1, tensor2, add_operation, 4));
    float *result = (float *)tinyaiGetTensorData(tensor1);
    for (int i = 0; i < 16; i++) {
        assert(result[i] == (float)(i + (i + 1)));
    }

    tinyaiFreeMemoryEfficientTensor(tensor1);
    tinyaiFreeMemoryEfficientTensor(tensor2);
}

// Test memory pool operations
static void test_memory_pool()
{
    // Create a tensor with pooled memory
    size_t            dims[] = {3, 3};
    TinyAITensorShape shape  = {.dims = dims, .num_dims = 2, .total_size = 9};

    TinyAIMemoryEfficientTensor *tensor =
        tinyaiCreateMemoryEfficientTensor(&shape, TINYAI_TENSOR_FLOAT32, TINYAI_MEMORY_POOLED);
    assert(tensor != NULL);

    // Test memory allocation from pool
    void *ptr1 = tinyaiTensorPoolAlloc(tensor, sizeof(float) * 4);
    assert(ptr1 != NULL);

    void *ptr2 = tinyaiTensorPoolAlloc(tensor, sizeof(float) * 3);
    assert(ptr2 != NULL);

    // Test memory freeing
    tinyaiTensorPoolFree(tensor, ptr1);
    tinyaiTensorPoolFree(tensor, ptr2);

    // Test pool resizing
    assert(tinyaiResizeTensorPool(tensor, tensor->pool_size * 2));
    assert(tensor->pool_size == 9 * sizeof(float) * 2);

    tinyaiFreeMemoryEfficientTensor(tensor);
}

// Test memory optimization
static void test_memory_optimization()
{
    // Create a non-contiguous tensor
    size_t            dims[] = {2, 3};
    TinyAITensorShape shape  = {.dims = dims, .num_dims = 2, .total_size = 6};

    TinyAIMemoryEfficientTensor *tensor =
        tinyaiCreateMemoryEfficientTensor(&shape, TINYAI_TENSOR_FLOAT32, TINYAI_MEMORY_STATIC);
    assert(tensor != NULL);

    // Make tensor non-contiguous
    tensor->is_contiguous = false;

    // Test memory optimization
    assert(tinyaiOptimizeTensorMemory(tensor));
    assert(tensor->is_contiguous == true);

    tinyaiFreeMemoryEfficientTensor(tensor);
}

int main()
{
    printf("Testing memory-efficient tensor operations...\n");

    test_tensor_creation();
    printf("Tensor creation test passed\n");

    test_inplace_operations();
    printf("In-place operations test passed\n");

    test_streaming_operations();
    printf("Streaming operations test passed\n");

    test_memory_pool();
    printf("Memory pool test passed\n");

    test_memory_optimization();
    printf("Memory optimization test passed\n");

    printf("All memory-efficient tensor tests passed successfully!\n");
    return 0;
}