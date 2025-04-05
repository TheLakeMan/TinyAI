/**
 * TinyAI Quantization Utilities Header
 * 
 * This header defines quantization utilities for TinyAI, enabling 4-bit and 8-bit
 * quantization of neural network weights and activations.
 */

#ifndef TINYAI_QUANTIZE_H
#define TINYAI_QUANTIZE_H

#include <stddef.h>
#include <stdint.h>

/* ----------------- Quantization Types ----------------- */

/**
 * Precision enumeration
 */
typedef enum {
    TINYAI_PRECISION_FP32,
    TINYAI_PRECISION_INT8,
    TINYAI_PRECISION_INT4
} TinyAIPrecision;

/**
 * 4-bit quantized matrix structure
 */
typedef struct {
    uint8_t *data;       /* Matrix data (4-bit packed, 2 values per byte) */
    uint32_t rows;       /* Number of rows */
    uint32_t cols;       /* Number of columns */
    float scale;         /* Scaling factor for quantization */
    float zeroPoint;     /* Zero point for quantization */
} TinyAIMatrix4bit;

/**
 * 8-bit quantized matrix structure
 */
typedef struct {
    int8_t *data;        /* Matrix data (8-bit values) */
    uint32_t rows;       /* Number of rows */
    uint32_t cols;       /* Number of columns */
    float scale;         /* Scaling factor for quantization */
    float zeroPoint;     /* Zero point for quantization */
} TinyAIMatrix8bit;

/**
 * FP32 matrix structure
 */
typedef struct {
    float *data;         /* Matrix data (32-bit float values) */
    uint32_t rows;       /* Number of rows */
    uint32_t cols;       /* Number of columns */
} TinyAIMatrixFP32;

/* ----------------- Matrix Operations ----------------- */

/**
 * Create a 4-bit quantized matrix
 * 
 * @param rows Number of rows
 * @param cols Number of columns
 * @return New matrix or NULL on error
 */
TinyAIMatrix4bit* tinyaiCreateMatrix4bit(uint32_t rows, uint32_t cols);

/**
 * Destroy a 4-bit quantized matrix
 * 
 * @param matrix Matrix to destroy
 */
void tinyaiDestroyMatrix4bit(TinyAIMatrix4bit *matrix);

/**
 * Create an 8-bit quantized matrix
 * 
 * @param rows Number of rows
 * @param cols Number of columns
 * @return New matrix or NULL on error
 */
TinyAIMatrix8bit* tinyaiCreateMatrix8bit(uint32_t rows, uint32_t cols);

/**
 * Destroy an 8-bit quantized matrix
 * 
 * @param matrix Matrix to destroy
 */
void tinyaiDestroyMatrix8bit(TinyAIMatrix8bit *matrix);

/**
 * Create a FP32 matrix
 * 
 * @param rows Number of rows
 * @param cols Number of columns
 * @return New matrix or NULL on error
 */
TinyAIMatrixFP32* tinyaiCreateMatrixFP32(uint32_t rows, uint32_t cols);

/**
 * Destroy a FP32 matrix
 * 
 * @param matrix Matrix to destroy
 */
void tinyaiDestroyMatrixFP32(TinyAIMatrixFP32 *matrix);

/**
 * Quantize a FP32 matrix to 4-bit
 * 
 * @param input Input FP32 matrix
 * @return Quantized 4-bit matrix or NULL on error
 */
TinyAIMatrix4bit* tinyaiQuantizeFP32To4bit(const TinyAIMatrixFP32 *input);

/**
 * Quantize a FP32 matrix to 8-bit
 * 
 * @param input Input FP32 matrix
 * @return Quantized 8-bit matrix or NULL on error
 */
TinyAIMatrix8bit* tinyaiQuantizeFP32To8bit(const TinyAIMatrixFP32 *input);

/**
 * Dequantize a 4-bit matrix to FP32
 * 
 * @param input Input 4-bit matrix
 * @return Dequantized FP32 matrix or NULL on error
 */
TinyAIMatrixFP32* tinyaiDequantize4bitToFP32(const TinyAIMatrix4bit *input);

/**
 * Dequantize an 8-bit matrix to FP32
 * 
 * @param input Input 8-bit matrix
 * @return Dequantized FP32 matrix or NULL on error
 */
TinyAIMatrixFP32* tinyaiDequantize8bitToFP32(const TinyAIMatrix8bit *input);

/**
 * Matrix multiplication: C = A * B
 * 
 * @param a Matrix A
 * @param b Matrix B
 * @param c Output matrix C
 * @param precision Precision to use for computation
 * @return 0 on success, non-zero on error
 */
int tinyaiMatrixMultiply(const void *a, const void *b, void *c, TinyAIPrecision precision);

/**
 * Matrix addition: C = A + B
 * 
 * @param a Matrix A
 * @param b Matrix B
 * @param c Output matrix C
 * @param precision Precision to use for computation
 * @return 0 on success, non-zero on error
 */
int tinyaiMatrixAdd(const void *a, const void *b, void *c, TinyAIPrecision precision);

/**
 * Apply activation function to matrix
 * 
 * @param input Input matrix
 * @param output Output matrix
 * @param activation Activation function ID (0 = none, 1 = ReLU, 2 = sigmoid, 3 = tanh)
 * @param precision Precision to use for computation
 * @return 0 on success, non-zero on error
 */
int tinyaiMatrixActivation(const void *input, void *output, int activation, TinyAIPrecision precision);

/* ----------------- Vector Operations ----------------- */

/**
 * Vector dot product
 * 
 * @param a Vector A
 * @param b Vector B
 * @param length Vector length
 * @param precision Precision to use for computation
 * @return Dot product result
 */
float tinyaiVectorDot(const void *a, const void *b, uint32_t length, TinyAIPrecision precision);

/**
 * Vector L2 norm (Euclidean distance)
 * 
 * @param a Vector A
 * @param length Vector length
 * @param precision Precision to use for computation
 * @return L2 norm
 */
float tinyaiVectorL2Norm(const void *a, uint32_t length, TinyAIPrecision precision);

/**
 * Vector cosine similarity
 * 
 * @param a Vector A
 * @param b Vector B
 * @param length Vector length
 * @param precision Precision to use for computation
 * @return Cosine similarity
 */
float tinyaiVectorCosineSimilarity(const void *a, const void *b, uint32_t length, TinyAIPrecision precision);

/* ----------------- Activation Functions ----------------- */

/**
 * ReLU activation function
 * 
 * @param x Input value
 * @return ReLU(x)
 */
float tinyaiActivationReLU(float x);

/**
 * Sigmoid activation function
 * 
 * @param x Input value
 * @return Sigmoid(x)
 */
float tinyaiActivationSigmoid(float x);

/**
 * Tanh activation function
 * 
 * @param x Input value
 * @return Tanh(x)
 */
float tinyaiActivationTanh(float x);

/**
 * GELU activation function
 * 
 * @param x Input value
 * @return GELU(x)
 */
float tinyaiActivationGELU(float x);

/**
 * Initialize activation function lookup tables
 * 
 * @return 0 on success, non-zero on error
 */
int tinyaiInitActivationTables();

/**
 * Clean up activation function lookup tables
 */
void tinyaiCleanupActivationTables();

/* ----------------- Utility Functions ----------------- */

/**
 * Find minimum and maximum values in FP32 array
 * 
 * @param data Input data
 * @param size Data size
 * @param min Pointer to store minimum value
 * @param max Pointer to store maximum value
 */
void tinyaiFindMinMax(const float *data, size_t size, float *min, float *max);

/**
 * Save a quantized matrix to a file
 * 
 * @param matrix Matrix to save
 * @param path File path
 * @param precision Precision of the matrix
 * @return 0 on success, non-zero on error
 */
int tinyaiSaveQuantizedMatrix(const void *matrix, const char *path, TinyAIPrecision precision);

/**
 * Load a quantized matrix from a file
 * 
 * @param path File path
 * @param precision Precision of the matrix
 * @return Loaded matrix or NULL on error
 */
void* tinyaiLoadQuantizedMatrix(const char *path, TinyAIPrecision precision);

#endif /* TINYAI_QUANTIZE_H */
