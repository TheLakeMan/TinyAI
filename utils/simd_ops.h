/**
 * @file simd_ops.h
 * @brief SIMD-accelerated operations for TinyAI
 *
 * This header provides SIMD-accelerated functions for matrix operations
 * with 4-bit quantized values.
 */

#ifndef TINYAI_SIMD_OPS_H
#define TINYAI_SIMD_OPS_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Check if SIMD extensions are available at runtime
 * @return true if SIMD acceleration is available
 */
bool tinyaiSimdAvailable(void);

/**
 * @brief SIMD-accelerated matrix-vector multiplication for 4-bit weights
 *
 * Multiplies a 4-bit quantized matrix by a vector of floats.
 * Each byte of the packed 4-bit matrix contains two weights.
 *
 * @param out Output vector
 * @param weights 4-bit quantized weight matrix (packed)
 * @param input Input vector
 * @param rows Number of rows in the weight matrix
 * @param cols Number of columns in the weight matrix
 * @param scaleFactors Scale factors for dequantizing weights
 */
void tinyaiSimdMatMul4Bit(float *out, const uint8_t *weights, const float *input, int rows,
                          int cols, const float *scaleFactors);

/**
 * @brief SIMD-accelerated vector addition
 *
 * Adds two vectors using SIMD instructions
 *
 * @param out Output vector (can be same as a)
 * @param a First vector
 * @param b Second vector
 * @param size Vector size
 */
void tinyaiSimdVecAdd(float *out, const float *a, const float *b, int size);

/**
 * @brief SIMD-accelerated vector activation function
 *
 * Applies an activation function to a vector using SIMD instructions
 *
 * @param inout Input/output vector
 * @param size Vector size
 * @param activationType Type of activation (0=ReLU, 1=GELU, 2=Sigmoid)
 */
void tinyaiSimdActivate(float *inout, int size, int activationType);

/**
 * @brief SIMD-accelerated matrix-matrix multiplication for 4-bit weights
 *
 * @param out Output matrix
 * @param a 4-bit quantized matrix (packed)
 * @param b Input matrix
 * @param rowsA Number of rows in matrix A
 * @param colsA Number of columns in matrix A
 * @param colsB Number of columns in matrix B
 * @param scaleFactors Scale factors for dequantizing weights
 */
void tinyaiSimdMatMul4BitMM(float *out, const uint8_t *a, const float *b, int rowsA, int colsA,
                            int colsB, const float *scaleFactors);

/**
 * @brief SIMD-accelerated dequantization from 4-bit to float
 *
 * @param out Output float array
 * @param in Input 4-bit packed array
 * @param size Number of elements (after unpacking)
 * @param scaleFactors Scale factors for dequantization
 */
void tinyaiSimdDequantize4Bit(float *out, const uint8_t *in, int size, const float *scaleFactors);

/**
 * @brief SIMD-accelerated quantization from float to 4-bit
 *
 * @param out Output 4-bit packed array
 * @param in Input float array
 * @param size Number of elements (before packing)
 * @param scaleFactors Scale factors for quantization (output)
 * @param blockSize Number of elements per scale factor
 */
void tinyaiSimdQuantize4Bit(uint8_t *out, const float *in, int size, float *scaleFactors,
                            int blockSize);

/**
 * @brief SIMD-accelerated 2D convolution with 4-bit quantized weights
 *
 * @param output Output feature map (outHeight x outWidth x outChannels)
 * @param input Input feature map (inHeight x inWidth x inChannels)
 * @param weights 4-bit quantized weights (kernelSize x kernelSize x inChannels x outChannels)
 * @param biases Bias values for each output channel
 * @param scaleFactors Scale factors for dequantizing weights
 * @param inWidth Width of input feature map
 * @param inHeight Height of input feature map
 * @param inChannels Number of input channels
 * @param outWidth Width of output feature map
 * @param outHeight Height of output feature map
 * @param outChannels Number of output channels
 * @param kernelSize Size of the convolution kernel (assuming square kernel)
 * @param stride Stride of the convolution
 * @param padding Padding size
 */
void tinyaiSimdConv2d4Bit(float *output, const float *input, const uint8_t *weights,
                          const float *biases, const float *scaleFactors, int inWidth, int inHeight,
                          int inChannels, int outWidth, int outHeight, int outChannels,
                          int kernelSize, int stride, int padding);

/**
 * @brief SIMD-accelerated depthwise convolution with 4-bit quantized weights
 *
 * @param output Output feature map (outHeight x outWidth x outChannels)
 * @param input Input feature map (inHeight x inWidth x inChannels)
 * @param weights 4-bit quantized weights (kernelSize x kernelSize x inChannels x multiplier)
 * @param biases Bias values for each output channel
 * @param scaleFactors Scale factors for dequantizing weights
 * @param inWidth Width of input feature map
 * @param inHeight Height of input feature map
 * @param inChannels Number of input channels
 * @param outWidth Width of output feature map
 * @param outHeight Height of output feature map
 * @param multiplier Channel multiplier
 * @param kernelSize Size of the convolution kernel (assuming square kernel)
 * @param stride Stride of the convolution
 * @param padding Padding size
 */
void tinyaiSimdDepthwiseConv2d4Bit(float *output, const float *input, const uint8_t *weights,
                                   const float *biases, const float *scaleFactors, int inWidth,
                                   int inHeight, int inChannels, int outWidth, int outHeight,
                                   int multiplier, int kernelSize, int stride, int padding);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TINYAI_SIMD_OPS_H */
