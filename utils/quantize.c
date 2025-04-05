/**
 * TinyAI Quantization Utilities Implementation
 * 
 * This file implements the quantization utilities for TinyAI.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "../core/memory.h"
#include "../core/io.h"
#include "quantize.h"

/* ----------------- Internal Constants and Variables ----------------- */

/* Activation function lookup tables */
#define ACTIVATION_TABLE_SIZE 8192
static float *activationTableReLU = NULL;
static float *activationTableSigmoid = NULL;
static float *activationTableTanh = NULL;
static float *activationTableGELU = NULL;

/* Activation range for lookup tables */
#define ACTIVATION_MIN -8.0f
#define ACTIVATION_MAX 8.0f

/* ----------------- Matrix Creation and Destruction ----------------- */

/**
 * Create a 4-bit quantized matrix
 */
TinyAIMatrix4bit* tinyaiCreateMatrix4bit(uint32_t rows, uint32_t cols) {
    TinyAIMatrix4bit *matrix = (TinyAIMatrix4bit*)TINYAI_MALLOC(sizeof(TinyAIMatrix4bit));
    if (!matrix) {
        return NULL;
    }
    
    /* Calculate data size (4-bit values, 2 per byte) */
    size_t dataSize = (rows * cols + 1) / 2; /* Ceiling division */
    
    matrix->data = (uint8_t*)TINYAI_MALLOC(dataSize);
    if (!matrix->data) {
        TINYAI_FREE(matrix);
        return NULL;
    }
    
    /* Initialize the matrix */
    memset(matrix->data, 0, dataSize);
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->scale = 1.0f;
    matrix->zeroPoint = 0.0f;
    
    return matrix;
}

/**
 * Destroy a 4-bit quantized matrix
 */
void tinyaiDestroyMatrix4bit(TinyAIMatrix4bit *matrix) {
    if (!matrix) {
        return;
    }
    
    if (matrix->data) {
        TINYAI_FREE(matrix->data);
    }
    
    TINYAI_FREE(matrix);
}

/**
 * Create an 8-bit quantized matrix
 */
TinyAIMatrix8bit* tinyaiCreateMatrix8bit(uint32_t rows, uint32_t cols) {
    TinyAIMatrix8bit *matrix = (TinyAIMatrix8bit*)TINYAI_MALLOC(sizeof(TinyAIMatrix8bit));
    if (!matrix) {
        return NULL;
    }
    
    matrix->data = (int8_t*)TINYAI_MALLOC(rows * cols);
    if (!matrix->data) {
        TINYAI_FREE(matrix);
        return NULL;
    }
    
    /* Initialize the matrix */
    memset(matrix->data, 0, rows * cols);
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->scale = 1.0f;
    matrix->zeroPoint = 0.0f;
    
    return matrix;
}

/**
 * Destroy an 8-bit quantized matrix
 */
void tinyaiDestroyMatrix8bit(TinyAIMatrix8bit *matrix) {
    if (!matrix) {
        return;
    }
    
    if (matrix->data) {
        TINYAI_FREE(matrix->data);
    }
    
    TINYAI_FREE(matrix);
}

/**
 * Create a FP32 matrix
 */
TinyAIMatrixFP32* tinyaiCreateMatrixFP32(uint32_t rows, uint32_t cols) {
    TinyAIMatrixFP32 *matrix = (TinyAIMatrixFP32*)TINYAI_MALLOC(sizeof(TinyAIMatrixFP32));
    if (!matrix) {
        return NULL;
    }
    
    matrix->data = (float*)TINYAI_MALLOC(rows * cols * sizeof(float));
    if (!matrix->data) {
        TINYAI_FREE(matrix);
        return NULL;
    }
    
    /* Initialize the matrix */
    memset(matrix->data, 0, rows * cols * sizeof(float));
    matrix->rows = rows;
    matrix->cols = cols;
    
    return matrix;
}

/**
 * Destroy a FP32 matrix
 */
void tinyaiDestroyMatrixFP32(TinyAIMatrixFP32 *matrix) {
    if (!matrix) {
        return;
    }
    
    if (matrix->data) {
        TINYAI_FREE(matrix->data);
    }
    
    TINYAI_FREE(matrix);
}

/* ----------------- Quantization and Dequantization ----------------- */

/**
 * Quantize a FP32 matrix to 4-bit
 */
TinyAIMatrix4bit* tinyaiQuantizeFP32To4bit(const TinyAIMatrixFP32 *input) {
    if (!input || !input->data) {
        return NULL;
    }
    
    TinyAIMatrix4bit *output = tinyaiCreateMatrix4bit(input->rows, input->cols);
    if (!output) {
        return NULL;
    }
    
    /* Find min and max values */
    float minVal = FLT_MAX;
    float maxVal = -FLT_MAX;
    
    size_t size = input->rows * input->cols;
    for (size_t i = 0; i < size; i++) {
        if (input->data[i] < minVal) minVal = input->data[i];
        if (input->data[i] > maxVal) maxVal = input->data[i];
    }
    
    /* Compute scale and zero point */
    output->zeroPoint = minVal;
    output->scale = (maxVal - minVal) / 15.0f; /* 15 is the max value in 4 bits */
    
    if (output->scale == 0) {
        /* All values are the same */
        output->scale = 1.0f;
    }
    
    /* Quantize the values */
    for (size_t i = 0; i < size; i += 2) {
        /* Quantize first value */
        int val1 = (int)((input->data[i] - output->zeroPoint) / output->scale + 0.5f);
        if (val1 < 0) val1 = 0;
        if (val1 > 15) val1 = 15;
        
        /* Quantize second value (or use 0 if at the end) */
        int val2 = 0;
        if (i + 1 < size) {
            val2 = (int)((input->data[i + 1] - output->zeroPoint) / output->scale + 0.5f);
            if (val2 < 0) val2 = 0;
            if (val2 > 15) val2 = 15;
        }
        
        /* Pack two 4-bit values into one byte */
        output->data[i / 2] = (val1 << 4) | val2;
    }
    
    return output;
}

/**
 * Quantize a FP32 matrix to 8-bit
 */
TinyAIMatrix8bit* tinyaiQuantizeFP32To8bit(const TinyAIMatrixFP32 *input) {
    if (!input || !input->data) {
        return NULL;
    }
    
    TinyAIMatrix8bit *output = tinyaiCreateMatrix8bit(input->rows, input->cols);
    if (!output) {
        return NULL;
    }
    
    /* Find min and max values */
    float minVal = FLT_MAX;
    float maxVal = -FLT_MAX;
    
    size_t size = input->rows * input->cols;
    for (size_t i = 0; i < size; i++) {
        if (input->data[i] < minVal) minVal = input->data[i];
        if (input->data[i] > maxVal) maxVal = input->data[i];
    }
    
    /* Compute scale and zero point */
    output->zeroPoint = minVal;
    output->scale = (maxVal - minVal) / 254.0f; /* Use -127 to 127 range for better precision */
    
    if (output->scale == 0) {
        /* All values are the same */
        output->scale = 1.0f;
    }
    
    /* Quantize the values */
    for (size_t i = 0; i < size; i++) {
        int val = (int)((input->data[i] - output->zeroPoint) / output->scale + 0.5f);
        if (val < -127) val = -127;
        if (val > 127) val = 127;
        
        output->data[i] = (int8_t)val;
    }
    
    return output;
}

/**
 * Dequantize a 4-bit matrix to FP32
 */
TinyAIMatrixFP32* tinyaiDequantize4bitToFP32(const TinyAIMatrix4bit *input) {
    if (!input || !input->data) {
        return NULL;
    }
    
    TinyAIMatrixFP32 *output = tinyaiCreateMatrixFP32(input->rows, input->cols);
    if (!output) {
        return NULL;
    }
    
    size_t size = input->rows * input->cols;
    for (size_t i = 0; i < size; i += 2) {
        /* Unpack first value */
        int val1 = (input->data[i / 2] >> 4) & 0x0F;
        output->data[i] = val1 * input->scale + input->zeroPoint;
        
        /* Unpack second value (if not at the end) */
        if (i + 1 < size) {
            int val2 = input->data[i / 2] & 0x0F;
            output->data[i + 1] = val2 * input->scale + input->zeroPoint;
        }
    }
    
    return output;
}

/**
 * Dequantize an 8-bit matrix to FP32
 */
TinyAIMatrixFP32* tinyaiDequantize8bitToFP32(const TinyAIMatrix8bit *input) {
    if (!input || !input->data) {
        return NULL;
    }
    
    TinyAIMatrixFP32 *output = tinyaiCreateMatrixFP32(input->rows, input->cols);
    if (!output) {
        return NULL;
    }
    
    size_t size = input->rows * input->cols;
    for (size_t i = 0; i < size; i++) {
        output->data[i] = input->data[i] * input->scale + input->zeroPoint;
    }
    
    return output;
}

/* ----------------- Matrix Operations ----------------- */

/**
 * Matrix multiplication: C = A * B
 */
int tinyaiMatrixMultiply(const void *a, const void *b, void *c, TinyAIPrecision precision) {
    /* Implementation varies based on precision */
    switch (precision) {
        case TINYAI_PRECISION_FP32: {
            const TinyAIMatrixFP32 *A = (const TinyAIMatrixFP32 *)a;
            const TinyAIMatrixFP32 *B = (const TinyAIMatrixFP32 *)b;
            TinyAIMatrixFP32 *C = (TinyAIMatrixFP32 *)c;
            
            if (!A || !B || !C || !A->data || !B->data || !C->data) {
                return -1;
            }
            
            if (A->cols != B->rows || C->rows != A->rows || C->cols != B->cols) {
                return -1;  /* Incompatible dimensions */
            }
            
            /* Compute matrix multiplication */
            for (uint32_t i = 0; i < A->rows; i++) {
                for (uint32_t j = 0; j < B->cols; j++) {
                    float sum = 0.0f;
                    for (uint32_t k = 0; k < A->cols; k++) {
                        sum += A->data[i * A->cols + k] * B->data[k * B->cols + j];
                    }
                    C->data[i * C->cols + j] = sum;
                }
            }
            
            return 0;
        }
        
        case TINYAI_PRECISION_INT8: {
            /* Simplified implementation for 8-bit quantized matrices */
            /* For a real implementation, we would use SIMD instructions and optimized code */
            const TinyAIMatrix8bit *A = (const TinyAIMatrix8bit *)a;
            const TinyAIMatrix8bit *B = (const TinyAIMatrix8bit *)b;
            TinyAIMatrix8bit *C = (TinyAIMatrix8bit *)c;
            
            if (!A || !B || !C || !A->data || !B->data || !C->data) {
                return -1;
            }
            
            if (A->cols != B->rows || C->rows != A->rows || C->cols != B->cols) {
                return -1;  /* Incompatible dimensions */
            }
            
            /* Compute scale for output */
            C->scale = A->scale * B->scale;
            C->zeroPoint = 0.0f;
            
            /* Compute matrix multiplication */
            for (uint32_t i = 0; i < A->rows; i++) {
                for (uint32_t j = 0; j < B->cols; j++) {
                    int32_t sum = 0;
                    for (uint32_t k = 0; k < A->cols; k++) {
                        sum += (int32_t)A->data[i * A->cols + k] * (int32_t)B->data[k * B->cols + j];
                    }
                    
                    /* Clip to INT8 range */
                    if (sum < -127) sum = -127;
                    if (sum > 127) sum = 127;
                    
                    C->data[i * C->cols + j] = (int8_t)sum;
                }
            }
            
            return 0;
        }
        
        case TINYAI_PRECISION_INT4: {
            /* Simplified implementation for 4-bit quantized matrices */
            /* This is a naive implementation for demonstration purposes */
            const TinyAIMatrix4bit *A = (const TinyAIMatrix4bit *)a;
            const TinyAIMatrix4bit *B = (const TinyAIMatrix4bit *)b;
            TinyAIMatrix4bit *C = (TinyAIMatrix4bit *)c;
            
            if (!A || !B || !C || !A->data || !B->data || !C->data) {
                return -1;
            }
            
            if (A->cols != B->rows || C->rows != A->rows || C->cols != B->cols) {
                return -1;  /* Incompatible dimensions */
            }
            
            /* For a real implementation, we would dequantize, compute, and requantize */
            /* This is too complex for this simplified version */
            
            /* Instead, we'll just dequantize to FP32, compute, and requantize */
            TinyAIMatrixFP32 *Afp32 = tinyaiDequantize4bitToFP32(A);
            TinyAIMatrixFP32 *Bfp32 = tinyaiDequantize4bitToFP32(B);
            TinyAIMatrixFP32 *Cfp32 = tinyaiCreateMatrixFP32(C->rows, C->cols);
            
            if (!Afp32 || !Bfp32 || !Cfp32) {
                if (Afp32) tinyaiDestroyMatrixFP32(Afp32);
                if (Bfp32) tinyaiDestroyMatrixFP32(Bfp32);
                if (Cfp32) tinyaiDestroyMatrixFP32(Cfp32);
                return -1;
            }
            
            /* Compute in FP32 */
            tinyaiMatrixMultiply(Afp32, Bfp32, Cfp32, TINYAI_PRECISION_FP32);
            
            /* Requantize */
            TinyAIMatrix4bit *Cnew = tinyaiQuantizeFP32To4bit(Cfp32);
            if (!Cnew) {
                tinyaiDestroyMatrixFP32(Afp32);
                tinyaiDestroyMatrixFP32(Bfp32);
                tinyaiDestroyMatrixFP32(Cfp32);
                return -1;
            }
            
            /* Copy results back to C */
            size_t dataSize = (C->rows * C->cols + 1) / 2;
            memcpy(C->data, Cnew->data, dataSize);
            C->scale = Cnew->scale;
            C->zeroPoint = Cnew->zeroPoint;
            
            /* Clean up */
            tinyaiDestroyMatrixFP32(Afp32);
            tinyaiDestroyMatrixFP32(Bfp32);
            tinyaiDestroyMatrixFP32(Cfp32);
            tinyaiDestroyMatrix4bit(Cnew);
            
            return 0;
        }
        
        default:
            return -1;  /* Unknown precision */
    }
}

/**
 * Matrix addition: C = A + B
 */
int tinyaiMatrixAdd(const void *a, const void *b, void *c, TinyAIPrecision precision) {
    /* Implementation varies based on precision */
    switch (precision) {
        case TINYAI_PRECISION_FP32: {
            const TinyAIMatrixFP32 *A = (const TinyAIMatrixFP32 *)a;
            const TinyAIMatrixFP32 *B = (const TinyAIMatrixFP32 *)b;
            TinyAIMatrixFP32 *C = (TinyAIMatrixFP32 *)c;
            
            if (!A || !B || !C || !A->data || !B->data || !C->data) {
                return -1;
            }
            
            if (A->rows != B->rows || A->cols != B->cols ||
                C->rows != A->rows || C->cols != A->cols) {
                return -1;  /* Incompatible dimensions */
            }
            
            /* Compute matrix addition */
            size_t size = A->rows * A->cols;
            for (size_t i = 0; i < size; i++) {
                C->data[i] = A->data[i] + B->data[i];
            }
            
            return 0;
        }
        
        /* Other precision implementations follow a similar pattern */
        /* For brevity, they're not included here */
        
        default:
            return -1;  /* Unknown precision */
    }
}

/**
 * Apply activation function to matrix
 */
int tinyaiMatrixActivation(const void *input, void *output, int activation, TinyAIPrecision precision) {
    /* Implementation varies based on precision */
    switch (precision) {
        case TINYAI_PRECISION_FP32: {
            const TinyAIMatrixFP32 *in = (const TinyAIMatrixFP32 *)input;
            TinyAIMatrixFP32 *out = (TinyAIMatrixFP32 *)output;
            
            if (!in || !out || !in->data || !out->data) {
                return -1;
            }
            
            if (in->rows != out->rows || in->cols != out->cols) {
                return -1;  /* Incompatible dimensions */
            }
            
            /* Apply activation function */
            size_t size = in->rows * in->cols;
            switch (activation) {
                case 0:  /* None */
                    memcpy(out->data, in->data, size * sizeof(float));
                    break;
                
                case 1:  /* ReLU */
                    for (size_t i = 0; i < size; i++) {
                        out->data[i] = tinyaiActivationReLU(in->data[i]);
                    }
                    break;
                
                case 2:  /* Sigmoid */
                    for (size_t i = 0; i < size; i++) {
                        out->data[i] = tinyaiActivationSigmoid(in->data[i]);
                    }
                    break;
                
                case 3:  /* Tanh */
                    for (size_t i = 0; i < size; i++) {
                        out->data[i] = tinyaiActivationTanh(in->data[i]);
                    }
                    break;
                
                case 4:  /* GELU */
                    for (size_t i = 0; i < size; i++) {
                        out->data[i] = tinyaiActivationGELU(in->data[i]);
                    }
                    break;
                
                default:
                    return -1;  /* Unknown activation */
            }
            
            return 0;
        }
        
        /* Other precision implementations follow a similar pattern */
        /* For brevity, they're not included here */
        
        default:
            return -1;  /* Unknown precision */
    }
}

/* ----------------- Vector Operations ----------------- */

/**
 * Vector dot product
 */
float tinyaiVectorDot(const void *a, const void *b, uint32_t length, TinyAIPrecision precision) {
    /* Implementation varies based on precision */
    switch (precision) {
        case TINYAI_PRECISION_FP32: {
            const float *A = (const float *)a;
            const float *B = (const float *)b;
            
            if (!A || !B) {
                return 0.0f;
            }
            
            /* Compute dot product */
            float sum = 0.0f;
            for (uint32_t i = 0; i < length; i++) {
                sum += A[i] * B[i];
            }
            
            return sum;
        }
        
        /* Other precision implementations follow a similar pattern */
        /* For brevity, they're not included here */
        
        default:
            return 0.0f;  /* Unknown precision */
    }
}

/**
 * Vector L2 norm (Euclidean distance)
 */
float tinyaiVectorL2Norm(const void *a, uint32_t length, TinyAIPrecision precision) {
    /* Implementation varies based on precision */
    switch (precision) {
        case TINYAI_PRECISION_FP32: {
            const float *A = (const float *)a;
            
            if (!A) {
                return 0.0f;
            }
            
            /* Compute L2 norm */
            float sum = 0.0f;
            for (uint32_t i = 0; i < length; i++) {
                sum += A[i] * A[i];
            }
            
            return sqrtf(sum);
        }
        
        /* Other precision implementations follow a similar pattern */
        /* For brevity, they're not included here */
        
        default:
            return 0.0f;  /* Unknown precision */
    }
}

/**
 * Vector cosine similarity
 */
float tinyaiVectorCosineSimilarity(const void *a, const void *b, uint32_t length, TinyAIPrecision precision) {
    /* Implementation varies based on precision */
    switch (precision) {
        case TINYAI_PRECISION_FP32: {
            const float *A = (const float *)a;
            const float *B = (const float *)b;
            
            if (!A || !B) {
                return 0.0f;
            }
            
            /* Compute cosine similarity */
            float dot = tinyaiVectorDot(A, B, length, precision);
            float normA = tinyaiVectorL2Norm(A, length, precision);
            float normB = tinyaiVectorL2Norm(B, length, precision);
            
            if (normA == 0.0f || normB == 0.0f) {
                return 0.0f;
            }
            
            return dot / (normA * normB);
        }
        
        /* Other precision implementations follow a similar pattern */
        /* For brevity, they're not included here */
        
        default:
            return 0.0f;  /* Unknown precision */
    }
}

/* ----------------- Activation Functions ----------------- */

/**
 * ReLU activation function
 */
float tinyaiActivationReLU(float x) {
    return x > 0.0f ? x : 0.0f;
}

/**
 * Sigmoid activation function
 */
float tinyaiActivationSigmoid(float x) {
    /* Check for lookup table */
    if (activationTableSigmoid) {
        /* Convert x to table index */
        int index = (int)((x - ACTIVATION_MIN) / (ACTIVATION_MAX - ACTIVATION_MIN) * ACTIVATION_TABLE_SIZE);
        if (index < 0) index = 0;
        if (index >= ACTIVATION_TABLE_SIZE) index = ACTIVATION_TABLE_SIZE - 1;
        
        return activationTableSigmoid[index];
    }
    
    /* Compute directly */
    if (x < -10.0f) return 0.0f;
    if (x > 10.0f) return 1.0f;
    return 1.0f / (1.0f + expf(-x));
}

/**
 * Tanh activation function
 */
float tinyaiActivationTanh(float x) {
    /* Check for lookup table */
    if (activationTableTanh) {
        /* Convert x to table index */
        int index = (int)((x - ACTIVATION_MIN) / (ACTIVATION_MAX - ACTIVATION_MIN) * ACTIVATION_TABLE_SIZE);
        if (index < 0) index = 0;
        if (index >= ACTIVATION_TABLE_SIZE) index = ACTIVATION_TABLE_SIZE - 1;
        
        return activationTableTanh[index];
    }
    
    /* Compute directly */
    if (x < -5.0f) return -1.0f;
    if (x > 5.0f) return 1.0f;
    float ex = expf(x);
    float enx = expf(-x);
    return (ex - enx) / (ex + enx);
}

/**
 * GELU activation function
 */
float tinyaiActivationGELU(float x) {
    /* Check for lookup table */
    if (activationTableGELU) {
        /* Convert x to table index */
        int index = (int)((x - ACTIVATION_MIN) / (ACTIVATION_MAX - ACTIVATION_MIN) * ACTIVATION_TABLE_SIZE);
        if (index < 0) index = 0;
        if (index >= ACTIVATION_TABLE_SIZE) index = ACTIVATION_TABLE_SIZE - 1;
        
        return activationTableGELU[index];
    }
    
    /* Compute directly (approximate) */
    return 0.5f * x * (1.0f + tinyaiActivationTanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

/**
 * Initialize activation function lookup tables
 */
int tinyaiInitActivationTables() {
    /* Allocate tables */
    activationTableReLU = (float*)TINYAI_MALLOC(ACTIVATION_TABLE_SIZE * sizeof(float));
    activationTableSigmoid = (float*)TINYAI_MALLOC(ACTIVATION_TABLE_SIZE * sizeof(float));
    activationTableTanh = (float*)TINYAI_MALLOC(ACTIVATION_TABLE_SIZE * sizeof(float));
    activationTableGELU = (float*)TINYAI_MALLOC(ACTIVATION_TABLE_SIZE * sizeof(float));
    
    if (!activationTableReLU || !activationTableSigmoid ||
        !activationTableTanh || !activationTableGELU) {
        tinyaiCleanupActivationTables();
        return -1;
    }
    
    /* Fill tables */
    for (int i = 0; i < ACTIVATION_TABLE_SIZE; i++) {
        float x = ACTIVATION_MIN + (ACTIVATION_MAX - ACTIVATION_MIN) * i / ACTIVATION_TABLE_SIZE;
        
        /* ReLU */
        activationTableReLU[i] = x > 0.0f ? x : 0.0f;
        
        /* Sigmoid */
        activationTableSigmoid[i] = 1.0f / (1.0f + expf(-x));
        
        /* Tanh */
        {
            float ex = expf(x);
            float enx = expf(-x);
            activationTableTanh[i] = (ex - enx) / (ex + enx);
        }
        
        /* GELU (approximate) */
        activationTableGELU[i] = 0.5f * x * (1.0f + activationTableTanh[i]); 
    }
    
    return 0;
}

/**
 * Clean up activation function lookup tables
 */
void tinyaiCleanupActivationTables() {
    if (activationTableReLU) {
        TINYAI_FREE(activationTableReLU);
        activationTableReLU = NULL;
    }
    
    if (activationTableSigmoid) {
        TINYAI_FREE(activationTableSigmoid);
        activationTableSigmoid = NULL;
    }
    
    if (activationTableTanh) {
        TINYAI_FREE(activationTableTanh);
        activationTableTanh = NULL;
    }
    
    if (activationTableGELU) {
        TINYAI_FREE(activationTableGELU);
        activationTableGELU = NULL;
    }
}

/* ----------------- Utility Functions ----------------- */

/**
 * Find minimum and maximum values in FP32 array
 */
void tinyaiFindMinMax(const float *data, size_t size, float *min, float *max) {
    if (!data || !min || !max || size == 0) {
        return;
    }
    
    float minVal = FLT_MAX;
    float maxVal = -FLT_MAX;
    
    for (size_t i = 0; i < size; i++) {
        if (data[i] < minVal) minVal = data[i];
        if (data[i] > maxVal) maxVal = data[i];
    }
    
    *min = minVal;
    *max = maxVal;
}

/**
 * Save a quantized matrix to a file
 */
int tinyaiSaveQuantizedMatrix(const void *matrix, const char *path, TinyAIPrecision precision) {
    if (!matrix || !path) {
        return -1;
    }
    
    TinyAIFile *file = tinyaiOpenFile(path, TINYAI_FILE_WRITE | TINYAI_FILE_BINARY | TINYAI_FILE_CREATE);
    if (!file) {
        return -1;
    }
    
    /* Write header with magic number, precision, and dimensions */
    uint32_t magic = 0x4D51544E; /* "TQNM" - TinyAI Quantized Matrix */
    tinyaiWriteFile(file, &magic, sizeof(magic));
    tinyaiWriteFile(file, &precision, sizeof(precision));
    
    switch (precision) {
        case TINYAI_PRECISION_FP32: {
            const TinyAIMatrixFP32 *mat = (const TinyAIMatrixFP32 *)matrix;
            tinyaiWriteFile(file, &mat->rows, sizeof(mat->rows));
            tinyaiWriteFile(file, &mat->cols, sizeof(mat->cols));
            
            /* Write the data */
            size_t dataSize = mat->rows * mat->cols * sizeof(float);
            tinyaiWriteFile(file, mat->data, dataSize);
            break;
        }
        
        case TINYAI_PRECISION_INT8: {
            const TinyAIMatrix8bit *mat = (const TinyAIMatrix8bit *)matrix;
            tinyaiWriteFile(file, &mat->rows, sizeof(mat->rows));
            tinyaiWriteFile(file, &mat->cols, sizeof(mat->cols));
            tinyaiWriteFile(file, &mat->scale, sizeof(mat->scale));
            tinyaiWriteFile(file, &mat->zeroPoint, sizeof(mat->zeroPoint));
            
            /* Write the data */
            size_t dataSize = mat->rows * mat->cols;
            tinyaiWriteFile(file, mat->data, dataSize);
            break;
        }
        
        case TINYAI_PRECISION_INT4: {
            const TinyAIMatrix4bit *mat = (const TinyAIMatrix4bit *)matrix;
            tinyaiWriteFile(file, &mat->rows, sizeof(mat->rows));
            tinyaiWriteFile(file, &mat->cols, sizeof(mat->cols));
            tinyaiWriteFile(file, &mat->scale, sizeof(mat->scale));
            tinyaiWriteFile(file, &mat->zeroPoint, sizeof(mat->zeroPoint));
            
            /* Write the data (4-bit packed, 2 values per byte) */
            size_t dataSize = (mat->rows * mat->cols + 1) / 2;
            tinyaiWriteFile(file, mat->data, dataSize);
            break;
        }
        
        default:
            tinyaiCloseFile(file);
            return -1;  /* Unknown precision */
    }
    
    tinyaiCloseFile(file);
    return 0;
}

/**
 * Load a quantized matrix from a file
 */
void* tinyaiLoadQuantizedMatrix(const char *path, TinyAIPrecision precision) {
    if (!path) {
        return NULL;
    }
    
    TinyAIFile *file = tinyaiOpenFile(path, TINYAI_FILE_READ | TINYAI_FILE_BINARY);
    if (!file) {
        return NULL;
    }
    
    /* Read and verify header */
    uint32_t magic;
    TinyAIPrecision filePrecision;
    
    if (tinyaiReadFile(file, &magic, sizeof(magic)) != sizeof(magic) ||
        magic != 0x4D51544E) {
        tinyaiCloseFile(file);
        return NULL;  /* Invalid magic number */
    }
    
    if (tinyaiReadFile(file, &filePrecision, sizeof(filePrecision)) != sizeof(filePrecision) ||
        filePrecision != precision) {
        tinyaiCloseFile(file);
        return NULL;  /* Precision mismatch */
    }
    
    /* Read data based on precision */
    void *result = NULL;
    
    switch (precision) {
        case TINYAI_PRECISION_FP32: {
            uint32_t rows, cols;
            
            if (tinyaiReadFile(file, &rows, sizeof(rows)) != sizeof(rows) ||
                tinyaiReadFile(file, &cols, sizeof(cols)) != sizeof(cols)) {
                tinyaiCloseFile(file);
                return NULL;
            }
            
            TinyAIMatrixFP32 *mat = tinyaiCreateMatrixFP32(rows, cols);
            if (!mat) {
                tinyaiCloseFile(file);
                return NULL;
            }
            
            /* Read the data */
            size_t dataSize = rows * cols * sizeof(float);
            if (tinyaiReadFile(file, mat->data, dataSize) != dataSize) {
                tinyaiDestroyMatrixFP32(mat);
                tinyaiCloseFile(file);
                return NULL;
            }
            
            result = mat;
            break;
        }
        
        case TINYAI_PRECISION_INT8: {
            uint32_t rows, cols;
            float scale, zeroPoint;
            
            if (tinyaiReadFile(file, &rows, sizeof(rows)) != sizeof(rows) ||
                tinyaiReadFile(file, &cols, sizeof(cols)) != sizeof(cols) ||
                tinyaiReadFile(file, &scale, sizeof(scale)) != sizeof(scale) ||
                tinyaiReadFile(file, &zeroPoint, sizeof(zeroPoint)) != sizeof(zeroPoint)) {
                tinyaiCloseFile(file);
                return NULL;
            }
            
            TinyAIMatrix8bit *mat = tinyaiCreateMatrix8bit(rows, cols);
            if (!mat) {
                tinyaiCloseFile(file);
                return NULL;
            }
            
            mat->scale = scale;
            mat->zeroPoint = zeroPoint;
            
            /* Read the data */
            size_t dataSize = rows * cols;
            if (tinyaiReadFile(file, mat->data, dataSize) != dataSize) {
                tinyaiDestroyMatrix8bit(mat);
                tinyaiCloseFile(file);
                return NULL;
            }
            
            result = mat;
            break;
        }
        
        case TINYAI_PRECISION_INT4: {
            uint32_t rows, cols;
            float scale, zeroPoint;
            
            if (tinyaiReadFile(file, &rows, sizeof(rows)) != sizeof(rows) ||
                tinyaiReadFile(file, &cols, sizeof(cols)) != sizeof(cols) ||
                tinyaiReadFile(file, &scale, sizeof(scale)) != sizeof(scale) ||
                tinyaiReadFile(file, &zeroPoint, sizeof(zeroPoint)) != sizeof(zeroPoint)) {
                tinyaiCloseFile(file);
                return NULL;
            }
            
            TinyAIMatrix4bit *mat = tinyaiCreateMatrix4bit(rows, cols);
            if (!mat) {
                tinyaiCloseFile(file);
                return NULL;
            }
            
            mat->scale = scale;
            mat->zeroPoint = zeroPoint;
            
            /* Read the data */
            size_t dataSize = (rows * cols + 1) / 2;
            if (tinyaiReadFile(file, mat->data, dataSize) != dataSize) {
                tinyaiDestroyMatrix4bit(mat);
                tinyaiCloseFile(file);
                return NULL;
            }
            
            result = mat;
            break;
        }
        
        default:
            break;  /* Unknown precision */
    }
    
    tinyaiCloseFile(file);
    return result;
}
