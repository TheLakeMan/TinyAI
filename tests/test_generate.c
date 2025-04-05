/**
 * TinyAI Text Generation Tests
 */

#include "../core/memory.h"           // For memory functions
#include "../models/text/generate.h"  // Include the generation module being tested
#include "../models/text/tokenizer.h" // For tokenization
#include "../utils/quantize.h"        // For matrix quantization helpers
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Basic assertion helper (consistent with other test files)
#define ASSERT(condition, message)                                                                 \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            fprintf(stderr, "Assertion Failed: %s (%s:%d)\n", message, __FILE__, __LINE__);        \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

// Helper to create a simple tokenizer for testing
TinyAITokenizer *create_test_tokenizer()
{
    TinyAITokenizer *tokenizer = tinyaiCreateTokenizer();

    // Add some basic tokens
    tinyaiAddToken(tokenizer, "the", 1000);
    tinyaiAddToken(tokenizer, "quick", 500);
    tinyaiAddToken(tokenizer, "brown", 400);
    tinyaiAddToken(tokenizer, "fox", 300);
    tinyaiAddToken(tokenizer, "jumps", 200);
    tinyaiAddToken(tokenizer, "over", 150);
    tinyaiAddToken(tokenizer, "lazy", 100);
    tinyaiAddToken(tokenizer, "dog", 90);
    tinyaiAddToken(tokenizer, ".", 80);

    return tokenizer;
}

// Helper to create a mock matrix
TinyAIMatrixFP32 *create_mock_matrix(int rows, int cols)
{
    TinyAIMatrixFP32 *matrix = (TinyAIMatrixFP32 *)TINYAI_MALLOC(sizeof(TinyAIMatrixFP32));
    if (!matrix)
        return NULL;

    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data = (float *)TINYAI_MALLOC(rows * cols * sizeof(float));

    if (!matrix->data) {
        TINYAI_FREE(matrix);
        return NULL;
    }

    // Fill with some pattern for testing
    for (int i = 0; i < rows * cols; i++) {
        matrix->data[i] = (float)(i % 10) / 10.0f;
    }

    return matrix;
}

// Helper to free a mock matrix
void free_mock_matrix(TinyAIMatrixFP32 *matrix)
{
    if (matrix) {
        if (matrix->data) {
            TINYAI_FREE(matrix->data);
        }
        TINYAI_FREE(matrix);
    }
}

// Test creating and destroying a model
void test_model_create_destroy()
{
    printf("  Testing model creation/destruction...\n");

    TinyAITokenizer *tokenizer = create_test_tokenizer();
    ASSERT(tokenizer != NULL, "create_test_tokenizer() should return non-NULL");

    // Create model
    uint32_t     hiddenSize  = 64;
    uint32_t     contextSize = 128;
    TinyAIModel *model =
        tinyaiCreateModel(TINYAI_MODEL_TYPE_RNN, hiddenSize, contextSize, tokenizer);
    ASSERT(model != NULL, "tinyaiCreateModel() should return non-NULL");
    ASSERT(model->hiddenSize == hiddenSize, "Model should have correct hidden size");
    ASSERT(model->contextSize == contextSize, "Model should have correct context size");
    ASSERT(model->tokenizer == tokenizer, "Model should reference the provided tokenizer");

    // Destroy model (should not free tokenizer)
    tinyaiDestroyModel(model);

    // Test that tokenizer is still valid
    const char *unknownStr = tinyaiGetTokenString(tokenizer, TINYAI_TOKEN_UNKNOWN);
    ASSERT(unknownStr != NULL, "Tokenizer should still be valid after model destruction");

    tinyaiDestroyTokenizer(tokenizer);
    printf("    PASS\n");
}

// Test adding layers to a model
void test_add_layers()
{
    printf("  Testing adding layers to model...\n");

    TinyAITokenizer *tokenizer = create_test_tokenizer();
    TinyAIModel     *model     = tinyaiCreateModel(TINYAI_MODEL_TYPE_RNN, 64, 128, tokenizer);

    // Add embedding layer
    int result1 = tinyaiAddLayer(model, TINYAI_LAYER_EMBEDDING, tokenizer->tokenCount, 64,
                                 TINYAI_ACTIVATION_NONE);
    ASSERT(result1 == 0, "Adding embedding layer should succeed");
    ASSERT(model->layerCount == 1, "Model should have 1 layer after adding embedding layer");

    // Add hidden layer
    int result2 = tinyaiAddLayer(model, TINYAI_LAYER_DENSE, 64, 64, TINYAI_ACTIVATION_RELU);
    ASSERT(result2 == 0, "Adding hidden layer should succeed");
    ASSERT(model->layerCount == 2, "Model should have 2 layers after adding hidden layer");

    // Add output layer
    int result3 = tinyaiAddLayer(model, TINYAI_LAYER_OUTPUT, 64, tokenizer->tokenCount,
                                 TINYAI_ACTIVATION_NONE);
    ASSERT(result3 == 0, "Adding output layer should succeed");
    ASSERT(model->layerCount == 3, "Model should have 3 layers after adding output layer");

    // Check layer properties
    ASSERT(model->layers[0].type == TINYAI_LAYER_EMBEDDING, "Layer 0 should be embedding layer");
    ASSERT(model->layers[1].type == TINYAI_LAYER_DENSE, "Layer 1 should be dense layer");
    ASSERT(model->layers[2].type == TINYAI_LAYER_OUTPUT, "Layer 2 should be output layer");

    ASSERT(model->layers[1].activation == TINYAI_ACTIVATION_RELU,
           "Layer 1 should have ReLU activation");

    tinyaiDestroyModel(model);
    tinyaiDestroyTokenizer(tokenizer);
    printf("    PASS\n");
}

// Test the model forward pass (simple case)
void test_model_forward_simple()
{
    printf("  Testing simple model forward pass...\n");

    TinyAITokenizer *tokenizer   = create_test_tokenizer();
    uint32_t         hiddenSize  = 4; // Small for testing
    uint32_t         contextSize = 8; // Small for testing
    TinyAIModel     *model =
        tinyaiCreateModel(TINYAI_MODEL_TYPE_RNN, hiddenSize, contextSize, tokenizer);

    // Add layers (minimal for testing)
    tinyaiAddLayer(model, TINYAI_LAYER_EMBEDDING, tokenizer->tokenCount, hiddenSize,
                   TINYAI_ACTIVATION_NONE);
    tinyaiAddLayer(model, TINYAI_LAYER_OUTPUT, hiddenSize, tokenizer->tokenCount,
                   TINYAI_ACTIVATION_NONE);

    // Create some dummy weights for embedding layer
    TinyAIMatrixFP32 *embedMatrix = create_mock_matrix(tokenizer->tokenCount, hiddenSize);
    ASSERT(embedMatrix != NULL, "Should create mock embedding matrix");

    // Quantize to 4-bit
    TinyAIMatrix4bit *quantized = tinyaiQuantizeFP32To4bit(embedMatrix);
    ASSERT(quantized != NULL, "Should quantize embedding matrix");

    // Set weights for embedding layer
    model->layers[0].weights = *quantized; // Copy the struct

    // Create some dummy weights for output layer
    TinyAIMatrixFP32 *outputMatrix = create_mock_matrix(hiddenSize, tokenizer->tokenCount);
    ASSERT(outputMatrix != NULL, "Should create mock output matrix");

    // Quantize to 4-bit
    TinyAIMatrix4bit *quantizedOutput = tinyaiQuantizeFP32To4bit(outputMatrix);
    ASSERT(quantizedOutput != NULL, "Should quantize output matrix");

    // Set weights for output layer
    model->layers[1].weights = *quantizedOutput; // Copy the struct

    // Now test forward pass
    int    inputTokens[3] = {TINYAI_TOKEN_BOS, 5, 7}; // Some token IDs
    float *outputLogits   = (float *)TINYAI_MALLOC(tokenizer->tokenCount * sizeof(float));
    ASSERT(outputLogits != NULL, "Should allocate output logits");

    int result = tinyaiModelForward(model, inputTokens, 3, outputLogits);
    ASSERT(result == 0, "Forward pass should succeed");

    // Check that we have some output (can't check specific values easily)
    int nonZeroFound = 0;
    for (uint32_t i = 0; i < tokenizer->tokenCount; i++) {
        if (outputLogits[i] != 0.0f) {
            nonZeroFound = 1;
            break;
        }
    }
    ASSERT(nonZeroFound, "Forward pass should produce non-zero outputs");

    // Clean up
    TINYAI_FREE(outputLogits);
    TINYAI_FREE(quantized);       // Free the struct but not the data (owned by model now)
    TINYAI_FREE(quantizedOutput); // Free the struct but not the data (owned by model now)
    free_mock_matrix(embedMatrix);
    free_mock_matrix(outputMatrix);
    tinyaiDestroyModel(model);
    tinyaiDestroyTokenizer(tokenizer);
    printf("    PASS\n");
}

// Test softmax temperature application
void test_softmax_temperature()
{
    printf("  Testing softmax and temperature application...\n");

    // Create logits for testing
    float logits[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float logits_copy[4];

    // Test various temperatures using the sampleToken function
    TinyAIGenerationParams params;
    params.maxTokens      = 10;
    params.samplingMethod = TINYAI_SAMPLING_TEMPERATURE;
    params.seed           = 42; // Fixed for reproducibility

    // Test with high temperature (more random)
    memcpy(logits_copy, logits, sizeof(logits));
    params.temperature = 1.5f;
    int token1         = tinyaiSampleToken(logits_copy, 4, &params);
    ASSERT(token1 >= 0 && token1 < 4, "Sampled token should be in range");

    // Test with low temperature (more deterministic)
    memcpy(logits_copy, logits, sizeof(logits));
    params.temperature = 0.1f;
    int token2         = tinyaiSampleToken(logits_copy, 4, &params);
    ASSERT(token2 >= 0 && token2 < 4, "Sampled token should be in range");

    // With very low temperature, it should pick the highest logit consistently
    memcpy(logits_copy, logits, sizeof(logits));
    params.temperature = 0.01f;
    int token3         = tinyaiSampleToken(logits_copy, 4, &params);
    ASSERT(token3 == 3, "With very low temperature, should pick highest logit (index 3)");

    printf("    PASS\n");
}

// Test top-K sampling
void test_top_k_sampling()
{
    printf("  Testing top-K sampling...\n");

    // Create logits for testing
    float logits[5] = {0.1f, 0.2f, 5.0f, 0.3f, 4.0f};
    float logits_copy[5];

    TinyAIGenerationParams params;
    params.maxTokens      = 10;
    params.samplingMethod = TINYAI_SAMPLING_TOP_K;
    params.seed           = 42; // Fixed for reproducibility
    params.temperature    = 1.0f;

    // Test with K=1 (should be deterministic, picking highest)
    memcpy(logits_copy, logits, sizeof(logits));
    params.topK = 1;
    int token1  = tinyaiSampleToken(logits_copy, 5, &params);
    ASSERT(token1 == 2, "With K=1, should pick highest logit (index 2)");

    // Test with K=2 (should be either index 2 or 4)
    memcpy(logits_copy, logits, sizeof(logits));
    params.topK = 2;
    int token2  = tinyaiSampleToken(logits_copy, 5, &params);
    ASSERT(token2 == 2 || token2 == 4, "With K=2, should pick from top 2 logits (indices 2 or 4)");

    // Test with K=5 (all tokens possible)
    memcpy(logits_copy, logits, sizeof(logits));
    params.topK = 5;
    int token3  = tinyaiSampleToken(logits_copy, 5, &params);
    ASSERT(token3 >= 0 && token3 < 5, "With K=5, any token can be sampled");

    printf("    PASS\n");
}

// Test top-P (nucleus) sampling
void test_top_p_sampling()
{
    printf("  Testing top-P (nucleus) sampling...\n");

    // Create logits for testing
    float logits[5] = {0.1f, 0.1f, 5.0f, 0.1f, 0.1f};
    float logits_copy[5];

    TinyAIGenerationParams params;
    params.maxTokens      = 10;
    params.samplingMethod = TINYAI_SAMPLING_TOP_P;
    params.seed           = 42; // Fixed for reproducibility
    params.temperature    = 1.0f;

    // Test with P=0.5 (should pick index 2 which has most probability mass)
    memcpy(logits_copy, logits, sizeof(logits));
    params.topP = 0.5f;
    int token1  = tinyaiSampleToken(logits_copy, 5, &params);
    ASSERT(token1 == 2, "With P=0.5, should mostly pick highest logit (index 2)");

    // Test with P=0.9 (more tokens possible)
    memcpy(logits_copy, logits, sizeof(logits));
    params.topP = 0.9f;
    int token2  = tinyaiSampleToken(logits_copy, 5, &params);
    ASSERT(token2 >= 0 && token2 < 5, "With P=0.9, more tokens can be sampled");

    // Test with P=1.0 (all tokens possible)
    memcpy(logits_copy, logits, sizeof(logits));
    params.topP = 1.0f;
    int token3  = tinyaiSampleToken(logits_copy, 5, &params);
    ASSERT(token3 >= 0 && token3 < 5, "With P=1.0, any token can be sampled");

    printf("    PASS\n");
}

// Test greedy sampling
void test_greedy_sampling()
{
    printf("  Testing greedy sampling...\n");

    // Create logits for testing
    float logits[5] = {0.1f, 0.2f, 5.0f, 0.3f, 4.0f};
    float logits_copy[5];

    TinyAIGenerationParams params;
    params.maxTokens      = 10;
    params.samplingMethod = TINYAI_SAMPLING_GREEDY;
    params.seed           = 42;   // Not used for greedy
    params.temperature    = 1.0f; // Not used for greedy

    // Greedy should always pick the highest logit
    memcpy(logits_copy, logits, sizeof(logits));
    int token = tinyaiSampleToken(logits_copy, 5, &params);
    ASSERT(token == 2, "Greedy sampling should pick highest logit (index 2)");

    printf("    PASS\n");
}

// Test simplified text generation
void test_text_generation()
{
    printf("  Testing text generation (simplified)...\n");

    TinyAITokenizer *tokenizer   = create_test_tokenizer();
    uint32_t         hiddenSize  = 4; // Small for testing
    uint32_t         contextSize = 8; // Small for testing
    TinyAIModel     *model =
        tinyaiCreateModel(TINYAI_MODEL_TYPE_RNN, hiddenSize, contextSize, tokenizer);

    // Add layers (minimal for testing)
    tinyaiAddLayer(model, TINYAI_LAYER_EMBEDDING, tokenizer->tokenCount, hiddenSize,
                   TINYAI_ACTIVATION_NONE);
    tinyaiAddLayer(model, TINYAI_LAYER_OUTPUT, hiddenSize, tokenizer->tokenCount,
                   TINYAI_ACTIVATION_NONE);

    // Create some dummy weights for embedding layer
    TinyAIMatrixFP32 *embedMatrix = create_mock_matrix(tokenizer->tokenCount, hiddenSize);
    TinyAIMatrix4bit *quantized   = tinyaiQuantizeFP32To4bit(embedMatrix);
    model->layers[0].weights      = *quantized; // Copy the struct

    // Create some dummy weights for output layer
    TinyAIMatrixFP32 *outputMatrix    = create_mock_matrix(hiddenSize, tokenizer->tokenCount);
    TinyAIMatrix4bit *quantizedOutput = tinyaiQuantizeFP32To4bit(outputMatrix);
    model->layers[1].weights          = *quantizedOutput; // Copy the struct

    // Set up generation parameters
    TinyAIGenerationParams params;
    params.maxTokens      = 5;    // Generate up to 5 tokens
    params.promptTokens   = NULL; // No prompt - will start with BOS token
    params.promptLength   = 0;
    params.samplingMethod = TINYAI_SAMPLING_GREEDY; // Deterministic for testing
    params.temperature    = 1.0f;
    params.topK           = 0;
    params.topP           = 0.0f;
    params.seed           = 42; // Fixed for reproducibility

    // Generate some text
    int outputTokens[10] = {0};
    int outputLength     = tinyaiGenerateText(model, &params, outputTokens, 10);

    // Check generation
    ASSERT(outputLength > 0, "Text generation should produce tokens");
    ASSERT(outputLength <= 10, "Text generation should respect max output tokens");
    ASSERT(outputLength <= params.maxTokens + 1,
           "Text generation should respect maxTokens parameter");

    // Check first token is BOS when no prompt
    ASSERT(outputTokens[0] == TINYAI_TOKEN_BOS, "First token should be BOS with no prompt");

    // Now test with a prompt
    int promptTokens[3] = {1, 2, 3}; // Some arbitrary token IDs
    params.promptTokens = promptTokens;
    params.promptLength = 3;

    memset(outputTokens, 0, sizeof(outputTokens));
    outputLength = tinyaiGenerateText(model, &params, outputTokens, 10);

    // Check generation with prompt
    ASSERT(outputLength >= 3,
           "Text generation with prompt should have at least prompt length tokens");
    ASSERT(outputLength <= 10, "Text generation should respect max output tokens");

    // Check prompt is preserved
    ASSERT(outputTokens[0] == promptTokens[0], "Prompt tokens should be preserved in output");
    ASSERT(outputTokens[1] == promptTokens[1], "Prompt tokens should be preserved in output");
    ASSERT(outputTokens[2] == promptTokens[2], "Prompt tokens should be preserved in output");

    // Clean up
    TINYAI_FREE(quantized);       // Free the struct but not the data (owned by model now)
    TINYAI_FREE(quantizedOutput); // Free the struct but not the data (owned by model now)
    free_mock_matrix(embedMatrix);
    free_mock_matrix(outputMatrix);
    tinyaiDestroyModel(model);
    tinyaiDestroyTokenizer(tokenizer);
    printf("    PASS\n");
}

// Stub for model loading test (requires actual model files)
void test_model_loading()
{
    printf("  Testing model loading (STUB)...\n");
    // This would require real model files to test properly
    printf("    SKIP - Requires model files\n");
}

// Function to be called by test_main.c
void run_generate_tests()
{
    printf("--- Running Text Generation Tests ---\n");

    test_model_create_destroy();
    test_add_layers();
    test_model_forward_simple();
    test_softmax_temperature();
    test_top_k_sampling();
    test_top_p_sampling();
    test_greedy_sampling();
    test_text_generation();
    test_model_loading();

    printf("--- Text Generation Tests Finished ---\n");
}
