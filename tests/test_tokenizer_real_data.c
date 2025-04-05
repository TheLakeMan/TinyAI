/**
 * TinyAI Tokenizer Real Data Tests
 *
 * This file implements tests for the tokenizer with actual data and realistic usage scenarios
 */

#include "../models/text/tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Basic assertion helper (consistent with other test files)
#define ASSERT(condition, message)                                                                 \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            fprintf(stderr, "Assertion Failed: %s (%s:%d)\n", message, __FILE__, __LINE__);        \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

// Maximum file size for test data
#define MAX_FILE_SIZE (1024 * 1024) // 1MB

// Load a text file for testing
char *load_text_file(const char *filepath)
{
    FILE *file = fopen(filepath, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", filepath);
        return NULL;
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);

    if (size > MAX_FILE_SIZE) {
        fprintf(stderr, "File too large: %ld bytes (max: %d)\n", size, MAX_FILE_SIZE);
        fclose(file);
        return NULL;
    }

    // Allocate buffer
    char *buffer = (char *)malloc(size + 1);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate memory for file content\n");
        fclose(file);
        return NULL;
    }

    // Read file content
    size_t bytesRead = fread(buffer, 1, size, file);
    fclose(file);

    if (bytesRead != (size_t)size) {
        fprintf(stderr, "Failed to read entire file\n");
        free(buffer);
        return NULL;
    }

    // Null terminate
    buffer[size] = '\0';

    return buffer;
}

// Test tokenization of real text data
void test_tokenize_real_data()
{
    printf("  Testing tokenization with real data...\n");

    // Try to load a test file
    char *text = load_text_file("data/sample_text.txt");
    if (!text) {
        printf("    SKIP - Could not load sample_text.txt\n");
        return;
    }

    // Create tokenizer
    TinyAITokenizer *tokenizer = tinyaiCreateTokenizer();
    ASSERT(tokenizer != NULL, "Failed to create tokenizer");

    // Create a minimal vocabulary from the text
    int vocabSize = 1000;
    int result    = tinyaiCreateMinimalVocabulary(tokenizer, text, vocabSize);

    ASSERT(result == 0, "Failed to create vocabulary from text");
    ASSERT(tokenizer->tokenCount > 0, "Vocabulary should not be empty");
    ASSERT(tokenizer->tokenCount <= vocabSize, "Vocabulary should not exceed requested size");

    printf("    Created vocabulary with %d tokens\n", tokenizer->tokenCount);

    // Encode the text
    int *tokens = (int *)malloc(MAX_FILE_SIZE * sizeof(int));
    ASSERT(tokens != NULL, "Failed to allocate memory for tokens");

    int tokenCount = tinyaiEncodeText(tokenizer, text, tokens, MAX_FILE_SIZE);
    ASSERT(tokenCount > 0, "Failed to encode text");

    printf("    Encoded text into %d tokens\n", tokenCount);

    // Calculate compression ratio
    double compressionRatio = (double)strlen(text) / (tokenCount * sizeof(int));
    printf("    Compression ratio: %.2fx\n", compressionRatio);

    // Decode the tokens back to text
    char *decoded = (char *)malloc(MAX_FILE_SIZE);
    ASSERT(decoded != NULL, "Failed to allocate memory for decoded text");

    int decodedLength = tinyaiDecodeTokens(tokenizer, tokens, tokenCount, decoded, MAX_FILE_SIZE);
    ASSERT(decodedLength > 0, "Failed to decode tokens");

    // Compare decoded text with original
    // Note: We don't expect exact match due to tokenization limitations,
    // but check if lengths are reasonably close
    double lengthRatio = (double)decodedLength / strlen(text);
    printf("    Original length: %zu, Decoded length: %d, Ratio: %.2f\n", strlen(text),
           decodedLength, lengthRatio);

    ASSERT(lengthRatio > 0.5 && lengthRatio < 1.5,
           "Decoded text length should be reasonably close to original");

    // Cleanup
    free(text);
    free(tokens);
    free(decoded);
    tinyaiDestroyTokenizer(tokenizer);

    printf("    PASS\n");
}

// Test loading a pre-built vocabulary
void test_load_vocabulary()
{
    printf("  Testing loading pre-built vocabulary...\n");

    // Try to load a vocabulary file
    TinyAITokenizer *tokenizer = tinyaiCreateTokenizer();
    ASSERT(tokenizer != NULL, "Failed to create tokenizer");

    int result = tinyaiLoadVocabulary(tokenizer, "data/tiny_vocab.tok");

    if (result != 0) {
        printf("    SKIP - Could not load tiny_vocab.tok\n");
        tinyaiDestroyTokenizer(tokenizer);
        return;
    }

    ASSERT(tokenizer->tokenCount > 0, "Loaded vocabulary should not be empty");
    printf("    Loaded vocabulary with %d tokens\n", tokenizer->tokenCount);

    // Test encoding/decoding with loaded vocabulary
    const char *testText = "The quick brown fox jumps over the lazy dog.";

    int tokens[100] = {0};
    int tokenCount  = tinyaiEncodeText(tokenizer, testText, tokens, 100);

    ASSERT(tokenCount > 0, "Failed to encode text with loaded vocabulary");
    printf("    Encoded test sentence into %d tokens\n", tokenCount);

    // Print first few tokens for inspection
    printf("    First tokens: ");
    for (int i = 0; i < (tokenCount > 5 ? 5 : tokenCount); i++) {
        printf("%d ", tokens[i]);
    }
    printf("...\n");

    char decoded[1000] = {0};
    int decodedLength = tinyaiDecodeTokens(tokenizer, tokens, tokenCount, decoded, sizeof(decoded));

    ASSERT(decodedLength > 0, "Failed to decode tokens with loaded vocabulary");
    printf("    Decoded: '%s'\n", decoded);

    tinyaiDestroyTokenizer(tokenizer);
    printf("    PASS\n");
}

// Test token frequency analysis
void test_token_frequency()
{
    printf("  Testing token frequency analysis...\n");

    // Try to load a test file
    char *text = load_text_file("data/sample_text.txt");
    if (!text) {
        printf("    SKIP - Could not load sample_text.txt\n");
        return;
    }

    // Create tokenizer with a vocabulary
    TinyAITokenizer *tokenizer = tinyaiCreateTokenizer();
    ASSERT(tokenizer != NULL, "Failed to create tokenizer");

    // Create vocabulary
    int vocabSize = 500;
    int result    = tinyaiCreateMinimalVocabulary(tokenizer, text, vocabSize);
    ASSERT(result == 0, "Failed to create vocabulary");

    // Encode the text
    int *tokens = (int *)malloc(MAX_FILE_SIZE * sizeof(int));
    ASSERT(tokens != NULL, "Failed to allocate memory for tokens");

    int tokenCount = tinyaiEncodeText(tokenizer, text, tokens, MAX_FILE_SIZE);
    ASSERT(tokenCount > 0, "Failed to encode text");

    // Count token frequencies
    int *frequencies = (int *)calloc(tokenizer->tokenCount, sizeof(int));
    ASSERT(frequencies != NULL, "Failed to allocate memory for frequencies");

    for (int i = 0; i < tokenCount; i++) {
        if (tokens[i] >= 0 && tokens[i] < tokenizer->tokenCount) {
            frequencies[tokens[i]]++;
        }
    }

    // Find most frequent tokens
    int topN = 10;
    printf("    Top %d most frequent tokens:\n", topN);

    for (int n = 0; n < topN; n++) {
        // Find max frequency
        int maxFreq = 0;
        int maxIdx  = -1;

        for (int i = 0; i < tokenizer->tokenCount; i++) {
            if (frequencies[i] > maxFreq) {
                maxFreq = frequencies[i];
                maxIdx  = i;
            }
        }

        if (maxIdx >= 0) {
            const char *tokenStr = tinyaiGetTokenString(tokenizer, maxIdx);
            printf("    %d. Token %d (\"%s\") - %d occurrences\n", n + 1, maxIdx,
                   tokenStr ? tokenStr : "<unknown>", maxFreq);

            // Zero out for next iteration
            frequencies[maxIdx] = 0;
        }
    }

    // Cleanup
    free(text);
    free(tokens);
    free(frequencies);
    tinyaiDestroyTokenizer(tokenizer);

    printf("    PASS\n");
}

// Test tokenizer on special sequences
void test_special_sequences()
{
    printf("  Testing tokenizer on special sequences...\n");

    TinyAITokenizer *tokenizer = tinyaiCreateTokenizer();
    ASSERT(tokenizer != NULL, "Failed to create tokenizer");

    // Add some basic tokens
    tinyaiAddToken(tokenizer, "a", 100);
    tinyaiAddToken(tokenizer, "b", 90);
    tinyaiAddToken(tokenizer, "c", 80);
    tinyaiAddToken(tokenizer, " ", 70);
    tinyaiAddToken(tokenizer, "\n", 60);
    tinyaiAddToken(tokenizer, ".", 50);
    tinyaiAddToken(tokenizer, ",", 40);
    tinyaiAddToken(tokenizer, "!", 30);
    tinyaiAddToken(tokenizer, "?", 20);

    // Test cases
    const char *testCases[] = {
        "",            // Empty string
        " ",           // Just space
        "\n\n",        // Just newlines
        "abc",         // Simple sequence
        "a b c",       // With spaces
        "a\nb\nc",     // With newlines
        "a,b,c.",      // With punctuation
        "a!!!",        // Repeated punctuation
        "  a  b  c  ", // Extra spaces
    };

    int numTestCases = sizeof(testCases) / sizeof(testCases[0]);

    for (int i = 0; i < numTestCases; i++) {
        const char *testCase = testCases[i];
        printf("    Test case %d: \"%s\"\n", i + 1, testCase);

        // Encode
        int tokens[100] = {0};
        int tokenCount  = tinyaiEncodeText(tokenizer, testCase, tokens, 100);

        ASSERT(tokenCount >= 0, "Encoding should not fail");
        printf("      Encoded into %d tokens: ", tokenCount);

        for (int j = 0; j < tokenCount; j++) {
            printf("%d ", tokens[j]);
        }
        printf("\n");

        if (tokenCount > 0) {
            // Decode
            char decoded[100] = {0};
            int  decodedLength =
                tinyaiDecodeTokens(tokenizer, tokens, tokenCount, decoded, sizeof(decoded));

            ASSERT(decodedLength >= 0, "Decoding should not fail");
            printf("      Decoded: \"%s\"\n", decoded);

            // For simple cases where all characters are in vocabulary, we should have lossless
            // encoding
            if (strspn(testCase, "abc \n.,!?") == strlen(testCase)) {
                ASSERT(strcmp(decoded, testCase) == 0,
                       "Decoded text should match original for known tokens");
            }
        }
    }

    tinyaiDestroyTokenizer(tokenizer);
    printf("    PASS\n");
}

// Test performance benchmarking
void test_tokenizer_performance()
{
    printf("  Testing tokenizer performance...\n");

    // Try to load a test file
    char *text = load_text_file("data/sample_text.txt");
    if (!text) {
        printf("    SKIP - Could not load sample_text.txt\n");
        return;
    }

    size_t textLength = strlen(text);
    printf("    Loaded text with %zu characters\n", textLength);

    // Create tokenizer
    TinyAITokenizer *tokenizer = tinyaiCreateTokenizer();
    ASSERT(tokenizer != NULL, "Failed to create tokenizer");

    // Create vocabulary
    int vocabSize = 1000;
    int result    = tinyaiCreateMinimalVocabulary(tokenizer, text, vocabSize);
    ASSERT(result == 0, "Failed to create vocabulary");

    int *tokens = (int *)malloc(MAX_FILE_SIZE * sizeof(int));
    ASSERT(tokens != NULL, "Failed to allocate memory for tokens");

    // Benchmark encoding
    clock_t startTime  = clock();
    int     tokenCount = tinyaiEncodeText(tokenizer, text, tokens, MAX_FILE_SIZE);
    clock_t endTime    = clock();

    ASSERT(tokenCount > 0, "Failed to encode text");

    double encodingTimeMs = (double)(endTime - startTime) * 1000.0 / CLOCKS_PER_SEC;
    double charsPerSecond = (textLength * 1000.0) / encodingTimeMs;

    printf("    Encoding: %.2f ms (%.2f chars/sec)\n", encodingTimeMs, charsPerSecond);

    // Benchmark decoding
    char *decoded = (char *)malloc(MAX_FILE_SIZE);
    ASSERT(decoded != NULL, "Failed to allocate memory for decoded text");

    startTime         = clock();
    int decodedLength = tinyaiDecodeTokens(tokenizer, tokens, tokenCount, decoded, MAX_FILE_SIZE);
    endTime           = clock();

    ASSERT(decodedLength > 0, "Failed to decode tokens");

    double decodingTimeMs  = (double)(endTime - startTime) * 1000.0 / CLOCKS_PER_SEC;
    double tokensPerSecond = (tokenCount * 1000.0) / decodingTimeMs;

    printf("    Decoding: %.2f ms (%.2f tokens/sec)\n", decodingTimeMs, tokensPerSecond);

    // Cleanup
    free(text);
    free(tokens);
    free(decoded);
    tinyaiDestroyTokenizer(tokenizer);

    printf("    PASS\n");
}

// Main test function to be called from main test suite
void run_tokenizer_real_data_tests()
{
    printf("--- Running Tokenizer Real Data Tests ---\n");

    test_tokenize_real_data();
    test_load_vocabulary();
    test_token_frequency();
    test_special_sequences();
    test_tokenizer_performance();

    printf("--- Tokenizer Real Data Tests Finished ---\n");
}
