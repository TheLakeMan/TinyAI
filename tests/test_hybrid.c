/**
 * @file test_hybrid.c
 * @brief Test suite for hybrid generation capabilities
 */

#include "../core/mcp/mcp_client.h"
#include "../models/text/hybrid_generate.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Mock model for local generation testing */
TinyAIModel mockModel = {
    .type         = TINYAI_MODEL_TYPE_TRANSFORMER,
    .layerCount   = 4,
    .layers       = NULL, /* We'll mock the generation function, so we don't need real layers */
    .tokenizer    = NULL,
    .hiddenSize   = 128,
    .contextSize  = 512,
    .activations  = {NULL, NULL},
    .activeBuffer = 0};

/* Mock generate function for testing */
int tinyaiGenerateText(TinyAIModel *model, const TinyAIGenerationParams *params, int *outputTokens,
                       int maxTokens)
{
    /* Generate some dummy tokens */
    int count = params->maxTokens < 5 ? params->maxTokens : 5;
    for (int i = 0; i < count; i++) {
        outputTokens[i] = i + 10; /* Placeholder tokens */
    }
    return count;
}

/* Test hybrid generation with local-only setup */
void testLocalOnlyGeneration()
{
    printf("Testing local-only generation...\n");

    /* Create a hybrid generation context with only local model */
    TinyAIHybridGenerate *hybrid = tinyaiCreateHybridGenerate(&mockModel, NULL);
    assert(hybrid != NULL);

    /* Create generation parameters */
    TinyAIGenerationParams params;
    int                    promptTokens[2] = {1, 2};
    params.promptTokens                    = promptTokens;
    params.promptLength                    = 2;
    params.maxTokens                       = 5;
    params.temperature                     = 0.7f;
    params.samplingMethod                  = 0; /* Greedy */
    params.topK                            = 0;
    params.topP                            = 0.0f;
    params.seed                            = 42;

    /* Generate text */
    int outputTokens[10] = {0};
    int tokensGenerated  = tinyaiHybridGenerateText(hybrid, &params, outputTokens, 10);

    /* Verify results */
    assert(tokensGenerated == 5);
    assert(!tinyaiHybridGenerateUsedRemote(hybrid));

    /* Get generation stats */
    double localTime, remoteTime, tokensPerSec;
    tinyaiHybridGenerateGetStats(hybrid, &localTime, &remoteTime, &tokensPerSec);

    printf("  Local time: %.2f ms\n", localTime);
    printf("  Remote time: %.2f ms\n", remoteTime);
    printf("  Tokens per second: %.2f\n", tokensPerSec);

    assert(localTime > 0.0);
    assert(remoteTime == 0.0);

    /* Clean up */
    tinyaiDestroyHybridGenerate(hybrid);
    printf("Local-only generation test passed!\n\n");
}

/* Test hybrid generation with mock MCP client */
void testHybridGeneration()
{
    printf("Testing hybrid generation with MCP...\n");

    /* Create a MCP client config */
    TinyAIMcpConfig config;
    tinyaiMcpGetDefaultConfig(&config);

    /* Create MCP client */
    TinyAIMcpClient *mcpClient = tinyaiMcpCreateClient(&config);
    assert(mcpClient != NULL);

    /* Connect to a mock server */
    bool connected = tinyaiMcpConnect(mcpClient, "mock://localhost:8080");
    assert(connected);

    /* Create hybrid generation context */
    TinyAIHybridGenerate *hybrid = tinyaiCreateHybridGenerate(&mockModel, mcpClient);
    assert(hybrid != NULL);

    /* Create generation parameters that should trigger remote execution */
    TinyAIGenerationParams params;
    int                    promptTokens[200];
    for (int i = 0; i < 200; i++) {
        promptTokens[i] = i;
    }
    params.promptTokens   = promptTokens;
    params.promptLength   = 200; /* Long prompt should trigger remote execution */
    params.maxTokens      = 50;
    params.temperature    = 0.8f;
    params.samplingMethod = 1; /* Top-K */
    params.topK           = 40;
    params.topP           = 0.9f;
    params.seed           = 123;

    /* Check if remote would be used */
    bool wouldUseRemote = tinyaiHybridGenerateWouldUseRemote(hybrid, &params);
    printf("  Would use remote: %s\n", wouldUseRemote ? "yes" : "no");

    /* Generate text */
    int outputTokens[50] = {0};
    int tokensGenerated  = tinyaiHybridGenerateText(hybrid, &params, outputTokens, 50);

    /* Verify results */
    assert(tokensGenerated > 0);

    /* Check if remote was used */
    bool usedRemote = tinyaiHybridGenerateUsedRemote(hybrid);
    printf("  Used remote: %s\n", usedRemote ? "yes" : "no");

    /* Get generation stats */
    double localTime, remoteTime, tokensPerSec;
    tinyaiHybridGenerateGetStats(hybrid, &localTime, &remoteTime, &tokensPerSec);

    printf("  Local time: %.2f ms\n", localTime);
    printf("  Remote time: %.2f ms\n", remoteTime);
    printf("  Tokens per second: %.2f\n", tokensPerSec);

    if (usedRemote) {
        assert(remoteTime > 0.0);
    }
    else {
        assert(localTime > 0.0);
    }

    /* Test forcing local execution */
    printf("Testing forced local execution...\n");

    bool forceSuccess = tinyaiHybridGenerateForceMode(hybrid, false);
    assert(forceSuccess);

    tokensGenerated = tinyaiHybridGenerateText(hybrid, &params, outputTokens, 50);
    assert(tokensGenerated > 0);
    assert(!tinyaiHybridGenerateUsedRemote(hybrid));

    /* Test forcing remote execution */
    printf("Testing forced remote execution...\n");

    forceSuccess = tinyaiHybridGenerateForceMode(hybrid, true);
    assert(forceSuccess);

    tokensGenerated = tinyaiHybridGenerateText(hybrid, &params, outputTokens, 50);
    assert(tokensGenerated > 0);
    /* Remote execution may fallback to local if connection fails, so don't assert this */

    /* Clean up */
    tinyaiDestroyHybridGenerate(hybrid);
    tinyaiMcpDisconnect(mcpClient);
    tinyaiMcpDestroyClient(mcpClient);

    printf("Hybrid generation test passed!\n\n");
}

/* Test hybrid generation with no model and no MCP client */
void testNoModelsGeneration()
{
    printf("Testing generation with no models...\n");

    /* Create hybrid generation context with no models */
    TinyAIHybridGenerate *hybrid = tinyaiCreateHybridGenerate(NULL, NULL);
    assert(hybrid != NULL);

    /* Try to generate text - should fail */
    TinyAIGenerationParams params;
    int                    promptTokens[2] = {1, 2};
    params.promptTokens                    = promptTokens;
    params.promptLength                    = 2;
    params.maxTokens                       = 5;
    params.temperature                     = 0.7f;

    int outputTokens[10] = {0};
    int tokensGenerated  = tinyaiHybridGenerateText(hybrid, &params, outputTokens, 10);

    /* Verify results */
    assert(tokensGenerated < 0); /* Should return negative value on error */

    /* Clean up */
    tinyaiDestroyHybridGenerate(hybrid);

    printf("No models generation test passed!\n\n");
}

/* Main test function */
int testHybridMain()
{
    printf("\n=== Testing Hybrid Generation ===\n\n");

    /* Run tests */
    testLocalOnlyGeneration();
    testHybridGeneration();
    testNoModelsGeneration();

    printf("=== All Hybrid Generation Tests Passed! ===\n\n");
    return 0;
}
