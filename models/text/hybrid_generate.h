/**
 * @file hybrid_generate.h
 * @brief Hybrid text generation interface for TinyAI
 *
 * This file defines the interface for text generation that can
 * transparently switch between local and remote execution based
 * on MCP client availability and configuration.
 */

#ifndef TINYAI_HYBRID_GENERATE_H
#define TINYAI_HYBRID_GENERATE_H

#include "../../core/mcp/mcp_client.h"
#include "generate.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Context for hybrid text generation
 */
typedef struct TinyAIHybridGenerate TinyAIHybridGenerate;

/**
 * @brief Create a hybrid generation context
 *
 * @param localModel Local model to use when MCP is unavailable or for local execution
 * @param mcpClient MCP client for remote execution (can be NULL for local-only)
 * @return TinyAIHybridGenerate* Context or NULL on failure
 */
TinyAIHybridGenerate *tinyaiCreateHybridGenerate(TinyAIModel     *localModel,
                                                 TinyAIMcpClient *mcpClient);

/**
 * @brief Destroy a hybrid generation context
 *
 * @param ctx Context to destroy
 */
void tinyaiDestroyHybridGenerate(TinyAIHybridGenerate *ctx);

/**
 * @brief Generate text using the hybrid context
 *
 * This function will automatically choose between local and remote execution
 * based on the MCP client configuration and availability.
 *
 * @param ctx Hybrid generation context
 * @param params Generation parameters
 * @param outputTokens Buffer to store output tokens
 * @param maxTokens Maximum number of tokens to generate
 * @return int Number of tokens generated or negative value on error
 */
int tinyaiHybridGenerateText(TinyAIHybridGenerate *ctx, const TinyAIGenerationParams *params,
                             int *outputTokens, int maxTokens);

/**
 * @brief Check if hybrid generation used remote execution for the last generation
 *
 * @param ctx Hybrid generation context
 * @return true if remote execution was used
 * @return false if local execution was used
 */
bool tinyaiHybridGenerateUsedRemote(TinyAIHybridGenerate *ctx);

/**
 * @brief Get execution statistics for the last generation
 *
 * @param ctx Hybrid generation context
 * @param localTimeMs Output parameter for local execution time in milliseconds
 * @param remoteTimeMs Output parameter for remote execution time in milliseconds
 * @param tokensPerSecond Output parameter for tokens per second
 */
void tinyaiHybridGenerateGetStats(TinyAIHybridGenerate *ctx, double *localTimeMs,
                                  double *remoteTimeMs, double *tokensPerSecond);

/**
 * @brief Force a specific execution mode for the next generation
 *
 * @param ctx Hybrid generation context
 * @param forceRemote Force remote execution if true, force local if false
 * @return true if the requested mode is available
 * @return false if the requested mode is not available (will use fallback)
 */
bool tinyaiHybridGenerateForceMode(TinyAIHybridGenerate *ctx, bool forceRemote);

/**
 * @brief Check if MCP remote generation is available
 *
 * @param ctx Hybrid generation context
 * @return true if remote generation is available
 * @return false if remote generation is not available
 */
bool tinyaiHybridGenerateHasRemote(TinyAIHybridGenerate *ctx);

/**
 * @brief Get the execution decision for a given prompt and parameters
 *
 * This function can be used to determine whether the generation would
 * use local or remote execution without actually performing the generation.
 *
 * @param ctx Hybrid generation context
 * @param params Generation parameters
 * @return true if remote execution would be used
 * @return false if local execution would be used
 */
bool tinyaiHybridGenerateWouldUseRemote(TinyAIHybridGenerate         *ctx,
                                        const TinyAIGenerationParams *params);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_HYBRID_GENERATE_H */
