/**
 * @file chat_model.h
 * @brief Header for memory-constrained chatbot functionality in TinyAI
 *
 * This header defines the chatbot functionality for TinyAI, focusing on
 * efficient operation within strict memory constraints.
 */

#ifndef TINYAI_CHAT_MODEL_H
#define TINYAI_CHAT_MODEL_H

#include "../../models/text/generate.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Message role
 */
typedef enum {
    TINYAI_ROLE_SYSTEM,   /* System instructions or context */
    TINYAI_ROLE_USER,     /* User messages */
    TINYAI_ROLE_ASSISTANT /* Assistant (model) responses */
} TinyAIChatRole;

/**
 * Chat message
 */
typedef struct {
    TinyAIChatRole role;        /* Role (system, user, assistant) */
    char          *content;     /* Message content */
    int            token_count; /* Number of tokens in the message (cached) */
} TinyAIChatMessage;

/**
 * Chat session configuration
 */
typedef struct {
    /* Model paths */
    const char *modelPath;     /* Path to language model structure */
    const char *weightsPath;   /* Path to model weights */
    const char *tokenizerPath; /* Path to tokenizer vocabulary */

    /* Memory constraints */
    int memoryLimitMB;    /* Maximum memory usage in MB */
    int maxContextTokens; /* Maximum context window in tokens */

    /* Generation parameters */
    int   maxTokens;   /* Maximum tokens in response */
    float temperature; /* Sampling temperature */
    float topP;        /* Top-p sampling parameter */

    /* Optimization */
    bool useQuantization; /* Whether to use quantization */
    bool useSIMD;         /* Whether to use SIMD acceleration */
} TinyAIChatConfig;

/**
 * Chat session
 */
typedef struct TinyAIChatSession TinyAIChatSession;

/**
 * Callback for streaming generation
 * @param token The generated token text
 * @param is_partial Whether this is a partial UTF-8 character
 * @param user_data User-provided data pointer
 * @return True to continue generation, false to stop
 */
typedef bool (*TinyAIChatStreamCallback)(const char *token, bool is_partial, void *user_data);

/**
 * Create a new chat session
 *
 * @param config Configuration for the chat session
 * @return New chat session or NULL on error
 */
TinyAIChatSession *tinyaiChatSessionCreate(const TinyAIChatConfig *config);

/**
 * Free a chat session
 *
 * @param session Chat session to free
 */
void tinyaiChatSessionFree(TinyAIChatSession *session);

/**
 * Add a message to the chat history
 *
 * @param session Chat session
 * @param role Role (system, user, assistant)
 * @param content Message content
 * @return True on success, false on failure
 */
bool tinyaiChatAddMessage(TinyAIChatSession *session, TinyAIChatRole role, const char *content);

/**
 * Generate a response to the conversation
 *
 * @param session Chat session
 * @param stream_callback Callback for streaming tokens (NULL for non-streaming)
 * @param user_data User data to pass to the callback
 * @return Generated response (caller must free) or NULL on error
 */
char *tinyaiChatGenerateResponse(TinyAIChatSession       *session,
                                 TinyAIChatStreamCallback stream_callback, void *user_data);

/**
 * Save chat history to a file (JSON format)
 *
 * @param session Chat session
 * @param filePath Path to save the history
 * @return True on success, false on failure
 */
bool tinyaiChatSaveHistory(TinyAIChatSession *session, const char *filePath);

/**
 * Load chat history from a file (JSON format)
 *
 * @param session Chat session
 * @param filePath Path to load the history from
 * @return True on success, false on failure
 */
bool tinyaiChatLoadHistory(TinyAIChatSession *session, const char *filePath);

/**
 * Get the number of messages in the chat history
 *
 * @param session Chat session
 * @return Number of messages
 */
int tinyaiChatGetMessageCount(TinyAIChatSession *session);

/**
 * Get a message from the chat history
 *
 * @param session Chat session
 * @param index Message index
 * @param role Output parameter for message role
 * @param content Output parameter for message content (don't free, owned by session)
 * @return True on success, false if index is out of bounds
 */
bool tinyaiChatGetMessage(TinyAIChatSession *session, int index, TinyAIChatRole *role,
                          const char **content);

/**
 * Clear the chat history
 *
 * @param session Chat session
 */
void tinyaiChatClearHistory(TinyAIChatSession *session);

/**
 * Get current memory usage statistics
 *
 * @param session Chat session
 * @param modelMemory Output parameter for model memory (in bytes)
 * @param historyMemory Output parameter for history memory (in bytes)
 * @param totalMemory Output parameter for total memory (in bytes)
 * @return True on success, false on failure
 */
bool tinyaiChatGetMemoryUsage(TinyAIChatSession *session, size_t *modelMemory,
                              size_t *historyMemory, size_t *totalMemory);

/**
 * Set generation parameters
 *
 * @param session Chat session
 * @param temperature Sampling temperature (0.0-1.5)
 * @param maxTokens Maximum tokens to generate
 * @param topP Top-p sampling parameter (0.0-1.0)
 * @return True on success, false on failure
 */
bool tinyaiChatSetGenerationParams(TinyAIChatSession *session, float temperature, int maxTokens,
                                   float topP);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_CHAT_MODEL_H */
