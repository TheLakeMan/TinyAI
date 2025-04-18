/**
 * @file chat_model.c
 * @brief Implementation of memory-constrained chatbot functionality in TinyAI
 */

#include "chat_model.h"
#include "../../core/io.h"
#include "../../core/memory.h"
#include "../../models/text/generate.h"
#include "../../models/text/tokenizer.h"
#include "../../utils/quantize.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Default values for memory constraints */
#define DEFAULT_MEMORY_LIMIT_MB 16
#define DEFAULT_MAX_CONTEXT_TOKENS 512
#define DEFAULT_MAX_TOKENS 100
#define DEFAULT_TEMPERATURE 0.7f
#define DEFAULT_TOP_P 0.9f

/* Maximum length of role prefix */
#define MAX_ROLE_PREFIX_LENGTH 32

/* Maximum number of characters to buffer for streaming */
#define MAX_STREAMING_BUFFER 16

/* JSON keys for saving/loading */
#define JSON_KEY_MESSAGES "messages"
#define JSON_KEY_ROLE "role"
#define JSON_KEY_CONTENT "content"

/**
 * Internal structure for chat session
 */
struct TinyAIChatSession {
    /* Model and tokenizer */
    TinyAIModel     *model;     /* Language model */
    TinyAITokenizer *tokenizer; /* Tokenizer */

    /* Message history */
    TinyAIChatMessage *messages;        /* Array of messages */
    int                messageCount;    /* Number of messages */
    int                messageCapacity; /* Capacity of messages array */

    /* Memory constraints */
    int memoryLimitMB;    /* Maximum memory usage in MB */
    int maxContextTokens; /* Maximum context window size in tokens */

    /* Generation parameters */
    int   maxTokens;   /* Maximum tokens in response */
    float temperature; /* Sampling temperature */
    float topP;        /* Top-p sampling parameter */
};

/* Streaming callback structure */
typedef struct {
    TinyAIChatStreamCallback userCallback;
    void                    *userData;
    char                    *fullResponse;
    size_t                   responseLen;
    size_t                   responseCapacity;
    TinyAITokenizer         *tokenizer;
} StreamingContext;

/* Role conversion utilities */
static const char *getRoleString(TinyAIChatRole role)
{
    switch (role) {
    case TINYAI_ROLE_SYSTEM:
        return "system";
    case TINYAI_ROLE_USER:
        return "user";
    case TINYAI_ROLE_ASSISTANT:
        return "assistant";
    default:
        return "unknown";
    }
}

static TinyAIChatRole getRoleFromString(const char *roleStr)
{
    if (strcmp(roleStr, "system") == 0) {
        return TINYAI_ROLE_SYSTEM;
    }
    else if (strcmp(roleStr, "user") == 0) {
        return TINYAI_ROLE_USER;
    }
    else if (strcmp(roleStr, "assistant") == 0) {
        return TINYAI_ROLE_ASSISTANT;
    }
    return TINYAI_ROLE_USER; /* Default to user if unknown */
}

/* Create prompt prefix for role */
static void createRolePrefix(TinyAIChatRole role, char *prefix, size_t maxLength)
{
    switch (role) {
    case TINYAI_ROLE_SYSTEM:
        snprintf(prefix, maxLength, "System: ");
        break;
    case TINYAI_ROLE_USER:
        snprintf(prefix, maxLength, "User: ");
        break;
    case TINYAI_ROLE_ASSISTANT:
        snprintf(prefix, maxLength, "Assistant: ");
        break;
    default:
        snprintf(prefix, maxLength, "Unknown: ");
        break;
    }
}

/* Calculate token count for a message */
static int calculateTokenCount(TinyAITokenizer *tokenizer, const char *text)
{
    if (!tokenizer || !text) {
        return 0;
    }

    int  tokenCount = 0;
    int *tokens     = tinyaiTokenizerEncodeText(tokenizer, text, &tokenCount);

    if (tokens) {
        free(tokens);
        return tokenCount;
    }

    return 0;
}

/* Calculate memory usage for chat history */
static size_t calculateHistoryMemoryUsage(TinyAIChatSession *session)
{
    if (!session) {
        return 0;
    }

    size_t total = 0;
    for (int i = 0; i < session->messageCount; i++) {
        /* Message struct size plus content string */
        total += sizeof(TinyAIChatMessage);
        if (session->messages[i].content) {
            total += strlen(session->messages[i].content) + 1;
        }
    }

    /* Add array overhead */
    total += sizeof(TinyAIChatMessage) * (session->messageCapacity - session->messageCount);

    return total;
}

/* Prune history to fit within token constraints */
static void pruneHistory(TinyAIChatSession *session)
{
    if (!session || !session->tokenizer || session->messageCount <= 0) {
        return;
    }

    /* Count total tokens in history */
    int totalTokens = 0;
    for (int i = 0; i < session->messageCount; i++) {
        totalTokens += session->messages[i].token_count;
    }

    /* If we're within limits, nothing to do */
    if (totalTokens <= session->maxContextTokens) {
        return;
    }

    /* We need to prune messages from the beginning (oldest first)
     * but always keep system messages and the most recent user message */

    int tokensToRemove = totalTokens - session->maxContextTokens;

    /* First pass: remove oldest messages that are not system or most recent user message */
    for (int i = 0; i < session->messageCount && tokensToRemove > 0;) {
        /* Skip system messages and the most recent user message */
        if (session->messages[i].role == TINYAI_ROLE_SYSTEM) {
            i++;
            continue;
        }

        /* Identify the index of the most recent user message */
        int lastUserIdx = -1;
        for (int j = session->messageCount - 1; j >= 0; j--) {
            if (session->messages[j].role == TINYAI_ROLE_USER) {
                lastUserIdx = j;
                break;
            }
        }

        /* Skip the most recent user message */
        if (i == lastUserIdx) {
            i++;
            continue;
        }

        /* Remove this message */
        tokensToRemove -= session->messages[i].token_count;

        /* Free the message content */
        free(session->messages[i].content);

        /* Shift remaining messages */
        for (int j = i; j < session->messageCount - 1; j++) {
            session->messages[j] = session->messages[j + 1];
        }

        /* Decrement message count */
        session->messageCount--;
    }

    /* Second pass: if we still need to remove tokens, trim assistant responses */
    for (int i = 0; i < session->messageCount && tokensToRemove > 0; i++) {
        if (session->messages[i].role == TINYAI_ROLE_ASSISTANT) {
            /* Get token count */
            int tokenCount = session->messages[i].token_count;

            /* If this message is small enough, removing it entirely is better */
            if (tokenCount <= tokensToRemove) {
                /* Remove this message */
                tokensToRemove -= tokenCount;

                /* Free the message content */
                free(session->messages[i].content);

                /* Shift remaining messages */
                for (int j = i; j < session->messageCount - 1; j++) {
                    session->messages[j] = session->messages[j + 1];
                }

                /* Decrement message count and index (since we shifted) */
                session->messageCount--;
                i--;
            }
            else {
                /* We need to truncate this message */

                /* Tokenize the message */
                int  count  = 0;
                int *tokens = tinyaiTokenizerEncodeText(session->tokenizer,
                                                        session->messages[i].content, &count);

                if (tokens) {
                    /* Calculate how many tokens to keep */
                    int tokensToDecode = count - tokensToRemove;
                    if (tokensToDecode > 0) {
                        /* Decode the truncated message */
                        char *newContent =
                            tinyaiTokenizerDecode(session->tokenizer, tokens, tokensToDecode);

                        if (newContent) {
                            /* Replace the message content */
                            free(session->messages[i].content);
                            session->messages[i].content     = newContent;
                            session->messages[i].token_count = tokensToDecode;

                            /* Update tokens to remove */
                            tokensToRemove = 0;
                        }
                    }

                    free(tokens);
                }

                /* If truncation didn't work, remove the message entirely */
                if (tokensToRemove > 0) {
                    /* Remove this message */
                    tokensToRemove -= session->messages[i].token_count;

                    /* Free the message content */
                    free(session->messages[i].content);

                    /* Shift remaining messages */
                    for (int j = i; j < session->messageCount - 1; j++) {
                        session->messages[j] = session->messages[j + 1];
                    }

                    /* Decrement message count and index (since we shifted) */
                    session->messageCount--;
                    i--;
                }
            }
        }
    }
}

/* Streaming token callback */
static int tokenCallbackFunc(int token, void *userData)
{
    StreamingContext *ctx = (StreamingContext *)userData;
    if (!ctx || !ctx->tokenizer) {
        return 0;
    }

    /* Decode the token */
    char *tokenText = tinyaiTokenizerDecodeToken(ctx->tokenizer, token);
    if (!tokenText) {
        return 0;
    }

    /* Call user callback */
    int shouldContinue = 1;
    if (ctx->userCallback) {
        shouldContinue = ctx->userCallback(tokenText, false, ctx->userData);
    }

    /* Append to full response */
    size_t tokenLen = strlen(tokenText);
    if (ctx->responseLen + tokenLen >= ctx->responseCapacity) {
        /* Resize buffer */
        size_t newCapacity = ctx->responseCapacity * 2;
        char  *newResponse = (char *)realloc(ctx->fullResponse, newCapacity);
        if (!newResponse) {
            free(tokenText);
            return 0;
        }
        ctx->fullResponse     = newResponse;
        ctx->responseCapacity = newCapacity;
    }

    /* Copy token to response */
    strcpy(ctx->fullResponse + ctx->responseLen, tokenText);
    ctx->responseLen += tokenLen;

    /* Free token text */
    free(tokenText);

    return shouldContinue;
}

/* Create a new chat session */
TinyAIChatSession *tinyaiChatSessionCreate(const TinyAIChatConfig *config)
{
    if (!config || !config->modelPath || !config->weightsPath || !config->tokenizerPath) {
        fprintf(stderr, "Invalid chat session configuration\n");
        return NULL;
    }

    /* Allocate session structure */
    TinyAIChatSession *session = (TinyAIChatSession *)malloc(sizeof(TinyAIChatSession));
    if (!session) {
        fprintf(stderr, "Failed to allocate chat session\n");
        return NULL;
    }

    /* Initialize with defaults */
    memset(session, 0, sizeof(TinyAIChatSession));
    session->memoryLimitMB =
        config->memoryLimitMB > 0 ? config->memoryLimitMB : DEFAULT_MEMORY_LIMIT_MB;
    session->maxContextTokens =
        config->maxContextTokens > 0 ? config->maxContextTokens : DEFAULT_MAX_CONTEXT_TOKENS;
    session->maxTokens   = config->maxTokens > 0 ? config->maxTokens : DEFAULT_MAX_TOKENS;
    session->temperature = config->temperature > 0 ? config->temperature : DEFAULT_TEMPERATURE;
    session->topP        = config->topP > 0 ? config->topP : DEFAULT_TOP_P;

    /* Initial capacity for messages */
    session->messageCapacity = 16;
    session->messages =
        (TinyAIChatMessage *)malloc(session->messageCapacity * sizeof(TinyAIChatMessage));
    if (!session->messages) {
        free(session);
        fprintf(stderr, "Failed to allocate chat history\n");
        return NULL;
    }

    /* Load tokenizer */
    session->tokenizer = tinyaiTokenizerCreate(config->tokenizerPath);
    if (!session->tokenizer) {
        free(session->messages);
        free(session);
        fprintf(stderr, "Failed to create tokenizer from %s\n", config->tokenizerPath);
        return NULL;
    }

    /* Load model */
    session->model = tinyaiLoadModel(config->modelPath, config->weightsPath, config->tokenizerPath);
    if (!session->model) {
        tinyaiTokenizerFree(session->tokenizer);
        free(session->messages);
        free(session);
        fprintf(stderr, "Failed to load model from %s and %s\n", config->modelPath,
                config->weightsPath);
        return NULL;
    }

    /* Apply quantization if requested */
    if (config->useQuantization) {
        if (tinyaiQuantizeModel(session->model) != 0) {
            fprintf(stderr, "Warning: Model quantization failed\n");
            /* Continue with unquantized model */
        }
    }

    return session;
}

/* Free chat session */
void tinyaiChatSessionFree(TinyAIChatSession *session)
{
    if (!session) {
        return;
    }

    /* Free the model */
    if (session->model) {
        tinyaiDestroyModel(session->model);
    }

    /* Free the tokenizer */
    if (session->tokenizer) {
        tinyaiTokenizerFree(session->tokenizer);
    }

    /* Free message contents */
    for (int i = 0; i < session->messageCount; i++) {
        free(session->messages[i].content);
    }

    /* Free messages array */
    free(session->messages);

    /* Free session structure */
    free(session);
}

/* Add a message to the chat history */
bool tinyaiChatAddMessage(TinyAIChatSession *session, TinyAIChatRole role, const char *content)
{
    if (!session || !content) {
        return false;
    }

    /* Check if we need to resize the messages array */
    if (session->messageCount >= session->messageCapacity) {
        int                newCapacity = session->messageCapacity * 2;
        TinyAIChatMessage *newMessages = (TinyAIChatMessage *)realloc(
            session->messages, newCapacity * sizeof(TinyAIChatMessage));
        if (!newMessages) {
            fprintf(stderr, "Failed to resize message history\n");
            return false;
        }

        session->messages        = newMessages;
        session->messageCapacity = newCapacity;
    }

    /* Add the new message */
    session->messages[session->messageCount].role    = role;
    session->messages[session->messageCount].content = _strdup(content);
    if (!session->messages[session->messageCount].content) {
        fprintf(stderr, "Failed to allocate message content\n");
        return false;
    }

    /* Calculate token count */
    session->messages[session->messageCount].token_count =
        calculateTokenCount(session->tokenizer, content);

    /* Increment message count */
    session->messageCount++;

    /* Prune history if needed */
    pruneHistory(session);

    return true;
}

/* Generate a response to the conversation */
char *tinyaiChatGenerateResponse(TinyAIChatSession       *session,
                                 TinyAIChatStreamCallback stream_callback, void *user_data)
{
    if (!session || !session->model || !session->tokenizer) {
        return NULL;
    }

    /* Build the chat context */
    char   rolePrefix[MAX_ROLE_PREFIX_LENGTH];
    size_t contextSize = 1; /* Start with 1 for null terminator */

    /* First, calculate required size */
    for (int i = 0; i < session->messageCount; i++) {
        createRolePrefix(session->messages[i].role, rolePrefix, MAX_ROLE_PREFIX_LENGTH);
        contextSize +=
            strlen(rolePrefix) + strlen(session->messages[i].content) + 2; /* +2 for newlines */
    }

    /* Add assistant prefix for the response */
    createRolePrefix(TINYAI_ROLE_ASSISTANT, rolePrefix, MAX_ROLE_PREFIX_LENGTH);
    contextSize += strlen(rolePrefix);

    /* Allocate context buffer */
    char *context = (char *)malloc(contextSize);
    if (!context) {
        fprintf(stderr, "Failed to allocate context buffer\n");
        return NULL;
    }

    /* Build context string */
    context[0] = '\0';
    for (int i = 0; i < session->messageCount; i++) {
        createRolePrefix(session->messages[i].role, rolePrefix, MAX_ROLE_PREFIX_LENGTH);
        strcat(context, rolePrefix);
        strcat(context, session->messages[i].content);
        strcat(context, "\n\n");
    }

    /* Add assistant prefix for the response */
    createRolePrefix(TINYAI_ROLE_ASSISTANT, rolePrefix, MAX_ROLE_PREFIX_LENGTH);
    strcat(context, rolePrefix);

    /* Tokenize context */
    int  contextTokens = 0;
    int *tokens        = tinyaiTokenizerEncodeText(session->tokenizer, context, &contextTokens);

    if (!tokens) {
        free(context);
        fprintf(stderr, "Failed to tokenize context\n");
        return NULL;
    }

    /* Set up generation parameters */
    TinyAIGenerationParams params;
    memset(&params, 0, sizeof(TinyAIGenerationParams));
    params.promptTokens   = tokens;
    params.promptLength   = contextTokens;
    params.maxTokens      = session->maxTokens;
    params.temperature    = session->temperature;
    params.topP           = session->topP;
    params.samplingMethod = TINYAI_SAMPLING_TOP_P;

    /* Generate response */
    char *response = NULL;

    if (stream_callback) {
        /* Streaming version */

        /* Create streaming context */
        StreamingContext streamCtx;
        memset(&streamCtx, 0, sizeof(StreamingContext));
        streamCtx.userCallback     = stream_callback;
        streamCtx.userData         = user_data;
        streamCtx.tokenizer        = session->tokenizer;
        streamCtx.responseCapacity = 1024;
        streamCtx.fullResponse     = (char *)malloc(streamCtx.responseCapacity);

        if (!streamCtx.fullResponse) {
            free(tokens);
            free(context);
            fprintf(stderr, "Failed to allocate response buffer\n");
            return NULL;
        }
        streamCtx.fullResponse[0] = '\0';

        /* Generate with streaming */
        int numTokens =
            tinyaiGenerateTextWithCallback(session->model, &params, tokenCallbackFunc, &streamCtx);

        if (numTokens > 0) {
            /* Return the full response */
            response = streamCtx.fullResponse;
        }
        else {
            /* Generation failed */
            free(streamCtx.fullResponse);
        }
    }
    else {
        /* Non-streaming version */

        /* Allocate buffer for output tokens */
        int *outputTokens = (int *)malloc(params.maxTokens * sizeof(int));
        if (!outputTokens) {
            free(tokens);
            free(context);
            fprintf(stderr, "Failed to allocate output tokens buffer\n");
            return NULL;
        }

        int numTokens = tinyaiGenerateText(session->model, &params, outputTokens, params.maxTokens);

        if (numTokens > 0) {
            /* Decode tokens to text */
            response = tinyaiTokenizerDecode(session->tokenizer, outputTokens, numTokens);
        }

        free(outputTokens);
    }

    /* Clean up */
    free(tokens);
    free(context);

    /* Add response to chat history if successful */
    if (response) {
        tinyaiChatAddMessage(session, TINYAI_ROLE_ASSISTANT, response);
    }

    return response;
}

/* Save chat history to a file (JSON format) */
bool tinyaiChatSaveHistory(TinyAIChatSession *session, const char *filePath)
{
    if (!session || !filePath) {
        return false;
    }

    /* Open file for writing */
    FILE *file = fopen(filePath, "w");
    if (!file) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filePath);
        return false;
    }

    /* Write JSON header */
    fprintf(file, "{\n  \"%s\": [\n", JSON_KEY_MESSAGES);

    /* Write messages */
    for (int i = 0; i < session->messageCount; i++) {
        fprintf(file, "    {\n");
        fprintf(file, "      \"%s\": \"%s\",\n", JSON_KEY_ROLE,
                getRoleString(session->messages[i].role));

        /* Escape JSON special characters in content */
        fprintf(file, "      \"%s\": \"", JSON_KEY_CONTENT);
        const char *content = session->messages[i].content;
        while (*content) {
            char c = *content++;
            switch (c) {
            case '\"':
                fprintf(file, "\\\"");
                break;
            case '\\':
                fprintf(file, "\\\\");
                break;
            case '\b':
                fprintf(file, "\\b");
                break;
            case '\f':
                fprintf(file, "\\f");
                break;
            case '\n':
                fprintf(file, "\\n");
                break;
            case '\r':
                fprintf(file, "\\r");
                break;
            case '\t':
                fprintf(file, "\\t");
                break;
            default:
                if ((unsigned char)c >= 32) {
                    fputc(c, file);
                }
                else {
                    fprintf(file, "\\u%04x", (unsigned char)c);
                }
            }
        }
        fprintf(file, "\"\n");

        fprintf(file, "    }%s\n", i < session->messageCount - 1 ? "," : "");
    }

    /* Write JSON footer */
    fprintf(file, "  ]\n}\n");

    /* Close file */
    fclose(file);

    return true;
}

/* Ultra-simple JSON parsing for loading chat history */
static char *extractJsonString(const char *json, const char *key)
{
    char keyPattern[256];
    snprintf(keyPattern, sizeof(keyPattern), "\"%s\":\\s*\"", key);

    const char *start = strstr(json, keyPattern);
    if (!start) {
        return NULL;
    }

    start = strchr(start + strlen(key) + 2, '"') + 1;
    if (!start) {
        return NULL;
    }

    /* Find end quote, accounting for escaped quotes */
    const char *end = start;
    while (1) {
        end = strchr(end, '"');
        if (!end) {
            return NULL;
        }

        /* Check if quote is escaped */
        if (end > start && *(end - 1) == '\\') {
            end++;
            continue;
        }

        break;
    }

    /* Extract the string */
    size_t len    = end - start;
    char  *result = (char *)malloc(len + 1);
    if (!result) {
        return NULL;
    }

    /* Copy and process escape sequences */
    size_t pos = 0;
    for (size_t i = 0; i < len; i++) {
        if (start[i] == '\\' && i + 1 < len) {
            i++;
            switch (start[i]) {
            case '"':
                result[pos++] = '"';
                break;
            case '\\':
                result[pos++] = '\\';
                break;
            case '/':
                result[pos++] = '/';
                break;
            case 'b':
                result[pos++] = '\b';
                break;
            case 'f':
                result[pos++] = '\f';
                break;
            case 'n':
                result[pos++] = '\n';
                break;
            case 'r':
                result[pos++] = '\r';
                break;
            case 't':
                result[pos++] = '\t';
                break;
            case 'u': /* Unicode escape - simplified handling */
                if (i + 4 < len) {
                    result[pos++] = '?'; /* Placeholder for Unicode character */
                    i += 4;
                }
                break;
            default:
                result[pos++] = start[i];
            }
        }
        else {
            result[pos++] = start[i];
        }
    }

    result[pos] = '\0';
    return result;
}

/* Load chat history from a file (JSON format) */
bool tinyaiChatLoadHistory(TinyAIChatSession *session, const char *filePath)
{
    if (!session || !filePath) {
        return false;
    }

    /* Clear existing history */
    tinyaiChatClearHistory(session);

    /* Read file contents */
    char *fileContents = tinyaiReadTextFile(filePath);
    if (!fileContents) {
        fprintf(stderr, "Failed to read file: %s\n", filePath);
        return false;
    }

    /* Find the beginning of the messages array */
    char *messagesStart = strstr(fileContents, JSON_KEY_MESSAGES);
    if (!messagesStart) {
        free(fileContents);
        fprintf(stderr, "Invalid JSON format: messages array not found\n");
        return false;
    }

    /* Iterate through messages */
    char *currentPos = messagesStart;

    while (1) {
        /* Find start of object */
        currentPos = strchr(currentPos, '{');
        if (!currentPos) {
            break;
        }

        /* Find end of object */
        char *endObject = strchr(currentPos, '}');
        if (!endObject) {
            break;
        }

        /* Extract object as a substring for parsing */
        size_t objectLen = endObject - currentPos + 1;
        char  *objectStr = (char *)malloc(objectLen + 1);
        if (!objectStr) {
            break;
        }

        strncpy(objectStr, currentPos, objectLen);
        objectStr[objectLen] = '\0';

        /* Extract role and content */
        char *roleStr = extractJsonString(objectStr, JSON_KEY_ROLE);
        char *content = extractJsonString(objectStr, JSON_KEY_CONTENT);

        /* Add message if valid */
        if (roleStr && content) {
            TinyAIChatRole role = getRoleFromString(roleStr);
            tinyaiChatAddMessage(session, role, content);
        }

        /* Free memory */
        free(roleStr);
        free(content);
        free(objectStr);

        /* Move to next object */
        currentPos = endObject + 1;
    }

    /* Clean up */
    free(fileContents);

    return session->messageCount > 0;
}

/* Get the number of messages in the chat history */
int tinyaiChatGetMessageCount(TinyAIChatSession *session)
{
    if (!session) {
        return 0;
    }

    return session->messageCount;
}

/* Get a message from the chat history */
bool tinyaiChatGetMessage(TinyAIChatSession *session, int index, TinyAIChatRole *role,
                          const char **content)
{
    if (!session || index < 0 || index >= session->messageCount || !role || !content) {
        return false;
    }

    *role    = session->messages[index].role;
    *content = session->messages[index].content;

    return true;
}

/* Clear the chat history */
void tinyaiChatClearHistory(TinyAIChatSession *session)
{
    if (!session) {
        return;
    }

    /* Free message contents */
    for (int i = 0; i < session->messageCount; i++) {
        free(session->messages[i].content);
    }

    /* Reset message count */
    session->messageCount = 0;
}

/* Get current memory usage statistics */
bool tinyaiChatGetMemoryUsage(TinyAIChatSession *session, size_t *modelMemory,
                              size_t *historyMemory, size_t *totalMemory)
{
    if (!session) {
        return false;
    }

    size_t mModel = 0;

    /* Calculate model memory */
    if (session->model) {
        /* Estimate based on model parameters */
        mModel = tinyaiGetModelSizeBytes(session->model);
    }

    /* Calculate history memory */
    size_t mHistory = calculateHistoryMemoryUsage(session);

    /* Calculate total memory */
    size_t mTotal = mModel + mHistory + sizeof(TinyAIChatSession);

    /* Add tokenizer memory (rough estimate) */
    if (session->tokenizer) {
        mTotal += 2 * 1024 * 1024; /* Rough estimate */
    }

    /* Set output parameters */
    if (modelMemory) {
        *modelMemory = mModel;
    }

    if (historyMemory) {
        *historyMemory = mHistory;
    }

    if (totalMemory) {
        *totalMemory = mTotal;
    }

    return true;
}

/* Set generation parameters */
bool tinyaiChatSetGenerationParams(TinyAIChatSession *session, float temperature, int maxTokens,
                                   float topP)
{
    if (!session) {
        return false;
    }

    /* Validate parameters */
    if (temperature < 0.0f || temperature > 1.5f) {
        fprintf(stderr, "Invalid temperature (must be between 0.0 and 1.5)\n");
        return false;
    }

    if (maxTokens <= 0) {
        fprintf(stderr, "Invalid maxTokens (must be positive)\n");
        return false;
    }

    if (topP < 0.0f || topP > 1.0f) {
        fprintf(stderr, "Invalid topP (must be between 0.0 and 1.0)\n");
        return false;
    }

    /* Set parameters */
    session->temperature = temperature;
    session->maxTokens   = maxTokens;
    session->topP        = topP;

    return true;
}
