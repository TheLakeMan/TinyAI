/**
 * @file mcp_client.c
 * @brief Implementation of the Model Context Protocol (MCP) client for TinyAI
 */

#include "mcp_client.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/**
 * @brief MCP client implementation structure
 */
struct TinyAIMcpClient {
    TinyAIMcpConfig          config;                   /* Client configuration */
    TinyAIMcpConnectionState connectionState;          /* Current connection state */
    char                     serverUrl[256];           /* Current server URL */
    time_t                   lastConnectionAttempt;    /* Timestamp of last connection attempt */
    int                      connectionAttempts;       /* Number of connection attempts */
    char                     lastError[256];           /* Last error message */
    bool                     isConnectionActive;       /* Whether connection is active */
    char                     serverCapabilities[1024]; /* Server capabilities */
};

/* Default configuration values */
#define DEFAULT_TIMEOUT_MS 5000
#define DEFAULT_MAX_RETRIES 3

void tinyaiMcpGetDefaultConfig(TinyAIMcpConfig *config)
{
    if (!config)
        return;

    config->execPreference      = TINYAI_EXEC_PREFER_LOCAL;
    config->enableAutoDiscovery = true;
    config->enableTelemetry     = false;
    config->connectionTimeoutMs = DEFAULT_TIMEOUT_MS;
    config->maxRetryAttempts    = DEFAULT_MAX_RETRIES;
    config->forceOffline        = false;
}

TinyAIMcpClient *tinyaiMcpCreateClient(const TinyAIMcpConfig *config)
{
    TinyAIMcpClient *client = (TinyAIMcpClient *)malloc(sizeof(TinyAIMcpClient));
    if (!client)
        return NULL;

    memset(client, 0, sizeof(TinyAIMcpClient));

    if (config) {
        memcpy(&client->config, config, sizeof(TinyAIMcpConfig));
    }
    else {
        tinyaiMcpGetDefaultConfig(&client->config);
    }

    client->connectionState    = TINYAI_MCP_DISCONNECTED;
    client->isConnectionActive = false;
    client->connectionAttempts = 0;

    return client;
}

void tinyaiMcpDestroyClient(TinyAIMcpClient *client)
{
    if (!client)
        return;

    /* If connected, disconnect first */
    if (client->connectionState == TINYAI_MCP_CONNECTED) {
        tinyaiMcpDisconnect(client);
    }

    free(client);
}

bool tinyaiMcpConnect(TinyAIMcpClient *client, const char *serverUrl)
{
    if (!client || !serverUrl)
        return false;

    /* Check if offline mode is forced */
    if (client->config.forceOffline) {
        strncpy(client->lastError, "Connection refused: offline mode is forced",
                sizeof(client->lastError) - 1);
        client->connectionState = TINYAI_MCP_ERROR;
        return false;
    }

    /* Check if already connected to the same server */
    if (client->connectionState == TINYAI_MCP_CONNECTED &&
        strcmp(client->serverUrl, serverUrl) == 0) {
        return true;
    }

    /* Disconnect from current server if connected */
    if (client->connectionState == TINYAI_MCP_CONNECTED) {
        tinyaiMcpDisconnect(client);
    }

    /* Store server URL */
    strncpy(client->serverUrl, serverUrl, sizeof(client->serverUrl) - 1);

    /* Reset connection state */
    client->connectionState       = TINYAI_MCP_CONNECTING;
    client->lastConnectionAttempt = time(NULL);
    client->connectionAttempts    = 1;

    /*
     * In a real implementation, we would:
     * 1. Establish a socket connection to the server
     * 2. Send handshake data and authenticate
     * 3. Receive capabilities from the server
     * 4. Update connection state based on response
     *
     * For this initial implementation, we'll simulate a successful connection
     */

    /* Simulate successful connection */
    client->connectionState    = TINYAI_MCP_CONNECTED;
    client->isConnectionActive = true;

    /* Set some default capabilities for testing */
    strcpy(
        client->serverCapabilities,
        "{"
        "  \"tools\": ["
        "    {\"name\": \"generate_text\", \"description\": \"Generate text with remote model\"},"
        "    {\"name\": \"tokenize_text\", \"description\": \"Tokenize text with remote model\"},"
        "    {\"name\": \"convert_model\", \"description\": \"Convert model to TinyAI format\"}"
        "  ],"
        "  \"resources\": ["
        "    {\"name\": \"model_repository\", \"description\": \"Access models from repository\"},"
        "    {\"name\": \"knowledge_base\", \"description\": \"Access knowledge base data\"}"
        "  ]"
        "}");

    return true;
}

void tinyaiMcpDisconnect(TinyAIMcpClient *client)
{
    if (!client)
        return;

    /* If not connected, nothing to do */
    if (client->connectionState != TINYAI_MCP_CONNECTED) {
        return;
    }

    /*
     * In a real implementation, we would:
     * 1. Send a disconnect message to the server
     * 2. Close the socket connection
     * 3. Clean up any resources
     */

    /* Reset connection state */
    client->connectionState    = TINYAI_MCP_DISCONNECTED;
    client->isConnectionActive = false;
    memset(client->serverUrl, 0, sizeof(client->serverUrl));
    client->connectionAttempts = 0;
}

TinyAIMcpConnectionState tinyaiMcpGetConnectionState(TinyAIMcpClient *client)
{
    if (!client)
        return TINYAI_MCP_DISCONNECTED;
    return client->connectionState;
}

bool tinyaiMcpIsAvailable(TinyAIMcpClient *client)
{
    if (!client)
        return false;

    /* Check if offline mode is forced */
    if (client->config.forceOffline) {
        return false;
    }

    /* Check if connected */
    return (client->connectionState == TINYAI_MCP_CONNECTED && client->isConnectionActive);
}

bool tinyaiMcpGetServerInfo(TinyAIMcpClient *client, TinyAIMcpServerInfo *info)
{
    if (!client || !info)
        return false;

    /* If not connected, can't get server info */
    if (client->connectionState != TINYAI_MCP_CONNECTED) {
        return false;
    }

    /* Fill in server info */
    strcpy(info->serverName, "TinyAI MCP Server");
    strcpy(info->serverUrl, client->serverUrl);
    strcpy(info->serverVersion, "0.1.0");
    info->connectionState = client->connectionState;
    strcpy(info->serverCapabilities, client->serverCapabilities);

    return true;
}

bool tinyaiMcpHasCapability(TinyAIMcpClient *client, const char *capability)
{
    if (!client || !capability)
        return false;

    /* If not connected, no capabilities */
    if (client->connectionState != TINYAI_MCP_CONNECTED) {
        return false;
    }

    /* Check if capability is in capabilities string */
    return (strstr(client->serverCapabilities, capability) != NULL);
}

int tinyaiMcpCallTool(TinyAIMcpClient *client, const char *toolName, const char *arguments,
                      char *result, int resultSize)
{
    if (!client || !toolName || !result || resultSize <= 0)
        return -1;

    /* If not connected, can't call tool */
    if (client->connectionState != TINYAI_MCP_CONNECTED) {
        strncpy(result, "Error: Not connected to MCP server", resultSize - 1);
        return -1;
    }

    /* Check if tool exists */
    if (!tinyaiMcpHasCapability(client, toolName)) {
        snprintf(result, resultSize - 1, "Error: Tool '%s' not supported by server", toolName);
        return -1;
    }

    /*
     * In a real implementation, we would:
     * 1. Format a request to the server
     * 2. Send the request
     * 3. Receive the response
     * 4. Parse the response and fill the result buffer
     */

    /* For now, return a placeholder response */
    if (strcmp(toolName, "generate_text") == 0) {
        snprintf(result, resultSize - 1, "Generated text based on arguments: %s",
                 arguments ? arguments : "none");
    }
    else if (strcmp(toolName, "tokenize_text") == 0) {
        snprintf(result, resultSize - 1, "Tokenized text based on arguments: %s",
                 arguments ? arguments : "none");
    }
    else if (strcmp(toolName, "convert_model") == 0) {
        snprintf(result, resultSize - 1, "Model conversion initiated with arguments: %s",
                 arguments ? arguments : "none");
    }
    else {
        snprintf(result, resultSize - 1, "Executed tool '%s' with arguments: %s", toolName,
                 arguments ? arguments : "none");
    }

    return strlen(result);
}

int tinyaiMcpAccessResource(TinyAIMcpClient *client, const char *resourceUri, char *result,
                            int resultSize)
{
    if (!client || !resourceUri || !result || resultSize <= 0)
        return -1;

    /* If not connected, can't access resource */
    if (client->connectionState != TINYAI_MCP_CONNECTED) {
        strncpy(result, "Error: Not connected to MCP server", resultSize - 1);
        return -1;
    }

    /*
     * In a real implementation, we would:
     * 1. Parse the resource URI
     * 2. Send a request to the server
     * 3. Receive the response
     * 4. Parse the response and fill the result buffer
     */

    /* For now, return a placeholder response */
    if (strstr(resourceUri, "model_repository") != NULL) {
        snprintf(result, resultSize - 1, "Model repository data for URI: %s", resourceUri);
    }
    else if (strstr(resourceUri, "knowledge_base") != NULL) {
        snprintf(result, resultSize - 1, "Knowledge base data for URI: %s", resourceUri);
    }
    else {
        snprintf(result, resultSize - 1, "Resource data for URI: %s", resourceUri);
    }

    return strlen(result);
}

void tinyaiMcpSetExecutionPreference(TinyAIMcpClient             *client,
                                     TinyAIMcpExecutionPreference preference)
{
    if (!client)
        return;
    client->config.execPreference = preference;
}

TinyAIMcpExecutionPreference tinyaiMcpGetExecutionPreference(TinyAIMcpClient *client)
{
    if (!client)
        return TINYAI_EXEC_ALWAYS_LOCAL;
    return client->config.execPreference;
}

void tinyaiMcpSetForceOffline(TinyAIMcpClient *client, bool forceOffline)
{
    if (!client)
        return;

    client->config.forceOffline = forceOffline;

    /* If forcing offline and currently connected, disconnect */
    if (forceOffline && client->connectionState == TINYAI_MCP_CONNECTED) {
        tinyaiMcpDisconnect(client);
    }
}

bool tinyaiMcpGetForceOffline(TinyAIMcpClient *client)
{
    if (!client)
        return true; /* Default to offline if no client */
    return client->config.forceOffline;
}
