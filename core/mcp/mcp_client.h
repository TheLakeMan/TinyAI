/**
 * @file mcp_client.h
 * @brief Model Context Protocol (MCP) client implementation for TinyAI
 *
 * This file defines the API for interacting with MCP servers, providing
 * hybrid local/remote execution capabilities for TinyAI.
 */

#ifndef TINYAI_MCP_CLIENT_H
#define TINYAI_MCP_CLIENT_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Execution preference mode for hybrid operations
 */
typedef enum {
    TINYAI_EXEC_ALWAYS_LOCAL, /**< Always use local execution only */
    TINYAI_EXEC_PREFER_LOCAL, /**< Prefer local, use MCP only when necessary */
    TINYAI_EXEC_PREFER_MCP,   /**< Prefer MCP when available, fallback to local */
    TINYAI_EXEC_CUSTOM_POLICY /**< Use custom policy defined per operation */
} TinyAIMcpExecutionPreference;

/**
 * @brief MCP Server connection state
 */
typedef enum {
    TINYAI_MCP_DISCONNECTED, /**< Not connected to MCP server */
    TINYAI_MCP_CONNECTING,   /**< Connection in progress */
    TINYAI_MCP_CONNECTED,    /**< Successfully connected to MCP server */
    TINYAI_MCP_ERROR         /**< Error connecting to MCP server */
} TinyAIMcpConnectionState;

/**
 * @brief MCP client configuration
 */
typedef struct {
    TinyAIMcpExecutionPreference execPreference;      /**< Execution preference */
    bool                         enableAutoDiscovery; /**< Automatically discover MCP servers */
    bool                         enableTelemetry;     /**< Allow sending telemetry data */
    int                          connectionTimeoutMs; /**< Connection timeout in milliseconds */
    int                          maxRetryAttempts;    /**< Maximum connection retry attempts */
    bool                         forceOffline;        /**< Force offline mode */
} TinyAIMcpConfig;

/**
 * @brief MCP server information
 */
typedef struct {
    char                     serverName[64];           /**< Server name */
    char                     serverUrl[256];           /**< Server URL */
    char                     serverVersion[32];        /**< Server version */
    TinyAIMcpConnectionState connectionState;          /**< Connection state */
    char                     serverCapabilities[1024]; /**< JSON string of capabilities */
} TinyAIMcpServerInfo;

/**
 * @brief MCP client context
 */
typedef struct TinyAIMcpClient TinyAIMcpClient;

/**
 * @brief Create an MCP client instance
 *
 * @param config Client configuration
 * @return TinyAIMcpClient* Client instance or NULL on failure
 */
TinyAIMcpClient *tinyaiMcpCreateClient(const TinyAIMcpConfig *config);

/**
 * @brief Destroy an MCP client instance
 *
 * @param client Client instance
 */
void tinyaiMcpDestroyClient(TinyAIMcpClient *client);

/**
 * @brief Connect to an MCP server
 *
 * @param client Client instance
 * @param serverUrl Server URL
 * @return true if connection was successful or in progress
 * @return false if connection failed
 */
bool tinyaiMcpConnect(TinyAIMcpClient *client, const char *serverUrl);

/**
 * @brief Disconnect from an MCP server
 *
 * @param client Client instance
 */
void tinyaiMcpDisconnect(TinyAIMcpClient *client);

/**
 * @brief Get connection state
 *
 * @param client Client instance
 * @return TinyAIMcpConnectionState Current connection state
 */
TinyAIMcpConnectionState tinyaiMcpGetConnectionState(TinyAIMcpClient *client);

/**
 * @brief Check if MCP capabilities are available
 *
 * @param client Client instance
 * @return true if MCP is available and connected
 * @return false if MCP is unavailable
 */
bool tinyaiMcpIsAvailable(TinyAIMcpClient *client);

/**
 * @brief Get server information
 *
 * @param client Client instance
 * @param info Output server information
 * @return true if information was successfully retrieved
 * @return false on failure
 */
bool tinyaiMcpGetServerInfo(TinyAIMcpClient *client, TinyAIMcpServerInfo *info);

/**
 * @brief Check if MCP server supports a specific capability
 *
 * @param client Client instance
 * @param capability Capability name
 * @return true if capability is supported
 * @return false if capability is not supported or client is not connected
 */
bool tinyaiMcpHasCapability(TinyAIMcpClient *client, const char *capability);

/**
 * @brief Call a remote MCP tool
 *
 * @param client Client instance
 * @param toolName Name of the tool to call
 * @param arguments JSON arguments for the tool
 * @param result Output buffer for result
 * @param resultSize Size of output buffer
 * @return int Number of bytes written to result buffer, or negative value on error
 */
int tinyaiMcpCallTool(TinyAIMcpClient *client, const char *toolName, const char *arguments,
                      char *result, int resultSize);

/**
 * @brief Access an MCP resource
 *
 * @param client Client instance
 * @param resourceUri URI of the resource to access
 * @param result Output buffer for resource content
 * @param resultSize Size of output buffer
 * @return int Number of bytes written to result buffer, or negative value on error
 */
int tinyaiMcpAccessResource(TinyAIMcpClient *client, const char *resourceUri, char *result,
                            int resultSize);

/**
 * @brief Set execution preference
 *
 * @param client Client instance
 * @param preference Execution preference
 */
void tinyaiMcpSetExecutionPreference(TinyAIMcpClient             *client,
                                     TinyAIMcpExecutionPreference preference);

/**
 * @brief Get execution preference
 *
 * @param client Client instance
 * @return TinyAIMcpExecutionPreference Current execution preference
 */
TinyAIMcpExecutionPreference tinyaiMcpGetExecutionPreference(TinyAIMcpClient *client);

/**
 * @brief Force offline mode (disable MCP capabilities)
 *
 * @param client Client instance
 * @param forceOffline Whether to force offline mode
 */
void tinyaiMcpSetForceOffline(TinyAIMcpClient *client, bool forceOffline);

/**
 * @brief Check if forced offline mode is enabled
 *
 * @param client Client instance
 * @return true if forced offline mode is enabled
 * @return false if forced offline mode is disabled
 */
bool tinyaiMcpGetForceOffline(TinyAIMcpClient *client);

/**
 * @brief Get default MCP client configuration
 *
 * @param config Output configuration structure
 */
void tinyaiMcpGetDefaultConfig(TinyAIMcpConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_MCP_CLIENT_H */
