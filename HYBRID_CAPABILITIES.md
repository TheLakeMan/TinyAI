# TinyAI Hybrid Capabilities

TinyAI now supports hybrid local/remote execution through the Model Context Protocol (MCP). This document provides an overview of the hybrid capabilities, architecture, and usage.

## Overview

The hybrid capabilities enable TinyAI to transparently switch between local execution and remote execution via MCP servers. This provides several benefits:

1. **Scalability**: Handle larger models than would be possible on the local device
2. **Performance**: Offload computation to more powerful remote servers when needed
3. **Flexibility**: Dynamically decide execution environment based on various factors
4. **Fallback**: Continue operation even if remote servers are unavailable

## Architecture

The hybrid functionality is implemented through several components:

### MCP Client (core/mcp/mcp_client.h)

The MCP client provides the interface to connect to and interact with MCP servers. Key features include:

- Connection management to MCP servers
- Capability discovery and querying
- Remote tool invocation
- Resource access
- Execution preference configuration

### Hybrid Generation (models/text/hybrid_generate.h)

The hybrid generation module provides a seamless interface for text generation that can transparently switch between local and remote execution. Key features include:

- Unified API for both local and remote generation
- Intelligent decision-making for execution environment
- Performance statistics for both local and remote execution
- Ability to force local or remote execution when needed
- Fallback mechanisms when preferred execution environment is unavailable

### CLI Integration (interface/cli.c)

The CLI has been enhanced with commands to manage MCP connections and control hybrid execution:

- `mcp` command for managing MCP server connections
- `hybrid` command for controlling hybrid execution mode
- Enhanced `generate` command that uses hybrid generation when enabled

## Usage

### MCP Server Connection

To connect to an MCP server, use the `mcp` command:

```
mcp connect <server-url>
```

To disconnect:

```
mcp disconnect
```

To check connection status:

```
mcp status
```

### Hybrid Mode

To enable hybrid generation mode (requires both a loaded model and an MCP connection):

```
hybrid on
```

To disable hybrid mode (use local-only generation):

```
hybrid off
```

To check hybrid mode status:

```
hybrid status
```

To force local execution for the next generation:

```
hybrid force-local
```

To force remote execution for the next generation:

```
hybrid force-remote
```

### Text Generation

The standard `generate` command will automatically use hybrid generation when hybrid mode is enabled:

```
generate "Your prompt here"
```

## Decision Making

The hybrid generation system decides whether to use local or remote execution based on several factors:

1. **Execution Preference**: The configured preference (Always Local, Prefer Local, Prefer MCP, Custom)
2. **Context Size**: If the prompt is approaching the local model's context size, remote execution is preferred
3. **Output Length**: If generating a large number of tokens, remote execution is preferred
4. **Complexity**: More complex requests may be routed to remote servers

The decision can be overridden using the `hybrid force-local` or `hybrid force-remote` commands.

## Extending Hybrid Capabilities

The hybrid architecture is designed to be extensible. Additional modules can be enhanced with hybrid capabilities following the same pattern:

1. Define a hybrid context that wraps both local and remote implementations
2. Implement decision logic for choosing the execution environment
3. Provide a unified API that abstracts the execution environment

## Troubleshooting

### Common Issues

- **Connection Failures**: Ensure the MCP server URL is correct and the server is running
- **Authentication Issues**: Check if the server requires authentication
- **Execution Failures**: If remote execution fails, the system will automatically fall back to local execution if possible
- **Performance Issues**: Use the `hybrid status` command to check performance statistics

### Logging

When troubleshooting, enable verbose logging with:

```
config verbose 2
```

This will provide additional information about the hybrid execution decision-making process and communication with MCP servers.
