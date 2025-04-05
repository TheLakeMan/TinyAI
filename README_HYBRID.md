# TinyAI Hybrid Execution

## Overview

TinyAI now supports hybrid local/remote execution through the Model Context Protocol (MCP). This allows the framework to transparently switch between local model execution and remote API calls to larger, more powerful models when appropriate.

## Features

- **Transparent Execution**: Automatically choose the best execution environment (local or remote) based on context
- **Fallback Support**: Continue with local execution if remote servers are unavailable
- **Performance Monitoring**: Track and compare local vs. remote execution times
- **Manual Control**: Force local or remote execution when needed
- **Flexible Configuration**: Customize hybrid execution parameters to fit your use case

## Setting Up Hybrid Execution

### 1. Connect to an MCP Server

```bash
# Connect to an MCP server
tinyai mcp connect your-mcp-server-url
```

### 2. Enable Hybrid Mode

```bash
# Enable hybrid mode
tinyai hybrid on
```

### 3. Generate Text

```bash
# Generate text using the best execution environment
tinyai generate "Your prompt here" 50 0.7 top_k
```

## Command Reference

### MCP Commands

- `mcp connect <url>`: Connect to an MCP server
- `mcp disconnect`: Disconnect from the current MCP server
- `mcp status`: Display current MCP connection status

### Hybrid Commands

- `hybrid on`: Enable hybrid execution mode
- `hybrid off`: Disable hybrid execution mode (use local-only)
- `hybrid status`: Display hybrid execution statistics and status
- `hybrid force-local`: Force local execution for the next generation
- `hybrid force-remote`: Force remote execution for the next generation

## Example Usage

```bash
# Connect to an MCP server
tinyai mcp connect mock://localhost:8080

# Enable hybrid mode
tinyai hybrid on

# Check hybrid status
tinyai hybrid status

# Generate text with hybrid execution
tinyai generate "TinyAI is an ultra-lightweight AI framework that" 30 0.7 top_k

# Force local execution for comparison
tinyai hybrid force-local
tinyai generate "TinyAI provides memory efficiency with" 30 0.7 top_k

# Force remote execution for comparison
tinyai hybrid force-remote
tinyai generate "The key benefits of TinyAI include" 30 0.7 top_k
```

## Configuration

The hybrid execution behavior can be configured in the model configuration file:

```json
{
  "hybrid": {
    "execution_preference": "prefer_local",  // prefer_local, prefer_remote, always_local
    "remote_threshold_tokens": 200,          // Number of tokens above which to prefer remote
    "context_threshold_ratio": 0.8,          // Context size ratio to trigger remote execution
    "fallback_enabled": true                 // Whether to fall back to local if remote fails
  }
}
```

## Testing

To test the hybrid capabilities, you can use the provided test scripts:

- Windows: `tools/test_pipeline.bat`
- Linux/macOS: `tools/test_pipeline.sh`

These scripts demonstrate the full TinyAI pipeline including tokenization, generation, and hybrid execution.

## Implementation Details

For more information about the hybrid implementation, see `HYBRID_CAPABILITIES.md`.
