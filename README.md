# TinyAI - Ultra-Lightweight AI Framework

TinyAI is an extremely memory-efficient AI framework designed to run on minimal hardware, including legacy systems. It uses 4-bit quantization for neural network weights, allowing models to run in as little as 50-100MB of RAM.

## Key Features

- **Extreme Memory Efficiency**: 4-bit quantization reduces model size by up to 8x compared to 32-bit floating point
- **Sparse Matrix Support**: CSR format with 4-bit quantization for up to 98% memory reduction on large sparse models
- **Cross-Platform**: Works on a wide range of hardware, from modern to legacy systems
- **Minimal Dependencies**: Pure C implementation with no external library requirements
- **Progressive Loading**: Components are loaded on-demand to minimize memory footprint
- **Multiple Model Types**: Supports both RNN and Transformer architectures
- **Flexible Text Generation**: Includes multiple sampling methods (greedy, top-k, top-p, temperature)
- **SIMD Acceleration**: Optimized matrix operations using AVX2, AVX, and SSE instructions
- **Hybrid Execution**: Seamlessly switch between local and remote execution via Model Context Protocol
- **Performance Monitoring**: Track and compare local vs. remote execution performance

## Architecture

TinyAI is built with a layered architecture:

1. **Core Layer**: Foundation components
   - Picol Interpreter: Extended Tcl interpreter
   - Runtime Environment: Module loading, resource management
   - Memory Management: Memory pools, 4-bit quantization
   - I/O System: Cross-platform I/O abstractions
   - Configuration: Flexible configuration system

2. **Model Layer**: AI model components
   - Text Generation: 4-bit quantized models
     - Tokenizer: Minimal vocabulary tokenizer
     - Generator: Text generation engine

3. **Interface Layer**: User interaction
   - Command Line: Interactive shell and commands
   - API: Programmatic access to functionality
   - Shell: Scripting environment

4. **Hybrid Capability**: Combined local/remote execution
   - Local Execution: Standalone operation without external dependencies
   - MCP Integration: Enhanced capabilities via Model Context Protocol
   - Dynamic Switching: Graceful transitions between local and remote processing
   - Distributed Computation: Offload heavy processing when beneficial
   - Performance Monitoring: Compare execution times for local and remote operations

## Building TinyAI

### Requirements

- C compiler (GCC, Clang, MSVC, etc.)
- CMake 3.10+

### Build Steps

```bash
# Clone the repository
git clone https://github.com/TheLakeMan/tinyai.git
cd tinyai

# Create a build directory
mkdir build
cd build

# Configure and build (adjust based on your system)

# Linux/macOS (using Makefiles)
cmake ..
cmake --build .

# Windows (using Visual Studio 2022 Build Tools)
# Run these commands in the "Developer Command Prompt for VS 2022"
cmake -G "Visual Studio 17 2022" ..
cmake --build . --config Debug # Or Release

# Run tests (after building)
# Linux/macOS:
# ctest
# Windows (in Developer Command Prompt):
# ctest -C Debug # Or Release
```

**Current Status:** The project now fully compiles and runs on Windows using Visual Studio 2022 Build Tools. All core components have been implemented, including the Picol interpreter, memory management, and I/O systems. The tokenizer and text generation modules have implementations but require further testing. The next development phase focuses on testing and enhancing these components.

## Using TinyAI

### Command Line Interface

The primary way to interact with TinyAI is through its command-line interface:

```bash
# Run the shell
./tinyai shell

# Generate text with a prompt
./tinyai generate "Once upon a time" --max-tokens 100 --temperature 0.7

# Load a custom model
./tinyai -m model.bin -t tokenizer.txt generate "Hello, world!"
```

### Interactive Shell

TinyAI provides an interactive shell for experimenting with models:

```
TinyAI Shell v0.1.0
Type 'help' for available commands, 'exit' to quit

> help
Available commands:
  help           Show help information
  generate       Generate text from a prompt
  tokenize       Tokenize text input
  model          Model management commands
  config         Configuration commands
  mcp            Model Context Protocol connections
  hybrid         Hybrid execution control
  exit           Exit the shell

> model load mymodel.bin
Loading model from mymodel.bin...

> generate "The quick brown fox" 50 0.8
Generating text (max 50 tokens, temp=0.80)...
The quick brown fox jumped over the lazy dog. The dog was not pleased with this arrangement and barked loudly. The fox, startled by the noise, scampered away into the forest.

> mcp connect mock://localhost:8080
Connecting to MCP server at mock://localhost:8080...
Connected to MCP server: TinyAI-MCP (version 0.1.0)

> hybrid on
Hybrid generation mode enabled.

> generate "The framework provides memory efficiency with" 30
Using hybrid generation mode...
Generation used local execution.
Local execution time: 12.45 ms
Tokens per second: 240.16
Generated Text:
The framework provides memory efficiency with 4-bit quantization, allowing it to run on devices with limited RAM. This approach makes it ideal for edge computing and embedded systems.
```

### Configuration

TinyAI can be configured through a configuration file (`tinyai.conf`) or command-line options:

```ini
# TinyAI Configuration File
system.name = "TinyAI"
system.version = "0.1.0"
system.data_dir = "./data"
system.model_dir = "./models"

# Memory settings
memory.pool_size = 1048576
memory.max_allocations = 10000
memory.track_leaks = true

# Model settings
model.context_size = 512
model.hidden_size = 256
model.temperature = 0.7
model.top_k = 40
model.top_p = 0.9
```

## API Usage

TinyAI can be embedded in your application through its C API:

### Basic Usage

```c
#include "tinyai.h"

int main() {
    // Initialize TinyAI
    tinyaiIOInit();
    tinyaiMemTrackInit();
    tinyaiConfigInit();
    
    // Load model and tokenizer
    TinyAITokenizer *tokenizer = tinyaiCreateTokenizer();
    tinyaiLoadVocabulary(tokenizer, "tokenizer.txt");
    
    TinyAIModel *model = tinyaiLoadModel("model.bin", "weights.bin", "tokenizer.txt");
    
    // Set up generation parameters
    TinyAIGenerationParams params;
    params.maxTokens = 100;
    params.samplingMethod = TINYAI_SAMPLING_TOP_P;
    params.temperature = 0.7f;
    params.topP = 0.9f;
    params.seed = time(NULL);
    
    // Encode prompt
    int promptTokens[64];
    int promptLength = tinyaiEncodeText(tokenizer, "Hello, world!", promptTokens, 64);
    params.promptTokens = promptTokens;
    params.promptLength = promptLength;
    
    // Generate text
    int outputTokens[1024];
    int outputLength = tinyaiGenerateText(model, &params, outputTokens, 1024);
    
    // Decode output
    char output[4096];
    tinyaiDecodeTokens(tokenizer, outputTokens, outputLength, output, 4096);
    printf("Generated: %s\n", output);
    
    // Clean up
    tinyaiDestroyModel(model);
    tinyaiDestroyTokenizer(tokenizer);
    tinyaiConfigCleanup();
    tinyaiIOCleanup();
    tinyaiMemTrackCleanup();
    
    return 0;
}
```

### Hybrid Execution Usage

```c
#include "tinyai.h"
#include "core/mcp/mcp_client.h"
#include "models/text/hybrid_generate.h"

int main() {
    // Initialize TinyAI
    tinyaiIOInit();
    tinyaiMemTrackInit();
    tinyaiConfigInit();
    
    // Load model and tokenizer
    TinyAITokenizer *tokenizer = tinyaiCreateTokenizer();
    tinyaiLoadVocabulary(tokenizer, "tokenizer.txt");
    
    TinyAIModel *model = tinyaiLoadModel("model.bin", "weights.bin", "tokenizer.txt");
    
    // Set up MCP client for hybrid execution
    TinyAIMcpConfig mcpConfig;
    tinyaiMcpGetDefaultConfig(&mcpConfig);
    mcpConfig.execPreference = TINYAI_EXEC_PREFER_LOCAL; // Prefer local execution when possible
    
    // Create MCP client
    TinyAIMcpClient *mcpClient = tinyaiMcpCreateClient(&mcpConfig);
    
    // Connect to MCP server
    bool connected = tinyaiMcpConnect(mcpClient, "mcp-server.example.com");
    
    // Create hybrid generation context
    TinyAIHybridGenerate *hybridGen = tinyaiCreateHybridGenerate(model, mcpClient);
    
    // Set up generation parameters
    TinyAIGenerationParams params;
    params.maxTokens = 100;
    params.samplingMethod = TINYAI_SAMPLING_TOP_P;
    params.temperature = 0.7f;
    params.topP = 0.9f;
    params.seed = time(NULL);
    
    // Encode prompt
    int promptTokens[64];
    int promptLength = tinyaiEncodeText(tokenizer, "Hello, world!", promptTokens, 64);
    params.promptTokens = promptTokens;
    params.promptLength = promptLength;
    
    // Generate text with hybrid execution
    int outputTokens[1024];
    int outputLength = tinyaiHybridGenerateText(hybridGen, &params, outputTokens, 1024);
    
    // Get information about execution environment used
    bool usedRemote = tinyaiHybridGenerateUsedRemote(hybridGen);
    
    // Get performance statistics
    double localTime, remoteTime, tokensPerSec;
    tinyaiHybridGenerateGetStats(hybridGen, &localTime, &remoteTime, &tokensPerSec);
    
    printf("Execution used: %s\n", usedRemote ? "Remote" : "Local");
    printf("Time taken: %.2f ms\n", usedRemote ? remoteTime : localTime);
    printf("Performance: %.2f tokens/sec\n", tokensPerSec);
    
    // Decode output
    char output[4096];
    tinyaiDecodeTokens(tokenizer, outputTokens, outputLength, output, 4096);
    printf("Generated: %s\n", output);
    
    // Clean up
    tinyaiDestroyHybridGenerate(hybridGen);
    tinyaiMcpDisconnect(mcpClient);
    tinyaiMcpDestroyClient(mcpClient);
    tinyaiDestroyModel(model);
    tinyaiDestroyTokenizer(tokenizer);
    tinyaiConfigCleanup();
    tinyaiIOCleanup();
    tinyaiMemTrackCleanup();
    
    return 0;
}
```

## Creating a Custom Model

To create a custom TinyAI model:

1. Train a model using your preferred framework (PyTorch, TensorFlow, etc.)
2. Convert the model to TinyAI format using the provided conversion tools
3. Quantize the model to 4-bit precision using `tinyaiQuantizeModel`
4. Save the model, weights, and tokenizer using the TinyAI format

## Memory Usage

TinyAI is designed to be extremely memory-efficient. Here's a comparison with other frameworks:

| Model Size | TinyAI (4-bit) | ONNX (int8) | PyTorch (fp16) | TensorFlow (fp16) |
|------------|----------------|-------------|----------------|-------------------|
| 100M params | ~50MB        | ~100MB      | ~200MB         | ~200MB            |
| 500M params | ~250MB       | ~500MB      | ~1GB           | ~1GB              |
| 1B params   | ~500MB       | ~1GB        | ~2GB           | ~2GB              |

## Contributing

Contributions to TinyAI are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

TinyAI is licensed under the MIT License. See [LICENSE](LICENSE) for details.
