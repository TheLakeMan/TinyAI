# Chatbot Example

This document explains how to use and customize the TinyAI chatbot example application.

## Overview

The TinyAI chatbot example demonstrates how to build a memory-efficient conversational AI using the TinyAI framework. The example includes:

- A small-footprint language model for text generation
- Memory management techniques for constrained environments
- Context handling for multi-turn conversations
- Custom prompt formatting for better responses

## Directory Structure

The example code is located in `examples/chatbot/`:

```
examples/chatbot/
├── CMakeLists.txt        # Build configuration
├── chat_model.c          # Chat model implementation
├── chat_model.h          # Chat model interface
├── main.c                # Example application entry point
└── README.md             # Quick start instructions
```

## Building the Example

To build the chatbot example:

```bash
# Navigate to the chatbot example directory
cd examples/chatbot

# Create a build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build
cmake --build .
```

## Running the Example

After building, run the chatbot application:

```bash
./chatbot_example
```

This will start an interactive chat session where you can type messages and receive responses from the model.

## How It Works

### Chat Model

The chat model (`chat_model.c`) implements a specialized wrapper around the core TinyAI text generation model with optimizations for memory efficiency and conversation handling:

```c
// Initialize the chat model
ChatModel* model = chat_model_init("path/to/model.tmai", 50*1024*1024); // 50MB memory limit

// Generate a response 
const char* response = chat_model_respond(model, "Hello, how are you?");
printf("Model: %s\n", response);

// Clean up
chat_model_free(model);
```

### Memory Management

The example demonstrates several techniques for efficient memory usage:

1. **Memory-mapped model loading**: The model weights are loaded on-demand from disk using memory mapping
2. **Token pruning**: The conversation history is pruned to maintain a fixed context window
3. **In-place token manipulation**: Token sequences are manipulated in-place to avoid extra memory allocation

### Conversation Context

The chatbot maintains a conversation history with a fixed token budget. When the history grows too large, the oldest parts of the conversation are removed to make room for new content while preserving critical context.

## Customizing the Example

### Changing the Model

You can use your own language model by changing the model path in `main.c`:

```c
// Replace with your preferred model
const char* model_path = "path/to/your/model.tmai";
ChatModel* model = chat_model_init(model_path, memory_limit);
```

### Tuning Conversation Parameters

Modify the conversation parameters in `chat_model.h`:

```c
// Maximum number of tokens to retain in conversation history
#define MAX_CONVERSATION_TOKENS 1024

// Maximum number of tokens to generate per response
#define MAX_RESPONSE_TOKENS 128

// Temperature for generation (higher = more random)
#define TEMPERATURE 0.7f
```

### Adding Custom Prompting

Customize the prompt format in `chat_model.c` to achieve different response styles:

```c
// Example of custom prompt format
static const char* PROMPT_FORMAT = 
    "The following is a conversation with an AI assistant. "
    "The assistant is helpful, creative, clever, and friendly.\n\n"
    "%s\n";
```

## Advanced Usage

### Non-interactive Mode

For non-interactive usage (e.g., in a larger application), use the API functions directly:

```c
ChatModel* model = chat_model_init(model_path, memory_limit);

// Add conversation turns
chat_model_add_user_message(model, "Hello, can you help me with a math problem?");
const char* response1 = chat_model_generate_response(model);
printf("Response: %s\n", response1);

chat_model_add_user_message(model, "What's the derivative of x^2?");
const char* response2 = chat_model_generate_response(model);
printf("Response: %s\n", response2);

chat_model_free(model);
```

### Integration with Other Modalities

To integrate the chatbot with other modalities (like image or audio):

```c
// Example of multimodal integration
#include "tinyai/models/multimodal.h"

// Initialize image model
TinyAIImageModel* img_model = tinyai_load_image_model("path/to/image_model.tmai");

// Process image and get description
const char* image_path = "input.jpg";
char image_description[512];
tinyai_describe_image(img_model, image_path, image_description, sizeof(image_description));

// Add the image description to chat context
char prompt[1024];
snprintf(prompt, sizeof(prompt), "I'm looking at an image that shows: %s", image_description);
chat_model_add_user_message(model, prompt);

// Generate response
const char* response = chat_model_generate_response(model);
printf("Response: %s\n", response);

// Clean up
tinyai_free_image_model(img_model);
chat_model_free(model);
```

## Performance Considerations

The chatbot example is optimized for memory efficiency but there are trade-offs with performance:

- Memory mapping improves startup time but can cause slight delays during generation
- Keeping the context window small improves performance but reduces conversation coherence
- Using 4-bit quantization significantly reduces memory usage with minimal quality loss

On typical embedded systems (e.g., Raspberry Pi 4), expect response times of 0.5-2 seconds per generation using the included small model.

## Next Steps

After exploring the chatbot example, consider:

1. [Memory Optimization Guide](../guides/memory_optimization.md) - For more techniques to optimize memory usage
2. [Text Generation API](../api/text_generation.md) - For advanced text generation options
3. [Example: Document Processor](./document_processor.md) - For document processing examples
