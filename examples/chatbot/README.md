# TinyAI On-Device Chatbot Example

This example demonstrates how to implement a memory-efficient chatbot using TinyAI on resource-constrained devices. The chatbot uses a small language model with efficient memory management to provide conversational capabilities while respecting strict memory limitations.

## Features

- **Memory-Constrained Operation**: Designed to work within tight memory limitations (as low as 16MB RAM)
- **Conversation Context Management**: Intelligently manages conversation history within memory constraints
- **Pre-Quantized Model Support**: Uses 4-bit quantized language models for efficient inference
- **Response Streaming**: Generates and displays responses token-by-token
- **Configurable Parameters**: Easily adjust temperature, response length, and memory usage
- **Persistent Chat History**: Option to save and load conversation history
- **Performance Metrics**: Built-in tracking of memory usage, tokens per second, and latency

## Building the Example

To build the chatbot example, ensure the `BUILD_EXAMPLES` option is set to `ON` when configuring TinyAI:

```bash
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=ON
cmake --build .
```

## Usage

The chatbot example provides a simple command-line interface:

```bash
chatbot [options]
```

### Options

- `--model <file>`: Path to model structure file
- `--weights <file>`: Path to model weights file
- `--tokenizer <file>`: Path to tokenizer vocabulary file
- `--memory-limit <MB>`: Maximum memory usage in MB (default: 16)
- `--max-tokens <n>`: Maximum tokens in response (default: 100)
- `--temperature <value>`: Sampling temperature (default: 0.7)
- `--system-prompt <file>`: File containing system prompt/instructions
- `--load-history <file>`: Load conversation history from file
- `--save-history <file>`: Save conversation history to file on exit
- `--quantized`: Use 4-bit quantization (default: enabled)
- `--simd`: Enable SIMD acceleration (default: enabled if available)
- `--help`: Show help message

## Example Usage

### Basic Usage

```bash
chatbot --model models/tiny_llm.json --weights models/tiny_llm_4bit.bin --tokenizer data/vocab.tok
```

### With System Prompt and History Saving

```bash
chatbot --model models/tiny_llm.json --weights models/tiny_llm_4bit.bin --tokenizer data/vocab.tok --system-prompt prompts/assistant.txt --save-history chat_log.json
```

### Fine-tuning Memory and Generation Parameters

```bash
chatbot --model models/tiny_llm.json --weights models/tiny_llm_4bit.bin --tokenizer data/vocab.tok --memory-limit 32 --max-tokens 200 --temperature 0.8
```

## Implementation Details

### Memory Management

The chatbot implements several techniques to operate within tight memory constraints:

1. **Conversation Pruning**: Automatically removes older messages when the context reaches a certain size
2. **Token Budget**: Allocates a fixed number of tokens between user inputs and chat history
3. **Memory Monitoring**: Continuously tracks memory usage to prevent exceeding limits
4. **Quantized Operation**: Uses 4-bit quantized weights to reduce model memory footprint

### Conversation Context Management

The context management system preserves conversation flow while respecting memory constraints:

1. **Importance Weighting**: More recent messages are prioritized over older ones
2. **Context Summarization**: Long conversations can be compressed through summarization
3. **Sliding Window**: Implements a sliding window approach to maintain most relevant context

## Model Support

The chatbot example works with any language model compatible with TinyAI's interface, but is specifically optimized for:

- Small language models (1-10M parameters)
- Models quantized to 4 or 8 bits
- Models trained for instruction-following or chat applications

Pre-quantized models are available in the `models/pretrained/` directory.

## Performance Considerations

For optimal performance on resource-constrained devices:

1. Use a model size appropriate for your hardware (2-5M parameters for very constrained devices)
2. Enable SIMD acceleration when available
3. Use 4-bit quantization to maximize memory efficiency
4. Adjust the conversation context length to balance memory usage and coherence

## Limitations

- Limited contextual understanding compared to larger models
- May produce incorrect or nonsensical responses, especially in complex domains
- Performance heavily depends on the quality and size of the underlying model
- Maximum token length must be configured to prevent memory exhaustion

## Extending the Example

You can extend this example in several ways:

1. Add domain-specific knowledge through system prompts
2. Integrate with external systems via simple APIs
3. Implement additional memory optimization techniques
4. Create a custom UI for better interaction experience
