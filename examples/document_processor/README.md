# TinyAI Document Processor Example

This example demonstrates how to use TinyAI's document processing capabilities for text classification, summarization, and information extraction. The document processor is built on top of TinyAI's text generation capabilities and uses 4-bit quantized models for efficient operation on resource-constrained devices.

## Features

- **Document Classification**: Categorize documents into predefined classes with confidence scores
- **Document Summarization**: Generate concise summaries of long documents
- **Information Extraction**: Extract specific information from documents using prompts
- **Memory Efficient**: Uses 4-bit quantized models for minimal memory footprint
- **Performance Optimized**: SIMD acceleration for faster processing when available

## Building the Example

To build the document processor example, ensure the `BUILD_EXAMPLES` option is set to `ON` when configuring TinyAI:

```bash
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=ON
cmake --build .
```

## Usage

The document processor example provides a command-line interface for processing documents:

```bash
document_processor [options] <mode> <document_file>
```

### Modes

- `classify`: Classify the document into predefined categories
- `summarize`: Generate a summary of the document
- `extract`: Extract specific information from the document

### Options

- `--model <file>`: Path to model structure file (required)
- `--weights <file>`: Path to model weights file (required)
- `--vocab <file>`: Path to tokenizer vocabulary file (required)
- `--classes <file>`: Path to classes file for classification (one class per line)
- `--prompt <text>`: Extraction prompt for extract mode
- `--max-input <n>`: Maximum input length in tokens (default: 1024)
- `--max-output <n>`: Maximum output length in tokens (default: 256)
- `--simd`: Enable SIMD acceleration
- `--quantized`: Use 4-bit quantization
- `--help`: Display help message

## Examples

### Document Classification

```bash
document_processor classify document.txt --model model.json --weights weights.bin --vocab vocab.txt --classes classes.txt
```

The classes file should contain one class label per line.

### Document Summarization

```bash
document_processor summarize document.txt --model model.json --weights weights.bin --vocab vocab.txt
```

### Information Extraction

```bash
document_processor extract document.txt --model model.json --weights weights.bin --vocab vocab.txt --prompt "Extract all dates and their corresponding events"
```

## Memory Usage

The document processor reports memory usage statistics when executed, showing:
- Weight memory (model parameters)
- Activation memory (temporary buffers)
- Total memory usage

By using 4-bit quantization, the memory footprint can be reduced by up to 8x compared to full-precision models.

## Performance

The document processor reports processing time when executed. When using SIMD acceleration, processing time can be significantly reduced, especially on platforms with AVX/AVX2 support.

Example output:
```
Memory usage:
  Weights: 3.50 MB
  Activations: 0.75 MB
  Total: 4.25 MB

Processing completed in 0.35 seconds
```

## Customization

You can customize the document processor by:
1. Training your own classification model
2. Creating your own vocabulary file
3. Adjusting the input and output token lengths
4. Using different prompts for information extraction

See the TinyAI documentation for more information on training and customizing models.
