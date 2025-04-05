# TinyAI Image Captioning Example

This example demonstrates how to implement efficient image captioning on resource-constrained devices using the TinyAI framework. Image captioning combines computer vision and natural language processing to generate textual descriptions of images, and is a perfect showcase of TinyAI's multimodal capabilities.

## Features

- **Multimodal Processing**: Integration of vision encoder and text decoder
- **Lightweight Implementation**: Optimized for devices with limited memory and processing power
- **Adaptive Beam Search**: Efficiently explore caption candidates with configurable beam width
- **Quantized Operation**: Support for 4-bit and 8-bit quantized models
- **SIMD Acceleration**: Optional SIMD acceleration for supported hardware
- **Streaming Generation**: Display captions as they're generated token-by-token
- **Memory-Constrained Operation**: Designed for efficient operation on devices with limited RAM

## Building the Example

To build the image captioning example, ensure the `BUILD_EXAMPLES` option is set to `ON` when configuring TinyAI:

```bash
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=ON
cmake --build .
```

## Usage

The image captioning example provides a command-line interface:

```bash
image_captioning [options] <image_file>
```

### Options

- `--vision-model <file>`: Path to vision model structure file (required)
- `--vision-weights <file>`: Path to vision model weights file (required)
- `--language-model <file>`: Path to language model structure file (required)
- `--language-weights <file>`: Path to language model weights file (required)
- `--tokenizer <file>`: Path to tokenizer vocabulary file (required)
- `--fusion <method>`: Fusion method - one of: concat, add, multiply, attention (default: attention)
- `--max-tokens <n>`: Maximum tokens in caption (default: 50)
- `--beam-width <n>`: Beam search width (default: 3, set to 1 for greedy search)
- `--temperature <value>`: Sampling temperature (default: 0.8)
- `--batch <file>`: Process multiple images listed in a file
- `--output <file>`: Save results to file instead of stdout
- `--quantized`: Use 4-bit quantization (default: enabled)
- `--no-quantize`: Disable quantization
- `--simd`: Enable SIMD acceleration
- `--no-simd`: Disable SIMD acceleration
- `--help`: Show help message

### Examples

#### Basic Image Captioning

```bash
image_captioning --vision-model models/resnet50.json --vision-weights models/resnet50_quant.bin \
                 --language-model models/captioner.json --language-weights models/captioner_quant.bin \
                 --tokenizer data/vocab.tok cat.jpg
```

#### Adjusting Generation Parameters

```bash
image_captioning --vision-model models/resnet50.json --vision-weights models/resnet50_quant.bin \
                 --language-model models/captioner.json --language-weights models/captioner_quant.bin \
                 --tokenizer data/vocab.tok --beam-width 5 --max-tokens 75 --temperature 0.6 cat.jpg
```

#### Batch Processing

Create a text file with image paths (one per line):
```
path/to/image1.jpg
path/to/image2.jpg
path/to/image3.jpg
```

Then run:
```bash
image_captioning --vision-model models/resnet50.json --vision-weights models/resnet50_quant.bin \
                 --language-model models/captioner.json --language-weights models/captioner_quant.bin \
                 --tokenizer data/vocab.tok --batch images.txt --output captions.txt
```

## How it Works

The image captioning process involves three main components:

1. **Vision Encoder**: Extracts visual features from the input image using a convolutional neural network (CNN) or vision transformer
2. **Multimodal Fusion**: Combines visual features with text features using methods like concatenation, addition, multiplication, or attention mechanisms
3. **Language Decoder**: Generates caption text based on the fused features using a language model

The process flow is:

1. Load and preprocess the input image
2. Run the vision encoder to extract image features
3. Initialize caption generation with a start token
4. For each generation step:
   - Combine image features with current text representation
   - Predict the next token in the caption
   - Add the token to the caption
   - Repeat until end token or maximum length is reached
5. Return the generated caption

## Performance Considerations

For optimal performance on resource-constrained devices:

1. Use quantized models (4-bit quantization reduces memory footprint by up to 8x)
2. Choose an appropriate model size based on your hardware capabilities
3. Enable SIMD acceleration when available
4. Use a smaller beam width for faster generation (beam_width=1 for greedy search)
5. Adjust max tokens to limit generation length for memory-constrained scenarios

## Sample Output

When running the example on an image, you'll see output similar to this:

```
Loading models...
Models loaded successfully in 1.23 seconds
Processing image: cat.jpg
Running vision encoder...
Vision encoding completed in 85.4 ms
Generating caption...
Caption generated in 132.8 ms

Generated Caption:
"A fluffy orange cat is sitting on a windowsill looking outside at birds."

Memory Usage:
  Vision Model:  4.2 MB
  Language Model: 2.8 MB
  Total:  7.5 MB
```

## Extending the Example

This example can be extended in several ways:

1. Add support for more vision encoders (EfficientNet, ViT, etc.)
2. Implement different fusion methods
3. Add fine-tuning capabilities for domain-specific captioning
4. Create a GUI for interactive captioning
5. Implement attention visualization to show what the model focuses on
