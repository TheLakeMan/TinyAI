# TinyAI Visual Question Answering (VQA) Example

This example demonstrates how to use TinyAI's multimodal capabilities to answer natural language questions about images. The visual question answering system combines computer vision and natural language processing to understand image content and provide relevant answers to user questions.

## Features

- **Ask Questions About Images**: Pose natural language questions about image content and receive detailed answers
- **Multiple Answer Styles**: Generate different types of answers
  - Concise: Brief, direct answers
  - Detailed: More comprehensive explanations
  - Factual: Focuses on objective information
  - Casual: Conversational, friendly style
  - Custom: User-defined prompt template for customized answers
- **Batch Processing**: Process multiple image-question pairs from a file
- **Memory Efficient**: Designed for resource-constrained devices
- **Optimized Performance**: Options for quantization and SIMD acceleration

## Building the Example

To build the visual question answering example, ensure the `BUILD_EXAMPLES` option is set to `ON` when configuring TinyAI:

```bash
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=ON
cmake --build .
```

## Usage

The visual question answering example provides a command-line interface for asking questions about images:

```bash
visual_qa [options] <image.jpg> "Your question about the image"
```

### Options

- `--model <file>`: Model structure file (required)
- `--weights <file>`: Model weights file (required)
- `--tokenizer <file>`: Tokenizer vocabulary file (required)
- `--style <style>`: Answer style (concise, detailed, factual, casual)
- `--custom-template <text>`: Custom prompt template for answers
- `--max-tokens <n>`: Maximum tokens in answer (default: 100)
- `--output <file>`: Output file for answers (default: output to terminal only)
- `--batch <file>`: Batch file with image paths and questions
- `--quantized`: Use 4-bit quantization for reduced memory usage
- `--simd`: Enable SIMD acceleration for faster processing
- `--help`: Show help message

## Examples

### Basic Question Answering

```bash
visual_qa --model models/vqa_model.json --weights models/vqa_weights.bin --tokenizer data/vocab.tok photo.jpg "What objects are in this image?"
```

### Using Different Answer Styles

```bash
# Detailed style
visual_qa --model models/vqa_model.json --weights models/vqa_weights.bin --tokenizer data/vocab.tok --style detailed photo.jpg "What is the weather like in this image?"

# Factual style
visual_qa --model models/vqa_model.json --weights models/vqa_weights.bin --tokenizer data/vocab.tok --style factual photo.jpg "How many people are in this image?"
```

### Custom Prompt Template

```bash
visual_qa --model models/vqa_model.json --weights models/vqa_weights.bin --tokenizer data/vocab.tok --custom-template "Answer this question about the image as a pirate would: " photo.jpg "What is happening in this scene?"
```

### Batch Processing

Create a batch file (e.g., `questions.txt`) with image paths and questions in the format:
```
path/to/image1.jpg|What is the main subject of this image?
path/to/image2.jpg|How many people are in this photo?
path/to/image3.jpg|What time of day is shown in this image?
```

Then run:
```bash
visual_qa --model models/vqa_model.json --weights models/vqa_weights.bin --tokenizer data/vocab.tok --batch questions.txt --output answers.txt
```

### Enabling Optimizations

```bash
visual_qa --model models/vqa_model.json --weights models/vqa_weights.bin --tokenizer data/vocab.tok --quantized --simd photo.jpg "What colors are prominent in this image?"
```

## Output Example

When asking a question about an image, the example will output information similar to this:

```
Initializing visual QA system...
Initialization completed in 0.48 seconds
Memory usage:
  Weights: 7.82 MB
  Activations: 1.42 MB
  Total: 9.24 MB

Processing question for image: landscape.jpg
Question: What time of day is shown in this image?

Answer: This image shows sunset or early evening. The warm orange and golden colors in the sky, the long shadows cast by the trees, and the overall soft lighting indicate that the sun is low on the horizon, which is characteristic of sunset.

Generation time: 0.652 seconds

Total processing time: 1.14 seconds
```

## Batch Processing Output

When using the `--batch` option with an output file, the results will be saved in a format like this:

```
Image: mountain.jpg
Question: How high is this mountain?
Answer: Based solely on this image, I cannot determine the exact height of the mountain. However, it appears to be a substantial peak with snow-covered upper sections, suggesting it's likely at least several thousand feet tall. For the precise height, you would need additional information not contained in the image.

Image: city.jpg
Question: What city is this?
Answer: I cannot identify the specific city with certainty from this image alone. The skyline shows a modern metropolitan area with several skyscrapers and what appears to be a body of water nearby, but without distinctive landmarks or additional context, I cannot name the exact city.

Image: family.jpg
Question: How many people are in this photo?
Answer: There are 5 people in this photo.
```

## How It Works

1. The image is loaded and preprocessed (resized, normalized) to match the model's input requirements.
2. The question is combined with a prompt template based on the selected answer style.
3. The multimodal model processes both the image and question together.
4. The model generates a textual answer through an autoregressive process.
5. The answer is post-processed and returned to the user.

The system leverages a multimodal architecture that can reason across both visual and textual information, allowing it to understand relationships between objects in the image and concepts mentioned in the question.

## Example Questions

Here are some example questions you can ask about images:

- **Object identification**: "What objects are in this image?"
- **Counting**: "How many people are in this photo?"
- **Spatial relationships**: "What is to the left of the cat?"
- **Colors**: "What color is the car?"
- **Scene description**: "Where was this photo taken?"
- **Activities**: "What are the people doing in this image?"
- **Time**: "What time of day is it in this photo?"
- **Weather**: "What is the weather like in this scene?"
- **Text recognition**: "What does the sign say?"
- **Emotions**: "How does the person in this image appear to be feeling?"

## Limitations

The quality of answers depends on several factors:

1. Image quality and clarity
2. Question complexity and specificity
3. Whether the question requires knowledge outside what's visible in the image
4. The training data and capabilities of the underlying model

The system performs best with clear, well-framed images and questions about visible content, rather than questions requiring external knowledge or complex reasoning.

## Integration

The visual question answering functionality can be integrated into your own applications by including the `visual_qa.h` header and linking against the TinyAI libraries. See the API documentation for details on how to use the VQA functions programmatically.
