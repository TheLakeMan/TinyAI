# TinyAI Media Tagging Example

This example demonstrates how to use TinyAI for automated tagging and description generation for different media types (images, text, and audio). The media tagger uses multimodal capabilities to identify content, generate relevant tags, and produce human-readable descriptions.

## Features

- **Automatic Media Type Detection**: Identifies and processes files based on their extensions (images, text documents, audio)
- **Multi-Category Tagging**: Provides tags for different aspects of the media content (objects, scenes, emotions, styles, topics)
- **Description Generation**: Creates human-readable descriptions of media content based on identified tags
- **Confidence Scoring**: Assigns confidence scores to tags for better filtering
- **Multiple Output Formats**: Saves tags in different formats (TXT, JSON, XML)
- **Batch Processing**: Support for processing multiple files or entire directories
- **Performance Optimized**: Includes options for 4-bit quantization and SIMD acceleration

## Building the Example

To build the media tagging example, ensure the `BUILD_EXAMPLES` option is set to `ON` when configuring TinyAI:

```bash
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=ON
cmake --build .
```

## Usage

The media tagging example provides a command-line interface for tagging media files:

```bash
media_tagging [options] file1 [file2 file3 ...]
```

### Options

- `--image-model <file>`: Image model structure file
- `--image-weights <file>`: Image model weights file
- `--text-model <file>`: Text model structure file
- `--text-weights <file>`: Text model weights file
- `--tokenizer <file>`: Tokenizer vocabulary file
- `--output <dir>`: Output directory for tag files (default: current dir)
- `--format <format>`: Output format: txt, json, xml (default: json)
- `--threshold <value>`: Confidence threshold (0.0-1.0, default: 0.5)
- `--max-tags <n>`: Maximum number of tags (default: 20)
- `--generate-description`: Generate descriptions for each media file
- `--quantized`: Use 4-bit quantization for models
- `--simd`: Use SIMD acceleration
- `--batch <directory>`: Process all supported files in directory
- `--help`: Show help message

## Examples

### Tagging Images

```bash
media_tagging --image-model models/mobilenet_v2.json --image-weights models/mobilenet_v2.bin photo.jpg
```

### Tagging Text Documents

```bash
media_tagging --text-model models/text_classifier.json --text-weights models/text_classifier.bin --tokenizer data/vocab.tok document.txt
```

### Batch Processing with Description Generation

```bash
media_tagging --image-model models/mobilenet_v2.json --image-weights models/mobilenet_v2.bin --text-model models/text_classifier.json --text-weights models/text_classifier.bin --tokenizer data/vocab.tok --generate-description --batch photos/vacation/
```

### Saving Tags in Different Formats

```bash
media_tagging --image-model models/mobilenet_v2.json --image-weights models/mobilenet_v2.bin --format xml image.jpg
```

## Output Examples

When processing media files, the example will output identified tags and optionally generate descriptions. For example:

```
Processing: landscape.jpg
Identified as: Image
Tags (15):
  mountain (0.92)
  sunset (0.87)
  landscape (0.85)
  nature (0.82)
  scenic (0.75)
  ...

Description:
A breathtaking landscape photograph featuring majestic mountains silhouetted
against a vibrant sunset. The natural scenery is captured with excellent 
composition, highlighting the serene beauty of the wilderness.

Tags saved to: ./landscape.jpg.tags.json
```

## Tag File Format Examples

### JSON (Default)

```json
{
  "tags": [
    {
      "text": "mountain",
      "confidence": 0.9241,
      "category": 0
    },
    {
      "text": "sunset",
      "confidence": 0.8723,
      "category": 0
    },
    ...
  ]
}
```

### XML

```xml
<?xml version="1.0" encoding="UTF-8"?>
<tags>
  <tag>
    <text>mountain</text>
    <confidence>0.9241</confidence>
    <category>0</category>
  </tag>
  <tag>
    <text>sunset</text>
    <confidence>0.8723</confidence>
    <category>0</category>
  </tag>
  ...
</tags>
```

### TXT

```
mountain,0.92,0
sunset,0.87,0
landscape,0.85,0
...
```

## Performance

The media tagger reports processing time and memory usage statistics. Using quantization and SIMD acceleration can significantly improve performance, especially for batch processing.

Example performance metrics:

```
Memory usage:
  Weights: 4567.84 KB
  Activations: 1024.00 KB
  Total: 5591.84 KB

Processed 25 files in 5.32 seconds
```

## Customization

You can customize the media tagger by:
1. Training your own image classification model
2. Training your own text classification model
3. Adjusting the confidence threshold to control tag quality
4. Configuring output formats for different use cases
5. Creating custom prompt templates for description generation

See the TinyAI documentation for more information on training and customizing models.
