# TinyAI Pre-Quantized Models

This directory contains pre-quantized models for various tasks, optimized for use with the TinyAI framework. These models are designed to run efficiently on resource-constrained devices while maintaining reasonable accuracy.

## Directory Structure

```
pretrained/
├── language_models/    # Text generation, classification, and summarization models
├── image_models/       # Image classification and feature extraction models
├── multimodal_models/  # Models combining vision and language capabilities
└── audio_models/       # Audio classification, keyword spotting, and speech models
```

## Available Models

### Language Models

| Model Name | Parameters | Size | Description | Use Case |
|------------|------------|------|-------------|----------|
| tiny_llm_1m | 1M | 0.5MB | Small language model for simple text generation | Chatbots, simple assistants |
| tiny_llm_3m | 3M | 1.6MB | Improved language model with better coherence | More complex dialogues, simple Q&A |
| tiny_llm_7m | 7M | 3.4MB | Enhanced language model for broader knowledge | Task assistance, content generation |
| tiny_classifier | 2M | 1MB | Text classification model | Sentiment analysis, topic classification |
| tiny_summarizer | 4M | 2MB | Text summarization model | Document summarization |

### Image Models

| Model Name | Parameters | Size | Description | Use Case |
|------------|------------|------|-------------|----------|
| mobilenet_v2_tiny | 1.4M | 0.7MB | Lightweight image classifier | Basic image classification |
| mobilenet_v2_small | 2.5M | 1.2MB | Improved image classifier | More accurate classification |
| mobilenet_v2_medium | 3.5M | 1.7MB | Enhanced image classifier | Detailed image classification |
| tiny_vision_encoder | 5M | 2.5MB | Vision feature extraction | Image feature extraction for multimodal tasks |

### Multimodal Models

| Model Name | Parameters | Size | Description | Use Case |
|------------|------------|------|-------------|----------|
| image_captioner_tiny | 6M | 3MB | Image captioning model | Generating descriptions for images |
| visual_qa_tiny | 8M | 4MB | Visual question answering | Answering questions about images |

### Audio Models

| Model Name | Parameters | Size | Description | Use Case |
|------------|------------|------|-------------|----------|
| keyword_spotter | 0.5M | 0.3MB | Keyword spotting model | Wake word detection |
| speech_commands | 1M | 0.5MB | Speech command recognition | Basic voice commands |
| audio_classifier | 2M | 1MB | Audio classification | Environmental sound classification |

## Model Format

Each model includes three files:

1. **[model_name].json**: Model architecture definition
2. **[model_name].bin**: Quantized model weights (4-bit)
3. **[model_name]_metadata.json**: Model metadata (input/output shapes, preprocessing info)

For language models, there's also a vocabulary file:
4. **[model_name]_vocab.tok**: Tokenizer vocabulary

## Usage

To use these models in your TinyAI application:

```c
// Example for loading a language model
TinyAIModel* model = tinyaiLoadModel(
    "models/pretrained/language_models/tiny_llm_3m.json",
    "models/pretrained/language_models/tiny_llm_3m.bin",
    "models/pretrained/language_models/tiny_llm_3m_vocab.tok"
);

// Example for loading an image model
TinyAIImageModel* imageModel = tinyaiLoadImageModel(
    "models/pretrained/image_models/mobilenet_v2_tiny.json",
    "models/pretrained/image_models/mobilenet_v2_tiny.bin"
);
```

See the example applications in the `examples/` directory for complete usage examples.

## Model Quantization

All models in this directory are quantized to 4 bits using TinyAI's symmetric quantization scheme, offering a compression ratio of approximately 8x compared to 32-bit floating point models, with minimal accuracy loss.

The quantization process includes:
1. Weight clustering to find optimal centroids
2. Linear quantization with symmetric ranges
3. Calibration using representative data

## Performance Characteristics

| Model | Inference Time (Raspberry Pi 4) | Memory Usage |
|-------|----------------------------------|--------------|
| tiny_llm_1m | ~20ms/token | ~2MB |
| tiny_llm_3m | ~50ms/token | ~5MB |
| tiny_llm_7m | ~120ms/token | ~10MB |
| mobilenet_v2_tiny | ~150ms/image | ~3MB |
| keyword_spotter | ~10ms/frame | ~1MB |

Performance measured using SIMD acceleration when available.

## Adding Your Own Models

You can add your own pre-quantized models to these directories. To convert and quantize your models, use the TinyAI conversion tools in the `tools/` directory:

```bash
# Example: Convert and quantize a TensorFlow model
python tools/convert_tensorflow_to_tinyai.py --model my_model.h5 --output models/pretrained/image_models/my_model --bits 4
```

## Limitations

- Models are optimized for size and speed rather than maximum accuracy
- Context lengths for language models are limited (typically 512-1024 tokens)
- Image models typically use 224x224 resolution inputs
- Audio models are designed for 16kHz mono audio

## License Information

These model weights are provided under the MIT license, same as the TinyAI framework. Some models are based on open-source models and may carry additional attributions - see the metadata files for details.
