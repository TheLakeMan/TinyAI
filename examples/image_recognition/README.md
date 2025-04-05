# TinyAI Image Recognition Example

This example demonstrates how to implement efficient image recognition on resource-constrained devices using the TinyAI framework. It provides a simple interface for classifying images using pre-trained models optimized for edge deployment.

## Features

- **Lightweight Implementation**: Designed for devices with limited memory and processing power
- **Quantized Models**: Support for 4-bit and 8-bit quantized models for efficient inference
- **Real-time Processing**: Optimized for low-latency image classification
- **SIMD Acceleration**: Optional SIMD acceleration for supported hardware
- **Adjustable Confidence Threshold**: Configure minimum confidence for predictions
- **Top-K Predictions**: Return multiple class predictions with confidence scores
- **Batch Processing**: Efficient processing of multiple images
- **Camera Integration**: Optional support for real-time camera feed processing

## Building the Example

To build the image recognition example, ensure the `BUILD_EXAMPLES` option is set to `ON` when configuring TinyAI:

```bash
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=ON
cmake --build .
```

## Usage

The image recognition example provides a command-line interface:

```bash
image_recognition [options] <image_file>
```

### Options

- `--model <file>`: Path to model structure file (required)
- `--weights <file>`: Path to model weights file (required)
- `--labels <file>`: Path to class labels file (required)
- `--top-k <n>`: Return top-K predictions (default: 5)
- `--threshold <value>`: Minimum confidence threshold (default: 0.1)
- `--input-size <size>`: Input image size in pixels (default: 224)
- `--batch <file>`: Process multiple images listed in a file
- `--output <file>`: Save results to file instead of stdout
- `--quantized`: Use 4-bit quantization (default: enabled)
- `--no-quantize`: Disable quantization
- `--simd`: Enable SIMD acceleration
- `--no-simd`: Disable SIMD acceleration
- `--camera`: Use camera feed instead of image file
- `--camera-id <id>`: Camera device ID (default: 0)
- `--help`: Show help message

### Examples

#### Basic Image Classification

```bash
image_recognition --model models/mobilenet_v2.json --weights models/mobilenet_v2_quant.bin --labels data/imagenet_labels.txt cat.jpg
```

#### Adjusting Confidence Threshold and Top-K

```bash
image_recognition --model models/mobilenet_v2.json --weights models/mobilenet_v2_quant.bin --labels data/imagenet_labels.txt --top-k 3 --threshold 0.2 dog.jpg
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
image_recognition --model models/mobilenet_v2.json --weights models/mobilenet_v2_quant.bin --labels data/imagenet_labels.txt --batch images.txt --output results.txt
```

#### Camera Mode

```bash
image_recognition --model models/mobilenet_v2.json --weights models/mobilenet_v2_quant.bin --labels data/imagenet_labels.txt --camera
```

## Supported Models

The example works with CNN-based image classification models like:

- MobileNet V1/V2
- EfficientNet-Lite
- SqueezeNet
- MNASNet
- And other models convertible to TinyAI format

Pre-quantized models are available in the `models/pretrained/` directory.

## Performance Considerations

For optimal performance on resource-constrained devices:

1. Use quantized models (4-bit or 8-bit)
2. Choose an appropriate model size based on your hardware capabilities
3. Enable SIMD acceleration when available
4. Use an appropriate input size for your model (smaller = faster)
5. Adjust the confidence threshold to filter low-confidence predictions

## Model Format

The example expects models in TinyAI's model format. This includes:

1. A JSON model structure file that defines the network architecture
2. A binary weights file containing the model parameters
3. A text file with class labels (one per line)

If you have models in other formats (TensorFlow, PyTorch, ONNX), you can convert them to TinyAI format using the conversion tools in the `tools/` directory.

## Sample Output

When running the example on an image, you'll see output similar to this:

```
Loading model from mobilenet_v2.json and mobilenet_v2_quant.bin...
Model loaded successfully (4.2 MB)
Processing image: cat.jpg
Preprocessing image to 224x224...
Running inference...
Inference completed in 45.3 ms

Top 5 Predictions:
1. Persian cat (99.2%)
2. Tabby cat (0.5%)
3. Egyptian cat (0.2%)
4. Siamese cat (0.1%)
5. Tiger cat (0.0%)
```

When in camera mode, the classifications are printed to the console in real-time as frames are processed.

## Extending the Example

This example can be extended in several ways:

1. Add support for more model architectures
2. Implement object detection (not just classification)
3. Add pre/post-processing steps for specific use cases
4. Integrate with a GUI for visual feedback
5. Combine with the audio example for multimodal sensing
