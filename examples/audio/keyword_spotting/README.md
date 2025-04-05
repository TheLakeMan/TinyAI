# Keyword Spotting Example

This example demonstrates how to use TinyAI's audio processing capabilities to perform real-time keyword spotting. Keyword spotting (also known as "wake word detection" or "hotword detection") is a technique used to detect specific words or phrases in an audio stream.

## Overview

The keyword spotting example showcases:

1. Loading and processing audio data in real-time or from files
2. Extracting MFCC (Mel-Frequency Cepstral Coefficients) features
3. Using a compact neural network to detect keywords
4. Visualizing and reporting detection results

## Features

- **Ultra-Lightweight Implementation**: The entire model requires less than 50KB of memory
- **4-bit Quantization**: Neural network weights are quantized to 4 bits for maximum efficiency
- **Multiple Keyword Support**: Can be configured to detect various keywords
- **Adjustable Sensitivity**: Customize detection threshold based on the use case
- **SIMD Acceleration**: Utilizes SIMD instructions for efficient processing when available

## Usage

```bash
# Basic usage with default parameters
./tinyai_kws input.wav --keyword "tinyai"

# Detect a different keyword
./tinyai_kws input.wav --keyword "hello"

# Adjust detection threshold (0.0-1.0, higher = less sensitive)
./tinyai_kws input.wav --keyword "tinyai" --threshold 0.7

# Real-time detection from microphone
./tinyai_kws --mic --keyword "tinyai"

# List available keywords in the model
./tinyai_kws --list-keywords

# Show detection visualization
./tinyai_kws input.wav --keyword "tinyai" --visualize
```

## Technical Implementation

The keyword spotting algorithm operates in several stages:

1. **Audio Preprocessing**:
   - Convert audio to 16kHz mono format
   - Apply pre-emphasis filter to enhance high frequencies
   - Split audio into overlapping frames (25ms with 10ms stride)
   - Apply Hamming window to each frame

2. **Feature Extraction**:
   - Extract MFCC features from each frame
   - Compute delta and delta-delta coefficients
   - Normalize features

3. **Neural Network Detection**:
   - Feed features into a compact convolutional neural network
   - Process multiple frames to capture temporal dynamics
   - Output detection score for each keyword

4. **Post-processing**:
   - Apply smoothing to reduce false positives
   - Use dynamic thresholding based on environmental noise
   - Group adjacent detections to avoid duplicates

## Model Architecture

The keyword spotting model uses a tiny convolutional neural network:

- Input: MFCC features (13 coefficients × 49 frames)
- Conv1: 8 filters, 4×4 kernel, ReLU activation
- MaxPool: 2×2 pool size
- Conv2: 16 filters, 3×3 kernel, ReLU activation
- MaxPool: 2×2 pool size
- Dense1: 32 neurons, ReLU activation
- Dense2: 1 neuron (per keyword), Sigmoid activation

Total parameters: ~12K parameters (using 4-bit quantization = ~6KB)

## Memory Usage

- Model weights: ~6KB
- Runtime buffers: ~10KB
- Code footprint: ~30KB
- Total RAM usage: <50KB

## Extension Points

This example can be extended in several ways:

1. Train custom keyword models using your own data
2. Implement multi-keyword detection with prioritization
3. Add noise-robust feature extraction for challenging environments
4. Integrate with other TinyAI components for multimodal applications
