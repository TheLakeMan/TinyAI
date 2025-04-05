# TinyAI Architecture

This document describes the architecture of the TinyAI system.

## System Overview

TinyAI is an ultra-lightweight AI model system designed to run on minimal hardware, including legacy systems. The architecture follows a modular design with the following key components:

```
TinyAI
├── Core Layer - Foundation components
│   ├── Picol Interpreter - Extended Tcl interpreter
│   ├── Runtime Environment - Module loading, resource management
│   ├── Memory Management - Memory pools, 4-bit quantization
│   ├── I/O System - Cross-platform I/O abstractions
│   ├── Configuration - Flexible configuration system
│   └── Sparse Matrix Operations - CSR format with 4-bit quantization
│
├── Model Layer - AI model components
│   ├── Text Generation - 4-bit quantized models
│   │   ├── Tokenizer - Minimal vocabulary tokenizer
│   │   └── Generator - Text generation engine
│   ├── Reasoning (planned) - Knowledge retrieval
│   └── Vision (optional) - Image processing
│
└── Interface Layer - User interaction
    ├── Command Line - Interactive shell and commands
    ├── API - Programmatic access to functionality
    └── Shell - Scripting environment
```

## Key Design Principles

1. **Minimal Resource Usage**: Every component is designed with extreme resource efficiency in mind, targeting systems with as little as 50-100MB of RAM.

2. **4-bit Quantization**: Neural network weights are quantized to 4 bits (16 values) to minimize memory footprint while maintaining reasonable accuracy.

3. **Progressive Loading**: Components are loaded on-demand and can be unloaded when not in use to minimize memory consumption.

4. **Cross-Platform Compatibility**: The system is designed to work on a wide range of platforms, from legacy Windows 95 systems to modern hardware.

5. **Modular Design**: Each component is self-contained and can be used independently, allowing for flexible deployment options.

## Data Flow

The typical data flow for text generation:

1. **Input Processing**: Text input is tokenized using a minimal vocabulary tokenizer
2. **Model Inference**: 4-bit quantized neural network processes tokens
3. **Progressive Generation**: Text is generated token by token with minimal memory overhead
4. **Output Formatting**: Generated tokens are decoded back to text

## Memory Management

Memory is managed through several specialized components:

1. **Memory Pools**: Fixed-size block allocation for efficient memory reuse
2. **4-bit Quantization**: Neural network weights stored with 2 values per byte
3. **Progressive Loading**: Model components loaded on-demand
4. **Resource Tracking**: All resources tracked and automatically released

## Implementation Details

### Core Components

1. **Picol Interpreter**: 
   - Extended from the original 550-line Tcl interpreter
   - Added module loading capabilities
   - Command registration system

2. **Runtime Environment**:
   - Module loading and dependency resolution
   - Resource tracking and management
   - Event system for callbacks

3. **Memory Management**:
   - Memory pools for efficient allocation
   - 4-bit quantization utilities
   - Memory-mapped file support

4. **I/O System**:
   - File operations with abstraction layer
   - Path operations for cross-platform compatibility
   - Binary data handling utilities

5. **Configuration System**:
   - Key-value store for settings
   - File-based configuration persistence
   - Command-line override support

6. **Sparse Matrix Operations**:
   - Compressed Sparse Row (CSR) format for memory-efficient storage
   - 4-bit quantized sparse matrices for extreme compression
   - SIMD-accelerated sparse matrix operations (AVX2, AVX, SSE)
   - Matrix-vector multiplication optimized for sparse data
   - Memory usage and compression ratio calculations
   - Up to 98% memory reduction for large sparse models

### Model Components

1. **Text Tokenizer**:
   - Minimal vocabulary (8,192 tokens)
   - Efficient encoding/decoding
   - Special token handling

2. **Text Generator**:
   - 4-bit quantized matrix operations
   - Multiple sampling methods
   - Context management

### Interface Components

1. **Command Line Interface**:
   - Interactive shell
   - Command parsing and execution
   - Help system

2. **API Layer**:
   - C function calls for embedding
   - Resource management helpers
   - Error handling

## Future Extensions

1. **Reasoning Module**:
   - Vector-based knowledge retrieval
   - Simple reasoning capabilities
   - Compressed knowledge base

2. **Vision Module**:
   - Basic image feature extraction
   - 8-bit quantized vision models
   - Object detection capabilities

3. **Model Training**:
   - Knowledge distillation for compression
   - Quantization-aware fine-tuning
   - Transfer learning from larger models
