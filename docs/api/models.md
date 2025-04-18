# Models API Reference

## Overview

The Models API provides functions for loading, managing, and using AI models in TinyAI, including text generation, image processing, and multimodal capabilities.

## Model Management

### `tinyai_load_model()`
```c
TinyAIModel* tinyai_load_model(const char* path);
```
Loads a model from a file.

**Parameters:**
- `path`: Path to the model file

**Returns:**
- Pointer to loaded model or NULL on failure

**Example:**
```c
TinyAIModel* model = tinyai_load_model("model.tmai");
if (!model) {
    printf("Error: %s\n", tinyai_get_error());
    return 1;
}
```

### `tinyai_free_model()`
```c
void tinyai_free_model(TinyAIModel* model);
```
Frees a loaded model.

**Parameters:**
- `model`: Pointer to model to free

## Text Generation

### `tinyai_generate_text()`
```c
void tinyai_generate_text(TinyAIModel* model, const char* prompt, char* output, size_t max_length, int max_tokens);
```
Generates text based on a prompt.

**Parameters:**
- `model`: Pointer to model
- `prompt`: Input prompt
- `output`: Buffer to store generated text
- `max_length`: Maximum output length
- `max_tokens`: Maximum number of tokens to generate

**Example:**
```c
const char* prompt = "TinyAI is";
char output[256];
tinyai_generate_text(model, prompt, output, sizeof(output), 50);
printf("Generated: %s\n", output);
```

### `tinyai_generate_text_with_config()`
```c
void tinyai_generate_text_with_config(TinyAIModel* model, const char* prompt, char* output, size_t max_length, const TinyAIGenerationConfig* config);
```
Generates text with custom configuration.

**Parameters:**
- `model`: Pointer to model
- `prompt`: Input prompt
- `output`: Buffer to store generated text
- `max_length`: Maximum output length
- `config`: Generation configuration

**Example:**
```c
TinyAIGenerationConfig config = {
    .temperature = 0.7f,
    .top_k = 40,
    .top_p = 0.9f
};
tinyai_generate_text_with_config(model, prompt, output, sizeof(output), &config);
```

## Image Processing

### `tinyai_process_image()`
```c
void tinyai_process_image(TinyAIModel* model, const TinyAIImage* image, TinyAIImage* output);
```
Processes an image using a model.

**Parameters:**
- `model`: Pointer to model
- `image`: Input image
- `output`: Output image

**Example:**
```c
TinyAIImage input, output;
// Load input image
tinyai_process_image(model, &input, &output);
// Use processed image
```

## Multimodal Operations

### `tinyai_process_multimodal()`
```c
void tinyai_process_multimodal(TinyAIModel* model, const TinyAIMultimodalInput* input, TinyAIMultimodalOutput* output);
```
Processes multimodal input (text and image).

**Parameters:**
- `model`: Pointer to model
- `input`: Multimodal input
- `output`: Multimodal output

**Example:**
```c
TinyAIMultimodalInput input = {
    .text = "Describe this image",
    .image = &image
};
TinyAIMultimodalOutput output;
tinyai_process_multimodal(model, &input, &output);
printf("Description: %s\n", output.text);
```

## Data Types

### `TinyAIModel`
```c
typedef struct {
    void* internal;
    TinyAIModelType type;
    size_t parameter_count;
} TinyAIModel;
```
Model structure.

### `TinyAIGenerationConfig`
```c
typedef struct {
    float temperature;
    int top_k;
    float top_p;
    int max_tokens;
} TinyAIGenerationConfig;
```
Text generation configuration.

### `TinyAIImage`
```c
typedef struct {
    int width;
    int height;
    int channels;
    uint8_t* data;
} TinyAIImage;
```
Image structure.

### `TinyAIMultimodalInput`
```c
typedef struct {
    const char* text;
    const TinyAIImage* image;
} TinyAIMultimodalInput;
```
Multimodal input structure.

### `TinyAIMultimodalOutput`
```c
typedef struct {
    char* text;
    TinyAIImage* image;
} TinyAIMultimodalOutput;
```
Multimodal output structure.

### `TinyAIModelType`
```c
typedef enum {
    TINYAI_MODEL_TEXT,
    TINYAI_MODEL_IMAGE,
    TINYAI_MODEL_MULTIMODAL
} TinyAIModelType;
```
Model type enumeration.

## Best Practices

1. Check model loading success
2. Configure generation parameters appropriately
3. Handle memory management properly
4. Use appropriate model types
5. Monitor performance metrics
6. Clean up resources

## Common Patterns

### Text Generation
```c
// Load model
TinyAIModel* model = tinyai_load_model("text_model.tmai");
if (!model) {
    printf("Error: %s\n", tinyai_get_error());
    return 1;
}

// Configure generation
TinyAIGenerationConfig config = {
    .temperature = 0.7f,
    .top_k = 40,
    .top_p = 0.9f
};

// Generate text
const char* prompt = "TinyAI is";
char output[256];
tinyai_generate_text_with_config(model, prompt, output, sizeof(output), &config);
printf("Generated: %s\n", output);

// Clean up
tinyai_free_model(model);
```

### Image Processing
```c
// Load model
TinyAIModel* model = tinyai_load_model("image_model.tmai");
if (!model) {
    printf("Error: %s\n", tinyai_get_error());
    return 1;
}

// Process image
TinyAIImage input, output;
// Load input image
tinyai_process_image(model, &input, &output);
// Use processed image

// Clean up
tinyai_free_model(model);
```

### Multimodal Processing
```c
// Load model
TinyAIModel* model = tinyai_load_model("multimodal_model.tmai");
if (!model) {
    printf("Error: %s\n", tinyai_get_error());
    return 1;
}

// Process multimodal input
TinyAIMultimodalInput input = {
    .text = "Describe this image",
    .image = &image
};
TinyAIMultimodalOutput output;
tinyai_process_multimodal(model, &input, &output);
printf("Description: %s\n", output.text);

// Clean up
tinyai_free_model(model);
```

## Related Documentation

- [Core API](core.md)
- [Memory Management API](memory.md)
- [Performance Tools API](performance.md) 