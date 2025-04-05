# Media Tagging Example

This document explains how to use and customize the TinyAI media tagging example application.

## Overview

The TinyAI media tagging example demonstrates how to build a system that can automatically analyze and tag various types of media, including images, text, and audio. The example includes:

- Image tagging and scene recognition
- Text content classification and keyword extraction
- Audio content analysis and classification
- Combined multimodal analysis for richer tagging

## Directory Structure

The example code is located in `examples/media_tagging/`:

```
examples/media_tagging/
├── CMakeLists.txt              # Build configuration
├── image_tagger.c              # Image tagging implementation
├── image_tagger.h              # Image tagging interface
├── text_tagger.c               # Text tagging implementation
├── text_tagger.h               # Text tagging interface
├── audio_tagger.c              # Audio tagging implementation
├── audio_tagger.h              # Audio tagging interface
├── multimodal_analyzer.c       # Multimodal analysis implementation
├── multimodal_analyzer.h       # Multimodal analysis interface
├── main.c                      # Example application entry point
└── README.md                   # Quick start instructions
```

## Building the Example

To build the media tagging example:

```bash
# Navigate to the media tagging example directory
cd examples/media_tagging

# Create a build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build
cmake --build .
```

## Running the Example

After building, run the media tagging application:

```bash
# Tag a single image
./media_tagger --image path/to/image.jpg

# Tag a text file
./media_tagger --text path/to/document.txt

# Tag an audio file
./media_tagger --audio path/to/audio.wav

# Multimodal analysis (combining different types)
./media_tagger --image path/to/image.jpg --text path/to/caption.txt
```

For batch processing:

```bash
# Process a directory of mixed media
./media_tagger --input-dir ./media/ --output-file tags.json
```

## How It Works

### Image Tagging

The image tagger (`image_tagger.c`) analyzes images to identify objects, scenes, and attributes:

```c
// Initialize the image tagger
ImageTagger* tagger = image_tagger_init("path/to/image_model.tmai");

// Tag an image
const char* image_path = "path/to/image.jpg";
ImageTags tags;
image_tagger_analyze(tagger, image_path, &tags);

// Print the tags
printf("Image tags (%d):\n", tags.count);
for (int i = 0; i < tags.count; i++) {
    printf("- %s (confidence: %.2f)\n", tags.items[i].name, tags.items[i].confidence);
}

// Clean up
image_tagger_free(tagger);
```

### Text Tagging

The text tagger (`text_tagger.c`) analyzes text to extract topics, sentiment, and keywords:

```c
// Initialize the text tagger
TextTagger* tagger = text_tagger_init("path/to/text_model.tmai");

// Analyze text
const char* text = "..."; // Text content
TextAnalysis analysis;
text_tagger_analyze(tagger, text, &analysis);

// Print the analysis results
printf("Topics: %s\n", analysis.topics);
printf("Sentiment: %s (score: %.2f)\n", 
       analysis.sentiment_score > 0 ? "Positive" : "Negative", 
       fabs(analysis.sentiment_score));
printf("Keywords: %s\n", analysis.keywords);

// Clean up
text_tagger_free(tagger);
```

### Audio Tagging

The audio tagger (`audio_tagger.c`) analyzes audio to detect sounds, speech, and music:

```c
// Initialize the audio tagger
AudioTagger* tagger = audio_tagger_init("path/to/audio_model.tmai");

// Analyze audio
const char* audio_path = "path/to/audio.wav";
AudioTags tags;
audio_tagger_analyze(tagger, audio_path, &tags);

// Print the audio tags
printf("Audio contains:\n");
if (tags.has_speech) printf("- Speech (confidence: %.2f)\n", tags.speech_confidence);
if (tags.has_music) printf("- Music (confidence: %.2f)\n", tags.music_confidence);
if (tags.ambient_sounds[0]) printf("- Ambient sounds: %s\n", tags.ambient_sounds);

// Clean up
audio_tagger_free(tagger);
```

### Multimodal Analysis

The multimodal analyzer (`multimodal_analyzer.c`) combines information from different modalities:

```c
// Initialize the multimodal analyzer
MultimodalAnalyzer* analyzer = multimodal_analyzer_init(
    "path/to/image_model.tmai",
    "path/to/text_model.tmai",
    "path/to/audio_model.tmai",
    "path/to/fusion_model.tmai"
);

// Load media files
Media media = {0};
media.image_path = "path/to/image.jpg";
media.text = "Image caption or related text";
media.audio_path = "path/to/audio.wav";

// Perform multimodal analysis
MultimodalAnalysis analysis;
multimodal_analyzer_analyze(analyzer, &media, &analysis);

// Print combined analysis
printf("Media content: %s\n", analysis.description);
printf("Tags: %s\n", analysis.tags);
printf("Content category: %s\n", analysis.category);

// Clean up
multimodal_analyzer_free(analyzer);
```

### Memory Management

The example uses several techniques for efficient memory usage:

1. **Shared models**: Models are loaded once and shared between different media files
2. **Progressive analysis**: Media is analyzed in stages, releasing intermediate resources
3. **Memory mapping**: Large models are loaded using memory mapping to reduce RAM usage

## Customizing the Example

### Using Custom Models

You can use your own models by changing the model paths:

```c
// Replace with your preferred models
const char* image_model_path = "path/to/your/image_model.tmai";
const char* text_model_path = "path/to/your/text_model.tmai";
const char* audio_model_path = "path/to/your/audio_model.tmai";
const char* fusion_model_path = "path/to/your/fusion_model.tmai";
```

### Custom Tag Categories

Modify the tag categories in `media_tagger.h`:

```c
// Default tag categories
typedef enum {
    TAG_OBJECT,      // Physical objects
    TAG_SCENE,       // Scene or environment
    TAG_ACTION,      // Activities or actions
    TAG_CONCEPT,     // Abstract concepts
    TAG_SENTIMENT,   // Emotional content
    TAG_STYLE        // Stylistic elements
} TagCategory;

// Add custom categories
#define TAG_CUSTOM_BRAND  64   // Brand identification
#define TAG_CUSTOM_PERSON 128  // Person identification
```

### Extending Supported Media Formats

Add support for additional media formats:

```c
// Register a new image format handler (using a third-party library)
bool load_webp_image(const char* file_path, ImageData* image) {
    // Implementation using WebP library
    // ...
}

// Register the format handler
ImageFormatHandler webp_handler = {
    .extension = ".webp",
    .load_func = load_webp_image
};
image_tagger_register_format(&webp_handler);
```

## Advanced Usage

### Tag-Based Media Organization

Build a media organization system using the tagging capabilities:

```c
// Create a media database
MediaDB* db = media_db_create("media_library.db");

// Process and index a collection of media files
const char* media_dir = "./media_collection/";
media_tagger_process_directory(tagger, media_dir, 
                              MEDIA_TAG_IMAGES | MEDIA_TAG_TEXT | MEDIA_TAG_AUDIO,
                              media_db_add_callback, db);

// Search media by tags
MediaSearchResult results;
const char* search_tags = "beach,sunset,waves";
media_db_search_by_tags(db, search_tags, &results);

// Print search results
printf("Found %d matching media files:\n", results.count);
for (int i = 0; i < results.count; i++) {
    printf("%d. %s (relevance: %.2f)\n", 
           i+1, results.items[i].path, results.items[i].relevance);
}

// Clean up
media_search_free(&results);
media_db_close(db);
```

### Integration with Other TinyAI Components

Combine the media tagger with other TinyAI capabilities:

```c
// Example: Integrating with document processor
#include "tinyai/document_processor.h"

// Initialize models
ImageTagger* img_tagger = image_tagger_init(image_model_path);
DocumentSummarizer* summarizer = doc_summarizer_init(summarizer_model_path);

// Process a document with embedded images
DocumentWithImages doc;
load_document_with_images("report.docx", &doc);

// Summarize document text
char summary[1024];
doc_summarizer_summarize(summarizer, doc.text, summary, sizeof(summary));

// Tag embedded images
for (int i = 0; i < doc.image_count; i++) {
    ImageTags tags;
    image_tagger_analyze_buffer(img_tagger, 
                              doc.images[i].data, 
                              doc.images[i].size, 
                              &tags);
    
    // Add image tags to document metadata
    add_image_tags_to_doc(&doc, i, &tags);
}

// Create enhanced document with tags and summary
save_enhanced_document("tagged_report.json", &doc, summary);

// Clean up
image_tagger_free(img_tagger);
doc_summarizer_free(summarizer);
```

## Performance Considerations

The media tagging example balances performance with accuracy:

- Image tagging is typically the most computationally intensive operation
- Audio analysis can be memory-intensive for longer audio files
- Text analysis is generally the fastest modality to process
- Multimodal analysis combines all modalities and is therefore slower but more comprehensive

Performance can be optimized by:
- Using the appropriate quantization level for each model
- Processing images at reduced resolution for faster analysis
- Using chunking for processing long audio files
- Running modality-specific analyses in parallel using threads

## Next Steps

After exploring the media tagging example, consider:

1. [Memory Optimization Guide](../guides/memory_optimization.md) - For more memory efficiency techniques
2. [Image Processing API](../api/image_processing.md) - For more advanced image operations
3. [Example: Chatbot](./chatbot.md) - For conversational AI examples
4. [Example: Document Processor](./document_processor.md) - For document analysis examples
