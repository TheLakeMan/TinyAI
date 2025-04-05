# Document Processor Example

This document explains how to use and customize the TinyAI document processing example application.

## Overview

The TinyAI document processor example demonstrates how to build document analysis tools using the TinyAI framework. The example includes:

- Text classification for document categorization
- Document summarization capabilities
- Keyword extraction and entity recognition
- Memory-efficient document processing

## Directory Structure

The example code is located in `examples/document_processor/`:

```
examples/document_processor/
├── CMakeLists.txt              # Build configuration
├── document_classifier.c       # Document classification implementation
├── document_classifier.h       # Document classification interface
├── document_summarizer.c       # Document summarization implementation
├── document_summarizer.h       # Document summarization interface
├── keyword_extractor.c         # Keyword extraction implementation
├── keyword_extractor.h         # Keyword extraction interface
├── main.c                      # Example application entry point
└── README.md                   # Quick start instructions
```

## Building the Example

To build the document processor example:

```bash
# Navigate to the document processor example directory
cd examples/document_processor

# Create a build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build
cmake --build .
```

## Running the Example

After building, run the document processor application:

```bash
# Process a single document
./document_processor --input document.txt --classify --summarize

# Process multiple documents
./document_processor --input-dir ./documents/ --output-dir ./results/ --classify --extract-keywords
```

## How It Works

### Document Classification

The document classifier (`document_classifier.c`) uses a small text classification model to categorize documents into predefined classes:

```c
// Initialize the classifier
DocumentClassifier* classifier = doc_classifier_init("path/to/classifier_model.tmai");

// Classify a document
const char* document_text = "..."; // Document content
ClassificationResult result = doc_classifier_classify(classifier, document_text);

// Print the classification results
printf("Document class: %s (confidence: %.2f%%)\n", result.class_name, result.confidence * 100.0f);

// Clean up
doc_classifier_free(classifier);
```

### Document Summarization

The document summarizer (`document_summarizer.c`) creates concise summaries of longer texts:

```c
// Initialize the summarizer
DocumentSummarizer* summarizer = doc_summarizer_init("path/to/summarizer_model.tmai");

// Summarize a document
const char* document_text = "..."; // Document content
char summary[1024];
doc_summarizer_summarize(summarizer, document_text, summary, sizeof(summary));

// Print the summary
printf("Summary: %s\n", summary);

// Clean up
doc_summarizer_free(summarizer);
```

### Keyword Extraction

The keyword extractor (`keyword_extractor.c`) identifies important terms and entities in documents:

```c
// Initialize the keyword extractor
KeywordExtractor* extractor = keyword_extractor_init("path/to/extractor_model.tmai");

// Extract keywords from a document
const char* document_text = "..."; // Document content
Keyword keywords[MAX_KEYWORDS];
int keyword_count = 0;
keyword_extractor_extract(extractor, document_text, keywords, MAX_KEYWORDS, &keyword_count);

// Print the keywords
printf("Keywords:\n");
for (int i = 0; i < keyword_count; i++) {
    printf("- %s (relevance: %.2f)\n", keywords[i].text, keywords[i].relevance);
}

// Clean up
keyword_extractor_free(extractor);
```

### Memory Management

The example uses several techniques for efficient memory usage:

1. **Streaming document processing**: Documents are processed in chunks to avoid loading large files entirely into memory
2. **Memory mapping**: Model weights are loaded on-demand from disk
3. **Token-level processing**: Operations are performed on token sequences rather than allocating strings

## Customizing the Example

### Using Custom Models

You can use your own models by changing the model paths:

```c
// Replace with your preferred models
const char* classifier_model_path = "path/to/your/classifier.tmai";
const char* summarizer_model_path = "path/to/your/summarizer.tmai";
const char* extractor_model_path = "path/to/your/extractor.tmai";
```

### Adding Custom Document Formats

The example supports plain text by default, but you can extend it to handle other formats by implementing format-specific readers:

```c
// Add to document_processor.h
typedef struct {
    const char* file_extension;
    bool (*read_document)(const char* file_path, char* buffer, size_t buffer_size);
} DocumentFormatReader;

// Example PDF reader implementation (using a third-party PDF library)
bool read_pdf_document(const char* file_path, char* buffer, size_t buffer_size) {
    // Implementation using a PDF library
    // ...
}

// Register the format reader
DocumentFormatReader pdf_reader = {
    .file_extension = ".pdf",
    .read_document = read_pdf_document
};
document_processor_register_format(&pdf_reader);
```

### Custom Classification Categories

Modify the classification categories in `document_classifier.c`:

```c
// Default categories
static const char* DOCUMENT_CATEGORIES[] = {
    "Business",
    "Technical",
    "Legal",
    "Financial",
    "Academic",
    "Marketing"
};

// Add your custom categories
static const char* DOCUMENT_CATEGORIES[] = {
    "Medical",
    "Research",
    "News",
    "Blog",
    "Email",
    "Report"
};
```

## Advanced Usage

### Batch Processing

For processing large numbers of documents efficiently:

```c
// Initialize processors
DocumentClassifier* classifier = doc_classifier_init(classifier_model_path);
DocumentSummarizer* summarizer = doc_summarizer_init(summarizer_model_path);

// Process documents in batch
const char* document_paths[] = {"doc1.txt", "doc2.txt", "doc3.txt"};
const int doc_count = 3;

// Process all documents
for (int i = 0; i < doc_count; i++) {
    // Load document content
    char document_text[MAX_DOCUMENT_SIZE];
    if (read_document(document_paths[i], document_text, sizeof(document_text))) {
        // Classify and summarize
        ClassificationResult class_result = doc_classifier_classify(classifier, document_text);
        
        char summary[1024];
        doc_summarizer_summarize(summarizer, document_text, summary, sizeof(summary));
        
        // Save results
        save_document_results(document_paths[i], &class_result, summary);
    }
}

// Clean up
doc_classifier_free(classifier);
doc_summarizer_free(summarizer);
```

### Integration with Other TinyAI Components

You can combine the document processor with other TinyAI capabilities:

```c
// Example: Using document processor with multimodal capabilities
#include "tinyai/models/multimodal.h"

// Initialize models
DocumentClassifier* classifier = doc_classifier_init(classifier_model_path);
TinyAIImageModel* img_model = tinyai_load_image_model(image_model_path);

// Process a document with images
DocumentWithImages doc;
load_document_with_images("report_with_figures.docx", &doc);

// Classify document text
ClassificationResult class_result = doc_classifier_classify(classifier, doc.text);

// Process embedded images
for (int i = 0; i < doc.image_count; i++) {
    char image_description[512];
    tinyai_describe_image(img_model, doc.images[i].data, doc.images[i].size, 
                          image_description, sizeof(image_description));
    
    // Add image descriptions to document metadata
    add_image_metadata(&doc, i, image_description);
}

// Save enhanced document
save_document_with_metadata("processed_report.json", &doc, &class_result);

// Clean up
doc_classifier_free(classifier);
tinyai_free_image_model(img_model);
```

## Performance Considerations

The document processor example balances performance with memory efficiency:

- Processing large documents in chunks reduces memory usage at the cost of slightly slower processing
- Using memory-mapped model loading provides fast startup but may have occasional delays during processing
- Classification is typically faster than summarization or keyword extraction
- Using 4-bit quantized models provides a good balance of speed and accuracy

On a typical desktop system, expect to process text at about 1-5 MB per second, depending on the operation.

## Next Steps

After exploring the document processor example, consider:

1. [Memory Optimization Guide](../guides/memory_optimization.md) - For more memory efficiency techniques
2. [Text Generation API](../api/text_generation.md) - For more advanced text processing
3. [Example: Chatbot](./chatbot.md) - For conversational AI examples
4. [Example: Media Tagging](./media_tagging.md) - For media content tagging and analysis
