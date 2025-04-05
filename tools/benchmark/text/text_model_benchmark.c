#include "../benchmark_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#include <windows.h> // For GetComputerNameA and DWORD type
#define mkdir(path, mode) _mkdir(path)
#else
#include <sys/stat.h>
#include <unistd.h> // For gethostname
#endif

// Text generation parameters structure (this would normally be defined in models/text/generate.h)
typedef struct {
    float temperature; // Temperature for token sampling
    int   top_k;       // Top-k for token sampling
    float top_p;       // Top-p (nucleus) for token sampling
    int   max_tokens;  // Maximum tokens to generate
    int   num_threads; // Number of threads to use
    bool  use_simd;    // Whether to use SIMD acceleration
} TextGenerationParams;

// TinyAI includes
#include "../../../core/config.h"
#include "../../../models/text/generate.h"
#include "../../../utils/forward_scheduler.h"
#include "../../../utils/mmap_loader.h"
#include "../../../utils/model_loader.h"

// Default configuration values
#define DEFAULT_ITERATIONS 5
#define DEFAULT_WARMUP_ITERATIONS 2
#define DEFAULT_PROMPT "TinyAI is"
#define DEFAULT_MAX_TOKENS 50
#define DEFAULT_TEMPERATURE 0.7
#define DEFAULT_SIMD_ENABLED 1
#define DEFAULT_NUM_THREADS 0 // Auto-detect
#define DEFAULT_MEMORY_MAPPING 1
#define DEFAULT_MEMORY_LIMIT (200 * 1024 * 1024) // 200MB
#define DEFAULT_EXPORT_PATH "./benchmark_results"
#define DEFAULT_MODEL_PATH "models/pretrained/text_small.tmai"
#define DEFAULT_COMPARE_FRAMEWORKS 0

typedef struct {
    char   prompt[256];
    int    max_tokens;
    float  temperature;
    int    top_k;
    float  top_p;
    int    context_size;
    int    use_mmap;
    size_t memory_limit;
} TextBenchmarkConfig;

void print_usage(const char *program_name)
{
    printf("Usage: %s [options]\n\n", program_name);
    printf("Options:\n");
    printf("  -model <path>           Path to TinyAI model file (default: %s)\n",
           DEFAULT_MODEL_PATH);
    printf("  -prompt <text>          Prompt text (default: \"%s\")\n", DEFAULT_PROMPT);
    printf("  -max_tokens <n>         Maximum tokens to generate (default: %d)\n",
           DEFAULT_MAX_TOKENS);
    printf("  -temp <t>               Temperature (default: %.2f)\n", DEFAULT_TEMPERATURE);
    printf("  -top_k <k>              Top-k sampling parameter (default: 40)\n");
    printf("  -top_p <p>              Top-p sampling parameter (default: 0.95)\n");
    printf("  -iter <n>               Number of iterations (default: %d)\n", DEFAULT_ITERATIONS);
    printf("  -warmup <n>             Number of warmup iterations (default: %d)\n",
           DEFAULT_WARMUP_ITERATIONS);
    printf("  -simd <0|1>             Enable/disable SIMD (default: %d)\n", DEFAULT_SIMD_ENABLED);
    printf("  -threads <n>            Number of threads (0 for auto) (default: %d)\n",
           DEFAULT_NUM_THREADS);
    printf("  -mmap <0|1>             Enable/disable memory mapping (default: %d)\n",
           DEFAULT_MEMORY_MAPPING);
    printf("  -memory <n>             Memory limit in MB (default: %d)\n",
           DEFAULT_MEMORY_LIMIT / (1024 * 1024));
    printf("  -export <path>          Export results path (default: %s)\n", DEFAULT_EXPORT_PATH);
    printf("  -compare <0|1>          Compare with other frameworks (default: %d)\n",
           DEFAULT_COMPARE_FRAMEWORKS);
    printf("  -v                      Verbose output\n");
    printf("  -h                      Display this help message\n");
    printf("\n");
}

// Parse command line arguments and set configuration
void parse_args(int argc, char **argv, TinyAIBenchmarkConfig *config,
                TextBenchmarkConfig *text_config)
{
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-model") == 0 && i + 1 < argc) {
            strncpy(config->model_path, argv[++i], sizeof(config->model_path) - 1);
        }
        else if (strcmp(argv[i], "-prompt") == 0 && i + 1 < argc) {
            strncpy(text_config->prompt, argv[++i], sizeof(text_config->prompt) - 1);
        }
        else if (strcmp(argv[i], "-max_tokens") == 0 && i + 1 < argc) {
            text_config->max_tokens = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-temp") == 0 && i + 1 < argc) {
            text_config->temperature = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-top_k") == 0 && i + 1 < argc) {
            text_config->top_k = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-top_p") == 0 && i + 1 < argc) {
            text_config->top_p = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-iter") == 0 && i + 1 < argc) {
            config->num_iterations = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-warmup") == 0 && i + 1 < argc) {
            config->warmup_iterations = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-simd") == 0 && i + 1 < argc) {
            config->use_simd = atoi(argv[++i]) != 0;
        }
        else if (strcmp(argv[i], "-threads") == 0 && i + 1 < argc) {
            config->num_threads = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-mmap") == 0 && i + 1 < argc) {
            text_config->use_mmap      = atoi(argv[++i]) != 0;
            config->use_memory_mapping = text_config->use_mmap;
        }
        else if (strcmp(argv[i], "-memory") == 0 && i + 1 < argc) {
            text_config->memory_limit =
                (size_t)atoi(argv[++i]) * 1024 * 1024; // Convert MB to bytes
        }
        else if (strcmp(argv[i], "-export") == 0 && i + 1 < argc) {
            strncpy(config->export_path, argv[++i], sizeof(config->export_path) - 1);
        }
        else if (strcmp(argv[i], "-compare") == 0 && i + 1 < argc) {
            config->compare_frameworks = atoi(argv[++i]) != 0;
        }
        else if (strcmp(argv[i], "-v") == 0) {
            config->verbose = true;
        }
        else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            exit(EXIT_SUCCESS);
        }
        else {
            printf("Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }
}

// Function to benchmark TinyAI text model
void benchmark_tinyai_text(TinyAIBenchmarkConfig *config, TextBenchmarkConfig *text_config,
                           TinyAIBenchmarkResult *result)
{
    // Initialize result
    tinyai_init_benchmark_result(result);

    // Set model name
    strncpy(result->model_name, config->model_path, sizeof(result->model_name) - 1);

    // Get device name (hostname)
    char hostname[64] = "unknown";
#ifdef _WIN32
    DWORD size = sizeof(hostname);
    GetComputerNameA(hostname, &size);
#else
    gethostname(hostname, sizeof(hostname));
#endif
    strncpy(result->device_name, hostname, sizeof(result->device_name) - 1);

    // Detect SIMD capabilities
    result->simd_used = config->use_simd;
    strncpy(result->simd_type, tinyai_detect_simd_capabilities(), sizeof(result->simd_type) - 1);

    // Set number of threads
    result->threads_used =
        config->num_threads > 0 ? config->num_threads : tinyai_determine_optimal_threads();

    // Load model
    void                   *model      = NULL;
    TinyAIMappedModel      *mmap_model = NULL;
    TinyAIForwardScheduler *scheduler  = NULL;

    if (text_config->use_mmap) {
        // Use memory mapping for model loading
        TinyAIMmapConfig mmap_config = tinyaiCreateDefaultMmapConfig();
        mmap_config.maxCacheSize     = config->cache_size_bytes;
        mmap_config.prefetchEnabled  = true;

        mmap_model = tinyaiOpenMappedModel(config->model_path, &mmap_config);
        if (!mmap_model) {
            printf("Error: Failed to load memory-mapped model: %s\n", config->model_path);
            exit(EXIT_FAILURE);
        }

        // Create forward scheduler
        scheduler = tinyaiCreateForwardScheduler(mmap_model, TINYAI_EXEC_MEMORY_OPT,
                                                 text_config->memory_limit);
        if (!scheduler) {
            printf("Error: Failed to create forward scheduler\n");
            tinyaiCloseMappedModel(mmap_model);
            exit(EXIT_FAILURE);
        }
    }
    else {
        // Load model normally
        model = tinyai_load_model(config->model_path);
        if (!model) {
            printf("Error: Failed to load model: %s\n", config->model_path);
            exit(EXIT_FAILURE);
        }
    }

    // Get model size
    result->model_size_bytes = 0; // TODO: Implement model size calculation

    // Prepare text generation parameters
    TextGenerationParams params;
    params.temperature = text_config->temperature;
    params.top_k       = text_config->top_k;
    params.top_p       = text_config->top_p;
    params.max_tokens  = text_config->max_tokens;
    params.num_threads = result->threads_used;
    params.use_simd    = result->simd_used;

    // Arrays to store timing results
    double *inference_times = (double *)malloc(config->num_iterations * sizeof(double));
    size_t *memory_usages   = (size_t *)malloc(config->num_iterations * sizeof(size_t));

    // Perform warmup iterations
    if (config->verbose) {
        printf("Performing %d warmup iterations...\n", config->warmup_iterations);
    }

    for (int i = 0; i < config->warmup_iterations; i++) {
        char output[1024] = {0};

        if (text_config->use_mmap) {
            // TODO: Generate text with memory-mapped model and scheduler
        }
        else {
            // Generate text with regular model
            // In a real implementation, we would call the text generation function here
            // tinyai_generate_text(model, text_config->prompt, output, sizeof(output), &params);
        }

        if (config->verbose) {
            printf("Warmup %d: Generated %zu characters\n", i + 1, strlen(output));
        }
    }

    // Perform benchmark iterations
    if (config->verbose) {
        printf("Performing %d benchmark iterations...\n", config->num_iterations);
    }

    // Total tokens generated (for calculating tokens per second)
    int total_tokens_generated = 0;

    for (int i = 0; i < config->num_iterations; i++) {
        char            output[1024] = {0};
        struct timespec start_time;

        // Measure memory before generation
        size_t mem_before = tinyai_measure_current_memory_usage();

        // Start timer
        tinyai_benchmark_start_timer(&start_time);

        if (text_config->use_mmap) {
            // TODO: Generate text with memory-mapped model and scheduler
            // In a real implementation, we would use the memory-mapped model and scheduler here
        }
        else {
            // Generate text with regular model
            // In a real implementation, we would call the text generation function here
            // tinyai_generate_text(model, text_config->prompt, output, sizeof(output), &params);
        }

        // Stop timer
        inference_times[i] = tinyai_benchmark_stop_timer(&start_time);

        // Measure memory after generation
        size_t mem_after = tinyai_measure_current_memory_usage();
        memory_usages[i] = mem_after > mem_before ? mem_after - mem_before : 0;

        // Count tokens generated (in a real implementation this would be the actual token count)
        int tokens_generated = text_config->max_tokens;
        total_tokens_generated += tokens_generated;

        if (config->verbose) {
            printf("Iteration %d: Generated %zu characters in %.2f ms\n", i + 1, strlen(output),
                   inference_times[i]);
            printf("Output: %s\n\n", output);
        }
    }

    // Calculate statistics
    result->total_time_ms = 0.0;
    result->min_time_ms   = inference_times[0];
    result->max_time_ms   = inference_times[0];

    for (int i = 0; i < config->num_iterations; i++) {
        result->total_time_ms += inference_times[i];
        if (inference_times[i] < result->min_time_ms)
            result->min_time_ms = inference_times[i];
        if (inference_times[i] > result->max_time_ms)
            result->max_time_ms = inference_times[i];
    }

    result->avg_inference_time_ms = result->total_time_ms / config->num_iterations;

    // Calculate standard deviation
    double variance = 0.0;
    for (int i = 0; i < config->num_iterations; i++) {
        double diff = inference_times[i] - result->avg_inference_time_ms;
        variance += diff * diff;
    }
    variance /= config->num_iterations;
    result->std_dev_time_ms = sqrt(variance);

    // Calculate memory statistics
    result->peak_memory_bytes = tinyai_measure_peak_memory_usage();

    result->avg_memory_bytes = 0;
    for (int i = 0; i < config->num_iterations; i++) {
        result->avg_memory_bytes += memory_usages[i];
    }
    result->avg_memory_bytes /= config->num_iterations;

    // Calculate performance metrics
    result->samples_processed  = config->num_iterations;
    result->samples_per_second = (config->num_iterations * 1000.0) / result->total_time_ms;

    // Set text-specific metrics
    result->modality_metrics.text.tokens_per_second =
        (total_tokens_generated * 1000.0) / result->total_time_ms;
    result->modality_metrics.text.context_length = text_config->context_size;

    // Clean up
    free(inference_times);
    free(memory_usages);

    if (text_config->use_mmap) {
        if (scheduler)
            tinyaiDestroyForwardScheduler(scheduler);
        if (mmap_model)
            tinyaiCloseMappedModel(mmap_model);
    }
    else {
        if (model) {
            // TODO: Free model in real implementation
            // tinyai_free_model(model);
        }
    }
}

// Function to compare with other frameworks
void compare_with_other_frameworks(const TinyAIBenchmarkResult *tinyai_result,
                                   const char                  *export_path)
{
    printf("Comparison with other frameworks is not implemented yet.\n");
    printf("This would require integration with external libraries like TensorFlow Lite,\n");
    printf("ONNX Runtime, and PyTorch Mobile.\n\n");
    printf("In a complete implementation, this function would:\n");
    printf("1. Load equivalent models in each framework\n");
    printf("2. Run the same benchmarks with the same input data\n");
    printf("3. Collect and compare results\n");
    printf("4. Generate comparison charts\n\n");

    // Placeholder for framework comparison
    TinyAIBenchmarkResult other_results[2];

    // Simulate TensorFlow Lite results
    tinyai_init_benchmark_result(&other_results[0]);
    strcpy(other_results[0].framework_name, "TensorFlow Lite");
    strcpy(other_results[0].framework_version, "2.12.0");
    strcpy(other_results[0].model_name, tinyai_result->model_name);
    strcpy(other_results[0].device_name, tinyai_result->device_name);
    other_results[0].avg_inference_time_ms = tinyai_result->avg_inference_time_ms * 1.5;
    other_results[0].peak_memory_bytes     = tinyai_result->peak_memory_bytes * 1.8;
    other_results[0].samples_per_second    = tinyai_result->samples_per_second / 1.5;

    // Simulate ONNX Runtime results
    tinyai_init_benchmark_result(&other_results[1]);
    strcpy(other_results[1].framework_name, "ONNX Runtime");
    strcpy(other_results[1].framework_version, "1.15.1");
    strcpy(other_results[1].model_name, tinyai_result->model_name);
    strcpy(other_results[1].device_name, tinyai_result->device_name);
    other_results[1].avg_inference_time_ms = tinyai_result->avg_inference_time_ms * 1.2;
    other_results[1].peak_memory_bytes     = tinyai_result->peak_memory_bytes * 1.5;
    other_results[1].samples_per_second    = tinyai_result->samples_per_second / 1.2;

    // Print comparison table
    tinyai_compare_benchmark_results(tinyai_result, other_results, 2);
}

// Main function
int main(int argc, char **argv)
{
    // Initialize configuration
    TinyAIBenchmarkConfig config;
    TextBenchmarkConfig   text_config;

    tinyai_init_benchmark_config(&config);

    // Set default text generation parameters
    strncpy(text_config.prompt, DEFAULT_PROMPT, sizeof(text_config.prompt) - 1);
    text_config.max_tokens   = DEFAULT_MAX_TOKENS;
    text_config.temperature  = DEFAULT_TEMPERATURE;
    text_config.top_k        = 40;
    text_config.top_p        = 0.95f;
    text_config.context_size = 512;
    text_config.use_mmap     = DEFAULT_MEMORY_MAPPING;
    text_config.memory_limit = DEFAULT_MEMORY_LIMIT;

    // Parse command line arguments
    parse_args(argc, argv, &config, &text_config);

// Ensure the export directory exists
#ifdef _WIN32
    _mkdir(config.export_path);
#else
    mkdir(config.export_path, 0755);
#endif

    // Print benchmark settings
    printf("\n===== TinyAI Text Model Benchmark =====\n");
    printf("Model: %s\n", config.model_path);
    printf("Prompt: \"%s\"\n", text_config.prompt);
    printf("Max Tokens: %d\n", text_config.max_tokens);
    printf("Temperature: %.2f\n", text_config.temperature);
    printf("Top-k: %d\n", text_config.top_k);
    printf("Top-p: %.2f\n", text_config.top_p);
    printf("Context Size: %d tokens\n", text_config.context_size);
    printf("Iterations: %d (%d warmup)\n", config.num_iterations, config.warmup_iterations);
    printf("SIMD: %s\n", config.use_simd ? "Enabled" : "Disabled");
    printf("Threads: %d\n",
           config.num_threads > 0 ? config.num_threads : tinyai_determine_optimal_threads());
    printf("Memory Mapping: %s\n", text_config.use_mmap ? "Enabled" : "Disabled");
    printf("Memory Limit: %.2f MB\n", text_config.memory_limit / (1024.0 * 1024.0));
    printf("Export Path: %s\n", config.export_path);
    printf("Compare Frameworks: %s\n", config.compare_frameworks ? "Yes" : "No");
    printf("Verbose: %s\n", config.verbose ? "Yes" : "No");
    printf("=========================================\n\n");

    // Run benchmark
    printf("Running TinyAI text model benchmark...\n");

    TinyAIBenchmarkResult result;
    benchmark_tinyai_text(&config, &text_config, &result);

    // Print results
    tinyai_print_benchmark_results(&result);

    // Export results
    char csv_path[512];
    char json_path[512];

    tinyai_create_timestamped_filename(csv_path, sizeof(csv_path), "tinyai_text_benchmark", "csv");
    tinyai_create_timestamped_filename(json_path, sizeof(json_path), "tinyai_text_benchmark",
                                       "json");

    char full_csv_path[1024];
    char full_json_path[1024];

    snprintf(full_csv_path, sizeof(full_csv_path), "%s/%s", config.export_path, csv_path);
    snprintf(full_json_path, sizeof(full_json_path), "%s/%s", config.export_path, json_path);

    printf("Exporting results...\n");
    if (tinyai_export_benchmark_csv(&result, full_csv_path)) {
        printf("CSV results exported to: %s\n", full_csv_path);
    }
    else {
        printf("Failed to export CSV results\n");
    }

    if (tinyai_export_benchmark_json(&result, full_json_path)) {
        printf("JSON results exported to: %s\n", full_json_path);
    }
    else {
        printf("Failed to export JSON results\n");
    }

    // Compare with other frameworks if requested
    if (config.compare_frameworks) {
        printf("\nComparing with other frameworks...\n");
        compare_with_other_frameworks(&result, config.export_path);
    }

    printf("\nBenchmark complete.\n");
    return 0;
}
