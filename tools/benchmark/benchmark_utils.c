#include "benchmark_utils.h"
#include <math.h>

// Platform-specific includes
#ifdef _WIN32
#include <psapi.h>
#include <time.h>
#include <windows.h>
#else
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <mach/mach.h>
#endif
#endif

// Current version of TinyAI framework
#define TINYAI_VERSION "0.1.0"

// Initialize benchmark configuration with default values
void tinyai_init_benchmark_config(TinyAIBenchmarkConfig *config)
{
    if (!config)
        return;

    // Set default general options
    config->use_simd          = true;
    config->num_threads       = 0; // Auto-detect
    config->num_iterations    = 10;
    config->warmup_iterations = 3;
    config->measure_memory    = true;
    config->verbose           = false;

    // Set default model options
    memset(config->model_path, 0, sizeof(config->model_path));
    config->use_memory_mapping = true;
    config->cache_size_bytes   = 100 * 1024 * 1024; // 100MB default cache

    // Set default export options
    config->export_csv  = true;
    config->export_json = true;
    strcpy(config->export_path, "./benchmark_results");

    // Set default comparison options
    config->compare_frameworks = false;
    memset(config->comparison_frameworks, 0, sizeof(config->comparison_frameworks));
}

// Initialize benchmark result structure
void tinyai_init_benchmark_result(TinyAIBenchmarkResult *result)
{
    if (!result)
        return;

    // Initialize timing metrics
    result->total_time_ms         = 0.0;
    result->avg_inference_time_ms = 0.0;
    result->std_dev_time_ms       = 0.0;
    result->min_time_ms           = 0.0;
    result->max_time_ms           = 0.0;

    // Initialize memory metrics
    result->peak_memory_bytes = 0;
    result->avg_memory_bytes  = 0;
    result->model_size_bytes  = 0;

    // Initialize hardware utilization
    result->cpu_utilization = 0.0;
    result->threads_used    = 0;
    result->simd_used       = false;
    memset(result->simd_type, 0, sizeof(result->simd_type));

    // Initialize performance metrics
    result->samples_processed  = 0;
    result->samples_per_second = 0.0;

    // Initialize framework identification
    strcpy(result->framework_name, "TinyAI");
    strcpy(result->framework_version, TINYAI_VERSION);
    memset(result->model_name, 0, sizeof(result->model_name));
    memset(result->device_name, 0, sizeof(result->device_name));

    // Initialize modality-specific metrics to zero
    memset(&result->modality_metrics, 0, sizeof(result->modality_metrics));
}

// Start benchmark timing
void tinyai_benchmark_start_timer(struct timespec *start_time)
{
    if (!start_time)
        return;
#ifdef _WIN32
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    start_time->tv_sec  = li.QuadPart / 1000000000;
    start_time->tv_nsec = li.QuadPart % 1000000000;
#else
    clock_gettime(CLOCK_MONOTONIC, start_time);
#endif
}

// Stop benchmark timing and calculate elapsed time in ms
double tinyai_benchmark_stop_timer(struct timespec *start_time)
{
    if (!start_time)
        return 0.0;

    struct timespec end_time;
#ifdef _WIN32
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    end_time.tv_sec  = li.QuadPart / 1000000000;
    end_time.tv_nsec = li.QuadPart % 1000000000;
#else
    clock_gettime(CLOCK_MONOTONIC, &end_time);
#endif

    double elapsed_sec  = (end_time.tv_sec - start_time->tv_sec);
    double elapsed_nsec = (end_time.tv_nsec - start_time->tv_nsec) / 1000000000.0;
    return (elapsed_sec + elapsed_nsec) * 1000.0; // Convert to milliseconds
}

// Export benchmark results to CSV
bool tinyai_export_benchmark_csv(const TinyAIBenchmarkResult *result, const char *filepath)
{
    if (!result || !filepath)
        return false;

    FILE *file = fopen(filepath, "w");
    if (!file)
        return false;

    // Write CSV header
    fprintf(file, "Metric,Value\n");

    // Write timing metrics
    fprintf(file, "Total Time (ms),%.2f\n", result->total_time_ms);
    fprintf(file, "Average Inference Time (ms),%.2f\n", result->avg_inference_time_ms);
    fprintf(file, "Standard Deviation Time (ms),%.2f\n", result->std_dev_time_ms);
    fprintf(file, "Minimum Time (ms),%.2f\n", result->min_time_ms);
    fprintf(file, "Maximum Time (ms),%.2f\n", result->max_time_ms);

    // Write memory metrics
    fprintf(file, "Peak Memory (bytes),%zu\n", result->peak_memory_bytes);
    fprintf(file, "Average Memory (bytes),%zu\n", result->avg_memory_bytes);
    fprintf(file, "Model Size (bytes),%zu\n", result->model_size_bytes);

    // Write hardware utilization
    fprintf(file, "CPU Utilization (%%),%%.2f\n", result->cpu_utilization);
    fprintf(file, "Threads Used,%d\n", result->threads_used);
    fprintf(file, "SIMD Used,%s\n", result->simd_used ? "true" : "false");
    fprintf(file, "SIMD Type,%s\n", result->simd_type);

    // Write performance metrics
    fprintf(file, "Samples Processed,%d\n", result->samples_processed);
    fprintf(file, "Samples Per Second,%.2f\n", result->samples_per_second);

    // Write framework identification
    fprintf(file, "Framework,%s\n", result->framework_name);
    fprintf(file, "Framework Version,%s\n", result->framework_version);
    fprintf(file, "Model,%s\n", result->model_name);
    fprintf(file, "Device,%s\n", result->device_name);

    // Write modality-specific metrics based on framework name
    if (strstr(result->model_name, "text") || strstr(result->model_name, "Text")) {
        fprintf(file, "Tokens Per Second,%.2f\n", result->modality_metrics.text.tokens_per_second);
        fprintf(file, "Context Length,%d\n", result->modality_metrics.text.context_length);
    }
    else if (strstr(result->model_name, "image") || strstr(result->model_name, "Image")) {
        fprintf(file, "Image Width,%d\n", result->modality_metrics.image.image_width);
        fprintf(file, "Image Height,%d\n", result->modality_metrics.image.image_height);
        fprintf(file, "FPS,%.2f\n", result->modality_metrics.image.fps);
    }
    else if (strstr(result->model_name, "audio") || strstr(result->model_name, "Audio")) {
        fprintf(file, "Sample Rate,%d\n", result->modality_metrics.audio.sample_rate);
        fprintf(file, "Audio Length (s),%.2f\n", result->modality_metrics.audio.audio_length_sec);
        fprintf(file, "Real-Time Factor,%.2f\n", result->modality_metrics.audio.real_time_factor);
    }
    else if (strstr(result->model_name, "multimodal") || strstr(result->model_name, "Multimodal")) {
        fprintf(file, "Number of Modalities,%d\n",
                result->modality_metrics.multimodal.num_modalities);
        fprintf(file, "Fusion Time (ms),%.2f\n",
                result->modality_metrics.multimodal.fusion_time_ms);
    }

    fclose(file);
    return true;
}

// Export benchmark results to JSON
bool tinyai_export_benchmark_json(const TinyAIBenchmarkResult *result, const char *filepath)
{
    if (!result || !filepath)
        return false;

    FILE *file = fopen(filepath, "w");
    if (!file)
        return false;

    // Write JSON header
    fprintf(file, "{\n");

    // Write timing metrics
    fprintf(file, "  \"timing\": {\n");
    fprintf(file, "    \"total_time_ms\": %.2f,\n", result->total_time_ms);
    fprintf(file, "    \"avg_inference_time_ms\": %.2f,\n", result->avg_inference_time_ms);
    fprintf(file, "    \"std_dev_time_ms\": %.2f,\n", result->std_dev_time_ms);
    fprintf(file, "    \"min_time_ms\": %.2f,\n", result->min_time_ms);
    fprintf(file, "    \"max_time_ms\": %.2f\n", result->max_time_ms);
    fprintf(file, "  },\n");

    // Write memory metrics
    fprintf(file, "  \"memory\": {\n");
    fprintf(file, "    \"peak_memory_bytes\": %zu,\n", result->peak_memory_bytes);
    fprintf(file, "    \"avg_memory_bytes\": %zu,\n", result->avg_memory_bytes);
    fprintf(file, "    \"model_size_bytes\": %zu\n", result->model_size_bytes);
    fprintf(file, "  },\n");

    // Write hardware utilization
    fprintf(file, "  \"hardware\": {\n");
    fprintf(file, "    \"cpu_utilization\": %.2f,\n", result->cpu_utilization);
    fprintf(file, "    \"threads_used\": %d,\n", result->threads_used);
    fprintf(file, "    \"simd_used\": %s,\n", result->simd_used ? "true" : "false");
    fprintf(file, "    \"simd_type\": \"%s\"\n", result->simd_type);
    fprintf(file, "  },\n");

    // Write performance metrics
    fprintf(file, "  \"performance\": {\n");
    fprintf(file, "    \"samples_processed\": %d,\n", result->samples_processed);
    fprintf(file, "    \"samples_per_second\": %.2f\n", result->samples_per_second);
    fprintf(file, "  },\n");

    // Write framework identification
    fprintf(file, "  \"framework\": {\n");
    fprintf(file, "    \"name\": \"%s\",\n", result->framework_name);
    fprintf(file, "    \"version\": \"%s\",\n", result->framework_version);
    fprintf(file, "    \"model\": \"%s\",\n", result->model_name);
    fprintf(file, "    \"device\": \"%s\"\n", result->device_name);
    fprintf(file, "  }");

    // Write modality-specific metrics based on framework name
    if (strstr(result->model_name, "text") || strstr(result->model_name, "Text")) {
        fprintf(file, ",\n  \"text_metrics\": {\n");
        fprintf(file, "    \"tokens_per_second\": %.2f,\n",
                result->modality_metrics.text.tokens_per_second);
        fprintf(file, "    \"context_length\": %d\n", result->modality_metrics.text.context_length);
        fprintf(file, "  }\n");
    }
    else if (strstr(result->model_name, "image") || strstr(result->model_name, "Image")) {
        fprintf(file, ",\n  \"image_metrics\": {\n");
        fprintf(file, "    \"image_width\": %d,\n", result->modality_metrics.image.image_width);
        fprintf(file, "    \"image_height\": %d,\n", result->modality_metrics.image.image_height);
        fprintf(file, "    \"fps\": %.2f\n", result->modality_metrics.image.fps);
        fprintf(file, "  }\n");
    }
    else if (strstr(result->model_name, "audio") || strstr(result->model_name, "Audio")) {
        fprintf(file, ",\n  \"audio_metrics\": {\n");
        fprintf(file, "    \"sample_rate\": %d,\n", result->modality_metrics.audio.sample_rate);
        fprintf(file, "    \"audio_length_sec\": %.2f,\n",
                result->modality_metrics.audio.audio_length_sec);
        fprintf(file, "    \"real_time_factor\": %.2f\n",
                result->modality_metrics.audio.real_time_factor);
        fprintf(file, "  }\n");
    }
    else if (strstr(result->model_name, "multimodal") || strstr(result->model_name, "Multimodal")) {
        fprintf(file, ",\n  \"multimodal_metrics\": {\n");
        fprintf(file, "    \"num_modalities\": %d,\n",
                result->modality_metrics.multimodal.num_modalities);
        fprintf(file, "    \"fusion_time_ms\": %.2f\n",
                result->modality_metrics.multimodal.fusion_time_ms);
        fprintf(file, "  }\n");
    }
    else {
        fprintf(file, "\n");
    }

    // Close JSON object
    fprintf(file, "}\n");

    fclose(file);
    return true;
}

// Print benchmark results to console
void tinyai_print_benchmark_results(const TinyAIBenchmarkResult *result)
{
    if (!result)
        return;

    printf("\n===== BENCHMARK RESULTS =====\n");
    printf("Model: %s\n", result->model_name);
    printf("Framework: %s v%s\n", result->framework_name, result->framework_version);
    printf("Device: %s\n", result->device_name);
    printf("\n");

    printf("-- Timing Metrics --\n");
    printf("Total Time: %.2f ms\n", result->total_time_ms);
    printf("Avg Inference Time: %.2f ms\n", result->avg_inference_time_ms);
    printf("Min/Max Time: %.2f / %.2f ms\n", result->min_time_ms, result->max_time_ms);
    printf("Standard Deviation: %.2f ms\n", result->std_dev_time_ms);
    printf("\n");

    printf("-- Memory Metrics --\n");
    printf("Peak Memory: %.2f MB\n", result->peak_memory_bytes / (1024.0 * 1024.0));
    printf("Avg Memory: %.2f MB\n", result->avg_memory_bytes / (1024.0 * 1024.0));
    printf("Model Size: %.2f MB\n", result->model_size_bytes / (1024.0 * 1024.0));
    printf("\n");

    printf("-- Hardware Utilization --\n");
    printf("CPU Utilization: %.2f%%\n", result->cpu_utilization);
    printf("Threads Used: %d\n", result->threads_used);
    printf("SIMD: %s (%s)\n", result->simd_used ? "Enabled" : "Disabled", result->simd_type);
    printf("\n");

    printf("-- Performance Metrics --\n");
    printf("Samples Processed: %d\n", result->samples_processed);
    printf("Samples Per Second: %.2f\n", result->samples_per_second);

    // Print modality-specific metrics
    if (strstr(result->model_name, "text") || strstr(result->model_name, "Text")) {
        printf("\n-- Text-Specific Metrics --\n");
        printf("Tokens Per Second: %.2f\n", result->modality_metrics.text.tokens_per_second);
        printf("Context Length: %d\n", result->modality_metrics.text.context_length);
    }
    else if (strstr(result->model_name, "image") || strstr(result->model_name, "Image")) {
        printf("\n-- Image-Specific Metrics --\n");
        printf("Resolution: %dx%d\n", result->modality_metrics.image.image_width,
               result->modality_metrics.image.image_height);
        printf("FPS: %.2f\n", result->modality_metrics.image.fps);
    }
    else if (strstr(result->model_name, "audio") || strstr(result->model_name, "Audio")) {
        printf("\n-- Audio-Specific Metrics --\n");
        printf("Sample Rate: %d Hz\n", result->modality_metrics.audio.sample_rate);
        printf("Audio Length: %.2f seconds\n", result->modality_metrics.audio.audio_length_sec);
        printf("Real-Time Factor: %.2f%s\n", result->modality_metrics.audio.real_time_factor,
               result->modality_metrics.audio.real_time_factor < 1.0 ? " (faster than real-time)"
                                                                     : "");
    }
    else if (strstr(result->model_name, "multimodal") || strstr(result->model_name, "Multimodal")) {
        printf("\n-- Multimodal-Specific Metrics --\n");
        printf("Number of Modalities: %d\n", result->modality_metrics.multimodal.num_modalities);
        printf("Fusion Time: %.2f ms\n", result->modality_metrics.multimodal.fusion_time_ms);
    }

    printf("\n=============================\n\n");
}

// Compare benchmark results with other frameworks
void tinyai_compare_benchmark_results(const TinyAIBenchmarkResult *tinyai_result,
                                      const TinyAIBenchmarkResult *other_results,
                                      int                          num_other_results)
{
    if (!tinyai_result || !other_results || num_other_results <= 0)
        return;

    printf("\n===== BENCHMARK COMPARISON =====\n");
    printf("Model: %s\n\n", tinyai_result->model_name);

    // Print header
    printf("%-15s %-15s %-15s %-15s %-15s %-15s\n", "Framework", "Avg Time (ms)", "Memory (MB)",
           "Samples/sec", "Speedup", "Size Ratio");
    printf("-------------------------------------------------------------------------------\n");

    // Print TinyAI results (baseline)
    printf("%-15s %-15.2f %-15.2f %-15.2f %-15s %-15s\n", tinyai_result->framework_name,
           tinyai_result->avg_inference_time_ms,
           tinyai_result->peak_memory_bytes / (1024.0 * 1024.0), tinyai_result->samples_per_second,
           "1.00x", "1.00x");

    // Print other frameworks
    for (int i = 0; i < num_other_results; i++) {
        double time_speedup =
            tinyai_result->avg_inference_time_ms / other_results[i].avg_inference_time_ms;
        double memory_ratio =
            tinyai_result->peak_memory_bytes / (double)other_results[i].peak_memory_bytes;

        printf("%-15s %-15.2f %-15.2f %-15.2f %-15.2fx %-15.2fx\n", other_results[i].framework_name,
               other_results[i].avg_inference_time_ms,
               other_results[i].peak_memory_bytes / (1024.0 * 1024.0),
               other_results[i].samples_per_second, time_speedup, memory_ratio);
    }

    printf("\n=================================\n\n");
}

// Platform-specific memory measurement functions
size_t tinyai_measure_current_memory_usage(void)
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize;
    }
    return 0;
#elif defined(__APPLE__)
    struct task_basic_info t_info;
    mach_msg_type_number_t t_info_count = TASK_BASIC_INFO_COUNT;

    if (task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&t_info, &t_info_count) !=
        KERN_SUCCESS) {
        return 0;
    }
    return t_info.resident_size;
#else
    // Linux
    FILE *file = fopen("/proc/self/status", "r");
    if (file == NULL) {
        return 0;
    }

    size_t result = 0;
    char   line[128];

    while (fgets(line, sizeof(line), file) != NULL) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            // Extract the memory usage value in kB
            long kb_usage = 0;
            if (sscanf(line + 6, "%ld", &kb_usage) == 1) {
                result = kb_usage * 1024; // Convert to bytes
            }
            break;
        }
    }

    fclose(file);
    return result;
#endif
}

size_t tinyai_measure_peak_memory_usage(void)
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.PeakWorkingSetSize;
    }
    return 0;
#elif defined(__APPLE__)
    // macOS doesn't provide a direct way to get peak memory,
    // so we'll have to track it manually in the benchmarking code
    return tinyai_measure_current_memory_usage();
#else
    // Linux
    FILE *file = fopen("/proc/self/status", "r");
    if (file == NULL) {
        return 0;
    }

    size_t result = 0;
    char   line[128];

    while (fgets(line, sizeof(line), file) != NULL) {
        if (strncmp(line, "VmHWM:", 6) == 0) {
            // Extract the peak memory usage value in kB
            long kb_usage = 0;
            if (sscanf(line + 6, "%ld", &kb_usage) == 1) {
                result = kb_usage * 1024; // Convert to bytes
            }
            break;
        }
    }

    fclose(file);
    return result;
#endif
}

// Function to detect and return the available SIMD capabilities
const char *tinyai_detect_simd_capabilities(void)
{
#ifdef __AVX2__
    return "AVX2";
#elif defined(__AVX__)
    return "AVX";
#elif defined(__SSE4_2__)
    return "SSE4.2";
#elif defined(__SSE4_1__)
    return "SSE4.1";
#elif defined(__SSE3__)
    return "SSE3";
#elif defined(__SSE2__)
    return "SSE2";
#else
    return "None";
#endif
}

// Function to determine the optimal number of threads
int tinyai_determine_optimal_threads(void)
{
#ifdef _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
#else
    return sysconf(_SC_NPROCESSORS_ONLN);
#endif
}

// Utility function to create a timestamped filename
void tinyai_create_timestamped_filename(char *buffer, size_t buffer_size, const char *prefix,
                                        const char *extension)
{
    if (!buffer || buffer_size == 0 || !prefix || !extension)
        return;

    time_t     now     = time(NULL);
    struct tm *tm_info = localtime(&now);

    char timestamp[20];
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", tm_info);

    snprintf(buffer, buffer_size, "%s_%s.%s", prefix, timestamp, extension);
}
