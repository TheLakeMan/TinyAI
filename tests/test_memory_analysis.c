#include "memory_analysis.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Test allocation tracking
static void test_allocation_tracking()
{
    TinyAIMemoryAnalysisConfig config = {.track_allocations   = true,
                                         .track_deallocations = true,
                                         .track_peak_usage    = true,
                                         .analyze_patterns    = true,
                                         .sample_interval_ms  = 100,
                                         .analysis_window_ms  = 1000};

    TinyAIMemoryAnalysis *analysis = tinyaiCreateMemoryAnalysis(&config);
    assert(analysis != NULL);

    // Test basic allocation
    void *ptr1 = malloc(1024);
    tinyaiRecordAllocation(analysis, ptr1, 1024, __FILE__, __LINE__, __func__);

    // Test another allocation
    void *ptr2 = malloc(2048);
    tinyaiRecordAllocation(analysis, ptr2, 2048, __FILE__, __LINE__, __func__);

    // Take a sample
    tinyaiTakeMemorySample(analysis);

    // Check pattern
    TinyAIMemoryPattern pattern = tinyaiGetMemoryPattern(analysis);
    assert(pattern.total_allocations == 2);
    assert(pattern.current_usage == 3072); // 1024 + 2048
    assert(pattern.peak_usage == 3072);

    // Test deallocation
    free(ptr1);
    tinyaiRecordDeallocation(analysis, ptr1);

    // Take another sample
    tinyaiTakeMemorySample(analysis);

    // Check updated pattern
    pattern = tinyaiGetMemoryPattern(analysis);
    assert(pattern.total_freed == 1);
    assert(pattern.current_usage == 2048);

    // Cleanup
    free(ptr2);
    tinyaiRecordDeallocation(analysis, ptr2);
    tinyaiFreeMemoryAnalysis(analysis);
}

// Test memory pattern analysis
static void test_pattern_analysis()
{
    TinyAIMemoryAnalysis *analysis = tinyaiCreateMemoryAnalysis(NULL);
    assert(analysis != NULL);

    // Create some allocations
    void *ptrs[10];
    for (int i = 0; i < 10; i++) {
        ptrs[i] = malloc(1024 * (i + 1));
        tinyaiRecordAllocation(analysis, ptrs[i], 1024 * (i + 1), __FILE__, __LINE__, __func__);
    }

    // Analyze patterns
    tinyaiAnalyzeMemoryPatterns(analysis);

    // Check pattern statistics
    TinyAIMemoryPattern pattern = tinyaiGetMemoryPattern(analysis);
    assert(pattern.total_allocations == 10);
    assert(pattern.current_usage == 1024 * 55); // Sum of 1+2+...+10

    // Generate report
    tinyaiGenerateMemoryReport(analysis, "memory_report.txt");

    // Cleanup
    for (int i = 0; i < 10; i++) {
        free(ptrs[i]);
        tinyaiRecordDeallocation(analysis, ptrs[i]);
    }
    tinyaiFreeMemoryAnalysis(analysis);
}

// Test memory leak detection
static void test_leak_detection()
{
    TinyAIMemoryAnalysis *analysis = tinyaiCreateMemoryAnalysis(NULL);
    assert(analysis != NULL);

    // Create some allocations
    void *ptr1 = malloc(1024);
    void *ptr2 = malloc(2048);
    void *ptr3 = malloc(4096);

    // Record allocations
    tinyaiRecordAllocation(analysis, ptr1, 1024, __FILE__, __LINE__, __func__);
    tinyaiRecordAllocation(analysis, ptr2, 2048, __FILE__, __LINE__, __func__);
    tinyaiRecordAllocation(analysis, ptr3, 4096, __FILE__, __LINE__, __func__);

    // Free only some allocations
    free(ptr1);
    tinyaiRecordDeallocation(analysis, ptr1);

    // Get leak candidates
    TinyAIMemoryAllocation *leaks;
    size_t                  num_leaks;
    tinyaiGetMemoryLeakCandidates(analysis, &leaks, &num_leaks);
    assert(num_leaks == 2); // ptr2 and ptr3 should be leaks

    // Cleanup
    free(ptr2);
    free(ptr3);
    free(leaks);
    tinyaiFreeMemoryAnalysis(analysis);
}

// Test allocation hotspots
static void test_allocation_hotspots()
{
    TinyAIMemoryAnalysis *analysis = tinyaiCreateMemoryAnalysis(NULL);
    assert(analysis != NULL);

    // Create allocations of different sizes
    void  *ptrs[5];
    size_t sizes[] = {1024, 2048, 4096, 8192, 16384};

    for (int i = 0; i < 5; i++) {
        ptrs[i] = malloc(sizes[i]);
        tinyaiRecordAllocation(analysis, ptrs[i], sizes[i], __FILE__, __LINE__, __func__);
    }

    // Get hotspots
    TinyAIMemoryAllocation *hotspots;
    size_t                  num_hotspots;
    tinyaiGetAllocationHotspots(analysis, &hotspots, &num_hotspots);
    assert(num_hotspots == 5);

    // Check that hotspots are sorted by size
    for (size_t i = 1; i < num_hotspots; i++) {
        assert(hotspots[i].size <= hotspots[i - 1].size);
    }

    // Cleanup
    for (int i = 0; i < 5; i++) {
        free(ptrs[i]);
    }
    free(hotspots);
    tinyaiFreeMemoryAnalysis(analysis);
}

// Test memory usage trend
static void test_usage_trend()
{
    TinyAIMemoryAnalysis *analysis = tinyaiCreateMemoryAnalysis(NULL);
    assert(analysis != NULL);

    // Create increasing allocations
    for (int i = 0; i < 20; i++) {
        void *ptr = malloc(1024 * (i + 1));
        tinyaiRecordAllocation(analysis, ptr, 1024 * (i + 1), __FILE__, __LINE__, __func__);
        free(ptr);
        tinyaiRecordDeallocation(analysis, ptr);
    }

    // Check trend
    double trend = tinyaiGetMemoryUsageTrend(analysis);
    assert(trend > 0); // Should be positive as allocations are increasing

    tinyaiFreeMemoryAnalysis(analysis);
}

// Test configuration changes
static void test_configuration()
{
    TinyAIMemoryAnalysis *analysis = tinyaiCreateMemoryAnalysis(NULL);
    assert(analysis != NULL);

    // Test enabling/disabling
    tinyaiEnableMemoryAnalysis(analysis, false);
    void *ptr = malloc(1024);
    tinyaiRecordAllocation(analysis, ptr, 1024, __FILE__, __LINE__, __func__);
    assert(tinyaiGetMemoryPattern(analysis).total_allocations == 0);

    // Test configuration changes
    TinyAIMemoryAnalysisConfig new_config = {.track_allocations   = true,
                                             .track_deallocations = true,
                                             .track_peak_usage    = true,
                                             .analyze_patterns    = true,
                                             .sample_interval_ms  = 50,
                                             .analysis_window_ms  = 500};
    tinyaiSetMemoryAnalysisConfig(analysis, &new_config);
    tinyaiEnableMemoryAnalysis(analysis, true);
    tinyaiRecordAllocation(analysis, ptr, 1024, __FILE__, __LINE__, __func__);
    assert(tinyaiGetMemoryPattern(analysis).total_allocations == 1);

    // Cleanup
    free(ptr);
    tinyaiRecordDeallocation(analysis, ptr);
    tinyaiFreeMemoryAnalysis(analysis);
}

int main()
{
    printf("Testing memory analysis tools...\n");

    test_allocation_tracking();
    printf("Allocation tracking test passed\n");

    test_pattern_analysis();
    printf("Pattern analysis test passed\n");

    test_leak_detection();
    printf("Leak detection test passed\n");

    test_allocation_hotspots();
    printf("Allocation hotspots test passed\n");

    test_usage_trend();
    printf("Usage trend test passed\n");

    test_configuration();
    printf("Configuration test passed\n");

    printf("All memory analysis tests passed successfully!\n");
    return 0;
}