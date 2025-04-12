cmake_minimum_required(VERSION 3.10)

# TinyAI Project
project(TinyAI VERSION 0.1.0 LANGUAGES C)

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Add compiler flags
if(MSVC)
    add_compile_options(/W4)
    # Enable SIMD instructions for MSVC
    add_compile_options(/arch:SSE2)
    option(ENABLE_AVX "Enable AVX instructions" ON)
    if(ENABLE_AVX)
        add_compile_options(/arch:AVX)
    endif()
    option(ENABLE_AVX2 "Enable AVX2 instructions" ON)
    if(ENABLE_AVX2)
        add_compile_options(/arch:AVX2)
    endif()
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    # Enable SIMD instructions for GCC/Clang
    add_compile_options(-msse2)
    option(ENABLE_AVX "Enable AVX instructions" ON)
    if(ENABLE_AVX)
        add_compile_options(-mavx)
    endif()
    option(ENABLE_AVX2 "Enable AVX2 instructions" ON)
    if(ENABLE_AVX2)
        add_compile_options(-mavx2)
    endif()
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor # Add vendor directory
)

# Collect source files
file(GLOB_RECURSE TINYAI_CORE_SOURCES "core/*.c")
file(GLOB_RECURSE TINYAI_UTILS_SOURCES "utils/*.c")
file(GLOB_RECURSE TINYAI_MODELS_SOURCES "models/*.c")
# Explicitly list interface sources for the main executable (excluding interface/main.c)
set(TINYAI_MAIN_INTERFACE_SOURCES
    interface/cli.c
    interface/web_server.c  # Web server interface
    # Add other interface files here if needed for tinyai.exe
)

# Define MCP_ENABLED to enable Model Context Protocol support
add_compile_definitions(MCP_ENABLED)

# Define the TINYAI_BUILD preprocessor symbol to exclude main() from picol.c
add_compile_definitions(TINYAI_BUILD)


# Main executable
add_executable(tinyai
    main.c # Add main.c back explicitly
    ${TINYAI_CORE_SOURCES} # Now including picol.c with TINYAI_BUILD defined
    ${TINYAI_UTILS_SOURCES}
    ${TINYAI_MODELS_SOURCES}
    ${TINYAI_MAIN_INTERFACE_SOURCES} # Use the explicit list
    vendor/mongoose/mongoose.c # Add Mongoose source
)

# Check if math library is needed (Unix/Linux)
if(UNIX)
    target_link_libraries(tinyai PRIVATE m)
endif()

# Link network library if needed (e.g., for Mongoose)
if(WIN32)
    target_link_libraries(tinyai PRIVATE ws2_32)
endif()

# Create install rules
install(TARGETS tinyai
    RUNTIME DESTINATION bin
)

# Install data directories
install(DIRECTORY data/
    DESTINATION share/tinyai/data
    PATTERN "*.md" EXCLUDE
)

# Install documentation
install(FILES
    README.md
    ARCHITECTURE.md
    IMPLEMENTATION_PLAN.md
    DESTINATION share/doc/tinyai
)

# Enable testing
enable_testing()

# Collect test source files (excluding main)
file(GLOB TINYAI_TEST_SOURCES "tests/test_*.c")

# Test executable
add_executable(tinyai_tests
    tests/test_main.c # Explicitly list main test file
    ${TINYAI_TEST_SOURCES} # Add other test files
    ${TINYAI_CORE_SOURCES} # Now including picol.c with TINYAI_BUILD defined
    ${TINYAI_UTILS_SOURCES}
    ${TINYAI_MODELS_SOURCES}
    # Add interface/cli.c explicitly as it's needed by tests but not in other lists
    interface/cli.c 
)

# Link math library for tests if needed
if(UNIX)
    target_link_libraries(tinyai_tests m)
endif()

# Add tests
add_test(NAME CoreTests COMMAND tinyai_tests core)
add_test(NAME UtilsTests COMMAND tinyai_tests utils)
add_test(NAME ModelsTests COMMAND tinyai_tests models)
add_test(NAME SimdTests COMMAND tinyai_tests simd)
add_test(NAME SparseMatrixTests COMMAND tinyai_tests sparse)
add_test(NAME MultimodalTests COMMAND tinyai_tests multimodal)

# Sparse matrix operations test executable
add_executable(sparse_matrix_test
    tests/test_sparse_ops.c
    ${TINYAI_UTILS_SOURCES}
)

# Link math library for sparse matrix test if needed
if(UNIX)
    target_link_libraries(sparse_matrix_test m)
endif()

# Image model test executable
add_executable(image_test
    models/image/image_test.c
    models/image/image_model.c
    models/image/image_utils.c
    ${TINYAI_UTILS_SOURCES}
)

# Link math library for image test if needed
if(UNIX)
    target_link_libraries(image_test m)
endif()

# Audio model test executable
add_executable(audio_test
    tests/test_audio_model.c
    models/audio/audio_model.c
    # TODO: Implement these audio module files:
    # models/audio/audio_features.c
    # models/audio/audio_utils.c
    ${TINYAI_UTILS_SOURCES}
)

# Link math library for audio test if needed
if(UNIX)
    target_link_libraries(audio_test m)
endif()

# Attention mechanism test executable
add_executable(attention_test
    tests/test_attention.c
    models/text/attention.c
    models/text/attention.h
    ${TINYAI_UTILS_SOURCES}
)

# Link math library for attention test if needed
if(UNIX)
    target_link_libraries(attention_test m)
endif()

# SIMD benchmark test executable
add_executable(simd_benchmark_test
    tests/test_simd_benchmark.c
    utils/simd_benchmark.c
    models/text/attention.c
    ${TINYAI_UTILS_SOURCES}
)

# Link math library for SIMD benchmark test if needed
if(UNIX)
    target_link_libraries(simd_benchmark_test m)
endif()

# Add a specific test for SIMD benchmarks
add_test(NAME SimdBenchmarkTest COMMAND simd_benchmark_test)

# Cache optimization test executable
add_executable(cache_opt_test
    tests/test_cache_opt.c
    utils/cache_opt.c
    ${TINYAI_UTILS_SOURCES}
)

# Link math library for cache optimization test if needed
if(UNIX)
    target_link_libraries(cache_opt_test m)
endif()

# Add a specific test for cache optimizations
add_test(NAME CacheOptimizationTest COMMAND cache_opt_test)

# Multimodal test executable
add_executable(multimodal_test
    tests/test_multimodal.c
    models/multimodal/multimodal_model.c
    models/multimodal/fusion.c
    ${TINYAI_UTILS_SOURCES}
    ${TINYAI_CORE_SOURCES}
    models/image/image_model.c
    models/image/image_utils.c
)

# Link math library for multimodal test if needed
if(UNIX)
    target_link_libraries(multimodal_test m)
endif()

# Add a specific test for multimodal capabilities
add_test(NAME MultimodalTest COMMAND multimodal_test)

# Memory-mapped model loader and forward scheduler test executable
add_executable(mmap_loader_test
    tests/test_mmap_loader.c
    utils/mmap_loader.c
    utils/forward_scheduler.c
    ${TINYAI_UTILS_SOURCES}
)

# Link math library for memory-mapped model loader test if needed
if(UNIX)
    target_link_libraries(mmap_loader_test m)
endif()

# Add a specific test for memory-mapped model loading and forward scheduling
add_test(NAME MemoryMappedModelTest COMMAND mmap_loader_test)

# Out-of-memory handling test executable
add_executable(oom_handling_test
    tests/test_oom_handling.c
    utils/memory_pool.c
    utils/mmap_loader.c
    utils/forward_scheduler.c
    ${TINYAI_UTILS_SOURCES}
)

# Link math library for OOM test if needed
if(UNIX)
    target_link_libraries(oom_handling_test m)
endif()

# Add a specific test for out-of-memory handling
add_test(NAME OutOfMemoryHandlingTest COMMAND oom_handling_test)

# Add examples
add_subdirectory(examples)

# Create a package
set(CPACK_PACKAGE_NAME "tinyai")
set(CPACK_PACKAGE_VENDOR "TinyAI Team")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Ultra-Lightweight AI Framework")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_INSTALL_DIRECTORY "TinyAI")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

# Set package generators
if(WIN32)
    set(CPACK_GENERATOR "ZIP;NSIS")
elseif(APPLE)
    set(CPACK_GENERATOR "ZIP;DragNDrop")
else()
    set(CPACK_GENERATOR "ZIP;TGZ;DEB;RPM")
endif()

include(CPack)
