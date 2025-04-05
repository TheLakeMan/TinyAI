# Document Processor Example CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# Create document processor executable
add_executable(document_processor
    main.c
    document_processor.c
)

# Include directories
target_include_directories(document_processor PRIVATE
    ${CMAKE_SOURCE_DIR}
)

# Link libraries
target_link_libraries(document_processor PRIVATE
    tinyai_core
    tinyai_models
    tinyai_utils
)

# Set optimization flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(document_processor PRIVATE -O2)
elseif(MSVC)
    target_compile_options(document_processor PRIVATE /O2)
endif()

# Install the executable
install(TARGETS document_processor
    RUNTIME DESTINATION bin/examples
)
