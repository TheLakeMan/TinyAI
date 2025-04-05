# Visual Question Answering Example CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# Create visual QA executable
add_executable(visual_qa
    main.c
    visual_qa.c
)

# Include directories
target_include_directories(visual_qa PRIVATE
    ${CMAKE_SOURCE_DIR}
)

# Link libraries
target_link_libraries(visual_qa PRIVATE
    tinyai_core
    tinyai_models
    tinyai_utils
)

# Set optimization flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(visual_qa PRIVATE -O2)
elseif(MSVC)
    target_compile_options(visual_qa PRIVATE /O2)
endif()

# Install the executable
install(TARGETS visual_qa
    RUNTIME DESTINATION bin/examples
)
