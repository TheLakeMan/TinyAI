# Image Captioning Example CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# Create image captioning executable
add_executable(image_captioning
    main.c
    image_captioner.c
)

# Include directories
target_include_directories(image_captioning PRIVATE
    ${CMAKE_SOURCE_DIR}
)

# Link libraries
target_link_libraries(image_captioning PRIVATE
    tinyai_core
    tinyai_models
    tinyai_utils
)

# Set optimization flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(image_captioning PRIVATE -O2)
elseif(MSVC)
    target_compile_options(image_captioning PRIVATE /O2)
endif()

# Install the executable
install(TARGETS image_captioning
    RUNTIME DESTINATION bin/examples
)
