# Image Recognition Example CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# Create image_recognition executable
add_executable(image_recognition
    main.c
    image_classifier.c
)

# Include directories
target_include_directories(image_recognition PRIVATE
    ${CMAKE_SOURCE_DIR}
)

# Link libraries
target_link_libraries(image_recognition PRIVATE
    tinyai_core
    tinyai_models
    tinyai_utils
)

# Add optimization flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(image_recognition PRIVATE -O2)
elseif(MSVC)
    target_compile_options(image_recognition PRIVATE /O2)
endif()

# Install executable
install(TARGETS image_recognition
    RUNTIME DESTINATION bin/examples
)
