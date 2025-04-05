# Media Tagging Example CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# Create media tagging executable
add_executable(media_tagging
    main.c
    media_tagger.c
)

# Include directories
target_include_directories(media_tagging PRIVATE
    ${CMAKE_SOURCE_DIR}
)

# Link libraries
target_link_libraries(media_tagging PRIVATE
    tinyai_core
    tinyai_models
    tinyai_utils
)

# Set optimization flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(media_tagging PRIVATE -O2)
elseif(MSVC)
    target_compile_options(media_tagging PRIVATE /O2)
endif()

# Install the executable
install(TARGETS media_tagging
    RUNTIME DESTINATION bin/examples
)
