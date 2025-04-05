# Chatbot Example CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# Create chatbot executable
add_executable(chatbot
    main.c
    chat_model.c
)

# Include directories
target_include_directories(chatbot PRIVATE
    ${CMAKE_SOURCE_DIR}
)

# Link libraries
target_link_libraries(chatbot PRIVATE
    tinyai_core
    tinyai_models
    tinyai_utils
)

# Set optimization flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(chatbot PRIVATE -O2)
elseif(MSVC)
    target_compile_options(chatbot PRIVATE /O2)
endif()

# Install the executable
install(TARGETS chatbot
    RUNTIME DESTINATION bin/examples
)
