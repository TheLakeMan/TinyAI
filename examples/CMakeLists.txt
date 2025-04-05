# Examples CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# Option to build examples
option(BUILD_EXAMPLES "Build TinyAI example applications" OFF)

if(BUILD_EXAMPLES)
    message(STATUS "Building TinyAI examples")
    
    # Include subdirectories
    add_subdirectory(chatbot)
    add_subdirectory(audio)
    add_subdirectory(multimodal)
    add_subdirectory(image_recognition)
    add_subdirectory(document_processor)
    add_subdirectory(media_tagging)
    
    # Install example data
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data
        DESTINATION share/tinyai/examples
        PATTERN "*.md" EXCLUDE
    )
else()
    message(STATUS "TinyAI examples will not be built (BUILD_EXAMPLES is OFF)")
endif()
