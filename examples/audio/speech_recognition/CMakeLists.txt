# Speech Recognition Example

# Source files
set(SPEECH_RECOGNITION_SOURCES
    main.c
    asr.c
)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}
)

# Create executable
add_executable(tinyai_asr ${SPEECH_RECOGNITION_SOURCES})

# Link against TinyAI library
target_link_libraries(tinyai_asr tinyai)

# Install executable
install(TARGETS tinyai_asr DESTINATION bin)

# Install sample data
install(FILES
    ${CMAKE_SOURCE_DIR}/data/sample_speech.wav
    DESTINATION share/tinyai/examples/audio
)
