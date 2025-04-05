# Voice Activity Detection Example

# Source files
set(VOICE_DETECTION_SOURCES
    main.c
    vad.c
)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}
)

# Create executable
add_executable(tinyai_vad ${VOICE_DETECTION_SOURCES})

# Link against TinyAI library
target_link_libraries(tinyai_vad tinyai)

# Install executable
install(TARGETS tinyai_vad DESTINATION bin)

# Install sample data
install(FILES
    ${CMAKE_SOURCE_DIR}/data/sample_speech.wav
    DESTINATION share/tinyai/examples/audio
)
