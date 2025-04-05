# Audio Examples for TinyAI

# Voice Activity Detection example
add_subdirectory(voice_detection)

# Keyword Spotting example
add_subdirectory(keyword_spotting)

# Speech Recognition example
add_subdirectory(speech_recognition)

# Audio Classification example
# Uncomment once implemented
# add_subdirectory(audio_classification)

# Create data directory for samples
set(AUDIO_SAMPLE_DIR ${CMAKE_BINARY_DIR}/data/audio)
file(MAKE_DIRECTORY ${AUDIO_SAMPLE_DIR})

# Copy sample audio files for testing
# This will be expanded as more samples are added
file(COPY 
    ${CMAKE_SOURCE_DIR}/data/sample_text.txt  # Replace with actual audio samples when available
    DESTINATION ${AUDIO_SAMPLE_DIR}
)

# Install audio examples and data
install(DIRECTORY ${AUDIO_SAMPLE_DIR}
    DESTINATION share/tinyai/examples
)
