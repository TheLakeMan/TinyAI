/**
 * TinyAI Logging System Header
 *
 * This header defines the logging system for TinyAI, providing functions
 * for logging messages with different severity levels and configuring
 * logging output destinations.
 */

#ifndef TINYAI_LOGGING_H
#define TINYAI_LOGGING_H

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------- Log Levels ----------------- */

/**
 * Log levels
 */
typedef enum {
    TINYAI_LOG_NONE  = 0, /* No logging */
    TINYAI_LOG_ERROR = 1, /* Critical errors */
    TINYAI_LOG_WARN  = 2, /* Warnings */
    TINYAI_LOG_INFO  = 3, /* Informational messages */
    TINYAI_LOG_DEBUG = 4, /* Debug information */
    TINYAI_LOG_TRACE = 5  /* Detailed trace information */
} TinyAILogLevel;

/* ----------------- Log Output ----------------- */

/**
 * Log output destinations
 */
typedef enum {
    TINYAI_LOG_OUTPUT_NONE    = 0,      /* No output */
    TINYAI_LOG_OUTPUT_CONSOLE = 1 << 0, /* Output to console */
    TINYAI_LOG_OUTPUT_FILE    = 1 << 1, /* Output to file */
    TINYAI_LOG_OUTPUT_CUSTOM  = 1 << 2  /* Output to custom handler */
} TinyAILogOutput;

/* ----------------- Log Format ----------------- */

/**
 * Log message formats
 */
typedef enum {
    TINYAI_LOG_FORMAT_PLAIN = 0, /* Plain text format */
    TINYAI_LOG_FORMAT_JSON  = 1, /* JSON format */
    TINYAI_LOG_FORMAT_CSV   = 2  /* CSV format */
} TinyAILogFormat;

/* ----------------- Log Configuration ----------------- */

/**
 * Log rotation options
 */
typedef struct {
    bool   enable_rotation;   /* Enable log rotation */
    size_t max_size;          /* Maximum log file size in bytes */
    int    max_files;         /* Maximum number of log files to keep */
    bool   rotate_on_startup; /* Rotate logs on startup */
} TinyAILogRotationConfig;

/**
 * Log configuration
 */
typedef struct {
    TinyAILogLevel          level;             /* Logging level */
    unsigned int            output;            /* Bit mask of output destinations */
    TinyAILogFormat         format;            /* Log message format */
    const char             *log_file_path;     /* Path to log file (if file output enabled) */
    TinyAILogRotationConfig rotation;          /* Log rotation configuration */
    bool                    include_timestamp; /* Include timestamp in log messages */
    bool                    include_level;     /* Include level in log messages */
    bool                    include_source;    /* Include source file/line in log messages */
    bool                    colorize_console;  /* Use colors in console output */
} TinyAILogConfig;

/**
 * Custom log handler function type
 *
 * @param level Log level
 * @param message Formatted log message
 * @param user_data User data passed to handler
 */
typedef void (*TinyAILogHandler)(TinyAILogLevel level, const char *message, void *user_data);

/* ----------------- Core Logging Functions ----------------- */

/**
 * Initialize the logging system with default configuration
 *
 * Default: INFO level, console output, plain format
 *
 * @return 1 on success, 0 on failure
 */
int tinyai_logging_init(void);

/**
 * Configure the logging system
 *
 * @param config Logging configuration
 * @return 1 on success, 0 on failure
 */
int tinyai_configure_logging(const TinyAILogConfig *config);

/**
 * Get the current logging configuration
 *
 * @param config Pointer to store the configuration
 */
void tinyai_get_logging_config(TinyAILogConfig *config);

/**
 * Set the logging level
 *
 * @param level Logging level
 */
void tinyai_set_log_level(TinyAILogLevel level);

/**
 * Get the current logging level
 *
 * @return Current logging level
 */
TinyAILogLevel tinyai_get_log_level(void);

/**
 * Configure log file output
 *
 * @param file_path Path to log file
 * @return 1 on success, 0 on failure
 */
int tinyai_set_log_file(const char *file_path);

/**
 * Configure log rotation
 *
 * @param rotation_config Log rotation configuration
 * @return 1 on success, 0 on failure
 */
int tinyai_configure_log_rotation(const TinyAILogRotationConfig *rotation_config);

/**
 * Register a custom log handler
 *
 * @param handler Handler function
 * @param user_data User data to pass to handler
 * @return 1 on success, 0 on failure
 */
int tinyai_register_log_handler(TinyAILogHandler handler, void *user_data);

/**
 * Shutdown the logging system
 */
void tinyai_logging_shutdown(void);

/* ----------------- Logging Functions ----------------- */

/**
 * Log a message with specified level
 *
 * @param level Log level
 * @param format Format string
 * @param ... Format arguments
 */
void tinyai_log(TinyAILogLevel level, const char *format, ...);

/**
 * Log a message with variable arguments
 *
 * @param level Log level
 * @param format Format string
 * @param args Variable argument list
 */
void tinyai_vlog(TinyAILogLevel level, const char *format, va_list args);

/**
 * Log a message with source file and line information
 *
 * @param level Log level
 * @param file Source file
 * @param line Line number
 * @param format Format string
 * @param ... Format arguments
 */
void tinyai_log_with_source(TinyAILogLevel level, const char *file, int line, const char *format,
                            ...);

/* ----------------- Convenience Logging Macros ----------------- */

#define TINYAI_LOG_ERROR(format, ...)                                                              \
    tinyai_log_with_source(TINYAI_LOG_ERROR, __FILE__, __LINE__, format, ##__VA_ARGS__)

#define TINYAI_LOG_WARN(format, ...)                                                               \
    tinyai_log_with_source(TINYAI_LOG_WARN, __FILE__, __LINE__, format, ##__VA_ARGS__)

#define TINYAI_LOG_INFO(format, ...)                                                               \
    tinyai_log_with_source(TINYAI_LOG_INFO, __FILE__, __LINE__, format, ##__VA_ARGS__)

#define TINYAI_LOG_DEBUG(format, ...)                                                              \
    tinyai_log_with_source(TINYAI_LOG_DEBUG, __FILE__, __LINE__, format, ##__VA_ARGS__)

#define TINYAI_LOG_TRACE(format, ...)                                                              \
    tinyai_log_with_source(TINYAI_LOG_TRACE, __FILE__, __LINE__, format, ##__VA_ARGS__)

/* ----------------- Conditional Logging Macros ----------------- */

#define TINYAI_LOG_IF_ERROR(condition, format, ...)                                                \
    do {                                                                                           \
        if (condition)                                                                             \
            TINYAI_LOG_ERROR(format, ##__VA_ARGS__);                                               \
    } while (0)

#define TINYAI_LOG_IF_WARN(condition, format, ...)                                                 \
    do {                                                                                           \
        if (condition)                                                                             \
            TINYAI_LOG_WARN(format, ##__VA_ARGS__);                                                \
    } while (0)

#define TINYAI_LOG_IF_INFO(condition, format, ...)                                                 \
    do {                                                                                           \
        if (condition)                                                                             \
            TINYAI_LOG_INFO(format, ##__VA_ARGS__);                                                \
    } while (0)

#define TINYAI_LOG_IF_DEBUG(condition, format, ...)                                                \
    do {                                                                                           \
        if (condition)                                                                             \
            TINYAI_LOG_DEBUG(format, ##__VA_ARGS__);                                               \
    } while (0)

#define TINYAI_LOG_IF_TRACE(condition, format, ...)                                                \
    do {                                                                                           \
        if (condition)                                                                             \
            TINYAI_LOG_TRACE(format, ##__VA_ARGS__);                                               \
    } while (0)

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_LOGGING_H */