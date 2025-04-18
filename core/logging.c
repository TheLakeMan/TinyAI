/**
 * TinyAI Logging System Implementation
 */

#include "logging.h"
#include "io.h"
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

/* ----------------- Private Data Types ----------------- */

/**
 * Log level name mapping
 */
static const char *LOG_LEVEL_NAMES[] = {"NONE", "ERROR", "WARN", "INFO", "DEBUG", "TRACE"};

/**
 * Log level colors (for console output)
 */
#ifdef _WIN32
static const WORD LOG_LEVEL_COLORS[] = {
    0,                                                         /* NONE */
    FOREGROUND_RED | FOREGROUND_INTENSITY,                     /* ERROR */
    FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY,  /* WARN */
    FOREGROUND_GREEN | FOREGROUND_INTENSITY,                   /* INFO */
    FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_INTENSITY, /* DEBUG */
    FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED        /* TRACE */
};
#else
static const char *LOG_LEVEL_COLORS[] = {
    "\033[0m",    /* NONE */
    "\033[1;31m", /* ERROR (Bold Red) */
    "\033[1;33m", /* WARN (Bold Yellow) */
    "\033[1;32m", /* INFO (Bold Green) */
    "\033[1;36m", /* DEBUG (Bold Cyan) */
    "\033[1;37m"  /* TRACE (Bold White) */
};
#endif

/**
 * Maximum log message length
 */
#define MAX_LOG_MESSAGE_LENGTH 4096

/**
 * Maximum timestamp length
 */
#define MAX_TIMESTAMP_LENGTH 64

/**
 * Maximum log file path length
 */
#define MAX_LOG_FILE_PATH 256

/**
 * Log system state
 */
typedef struct {
    bool             initialized;
    TinyAILogConfig  config;
    FILE            *log_file;
    TinyAILogHandler custom_handler;
    void            *custom_handler_data;
    char             log_file_path[MAX_LOG_FILE_PATH];
    size_t           current_file_size;
    int              current_file_index;
} LogSystemState;

/**
 * Log system singleton instance
 */
static LogSystemState log_state = {0};

/* ----------------- Forward Declarations ----------------- */

static void format_timestamp(char *buffer, size_t buffer_size);
static void format_log_message(TinyAILogLevel level, const char *file, int line, const char *format,
                               va_list args, char *buffer, size_t buffer_size);
static void format_log_message_json(TinyAILogLevel level, const char *file, int line,
                                    const char *format, va_list args, char *buffer,
                                    size_t buffer_size);
static void format_log_message_csv(TinyAILogLevel level, const char *file, int line,
                                   const char *format, va_list args, char *buffer,
                                   size_t buffer_size);
static void write_to_console(TinyAILogLevel level, const char *message);
static int  write_to_file(const char *message);
static int  rotate_log_file_if_needed(void);
static void escape_json_string(const char *input, char *output, size_t output_size);

/* ----------------- Core Functions ----------------- */

int tinyai_logging_init(void)
{
    if (log_state.initialized) {
        return 1; /* Already initialized */
    }

    /* Set default configuration */
    memset(&log_state, 0, sizeof(LogSystemState));
    log_state.config.level                      = TINYAI_LOG_INFO;
    log_state.config.output                     = TINYAI_LOG_OUTPUT_CONSOLE;
    log_state.config.format                     = TINYAI_LOG_FORMAT_PLAIN;
    log_state.config.include_timestamp          = true;
    log_state.config.include_level              = true;
    log_state.config.include_source             = false;
    log_state.config.colorize_console           = true;
    log_state.config.log_file_path              = NULL;
    log_state.config.rotation.enable_rotation   = false;
    log_state.config.rotation.max_size          = 10 * 1024 * 1024; /* 10 MB */
    log_state.config.rotation.max_files         = 5;
    log_state.config.rotation.rotate_on_startup = false;

    log_state.log_file            = NULL;
    log_state.custom_handler      = NULL;
    log_state.custom_handler_data = NULL;
    log_state.current_file_size   = 0;
    log_state.current_file_index  = 0;

    log_state.initialized = true;

    TINYAI_LOG_INFO("Logging system initialized");
    return 1;
}

int tinyai_configure_logging(const TinyAILogConfig *config)
{
    if (!log_state.initialized) {
        if (!tinyai_logging_init()) {
            return 0;
        }
    }

    if (!config) {
        return 0;
    }

    /* Close existing log file if we're changing output settings */
    if ((log_state.config.output & TINYAI_LOG_OUTPUT_FILE) && (log_state.log_file != NULL) &&
        ((!(config->output & TINYAI_LOG_OUTPUT_FILE)) ||
         (config->log_file_path != log_state.config.log_file_path))) {
        fclose(log_state.log_file);
        log_state.log_file = NULL;
    }

    /* Copy configuration */
    log_state.config.level             = config->level;
    log_state.config.output            = config->output;
    log_state.config.format            = config->format;
    log_state.config.include_timestamp = config->include_timestamp;
    log_state.config.include_level     = config->include_level;
    log_state.config.include_source    = config->include_source;
    log_state.config.colorize_console  = config->colorize_console;

    /* Configure rotation */
    log_state.config.rotation.enable_rotation   = config->rotation.enable_rotation;
    log_state.config.rotation.max_size          = config->rotation.max_size;
    log_state.config.rotation.max_files         = config->rotation.max_files;
    log_state.config.rotation.rotate_on_startup = config->rotation.rotate_on_startup;

    /* Configure log file */
    if (config->output & TINYAI_LOG_OUTPUT_FILE) {
        if (config->log_file_path) {
            log_state.config.log_file_path = config->log_file_path;
            strncpy(log_state.log_file_path, config->log_file_path, MAX_LOG_FILE_PATH - 1);
            log_state.log_file_path[MAX_LOG_FILE_PATH - 1] = '\0';

            /* Rotate logs on startup if configured */
            if (config->rotation.rotate_on_startup) {
                /* Try to rename existing log file if it exists */
                FILE *test_file = fopen(log_state.log_file_path, "r");
                if (test_file) {
                    fclose(test_file);

                    char       backup_path[MAX_LOG_FILE_PATH];
                    time_t     now      = time(NULL);
                    struct tm *timeinfo = localtime(&now);

                    snprintf(backup_path, MAX_LOG_FILE_PATH, "%s.%04d%02d%02d-%02d%02d%02d",
                             log_state.log_file_path, timeinfo->tm_year + 1900,
                             timeinfo->tm_mon + 1, timeinfo->tm_mday, timeinfo->tm_hour,
                             timeinfo->tm_min, timeinfo->tm_sec);

                    rename(log_state.log_file_path, backup_path);
                }
            }

            /* Open log file */
            log_state.log_file = fopen(log_state.log_file_path, "a");
            if (!log_state.log_file) {
                fprintf(stderr, "Failed to open log file: %s\n", log_state.log_file_path);
                log_state.config.output &= ~TINYAI_LOG_OUTPUT_FILE;
                return 0;
            }

            /* Get current file size */
            fseek(log_state.log_file, 0, SEEK_END);
            log_state.current_file_size = ftell(log_state.log_file);
        }
    }

    TINYAI_LOG_INFO("Logging configuration updated");
    return 1;
}

void tinyai_get_logging_config(TinyAILogConfig *config)
{
    if (!log_state.initialized) {
        tinyai_logging_init();
    }

    if (config) {
        memcpy(config, &log_state.config, sizeof(TinyAILogConfig));
    }
}

void tinyai_set_log_level(TinyAILogLevel level)
{
    if (!log_state.initialized) {
        tinyai_logging_init();
    }

    log_state.config.level = level;
}

TinyAILogLevel tinyai_get_log_level(void)
{
    if (!log_state.initialized) {
        tinyai_logging_init();
    }

    return log_state.config.level;
}

int tinyai_set_log_file(const char *file_path)
{
    if (!log_state.initialized) {
        tinyai_logging_init();
    }

    if (!file_path) {
        if (log_state.log_file) {
            fclose(log_state.log_file);
            log_state.log_file = NULL;
        }
        log_state.config.output &= ~TINYAI_LOG_OUTPUT_FILE;
        return 1;
    }

    /* Close existing log file if open */
    if (log_state.log_file) {
        fclose(log_state.log_file);
        log_state.log_file = NULL;
    }

    /* Update config */
    log_state.config.log_file_path = file_path;
    log_state.config.output |= TINYAI_LOG_OUTPUT_FILE;
    strncpy(log_state.log_file_path, file_path, MAX_LOG_FILE_PATH - 1);
    log_state.log_file_path[MAX_LOG_FILE_PATH - 1] = '\0';

    /* Open new log file */
    log_state.log_file = fopen(log_state.log_file_path, "a");
    if (!log_state.log_file) {
        fprintf(stderr, "Failed to open log file: %s\n", log_state.log_file_path);
        log_state.config.output &= ~TINYAI_LOG_OUTPUT_FILE;
        return 0;
    }

    /* Get current file size */
    fseek(log_state.log_file, 0, SEEK_END);
    log_state.current_file_size = ftell(log_state.log_file);

    return 1;
}

int tinyai_configure_log_rotation(const TinyAILogRotationConfig *rotation_config)
{
    if (!log_state.initialized) {
        tinyai_logging_init();
    }

    if (!rotation_config) {
        return 0;
    }

    /* Update rotation config */
    log_state.config.rotation.enable_rotation   = rotation_config->enable_rotation;
    log_state.config.rotation.max_size          = rotation_config->max_size;
    log_state.config.rotation.max_files         = rotation_config->max_files;
    log_state.config.rotation.rotate_on_startup = rotation_config->rotate_on_startup;

    return 1;
}

int tinyai_register_log_handler(TinyAILogHandler handler, void *user_data)
{
    if (!log_state.initialized) {
        tinyai_logging_init();
    }

    log_state.custom_handler      = handler;
    log_state.custom_handler_data = user_data;
    log_state.config.output |= TINYAI_LOG_OUTPUT_CUSTOM;

    return 1;
}

void tinyai_logging_shutdown(void)
{
    if (!log_state.initialized) {
        return;
    }

    TINYAI_LOG_INFO("Logging system shutting down");

    /* Close log file if open */
    if (log_state.log_file) {
        fclose(log_state.log_file);
        log_state.log_file = NULL;
    }

    log_state.initialized = false;
}

/* ----------------- Logging Functions ----------------- */

void tinyai_log(TinyAILogLevel level, const char *format, ...)
{
    va_list args;
    va_start(args, format);
    tinyai_vlog(level, format, args);
    va_end(args);
}

void tinyai_vlog(TinyAILogLevel level, const char *format, va_list args)
{
    if (!log_state.initialized) {
        tinyai_logging_init();
    }

    /* Check log level */
    if (level > log_state.config.level || level == TINYAI_LOG_NONE) {
        return;
    }

    /* Format message based on selected format */
    char    message[MAX_LOG_MESSAGE_LENGTH];
    va_list args_copy;
    va_copy(args_copy, args);

    switch (log_state.config.format) {
    case TINYAI_LOG_FORMAT_JSON:
        format_log_message_json(level, NULL, 0, format, args_copy, message, sizeof(message));
        break;
    case TINYAI_LOG_FORMAT_CSV:
        format_log_message_csv(level, NULL, 0, format, args_copy, message, sizeof(message));
        break;
    case TINYAI_LOG_FORMAT_PLAIN:
    default:
        format_log_message(level, NULL, 0, format, args_copy, message, sizeof(message));
        break;
    }

    va_end(args_copy);

    /* Write to enabled output destinations */
    if (log_state.config.output & TINYAI_LOG_OUTPUT_CONSOLE) {
        write_to_console(level, message);
    }

    if (log_state.config.output & TINYAI_LOG_OUTPUT_FILE) {
        write_to_file(message);
    }

    if ((log_state.config.output & TINYAI_LOG_OUTPUT_CUSTOM) && log_state.custom_handler) {
        log_state.custom_handler(level, message, log_state.custom_handler_data);
    }
}

void tinyai_log_with_source(TinyAILogLevel level, const char *file, int line, const char *format,
                            ...)
{
    if (!log_state.initialized) {
        tinyai_logging_init();
    }

    /* Check log level */
    if (level > log_state.config.level || level == TINYAI_LOG_NONE) {
        return;
    }

    /* Format message based on selected format */
    char    message[MAX_LOG_MESSAGE_LENGTH];
    va_list args;
    va_start(args, format);

    switch (log_state.config.format) {
    case TINYAI_LOG_FORMAT_JSON:
        format_log_message_json(level, file, line, format, args, message, sizeof(message));
        break;
    case TINYAI_LOG_FORMAT_CSV:
        format_log_message_csv(level, file, line, format, args, message, sizeof(message));
        break;
    case TINYAI_LOG_FORMAT_PLAIN:
    default:
        format_log_message(level, file, line, format, args, message, sizeof(message));
        break;
    }

    va_end(args);

    /* Write to enabled output destinations */
    if (log_state.config.output & TINYAI_LOG_OUTPUT_CONSOLE) {
        write_to_console(level, message);
    }

    if (log_state.config.output & TINYAI_LOG_OUTPUT_FILE) {
        write_to_file(message);
    }

    if ((log_state.config.output & TINYAI_LOG_OUTPUT_CUSTOM) && log_state.custom_handler) {
        log_state.custom_handler(level, message, log_state.custom_handler_data);
    }
}

/* ----------------- Helper Functions ----------------- */

static void format_timestamp(char *buffer, size_t buffer_size)
{
    time_t     now      = time(NULL);
    struct tm *timeinfo = localtime(&now);

    strftime(buffer, buffer_size, "%Y-%m-%d %H:%M:%S", timeinfo);
}

static void format_log_message(TinyAILogLevel level, const char *file, int line, const char *format,
                               va_list args, char *buffer, size_t buffer_size)
{
    char   timestamp[MAX_TIMESTAMP_LENGTH] = {0};
    char  *buffer_pos                      = buffer;
    size_t remaining                       = buffer_size;
    int    written                         = 0;

    /* Add timestamp if enabled */
    if (log_state.config.include_timestamp) {
        format_timestamp(timestamp, sizeof(timestamp));
        written = snprintf(buffer_pos, remaining, "[%s] ", timestamp);
        buffer_pos += written;
        remaining -= written;
    }

    /* Add level if enabled */
    if (log_state.config.include_level && level > 0 &&
        level < sizeof(LOG_LEVEL_NAMES) / sizeof(LOG_LEVEL_NAMES[0])) {
        written = snprintf(buffer_pos, remaining, "[%s] ", LOG_LEVEL_NAMES[level]);
        buffer_pos += written;
        remaining -= written;
    }

    /* Add source info if enabled and available */
    if (log_state.config.include_source && file && line > 0) {
        /* Extract filename without path */
        const char *filename  = file;
        const char *lastSlash = strrchr(file, '/');
        if (lastSlash) {
            filename = lastSlash + 1;
        }
        else {
            lastSlash = strrchr(file, '\\');
            if (lastSlash) {
                filename = lastSlash + 1;
            }
        }

        written = snprintf(buffer_pos, remaining, "[%s:%d] ", filename, line);
        buffer_pos += written;
        remaining -= written;
    }

    /* Add message */
    vsnprintf(buffer_pos, remaining, format, args);
}

static void format_log_message_json(TinyAILogLevel level, const char *file, int line,
                                    const char *format, va_list args, char *buffer,
                                    size_t buffer_size)
{
    char timestamp[MAX_TIMESTAMP_LENGTH]         = {0};
    char message[MAX_LOG_MESSAGE_LENGTH]         = {0};
    char escaped_message[MAX_LOG_MESSAGE_LENGTH] = {0};

    /* Format the message text */
    vsnprintf(message, sizeof(message), format, args);

    /* Escape JSON strings */
    escape_json_string(message, escaped_message, sizeof(escaped_message));

    /* Get timestamp */
    format_timestamp(timestamp, sizeof(timestamp));

    /* Build JSON */
    snprintf(buffer, buffer_size, "{\"timestamp\":\"%s\",\"level\":\"%s\",\"message\":\"%s\"%s%s}",
             timestamp,
             level > 0 && level < sizeof(LOG_LEVEL_NAMES) / sizeof(LOG_LEVEL_NAMES[0])
                 ? LOG_LEVEL_NAMES[level]
                 : "UNKNOWN",
             escaped_message, file ? ",\"file\":\"" : "", file ? file : "");

    if (file && line > 0) {
        /* Append line number to JSON */
        size_t current_length = strlen(buffer);
        if (current_length < buffer_size - 20) {
            snprintf(buffer + current_length - 1, buffer_size - current_length, ",\"line\":%d}",
                     line);
        }
    }
}

static void format_log_message_csv(TinyAILogLevel level, const char *file, int line,
                                   const char *format, va_list args, char *buffer,
                                   size_t buffer_size)
{
    char   timestamp[MAX_TIMESTAMP_LENGTH] = {0};
    char   message[MAX_LOG_MESSAGE_LENGTH] = {0};
    char  *buffer_pos                      = buffer;
    size_t remaining                       = buffer_size;
    int    written                         = 0;

    /* Format timestamp */
    format_timestamp(timestamp, sizeof(timestamp));

    /* Format the message text */
    vsnprintf(message, sizeof(message), format, args);

    /* Replace commas in message with escape sequence to prevent CSV issues */
    char *p = message;
    while ((p = strchr(p, ',')) != NULL) {
        *p = ' ';
    }

    /* Replace newlines with spaces */
    p = message;
    while ((p = strchr(p, '\n')) != NULL) {
        *p = ' ';
    }

    /* Format CSV line */
    written = snprintf(buffer_pos, remaining, "%s,%s,", timestamp,
                       level > 0 && level < sizeof(LOG_LEVEL_NAMES) / sizeof(LOG_LEVEL_NAMES[0])
                           ? LOG_LEVEL_NAMES[level]
                           : "UNKNOWN");
    buffer_pos += written;
    remaining -= written;

    /* Add file and line if available */
    if (file && line > 0) {
        written = snprintf(buffer_pos, remaining, "%s,%d,", file, line);
    }
    else {
        written = snprintf(buffer_pos, remaining, ",,");
    }
    buffer_pos += written;
    remaining -= written;

    /* Add message */
    snprintf(buffer_pos, remaining, "%s", message);
}

static void write_to_console(TinyAILogLevel level, const char *message)
{
#ifdef _WIN32
    /* Windows console coloring */
    HANDLE                     console = GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_SCREEN_BUFFER_INFO console_info;
    WORD                       original_attributes = 0;

    if (log_state.config.colorize_console && level > 0 &&
        level < sizeof(LOG_LEVEL_COLORS) / sizeof(LOG_LEVEL_COLORS[0])) {
        GetConsoleScreenBufferInfo(console, &console_info);
        original_attributes = console_info.wAttributes;
        SetConsoleTextAttribute(console, LOG_LEVEL_COLORS[level]);
    }

    printf("%s\n", message);

    if (log_state.config.colorize_console && level > 0 &&
        level < sizeof(LOG_LEVEL_COLORS) / sizeof(LOG_LEVEL_COLORS[0])) {
        SetConsoleTextAttribute(console, original_attributes);
    }
#else
    /* ANSI color codes for Unix-like systems */
    if (log_state.config.colorize_console && level > 0 &&
        level < sizeof(LOG_LEVEL_COLORS) / sizeof(LOG_LEVEL_COLORS[0])) {
        printf("%s%s\033[0m\n", LOG_LEVEL_COLORS[level], message);
    }
    else {
        printf("%s\n", message);
    }
#endif
}

static int write_to_file(const char *message)
{
    if (!log_state.log_file) {
        /* Try to reopen log file */
        if (log_state.log_file_path[0] == '\0') {
            return 0;
        }

        log_state.log_file = fopen(log_state.log_file_path, "a");
        if (!log_state.log_file) {
            fprintf(stderr, "Failed to open log file: %s\n", log_state.log_file_path);
            return 0;
        }
    }

    /* Rotate log file if needed */
    if (log_state.config.rotation.enable_rotation) {
        if (!rotate_log_file_if_needed()) {
            return 0;
        }
    }

    /* Write message to file */
    fprintf(log_state.log_file, "%s\n", message);
    fflush(log_state.log_file);

    /* Update file size */
    log_state.current_file_size += strlen(message) + 1; /* +1 for newline */

    return 1;
}

static int rotate_log_file_if_needed(void)
{
    if (!log_state.config.rotation.enable_rotation ||
        log_state.current_file_size < log_state.config.rotation.max_size) {
        return 1; /* No rotation needed */
    }

    /* Close current log file */
    if (log_state.log_file) {
        fclose(log_state.log_file);
        log_state.log_file = NULL;
    }

    /* Rotate existing log files */
    if (log_state.config.rotation.max_files > 0) {
        char old_path[MAX_LOG_FILE_PATH];
        char new_path[MAX_LOG_FILE_PATH];

        /* Delete oldest log file if it exists */
        snprintf(old_path, sizeof(old_path), "%s.%d", log_state.log_file_path,
                 log_state.config.rotation.max_files - 1);
        remove(old_path);

        /* Rename other log files */
        for (int i = log_state.config.rotation.max_files - 2; i >= 0; i--) {
            if (i == 0) {
                strncpy(old_path, log_state.log_file_path, sizeof(old_path) - 1);
            }
            else {
                snprintf(old_path, sizeof(old_path), "%s.%d", log_state.log_file_path, i);
            }

            snprintf(new_path, sizeof(new_path), "%s.%d", log_state.log_file_path, i + 1);

            /* Check if file exists before renaming */
            FILE *test_file = fopen(old_path, "r");
            if (test_file) {
                fclose(test_file);
                rename(old_path, new_path);
            }
        }
    }

    /* Open new log file */
    log_state.log_file = fopen(log_state.log_file_path, "w");
    if (!log_state.log_file) {
        fprintf(stderr, "Failed to open new log file after rotation: %s\n",
                log_state.log_file_path);
        return 0;
    }

    /* Reset file size counter */
    log_state.current_file_size = 0;

    /* Log rotation message */
    char timestamp[MAX_TIMESTAMP_LENGTH];
    format_timestamp(timestamp, sizeof(timestamp));
    fprintf(log_state.log_file, "[%s] [INFO] Log file rotated\n", timestamp);
    fflush(log_state.log_file);
    log_state.current_file_size += strlen(timestamp) + 28; /* 28 = length of rotation message */

    return 1;
}

static void escape_json_string(const char *input, char *output, size_t output_size)
{
    size_t j = 0;

    for (size_t i = 0; input[i] != '\0' && j < output_size - 1; i++) {
        switch (input[i]) {
        case '\\':
            if (j < output_size - 2) {
                output[j++] = '\\';
                output[j++] = '\\';
            }
            break;
        case '\"':
            if (j < output_size - 2) {
                output[j++] = '\\';
                output[j++] = '\"';
            }
            break;
        case '\b':
            if (j < output_size - 2) {
                output[j++] = '\\';
                output[j++] = 'b';
            }
            break;
        case '\f':
            if (j < output_size - 2) {
                output[j++] = '\\';
                output[j++] = 'f';
            }
            break;
        case '\n':
            if (j < output_size - 2) {
                output[j++] = '\\';
                output[j++] = 'n';
            }
            break;
        case '\r':
            if (j < output_size - 2) {
                output[j++] = '\\';
                output[j++] = 'r';
            }
            break;
        case '\t':
            if (j < output_size - 2) {
                output[j++] = '\\';
                output[j++] = 't';
            }
            break;
        default:
            if ((unsigned char)input[i] < 32) {
                /* Control characters */
                if (j < output_size - 6) {
                    snprintf(output + j, output_size - j, "\\u%04x", input[i]);
                    j += 6;
                }
            }
            else {
                output[j++] = input[i];
            }
            break;
        }
    }

    output[j] = '\0';
}