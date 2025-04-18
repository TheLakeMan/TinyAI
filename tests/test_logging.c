/**
 * TinyAI Logging System Test
 *
 * This file contains tests for the TinyAI logging system.
 */

#include "../core/logging.h"
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Test log handler callback data */
typedef struct {
    int            messages_received;
    TinyAILogLevel last_level;
    char           last_message[4096];
} TestLogHandlerData;

/* Test log handler callback function */
static void test_log_handler(TinyAILogLevel level, const char *message, void *user_data)
{
    TestLogHandlerData *data = (TestLogHandlerData *)user_data;
    data->messages_received++;
    data->last_level = level;
    strncpy(data->last_message, message, sizeof(data->last_message) - 1);
    data->last_message[sizeof(data->last_message) - 1] = '\0';
}

/* Test basic logging functionality */
void test_basic_logging()
{
    printf("Testing basic logging functionality... ");

    /* Initialize logging system */
    tinyai_logging_init();

    /* Set log level to INFO */
    tinyai_set_log_level(TINYAI_LOG_INFO);

    /* Log messages at different levels */
    TINYAI_LOG_ERROR("This is an error message");
    TINYAI_LOG_WARN("This is a warning message");
    TINYAI_LOG_INFO("This is an info message");
    TINYAI_LOG_DEBUG("This is a debug message (should not be logged)");
    TINYAI_LOG_TRACE("This is a trace message (should not be logged)");

    /* Change log level to DEBUG */
    tinyai_set_log_level(TINYAI_LOG_DEBUG);

    /* Log a debug message (should be logged now) */
    TINYAI_LOG_DEBUG("This is a debug message (should be logged)");

    /* No assertions here, just visual inspection */
    printf("PASS\n");
}

/* Test custom log handler */
void test_custom_handler()
{
    printf("Testing custom log handler... ");

    /* Initialize logging system */
    tinyai_logging_init();

    /* Set up custom handler data */
    TestLogHandlerData handler_data = {0};

    /* Register custom handler */
    tinyai_register_log_handler(test_log_handler, &handler_data);

    /* Set log level */
    tinyai_set_log_level(TINYAI_LOG_DEBUG);

    /* Configure logging output to only use custom handler */
    TinyAILogConfig config;
    tinyai_get_logging_config(&config);
    config.output = TINYAI_LOG_OUTPUT_CUSTOM;
    tinyai_configure_logging(&config);

    /* Log messages */
    tinyai_log(TINYAI_LOG_ERROR, "Custom handler error message");
    assert(handler_data.messages_received == 1);
    assert(handler_data.last_level == TINYAI_LOG_ERROR);
    assert(strstr(handler_data.last_message, "Custom handler error message") != NULL);

    tinyai_log(TINYAI_LOG_DEBUG, "Custom handler debug message");
    assert(handler_data.messages_received == 2);
    assert(handler_data.last_level == TINYAI_LOG_DEBUG);
    assert(strstr(handler_data.last_message, "Custom handler debug message") != NULL);

    tinyai_log(TINYAI_LOG_TRACE, "Custom handler trace message (should not be logged)");
    assert(handler_data.messages_received == 2); /* Should not have increased */

    /* Reset output configuration */
    config.output = TINYAI_LOG_OUTPUT_CONSOLE;
    tinyai_configure_logging(&config);

    printf("PASS\n");
}

/* Test log file output */
void test_log_file_output()
{
    printf("Testing log file output... ");

    /* Initialize logging system */
    tinyai_logging_init();

    /* Set log file path */
    const char *test_log_file = "test_log.txt";
    tinyai_set_log_file(test_log_file);

    /* Configure logging output to use both console and file */
    TinyAILogConfig config;
    tinyai_get_logging_config(&config);
    config.output = TINYAI_LOG_OUTPUT_CONSOLE | TINYAI_LOG_OUTPUT_FILE;
    config.level  = TINYAI_LOG_INFO;
    tinyai_configure_logging(&config);

    /* Log some messages */
    char       timestamp_str[32];
    time_t     now     = time(NULL);
    struct tm *tm_info = localtime(&now);
    strftime(timestamp_str, sizeof(timestamp_str), "%Y-%m-%d %H:%M", tm_info);

    TINYAI_LOG_INFO("Test log file message at %s", timestamp_str);

    /* Close log file */
    tinyai_set_log_file(NULL);

    /* Verify log file exists and contains message */
    FILE *log_file = fopen(test_log_file, "r");
    assert(log_file != NULL);

    char   file_contents[4096] = {0};
    size_t bytes_read          = fread(file_contents, 1, sizeof(file_contents) - 1, log_file);
    file_contents[bytes_read]  = '\0';
    fclose(log_file);

    assert(strstr(file_contents, timestamp_str) != NULL);
    assert(strstr(file_contents, "Test log file message") != NULL);

    /* Clean up */
    remove(test_log_file);

    printf("PASS\n");
}

/* Test log formatting */
void test_log_formatting()
{
    printf("Testing log formatting... ");

    /* Initialize logging system */
    tinyai_logging_init();

    /* Set up custom handler data */
    TestLogHandlerData handler_data = {0};

    /* Register custom handler */
    tinyai_register_log_handler(test_log_handler, &handler_data);

    /* Set configuration to use custom handler only */
    TinyAILogConfig config;
    tinyai_get_logging_config(&config);
    config.output            = TINYAI_LOG_OUTPUT_CUSTOM;
    config.level             = TINYAI_LOG_INFO;
    config.include_timestamp = true;
    config.include_level     = true;
    config.include_source    = true;
    tinyai_configure_logging(&config);

    /* Test plain format */
    config.format = TINYAI_LOG_FORMAT_PLAIN;
    tinyai_configure_logging(&config);

    TINYAI_LOG_INFO("Plain format message");
    assert(handler_data.messages_received == 1);
    assert(strstr(handler_data.last_message, "[INFO]") != NULL);
    assert(strstr(handler_data.last_message, "Plain format message") != NULL);
    assert(strstr(handler_data.last_message, "test_logging.c") != NULL);

    /* Test JSON format */
    config.format = TINYAI_LOG_FORMAT_JSON;
    tinyai_configure_logging(&config);

    TINYAI_LOG_INFO("JSON format message");
    assert(handler_data.messages_received == 2);
    assert(strstr(handler_data.last_message, "\"level\":\"INFO\"") != NULL);
    assert(strstr(handler_data.last_message, "\"message\":\"JSON format message\"") != NULL);
    assert(strstr(handler_data.last_message, "\"file\":\"") != NULL);

    /* Test CSV format */
    config.format = TINYAI_LOG_FORMAT_CSV;
    tinyai_configure_logging(&config);

    TINYAI_LOG_INFO("CSV format message");
    assert(handler_data.messages_received == 3);

    /* Reset output configuration */
    config.format = TINYAI_LOG_FORMAT_PLAIN;
    config.output = TINYAI_LOG_OUTPUT_CONSOLE;
    tinyai_configure_logging(&config);

    printf("PASS\n");
}

/* Test log rotation */
void test_log_rotation()
{
    printf("Testing log rotation... ");

    /* Initialize logging system */
    tinyai_logging_init();

    /* Set log file path */
    const char *test_log_file = "rotation_test.log";

    /* Configure log rotation */
    TinyAILogConfig config;
    tinyai_get_logging_config(&config);
    config.output                   = TINYAI_LOG_OUTPUT_FILE;
    config.level                    = TINYAI_LOG_INFO;
    config.log_file_path            = test_log_file;
    config.rotation.enable_rotation = true;
    config.rotation.max_size        = 1024; /* Small size for testing */
    config.rotation.max_files       = 3;
    tinyai_configure_logging(&config);

    /* Generate enough logs to trigger multiple rotations */
    char log_line[100];
    for (int i = 0; i < 50; i++) {
        snprintf(log_line, sizeof(log_line),
                 "Log line %d with some padding to make the log file grow faster.............", i);
        TINYAI_LOG_INFO("%s", log_line);
    }

    /* Close log file */
    tinyai_set_log_file(NULL);

    /* Check if rotation files exist */
    FILE *file;
    int   rotation_count = 0;

    /* Check main log file */
    file = fopen(test_log_file, "r");
    if (file) {
        fclose(file);
        rotation_count++;
    }

    /* Check rotation files */
    char rotation_path[256];
    for (int i = 1; i <= config.rotation.max_files; i++) {
        snprintf(rotation_path, sizeof(rotation_path), "%s.%d", test_log_file, i);
        file = fopen(rotation_path, "r");
        if (file) {
            fclose(file);
            rotation_count++;
            remove(rotation_path); /* Clean up */
        }
    }

    /* At least some rotation files should exist */
    assert(rotation_count > 1);

    /* Clean up */
    remove(test_log_file);

    printf("PASS\n");
}

/* Test conditional logging macros */
void test_conditional_logging()
{
    printf("Testing conditional logging macros... ");

    /* Initialize logging system */
    tinyai_logging_init();

    /* Set up custom handler data */
    TestLogHandlerData handler_data = {0};

    /* Register custom handler */
    tinyai_register_log_handler(test_log_handler, &handler_data);

    /* Set configuration to use custom handler only */
    TinyAILogConfig config;
    tinyai_get_logging_config(&config);
    config.output = TINYAI_LOG_OUTPUT_CUSTOM;
    config.level  = TINYAI_LOG_INFO;
    tinyai_configure_logging(&config);

    /* Reset message count */
    handler_data.messages_received = 0;

    /* Test conditional logging when condition is true */
    int error_code = 1;
    TINYAI_LOG_IF_ERROR(error_code != 0, "Error occurred with code %d", error_code);
    assert(handler_data.messages_received == 1);
    assert(strstr(handler_data.last_message, "Error occurred with code 1") != NULL);

    /* Test conditional logging when condition is false */
    error_code = 0;
    TINYAI_LOG_IF_ERROR(error_code != 0, "This should not be logged");
    assert(handler_data.messages_received == 1); /* Count should not change */

    /* Reset output configuration */
    config.output = TINYAI_LOG_OUTPUT_CONSOLE;
    tinyai_configure_logging(&config);

    printf("PASS\n");
}

/* Main function */
int main()
{
    printf("Running TinyAI logging system tests...\n");

    /* Run tests */
    test_basic_logging();
    test_custom_handler();
    test_log_file_output();
    test_log_formatting();
    test_log_rotation();
    test_conditional_logging();

    /* Shut down logging system */
    tinyai_logging_shutdown();

    printf("All logging tests passed!\n");
    return 0;
}