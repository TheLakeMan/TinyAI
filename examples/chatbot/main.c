/**
 * @file main.c
 * @brief Main program for the TinyAI memory-constrained chatbot example
 */

#include "../../core/io.h"
#include "chat_model.h"
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#define COLOR_RESET "\033[0m"
#define COLOR_BOLD "\033[1m"
#define COLOR_RED "\033[31m"
#define COLOR_GREEN "\033[32m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_BLUE "\033[34m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_CYAN "\033[36m"
#else
#include <unistd.h>
#define COLOR_RESET "\033[0m"
#define COLOR_BOLD "\033[1m"
#define COLOR_RED "\033[31m"
#define COLOR_GREEN "\033[32m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_BLUE "\033[34m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_CYAN "\033[36m"
#endif

/* Global variables */
static volatile int       g_running = 1;    /* Control flag for main loop */
static TinyAIChatSession *g_session = NULL; /* Global session for signal handler */

/* Maximum input length */
#define MAX_INPUT_LENGTH 4096

/* Maximum path length */
#define MAX_PATH_LENGTH 512

/* Buffer size for reading input */
#define READ_BUFFER_SIZE 1024

/* Default save history file name */
#define DEFAULT_HISTORY_FILE "chat_history.json"

/* Signal handler to handle Ctrl+C */
void signalHandler(int signal)
{
    if (signal == SIGINT) {
        printf("\n%sInterrupted, exiting gracefully...%s\n", COLOR_YELLOW, COLOR_RESET);
        g_running = 0;
    }
}

/* Read multiline input from the console */
char *readMultilineInput()
{
    char   buffer[READ_BUFFER_SIZE];
    char  *input       = NULL;
    size_t inputLength = 0;
    size_t bufferLength;

    printf("%s>>> %s", COLOR_GREEN, COLOR_RESET);
    fflush(stdout);

    /* Read until empty line */
    while (fgets(buffer, READ_BUFFER_SIZE, stdin) != NULL) {
        bufferLength = strlen(buffer);

        /* Check for empty line (just newline) to end input */
        if (bufferLength == 1 && buffer[0] == '\n') {
            break;
        }

        /* Allocate or reallocate the input buffer */
        if (input == NULL) {
            input = (char *)malloc(bufferLength + 1);
            if (input == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                return NULL;
            }
            input[0] = '\0'; /* Ensure null-terminated */
        }
        else {
            char *newInput = (char *)realloc(input, inputLength + bufferLength + 1);
            if (newInput == NULL) {
                fprintf(stderr, "Memory reallocation failed\n");
                free(input);
                return NULL;
            }
            input = newInput;
        }

        /* Append the buffer to the input */
        strcpy(input + inputLength, buffer);
        inputLength += bufferLength;

        /* Show continuation prompt */
        printf("%s... %s", COLOR_GREEN, COLOR_RESET);
        fflush(stdout);
    }

    return input;
}

/* Token streaming callback */
bool streamingCallback(const char *token, bool is_partial, void *user_data)
{
    /* Simply print the token to stdout */
    printf("%s", token);
    fflush(stdout);

    /* Continue generation */
    return true;
}

/* Print usage instructions */
void printUsage(const char *progname)
{
    printf("TinyAI On-Device Chatbot Example\n\n");
    printf("Usage: %s [options]\n\n", progname);
    printf("Options:\n");
    printf("  --model <file>         Path to model structure file\n");
    printf("  --weights <file>       Path to model weights file\n");
    printf("  --tokenizer <file>     Path to tokenizer vocabulary file\n");
    printf("  --memory-limit <MB>    Maximum memory usage in MB (default: 16)\n");
    printf("  --max-tokens <n>       Maximum tokens in response (default: 100)\n");
    printf("  --temperature <value>  Sampling temperature (default: 0.7)\n");
    printf("  --system-prompt <file> File containing system prompt/instructions\n");
    printf("  --load-history <file>  Load conversation history from file\n");
    printf("  --save-history <file>  Save conversation history to file on exit\n");
    printf("  --no-stream            Disable streaming generation\n");
    printf("  --quantized            Use 4-bit quantization (default: enabled)\n");
    printf("  --no-quantize          Disable quantization\n");
    printf("  --simd                 Enable SIMD acceleration\n");
    printf("  --no-simd              Disable SIMD acceleration\n");
    printf("  --help                 Show this help message\n");
    printf("\nEnter your messages and press Enter twice (empty line) to submit.\n");
    printf("Type 'exit', 'quit', or press Ctrl+C to exit.\n");
}

/* Print memory usage statistics */
void printMemoryUsage(TinyAIChatSession *session)
{
    size_t modelMem, historyMem, totalMem;

    if (tinyaiChatGetMemoryUsage(session, &modelMem, &historyMem, &totalMem)) {
        printf("\n%sMemory Usage:%s\n", COLOR_BOLD, COLOR_RESET);
        printf("  Model:    %6.2f MB\n", modelMem / (1024.0 * 1024.0));
        printf("  History:  %6.2f MB\n", historyMem / (1024.0 * 1024.0));
        printf("  Total:    %6.2f MB\n", totalMem / (1024.0 * 1024.0));
    }
}

/* Main function */
int main(int argc, char *argv[])
{
    /* Default settings */
    const char *model_path         = NULL;
    const char *weights_path       = NULL;
    const char *tokenizer_path     = NULL;
    const char *system_prompt_file = NULL;
    const char *load_history_file  = NULL;
    const char *save_history_file  = NULL;
    int         memory_limit_mb    = 16;
    int         max_tokens         = 100;
    float       temperature        = 0.7f;
    float       top_p              = 0.9f;
    bool        use_streaming      = true;
    bool        use_quantization   = true;
    bool        use_simd           = true;

    /* Parse command line arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        }
        else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            weights_path = argv[++i];
        }
        else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            tokenizer_path = argv[++i];
        }
        else if (strcmp(argv[i], "--memory-limit") == 0 && i + 1 < argc) {
            memory_limit_mb = atoi(argv[++i]);
            if (memory_limit_mb <= 0) {
                fprintf(stderr, "Invalid memory limit: %s\n", argv[i]);
                return 1;
            }
        }
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
            if (max_tokens <= 0) {
                fprintf(stderr, "Invalid max tokens: %s\n", argv[i]);
                return 1;
            }
        }
        else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            temperature = (float)atof(argv[++i]);
            if (temperature < 0.0f || temperature > 1.5f) {
                fprintf(stderr, "Invalid temperature (must be between 0.0 and 1.5): %s\n", argv[i]);
                return 1;
            }
        }
        else if (strcmp(argv[i], "--system-prompt") == 0 && i + 1 < argc) {
            system_prompt_file = argv[++i];
        }
        else if (strcmp(argv[i], "--load-history") == 0 && i + 1 < argc) {
            load_history_file = argv[++i];
        }
        else if (strcmp(argv[i], "--save-history") == 0 && i + 1 < argc) {
            save_history_file = argv[++i];
        }
        else if (strcmp(argv[i], "--save-history") == 0) {
            save_history_file = DEFAULT_HISTORY_FILE;
        }
        else if (strcmp(argv[i], "--no-stream") == 0) {
            use_streaming = false;
        }
        else if (strcmp(argv[i], "--quantized") == 0) {
            use_quantization = true;
        }
        else if (strcmp(argv[i], "--no-quantize") == 0) {
            use_quantization = false;
        }
        else if (strcmp(argv[i], "--simd") == 0) {
            use_simd = true;
        }
        else if (strcmp(argv[i], "--no-simd") == 0) {
            use_simd = false;
        }
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    /* Check for required arguments */
    if (!model_path || !weights_path || !tokenizer_path) {
        fprintf(stderr, "Error: Model, weights, and tokenizer paths are required\n");
        printUsage(argv[0]);
        return 1;
    }

    /* Setup signal handler */
    signal(SIGINT, signalHandler);

    /* Initialize chat session */
    printf("Initializing chatbot...\n");

    TinyAIChatConfig config;
    memset(&config, 0, sizeof(TinyAIChatConfig));
    config.modelPath        = model_path;
    config.weightsPath      = weights_path;
    config.tokenizerPath    = tokenizer_path;
    config.memoryLimitMB    = memory_limit_mb;
    config.maxContextTokens = memory_limit_mb * 1024 / 2; /* Rough estimate: 2 bytes per token */
    config.maxTokens        = max_tokens;
    config.temperature      = temperature;
    config.topP             = top_p;
    config.useQuantization  = use_quantization;
    config.useSIMD          = use_simd;

    /* Create session */
    clock_t            start   = clock();
    TinyAIChatSession *session = tinyaiChatSessionCreate(&config);

    if (!session) {
        fprintf(stderr, "Failed to create chat session\n");
        return 1;
    }

    /* Store in global for signal handler */
    g_session = session;

    clock_t end       = clock();
    double  init_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Initialization completed in %.2f seconds\n", init_time);

    /* Add system prompt if provided */
    if (system_prompt_file) {
        char *system_prompt = tinyaiReadTextFile(system_prompt_file);
        if (system_prompt) {
            tinyaiChatAddMessage(session, TINYAI_ROLE_SYSTEM, system_prompt);
            printf("Added system prompt from %s\n", system_prompt_file);
            free(system_prompt);
        }
        else {
            fprintf(stderr, "Failed to read system prompt file: %s\n", system_prompt_file);
        }
    }

    /* Load chat history if provided */
    if (load_history_file) {
        if (tinyaiChatLoadHistory(session, load_history_file)) {
            int msg_count = tinyaiChatGetMessageCount(session);
            printf("Loaded %d messages from %s\n", msg_count, load_history_file);
        }
        else {
            fprintf(stderr, "Failed to load chat history from: %s\n", load_history_file);
        }
    }

    /* Display initial memory usage */
    printMemoryUsage(session);

    /* Welcome message */
    printf("\n%s==========================================%s\n", COLOR_BOLD, COLOR_RESET);
    printf("%s      TinyAI Memory-Constrained Chatbot      %s\n", COLOR_BOLD, COLOR_RESET);
    printf("%s==========================================%s\n\n", COLOR_BOLD, COLOR_RESET);
    printf("Enter your messages and press Enter twice (empty line) to submit.\n");
    printf("Type 'exit', 'quit', or press Ctrl+C to exit.\n\n");

    /* Main conversation loop */
    while (g_running) {
        /* Read user input */
        char *input = readMultilineInput();

        /* Check for input */
        if (!input || input[0] == '\0') {
            free(input);
            continue;
        }

        /* Check for exit commands */
        if (strcmp(input, "exit\n") == 0 || strcmp(input, "quit\n") == 0) {
            free(input);
            break;
        }

        /* Add user message to history */
        tinyaiChatAddMessage(session, TINYAI_ROLE_USER, input);

        /* Get model response */
        printf("\n%s%s: %s", COLOR_BOLD, "Assistant", COLOR_RESET);
        fflush(stdout);

        start = clock();

        /* Generate response with or without streaming */
        char *response;
        if (use_streaming) {
            response = tinyaiChatGenerateResponse(session, streamingCallback, NULL);
        }
        else {
            response = tinyaiChatGenerateResponse(session, NULL, NULL);
            if (response) {
                printf("%s", response);
            }
            else {
                printf("%sError: Failed to generate response%s", COLOR_RED, COLOR_RESET);
            }
        }

        end                    = clock();
        double generation_time = ((double)(end - start)) / CLOCKS_PER_SEC;

        /* Free the response (it's already added to history by the API) */
        free(response);

        /* Free user input */
        free(input);

        /* Print generation statistics */
        printf("\n\n%s[Generated in %.2f seconds]%s\n\n", COLOR_YELLOW, generation_time,
               COLOR_RESET);
    }

    /* Save chat history if requested */
    if (save_history_file) {
        if (tinyaiChatSaveHistory(session, save_history_file)) {
            printf("Chat history saved to %s\n", save_history_file);
        }
        else {
            fprintf(stderr, "Failed to save chat history to %s\n", save_history_file);
        }
    }

    /* Final memory usage */
    printMemoryUsage(session);

    /* Clean up */
    tinyaiChatSessionFree(session);
    g_session = NULL;

    printf("Goodbye!\n");

    return 0;
}
