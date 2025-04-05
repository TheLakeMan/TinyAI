/**
 * TinyAI Command Line Interface Implementation
 */

#include "cli.h"
#include "../core/config.h"           // Potentially needed for config command
#include "../core/io.h"               // For file operations (loading models etc.)
#include "../core/memory.h"           // For memory allocation
#include "../models/text/generate.h"  // For model/tokenizer types
#include "../models/text/tokenizer.h" // For tokenizer type

#include <ctype.h> // For isspace
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Define TinyAI Version (replace with actual versioning later)
#define TINYAI_VERSION "0.1.0-alpha"

/* ----------------- Internal State ----------------- */

static TinyAICommand g_commands[TINYAI_CLI_MAX_COMMANDS];
static int           g_commandCount = 0;

/* ----------------- Helper Functions ----------------- */

// Simple command line parsing into argc/argv
// Note: Modifies the input string by inserting null terminators!
static int parseCommandLine(char *line, char **argv, int maxArgs)
{
    int   argc       = 0;
    char *current    = line;
    char *tokenStart = NULL;
    int   inToken    = 0;
    int   inQuotes   = 0;

    while (*current != '\0' && argc < maxArgs - 1) {
        if (*current == '"') {
            inQuotes = !inQuotes;
            if (!inToken) {
                tokenStart = current + 1; // Start token after quote
                inToken    = 1;
            }
            else if (!inQuotes) {    // Closing quote
                *current     = '\0'; // Terminate token
                argv[argc++] = tokenStart;
                inToken      = 0;
            }
        }
        else if (isspace((unsigned char)*current)) {
            if (inToken && !inQuotes) {
                *current     = '\0'; // Terminate token
                argv[argc++] = tokenStart;
                inToken      = 0;
            }
        }
        else {
            if (!inToken) {
                tokenStart = current;
                inToken    = 1;
            }
        }
        current++;
    }

    // Handle the last token if the line doesn't end with whitespace
    if (inToken) {
        // If still in quotes at the end, it's an error or unterminated string
        if (inQuotes) {
            fprintf(stderr, "Error: Unterminated quote in command.\n");
            return -1; // Indicate parse error
        }
        // No need to terminate, already points within the original string which is null-terminated
        argv[argc++] = tokenStart;
    }

    argv[argc] = NULL; // Null-terminate the argv array
    return argc;
}

// Find a command by name
static TinyAICommand *findCommand(const char *name)
{
    for (int i = 0; i < g_commandCount; ++i) {
        if (strcmp(g_commands[i].name, name) == 0) {
            return &g_commands[i];
        }
    }
    return NULL;
}

/* ----------------- API Functions Implementation ----------------- */

int tinyaiCLIInit(TinyAICLIContext *context)
{
    if (!context)
        return TINYAI_CLI_EXIT_ERROR;

    // Initialize context defaults
    memset(context, 0, sizeof(TinyAICLIContext));

    // CLI state
    context->interactive = 0; // Default to non-interactive
    context->verbose     = 0;

    // Generation parameters
    context->params.maxTokens   = 50; // Use maxTokens
    context->params.temperature = 0.7f;
    context->params.topK        = 50;
    context->params.topP        = 0.9f;

    // MCP and hybrid initialization
    context->mcpClient   = NULL;
    context->hybridGen   = NULL;
    context->useHybrid   = 0; // Default to local-only generation
    context->forceRemote = 0;
    context->forceLocal  = 0;

    g_commandCount = 0;

    // Register built-in commands
    tinyaiCLIRegisterCommand("help", "Show help information.", "help [command]", tinyaiCommandHelp);
    tinyaiCLIRegisterCommand("version", "Show TinyAI version.", "version", tinyaiCommandVersion);
    tinyaiCLIRegisterCommand("generate", "Generate text using the loaded model.",
                             "generate <prompt>", tinyaiCommandGenerate);
    tinyaiCLIRegisterCommand("tokenize", "Tokenize input text.", "tokenize <text>",
                             tinyaiCommandTokenize);
    tinyaiCLIRegisterCommand("model", "Load or inspect the model.", "model load <path> | info",
                             tinyaiCommandModel);
    tinyaiCLIRegisterCommand("config", "Set or view configuration parameters.",
                             "config [param] [value]", tinyaiCommandConfig);
    tinyaiCLIRegisterCommand("mcp", "Connect to or control MCP server.",
                             "mcp connect <url> | disconnect | status", tinyaiCommandMcp);
    tinyaiCLIRegisterCommand("hybrid", "Control hybrid local/remote execution mode.",
                             "hybrid on | off | status | force-local | force-remote",
                             tinyaiCommandHybrid);
    tinyaiCLIRegisterCommand("exit", "Exit the interactive shell.", "exit", tinyaiCommandExit);
    tinyaiCLIRegisterCommand("quit", "Exit the interactive shell.", "quit",
                             tinyaiCommandExit); // Alias for exit

    // Initialize subsystems if needed (though likely done elsewhere)
    // tinyaiIOInit();
    // tinyaiMemTrackInit(); // If tracking enabled

    return TINYAI_CLI_EXIT_SUCCESS;
}

void tinyaiCLICleanup(TinyAICLIContext *context)
{
    if (!context)
        return;

    // Free context resources - paths
    if (context->modelPath)
        free(context->modelPath);
    if (context->tokenizerPath)
        free(context->tokenizerPath);
    if (context->mcpServerUrl)
        free(context->mcpServerUrl);

    // Clean up model and tokenizer
    if (context->model)
        tinyaiDestroyModel(context->model);
    if (context->tokenizer)
        tinyaiDestroyTokenizer(context->tokenizer);

    // Clean up MCP client and hybrid generation
    if (context->hybridGen)
        tinyaiDestroyHybridGenerate(context->hybridGen);
    if (context->mcpClient) {
        if (tinyaiMcpGetConnectionState(context->mcpClient) == TINYAI_MCP_CONNECTED)
            tinyaiMcpDisconnect(context->mcpClient);
        tinyaiMcpDestroyClient(context->mcpClient);
    }

    // Reset command registry (optional, as it's static)
    g_commandCount = 0;

    // Cleanup subsystems if initialized here
    // tinyaiIOCleanup();
    // tinyaiMemTrackCleanup(); // If tracking enabled
}

int tinyaiCLIParseArgs(TinyAICLIContext *context, int argc, char **argv)
{
    // Basic argument parsing (replace with getopt or similar later if needed)
    // This function is primarily for parsing arguments passed to the program itself,
    // not for parsing commands entered in the interactive shell.

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--interactive") == 0) {
            context->interactive = 1;
        }
        else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            context->verbose++;
        }
        else if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            // Use TINYAI_FREE/MALLOC if integrated
            if (context->modelPath)
                free(context->modelPath);
            context->modelPath = _strdup(argv[++i]); // Use _strdup
            if (!context->modelPath)
                return TINYAI_CLI_EXIT_ERROR; // Memory error
        }
        else if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--tokenizer") == 0) &&
                 i + 1 < argc) {
            if (context->tokenizerPath)
                free(context->tokenizerPath);
            context->tokenizerPath = _strdup(argv[++i]); // Use _strdup
            if (!context->tokenizerPath)
                return TINYAI_CLI_EXIT_ERROR; // Memory error
        }
        else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            context->params.temperature = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--topk") == 0 && i + 1 < argc) {
            context->params.topK = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--topp") == 0 && i + 1 < argc) {
            context->params.topP = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--maxlen") == 0 && i + 1 < argc) {
            context->params.maxTokens = atoi(argv[++i]); // Use maxTokens
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            tinyaiCLIPrintHelp(context, NULL);
            return TINYAI_CLI_EXIT_QUIT; // Exit after showing help
        }
        else {
            // Assume remaining arguments might be a command and its args for non-interactive mode
            // Or potentially a prompt for direct generation
            break; // Stop parsing options here
        }
    }
    return TINYAI_CLI_EXIT_SUCCESS;
}

int tinyaiCLIRun(TinyAICLIContext *context, int argc, char **argv)
{
    int parseResult = tinyaiCLIParseArgs(context, argc, argv);
    if (parseResult != TINYAI_CLI_EXIT_SUCCESS) {
        return parseResult; // Return error or quit code from parsing
    }

    // Load model/tokenizer if paths were provided
    if (context->modelPath) {
        // Call model loading function (implementation needed)
        printf("Info: Attempting to load model from %s\n", context->modelPath);
        // context->model = tinyaiLoadModel(context->modelPath); // Example
        // if (!context->model) { fprintf(stderr, "Error loading model.\n"); return
        // TINYAI_CLI_EXIT_ERROR; }
    }
    if (context->tokenizerPath) {
        printf("Info: Attempting to load tokenizer from %s\n", context->tokenizerPath);
        // context->tokenizer = tinyaiLoadTokenizer(context->tokenizerPath); // Example
        // if (!context->tokenizer) { fprintf(stderr, "Error loading tokenizer.\n"); return
        // TINYAI_CLI_EXIT_ERROR; }
    }

    if (context->interactive) {
        return tinyaiCLIRunShell(context);
    }
    else {
        // Non-interactive mode: try to execute a command from remaining args
        int firstCmdArgIndex = 1; // Find where options stopped
        for (; firstCmdArgIndex < argc; ++firstCmdArgIndex) {
            // Simple check: stop at first arg not starting with '-'? Needs refinement.
            if (argv[firstCmdArgIndex][0] != '-')
                break;
            // Skip option arguments
            if ((strcmp(argv[firstCmdArgIndex], "-m") == 0 ||
                 strcmp(argv[firstCmdArgIndex], "--model") == 0 ||
                 strcmp(argv[firstCmdArgIndex], "-t") == 0 ||
                 strcmp(argv[firstCmdArgIndex], "--tokenizer") == 0 ||
                 strcmp(argv[firstCmdArgIndex], "--temp") == 0 ||
                 strcmp(argv[firstCmdArgIndex], "--topk") == 0 ||
                 strcmp(argv[firstCmdArgIndex], "--topp") == 0 ||
                 strcmp(argv[firstCmdArgIndex], "--maxlen") == 0) &&
                firstCmdArgIndex + 1 < argc) {
                firstCmdArgIndex++;
            }
        }

        if (firstCmdArgIndex < argc) {
            // Treat remaining args as a command
            TinyAICommand *cmd = findCommand(argv[firstCmdArgIndex]);
            if (cmd) {
                return cmd->handler(argc - firstCmdArgIndex, argv + firstCmdArgIndex, context);
            }
            else {
                // If no command matches, maybe treat as a prompt for generate?
                fprintf(stderr, "Error: Unknown command '%s' in non-interactive mode.\n",
                        argv[firstCmdArgIndex]);
                tinyaiCLIPrintHelp(context, NULL);
                return TINYAI_CLI_EXIT_ERROR;
            }
        }
        else {
            // No command provided in non-interactive mode, show help or default action?
            fprintf(stderr, "Error: No command specified for non-interactive mode.\n");
            tinyaiCLIPrintHelp(context, NULL);
            return TINYAI_CLI_EXIT_ERROR;
        }
    }
}

int tinyaiCLIRunShell(TinyAICLIContext *context)
{
    char line[TINYAI_CLI_MAX_COMMAND_LENGTH];
    int  exitCode = TINYAI_CLI_EXIT_SUCCESS;

    printf("TinyAI Interactive Shell (v%s)\n", TINYAI_VERSION);
    printf("Type 'help' for available commands, 'exit' or 'quit' to leave.\n");

    while (exitCode != TINYAI_CLI_EXIT_QUIT) {
        printf("> ");
        fflush(stdout); // Ensure prompt is displayed

        if (fgets(line, sizeof(line), stdin) == NULL) {
            // End of input (e.g., Ctrl+D)
            printf("\nExiting.\n");
            exitCode = TINYAI_CLI_EXIT_QUIT;
            break;
        }

        // Remove trailing newline
        line[strcspn(line, "\r\n")] = 0;

        // Skip empty lines
        char *trimmed_line = line;
        while (isspace((unsigned char)*trimmed_line))
            trimmed_line++;
        if (*trimmed_line == '\0') {
            continue;
        }

        exitCode = tinyaiCLIProcessCommand(context, trimmed_line);

        if (exitCode != TINYAI_CLI_EXIT_SUCCESS && exitCode != TINYAI_CLI_EXIT_QUIT) {
            // Print error message based on exit code?
            fprintf(stderr, "Command failed with exit code %d\n", exitCode);
            // Reset exit code to continue shell unless it was QUIT
            exitCode = TINYAI_CLI_EXIT_SUCCESS;
        }
    }

    return exitCode == TINYAI_CLI_EXIT_QUIT ? TINYAI_CLI_EXIT_SUCCESS
                                            : exitCode; // Return 0 on normal quit
}

int tinyaiCLIProcessCommand(TinyAICLIContext *context, const char *command_const)
{
    char  command[TINYAI_CLI_MAX_COMMAND_LENGTH];
    char *argv[TINYAI_CLI_MAX_ARGS];

    // Make a mutable copy for parsing
    // Use safer strncpy_s with truncation
    strncpy_s(command, sizeof(command), command_const, _TRUNCATE);
    // command[sizeof(command) - 1] = '\0'; // strncpy_s guarantees null termination if space allows

    int argc = parseCommandLine(command, argv, TINYAI_CLI_MAX_ARGS);

    if (argc <= 0) {
        return TINYAI_CLI_EXIT_ERROR; // Parsing error or empty command
    }

    TinyAICommand *cmd = findCommand(argv[0]);
    if (cmd) {
        return cmd->handler(argc, argv, context);
    }
    else {
        fprintf(stderr, "Error: Unknown command '%s'. Type 'help' for available commands.\n",
                argv[0]);
        return TINYAI_CLI_EXIT_ERROR;
    }
}

int tinyaiCLIRegisterCommand(const char *name, const char *description, const char *usage,
                             TinyAICommandHandler handler)
{
    if (g_commandCount >= TINYAI_CLI_MAX_COMMANDS) {
        fprintf(stderr, "Error: Maximum number of commands reached (%d).\n",
                TINYAI_CLI_MAX_COMMANDS);
        return 1; // Error
    }
    if (!name || !description || !usage || !handler) {
        fprintf(stderr, "Error: Invalid arguments for registering command '%s'.\n",
                name ? name : "(null)");
        return 1; // Error
    }
    if (findCommand(name)) {
        fprintf(stderr, "Warning: Command '%s' already registered. Overwriting.\n", name);
        // Allow overwriting for now, could return error instead
    }

    g_commands[g_commandCount].name        = name;
    g_commands[g_commandCount].description = description;
    g_commands[g_commandCount].usage       = usage;
    g_commands[g_commandCount].handler     = handler;
    g_commandCount++;

    return 0; // Success
}

int tinyaiCLIPrintHelp(TinyAICLIContext *context, const char *commandName)
{
    (void)context; // Context might be used later (e.g., to show context-specific help)

    if (commandName) {
        TinyAICommand *cmd = findCommand(commandName);
        if (cmd) {
            printf("Usage: %s\n\n", cmd->usage);
            printf("%s\n", cmd->description);
        }
        else {
            fprintf(stderr, "Error: Unknown command '%s'.\n", commandName);
            return TINYAI_CLI_EXIT_ERROR;
        }
    }
    else {
        printf("TinyAI CLI v%s\n", TINYAI_VERSION);
        printf("Available commands:\n");
        for (int i = 0; i < g_commandCount; ++i) {
            printf("  %-15s %s\n", g_commands[i].name, g_commands[i].description);
        }
        printf("\nType 'help <command>' for more information on a specific command.\n");
    }
    return TINYAI_CLI_EXIT_SUCCESS;
}

/* ----------------- Built-in Command Handlers (Stubs) ----------------- */

int tinyaiCommandHelp(int argc, char **argv, void *context)
{
    if (argc > 2) {
        fprintf(stderr, "Usage: help [command]\n");
        return TINYAI_CLI_EXIT_ERROR;
    }
    return tinyaiCLIPrintHelp((TinyAICLIContext *)context, (argc == 2) ? argv[1] : NULL);
}

int tinyaiCommandVersion(int argc, char **argv, void *context)
{
    (void)argc;
    (void)argv;
    (void)context; // Unused
    printf("TinyAI version %s\n", TINYAI_VERSION);
    return TINYAI_CLI_EXIT_SUCCESS;
}

int tinyaiCommandGenerate(int argc, char **argv, void *context)
{
    TinyAICLIContext *ctx = (TinyAICLIContext *)context;
    if (argc < 2) {
        fprintf(stderr, "Usage: generate <prompt> [max_tokens] [temperature] [sampling_method]\n");
        fprintf(stderr, "   Sampling methods: greedy, temp, topk, topp\n");
        return TINYAI_CLI_EXIT_ERROR;
    }
    if (!ctx->model || !ctx->tokenizer) {
        fprintf(stderr,
                "Error: Model and tokenizer must be loaded first (use 'model load ...').\n");
        return TINYAI_CLI_EXIT_ERROR;
    }

    // Parse arguments
    const char *prompt = argv[1];

    // Initialize generation parameters from context's defaults
    TinyAIGenerationParams params = ctx->params;

    // Override with command-line arguments if provided
    if (argc > 2) {
        params.maxTokens = atoi(argv[2]);
    }

    if (argc > 3) {
        params.temperature = (float)atof(argv[3]);
    }

    if (argc > 4) {
        const char *samplingMethod = argv[4];
        if (strcmp(samplingMethod, "greedy") == 0) {
            params.samplingMethod = TINYAI_SAMPLING_GREEDY;
        }
        else if (strcmp(samplingMethod, "temp") == 0) {
            params.samplingMethod = TINYAI_SAMPLING_TEMPERATURE;
        }
        else if (strcmp(samplingMethod, "topk") == 0) {
            params.samplingMethod = TINYAI_SAMPLING_TOP_K;
        }
        else if (strcmp(samplingMethod, "topp") == 0) {
            params.samplingMethod = TINYAI_SAMPLING_TOP_P;
        }
        else {
            fprintf(stderr, "Unknown sampling method: %s\n", samplingMethod);
            return TINYAI_CLI_EXIT_ERROR;
        }
    }

    printf("Generating text for prompt: \"%s\"\n", prompt);
    printf("Parameters: max_tokens=%d, temp=%.2f, top_k=%d, top_p=%.2f, sampling=%d\n",
           params.maxTokens, params.temperature, params.topK, params.topP, params.samplingMethod);

    // Tokenize prompt
    int promptTokens[1024]; // Buffer for prompt tokens
    int promptLength = tinyaiEncodeText(ctx->tokenizer, prompt, promptTokens, 1024);

    if (promptLength <= 0) {
        fprintf(stderr, "Error: Failed to tokenize prompt.\n");
        return TINYAI_CLI_EXIT_ERROR;
    }

    printf("Prompt tokenized into %d tokens\n", promptLength);

    // Set prompt in generation parameters
    params.promptTokens = promptTokens;
    params.promptLength = promptLength;
    params.seed         = (uint32_t)time(NULL); // Random seed based on current time

    // Generate text
    int outputTokens[4096]; // Large buffer for generated tokens
    int outputLength;

    // Use hybrid generation if enabled, otherwise use local generation
    if (ctx->useHybrid && ctx->hybridGen) {
        printf("Using hybrid generation mode...\n");

        // Apply force flags if set
        if (ctx->forceLocal) {
            printf("Forcing local execution for this generation.\n");
            tinyaiHybridGenerateForceMode(ctx->hybridGen, false);
        }
        else if (ctx->forceRemote) {
            printf("Forcing remote execution for this generation.\n");
            tinyaiHybridGenerateForceMode(ctx->hybridGen, true);
        }

        // Generate text using hybrid context
        outputLength = tinyaiHybridGenerateText(ctx->hybridGen, &params, outputTokens, 4096);

        // Reset force flags after generation
        ctx->forceLocal  = 0;
        ctx->forceRemote = 0;

        // Display execution mode used
        if (outputLength > 0) {
            printf("Generation used %s execution.\n",
                   tinyaiHybridGenerateUsedRemote(ctx->hybridGen) ? "remote" : "local");

            // Get generation stats
            double localTime, remoteTime, tokensPerSec;
            tinyaiHybridGenerateGetStats(ctx->hybridGen, &localTime, &remoteTime, &tokensPerSec);

            if (tinyaiHybridGenerateUsedRemote(ctx->hybridGen)) {
                printf("Remote execution time: %.2f ms\n", remoteTime);
            }
            else {
                printf("Local execution time: %.2f ms\n", localTime);
            }
            printf("Tokens per second: %.2f\n", tokensPerSec);
        }
    }
    else {
        // Use local generation with direct model access
        outputLength = tinyaiGenerateText(ctx->model, &params, outputTokens, 4096);
    }

    if (outputLength <= 0) {
        fprintf(stderr, "Error: Text generation failed.\n");
        return TINYAI_CLI_EXIT_ERROR;
    }

    // Decode generated tokens to text
    char outputText[16384] = {0}; // Large buffer for decoded text
    int  textLength = tinyaiDecodeTokens(ctx->tokenizer, outputTokens, outputLength, outputText,
                                         sizeof(outputText));

    if (textLength <= 0) {
        fprintf(stderr, "Error: Failed to decode generated tokens.\n");
        return TINYAI_CLI_EXIT_ERROR;
    }

    // Print the generated text
    printf("\n--- Generated Text ---\n%s\n---------------------\n", outputText);

    return TINYAI_CLI_EXIT_SUCCESS;
}

int tinyaiCommandTokenize(int argc, char **argv, void *context)
{
    TinyAICLIContext *ctx = (TinyAICLIContext *)context;
    if (argc != 2) {
        fprintf(stderr, "Usage: tokenize <text>\n");
        return TINYAI_CLI_EXIT_ERROR;
    }
    if (!ctx->tokenizer) {
        fprintf(stderr, "Error: Tokenizer must be loaded first (use 'model load ...').\n");
        return TINYAI_CLI_EXIT_ERROR;
    }

    const char *text = argv[1];
    printf("Tokenizing text: \"%s\"\n", text);

    // Tokenize the input text
    int tokens[1024] = {0}; // Buffer for tokens
    int tokenCount   = tinyaiEncodeText(ctx->tokenizer, text, tokens, 1024);

    if (tokenCount <= 0) {
        fprintf(stderr, "Error during tokenization.\n");
        return TINYAI_CLI_EXIT_ERROR;
    }

    // Display the tokens
    printf("Tokens (%d): [", tokenCount);
    for (int i = 0; i < tokenCount; ++i) {
        // Print token ID and string representation
        const char *tokenStr = tinyaiGetTokenString(ctx->tokenizer, tokens[i]);
        printf("%d=\"%s\"%s", tokens[i], tokenStr ? tokenStr : "<unknown>",
               (i == tokenCount - 1) ? "" : ", ");
    }
    printf("]\n");

    // Test decoding (should match original input, modulo special tokens handling)
    char decoded[4096] = {0};
    int  decodedLength =
        tinyaiDecodeTokens(ctx->tokenizer, tokens, tokenCount, decoded, sizeof(decoded));

    if (decodedLength > 0) {
        printf("Decoded back: \"%s\"\n", decoded);
    }
    else {
        fprintf(stderr, "Warning: Could not decode tokens back to text.\n");
    }

    return TINYAI_CLI_EXIT_SUCCESS;
}

int tinyaiCommandModel(int argc, char **argv, void *context)
{
    TinyAICLIContext *ctx = (TinyAICLIContext *)context;
    if (argc < 2) {
        fprintf(stderr, "Usage: model <load <model_path> [tokenizer_path] | info | create-vocab "
                        "<corpus_file> <vocab_size> <output_path>>\n");
        return TINYAI_CLI_EXIT_ERROR;
    }

    if (strcmp(argv[1], "load") == 0) {
        if (argc < 3) {
            fprintf(stderr, "Usage: model load <model_path> [tokenizer_path]\n");
            return TINYAI_CLI_EXIT_ERROR;
        }
        const char *modelPath     = argv[2];
        const char *tokenizerPath = (argc > 3) ? argv[3] : NULL; // Optional tokenizer path

        // Free existing paths if any
        if (ctx->modelPath)
            free(ctx->modelPath);
        if (ctx->tokenizerPath)
            free(ctx->tokenizerPath);
        ctx->modelPath     = NULL;
        ctx->tokenizerPath = NULL;

        // Store new paths (use TINYAI_MALLOC if integrated)
        ctx->modelPath = _strdup(modelPath); // Use _strdup
        if (!ctx->modelPath)
            return TINYAI_CLI_EXIT_ERROR; // Memory error

        if (tokenizerPath) {
            ctx->tokenizerPath = _strdup(tokenizerPath); // Use _strdup
            if (!ctx->tokenizerPath) {
                free(ctx->modelPath);
                ctx->modelPath = NULL;
                return TINYAI_CLI_EXIT_ERROR;
            }
        }
        else {
            // Try to infer tokenizer path from model path (e.g., replace extension)
            // Basic example: replace .tmodel with .tok
            size_t      len = strlen(modelPath);
            const char *ext = tinyaiGetFileExt(modelPath);
            if (ext && len > strlen(ext)) {
                size_t basePathLen = len - strlen(ext);
                // Need basePathLen + "tok" + null terminator
                size_t inferredPathSize = basePathLen + 4;
                char  *inferredPath     = (char *)malloc(inferredPathSize);
                if (inferredPath) {
                    // Use safer string functions
                    strncpy_s(inferredPath, inferredPathSize, modelPath, basePathLen);
                    strcpy_s(inferredPath + basePathLen, inferredPathSize - basePathLen, "tok");
                    // Check if inferred path exists?
                    if (tinyaiFileExists(inferredPath) == 1) {
                        ctx->tokenizerPath = inferredPath;
                        printf("Info: Inferred tokenizer path: %s\n", ctx->tokenizerPath);
                    }
                    else {
                        printf("Warning: Could not infer tokenizer path from model path. Please "
                               "provide explicitly.\n");
                        free(inferredPath);
                    }
                }
            }
            if (!ctx->tokenizerPath) {
                fprintf(stderr, "Error: Tokenizer path not provided and could not be inferred.\n");
                free(ctx->modelPath);
                ctx->modelPath = NULL;
                return TINYAI_CLI_EXIT_ERROR;
            }
        }

        printf("Loading model from: %s\n", ctx->modelPath);
        printf("Loading tokenizer from: %s\n", ctx->tokenizerPath);

        // Load tokenizer first
        if (ctx->tokenizer) {
            tinyaiDestroyTokenizer(ctx->tokenizer); // Free previous tokenizer
            ctx->tokenizer = NULL;
        }

        ctx->tokenizer = tinyaiCreateTokenizer();
        if (!ctx->tokenizer) {
            fprintf(stderr, "Error: Failed to create tokenizer.\n");
            return TINYAI_CLI_EXIT_ERROR;
        }

        int result = tinyaiLoadVocabulary(ctx->tokenizer, ctx->tokenizerPath);
        if (result != 0) {
            fprintf(stderr, "Error: Failed to load vocabulary from %s.\n", ctx->tokenizerPath);
            tinyaiDestroyTokenizer(ctx->tokenizer);
            ctx->tokenizer = NULL;
            return TINYAI_CLI_EXIT_ERROR;
        }

        printf("Tokenizer loaded successfully with %d tokens.\n", ctx->tokenizer->tokenCount);

        // Try to load model
        if (ctx->model) {
            tinyaiDestroyModel(ctx->model); // Free previous model
            ctx->model = NULL;
        }

        ctx->model = tinyaiLoadModel(modelPath, modelPath, ctx->tokenizerPath);
        if (!ctx->model) {
            fprintf(stderr, "Warning: Could not load model from %s. Only tokenizer is available.\n",
                    modelPath);
            // Continue with just the tokenizer for now
        }
        else {
            printf("Model loaded successfully.\n");
        }
    }
    else if (strcmp(argv[1], "create-vocab") == 0) {
        // Subcommand to create a vocabulary from a text corpus
        if (argc < 5) {
            fprintf(stderr, "Usage: model create-vocab <corpus_file> <vocab_size> <output_path>\n");
            return TINYAI_CLI_EXIT_ERROR;
        }

        const char *corpusPath = argv[2];
        int         vocabSize  = atoi(argv[3]);
        const char *outputPath = argv[4];

        if (vocabSize <= 0 || vocabSize > 65536) {
            fprintf(stderr,
                    "Error: Invalid vocabulary size. Please use a value between 1 and 65536.\n");
            return TINYAI_CLI_EXIT_ERROR;
        }

        printf("Creating vocabulary with up to %d tokens from %s\n", vocabSize, corpusPath);

        // Load corpus file
        FILE *corpusFile = fopen(corpusPath, "r");
        if (!corpusFile) {
            fprintf(stderr, "Error: Could not open corpus file %s\n", corpusPath);
            return TINYAI_CLI_EXIT_ERROR;
        }

        // Determine file size
        fseek(corpusFile, 0, SEEK_END);
        long fileSize = ftell(corpusFile);
        fseek(corpusFile, 0, SEEK_SET);

        if (fileSize <= 0) {
            fprintf(stderr, "Error: Corpus file is empty or invalid.\n");
            fclose(corpusFile);
            return TINYAI_CLI_EXIT_ERROR;
        }

        // Allocate memory for corpus
        char *corpus = (char *)malloc(fileSize + 1);
        if (!corpus) {
            fprintf(stderr, "Error: Failed to allocate memory for corpus.\n");
            fclose(corpusFile);
            return TINYAI_CLI_EXIT_ERROR;
        }

        // Read corpus
        size_t bytesRead = fread(corpus, 1, fileSize, corpusFile);
        fclose(corpusFile);

        if (bytesRead <= 0) {
            fprintf(stderr, "Error: Failed to read corpus file.\n");
            free(corpus);
            return TINYAI_CLI_EXIT_ERROR;
        }

        corpus[bytesRead] = '\0'; // Ensure null termination

        // Create new tokenizer
        TinyAITokenizer *tokenizer = tinyaiCreateTokenizer();
        if (!tokenizer) {
            fprintf(stderr, "Error: Failed to create tokenizer.\n");
            free(corpus);
            return TINYAI_CLI_EXIT_ERROR;
        }

        // Build vocabulary
        int result = tinyaiCreateMinimalVocabulary(tokenizer, corpus, vocabSize);
        if (result != 0) {
            fprintf(stderr, "Error: Failed to create vocabulary.\n");
            tinyaiDestroyTokenizer(tokenizer);
            free(corpus);
            return TINYAI_CLI_EXIT_ERROR;
        }

        printf("Created vocabulary with %d tokens.\n", tokenizer->tokenCount);

        // Save vocabulary
        result = tinyaiSaveVocabulary(tokenizer, outputPath);
        if (result != 0) {
            fprintf(stderr, "Error: Failed to save vocabulary to %s.\n", outputPath);
            tinyaiDestroyTokenizer(tokenizer);
            free(corpus);
            return TINYAI_CLI_EXIT_ERROR;
        }

        printf("Vocabulary saved to %s.\n", outputPath);

        // Cleanup
        tinyaiDestroyTokenizer(tokenizer);
        free(corpus);
    }
    else if (strcmp(argv[1], "info") == 0) {
        printf("Current Model Path: %s\n", ctx->modelPath ? ctx->modelPath : "(None)");
        printf("Current Tokenizer Path: %s\n", ctx->tokenizerPath ? ctx->tokenizerPath : "(None)");
        printf("Model Loaded: %s\n", ctx->model ? "Yes" : "No");
        printf("Tokenizer Loaded: %s\n", ctx->tokenizer ? "Yes" : "No");

        if (ctx->tokenizer) {
            printf("Tokenizer Information:\n");
            printf("  Vocabulary Size: %d tokens\n", ctx->tokenizer->tokenCount);
            printf("  Special Tokens:\n");
            printf("    <unk> (ID %d): %s\n", TINYAI_TOKEN_UNKNOWN,
                   tinyaiGetTokenString(ctx->tokenizer, TINYAI_TOKEN_UNKNOWN));
            printf("    <bos> (ID %d): %s\n", TINYAI_TOKEN_BOS,
                   tinyaiGetTokenString(ctx->tokenizer, TINYAI_TOKEN_BOS));
            printf("    <eos> (ID %d): %s\n", TINYAI_TOKEN_EOS,
                   tinyaiGetTokenString(ctx->tokenizer, TINYAI_TOKEN_EOS));
            printf("    <pad> (ID %d): %s\n", TINYAI_TOKEN_PAD,
                   tinyaiGetTokenString(ctx->tokenizer, TINYAI_TOKEN_PAD));
        }

        if (ctx->model) {
            printf("Model Information:\n");
            printf("  Type: %s\n",
                   ctx->model->type == TINYAI_MODEL_TYPE_RNN
                       ? "RNN"
                       : (ctx->model->type == TINYAI_MODEL_TYPE_TRANSFORMER ? "Transformer"
                                                                            : "Unknown"));
            printf("  Hidden Size: %d\n", ctx->model->hiddenSize);
            printf("  Context Size: %d\n", ctx->model->contextSize);
            printf("  Layers: %d\n", ctx->model->layerCount);
        }
    }
    else {
        fprintf(stderr,
                "Error: Unknown model subcommand '%s'. Use 'load', 'create-vocab', or 'info'.\n",
                argv[1]);
        return TINYAI_CLI_EXIT_ERROR;
    }

    return TINYAI_CLI_EXIT_SUCCESS;
}

int tinyaiCommandConfig(int argc, char **argv, void *context)
{
    TinyAICLIContext *ctx = (TinyAICLIContext *)context;

    if (argc == 1) {
        // Print current config
        printf("Current Generation Parameters:\n");
        printf("  maxTokens    = %d\n", ctx->params.maxTokens); // Use maxTokens
        printf("  temperature  = %.2f\n", ctx->params.temperature);
        printf("  topK         = %d\n", ctx->params.topK);
        printf("  topP         = %.2f\n", ctx->params.topP);
        printf("Verbosity: %d\n", ctx->verbose);
        // Add other relevant config from context
    }
    else if (argc == 3) {
        // Set config parameter
        const char *param = argv[1];
        const char *value = argv[2];
        if (strcmp(param, "maxTokens") == 0) {                     // Use maxTokens
            ctx->params.maxTokens = atoi(value);                   // Use maxTokens
            printf("Set maxTokens = %d\n", ctx->params.maxTokens); // Use maxTokens
        }
        else if (strcmp(param, "temperature") == 0) {
            ctx->params.temperature = (float)atof(value);
            printf("Set temperature = %.2f\n", ctx->params.temperature);
        }
        else if (strcmp(param, "topK") == 0) {
            ctx->params.topK = atoi(value);
            printf("Set topK = %d\n", ctx->params.topK);
        }
        else if (strcmp(param, "topP") == 0) {
            ctx->params.topP = (float)atof(value);
            printf("Set topP = %.2f\n", ctx->params.topP);
        }
        else if (strcmp(param, "verbose") == 0) {
            ctx->verbose = atoi(value);
            printf("Set verbose = %d\n", ctx->verbose);
        }
        else {
            fprintf(stderr, "Error: Unknown configuration parameter '%s'.\n", param);
            return TINYAI_CLI_EXIT_ERROR;
        }
    }
    else {
        fprintf(stderr, "Usage: config [parameter] [value]\n");
        fprintf(stderr, "       config (to view current settings)\n");
        return TINYAI_CLI_EXIT_ERROR;
    }

    return TINYAI_CLI_EXIT_SUCCESS;
}

int tinyaiCommandExit(int argc, char **argv, void *context)
{
    (void)argc;
    (void)argv;
    (void)context; // Unused
    printf("Exiting TinyAI shell.\n");
    return TINYAI_CLI_EXIT_QUIT; // Special code to signal shell exit
}

/**
 * MCP command handler implementation
 */
int tinyaiCommandMcp(int argc, char **argv, void *context)
{
    TinyAICLIContext *ctx = (TinyAICLIContext *)context;

    if (argc < 2) {
        fprintf(stderr, "Usage: mcp <connect <url> | disconnect | status>\n");
        return TINYAI_CLI_EXIT_ERROR;
    }

    const char *subcmd = argv[1];

    if (strcmp(subcmd, "connect") == 0) {
        if (argc < 3) {
            fprintf(stderr, "Usage: mcp connect <url>\n");
            return TINYAI_CLI_EXIT_ERROR;
        }

        const char *url = argv[2];

        // Already connected? Disconnect first
        if (ctx->mcpClient) {
            printf("Disconnecting from existing MCP server...\n");
            tinyaiMcpDisconnect(ctx->mcpClient);
            tinyaiMcpDestroyClient(ctx->mcpClient);
            ctx->mcpClient = NULL;

            // Also destroy hybrid generation context if it exists
            if (ctx->hybridGen) {
                tinyaiDestroyHybridGenerate(ctx->hybridGen);
                ctx->hybridGen = NULL;
            }

            if (ctx->mcpServerUrl) {
                free(ctx->mcpServerUrl);
                ctx->mcpServerUrl = NULL;
            }
        }

        // Create MCP client config
        TinyAIMcpConfig config;
        tinyaiMcpGetDefaultConfig(&config);

        // Adjust config based on user preferences (could be from ctx)
        config.execPreference = TINYAI_EXEC_PREFER_LOCAL; // Default to local-first

        // Create client
        ctx->mcpClient = tinyaiMcpCreateClient(&config);
        if (!ctx->mcpClient) {
            fprintf(stderr, "Error: Failed to create MCP client.\n");
            return TINYAI_CLI_EXIT_ERROR;
        }

        // Connect to server
        printf("Connecting to MCP server at %s...\n", url);
        bool connected = tinyaiMcpConnect(ctx->mcpClient, url);

        if (!connected) {
            fprintf(stderr, "Error: Failed to connect to MCP server at %s.\n", url);
            tinyaiMcpDestroyClient(ctx->mcpClient);
            ctx->mcpClient = NULL;
            return TINYAI_CLI_EXIT_ERROR;
        }

        // Store server URL
        ctx->mcpServerUrl = _strdup(url);
        if (!ctx->mcpServerUrl) {
            fprintf(stderr, "Error: Memory allocation failure for server URL.\n");
            tinyaiMcpDisconnect(ctx->mcpClient);
            tinyaiMcpDestroyClient(ctx->mcpClient);
            ctx->mcpClient = NULL;
            return TINYAI_CLI_EXIT_ERROR;
        }

        // Create hybrid generation context if both model and MCP client are available
        if (ctx->model) {
            ctx->hybridGen = tinyaiCreateHybridGenerate(ctx->model, ctx->mcpClient);
            if (!ctx->hybridGen) {
                fprintf(stderr, "Warning: Failed to create hybrid generation context.\n");
                // Continue without hybrid generation
            }
            else {
                ctx->useHybrid = 1;
                printf("Hybrid generation mode enabled.\n");
            }
        }

        // Get server info
        TinyAIMcpServerInfo serverInfo;
        if (tinyaiMcpGetServerInfo(ctx->mcpClient, &serverInfo)) {
            printf("Connected to MCP server: %s (version %s)\n", serverInfo.serverName,
                   serverInfo.serverVersion);

            // Print capabilities
            printf("Server capabilities: %s\n", serverInfo.serverCapabilities);
        }
        else {
            printf("Connected to MCP server (no server info available).\n");
        }
    }
    else if (strcmp(subcmd, "disconnect") == 0) {
        if (!ctx->mcpClient) {
            fprintf(stderr, "Error: No active MCP connection.\n");
            return TINYAI_CLI_EXIT_ERROR;
        }

        printf("Disconnecting from MCP server...\n");
        tinyaiMcpDisconnect(ctx->mcpClient);
        tinyaiMcpDestroyClient(ctx->mcpClient);
        ctx->mcpClient = NULL;

        // Also destroy hybrid generation context if it exists
        if (ctx->hybridGen) {
            tinyaiDestroyHybridGenerate(ctx->hybridGen);
            ctx->hybridGen = NULL;
        }

        if (ctx->mcpServerUrl) {
            free(ctx->mcpServerUrl);
            ctx->mcpServerUrl = NULL;
        }

        ctx->useHybrid = 0;

        printf("Disconnected from MCP server.\n");
    }
    else if (strcmp(subcmd, "status") == 0) {
        if (!ctx->mcpClient) {
            printf("MCP status: Not connected\n");
            return TINYAI_CLI_EXIT_SUCCESS;
        }

        TinyAIMcpConnectionState state = tinyaiMcpGetConnectionState(ctx->mcpClient);
        printf("MCP status: %s\n", state == TINYAI_MCP_CONNECTED      ? "Connected"
                                   : state == TINYAI_MCP_CONNECTING   ? "Connecting"
                                   : state == TINYAI_MCP_DISCONNECTED ? "Disconnected"
                                   : state == TINYAI_MCP_ERROR        ? "Error"
                                                                      : "Unknown");

        if (state == TINYAI_MCP_CONNECTED) {
            printf("Connected to: %s\n", ctx->mcpServerUrl ? ctx->mcpServerUrl : "(unknown)");

            // Get execution preference
            TinyAIMcpExecutionPreference pref = tinyaiMcpGetExecutionPreference(ctx->mcpClient);
            printf("Execution preference: %s\n", pref == TINYAI_EXEC_ALWAYS_LOCAL   ? "Always local"
                                                 : pref == TINYAI_EXEC_PREFER_LOCAL ? "Prefer local"
                                                 : pref == TINYAI_EXEC_PREFER_MCP   ? "Prefer MCP"
                                                 : pref == TINYAI_EXEC_CUSTOM_POLICY
                                                     ? "Custom policy"
                                                     : "Unknown");

            // Print whether force offline mode is enabled
            printf("Force offline mode: %s\n",
                   tinyaiMcpGetForceOffline(ctx->mcpClient) ? "Yes" : "No");

            // Print hybrid status
            printf("Hybrid generation: %s\n", ctx->useHybrid ? "Enabled" : "Disabled");
            if (ctx->useHybrid && ctx->hybridGen) {
                printf("Remote generation available: %s\n",
                       tinyaiHybridGenerateHasRemote(ctx->hybridGen) ? "Yes" : "No");
            }
        }
    }
    else {
        fprintf(stderr,
                "Error: Unknown MCP subcommand '%s'. Use 'connect', 'disconnect', or 'status'.\n",
                subcmd);
        return TINYAI_CLI_EXIT_ERROR;
    }

    return TINYAI_CLI_EXIT_SUCCESS;
}

/**
 * Hybrid command handler implementation
 */
int tinyaiCommandHybrid(int argc, char **argv, void *context)
{
    TinyAICLIContext *ctx = (TinyAICLIContext *)context;

    if (argc < 2) {
        fprintf(stderr, "Usage: hybrid <on | off | status | force-local | force-remote>\n");
        return TINYAI_CLI_EXIT_ERROR;
    }

    const char *subcmd = argv[1];

    if (strcmp(subcmd, "on") == 0) {
        // Check if we have both model and MCP client
        if (!ctx->model) {
            fprintf(stderr, "Error: No model loaded. Hybrid generation requires a local model.\n");
            return TINYAI_CLI_EXIT_ERROR;
        }

        if (!ctx->mcpClient) {
            fprintf(stderr,
                    "Error: No MCP connection. Hybrid generation requires an MCP connection.\n");
            fprintf(stderr, "Use 'mcp connect <url>' to connect to an MCP server.\n");
            return TINYAI_CLI_EXIT_ERROR;
        }

        // Create hybrid generation context if needed
        if (!ctx->hybridGen) {
            ctx->hybridGen = tinyaiCreateHybridGenerate(ctx->model, ctx->mcpClient);
            if (!ctx->hybridGen) {
                fprintf(stderr, "Error: Failed to create hybrid generation context.\n");
                return TINYAI_CLI_EXIT_ERROR;
            }
        }

        ctx->useHybrid = 1;
        printf("Hybrid generation mode enabled.\n");

        // Reset force flags
        ctx->forceLocal  = 0;
        ctx->forceRemote = 0;
    }
    else if (strcmp(subcmd, "off") == 0) {
        ctx->useHybrid = 0;

        // Don't destroy the hybrid context, just disable its use
        // This allows quick toggling between modes without recreating the context

        printf("Hybrid generation mode disabled. Using local generation only.\n");
    }
    else if (strcmp(subcmd, "status") == 0) {
        printf("Hybrid generation: %s\n", ctx->useHybrid ? "Enabled" : "Disabled");

        if (ctx->hybridGen) {
            printf("Hybrid context: Available\n");
            printf("Remote generation available: %s\n",
                   tinyaiHybridGenerateHasRemote(ctx->hybridGen) ? "Yes" : "No");

            // Get execution stats if available
            double localTime, remoteTime, tokensPerSec;
            tinyaiHybridGenerateGetStats(ctx->hybridGen, &localTime, &remoteTime, &tokensPerSec);

            if (localTime > 0.0 || remoteTime > 0.0) {
                printf("Last generation stats:\n");
                printf("  Local time: %.2f ms\n", localTime);
                printf("  Remote time: %.2f ms\n", remoteTime);
                printf("  Tokens per second: %.2f\n", tokensPerSec);
                printf("  Last execution: %s\n",
                       tinyaiHybridGenerateUsedRemote(ctx->hybridGen) ? "Remote" : "Local");
            }
        }
        else {
            printf("Hybrid context: Not available\n");
        }

        // Force flags
        printf("Force local for next generation: %s\n", ctx->forceLocal ? "Yes" : "No");
        printf("Force remote for next generation: %s\n", ctx->forceRemote ? "Yes" : "No");
    }
    else if (strcmp(subcmd, "force-local") == 0) {
        if (!ctx->useHybrid || !ctx->hybridGen) {
            fprintf(stderr, "Error: Hybrid mode is not enabled.\n");
            return TINYAI_CLI_EXIT_ERROR;
        }

        ctx->forceLocal  = 1;
        ctx->forceRemote = 0;

        // Apply to hybrid context
        if (!tinyaiHybridGenerateForceMode(ctx->hybridGen, false)) {
            fprintf(stderr, "Warning: Could not force local mode in hybrid context.\n");
        }

        printf("Next generation will use local execution.\n");
    }
    else if (strcmp(subcmd, "force-remote") == 0) {
        if (!ctx->useHybrid || !ctx->hybridGen) {
            fprintf(stderr, "Error: Hybrid mode is not enabled.\n");
            return TINYAI_CLI_EXIT_ERROR;
        }

        if (!tinyaiHybridGenerateHasRemote(ctx->hybridGen)) {
            fprintf(stderr, "Error: Remote generation is not available.\n");
            return TINYAI_CLI_EXIT_ERROR;
        }

        ctx->forceLocal  = 0;
        ctx->forceRemote = 1;

        // Apply to hybrid context
        if (!tinyaiHybridGenerateForceMode(ctx->hybridGen, true)) {
            fprintf(stderr, "Warning: Could not force remote mode in hybrid context.\n");
            return TINYAI_CLI_EXIT_ERROR;
        }

        printf("Next generation will use remote execution.\n");
    }
    else {
        fprintf(stderr,
                "Error: Unknown hybrid subcommand '%s'. Use 'on', 'off', 'status', 'force-local', "
                "or 'force-remote'.\n",
                subcmd);
        return TINYAI_CLI_EXIT_ERROR;
    }

    return TINYAI_CLI_EXIT_SUCCESS;
}
