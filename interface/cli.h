/**
 * TinyAI Command Line Interface Header
 *
 * This header defines the command-line interface for TinyAI.
 */

#ifndef TINYAI_CLI_H
#define TINYAI_CLI_H

#include "../core/mcp/mcp_client.h"
#include "../models/text/generate.h"
#include "../models/text/hybrid_generate.h"
#include <stdint.h>

/* ----------------- Constants ----------------- */

/* Maximum command length */
#define TINYAI_CLI_MAX_COMMAND_LENGTH 1024

/* Maximum number of arguments */
#define TINYAI_CLI_MAX_ARGS 64

/* Maximum number of commands */
#define TINYAI_CLI_MAX_COMMANDS 32

/* Command exit codes */
#define TINYAI_CLI_EXIT_SUCCESS 0
#define TINYAI_CLI_EXIT_ERROR 1
#define TINYAI_CLI_EXIT_QUIT 2

/* ----------------- Types ----------------- */

/**
 * Command handler function type
 */
typedef int (*TinyAICommandHandler)(int argc, char **argv, void *context);

/**
 * Command structure
 */
typedef struct {
    const char          *name;        /* Command name */
    const char          *description; /* Command description */
    const char          *usage;       /* Command usage */
    TinyAICommandHandler handler;     /* Command handler */
} TinyAICommand;

/**
 * CLI context structure
 */
typedef struct {
    /* Model and tokenizer */
    TinyAIModel     *model;         /* Current model */
    TinyAITokenizer *tokenizer;     /* Current tokenizer */
    char            *modelPath;     /* Current model path */
    char            *tokenizerPath; /* Current tokenizer path */

    /* MCP and hybrid generation */
    TinyAIMcpClient      *mcpClient;    /* MCP client (if connected) */
    TinyAIHybridGenerate *hybridGen;    /* Hybrid generation context */
    char                 *mcpServerUrl; /* MCP server URL */
    int                   useHybrid;    /* Whether to use hybrid generation */
    int                   forceRemote;  /* Force remote execution for next generation */
    int                   forceLocal;   /* Force local execution for next generation */

    /* CLI state */
    int                    interactive; /* Whether in interactive mode */
    int                    verbose;     /* Verbosity level */
    TinyAIGenerationParams params;      /* Current generation parameters */
} TinyAICLIContext;

/* ----------------- API Functions ----------------- */

/**
 * Initialize the CLI
 *
 * @param context CLI context
 * @return 0 on success, non-zero on error
 */
int tinyaiCLIInit(TinyAICLIContext *context);

/**
 * Clean up the CLI
 *
 * @param context CLI context
 */
void tinyaiCLICleanup(TinyAICLIContext *context);

/**
 * Parse command-line arguments
 *
 * @param context CLI context
 * @param argc Argument count
 * @param argv Argument array
 * @return 0 on success, non-zero on error
 */
int tinyaiCLIParseArgs(TinyAICLIContext *context, int argc, char **argv);

/**
 * Run the CLI
 *
 * @param context CLI context
 * @param argc Argument count
 * @param argv Argument array
 * @return Exit code
 */
int tinyaiCLIRun(TinyAICLIContext *context, int argc, char **argv);

/**
 * Run the interactive shell
 *
 * @param context CLI context
 * @return Exit code
 */
int tinyaiCLIRunShell(TinyAICLIContext *context);

/**
 * Process a single command
 *
 * @param context CLI context
 * @param command Command line
 * @return Exit code
 */
int tinyaiCLIProcessCommand(TinyAICLIContext *context, const char *command);

/**
 * Register a command
 *
 * @param name Command name
 * @param description Command description
 * @param usage Command usage
 * @param handler Command handler
 * @return 0 on success, non-zero on error
 */
int tinyaiCLIRegisterCommand(const char *name, const char *description, const char *usage,
                             TinyAICommandHandler handler);

/**
 * Print help information
 *
 * @param context CLI context
 * @param command Command name (NULL for general help)
 * @return 0 on success, non-zero on error
 */
int tinyaiCLIPrintHelp(TinyAICLIContext *context, const char *command);

/* ----------------- Built-in Command Handlers ----------------- */

/**
 * Help command handler
 */
int tinyaiCommandHelp(int argc, char **argv, void *context);

/**
 * Version command handler
 */
int tinyaiCommandVersion(int argc, char **argv, void *context);

/**
 * Generate command handler
 */
int tinyaiCommandGenerate(int argc, char **argv, void *context);

/**
 * Tokenize command handler
 */
int tinyaiCommandTokenize(int argc, char **argv, void *context);

/**
 * Model command handler
 */
int tinyaiCommandModel(int argc, char **argv, void *context);

/**
 * Config command handler
 */
int tinyaiCommandConfig(int argc, char **argv, void *context);

/**
 * Exit command handler
 */
int tinyaiCommandExit(int argc, char **argv, void *context);

/**
 * MCP command handler for Model Context Protocol operations
 */
int tinyaiCommandMcp(int argc, char **argv, void *context);

/**
 * Hybrid command handler for controlling hybrid local/remote execution
 */
int tinyaiCommandHybrid(int argc, char **argv, void *context);

#endif /* TINYAI_CLI_H */
