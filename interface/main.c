/**
 * TinyAI Main Entry Point
 * 
 * This file implements the main entry point for the TinyAI system.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../core/picol.h"
#include "../core/runtime.h"
#include "../core/io.h"
#include "../core/memory.h"
#include "../core/config.h"
#include "../models/text/tokenizer.h"
#include "../models/text/generate.h"

/* Default configuration path */
#define DEFAULT_CONFIG_PATH "tinyai.conf"

/* Print version information */
static void printVersion() {
    printf("TinyAI v0.1.0 - Ultra-lightweight AI Model\n");
    printf("(c) 2025 TinyAI Contributors\n");
}

/* Print usage information */
static void printUsage(const char *programName) {
    printf("Usage: %s [options] [command]\n\n", programName);
    printf("Options:\n");
    printf("  -h, --help              Show this help message\n");
    printf("  -v, --version           Show version information\n");
    printf("  -c, --config <file>     Use specified config file\n");
    printf("  -m, --model <file>      Load the specified model\n");
    printf("  -t, --tokenizer <file>  Load the specified tokenizer\n");
    printf("  -o, --output <file>     Write output to file instead of stdout\n");
    printf("\n");
    printf("Commands:\n");
    printf("  generate <prompt>       Generate text from a prompt\n");
    printf("  train <file>            Train a model on text file\n");
    printf("  quantize <model> <out>  Quantize a model to 4-bit precision\n");
    printf("  shell                   Start interactive shell\n");
    printf("\n");
}

/* Interactive shell command handler */
static int handleShellCommand(picolInterp *interp, const char *command) {
    int result = picolEval(interp, (char *)command);
    
    if (result == PICOL_OK) {
        if (interp->resultString && interp->resultString[0]) {
            printf("%s\n", interp->resultString);
        }
    } else {
        printf("Error: %s\n", interp->resultString);
    }
    
    return result != PICOL_ERR;
}

/* Interactive shell loop */
static int runShell(picolInterp *interp) {
    char command[1024];
    
    printf("TinyAI Shell v0.1.0\n");
    printf("Type 'exit' to quit, 'help' for available commands\n");
    
    while (1) {
        printf("> ");
        fflush(stdout);
        
        if (!fgets(command, sizeof(command), stdin)) {
            break;
        }
        
        /* Remove trailing newline */
        size_t len = strlen(command);
        if (len > 0 && command[len - 1] == '\n') {
            command[len - 1] = '\0';
        }
        
        /* Check for exit command */
        if (strcmp(command, "exit") == 0 || strcmp(command, "quit") == 0) {
            break;
        }
        
        /* Handle the command */
        handleShellCommand(interp, command);
    }
    
    return 0;
}

/* Command: generate text */
static int cmdGenerate(picolInterp *interp, const char *modelPath, 
                      const char *tokenizerPath, const char *prompt) {
    printf("Loading model from %s...\n", modelPath);
    printf("Loading tokenizer from %s...\n", tokenizerPath);
    printf("Generating text from prompt: \"%s\"\n", prompt);
    
    /* TODO: Implement actual text generation once the model is implemented */
    printf("Text generation not yet implemented\n");
    
    return 0;
}

/* Main entry point */
int main(int argc, char **argv) {
    int i;
    int result = 0;
    const char *configPath = DEFAULT_CONFIG_PATH;
    const char *modelPath = NULL;
    const char *tokenizerPath = NULL;
    const char *outputPath = NULL;
    const char *command = NULL;
    
    /* Initialize the system */
    tinyaiIOInit();
    tinyaiMemTrackInit();
    tinyaiConfigInit();
    
    /* Create the interpreter */
    picolInterp *interp = picolCreateInterp();
    if (!interp) {
        fprintf(stderr, "Error: Failed to create interpreter\n");
        return 1;
    }
    
    /* Initialize the runtime */
    tinyaiRuntimeInit(interp);
    
    /* Parse command line arguments */
    for (i = 1; i < argc; i++) {
        if (argv[i][0] == '-') {
            /* Option */
            if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
                printUsage(argv[0]);
                result = 0;
                goto cleanup;
            } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--version") == 0) {
                printVersion();
                result = 0;
                goto cleanup;
            } else if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--config") == 0) {
                if (i + 1 < argc) {
                    configPath = argv[++i];
                } else {
                    fprintf(stderr, "Error: Missing argument for %s\n", argv[i]);
                    result = 1;
                    goto cleanup;
                }
            } else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
                if (i + 1 < argc) {
                    modelPath = argv[++i];
                } else {
                    fprintf(stderr, "Error: Missing argument for %s\n", argv[i]);
                    result = 1;
                    goto cleanup;
                }
            } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--tokenizer") == 0) {
                if (i + 1 < argc) {
                    tokenizerPath = argv[++i];
                } else {
                    fprintf(stderr, "Error: Missing argument for %s\n", argv[i]);
                    result = 1;
                    goto cleanup;
                }
            } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
                if (i + 1 < argc) {
                    outputPath = argv[++i];
                } else {
                    fprintf(stderr, "Error: Missing argument for %s\n", argv[i]);
                    result = 1;
                    goto cleanup;
                }
            } else {
                fprintf(stderr, "Error: Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                result = 1;
                goto cleanup;
            }
        } else {
            /* Command or argument */
            if (!command) {
                command = argv[i];
            } else {
                /* Command arguments are handled below */
                break;
            }
        }
    }
    
    /* Load configuration */
    if (tinyaiConfigLoad(configPath) != 0) {
        printf("Warning: Failed to load configuration from %s, using defaults\n", 
               configPath);
        tinyaiConfigSetDefaults();
    }
    
    /* Apply command line overrides */
    tinyaiConfigApplyCommandLine(argc, argv);
    
    /* Execute command */
    if (!command) {
        /* No command, start shell */
        runShell(interp);
    } else if (strcmp(command, "shell") == 0) {
        /* Explicit shell command */
        runShell(interp);
    } else if (strcmp(command, "generate") == 0) {
        /* Generate text */
        if (i < argc) {
            const char *prompt = argv[i];
            cmdGenerate(interp, modelPath, tokenizerPath, prompt);
        } else {
            fprintf(stderr, "Error: Missing prompt for generate command\n");
            result = 1;
        }
    } else if (strcmp(command, "train") == 0) {
        /* Train model */
        fprintf(stderr, "Error: Training not yet implemented\n");
        result = 1;
    } else if (strcmp(command, "quantize") == 0) {
        /* Quantize model */
        fprintf(stderr, "Error: Quantization not yet implemented\n");
        result = 1;
    } else {
        /* Unknown command */
        fprintf(stderr, "Error: Unknown command: %s\n", command);
        printUsage(argv[0]);
        result = 1;
    }
    
cleanup:
    /* Clean up */
    tinyaiRuntimeCleanup(interp);
    picolFreeInterp(interp);
    tinyaiConfigCleanup();
    tinyaiIOCleanup();
    
    /* Check for memory leaks if in debug mode */
#ifdef TINYAI_MEM_TRACK
    tinyaiMemCheckLeaks();
#endif
    tinyaiMemTrackCleanup();
    
    return result;
}
