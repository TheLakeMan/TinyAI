/**
 * TinyAI Runtime Header
 * 
 * This header defines the runtime environment for TinyAI, including module
 * loading, resource management, error handling, and event systems.
 */

#ifndef TINYAI_RUNTIME_H
#define TINYAI_RUNTIME_H

#include "picol.h"

/* ----------------- Module System ----------------- */

/**
 * Add a module search path
 * 
 * @param path Directory path to search for modules
 * @return 1 on success, 0 on failure
 */
int tinyaiAddModulePath(const char *path);

/**
 * Register a module
 * 
 * @param name Module name
 * @param initFunc Initialization function
 * @param cleanupFunc Cleanup function
 * @param version Module version string
 * @param dependencies Comma-separated list of dependencies
 * @return 1 on success, 0 on failure
 */
int tinyaiRegisterModule(const char *name, int (*initFunc)(picolInterp*), 
                      int (*cleanupFunc)(picolInterp*), const char *version,
                      const char *dependencies);

/**
 * Load a module
 * 
 * @param interp Picol interpreter
 * @param name Module name
 * @return 1 on success, 0 on failure
 */
int tinyaiLoadModule(picolInterp *interp, const char *name);

/**
 * Unload a module
 * 
 * @param interp Picol interpreter
 * @param name Module name
 * @return 1 on success, 0 on failure
 */
int tinyaiUnloadModule(picolInterp *interp, const char *name);

/* ----------------- Resource Management ----------------- */

/**
 * Resource types
 */
typedef enum {
    RESOURCE_FILE,
    RESOURCE_MEMORY,
    RESOURCE_OTHER
} ResourceType;

/**
 * Register a resource for tracking
 * 
 * @param type Resource type
 * @param handle Resource handle
 * @param description Resource description
 * @param cleanupFunc Cleanup function
 * @return Resource ID or -1 on error
 */
int tinyaiRegisterResource(ResourceType type, void *handle, const char *description,
                        int (*cleanupFunc)(void *handle));

/**
 * Release a resource
 * 
 * @param resourceId Resource ID
 * @return 1 on success, 0 on failure
 */
int tinyaiReleaseResource(int resourceId);

/**
 * Release all resources
 */
void tinyaiReleaseAllResources();

/* ----------------- Error Handling ----------------- */

/**
 * Error types
 */
typedef enum {
    ERROR_NONE,
    ERROR_SYNTAX,
    ERROR_RUNTIME,
    ERROR_MEMORY,
    ERROR_IO,
    ERROR_MODULE,
    ERROR_OTHER
} ErrorType;

/**
 * Set an error
 * 
 * @param type Error type
 * @param code Error code
 * @param format Error message format string
 * @param ... Format arguments
 */
void tinyaiSetError(ErrorType type, int code, const char *format, ...);

/**
 * Get the last error
 * 
 * @param type Pointer to store the error type
 * @param code Pointer to store the error code
 * @param message Buffer to store the error message
 * @param maxLen Maximum length of the message buffer
 */
void tinyaiGetError(ErrorType *type, int *code, char *message, size_t maxLen);

/**
 * Clear the last error
 */
void tinyaiClearError();

/* ----------------- Event System ----------------- */

/**
 * Register an event
 * 
 * @param name Event name
 * @return Event ID or -1 on error
 */
int tinyaiRegisterEvent(const char *name);

/**
 * Register an event handler
 * 
 * @param eventName Event name
 * @param priority Handler priority (higher = called first)
 * @param handler Handler function
 * @param userData User data to pass to the handler
 * @return 1 on success, 0 on failure
 */
int tinyaiRegisterEventHandler(const char *eventName, int priority,
                            int (*handler)(void *data), void *userData);

/**
 * Trigger an event
 * 
 * @param eventName Event name
 * @param data Data to pass to handlers
 * @return 1 on success, 0 on failure
 */
int tinyaiTriggerEvent(const char *eventName, void *data);

/* ----------------- Initialization ----------------- */

/**
 * Initialize the runtime
 * 
 * @param interp Picol interpreter
 * @return 0 on success, non-zero on error
 */
int tinyaiRuntimeInit(picolInterp *interp);

/**
 * Clean up the runtime
 * 
 * @param interp Picol interpreter
 */
void tinyaiRuntimeCleanup(picolInterp *interp);

#endif /* TINYAI_RUNTIME_H */
