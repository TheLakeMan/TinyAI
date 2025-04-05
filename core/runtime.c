/**
 * TinyAI Runtime Environment
 * 
 * This file implements the runtime environment for TinyAI, extending the
 * picol interpreter with module loading capabilities, command registration,
 * resource management, error handling, and event systems.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "picol.h" // Include the header from the same directory
#include "io.h"
#include "config.h"
#include "memory.h"

/* ----------------- Module System ----------------- */

#define MAX_MODULES 32
#define MAX_MODULE_PATH 256
#define MAX_MODULE_NAME 64

typedef struct tinyaiModule {
    char name[MAX_MODULE_NAME];
    int (*initFunc)(picolInterp *interp);
    int (*cleanupFunc)(picolInterp *interp);
    void *handle;          /* Dynamic library handle */
    int isLoaded;          /* Whether the module is currently loaded */
    char version[32];      /* Module version string */
    char dependencies[256]; /* Comma-separated list of dependencies */
} tinyaiModule;

/* Global module registry */
static tinyaiModule modules[MAX_MODULES];
static int moduleCount = 0;

/* Module search paths */
static char modulePaths[10][MAX_MODULE_PATH];
static int modulePathCount = 0;

/**
 * Add a module search path
 */
int tinyaiAddModulePath(const char *path) {
    if (modulePathCount >= 10) return 0; /* Too many paths */
    
    strncpy(modulePaths[modulePathCount], path, MAX_MODULE_PATH - 1);
    modulePaths[modulePathCount][MAX_MODULE_PATH - 1] = '\0';
    modulePathCount++;
    return 1;
}

/**
 * Register a statically linked module
 */
int tinyaiRegisterModule(const char *name, int (*initFunc)(picolInterp*), 
                      int (*cleanupFunc)(picolInterp*), const char *version,
                      const char *dependencies) {
    if (moduleCount >= MAX_MODULES) return 0; /* Too many modules */
    
    tinyaiModule *mod = &modules[moduleCount++];
    strncpy(mod->name, name, MAX_MODULE_NAME - 1);
    mod->name[MAX_MODULE_NAME - 1] = '\0';
    mod->initFunc = initFunc;
    mod->cleanupFunc = cleanupFunc;
    mod->handle = NULL; /* Static module, no handle */
    mod->isLoaded = 0;
    
    strncpy(mod->version, version, sizeof(mod->version) - 1);
    mod->version[sizeof(mod->version) - 1] = '\0';
    
    strncpy(mod->dependencies, dependencies, sizeof(mod->dependencies) - 1);
    mod->dependencies[sizeof(mod->dependencies) - 1] = '\0';
    
    return 1;
}

/**
 * Load a module by name
 */
int tinyaiLoadModule(picolInterp *interp, const char *name) {
    /* Search for the module in the registry */
    for (int i = 0; i < moduleCount; i++) {
        if (strcmp(modules[i].name, name) == 0) {
            if (modules[i].isLoaded) {
                return 1; /* Already loaded */
            }
            
            /* Initialize the module */
            if (modules[i].initFunc && modules[i].initFunc(interp) != 0) {
                /* Initialization failed */
                return 0;
            }
            
            modules[i].isLoaded = 1;
            return 1;
        }
    }
    
    /* Module not found in registry, try to load dynamically - future enhancement */
    
    return 0; /* Module not found */
}

/**
 * Unload a module by name
 */
int tinyaiUnloadModule(picolInterp *interp, const char *name) {
    /* Search for the module in the registry */
    for (int i = 0; i < moduleCount; i++) {
        if (strcmp(modules[i].name, name) == 0) {
            if (!modules[i].isLoaded) {
                return 1; /* Not loaded, nothing to do */
            }
            
            /* Clean up the module */
            if (modules[i].cleanupFunc && modules[i].cleanupFunc(interp) != 0) {
                /* Cleanup failed */
                return 0;
            }
            
            modules[i].isLoaded = 0;
            return 1;
        }
    }
    
    return 0; /* Module not found */
}

/* ----------------- Resource Management ----------------- */

/* Resource tracker structure */
#define MAX_RESOURCES 256

typedef enum {
    RESOURCE_FILE,
    RESOURCE_MEMORY,
    RESOURCE_OTHER
} ResourceType;

typedef struct {
    ResourceType type;
    void *handle;
    const char *description;
    int (*cleanupFunc)(void *handle);
} Resource;

static Resource resourceRegistry[MAX_RESOURCES];
static int resourceCount = 0;

/**
 * Register a resource for tracking
 */
int tinyaiRegisterResource(ResourceType type, void *handle, const char *description,
                        int (*cleanupFunc)(void *handle)) {
    if (resourceCount >= MAX_RESOURCES) return -1; /* Too many resources */
    
    Resource *res = &resourceRegistry[resourceCount];
    res->type = type;
    res->handle = handle;
    res->description = description;
    res->cleanupFunc = cleanupFunc;
    
    return resourceCount++;
}

/**
 * Release a resource by ID
 */
int tinyaiReleaseResource(int resourceId) {
    if (resourceId < 0 || resourceId >= resourceCount) return 0; /* Invalid ID */
    
    Resource *res = &resourceRegistry[resourceId];
    if (!res->handle) return 1; /* Already released */
    
    if (res->cleanupFunc && res->cleanupFunc(res->handle) != 0) {
        return 0; /* Cleanup failed */
    }
    
    res->handle = NULL;
    return 1;
}

/**
 * Release all resources
 */
void tinyaiReleaseAllResources() {
    for (int i = 0; i < resourceCount; i++) {
        if (resourceRegistry[i].handle) {
            if (resourceRegistry[i].cleanupFunc) {
                resourceRegistry[i].cleanupFunc(resourceRegistry[i].handle);
            }
            resourceRegistry[i].handle = NULL;
        }
    }
    resourceCount = 0;
}

/* ----------------- Error Handling System ----------------- */

#define MAX_ERROR_MSG 1024

typedef enum {
    ERROR_NONE,
    ERROR_SYNTAX,
    ERROR_RUNTIME,
    ERROR_MEMORY,
    ERROR_IO,
    ERROR_MODULE,
    ERROR_OTHER
} ErrorType;

typedef struct {
    ErrorType type;
    char message[MAX_ERROR_MSG];
    int code;
} TinyAIError;

static TinyAIError lastError = {ERROR_NONE, "", 0};

/**
 * Set the last error
 */
void tinyaiSetError(ErrorType type, int code, const char *format, ...) {
    va_list args;
    
    lastError.type = type;
    lastError.code = code;
    
    va_start(args, format);
    vsnprintf(lastError.message, MAX_ERROR_MSG - 1, format, args);
    va_end(args);
    
    lastError.message[MAX_ERROR_MSG - 1] = '\0';
}

/**
 * Get the last error
 */
void tinyaiGetError(ErrorType *type, int *code, char *message, size_t maxLen) {
    if (type) *type = lastError.type;
    if (code) *code = lastError.code;
    if (message && maxLen > 0) {
        strncpy(message, lastError.message, maxLen - 1);
        message[maxLen - 1] = '\0';
    }
}

/**
 * Clear the last error
 */
void tinyaiClearError() {
    lastError.type = ERROR_NONE;
    lastError.code = 0;
    lastError.message[0] = '\0';
}

/* ----------------- Event System ----------------- */

#define MAX_EVENTS 32
#define MAX_EVENT_HANDLERS 64

typedef struct {
    char name[64];
    int priority;
    int (*handler)(void *data);
    void *userData;
} EventHandler;

typedef struct {
    char name[64];
    EventHandler handlers[MAX_EVENT_HANDLERS];
    int handlerCount;
} Event;

static Event events[MAX_EVENTS];
static int eventCount = 0;

/**
 * Register an event
 */
int tinyaiRegisterEvent(const char *name) {
    if (eventCount >= MAX_EVENTS) return -1; /* Too many events */
    
    /* Check if event already exists */
    for (int i = 0; i < eventCount; i++) {
        if (strcmp(events[i].name, name) == 0) {
            return i; /* Event already exists */
        }
    }
    
    /* Create new event */
    Event *event = &events[eventCount];
    strncpy(event->name, name, sizeof(event->name) - 1);
    event->name[sizeof(event->name) - 1] = '\0';
    event->handlerCount = 0;
    
    return eventCount++;
}

/**
 * Register an event handler
 */
int tinyaiRegisterEventHandler(const char *eventName, int priority,
                            int (*handler)(void *data), void *userData) {
    /* Find the event */
    int eventId = -1;
    for (int i = 0; i < eventCount; i++) {
        if (strcmp(events[i].name, eventName) == 0) {
            eventId = i;
            break;
        }
    }
    
    if (eventId == -1) {
        /* Event doesn't exist, create it */
        eventId = tinyaiRegisterEvent(eventName);
        if (eventId == -1) return 0; /* Failed to create event */
    }
    
    Event *event = &events[eventId];
    if (event->handlerCount >= MAX_EVENT_HANDLERS) {
        return 0; /* Too many handlers for this event */
    }
    
    /* Add the handler */
    EventHandler *h = &event->handlers[event->handlerCount++];
    strncpy(h->name, eventName, sizeof(h->name) - 1);
    h->name[sizeof(h->name) - 1] = '\0';
    h->priority = priority;
    h->handler = handler;
    h->userData = userData;
    
    /* Sort handlers by priority (higher priority first) */
    for (int i = event->handlerCount - 1; i > 0; i--) {
        if (event->handlers[i].priority > event->handlers[i-1].priority) {
            /* Swap */
            EventHandler temp = event->handlers[i];
            event->handlers[i] = event->handlers[i-1];
            event->handlers[i-1] = temp;
        } else {
            break;
        }
    }
    
    return 1;
}

/**
 * Trigger an event
 */
int tinyaiTriggerEvent(const char *eventName, void *data) {
    /* Find the event */
    int eventId = -1;
    for (int i = 0; i < eventCount; i++) {
        if (strcmp(events[i].name, eventName) == 0) {
            eventId = i;
            break;
        }
    }
    
    if (eventId == -1) {
        return 0; /* Event doesn't exist */
    }
    
    Event *event = &events[eventId];
    
    /* Call all handlers in priority order */
    for (int i = 0; i < event->handlerCount; i++) {
        int result = event->handlers[i].handler(data);
        if (result != 0) {
            /* Handler requested to stop propagation */
            return 1;
        }
    }
    
    return 1;
}

/* ----------------- Picol Command Wrappers ----------------- */

/**
 * Command: loadmodule
 * 
 * Usage: loadmodule <module_name>
 */
int picolCommandLoadModule(picolInterp *i, int argc, char **argv, void *pd) {
    if (argc != 2) {
        picolSetResult(i, "wrong # args: should be 'loadmodule module_name'");
        return PICOL_ERR; // Return error code
    }
    
    if (tinyaiLoadModule(i, argv[1])) {
        picolSetResult(i, "");
        return PICOL_OK; // Added return
    } else {
        picolSetResult(i, "failed to load module");
        return PICOL_ERR; // Added return
    }
}

/**
 * Command: unloadmodule
 * 
 * Usage: unloadmodule <module_name>
 */
int picolCommandUnloadModule(picolInterp *i, int argc, char **argv, void *pd) {
    if (argc != 2) {
        picolSetResult(i, "wrong # args: should be 'unloadmodule module_name'");
        return PICOL_ERR; // Return error code
    }
    
    if (tinyaiUnloadModule(i, argv[1])) {
        picolSetResult(i, "");
        return PICOL_OK; // Added return
    } else {
        picolSetResult(i, "failed to unload module");
        return PICOL_ERR; // Added return
    }
}

/**
 * Command: listmodules
 * 
 * Usage: listmodules
 */
int picolCommandListModules(picolInterp *i, int argc, char **argv, void *pd) {
    if (argc != 1) {
        picolSetResult(i, "wrong # args: should be 'listmodules'");
        return PICOL_ERR; // Return error code
    }
    
    char result[PICOL_MAX_STR] = "";
    size_t resultLen = 0;
    
    for (int j = 0; j < moduleCount; j++) {
        size_t len = strlen(modules[j].name);
        if (resultLen + len + 10 >= PICOL_MAX_STR) {
            /* Not enough space */
            break;
        }
        
        if (resultLen > 0) {
            strcat(result, " ");
            resultLen++;
        }
        
        strcat(result, modules[j].name);
        if (modules[j].isLoaded) {
            strcat(result, "(loaded)");
        }
        resultLen += len + (modules[j].isLoaded ? 8 : 0);
    }
    
    picolSetResult(i, result);
    return PICOL_OK; // Added return
}

/**
 * Initialize the runtime environment
 */
int tinyaiRuntimeInit(picolInterp *interp) {
    /* Register module-related commands */
    picolRegisterCommand(interp, "loadmodule", picolCommandLoadModule, NULL);
    picolRegisterCommand(interp, "unloadmodule", picolCommandUnloadModule, NULL);
    picolRegisterCommand(interp, "listmodules", picolCommandListModules, NULL);
    
    /* Initialize default module search paths */
    tinyaiAddModulePath("./modules");
    tinyaiAddModulePath("../modules");
    
    /* Register core events */
    tinyaiRegisterEvent("init");
    tinyaiRegisterEvent("shutdown");
    tinyaiRegisterEvent("command");
    tinyaiRegisterEvent("error");
    
    /* Trigger init event */
    tinyaiTriggerEvent("init", interp);
    
    return 0;
}

/**
 * Cleanup the runtime environment
 */
void tinyaiRuntimeCleanup(picolInterp *interp) {
    /* Trigger shutdown event */
    tinyaiTriggerEvent("shutdown", interp);
    
    /* Unload all modules */
    for (int i = 0; i < moduleCount; i++) {
        if (modules[i].isLoaded) {
            tinyaiUnloadModule(interp, modules[i].name);
        }
    }
    
    /* Release all resources */
    tinyaiReleaseAllResources();
}
