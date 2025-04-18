/**
 * TinyAI Configuration Implementation
 * 
 * This file implements the configuration management system for TinyAI.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "config.h"
#include "memory.h"
#include "io.h"

/* ----------------- Configuration Storage ----------------- */

#define MAX_CONFIG_ENTRIES 256
#define MAX_KEY_LENGTH 64
#define MAX_VALUE_LENGTH 1024

typedef struct {
    char key[MAX_KEY_LENGTH];
    TinyAIConfigValue value;
    int active;  /* 1 if set, 0 if deleted or inactive */
} ConfigEntry;

static ConfigEntry configEntries[MAX_CONFIG_ENTRIES];
static int configEntryCount = 0;
static int configInitialized = 0;

/* ----------------- Helper Functions ----------------- */

/**
 * Find a configuration entry by key
 */
static int findConfigEntry(const char *key) {
    for (int i = 0; i < configEntryCount; i++) {
        if (configEntries[i].active && strcmp(configEntries[i].key, key) == 0) {
            return i;
        }
    }
    return -1;
}

/**
 * Create a new configuration entry
 */
static int createConfigEntry(const char *key, TinyAIConfigType type) {
    /* Check if key already exists */
    int index = findConfigEntry(key);
    if (index >= 0) {
        /* Key exists, update type */
        configEntries[index].value.type = type;
        
        /* Free string value if needed */
        if (type != TINYAI_CONFIG_STRING && 
            configEntries[index].value.type == TINYAI_CONFIG_STRING &&
            configEntries[index].value.value.stringValue) {
            free(configEntries[index].value.value.stringValue);
            configEntries[index].value.value.stringValue = NULL;
        }
        
        return index;
    }
    
    /* Check if we have space */
    if (configEntryCount >= MAX_CONFIG_ENTRIES) {
        /* Look for an inactive entry */
        for (int i = 0; i < configEntryCount; i++) {
            if (!configEntries[i].active) {
                /* Found an inactive entry, reuse it */
                strncpy(configEntries[i].key, key, MAX_KEY_LENGTH - 1);
                configEntries[i].key[MAX_KEY_LENGTH - 1] = '\0';
                configEntries[i].value.type = type;
                configEntries[i].active = 1;
                
                /* Clean up possible string value */
                if (configEntries[i].value.type == TINYAI_CONFIG_STRING &&
                    configEntries[i].value.value.stringValue) {
                    free(configEntries[i].value.value.stringValue);
                    configEntries[i].value.value.stringValue = NULL;
                }
                
                return i;
            }
        }
        
        /* No space available */
        return -1;
    }
    
    /* Create a new entry */
    index = configEntryCount++;
    strncpy(configEntries[index].key, key, MAX_KEY_LENGTH - 1);
    configEntries[index].key[MAX_KEY_LENGTH - 1] = '\0';
    configEntries[index].value.type = type;
    configEntries[index].active = 1;
    
    /* Initialize string value to NULL */
    if (type == TINYAI_CONFIG_STRING) {
        configEntries[index].value.value.stringValue = NULL;
    }
    
    return index;
}

/**
 * Trim whitespace from string
 */
static char* trimWhitespace(char *str) {
    if (!str) return NULL;
    
    /* Trim leading space */
    while (isspace((unsigned char)*str)) str++;
    
    if (*str == 0) return str;  /* All spaces */
    
    /* Trim trailing space */
    char *end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--;
    
    /* Write new null terminator */
    *(end + 1) = 0;
    
    return str;
}

/**
 * Parse a configuration line
 * 
 * Format: key = value
 */
static int parseConfigLine(char *line) {
    if (!line) return 0;
    
    /* Skip comments and empty lines */
    line = trimWhitespace(line);
    if (!line[0] || line[0] == '#' || line[0] == ';') {
        return 0;
    }
    
    /* Find the equal sign */
    char *equalSign = strchr(line, '=');
    if (!equalSign) {
        /* No equal sign, return an error */
        return -1;
    }
    
    /* Split the line at the equal sign */
    *equalSign = '\0';
    char *key = trimWhitespace(line);
    char *value = trimWhitespace(equalSign + 1);
    
    /* Check for empty key or value */
    if (!key[0] || !value[0]) {
        return -1;
    }
    
    /* Determine the value type */
    if ((value[0] == '"' && value[strlen(value) - 1] == '"') ||
        (value[0] == '\'' && value[strlen(value) - 1] == '\'')) {
        /* String value */
        value[strlen(value) - 1] = '\0';  /* Remove trailing quote */
        value++;  /* Skip leading quote */
        
        tinyaiConfigSetString(key, value);
    } else if (strcmp(value, "true") == 0 || strcmp(value, "false") == 0) {
        /* Boolean value */
        tinyaiConfigSetBool(key, strcmp(value, "true") == 0);
    } else {
        /* Try to parse as a number */
        char *endPtr;
        float floatValue = strtof(value, &endPtr);
        
        if (*endPtr == '\0') {
            /* Valid number */
            if (floatValue == (int)floatValue) {
                /* Integer */
                tinyaiConfigSetInt(key, (int)floatValue);
            } else {
                /* Float */
                tinyaiConfigSetFloat(key, floatValue);
            }
        } else {
            /* Treat as string */
            tinyaiConfigSetString(key, value);
        }
    }
    
    return 0;
}

/* ----------------- Public API ----------------- */

/**
 * Initialize the configuration system
 */
int tinyaiConfigInit() {
    if (configInitialized) {
        return 0;  /* Already initialized */
    }
    
    /* Clear the config entries */
    memset(configEntries, 0, sizeof(configEntries));
    configEntryCount = 0;
    configInitialized = 1;
    
    return 0;
}

/**
 * Clean up the configuration system
 */
void tinyaiConfigCleanup() {
    /* Clean up string values */
    for (int i = 0; i < configEntryCount; i++) {
        if (configEntries[i].active && 
            configEntries[i].value.type == TINYAI_CONFIG_STRING &&
            configEntries[i].value.value.stringValue) {
            free(configEntries[i].value.value.stringValue);
            configEntries[i].value.value.stringValue = NULL;
        }
    }
    
    /* Reset state */
    memset(configEntries, 0, sizeof(configEntries));
    configEntryCount = 0;
    configInitialized = 0;
}

/**
 * Load configuration from a file
 */
int tinyaiConfigLoad(const char *path) {
    TinyAIFile *file = tinyaiOpenFile(path, TINYAI_FILE_READ);
    if (!file) {
        return -1;
    }
    
    char line[MAX_KEY_LENGTH + MAX_VALUE_LENGTH + 10];
    int lineNum = 0;
    int errors = 0;
    
    while (tinyaiReadLine(file, line, sizeof(line)) > 0) {
        lineNum++;
        
        if (parseConfigLine(line) < 0) {
            /* Error parsing line */
            fprintf(stderr, "Error parsing config line %d: %s\n", lineNum, line);
            errors++;
        }
    }
    
    tinyaiCloseFile(file);
    
    return errors ? -1 : 0;
}

/**
 * Save configuration to a file
 */
int tinyaiConfigSave(const char *path) {
    TinyAIFile *file = tinyaiOpenFile(path, TINYAI_FILE_WRITE | TINYAI_FILE_CREATE);
    if (!file) {
        return -1;
    }
    
    /* Write a header */
    char header[256];
    sprintf(header, "# TinyAI Configuration File\n# Generated automatically\n\n");
    tinyaiWriteFile(file, header, strlen(header));
    
    /* Write each entry */
    for (int i = 0; i < configEntryCount; i++) {
        if (!configEntries[i].active) {
            continue;
        }
        
        char line[MAX_KEY_LENGTH + MAX_VALUE_LENGTH + 10];
        
        switch (configEntries[i].value.type) {
            case TINYAI_CONFIG_INTEGER:
                sprintf(line, "%s = %d\n", configEntries[i].key, 
                        configEntries[i].value.value.intValue);
                break;
            
            case TINYAI_CONFIG_FLOAT:
                sprintf(line, "%s = %f\n", configEntries[i].key, 
                        configEntries[i].value.value.floatValue);
                break;
            
            case TINYAI_CONFIG_STRING:
                sprintf(line, "%s = \"%s\"\n", configEntries[i].key, 
                        configEntries[i].value.value.stringValue ? 
                        configEntries[i].value.value.stringValue : "");
                break;
            
            case TINYAI_CONFIG_BOOLEAN:
                sprintf(line, "%s = %s\n", configEntries[i].key, 
                        configEntries[i].value.value.boolValue ? "true" : "false");
                break;
        }
        
        tinyaiWriteFile(file, line, strlen(line));
    }
    
    tinyaiCloseFile(file);
    
    return 0;
}

/**
 * Set a configuration integer value
 */
int tinyaiConfigSetInt(const char *key, int value) {
    if (!configInitialized) {
        if (tinyaiConfigInit() != 0) {
            return -1;
        }
    }
    
    int index = createConfigEntry(key, TINYAI_CONFIG_INTEGER);
    if (index < 0) {
        return -1;
    }
    
    configEntries[index].value.value.intValue = value;
    
    return 0;
}

/**
 * Get a configuration integer value
 */
int tinyaiConfigGetInt(const char *key, int defaultValue) {
    if (!configInitialized) {
        return defaultValue;
    }
    
    int index = findConfigEntry(key);
    if (index < 0) {
        return defaultValue;
    }
    
    switch (configEntries[index].value.type) {
        case TINYAI_CONFIG_INTEGER:
            return configEntries[index].value.value.intValue;
        
        case TINYAI_CONFIG_FLOAT:
            return (int)configEntries[index].value.value.floatValue;
        
        case TINYAI_CONFIG_BOOLEAN:
            return configEntries[index].value.value.boolValue ? 1 : 0;
        
        case TINYAI_CONFIG_STRING:
            if (configEntries[index].value.value.stringValue) {
                return atoi(configEntries[index].value.value.stringValue);
            }
            break;
    }
    
    return defaultValue;
}

/**
 * Set a configuration float value
 */
int tinyaiConfigSetFloat(const char *key, float value) {
    if (!configInitialized) {
        if (tinyaiConfigInit() != 0) {
            return -1;
        }
    }
    
    int index = createConfigEntry(key, TINYAI_CONFIG_FLOAT);
    if (index < 0) {
        return -1;
    }
    
    configEntries[index].value.value.floatValue = value;
    
    return 0;
}

/**
 * Get a configuration float value
 */
float tinyaiConfigGetFloat(const char *key, float defaultValue) {
    if (!configInitialized) {
        return defaultValue;
    }
    
    int index = findConfigEntry(key);
    if (index < 0) {
        return defaultValue;
    }
    
    switch (configEntries[index].value.type) {
        case TINYAI_CONFIG_INTEGER:
            return (float)configEntries[index].value.value.intValue;
        
        case TINYAI_CONFIG_FLOAT:
            return configEntries[index].value.value.floatValue;
        
        case TINYAI_CONFIG_BOOLEAN:
            return configEntries[index].value.value.boolValue ? 1.0f : 0.0f;
        
        case TINYAI_CONFIG_STRING:
            if (configEntries[index].value.value.stringValue) {
                return atof(configEntries[index].value.value.stringValue);
            }
            break;
    }
    
    return defaultValue;
}

/**
 * Set a configuration string value
 */
int tinyaiConfigSetString(const char *key, const char *value) {
    if (!configInitialized) {
        if (tinyaiConfigInit() != 0) {
            return -1;
        }
    }
    
    int index = createConfigEntry(key, TINYAI_CONFIG_STRING);
    if (index < 0) {
        return -1;
    }
    
    /* Free existing value if any */
    if (configEntries[index].value.value.stringValue) {
        free(configEntries[index].value.value.stringValue);
    }
    
    /* Copy the new value */
    if (value) {
        configEntries[index].value.value.stringValue = _strdup(value);
        if (!configEntries[index].value.value.stringValue) {
            return -1;
        }
    } else {
        configEntries[index].value.value.stringValue = NULL;
    }
    
    return 0;
}

/**
 * Get a configuration string value
 */
const char* tinyaiConfigGetString(const char *key, const char *defaultValue) {
    if (!configInitialized) {
        return defaultValue;
    }
    
    int index = findConfigEntry(key);
    if (index < 0) {
        return defaultValue;
    }
    
    static char buffer[MAX_VALUE_LENGTH];
    
    switch (configEntries[index].value.type) {
        case TINYAI_CONFIG_INTEGER:
            sprintf(buffer, "%d", configEntries[index].value.value.intValue);
            return buffer;
        
        case TINYAI_CONFIG_FLOAT:
            sprintf(buffer, "%f", configEntries[index].value.value.floatValue);
            return buffer;
        
        case TINYAI_CONFIG_BOOLEAN:
            return configEntries[index].value.value.boolValue ? "true" : "false";
        
        case TINYAI_CONFIG_STRING:
            return configEntries[index].value.value.stringValue ? 
                   configEntries[index].value.value.stringValue : defaultValue;
    }
    
    return defaultValue;
}

/**
 * Set a configuration boolean value
 */
int tinyaiConfigSetBool(const char *key, int value) {
    if (!configInitialized) {
        if (tinyaiConfigInit() != 0) {
            return -1;
        }
    }
    
    int index = createConfigEntry(key, TINYAI_CONFIG_BOOLEAN);
    if (index < 0) {
        return -1;
    }
    
    configEntries[index].value.value.boolValue = value ? 1 : 0;
    
    return 0;
}

/**
 * Get a configuration boolean value
 */
int tinyaiConfigGetBool(const char *key, int defaultValue) {
    if (!configInitialized) {
        return defaultValue;
    }
    
    int index = findConfigEntry(key);
    if (index < 0) {
        return defaultValue;
    }
    
    switch (configEntries[index].value.type) {
        case TINYAI_CONFIG_INTEGER:
            return configEntries[index].value.value.intValue != 0;
        
        case TINYAI_CONFIG_FLOAT:
            return configEntries[index].value.value.floatValue != 0.0f;
        
        case TINYAI_CONFIG_BOOLEAN:
            return configEntries[index].value.value.boolValue;
        
        case TINYAI_CONFIG_STRING:
            if (configEntries[index].value.value.stringValue) {
                return strcmp(configEntries[index].value.value.stringValue, "true") == 0 ||
                       strcmp(configEntries[index].value.value.stringValue, "1") == 0 ||
                       strcmp(configEntries[index].value.value.stringValue, "yes") == 0 ||
                       strcmp(configEntries[index].value.value.stringValue, "y") == 0 ||
                       strcmp(configEntries[index].value.value.stringValue, "on") == 0;
            }
            break;
    }
    
    return defaultValue;
}

/**
 * Check if a configuration key exists
 */
int tinyaiConfigHasKey(const char *key) {
    if (!configInitialized) {
        return 0;
    }
    
    return findConfigEntry(key) >= 0;
}

/**
 * Remove a configuration key
 */
int tinyaiConfigRemoveKey(const char *key) {
    if (!configInitialized) {
        return 0;
    }
    
    int index = findConfigEntry(key);
    if (index < 0) {
        return 0;
    }
    
    /* Free string value if needed */
    if (configEntries[index].value.type == TINYAI_CONFIG_STRING &&
        configEntries[index].value.value.stringValue) {
        free(configEntries[index].value.value.stringValue);
        configEntries[index].value.value.stringValue = NULL;
    }
    
    configEntries[index].active = 0;
    
    return 1;
}

/**
 * Get all configuration keys
 */
int tinyaiConfigGetKeys(char **keys, int maxKeys) {
    if (!configInitialized || !keys || maxKeys <= 0) {
        return 0;
    }
    
    int count = 0;
    
    for (int i = 0; i < configEntryCount && count < maxKeys; i++) {
        if (configEntries[i].active) {
            keys[count++] = configEntries[i].key;
        }
    }
    
    return count;
}

/**
 * Set default configuration values
 */
int tinyaiConfigSetDefaults() {
    if (!configInitialized) {
        if (tinyaiConfigInit() != 0) {
            return -1;
        }
    }
    
    /* System settings */
    tinyaiConfigSetString("system.name", "TinyAI");
    tinyaiConfigSetString("system.version", "0.1.0");
    tinyaiConfigSetString("system.data_dir", "./data");
    tinyaiConfigSetString("system.model_dir", "./models");
    
    /* Memory settings */
    tinyaiConfigSetInt("memory.pool_size", 1024 * 1024);  /* 1MB */
    tinyaiConfigSetInt("memory.max_allocations", 10000);
    tinyaiConfigSetBool("memory.track_leaks", 1);
    
    /* Model settings */
    tinyaiConfigSetInt("model.context_size", 512);
    tinyaiConfigSetInt("model.hidden_size", 256);
    tinyaiConfigSetFloat("model.temperature", 0.7f);
    tinyaiConfigSetInt("model.top_k", 40);
    tinyaiConfigSetFloat("model.top_p", 0.9f);
    
    /* Text generation settings */
    tinyaiConfigSetInt("generate.max_tokens", 100);
    tinyaiConfigSetBool("generate.add_bos", 1);
    
    /* Tokenizer settings */
    tinyaiConfigSetInt("tokenizer.vocab_size", 8192);
    tinyaiConfigSetBool("tokenizer.case_sensitive", 0);
    
    return 0;
}

/**
 * Override a configuration value from command line
 */
int tinyaiConfigOverride(const char *key, const char *value) {
    if (!key || !value) {
        return -1;
    }
    
    /* Parse the value based on type */
    if ((value[0] == '"' && value[strlen(value) - 1] == '"') ||
        (value[0] == '\'' && value[strlen(value) - 1] == '\'')) {
        /* String value */
        char *stringValue = _strdup(value + 1);
        stringValue[strlen(stringValue) - 1] = '\0';
        
        int result = tinyaiConfigSetString(key, stringValue);
        free(stringValue);
        return result;
    } else if (strcmp(value, "true") == 0 || strcmp(value, "false") == 0) {
        /* Boolean value */
        return tinyaiConfigSetBool(key, strcmp(value, "true") == 0);
    } else {
        /* Try to parse as a number */
        char *endPtr;
        float floatValue = strtof(value, &endPtr);
        
        if (*endPtr == '\0') {
            /* Valid number */
            if (floatValue == (int)floatValue) {
                /* Integer */
                return tinyaiConfigSetInt(key, (int)floatValue);
            } else {
                /* Float */
                return tinyaiConfigSetFloat(key, floatValue);
            }
        } else {
            /* Treat as string */
            return tinyaiConfigSetString(key, value);
        }
    }
}

/**
 * Apply command line overrides
 */
int tinyaiConfigApplyCommandLine(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        /* Look for --key=value or -key=value format */
        if ((argv[i][0] == '-' && argv[i][1] == '-') || 
            (argv[i][0] == '-')) {
            /* Get the key (skip leading dashes) */
            char *key = argv[i] + (argv[i][1] == '-' ? 2 : 1);
            
            /* Find the equal sign */
            char *equalSign = strchr(key, '=');
            if (equalSign) {
                /* Split the string at the equal sign */
                *equalSign = '\0';
                char *value = equalSign + 1;
                
                /* Override the configuration */
                tinyaiConfigOverride(key, value);
                
                /* Restore the equal sign for display purposes */
                *equalSign = '=';
            } else if (i + 1 < argc && argv[i + 1][0] != '-') {
                /* Format: --key value or -key value */
                tinyaiConfigOverride(key, argv[i + 1]);
                i++;  /* Skip the value */
            }
        }
    }
    
    return 0;
}
