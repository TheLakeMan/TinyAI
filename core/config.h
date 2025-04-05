/**
 * TinyAI Configuration Header
 * 
 * This header defines the configuration system for TinyAI, allowing
 * for flexible runtime configuration of the framework.
 */

#ifndef TINYAI_CONFIG_H
#define TINYAI_CONFIG_H

/* ----------------- Configuration Value Types ----------------- */

/**
 * Configuration value type enumeration
 */
typedef enum {
    TINYAI_CONFIG_INTEGER,    /* Integer value */
    TINYAI_CONFIG_FLOAT,      /* Float value */
    TINYAI_CONFIG_STRING,     /* String value */
    TINYAI_CONFIG_BOOLEAN     /* Boolean value */
} TinyAIConfigType;

/**
 * Configuration value union
 */
typedef struct {
    TinyAIConfigType type;    /* Value type */
    union {
        int intValue;         /* Integer value */
        float floatValue;     /* Float value */
        char *stringValue;    /* String value */
        int boolValue;        /* Boolean value */
    } value;
} TinyAIConfigValue;

/* ----------------- Configuration API ----------------- */

/**
 * Initialize the configuration system
 * 
 * @return 0 on success, non-zero on error
 */
int tinyaiConfigInit();

/**
 * Clean up the configuration system
 */
void tinyaiConfigCleanup();

/**
 * Load configuration from a file
 * 
 * @param path File path
 * @return 0 on success, non-zero on error
 */
int tinyaiConfigLoad(const char *path);

/**
 * Save configuration to a file
 * 
 * @param path File path
 * @return 0 on success, non-zero on error
 */
int tinyaiConfigSave(const char *path);

/**
 * Set an integer configuration value
 * 
 * @param key Configuration key
 * @param value Integer value
 * @return 0 on success, non-zero on error
 */
int tinyaiConfigSetInt(const char *key, int value);

/**
 * Get an integer configuration value
 * 
 * @param key Configuration key
 * @param defaultValue Default value if key not found
 * @return Configuration value or default value
 */
int tinyaiConfigGetInt(const char *key, int defaultValue);

/**
 * Set a float configuration value
 * 
 * @param key Configuration key
 * @param value Float value
 * @return 0 on success, non-zero on error
 */
int tinyaiConfigSetFloat(const char *key, float value);

/**
 * Get a float configuration value
 * 
 * @param key Configuration key
 * @param defaultValue Default value if key not found
 * @return Configuration value or default value
 */
float tinyaiConfigGetFloat(const char *key, float defaultValue);

/**
 * Set a string configuration value
 * 
 * @param key Configuration key
 * @param value String value
 * @return 0 on success, non-zero on error
 */
int tinyaiConfigSetString(const char *key, const char *value);

/**
 * Get a string configuration value
 * 
 * @param key Configuration key
 * @param defaultValue Default value if key not found
 * @return Configuration value or default value
 */
const char* tinyaiConfigGetString(const char *key, const char *defaultValue);

/**
 * Set a boolean configuration value
 * 
 * @param key Configuration key
 * @param value Boolean value (0 = false, non-zero = true)
 * @return 0 on success, non-zero on error
 */
int tinyaiConfigSetBool(const char *key, int value);

/**
 * Get a boolean configuration value
 * 
 * @param key Configuration key
 * @param defaultValue Default value if key not found
 * @return Configuration value or default value
 */
int tinyaiConfigGetBool(const char *key, int defaultValue);

/**
 * Check if a configuration key exists
 * 
 * @param key Configuration key
 * @return 1 if key exists, 0 if not
 */
int tinyaiConfigHasKey(const char *key);

/**
 * Remove a configuration key
 * 
 * @param key Configuration key
 * @return 1 if key was removed, 0 if key not found
 */
int tinyaiConfigRemoveKey(const char *key);

/**
 * Get all configuration keys
 * 
 * @param keys Array to store keys
 * @param maxKeys Maximum number of keys to store
 * @return Number of keys stored
 */
int tinyaiConfigGetKeys(char **keys, int maxKeys);

/**
 * Set default configuration values
 * 
 * @return 0 on success, non-zero on error
 */
int tinyaiConfigSetDefaults();

/**
 * Override a configuration value from command line
 * 
 * @param key Configuration key
 * @param value Value string
 * @return 0 on success, non-zero on error
 */
int tinyaiConfigOverride(const char *key, const char *value);

/**
 * Apply command line overrides
 * 
 * @param argc Argument count
 * @param argv Argument array
 * @return 0 on success, non-zero on error
 */
int tinyaiConfigApplyCommandLine(int argc, char **argv);

#endif /* TINYAI_CONFIG_H */
