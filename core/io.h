/**
 * TinyAI I/O System Header
 * 
 * This header defines the I/O system for TinyAI, providing cross-platform
 * file operations for loading models and saving configurations.
 */

#ifndef TINYAI_IO_H
#define TINYAI_IO_H

#include <stddef.h>
#include <stdint.h>

/* ----------------- File Mode Flags ----------------- */

#define TINYAI_FILE_READ        0x01    /* Open for reading */
#define TINYAI_FILE_WRITE       0x02    /* Open for writing */
#define TINYAI_FILE_APPEND      0x04    /* Open for appending */
#define TINYAI_FILE_BINARY      0x08    /* Open in binary mode */
#define TINYAI_FILE_CREATE      0x10    /* Create if doesn't exist */
#define TINYAI_FILE_TRUNCATE    0x20    /* Truncate if exists */

/* ----------------- File Error Codes ----------------- */

#define TINYAI_IO_SUCCESS       0       /* Operation successful */
#define TINYAI_IO_ERROR        -1       /* Generic I/O error */
#define TINYAI_IO_NOT_FOUND    -2       /* File not found */
#define TINYAI_IO_ACCESS       -3       /* Permission denied */
#define TINYAI_IO_EXISTS       -4       /* File already exists */
#define TINYAI_IO_INVALID      -5       /* Invalid argument */
#define TINYAI_IO_NO_MEMORY    -6       /* Out of memory */
#define TINYAI_IO_EOF          -7       /* End of file */

/* ----------------- Type Definitions ----------------- */

/* File handle type */
typedef struct TinyAIFile TinyAIFile;

/* Directory handle type */
typedef struct TinyAIDir TinyAIDir;

/* File information structure */
typedef struct {
    char *path;             /* Full path to file */
    uint64_t size;          /* File size in bytes */
    uint64_t modTime;       /* Last modification time */
    uint32_t mode;          /* File mode */
    int isDirectory;        /* Whether it's a directory */
} TinyAIFileInfo;

/* ----------------- System Functions ----------------- */

/**
 * Initialize I/O system
 * 
 * @return 0 on success, non-zero on error
 */
int tinyaiIOInit();

/**
 * Clean up I/O system
 */
void tinyaiIOCleanup();

/**
 * Get last I/O error code
 * 
 * @return Last error code
 */
int tinyaiIOGetLastError();

/**
 * Get error message for error code
 * 
 * @param error Error code
 * @return Error message string
 */
const char* tinyaiIOGetErrorString(int error);

/* ----------------- File Operations ----------------- */

/**
 * Open a file
 * 
 * @param path File path
 * @param mode File mode flags
 * @return File handle or NULL on error
 */
TinyAIFile* tinyaiOpenFile(const char *path, int mode);

/**
 * Close a file
 * 
 * @param file File handle
 */
void tinyaiCloseFile(TinyAIFile *file);

/**
 * Read from a file
 * 
 * @param file File handle
 * @param buffer Buffer to read into
 * @param size Number of bytes to read
 * @return Number of bytes read or negative error code
 */
int64_t tinyaiReadFile(TinyAIFile *file, void *buffer, size_t size);

/**
 * Write to a file
 * 
 * @param file File handle
 * @param buffer Buffer to write from
 * @param size Number of bytes to write
 * @return Number of bytes written or negative error code
 */
int64_t tinyaiWriteFile(TinyAIFile *file, const void *buffer, size_t size);

/**
 * Read a line from a file
 * 
 * @param file File handle
 * @param buffer Buffer to read into
 * @param size Buffer size
 * @return Number of bytes read or negative error code
 */
int tinyaiReadLine(TinyAIFile *file, char *buffer, size_t size);

/**
 * Seek to position in file
 * 
 * @param file File handle
 * @param offset Offset in bytes
 * @param whence Seek origin (0=start, 1=current, 2=end)
 * @return New position or negative error code
 */
int64_t tinyaiSeekFile(TinyAIFile *file, int64_t offset, int whence);

/**
 * Get current position in file
 * 
 * @param file File handle
 * @return Current position or negative error code
 */
int64_t tinyaiTellFile(TinyAIFile *file);

/**
 * Flush file buffers
 * 
 * @param file File handle
 * @return 0 on success, negative error code on failure
 */
int tinyaiFlushFile(TinyAIFile *file);

/**
 * Check if end of file
 * 
 * @param file File handle
 * @return 1 if EOF, 0 if not, negative error code on failure
 */
int tinyaiEOF(TinyAIFile *file);

/* ----------------- File System Operations ----------------- */

/**
 * Check if file exists
 * 
 * @param path File path
 * @return 1 if exists, 0 if not, negative error code on failure
 */
int tinyaiFileExists(const char *path);

/**
 * Delete a file
 * 
 * @param path File path
 * @return 0 on success, negative error code on failure
 */
int tinyaiDeleteFile(const char *path);

/**
 * Rename a file
 * 
 * @param oldPath Old file path
 * @param newPath New file path
 * @return 0 on success, negative error code on failure
 */
int tinyaiRenameFile(const char *oldPath, const char *newPath);

/**
 * Get file information
 * 
 * @param path File path
 * @param info File information structure
 * @return 0 on success, negative error code on failure
 */
int tinyaiGetFileInfo(const char *path, TinyAIFileInfo *info);

/**
 * Free file information structure
 * 
 * @param info File information structure
 */
void tinyaiFreeFileInfo(TinyAIFileInfo *info);

/* ----------------- Directory Operations ----------------- */

/**
 * Create a directory
 * 
 * @param path Directory path
 * @return 0 on success, negative error code on failure
 */
int tinyaiCreateDir(const char *path);

/**
 * Delete a directory
 * 
 * @param path Directory path
 * @param recursive Whether to delete recursively
 * @return 0 on success, negative error code on failure
 */
int tinyaiDeleteDir(const char *path, int recursive);

/**
 * Open a directory for reading
 * 
 * @param path Directory path
 * @return Directory handle or NULL on error
 */
TinyAIDir* tinyaiOpenDir(const char *path);

/**
 * Close a directory
 * 
 * @param dir Directory handle
 */
void tinyaiCloseDir(TinyAIDir *dir);

/**
 * Read next directory entry
 * 
 * @param dir Directory handle
 * @param name Buffer to store entry name
 * @param size Buffer size
 * @return 1 if entry read, 0 if no more entries, negative error code on failure
 */
int tinyaiReadDir(TinyAIDir *dir, char *name, size_t size);

/**
 * Get current working directory
 * 
 * @param buffer Buffer to store path
 * @param size Buffer size
 * @return 0 on success, negative error code on failure
 */
int tinyaiGetCWD(char *buffer, size_t size);

/**
 * Change current working directory
 * 
 * @param path New directory path
 * @return 0 on success, negative error code on failure
 */
int tinyaiSetCWD(const char *path);

/* ----------------- Path Operations ----------------- */

/**
 * Get path separator
 * 
 * @return Path separator character ('/' or '\')
 */
char tinyaiGetPathSeparator();

/**
 * Join path components
 * 
 * @param buffer Buffer to store result
 * @param size Buffer size
 * @param count Number of components
 * @param ... Path components
 * @return 0 on success, negative error code on failure
 */
int tinyaiJoinPath(char *buffer, size_t size, int count, ...);

/**
 * Get file name from path
 * 
 * @param path File path
 * @return File name
 */
const char* tinyaiGetFileName(const char *path);

/**
 * Get file extension from path
 * 
 * @param path File path
 * @return File extension
 */
const char* tinyaiGetFileExt(const char *path);

/**
 * Get directory name from path
 * 
 * @param path File path
 * @param buffer Buffer to store result
 * @param size Buffer size
 * @return 0 on success, negative error code on failure
 */
int tinyaiGetDirName(const char *path, char *buffer, size_t size);

/**
 * Get absolute path
 * 
 * @param path Path to resolve
 * @param buffer Buffer to store result
 * @param size Buffer size
 * @return 0 on success, negative error code on failure
 */
int tinyaiGetAbsPath(const char *path, char *buffer, size_t size);

#endif /* TINYAI_IO_H */
