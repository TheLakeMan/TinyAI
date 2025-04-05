/**
 * TinyAI I/O System Implementation
 */

#include "io.h"
#include "memory.h" // Use TinyAI memory functions if available/needed

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h> // For stat, mkdir
#include <sys/types.h>
#include <stdarg.h> // For va_list in tinyaiJoinPath

// Platform-specific includes
#ifdef _WIN32
#include <windows.h>
#include <direct.h> // For _mkdir, _rmdir, _getcwd, _chdir
#define stat _stat
#define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
#define S_ISREG(m) (((m) & S_IFMT) == S_IFREG)
#define getcwd _getcwd
#define chdir _chdir
#define mkdir(path, mode) _mkdir(path) // mode is ignored on Windows _mkdir
#define rmdir _rmdir
#define PATH_SEPARATOR '\\'
#else // POSIX
#include <unistd.h> // For rmdir, getcwd, chdir, access
#include <dirent.h> // For opendir, readdir, closedir
#define PATH_SEPARATOR '/'
#endif

/* ----------------- Internal State & Error Handling ----------------- */

static int g_lastIOError = TINYAI_IO_SUCCESS;

// Helper to map errno to TinyAI error codes
static int mapErrnoToTinyAIError(int err) {
    switch (err) {
        case 0: return TINYAI_IO_SUCCESS;
        case ENOENT: return TINYAI_IO_NOT_FOUND;
        case EACCES: return TINYAI_IO_ACCESS;
        case EEXIST: return TINYAI_IO_EXISTS;
        case EINVAL: return TINYAI_IO_INVALID;
        case ENOMEM: return TINYAI_IO_NO_MEMORY;
        // Add more mappings as needed
        default: return TINYAI_IO_ERROR; // Generic error
    }
}

// Helper to set the last error
static void setLastError(int tinyaiError) {
    g_lastIOError = tinyaiError;
}

static void setLastErrorFromErrno() {
    g_lastIOError = mapErrnoToTinyAIError(errno);
}

/* ----------------- Type Definitions ----------------- */

// File handle structure wraps standard FILE*
struct TinyAIFile {
    FILE *fp;
    int mode; // Store the mode flags used to open
};

// Directory handle structure (Platform-dependent)
#ifdef _WIN32
struct TinyAIDir {
    HANDLE hFind;
    WIN32_FIND_DATA findData;
    char *path; // Store path for subsequent FindNextFile calls
    int firstEntry; // Flag to handle the first FindFirstFile call
};
#else // POSIX
struct TinyAIDir {
    DIR *dp;
    struct dirent *entry;
};
#endif


/* ----------------- System Functions ----------------- */

int tinyaiIOInit() {
    // Nothing specific to initialize for basic stdio usage
    g_lastIOError = TINYAI_IO_SUCCESS;
    return 0;
}

void tinyaiIOCleanup() {
    // Nothing specific to clean up
}

int tinyaiIOGetLastError() {
    return g_lastIOError;
}

const char* tinyaiIOGetErrorString(int error) {
    switch (error) {
        case TINYAI_IO_SUCCESS: return "Success";
        case TINYAI_IO_ERROR: return "Generic I/O error";
        case TINYAI_IO_NOT_FOUND: return "File not found";
        case TINYAI_IO_ACCESS: return "Permission denied";
        case TINYAI_IO_EXISTS: return "File already exists";
        case TINYAI_IO_INVALID: return "Invalid argument or operation";
        case TINYAI_IO_NO_MEMORY: return "Out of memory";
        case TINYAI_IO_EOF: return "End of file";
        default: return "Unknown error code";
    }
}

/* ----------------- File Operations ----------------- */

TinyAIFile* tinyaiOpenFile(const char *path, int mode) {
    char fopen_mode[5] = {0}; // Max "rb+" + null terminator
    int current_pos = 0;

    if (mode & TINYAI_FILE_READ && mode & TINYAI_FILE_WRITE) {
        fopen_mode[current_pos++] = 'r';
        fopen_mode[current_pos++] = '+';
    } else if (mode & TINYAI_FILE_WRITE) {
        fopen_mode[current_pos++] = 'w';
         if (mode & TINYAI_FILE_READ) fopen_mode[current_pos++] = '+';
    } else if (mode & TINYAI_FILE_APPEND) {
        fopen_mode[current_pos++] = 'a';
         if (mode & TINYAI_FILE_READ) fopen_mode[current_pos++] = '+';
    } else if (mode & TINYAI_FILE_READ) {
        fopen_mode[current_pos++] = 'r';
    } else {
        setLastError(TINYAI_IO_INVALID); // No read/write/append specified
        return NULL;
    }

    if (mode & TINYAI_FILE_BINARY) {
        fopen_mode[current_pos++] = 'b';
    }
    
    // Note: TINYAI_FILE_CREATE and TINYAI_FILE_TRUNCATE are implicitly handled by 'w' and 'a' modes.
    // 'w' creates or truncates. 'a' creates or appends. 'r+' requires file to exist.
    // More complex logic could be added if exact POSIX open() flag behavior is needed.

    FILE *fp = fopen(path, fopen_mode);
    if (!fp) {
        setLastErrorFromErrno();
        return NULL;
    }

    // Allocate TinyAIFile structure
    // Use TINYAI_MALLOC if memory tracking is enabled and integrated
    TinyAIFile *taiFile = (TinyAIFile*)malloc(sizeof(TinyAIFile)); 
    if (!taiFile) {
        fclose(fp);
        setLastError(TINYAI_IO_NO_MEMORY);
        return NULL;
    }

    taiFile->fp = fp;
    taiFile->mode = mode;
    setLastError(TINYAI_IO_SUCCESS);
    return taiFile;
}

void tinyaiCloseFile(TinyAIFile *file) {
    if (file && file->fp) {
        fclose(file->fp);
        // Use TINYAI_FREE if memory tracking is enabled and integrated
        free(file); 
    }
}

int64_t tinyaiReadFile(TinyAIFile *file, void *buffer, size_t size) {
    if (!file || !file->fp || !buffer) {
        setLastError(TINYAI_IO_INVALID);
        return TINYAI_IO_INVALID;
    }

    size_t bytesRead = fread(buffer, 1, size, file->fp);
    if (bytesRead < size) {
        if (feof(file->fp)) {
            // Reached EOF, return bytes read, set EOF state if bytesRead is 0?
            // Standard fread returns partial read on EOF.
             setLastError(TINYAI_IO_EOF); // Indicate EOF was reached
             // return bytesRead; // Return actual bytes read before EOF
        } else if (ferror(file->fp)) {
            setLastErrorFromErrno(); // Or just TINYAI_IO_ERROR
            return g_lastIOError;
        }
        // If bytesRead < size and not EOF and not error, it's unexpected.
    } else {
         setLastError(TINYAI_IO_SUCCESS);
    }
    return (int64_t)bytesRead;
}

int64_t tinyaiWriteFile(TinyAIFile *file, const void *buffer, size_t size) {
     if (!file || !file->fp || !buffer) {
        setLastError(TINYAI_IO_INVALID);
        return TINYAI_IO_INVALID;
    }
     if (!(file->mode & (TINYAI_FILE_WRITE | TINYAI_FILE_APPEND))) {
         setLastError(TINYAI_IO_ACCESS); // Opened read-only
         return TINYAI_IO_ACCESS;
     }

    size_t bytesWritten = fwrite(buffer, 1, size, file->fp);
     if (bytesWritten < size) {
         // Error occurred
         setLastErrorFromErrno(); // Or just TINYAI_IO_ERROR
         return g_lastIOError;
     }
    
    setLastError(TINYAI_IO_SUCCESS);
    return (int64_t)bytesWritten;
}

int tinyaiReadLine(TinyAIFile *file, char *buffer, size_t size) {
    if (!file || !file->fp || !buffer || size == 0) {
        setLastError(TINYAI_IO_INVALID);
        return TINYAI_IO_INVALID;
    }

    char *result = fgets(buffer, (int)size, file->fp);
    if (result == NULL) {
        if (feof(file->fp)) {
            setLastError(TINYAI_IO_EOF);
            buffer[0] = '\0'; // Ensure buffer is empty on EOF
            return TINYAI_IO_EOF; // Return specific EOF code
        } else {
            setLastErrorFromErrno(); // Or TINYAI_IO_ERROR
            return g_lastIOError;
        }
    }
    
    // Remove trailing newline if present and buffer has space
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len - 1] == '\n') {
        buffer[len - 1] = '\0';
        len--;
         if (len > 0 && buffer[len - 1] == '\r') { // Handle CRLF
             buffer[len - 1] = '\0';
             len--;
         }
    }

    setLastError(TINYAI_IO_SUCCESS);
    return (int)len; // Return number of bytes read (excluding null terminator)
}


int64_t tinyaiSeekFile(TinyAIFile *file, int64_t offset, int whence) {
    if (!file || !file->fp) {
        setLastError(TINYAI_IO_INVALID);
        return TINYAI_IO_INVALID;
    }

    int fseek_whence;
    switch (whence) {
        case 0: fseek_whence = SEEK_SET; break;
        case 1: fseek_whence = SEEK_CUR; break;
        case 2: fseek_whence = SEEK_END; break;
        default:
            setLastError(TINYAI_IO_INVALID);
            return TINYAI_IO_INVALID;
    }

    // fseek returns 0 on success
    if (fseek(file->fp, (long)offset, fseek_whence) != 0) {
        setLastErrorFromErrno(); // Or TINYAI_IO_ERROR
        return g_lastIOError;
    }

    // Return the new position after seeking
    return tinyaiTellFile(file);
}

int64_t tinyaiTellFile(TinyAIFile *file) {
    if (!file || !file->fp) {
        setLastError(TINYAI_IO_INVALID);
        return TINYAI_IO_INVALID;
    }
    long pos = ftell(file->fp);
    if (pos == -1L) {
        setLastErrorFromErrno(); // Or TINYAI_IO_ERROR
        return g_lastIOError;
    }
    setLastError(TINYAI_IO_SUCCESS);
    return (int64_t)pos;
}

int tinyaiFlushFile(TinyAIFile *file) {
    if (!file || !file->fp) {
        setLastError(TINYAI_IO_INVALID);
        return TINYAI_IO_INVALID;
    }
    if (fflush(file->fp) != 0) {
        setLastErrorFromErrno(); // Or TINYAI_IO_ERROR
        return g_lastIOError;
    }
    setLastError(TINYAI_IO_SUCCESS);
    return TINYAI_IO_SUCCESS;
}

int tinyaiEOF(TinyAIFile *file) {
     if (!file || !file->fp) {
        setLastError(TINYAI_IO_INVALID);
        return TINYAI_IO_INVALID; // Indicate error
    }
    int eof_status = feof(file->fp);
    setLastError(TINYAI_IO_SUCCESS); // Checking EOF is not an error itself
    return eof_status ? 1 : 0; // Return 1 if EOF, 0 otherwise
}


/* ----------------- File System Operations ----------------- */

int tinyaiFileExists(const char *path) {
    struct stat buffer;
    if (stat(path, &buffer) == 0) {
        setLastError(TINYAI_IO_SUCCESS);
        return 1; // Exists
    } else {
        if (errno == ENOENT) {
            setLastError(TINYAI_IO_NOT_FOUND);
            return 0; // Does not exist
        } else {
            setLastErrorFromErrno(); // Other error (e.g., permission)
            return g_lastIOError; // Return error code
        }
    }
}

int tinyaiDeleteFile(const char *path) {
    if (remove(path) == 0) {
        setLastError(TINYAI_IO_SUCCESS);
        return TINYAI_IO_SUCCESS;
    } else {
        setLastErrorFromErrno();
        return g_lastIOError;
    }
}

int tinyaiRenameFile(const char *oldPath, const char *newPath) {
    if (rename(oldPath, newPath) == 0) {
        setLastError(TINYAI_IO_SUCCESS);
        return TINYAI_IO_SUCCESS;
    } else {
        setLastErrorFromErrno();
        return g_lastIOError;
    }
}

int tinyaiGetFileInfo(const char *path, TinyAIFileInfo *info) {
    if (!path || !info) {
         setLastError(TINYAI_IO_INVALID);
         return TINYAI_IO_INVALID;
    }
    
    struct stat statbuf;
    if (stat(path, &statbuf) != 0) {
        setLastErrorFromErrno();
        return g_lastIOError;
    }

    // Allocate memory for the path copy
    // Use TINYAI_MALLOC if integrated
    info->path = (char*)malloc(strlen(path) + 1); 
    if (!info->path) {
        setLastError(TINYAI_IO_NO_MEMORY);
        return TINYAI_IO_NO_MEMORY;
    }
    strcpy(info->path, path);

    info->size = (uint64_t)statbuf.st_size;
    info->modTime = (uint64_t)statbuf.st_mtime;
    info->mode = (uint32_t)statbuf.st_mode; // Keep platform mode bits
    info->isDirectory = S_ISDIR(statbuf.st_mode);

    setLastError(TINYAI_IO_SUCCESS);
    return TINYAI_IO_SUCCESS;
}

void tinyaiFreeFileInfo(TinyAIFileInfo *info) {
    if (info && info->path) {
        // Use TINYAI_FREE if integrated
        free(info->path); 
        info->path = NULL; // Prevent double free
    }
    // No need to free the struct itself if allocated on stack
}


/* ----------------- Directory Operations (Basic POSIX/Windows Stubs) ----------------- */

int tinyaiCreateDir(const char *path) {
    // Note: mode 0777 is common for POSIX, ignored by Windows _mkdir
    if (mkdir(path, 0777) == 0) {
        setLastError(TINYAI_IO_SUCCESS);
        return TINYAI_IO_SUCCESS;
    } else {
        // Check if it already exists (EEXIST is often success for mkdir)
        if (errno == EEXIST) {
             // Verify it's actually a directory
             struct stat st;
             if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) {
                 setLastError(TINYAI_IO_SUCCESS); // Treat existing dir as success
                 return TINYAI_IO_SUCCESS;
             }
        }
        setLastErrorFromErrno();
        return g_lastIOError;
    }
}

int tinyaiDeleteDir(const char *path, int recursive) {
    // Recursive delete is complex and platform-specific.
    // Provide non-recursive only for now.
    if (recursive) {
        fprintf(stderr, "Warning: Recursive directory delete not yet implemented in tinyaiDeleteDir.\n");
        setLastError(TINYAI_IO_INVALID); // Or a specific "not implemented" error
        return TINYAI_IO_INVALID; 
    }

    if (rmdir(path) == 0) {
        setLastError(TINYAI_IO_SUCCESS);
        return TINYAI_IO_SUCCESS;
    } else {
        setLastErrorFromErrno();
        return g_lastIOError;
    }
}

TinyAIDir* tinyaiOpenDir(const char *path) {
#ifdef _WIN32
    // Windows implementation using FindFirstFile/FindNextFile
    TinyAIDir *dir = (TinyAIDir*)malloc(sizeof(TinyAIDir));
    if (!dir) { setLastError(TINYAI_IO_NO_MEMORY); return NULL; }

    size_t pathLen = strlen(path);
    // Append \*.* for FindFirstFile
    dir->path = (char*)malloc(pathLen + 5); // path + "\\*.*" + null
    if (!dir->path) { free(dir); setLastError(TINYAI_IO_NO_MEMORY); return NULL; }
    strcpy(dir->path, path);
    // Ensure trailing separator
    if (pathLen > 0 && dir->path[pathLen - 1] != PATH_SEPARATOR) {
        dir->path[pathLen] = PATH_SEPARATOR;
        dir->path[pathLen + 1] = '\0';
    }
    strcat(dir->path, "*.*");

    dir->hFind = FindFirstFile(dir->path, &dir->findData);
    if (dir->hFind == INVALID_HANDLE_VALUE) {
        free(dir->path);
        free(dir);
        setLastErrorFromErrno(); // Map GetLastError() if needed
        return NULL;
    }
    dir->firstEntry = 1;
    setLastError(TINYAI_IO_SUCCESS);
    return dir;
#else // POSIX
    DIR *dp = opendir(path);
    if (!dp) {
        setLastErrorFromErrno();
        return NULL;
    }
    TinyAIDir *dir = (TinyAIDir*)malloc(sizeof(TinyAIDir));
     if (!dir) { 
         closedir(dp); 
         setLastError(TINYAI_IO_NO_MEMORY); 
         return NULL; 
     }
    dir->dp = dp;
    dir->entry = NULL;
    setLastError(TINYAI_IO_SUCCESS);
    return dir;
#endif
}

void tinyaiCloseDir(TinyAIDir *dir) {
    if (!dir) return;
#ifdef _WIN32
    if (dir->hFind != INVALID_HANDLE_VALUE) {
        FindClose(dir->hFind);
    }
    free(dir->path);
    free(dir);
#else // POSIX
    if (dir->dp) {
        closedir(dir->dp);
    }
    free(dir);
#endif
}

int tinyaiReadDir(TinyAIDir *dir, char *name, size_t size) {
     if (!dir || !name || size == 0) {
         setLastError(TINYAI_IO_INVALID);
         return TINYAI_IO_INVALID;
     }
#ifdef _WIN32
    if (dir->hFind == INVALID_HANDLE_VALUE) {
         setLastError(TINYAI_IO_INVALID); // Already closed or error
         return TINYAI_IO_INVALID;
    }

    // Handle the first entry from FindFirstFile
    if (dir->firstEntry) {
        dir->firstEntry = 0; // Consume the first entry
         // Skip "." and ".." directories
        while (strcmp(dir->findData.cFileName, ".") == 0 || strcmp(dir->findData.cFileName, "..") == 0) {
             if (!FindNextFile(dir->hFind, &dir->findData)) {
                 if (GetLastError() == ERROR_NO_MORE_FILES) {
                     setLastError(TINYAI_IO_SUCCESS); // No more entries is not an error here
                     return 0; // No more entries
                 } else {
                     setLastErrorFromErrno(); // Map GetLastError()
                     return g_lastIOError;
                 }
             }
        }
        strncpy(name, dir->findData.cFileName, size - 1);
        name[size - 1] = '\0'; // Ensure null termination
        setLastError(TINYAI_IO_SUCCESS);
        return 1; // Entry read
    }

    // Subsequent entries using FindNextFile
    while (FindNextFile(dir->hFind, &dir->findData)) {
         // Skip "." and ".." directories
         if (strcmp(dir->findData.cFileName, ".") != 0 && strcmp(dir->findData.cFileName, "..") != 0) {
             strncpy(name, dir->findData.cFileName, size - 1);
             name[size - 1] = '\0'; // Ensure null termination
             setLastError(TINYAI_IO_SUCCESS);
             return 1; // Entry read
         }
    }

    // Check why FindNextFile failed
    if (GetLastError() == ERROR_NO_MORE_FILES) {
        setLastError(TINYAI_IO_SUCCESS); // No more entries is not an error here
        return 0; // No more entries
    } else {
        setLastErrorFromErrno(); // Map GetLastError()
        return g_lastIOError;
    }
#else // POSIX
    if (!dir->dp) {
         setLastError(TINYAI_IO_INVALID);
         return TINYAI_IO_INVALID;
    }
    
    errno = 0; // Reset errno before calling readdir
    while ((dir->entry = readdir(dir->dp)) != NULL) {
        // Skip "." and ".."
        if (strcmp(dir->entry->d_name, ".") == 0 || strcmp(dir->entry->d_name, "..") == 0) {
            continue;
        }
        strncpy(name, dir->entry->d_name, size - 1);
        name[size - 1] = '\0'; // Ensure null termination
        setLastError(TINYAI_IO_SUCCESS);
        return 1; // Entry read
    }

    // Check if readdir finished or encountered an error
    if (errno != 0) {
        setLastErrorFromErrno();
        return g_lastIOError; // Error reading directory
    } else {
        setLastError(TINYAI_IO_SUCCESS); // End of directory
        return 0; // No more entries
    }
#endif
}


int tinyaiGetCWD(char *buffer, size_t size) {
    if (getcwd(buffer, size) != NULL) {
        setLastError(TINYAI_IO_SUCCESS);
        return TINYAI_IO_SUCCESS;
    } else {
        setLastErrorFromErrno();
        return g_lastIOError;
    }
}

int tinyaiSetCWD(const char *path) {
    if (chdir(path) == 0) {
        setLastError(TINYAI_IO_SUCCESS);
        return TINYAI_IO_SUCCESS;
    } else {
        setLastErrorFromErrno();
        return g_lastIOError;
    }
}

/* ----------------- Path Operations (Basic Implementations) ----------------- */

char tinyaiGetPathSeparator() {
    return PATH_SEPARATOR;
}

int tinyaiJoinPath(char *buffer, size_t size, int count, ...) {
    if (!buffer || size == 0 || count <= 0) {
        setLastError(TINYAI_IO_INVALID);
        return TINYAI_IO_INVALID;
    }

    va_list args;
    va_start(args, count);

    buffer[0] = '\0';
    size_t currentLen = 0;
    char separator[2] = {PATH_SEPARATOR, '\0'};

    for (int i = 0; i < count; ++i) {
        const char *component = va_arg(args, const char*);
        if (!component) continue; // Skip null components

        size_t componentLen = strlen(component);
        if (componentLen == 0) continue; // Skip empty components

        // Calculate required length
        // Need space for component, potentially a separator, and null terminator
        size_t needed = componentLen + (currentLen > 0 ? 1 : 0) + 1; 

        if (currentLen + needed > size) {
            va_end(args);
            setLastError(TINYAI_IO_NO_MEMORY); // Buffer too small (treat as memory error)
            // Consider adding a TINYAI_IO_BUFFER_TOO_SMALL error code
            return TINYAI_IO_NO_MEMORY; 
        }

        // Add separator if not the first component and buffer not empty
        if (currentLen > 0) {
            // Avoid double separators if component starts with one or buffer ends with one
            if (buffer[currentLen - 1] != PATH_SEPARATOR && component[0] != PATH_SEPARATOR) {
                 strcat(buffer, separator);
                 currentLen++;
            } else if (buffer[currentLen - 1] == PATH_SEPARATOR && component[0] == PATH_SEPARATOR) {
                 component++; // Skip separator in component
                 componentLen--;
            }
        }
        
        // Concatenate component, handling potential leading separator skip
        strcat(buffer, component);
        currentLen += componentLen;
    }

    va_end(args);
    setLastError(TINYAI_IO_SUCCESS);
    return TINYAI_IO_SUCCESS;
}


const char* tinyaiGetFileName(const char *path) {
    if (!path) return NULL;
    const char *lastSep = strrchr(path, PATH_SEPARATOR);
#ifdef _WIN32
    // Also check for forward slash on Windows
    const char *lastForwardSep = strrchr(path, '/');
    if (lastForwardSep > lastSep) {
        lastSep = lastForwardSep;
    }
#endif
    return (lastSep == NULL) ? path : lastSep + 1;
}

const char* tinyaiGetFileExt(const char *path) {
     if (!path) return NULL;
     const char *fileName = tinyaiGetFileName(path); // Get filename part first
     const char *lastDot = strrchr(fileName, '.');
     // Ensure dot is not the first character of the filename
     return (lastDot == NULL || lastDot == fileName) ? NULL : lastDot + 1;
}

int tinyaiGetDirName(const char *path, char *buffer, size_t size) {
    if (!path || !buffer || size == 0) {
        setLastError(TINYAI_IO_INVALID);
        return TINYAI_IO_INVALID;
    }

    const char *fileName = tinyaiGetFileName(path);
    if (fileName == path) { // No separator found, path is just a filename
        // Return "." for current directory? Or empty string? Let's go with "."
        if (size < 2) {
             setLastError(TINYAI_IO_NO_MEMORY); // Buffer too small
             return TINYAI_IO_NO_MEMORY;
        }
        strcpy(buffer, ".");
    } else {
        size_t dirLen = (size_t)(fileName - path); // Length including trailing separator
        if (dirLen == 0) { // Path starts with separator (e.g., "/file")
             if (size < 2) {
                 setLastError(TINYAI_IO_NO_MEMORY);
                 return TINYAI_IO_NO_MEMORY;
             }
             buffer[0] = PATH_SEPARATOR;
             buffer[1] = '\0';
        } else {
             // Remove trailing separator if present and it's not the root dir separator
             if (dirLen > 1 && path[dirLen - 1] == PATH_SEPARATOR) {
                 dirLen--;
             }
             if (dirLen + 1 > size) { // Need space for null terminator
                 setLastError(TINYAI_IO_NO_MEMORY);
                 return TINYAI_IO_NO_MEMORY;
             }
             strncpy(buffer, path, dirLen);
             buffer[dirLen] = '\0';
        }
    }
    setLastError(TINYAI_IO_SUCCESS);
    return TINYAI_IO_SUCCESS;
}

int tinyaiGetAbsPath(const char *path, char *buffer, size_t size) {
    if (!path || !buffer || size == 0) {
        setLastError(TINYAI_IO_INVALID);
        return TINYAI_IO_INVALID;
    }

#ifdef _WIN32
    // Use Windows API GetFullPathName
    DWORD result = GetFullPathName(path, (DWORD)size, buffer, NULL);
    if (result == 0) {
        // Error occurred
        setLastErrorFromErrno(); // Map GetLastError() if needed
        return g_lastIOError;
    } else if (result > size) {
        // Buffer too small
        setLastError(TINYAI_IO_NO_MEMORY);
        return TINYAI_IO_NO_MEMORY;
    } else {
        setLastError(TINYAI_IO_SUCCESS);
        return TINYAI_IO_SUCCESS;
    }
#else // POSIX
    // Use POSIX realpath
    char *real_path = realpath(path, buffer); // realpath resolves symlinks etc.
    if (real_path == NULL) {
        // If buffer is NULL, realpath allocates memory. If buffer is provided,
        // it uses the buffer. Error could be ENOENT, EACCES, ENAMETOOLONG, etc.
        // or ENOMEM if buffer is NULL and malloc fails.
        // If buffer is provided and too small, it returns NULL and sets errno to ERANGE.
        if (errno == ERANGE && buffer != NULL) {
             setLastError(TINYAI_IO_NO_MEMORY); // Treat ERANGE as buffer too small
        } else {
             setLastErrorFromErrno();
        }
        return g_lastIOError;
    }
    // If realpath succeeded and used our buffer, real_path == buffer.
    // If realpath allocated memory (buffer was NULL), we'd need to copy and free.
    // Since we provide the buffer, we assume success means buffer is filled.
    setLastError(TINYAI_IO_SUCCESS);
    return TINYAI_IO_SUCCESS;
#endif
}
