/**
 * @file cache_opt.c
 * @brief Implementation of cache optimization utilities for TinyAI
 */

#include "cache_opt.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <intrin.h>
#include <windows.h>
#elif defined(__APPLE__) || defined(__linux__)
#include <sys/sysctl.h>
#include <unistd.h>
#endif

/* Default cache line size if detection fails */
#define DEFAULT_CACHE_LINE_SIZE 64

/* Default block sizes for various operations */
#define DEFAULT_MATRIX_BLOCK_X 32
#define DEFAULT_MATRIX_BLOCK_Y 32
#define DEFAULT_CONV_BLOCK_X 16
#define DEFAULT_CONV_BLOCK_Y 16
#define DEFAULT_PREFETCH_DISTANCE 8

/* Cache-specific optimization guidelines */
#define L1_CACHE_BLOCK_SIZE_MULTIPLIER 0.25f  /* Block size as fraction of L1 cache */
#define L2_CACHE_BLOCK_SIZE_MULTIPLIER 0.125f /* Block size as fraction of L2 cache */

/**
 * Initialize default cache optimization configuration
 * @return Default configuration structure
 */
TinyAICacheOptConfig tinyai_cache_opt_init_default(void)
{
    TinyAICacheOptConfig config;

    /* Initialize with reasonable defaults */
    config.blockSizeX       = DEFAULT_MATRIX_BLOCK_X;
    config.blockSizeY       = DEFAULT_MATRIX_BLOCK_Y;
    config.prefetchDistance = DEFAULT_PREFETCH_DISTANCE;
    config.enablePrefetch   = true;
    config.enableTiling     = true;

    /* Attempt to tune based on detected cache sizes */
    TinyAICacheInfo cacheInfo = tinyai_get_cache_info();

    if (cacheInfo.l1dCacheSize > 0) {
        /* Adjust block sizes based on L1 cache size */
        size_t blockSize = (size_t)(sqrtf(
            (float)(cacheInfo.l1dCacheSize * L1_CACHE_BLOCK_SIZE_MULTIPLIER) / sizeof(float)));

        /* Ensure block size is a multiple of cache line size */
        if (cacheInfo.cacheLineSize > 0) {
            blockSize = (blockSize / cacheInfo.cacheLineSize) * cacheInfo.cacheLineSize;
        }

        /* Ensure block size is reasonable */
        if (blockSize > 8 && blockSize < 256) {
            config.blockSizeX = blockSize;
            config.blockSizeY = blockSize;
        }
    }

    return config;
}

/**
 * Detect cache information on current system
 * This function attempts to detect cache characteristics on various platforms
 * and falls back to reasonable defaults if detection fails
 */
TinyAICacheInfo tinyai_get_cache_info(void)
{
    TinyAICacheInfo info;
    memset(&info, 0, sizeof(TinyAICacheInfo));

    /* Set default values in case detection fails */
    info.l1dCacheSize    = 32 * 1024;       /* 32 KB */
    info.l2CacheSize     = 256 * 1024;      /* 256 KB */
    info.l3CacheSize     = 4 * 1024 * 1024; /* 4 MB */
    info.cacheLineSize   = DEFAULT_CACHE_LINE_SIZE;
    info.l1Associativity = 8;
    info.l2Associativity = 8;
    info.l3Associativity = 16;

#ifdef _WIN32
    /* Windows cache detection using Windows API and CPUID */
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    info.cacheLineSize = sysInfo.dwPageSize > 0 ? sysInfo.dwPageSize : DEFAULT_CACHE_LINE_SIZE;

    int cpuInfo[4] = {0};
    /* Get cache information using CPUID */
    __cpuid(cpuInfo, 0x80000006);
    info.l2CacheSize = ((cpuInfo[2] >> 16) & 0xFFFF) * 1024; /* L2 size in KB */

    /* Get L1 cache size */
    __cpuid(cpuInfo, 0x80000005);
    info.l1dCacheSize = ((cpuInfo[2] >> 24) & 0xFF) * 1024; /* L1 data cache size in KB */

#elif defined(__APPLE__)
    /* macOS cache detection using sysctl */
    size_t size = sizeof(size_t);
    sysctlbyname("hw.l1dcachesize", &info.l1dCacheSize, &size, NULL, 0);
    sysctlbyname("hw.l2cachesize", &info.l2CacheSize, &size, NULL, 0);
    sysctlbyname("hw.l3cachesize", &info.l3CacheSize, &size, NULL, 0);
    sysctlbyname("hw.cachelinesize", &info.cacheLineSize, &size, NULL, 0);

#elif defined(__linux__)
    /* Linux cache detection - read from /sys/devices/system/cpu/cpu0/cache/ */
    FILE *fp;
    char  path[256];
    char  line[256];
    int   level, type;

    /* Find the data caches */
    for (level = 0; level < 4; level++) {
        sprintf(path, "/sys/devices/system/cpu/cpu0/cache/index%d/level", level);
        fp = fopen(path, "r");
        if (!fp)
            continue;

        fgets(line, sizeof(line), fp);
        int cacheLevel = atoi(line);
        fclose(fp);

        /* Check if this is a data cache */
        sprintf(path, "/sys/devices/system/cpu/cpu0/cache/index%d/type", level);
        fp = fopen(path, "r");
        if (!fp)
            continue;

        fgets(line, sizeof(line), fp);
        fclose(fp);

        /* Skip if not a data cache */
        if (strncmp(line, "Data", 4) != 0 && strncmp(line, "Unified", 7) != 0) {
            continue;
        }

        /* Get size */
        sprintf(path, "/sys/devices/system/cpu/cpu0/cache/index%d/size", level);
        fp = fopen(path, "r");
        if (!fp)
            continue;

        fgets(line, sizeof(line), fp);
        fclose(fp);

        size_t size = atoi(line);
        /* Check if KB or MB */
        if (strstr(line, "K")) {
            size *= 1024;
        }
        else if (strstr(line, "M")) {
            size *= 1024 * 1024;
        }

        /* Store in appropriate field based on level */
        if (cacheLevel == 1) {
            info.l1dCacheSize = size;
        }
        else if (cacheLevel == 2) {
            info.l2CacheSize = size;
        }
        else if (cacheLevel == 3) {
            info.l3CacheSize = size;
        }

        /* Get line size if this is L1 cache */
        if (cacheLevel == 1) {
            sprintf(path, "/sys/devices/system/cpu/cpu0/cache/index%d/coherency_line_size", level);
            fp = fopen(path, "r");
            if (fp) {
                fgets(line, sizeof(line), fp);
                info.cacheLineSize = atoi(line);
                fclose(fp);
            }
        }

        /* Get associativity */
        sprintf(path, "/sys/devices/system/cpu/cpu0/cache/index%d/ways_of_associativity", level);
        fp = fopen(path, "r");
        if (fp) {
            fgets(line, sizeof(line), fp);
            int assoc = atoi(line);

            if (cacheLevel == 1) {
                info.l1Associativity = assoc;
            }
            else if (cacheLevel == 2) {
                info.l2Associativity = assoc;
            }
            else if (cacheLevel == 3) {
                info.l3Associativity = assoc;
            }

            fclose(fp);
        }
    }
#endif

    /* Ensure we have a reasonable cache line size */
    if (info.cacheLineSize == 0) {
        info.cacheLineSize = DEFAULT_CACHE_LINE_SIZE;
    }

    return info;
}

/**
 * Optimize configuration for matrix multiplication
 * @param rows Number of rows in the first matrix
 * @param cols Number of columns in the second matrix
 * @param inner Inner dimension (columns of first, rows of second)
 * @param config Pointer to configuration structure to optimize
 */
void tinyai_cache_opt_matrix_multiply(size_t rows, size_t cols, size_t inner,
                                      TinyAICacheOptConfig *config)
{
    if (!config)
        return;

    /* Get cache information */
    TinyAICacheInfo cacheInfo = tinyai_get_cache_info();

    /* Compute total data size */
    size_t totalSize = (rows * inner + inner * cols + rows * cols) * sizeof(float);
    size_t l1Size    = cacheInfo.l1dCacheSize;
    size_t l2Size    = cacheInfo.l2CacheSize;

    /* Determine optimal block size based on problem size and cache sizes */
    if (totalSize <= l1Size) {
        /* Problem fits in L1 cache, no need for tiling */
        config->enableTiling = false;
    }
    else if (totalSize <= l2Size) {
        /* Problem fits in L2 cache, moderate tiling */
        size_t blockSize =
            (size_t)sqrtf((float)(l1Size * L1_CACHE_BLOCK_SIZE_MULTIPLIER) / sizeof(float) / 3);

        /* Ensure block size is a multiple of cache line size and reasonable */
        blockSize = (blockSize / cacheInfo.cacheLineSize) * cacheInfo.cacheLineSize;
        if (blockSize < 16)
            blockSize = 16;
        if (blockSize > 64)
            blockSize = 64;

        config->blockSizeX   = blockSize;
        config->blockSizeY   = blockSize;
        config->enableTiling = true;
    }
    else {
        /* Problem doesn't fit in L2 cache, aggressive tiling */
        size_t blockSize =
            (size_t)sqrtf((float)(l1Size * L1_CACHE_BLOCK_SIZE_MULTIPLIER) / sizeof(float) / 3);

        /* Ensure block size is a multiple of cache line size and reasonable */
        blockSize = (blockSize / cacheInfo.cacheLineSize) * cacheInfo.cacheLineSize;
        if (blockSize < 16)
            blockSize = 16;
        if (blockSize > 64)
            blockSize = 64;

        config->blockSizeX   = blockSize;
        config->blockSizeY   = blockSize;
        config->enableTiling = true;
    }

    /* Adjust prefetch distance based on problem size */
    if (inner > 512) {
        config->prefetchDistance = 16;
    }
    else if (inner > 128) {
        config->prefetchDistance = 8;
    }
    else {
        config->prefetchDistance = 4;
    }
}

/**
 * Optimize configuration for convolution operations
 * @param inputWidth Width of input feature map
 * @param inputHeight Height of input feature map
 * @param inputChannels Number of input channels
 * @param kernelSize Size of convolution kernel (assume square)
 * @param outputChannels Number of output channels
 * @param config Pointer to configuration structure to optimize
 */
void tinyai_cache_opt_convolution(size_t inputWidth, size_t inputHeight, size_t inputChannels,
                                  size_t kernelSize, size_t outputChannels,
                                  TinyAICacheOptConfig *config)
{
    if (!config)
        return;

    /* Get cache information */
    TinyAICacheInfo cacheInfo = tinyai_get_cache_info();

    /* Compute total data size */
    size_t inputSize    = inputWidth * inputHeight * inputChannels * sizeof(float);
    size_t kernelSize2  = kernelSize * kernelSize;
    size_t weightSize   = kernelSize2 * inputChannels * outputChannels * sizeof(float);
    size_t outputWidth  = inputWidth - kernelSize + 1;
    size_t outputHeight = inputHeight - kernelSize + 1;
    size_t outputSize   = outputWidth * outputHeight * outputChannels * sizeof(float);
    size_t totalSize    = inputSize + weightSize + outputSize;

    size_t l1Size = cacheInfo.l1dCacheSize;

    /* For convolutions, we want block sizes related to output dimensions */
    if (totalSize <= l1Size) {
        /* Small problem that fits in L1 cache */
        config->enableTiling = false;
    }
    else {
        /* Need tiling - calculate block sizes relative to output dimensions */
        size_t maxBlockSize = (size_t)sqrtf(
            (float)(l1Size * 0.3f) /
            (sizeof(float) * (1 + kernelSize2 * inputChannels / outputWidth / outputHeight)));

        /* Ensure block sizes are reasonable */
        if (maxBlockSize < 8)
            maxBlockSize = 8;
        if (maxBlockSize > outputWidth)
            maxBlockSize = outputWidth;
        if (maxBlockSize > outputHeight)
            maxBlockSize = outputHeight;

        config->blockSizeX   = maxBlockSize;
        config->blockSizeY   = maxBlockSize;
        config->enableTiling = true;
    }

    /* Adjust prefetch distance based on kernel size and channels */
    if (kernelSize <= 3 && inputChannels <= 64) {
        config->prefetchDistance = 2; /* Small kernel, shorter prefetch */
    }
    else if (kernelSize <= 5 && inputChannels <= 128) {
        config->prefetchDistance = 4; /* Medium kernel */
    }
    else {
        config->prefetchDistance = 8; /* Large kernel or many channels */
    }
}

/**
 * Perform software prefetch of memory address
 * @param addr Memory address to prefetch
 * @param readWrite 0 for read, 1 for write
 * @param locality Temporal locality hint (0-3, where 3 is highest locality)
 */
void tinyai_prefetch(const void *addr, int readWrite, int locality)
{
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
    /* MSVC intrinsics for x86/x64 */
    switch (locality) {
    case 0:
        _mm_prefetch((const char *)addr, _MM_HINT_NTA);
        break;
    case 1:
        _mm_prefetch((const char *)addr, _MM_HINT_T2);
        break;
    case 2:
        _mm_prefetch((const char *)addr, _MM_HINT_T1);
        break;
    case 3:
        _mm_prefetch((const char *)addr, _MM_HINT_T0);
        break;
    default:
        _mm_prefetch((const char *)addr, _MM_HINT_T0);
        break;
    }
#elif defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))
    /* GCC intrinsics for x86/x64 */
    switch (locality) {
    case 0:
        __builtin_prefetch(addr, readWrite, 0);
        break;
    case 1:
        __builtin_prefetch(addr, readWrite, 1);
        break;
    case 2:
        __builtin_prefetch(addr, readWrite, 2);
        break;
    case 3:
        __builtin_prefetch(addr, readWrite, 3);
        break;
    default:
        __builtin_prefetch(addr, readWrite, 0);
        break;
    }
#else
    /* No prefetch support - do nothing */
    (void)addr;
    (void)readWrite;
    (void)locality;
#endif
}

/**
 * Transpose matrix in a cache-friendly way
 * @param dest Destination matrix
 * @param src Source matrix
 * @param rows Number of rows
 * @param cols Number of columns
 * @param elemSize Size of each element in bytes
 */
void tinyai_transpose_blocked(void *dest, const void *src, size_t rows, size_t cols,
                              size_t elemSize)
{
    /* Get default cache configuration */
    TinyAICacheOptConfig config = tinyai_cache_opt_init_default();

    /* Determine block size */
    size_t blockSize = config.blockSizeX;
    if (blockSize > 32)
        blockSize = 32; /* Cap block size for transpose */

    /* Allocate temporary buffer for block transposition */
    char *buffer = (char *)malloc(blockSize * blockSize * elemSize);
    if (!buffer) {
        /* Fallback to non-blocked implementation if allocation fails */
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                memcpy((char *)dest + (j * rows + i) * elemSize,
                       (char *)src + (i * cols + j) * elemSize, elemSize);
            }
        }
        return;
    }

    /* Blocked transpose implementation */
    for (size_t i = 0; i < rows; i += blockSize) {
        size_t iLimit = (i + blockSize < rows) ? i + blockSize : rows;

        for (size_t j = 0; j < cols; j += blockSize) {
            size_t jLimit = (j + blockSize < cols) ? j + blockSize : cols;

            /* Transpose current block */
            for (size_t bi = i; bi < iLimit; bi++) {
                for (size_t bj = j; bj < jLimit; bj++) {
                    memcpy((char *)dest + (bj * rows + bi) * elemSize,
                           (char *)src + (bi * cols + bj) * elemSize, elemSize);
                }
            }
        }
    }

    /* Free temporary buffer */
    free(buffer);
}
