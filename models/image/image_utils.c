/**
 * @file image_utils.c
 * @brief Image utility functions for TinyAI
 */

#include "image_model.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Create a new image
 * @param width Image width
 * @param height Image height
 * @param format Image format
 * @return Newly allocated image, or NULL on failure
 */
TinyAIImage *tinyaiImageCreate(int width, int height, TinyAIImageFormat format)
{
    if (width <= 0 || height <= 0) {
        fprintf(stderr, "Invalid image dimensions: %dx%d\n", width, height);
        return NULL;
    }

    /* Determine number of channels based on format */
    int channels;
    switch (format) {
    case TINYAI_IMAGE_FORMAT_GRAYSCALE:
        channels = 1;
        break;
    case TINYAI_IMAGE_FORMAT_RGB:
    case TINYAI_IMAGE_FORMAT_BGR:
        channels = 3;
        break;
    case TINYAI_IMAGE_FORMAT_RGBA:
        channels = 4;
        break;
    default:
        fprintf(stderr, "Unsupported image format\n");
        return NULL;
    }

    /* Allocate image structure */
    TinyAIImage *image = (TinyAIImage *)malloc(sizeof(TinyAIImage));
    if (!image) {
        fprintf(stderr, "Failed to allocate image structure\n");
        return NULL;
    }

    /* Allocate pixel data */
    size_t dataSize = width * height * channels;
    image->data     = (uint8_t *)malloc(dataSize);
    if (!image->data) {
        fprintf(stderr, "Failed to allocate image data (%zu bytes)\n", dataSize);
        free(image);
        return NULL;
    }

    /* Initialize image */
    image->width    = width;
    image->height   = height;
    image->format   = format;
    image->ownsData = true;

    /* Clear pixel data */
    memset(image->data, 0, dataSize);

    return image;
}

/**
 * Free an image
 * @param image The image to free
 */
void tinyaiImageFree(TinyAIImage *image)
{
    if (!image) {
        return;
    }

    /* Free pixel data if we own it */
    if (image->ownsData && image->data) {
        free(image->data);
    }

    /* Free the image structure */
    free(image);
}

/**
 * Create a copy of an image
 * @param image The image to copy
 * @return Newly allocated copy of image, or NULL on failure
 */
TinyAIImage *tinyaiImageCopy(const TinyAIImage *image)
{
    if (!image) {
        return NULL;
    }

    /* Create a new image with the same dimensions and format */
    TinyAIImage *copy = tinyaiImageCreate(image->width, image->height, image->format);
    if (!copy) {
        return NULL;
    }

    /* Determine number of channels */
    int channels;
    switch (image->format) {
    case TINYAI_IMAGE_FORMAT_GRAYSCALE:
        channels = 1;
        break;
    case TINYAI_IMAGE_FORMAT_RGB:
    case TINYAI_IMAGE_FORMAT_BGR:
        channels = 3;
        break;
    case TINYAI_IMAGE_FORMAT_RGBA:
        channels = 4;
        break;
    default:
        fprintf(stderr, "Unsupported image format\n");
        tinyaiImageFree(copy);
        return NULL;
    }

    /* Copy pixel data */
    size_t dataSize = image->width * image->height * channels;
    memcpy(copy->data, image->data, dataSize);

    return copy;
}

/**
 * Resize an image using bilinear interpolation
 * @param image The image to resize
 * @param newWidth New width
 * @param newHeight New height
 * @return Newly allocated resized image, or NULL on failure
 */
TinyAIImage *tinyaiImageResize(const TinyAIImage *image, int newWidth, int newHeight)
{
    if (!image || newWidth <= 0 || newHeight <= 0) {
        return NULL;
    }

    /* Determine number of channels */
    int channels;
    switch (image->format) {
    case TINYAI_IMAGE_FORMAT_GRAYSCALE:
        channels = 1;
        break;
    case TINYAI_IMAGE_FORMAT_RGB:
    case TINYAI_IMAGE_FORMAT_BGR:
        channels = 3;
        break;
    case TINYAI_IMAGE_FORMAT_RGBA:
        channels = 4;
        break;
    default:
        fprintf(stderr, "Unsupported image format\n");
        return NULL;
    }

    /* Create a new image with the new dimensions */
    TinyAIImage *resized = tinyaiImageCreate(newWidth, newHeight, image->format);
    if (!resized) {
        return NULL;
    }

    /* Bilinear interpolation */
    float scaleX = (float)image->width / newWidth;
    float scaleY = (float)image->height / newHeight;

    for (int y = 0; y < newHeight; y++) {
        float origY = y * scaleY;
        int   y1    = (int)origY;
        int   y2    = (y1 + 1 < image->height) ? y1 + 1 : y1;
        float dy    = origY - y1;

        for (int x = 0; x < newWidth; x++) {
            float origX = x * scaleX;
            int   x1    = (int)origX;
            int   x2    = (x1 + 1 < image->width) ? x1 + 1 : x1;
            float dx    = origX - x1;

            for (int c = 0; c < channels; c++) {
                /* Get pixel values at the four corners */
                uint8_t p11 = image->data[(y1 * image->width + x1) * channels + c];
                uint8_t p12 = image->data[(y1 * image->width + x2) * channels + c];
                uint8_t p21 = image->data[(y2 * image->width + x1) * channels + c];
                uint8_t p22 = image->data[(y2 * image->width + x2) * channels + c];

                /* Bilinear interpolation */
                float   top    = p11 * (1 - dx) + p12 * dx;
                float   bottom = p21 * (1 - dx) + p22 * dx;
                uint8_t value  = (uint8_t)(top * (1 - dy) + bottom * dy);

                resized->data[(y * newWidth + x) * channels + c] = value;
            }
        }
    }

    return resized;
}

/**
 * Convert image to grayscale
 * @param image The image to convert
 * @return Newly allocated grayscale image, or NULL on failure
 */
TinyAIImage *tinyaiImageToGrayscale(const TinyAIImage *image)
{
    if (!image) {
        return NULL;
    }

    /* Check if already grayscale */
    if (image->format == TINYAI_IMAGE_FORMAT_GRAYSCALE) {
        return tinyaiImageCopy(image);
    }

    /* Create a new grayscale image */
    TinyAIImage *gray =
        tinyaiImageCreate(image->width, image->height, TINYAI_IMAGE_FORMAT_GRAYSCALE);
    if (!gray) {
        return NULL;
    }

    /* Determine conversion based on source format */
    switch (image->format) {
    case TINYAI_IMAGE_FORMAT_RGB:
        /* Use standard RGB to grayscale conversion */
        for (int i = 0; i < image->width * image->height; i++) {
            uint8_t r = image->data[i * 3 + 0];
            uint8_t g = image->data[i * 3 + 1];
            uint8_t b = image->data[i * 3 + 2];

            /* Weighted average (ITU-R BT.601) */
            gray->data[i] = (uint8_t)(0.299f * r + 0.587f * g + 0.114f * b);
        }
        break;

    case TINYAI_IMAGE_FORMAT_BGR:
        /* Same as RGB but with different channel order */
        for (int i = 0; i < image->width * image->height; i++) {
            uint8_t b = image->data[i * 3 + 0];
            uint8_t g = image->data[i * 3 + 1];
            uint8_t r = image->data[i * 3 + 2];

            /* Weighted average (ITU-R BT.601) */
            gray->data[i] = (uint8_t)(0.299f * r + 0.587f * g + 0.114f * b);
        }
        break;

    case TINYAI_IMAGE_FORMAT_RGBA:
        /* Ignore alpha channel */
        for (int i = 0; i < image->width * image->height; i++) {
            uint8_t r = image->data[i * 4 + 0];
            uint8_t g = image->data[i * 4 + 1];
            uint8_t b = image->data[i * 4 + 2];

            /* Weighted average (ITU-R BT.601) */
            gray->data[i] = (uint8_t)(0.299f * r + 0.587f * g + 0.114f * b);
        }
        break;

    default:
        fprintf(stderr, "Unsupported conversion to grayscale\n");
        tinyaiImageFree(gray);
        return NULL;
    }

    return gray;
}

/**
 * Convert image to float array
 * @param image The image to convert
 * @param output Float array to store result (must be pre-allocated)
 * @param normalize Whether to normalize to 0-1 range
 * @return true on success, false on failure
 */
bool tinyaiImageToFloatArray(const TinyAIImage *image, float *output, bool normalize)
{
    if (!image || !output) {
        return false;
    }

    /* Determine number of channels */
    int channels;
    switch (image->format) {
    case TINYAI_IMAGE_FORMAT_GRAYSCALE:
        channels = 1;
        break;
    case TINYAI_IMAGE_FORMAT_RGB:
    case TINYAI_IMAGE_FORMAT_BGR:
        channels = 3;
        break;
    case TINYAI_IMAGE_FORMAT_RGBA:
        channels = 4;
        break;
    default:
        fprintf(stderr, "Unsupported image format\n");
        return false;
    }

    /* Convert to float */
    int numPixels = image->width * image->height;
    int numValues = numPixels * channels;

    if (normalize) {
        /* Normalize to 0-1 range */
        for (int i = 0; i < numValues; i++) {
            output[i] = image->data[i] / 255.0f;
        }
    }
    else {
        /* Direct conversion */
        for (int i = 0; i < numValues; i++) {
            output[i] = (float)image->data[i];
        }
    }

    return true;
}

/**
 * Set default preprocessing parameters
 * @param params The parameters structure to initialize
 */
void tinyaiImagePreprocessParamsDefault(TinyAIImagePreprocessParams *params)
{
    if (!params) {
        return;
    }

    /* Set default values */
    params->targetWidth  = 224;    /* Common input size for CNNs */
    params->targetHeight = 224;    /* Common input size for CNNs */
    params->meanR        = 127.5f; /* Default mean (centered at 127.5) */
    params->meanG        = 127.5f;
    params->meanB        = 127.5f;
    params->stdR         = 127.5f; /* Default std (scaled by 127.5) */
    params->stdG         = 127.5f;
    params->stdB         = 127.5f;
    params->centerCrop   = false;  /* No center crop by default */
    params->cropRatio    = 0.875f; /* Standard crop ratio if enabled */
}

/**
 * Preprocess an image for model input
 * @param image The image to preprocess
 * @param params Preprocessing parameters
 * @return Newly allocated preprocessed image, or NULL on failure
 */
TinyAIImage *tinyaiImagePreprocess(const TinyAIImage                 *image,
                                   const TinyAIImagePreprocessParams *params)
{
    if (!image || !params) {
        return NULL;
    }

    /* Ensure image is in RGB format */
    TinyAIImage *rgbImage = NULL;
    if (image->format != TINYAI_IMAGE_FORMAT_RGB) {
        /* Convert to RGB if needed */
        switch (image->format) {
        case TINYAI_IMAGE_FORMAT_GRAYSCALE:
            /* Grayscale to RGB: duplicate channel */
            rgbImage = tinyaiImageCreate(image->width, image->height, TINYAI_IMAGE_FORMAT_RGB);
            if (!rgbImage) {
                return NULL;
            }

            for (int i = 0; i < image->width * image->height; i++) {
                uint8_t gray              = image->data[i];
                rgbImage->data[i * 3 + 0] = gray;
                rgbImage->data[i * 3 + 1] = gray;
                rgbImage->data[i * 3 + 2] = gray;
            }
            break;

        case TINYAI_IMAGE_FORMAT_BGR:
            /* BGR to RGB: swap channels */
            rgbImage = tinyaiImageCreate(image->width, image->height, TINYAI_IMAGE_FORMAT_RGB);
            if (!rgbImage) {
                return NULL;
            }

            for (int i = 0; i < image->width * image->height; i++) {
                rgbImage->data[i * 3 + 0] = image->data[i * 3 + 2]; /* R <- B */
                rgbImage->data[i * 3 + 1] = image->data[i * 3 + 1]; /* G stays */
                rgbImage->data[i * 3 + 2] = image->data[i * 3 + 0]; /* B <- R */
            }
            break;

        case TINYAI_IMAGE_FORMAT_RGBA:
            /* RGBA to RGB: drop alpha */
            rgbImage = tinyaiImageCreate(image->width, image->height, TINYAI_IMAGE_FORMAT_RGB);
            if (!rgbImage) {
                return NULL;
            }

            for (int i = 0; i < image->width * image->height; i++) {
                rgbImage->data[i * 3 + 0] = image->data[i * 4 + 0]; /* R */
                rgbImage->data[i * 3 + 1] = image->data[i * 4 + 1]; /* G */
                rgbImage->data[i * 3 + 2] = image->data[i * 4 + 2]; /* B */
            }
            break;

        default:
            fprintf(stderr, "Unsupported image format for preprocessing\n");
            return NULL;
        }
    }
    else {
        /* Already RGB, just make a copy */
        rgbImage = tinyaiImageCopy(image);
        if (!rgbImage) {
            return NULL;
        }
    }

    /* Apply center crop if requested */
    TinyAIImage *croppedImage = NULL;
    if (params->centerCrop) {
        int cropWidth  = (int)(rgbImage->width * params->cropRatio);
        int cropHeight = (int)(rgbImage->height * params->cropRatio);

        /* Ensure crop dimensions are valid */
        cropWidth  = (cropWidth <= 0) ? rgbImage->width : cropWidth;
        cropHeight = (cropHeight <= 0) ? rgbImage->height : cropHeight;

        /* Calculate crop coordinates */
        int x = (rgbImage->width - cropWidth) / 2;
        int y = (rgbImage->height - cropHeight) / 2;

        /* Create a new image for the cropped result */
        croppedImage = tinyaiImageCreate(cropWidth, cropHeight, TINYAI_IMAGE_FORMAT_RGB);
        if (!croppedImage) {
            tinyaiImageFree(rgbImage);
            return NULL;
        }

        /* Copy cropped region */
        for (int cy = 0; cy < cropHeight; cy++) {
            for (int cx = 0; cx < cropWidth; cx++) {
                int srcIdx = ((y + cy) * rgbImage->width + (x + cx)) * 3;
                int dstIdx = (cy * cropWidth + cx) * 3;

                croppedImage->data[dstIdx + 0] = rgbImage->data[srcIdx + 0];
                croppedImage->data[dstIdx + 1] = rgbImage->data[srcIdx + 1];
                croppedImage->data[dstIdx + 2] = rgbImage->data[srcIdx + 2];
            }
        }

        /* Free the RGB image as we don't need it anymore */
        tinyaiImageFree(rgbImage);
        rgbImage = croppedImage;
    }

    /* Resize to target dimensions if needed */
    TinyAIImage *resizedImage = NULL;
    if (rgbImage->width != params->targetWidth || rgbImage->height != params->targetHeight) {
        resizedImage = tinyaiImageResize(rgbImage, params->targetWidth, params->targetHeight);
        if (!resizedImage) {
            tinyaiImageFree(rgbImage);
            return NULL;
        }

        /* Free the previous image as we don't need it anymore */
        tinyaiImageFree(rgbImage);
        rgbImage = resizedImage;
    }

    /* Normalize pixel values (applies mean and std) */
    for (int i = 0; i < rgbImage->width * rgbImage->height; i++) {
        /* Apply normalization to each channel */
        uint8_t r = rgbImage->data[i * 3 + 0];
        uint8_t g = rgbImage->data[i * 3 + 1];
        uint8_t b = rgbImage->data[i * 3 + 2];

        /* Convert to float, subtract mean, and divide by std */
        float normalized_r = (r - params->meanR) / params->stdR;
        float normalized_g = (g - params->meanG) / params->stdG;
        float normalized_b = (b - params->meanB) / params->stdB;

        /* Convert back to uint8_t with clamping */
        rgbImage->data[i * 3 + 0] = (uint8_t)fmaxf(0, fminf(255, normalized_r + 128));
        rgbImage->data[i * 3 + 1] = (uint8_t)fmaxf(0, fminf(255, normalized_g + 128));
        rgbImage->data[i * 3 + 2] = (uint8_t)fmaxf(0, fminf(255, normalized_b + 128));
    }

    return rgbImage;
}
