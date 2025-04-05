/**
 * @file stb_image_loader.c
 * @brief Image loading implementation using STB Image library for TinyAI
 */

#include "image_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Define STB_IMAGE_IMPLEMENTATION in only one source file */
#define STB_IMAGE_IMPLEMENTATION
#include "../../third_party/stb/stb_image.h"

/* Define STB_IMAGE_WRITE_IMPLEMENTATION for saving images */
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../third_party/stb/stb_image_write.h"

/**
 * Load an image from a file using STB Image library
 * @param filepath Path to the image file
 * @return Newly allocated TinyAIImage, or NULL on failure
 */
TinyAIImage *tinyaiImageLoadFromFile(const char *filepath)
{
    if (!filepath) {
        fprintf(stderr, "Error: NULL filepath provided to tinyaiImageLoadFromFile\n");
        return NULL;
    }

    /* Load image data using STB Image */
    int width, height, channels;
    stbi_set_flip_vertically_on_load(1); /* Flip images so that 0,0 is bottom-left */

    unsigned char *data = stbi_load(filepath, &width, &height, &channels, 0);
    if (!data) {
        fprintf(stderr, "Error loading image %s: %s\n", filepath, stbi_failure_reason());
        return NULL;
    }

    /* Convert channels to our format enum */
    TinyAIImageFormat format;
    switch (channels) {
    case 1:
        format = TINYAI_IMAGE_FORMAT_GRAYSCALE;
        break;
    case 3:
        format = TINYAI_IMAGE_FORMAT_RGB;
        break;
    case 4:
        format = TINYAI_IMAGE_FORMAT_RGBA;
        break;
    default:
        fprintf(stderr, "Unsupported number of channels: %d\n", channels);
        stbi_image_free(data);
        return NULL;
    }

    /* Create our image structure */
    TinyAIImage *image = tinyaiImageCreate(width, height, format);
    if (!image) {
        stbi_image_free(data);
        return NULL;
    }

    /* Copy the data */
    memcpy(image->data, data, width * height * channels);

    /* Free STB's image data */
    stbi_image_free(data);

    return image;
}

/**
 * Save an image to a file using STB Image Write
 * @param image The image to save
 * @param filepath Path where to save the image
 * @param format Output format (jpg, png, bmp, tga)
 * @return true on success, false on failure
 */
bool tinyaiImageSaveToFile(const TinyAIImage *image, const char *filepath, const char *format)
{
    if (!image || !filepath || !format) {
        return false;
    }

    int result   = 0;
    int channels = 0;

    /* Determine channels from our format */
    switch (image->format) {
    case TINYAI_IMAGE_FORMAT_GRAYSCALE:
        channels = 1;
        break;
    case TINYAI_IMAGE_FORMAT_RGB:
        channels = 3;
        break;
    case TINYAI_IMAGE_FORMAT_BGR:
        /* Need to convert BGR to RGB for saving */
        {
            unsigned char *rgb_data = (unsigned char *)malloc(image->width * image->height * 3);
            if (!rgb_data) {
                return false;
            }

            for (int i = 0; i < image->width * image->height; i++) {
                rgb_data[i * 3 + 0] = image->data[i * 3 + 2]; /* R <- B */
                rgb_data[i * 3 + 1] = image->data[i * 3 + 1]; /* G stays */
                rgb_data[i * 3 + 2] = image->data[i * 3 + 0]; /* B <- R */
            }

            /* Save with the converted data */
            if (strcmp(format, "png") == 0) {
                result = stbi_write_png(filepath, image->width, image->height, 3, rgb_data,
                                        image->width * 3);
            }
            else if (strcmp(format, "jpg") == 0 || strcmp(format, "jpeg") == 0) {
                result = stbi_write_jpg(filepath, image->width, image->height, 3, rgb_data, 90);
            }
            else if (strcmp(format, "bmp") == 0) {
                result = stbi_write_bmp(filepath, image->width, image->height, 3, rgb_data);
            }
            else if (strcmp(format, "tga") == 0) {
                result = stbi_write_tga(filepath, image->width, image->height, 3, rgb_data);
            }

            free(rgb_data);
            return result != 0;
        }
        break;
    case TINYAI_IMAGE_FORMAT_RGBA:
        channels = 4;
        break;
    default:
        return false;
    }

    /* Save the image in the requested format */
    if (strcmp(format, "png") == 0) {
        result = stbi_write_png(filepath, image->width, image->height, channels, image->data,
                                image->width * channels);
    }
    else if (strcmp(format, "jpg") == 0 || strcmp(format, "jpeg") == 0) {
        result = stbi_write_jpg(filepath, image->width, image->height, channels, image->data, 90);
    }
    else if (strcmp(format, "bmp") == 0) {
        result = stbi_write_bmp(filepath, image->width, image->height, channels, image->data);
    }
    else if (strcmp(format, "tga") == 0) {
        result = stbi_write_tga(filepath, image->width, image->height, channels, image->data);
    }
    else {
        fprintf(stderr, "Unsupported output format: %s\n", format);
        return false;
    }

    return result != 0;
}

/**
 * Load a set of image files (dataset) using STB Image
 * @param filepaths Array of file paths to load
 * @param numImages Number of images in the array
 * @param targetWidth Width to resize all images to (0 for no resize)
 * @param targetHeight Height to resize all images to (0 for no resize)
 * @return Array of loaded images, or NULL on failure
 */
TinyAIImage **tinyaiImageLoadDataset(const char **filepaths, int numImages, int targetWidth,
                                     int targetHeight)
{
    if (!filepaths || numImages <= 0) {
        return NULL;
    }

    /* Allocate array for images */
    TinyAIImage **images = (TinyAIImage **)malloc(numImages * sizeof(TinyAIImage *));
    if (!images) {
        return NULL;
    }

    /* Zero out the array */
    memset(images, 0, numImages * sizeof(TinyAIImage *));

    /* Load each image */
    bool success = true;
    for (int i = 0; i < numImages; i++) {
        /* Load the original image */
        images[i] = tinyaiImageLoadFromFile(filepaths[i]);
        if (!images[i]) {
            success = false;
            break;
        }

        /* If resize is requested */
        if (targetWidth > 0 && targetHeight > 0 &&
            (images[i]->width != targetWidth || images[i]->height != targetHeight)) {

            /* Create a new preprocessed image */
            TinyAIImagePreprocessParams params;
            tinyaiImagePreprocessParamsDefault(&params);
            params.targetWidth  = targetWidth;
            params.targetHeight = targetHeight;

            TinyAIImage *resized = tinyaiImagePreprocess(images[i], &params);
            if (!resized) {
                success = false;
                break;
            }

            /* Replace the original with the resized version */
            tinyaiImageFree(images[i]);
            images[i] = resized;
        }
    }

    /* Clean up on failure */
    if (!success) {
        for (int i = 0; i < numImages; i++) {
            if (images[i]) {
                tinyaiImageFree(images[i]);
            }
        }
        free(images);
        return NULL;
    }

    return images;
}

/**
 * Free a dataset of images
 * @param images Array of images to free
 * @param numImages Number of images in the array
 */
void tinyaiImageFreeDataset(TinyAIImage **images, int numImages)
{
    if (!images) {
        return;
    }

    for (int i = 0; i < numImages; i++) {
        if (images[i]) {
            tinyaiImageFree(images[i]);
        }
    }

    free(images);
}
