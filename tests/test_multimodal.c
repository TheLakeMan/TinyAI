/**
 * @file test_multimodal.c
 * @brief Test suite for TinyAI multimodal capabilities
 */

#include "../models/image/image_model.h"
#include "../models/multimodal/fusion.h"
#include "../models/multimodal/multimodal_model.h"
#include "../models/text/generate.h"
#include "../utils/benchmark.h"
#include "../utils/quantize.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Test constants */
#define TEST_IMAGE_PATH "data/test_image.jpg"
#define TEST_VISION_MODEL_PATH "models/test_resnet.json"
#define TEST_VISION_WEIGHTS_PATH "models/test_resnet.bin"
#define TEST_TEXT_MODEL_PATH "models/test_decoder.json"
#define TEST_TEXT_WEIGHTS_PATH "models/test_decoder.bin"
#define TEST_TOKENIZER_PATH "data/tiny_vocab.tok"

/* Maximum length for test strings */
#define MAX_TEST_STR_LEN 1024

/* Test feature size */
#define TEST_FEATURE_SIZE 256

/* Test text output length */
#define TEST_OUTPUT_LENGTH 50

/* Forward declarations for test functions */
static void test_multimodal_model_creation();
static void test_multimodal_feature_fusion();
static void test_cross_attention();
static void test_multimodal_end_to_end();
static void test_multimodal_quantization();
static void test_multimodal_simd_acceleration();
static void test_feature_fusion_methods();

/* Utility function to create a simple test image */
static TinyAIImage *create_test_image()
{
    /* Create a small test image (RGB, 32x32) */
    const int width    = 32;
    const int height   = 32;
    const int channels = 3;

    TinyAIImage *image = tinyaiImageCreate(width, height, channels);
    if (!image) {
        fprintf(stderr, "Failed to create test image\n");
        return NULL;
    }

    /* Fill with simple gradient pattern */
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            /* Red gradient horizontally */
            image->data[(y * width + x) * channels + 0] = (uint8_t)(x * 255 / width);
            /* Green gradient vertically */
            image->data[(y * width + x) * channels + 1] = (uint8_t)(y * 255 / height);
            /* Blue constant */
            image->data[(y * width + x) * channels + 2] = 128;
        }
    }

    return image;
}

/* Utility function to create test features */
static float *create_test_features(int size)
{
    float *features = (float *)malloc(size * sizeof(float));
    if (!features) {
        fprintf(stderr, "Failed to allocate test features\n");
        return NULL;
    }

    /* Fill with simple pattern */
    for (int i = 0; i < size; i++) {
        features[i] = (float)i / size;
    }

    return features;
}

/* Main test function */
int test_multimodal_main()
{
    printf("Running multimodal tests...\n");

    /* Initialize random seed */
    srand((unsigned int)time(NULL));

    /* Run tests */
    test_multimodal_model_creation();
    test_multimodal_feature_fusion();
    test_cross_attention();
    test_multimodal_end_to_end();
    test_multimodal_quantization();
    test_multimodal_simd_acceleration();
    test_feature_fusion_methods();

    printf("All multimodal tests passed!\n");
    return 0;
}

/* Test multimodal model creation */
static void test_multimodal_model_creation()
{
    printf("Testing multimodal model creation...\n");

    /* Create models */
    TinyAIModel *visionModel =
        tinyaiLoadModel(TEST_VISION_MODEL_PATH, TEST_VISION_WEIGHTS_PATH, NULL);
    assert(visionModel != NULL);

    TinyAIModel *textModel =
        tinyaiLoadModel(TEST_TEXT_MODEL_PATH, TEST_TEXT_WEIGHTS_PATH, TEST_TOKENIZER_PATH);
    assert(textModel != NULL);

    /* Create multimodal model */
    TinyAIMultimodalModel *multimodalModel = tinyaiCreateMultimodalModel(visionModel, textModel);
    assert(multimodalModel != NULL);

    /* Check properties */
    assert(multimodalModel->visionModel == visionModel);
    assert(multimodalModel->textModel == textModel);

    /* Free multimodal model */
    tinyaiFreeMultimodalModel(multimodalModel);

    printf("Multimodal model creation test passed!\n");
}

/* Test feature fusion methods */
static void test_multimodal_feature_fusion()
{
    printf("Testing multimodal feature fusion...\n");

    /* Create test features */
    const int visionFeatureSize = TEST_FEATURE_SIZE;
    const int textFeatureSize   = TEST_FEATURE_SIZE / 2;

    float *visionFeatures = create_test_features(visionFeatureSize);
    assert(visionFeatures != NULL);

    float *textFeatures = create_test_features(textFeatureSize);
    assert(textFeatures != NULL);

    /* Test concatenation fusion */
    float *concatFeatures    = NULL;
    int    concatFeatureSize = 0;

    bool concatResult =
        tinyaiFuseFeaturesConcat(visionFeatures, visionFeatureSize, textFeatures, textFeatureSize,
                                 &concatFeatures, &concatFeatureSize);

    assert(concatResult == true);
    assert(concatFeatures != NULL);
    assert(concatFeatureSize == visionFeatureSize + textFeatureSize);

    /* Verify concatenation */
    for (int i = 0; i < visionFeatureSize; i++) {
        assert(concatFeatures[i] == visionFeatures[i]);
    }

    for (int i = 0; i < textFeatureSize; i++) {
        assert(concatFeatures[visionFeatureSize + i] == textFeatures[i]);
    }

    /* Test addition fusion (requires same feature size) */
    float *tempTextFeatures = create_test_features(visionFeatureSize);
    assert(tempTextFeatures != NULL);

    float *addFeatures    = NULL;
    int    addFeatureSize = 0;

    bool addResult = tinyaiFuseFeaturesAdd(visionFeatures, visionFeatureSize, tempTextFeatures,
                                           visionFeatureSize, &addFeatures, &addFeatureSize);

    assert(addResult == true);
    assert(addFeatures != NULL);
    assert(addFeatureSize == visionFeatureSize);

    /* Verify addition */
    for (int i = 0; i < visionFeatureSize; i++) {
        assert(addFeatures[i] == visionFeatures[i] + tempTextFeatures[i]);
    }

    /* Test multiplication fusion (requires same feature size) */
    float *mulFeatures    = NULL;
    int    mulFeatureSize = 0;

    bool mulResult = tinyaiFuseFeaturesMul(visionFeatures, visionFeatureSize, tempTextFeatures,
                                           visionFeatureSize, &mulFeatures, &mulFeatureSize);

    assert(mulResult == true);
    assert(mulFeatures != NULL);
    assert(mulFeatureSize == visionFeatureSize);

    /* Verify multiplication */
    for (int i = 0; i < visionFeatureSize; i++) {
        assert(mulFeatures[i] == visionFeatures[i] * tempTextFeatures[i]);
    }

    /* Clean up */
    free(visionFeatures);
    free(textFeatures);
    free(tempTextFeatures);
    free(concatFeatures);
    free(addFeatures);
    free(mulFeatures);

    printf("Multimodal feature fusion test passed!\n");
}

/* Test cross-attention between modalities */
static void test_cross_attention()
{
    printf("Testing cross-attention between modalities...\n");

    /* Create test features */
    const int visionFeatureSize = TEST_FEATURE_SIZE;
    const int textFeatureSize   = TEST_FEATURE_SIZE;
    const int seqLength         = 10;

    /* Create vision features (treated as a single feature vector) */
    float *visionFeatures = create_test_features(visionFeatureSize);
    assert(visionFeatures != NULL);

    /* Create text features (sequence of feature vectors) */
    float *textFeatures = (float *)malloc(seqLength * textFeatureSize * sizeof(float));
    assert(textFeatures != NULL);

    for (int i = 0; i < seqLength; i++) {
        for (int j = 0; j < textFeatureSize; j++) {
            textFeatures[i * textFeatureSize + j] = (float)(i + j) / (seqLength + textFeatureSize);
        }
    }

    /* Test cross-attention fusion */
    float *attentionFeatures    = NULL;
    int    attentionFeatureSize = 0;

    bool attentionResult = tinyaiFuseFeaturesAttention(visionFeatures, visionFeatureSize,
                                                       textFeatures, textFeatureSize, seqLength,
                                                       &attentionFeatures, &attentionFeatureSize);

    assert(attentionResult == true);
    assert(attentionFeatures != NULL);
    assert(attentionFeatureSize == textFeatureSize);

    /* Clean up */
    free(visionFeatures);
    free(textFeatures);
    free(attentionFeatures);

    printf("Cross-attention test passed!\n");
}

/* Test end-to-end multimodal model operation */
static void test_multimodal_end_to_end()
{
    printf("Testing end-to-end multimodal operation...\n");

    /* Create test image */
    TinyAIImage *testImage = create_test_image();
    assert(testImage != NULL);

    /* Create multimodal config */
    TinyAIMultimodalConfig config;
    memset(&config, 0, sizeof(TinyAIMultimodalConfig));

    config.visionModelPath   = TEST_VISION_MODEL_PATH;
    config.visionWeightsPath = TEST_VISION_WEIGHTS_PATH;
    config.textModelPath     = TEST_TEXT_MODEL_PATH;
    config.textWeightsPath   = TEST_TEXT_WEIGHTS_PATH;
    config.tokenizerPath     = TEST_TOKENIZER_PATH;
    config.fusionMethod      = TINYAI_FUSION_ATTENTION;
    config.maxOutputTokens   = TEST_OUTPUT_LENGTH;
    config.useQuantization   = false;
    config.useSIMD           = true;

    /* Create multimodal model */
    TinyAIMultimodalModel *model = tinyaiCreateMultimodalModelFromConfig(&config);
    assert(model != NULL);

    /* Process image */
    float *imageFeatures = NULL;
    int    featureSize   = 0;

    bool encodeResult = tinyaiEncodeImage(model, testImage, &imageFeatures, &featureSize);
    assert(encodeResult == true);
    assert(imageFeatures != NULL);
    assert(featureSize > 0);

    /* Generate text from image features */
    char *outputText =
        tinyaiGenerateTextFromFeatures(model, imageFeatures, featureSize, NULL, NULL);
    assert(outputText != NULL);

    /* Verify output is not empty */
    assert(strlen(outputText) > 0);

    /* Clean up */
    free(outputText);
    free(imageFeatures);
    tinyaiFreeMultimodalModel(model);
    tinyaiImageFree(testImage);

    printf("End-to-end multimodal test passed!\n");
}

/* Test quantized multimodal model */
static void test_multimodal_quantization()
{
    printf("Testing multimodal model quantization...\n");

    /* Create test image */
    TinyAIImage *testImage = create_test_image();
    assert(testImage != NULL);

    /* Create multimodal config with quantization enabled */
    TinyAIMultimodalConfig config;
    memset(&config, 0, sizeof(TinyAIMultimodalConfig));

    config.visionModelPath   = TEST_VISION_MODEL_PATH;
    config.visionWeightsPath = TEST_VISION_WEIGHTS_PATH;
    config.textModelPath     = TEST_TEXT_MODEL_PATH;
    config.textWeightsPath   = TEST_TEXT_WEIGHTS_PATH;
    config.tokenizerPath     = TEST_TOKENIZER_PATH;
    config.fusionMethod      = TINYAI_FUSION_ATTENTION;
    config.maxOutputTokens   = TEST_OUTPUT_LENGTH;
    config.useQuantization   = true; /* Enable quantization */
    config.useSIMD           = true;

    /* Create multimodal model */
    TinyAIMultimodalModel *model = tinyaiCreateMultimodalModelFromConfig(&config);
    assert(model != NULL);

    /* Verify quantization status */
    assert(tinyaiIsModelQuantized(model->visionModel) == true);
    assert(tinyaiIsModelQuantized(model->textModel) == true);

    /* Measure memory usage */
    size_t memoryUsage = tinyaiGetMultimodalModelSizeBytes(model);
    printf("Quantized model memory usage: %zu bytes\n", memoryUsage);

    /* Process image */
    float *imageFeatures = NULL;
    int    featureSize   = 0;

    bool encodeResult = tinyaiEncodeImage(model, testImage, &imageFeatures, &featureSize);
    assert(encodeResult == true);

    /* Generate text from image features */
    char *outputText =
        tinyaiGenerateTextFromFeatures(model, imageFeatures, featureSize, NULL, NULL);
    assert(outputText != NULL);

    /* Verify output is not empty */
    assert(strlen(outputText) > 0);

    /* Clean up */
    free(outputText);
    free(imageFeatures);
    tinyaiFreeMultimodalModel(model);
    tinyaiImageFree(testImage);

    printf("Multimodal quantization test passed!\n");
}

/* Test SIMD acceleration for multimodal operations */
static void test_multimodal_simd_acceleration()
{
    printf("Testing SIMD acceleration for multimodal operations...\n");

    /* Create test image */
    TinyAIImage *testImage = create_test_image();
    assert(testImage != NULL);

    /* Create multimodal config */
    TinyAIMultimodalConfig config;
    memset(&config, 0, sizeof(TinyAIMultimodalConfig));

    config.visionModelPath   = TEST_VISION_MODEL_PATH;
    config.visionWeightsPath = TEST_VISION_WEIGHTS_PATH;
    config.textModelPath     = TEST_TEXT_MODEL_PATH;
    config.textWeightsPath   = TEST_TEXT_WEIGHTS_PATH;
    config.tokenizerPath     = TEST_TOKENIZER_PATH;
    config.fusionMethod      = TINYAI_FUSION_ATTENTION;
    config.maxOutputTokens   = TEST_OUTPUT_LENGTH;
    config.useQuantization   = true;
    config.useSIMD           = true; /* Enable SIMD */

    /* Create multimodal model with SIMD */
    TinyAIMultimodalModel *modelWithSIMD = tinyaiCreateMultimodalModelFromConfig(&config);
    assert(modelWithSIMD != NULL);

    /* Update config to disable SIMD */
    config.useSIMD = false;

    /* Create multimodal model without SIMD */
    TinyAIMultimodalModel *modelWithoutSIMD = tinyaiCreateMultimodalModelFromConfig(&config);
    assert(modelWithoutSIMD != NULL);

    /* Set up benchmarking */
    TinyAIBenchmarkConfig benchConfig;
    benchConfig.numRuns    = 5;
    benchConfig.warmupRuns = 2;

    /* Benchmark image encoding with SIMD */
    double simdTime = tinyaiBenchmarkOperation(&benchConfig, ^{
      float *features    = NULL;
      int    featureSize = 0;
      bool   result      = tinyaiEncodeImage(modelWithSIMD, testImage, &features, &featureSize);
      assert(result == true);
      free(features);
    });

    /* Benchmark image encoding without SIMD */
    double nonSimdTime = tinyaiBenchmarkOperation(&benchConfig, ^{
      float *features    = NULL;
      int    featureSize = 0;
      bool   result      = tinyaiEncodeImage(modelWithoutSIMD, testImage, &features, &featureSize);
      assert(result == true);
      free(features);
    });

    /* Print results */
    printf("SIMD enabled time: %.4f ms\n", simdTime);
    printf("SIMD disabled time: %.4f ms\n", nonSimdTime);
    printf("Speedup: %.2fx\n", nonSimdTime / simdTime);

    /* Verify SIMD provides speedup (may not always be true on all hardware) */
    /* Commented out as it depends on hardware support
    assert(simdTime < nonSimdTime);
    */

    /* Clean up */
    tinyaiFreeMultimodalModel(modelWithSIMD);
    tinyaiFreeMultimodalModel(modelWithoutSIMD);
    tinyaiImageFree(testImage);

    printf("SIMD acceleration test completed!\n");
}

/* Test various feature fusion methods */
static void test_feature_fusion_methods()
{
    printf("Testing different fusion methods...\n");

    /* Create test image */
    TinyAIImage *testImage = create_test_image();
    assert(testImage != NULL);

    /* Define fusion methods to test */
    TinyAIFusionMethod fusionMethods[] = {TINYAI_FUSION_CONCATENATION, TINYAI_FUSION_ADDITION,
                                          TINYAI_FUSION_MULTIPLICATION, TINYAI_FUSION_ATTENTION};

    const char *fusionMethodNames[] = {"Concatenation", "Addition", "Multiplication", "Attention"};

    const int numMethods = sizeof(fusionMethods) / sizeof(fusionMethods[0]);

    /* Base config */
    TinyAIMultimodalConfig config;
    memset(&config, 0, sizeof(TinyAIMultimodalConfig));

    config.visionModelPath   = TEST_VISION_MODEL_PATH;
    config.visionWeightsPath = TEST_VISION_WEIGHTS_PATH;
    config.textModelPath     = TEST_TEXT_MODEL_PATH;
    config.textWeightsPath   = TEST_TEXT_WEIGHTS_PATH;
    config.tokenizerPath     = TEST_TOKENIZER_PATH;
    config.maxOutputTokens   = TEST_OUTPUT_LENGTH;
    config.useQuantization   = true;
    config.useSIMD           = true;

    /* Test each fusion method */
    for (int i = 0; i < numMethods; i++) {
        printf("Testing %s fusion...\n", fusionMethodNames[i]);

        /* Set fusion method */
        config.fusionMethod = fusionMethods[i];

        /* Create multimodal model */
        TinyAIMultimodalModel *model = tinyaiCreateMultimodalModelFromConfig(&config);
        assert(model != NULL);

        /* Process image */
        float *imageFeatures = NULL;
        int    featureSize   = 0;

        bool encodeResult = tinyaiEncodeImage(model, testImage, &imageFeatures, &featureSize);
        assert(encodeResult == true);

        /* Generate text */
        char *outputText =
            tinyaiGenerateTextFromFeatures(model, imageFeatures, featureSize, NULL, NULL);
        assert(outputText != NULL);

        /* Print sample output */
        printf("%s fusion output: %.40s...\n", fusionMethodNames[i], outputText);

        /* Clean up */
        free(outputText);
        free(imageFeatures);
        tinyaiFreeMultimodalModel(model);
    }

    /* Clean up */
    tinyaiImageFree(testImage);

    printf("Feature fusion methods test passed!\n");
}
