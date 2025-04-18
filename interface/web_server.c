#include <stdio.h>
#include <stdlib.h> // For getenv, atoi
#include <string.h>

#include "../core/config.h"              // For accessing config values like model paths
#include "../models/text/generate.h"     // For text generation functions
#include "../models/text/tokenizer.h"    // For tokenizer
#include "../vendor/mongoose/mongoose.h" // Absolute path from project root
#include "web_server.h"

// --- Global State (Consider managing this better in a real app) ---
static TinyAIModel     *g_model     = NULL;
static TinyAITokenizer *g_tokenizer = NULL;
static picolInterp     *g_interp    = NULL; // Store interpreter if needed for commands
static volatile int     s_exit_flag = 0;    // Flag to signal server shutdown
// ---

// --- Forward Declarations ---
static void handle_api_generate(struct mg_connection *c, struct mg_http_message *hm);
// ---

// Mongoose event handler function
static void fn(struct mg_connection *c, int ev, void *ev_data, void *fn_data)
{
    if (ev == MG_EV_HTTP_MSG) {
        struct mg_http_message *hm            = (struct mg_http_message *)ev_data;
        const char             *document_root = (const char *)fn_data;

        // Check if it's an API call
        if (mg_http_match_uri(hm, "/api/generate") && mg_match(hm->method, mg_str("POST"), NULL)) {
            handle_api_generate(c, hm);
        }
        else {
            // Serve static files
            struct mg_http_serve_opts opts = {
                .root_dir = document_root,
                // .extra_headers = "Cache-Control: max-age=3600\r\n" // Optional: Add caching
            };
            mg_http_serve_dir(c, hm, &opts);
        }
    }
    else if (ev == MG_EV_CLOSE) {
        // Connection closed
    }
}

// API Handler for /api/generate
static void handle_api_generate(struct mg_connection *c, struct mg_http_message *hm)
{
    char  prompt_buf[512] = {0}; // Buffer for the prompt
    char *result_text     = NULL;
    int   result_len      = 0;

    // 1. Parse prompt from JSON body: {"prompt": "..."}
    char *prompt_val_ptr = mg_json_get_str(hm->body, "$.prompt");
    if (prompt_val_ptr != NULL) {
        // Copy the value, ensuring not to overflow prompt_buf
        strncpy(prompt_buf, prompt_val_ptr, sizeof(prompt_buf) - 1);
        prompt_buf[sizeof(prompt_buf) - 1] = '\0'; // Ensure null termination
        // NOTE: mg_json_get_str returns a pointer into the original JSON string (hm->body).
        // We don't free prompt_val_ptr itself.
    }
    else {
        // Handle error: prompt not found or parsing failed
        mg_http_reply(c, 400, "Content-Type: application/json\r\n",
                      "{\"error\":\"Invalid JSON or missing/invalid 'prompt' field\"}\n");
        return;
    }

    // Check if prompt_buf is still empty after potential copy
    if (strlen(prompt_buf) == 0) {
        mg_http_reply(c, 400, "Content-Type: application/json\r\n",
                      "{\"error\":\"Empty 'prompt' value provided\"}\n");
        return;
    }

    // 2. Check if model is loaded
    if (!g_model || !g_tokenizer) {
        mg_http_reply(c, 503, "Content-Type: application/json\r\n",
                      "{\"error\":\"Model or tokenizer not loaded\"}\n");
        return;
    }

    // 3. Prepare generation parameters (use defaults or config)
    TinyAIGenerationParams params = {0};
    params.maxTokens = tinyaiConfigGetInt("generate.max_tokens", 128); // Get from config or default
    params.samplingMethod = TINYAI_SAMPLING_TEMPERATURE;               // Example
    params.temperature =
        tinyaiConfigGetFloat("generate.temperature", 0.7f); // Get from config or default
    params.topK = tinyaiConfigGetInt("generate.top_k", 40);
    params.topP = tinyaiConfigGetFloat("generate.top_p", 0.9f);
    params.seed = tinyaiConfigGetInt("generate.seed", 0); // 0 for random

    // 4. Tokenize the prompt
    int prompt_tokens[512]; // Adjust size as needed
    params.promptLength = tinyaiTokenize(g_tokenizer, prompt_buf, prompt_tokens,
                                         sizeof(prompt_tokens) / sizeof(prompt_tokens[0]));
    if (params.promptLength <= 0) {
        mg_http_reply(c, 400, "Content-Type: application/json\r\n",
                      "{\"error\":\"Failed to tokenize prompt or prompt too long\"}\n");
        return;
    }
    params.promptTokens = prompt_tokens;

    // 5. Allocate buffer for output tokens - dynamically to avoid VLA issues
    int *output_tokens = (int *)malloc((params.maxTokens + params.promptLength) * sizeof(int));
    if (!output_tokens) {
        mg_http_reply(c, 500, "Content-Type: application/json\r\n",
                      "{\"error\":\"Failed to allocate memory for generation\"}\n");
        return;
    }

    // 6. Generate text
    printf("Generating text for prompt: \"%s\"\n", prompt_buf); // Log
    int generated_count = tinyaiGenerateText(g_model, &params, output_tokens, params.maxTokens);

    // 7. Decode the generated tokens (excluding prompt)
    if (generated_count > 0) {
        result_text = tinyaiDecode(g_tokenizer, output_tokens, generated_count);
        if (result_text) {
            result_len = strlen(result_text);
            printf("Generated text: %s\n", result_text); // Log
        }
        else {
            mg_http_reply(c, 500, "Content-Type: application/json\r\n",
                          "{\"error\":\"Failed to decode generated tokens\"}\n");
            return;
        }
    }
    else {
        // Handle generation failure or empty output
        mg_http_reply(c, 500, "Content-Type: application/json\r\n",
                      "{\"error\":\"Text generation failed or produced no output\"}\n");
        return;
    }

    // 8. Send JSON response: {"result": "..."}
    // Use mg_http_reply which handles chunking if needed
    mg_printf(
        c,
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n");
    mg_http_printf_chunk(c, "{\"result\": ");
    mg_http_write_chunk(c, "\"", 1); // Start JSON string
    // Simple JSON escaping (replace " with \" and \ with \\) - Mongoose might have helpers?
    // For simplicity, let's assume basic escaping is sufficient for now.
    // A proper JSON library would be better for complex text.
    char *escaped_result = mg_json_esc(NULL, result_text, result_len); // Mongoose helper
    if (escaped_result) {
        mg_http_write_chunk(c, escaped_result, strlen(escaped_result));
        free(escaped_result);
    }
    else {
        // Fallback if escaping fails (shouldn't happen often)
        mg_http_write_chunk(c, "Error escaping result", strlen("Error escaping result"));
    }
    mg_http_write_chunk(c, "\"", 1); // End JSON string
    mg_http_printf_chunk(c, "}");
    mg_http_printf_chunk(c, ""); // End chunked response

    // 9. Clean up allocated memory
    if (result_text) {
        free(result_text); // Assuming tinyaiDecode allocates memory
    }
    free(output_tokens);
}

// Function to start the web server
int start_web_server(picolInterp *interp, const char *port, const char *document_root)
{
    struct mg_mgr mgr;
    char          addr[64];

    // --- Load Model and Tokenizer ---
    // Get paths from config (assuming they are set via CLI or config file)
    const char *model_file = tinyaiConfigGet("model.path", NULL);
    const char *weights_file =
        tinyaiConfigGet("model.weights_path", NULL); // Assuming separate weights
    const char *tokenizer_file = tinyaiConfigGet("tokenizer.path", NULL);

    if (!model_file || !tokenizer_file) {
        fprintf(stderr, "Error: Model or Tokenizer path not configured.\n");
        fprintf(stderr, "Please specify using --model and --tokenizer options or config file.\n");
        return 1;
    }
    // If weights are separate, check for weights_file too

    printf("Loading tokenizer from %s...\n", tokenizer_file);
    g_tokenizer = tinyaiLoadTokenizer(tokenizer_file);
    if (!g_tokenizer) {
        fprintf(stderr, "Error: Failed to load tokenizer from %s\n", tokenizer_file);
        return 1;
    }

    printf("Loading model from %s...\n", model_file);
    // Assuming tinyaiLoadModel handles both structure and weights if weights_file is NULL
    g_model = tinyaiLoadModel(model_file, weights_file, tokenizer_file);
    if (!g_model) {
        fprintf(stderr, "Error: Failed to load model from %s\n", model_file);
        tinyaiDestroyTokenizer(g_tokenizer); // Clean up tokenizer
        g_tokenizer = NULL;
        return 1;
    }
    // Assign tokenizer to model if not done by tinyaiLoadModel
    if (g_model->tokenizer == NULL) {
        g_model->tokenizer = g_tokenizer;
    }
    // --- Model Loaded ---

    g_interp = interp; // Store interpreter if needed later

    // Initialize Mongoose manager
    mg_mgr_init(&mgr);
    snprintf(addr, sizeof(addr), "http://0.0.0.0:%s", port);

    printf("Starting web server on %s, serving %s\n", addr, document_root);

    // Create listening connection
    if (mg_http_listen(&mgr, addr, fn, (void *)document_root) == NULL) {
        fprintf(stderr, "Error: Cannot listen on %s. Is the port already in use?\n", addr);
        mg_mgr_free(&mgr);
        tinyaiDestroyModel(g_model);
        tinyaiDestroyTokenizer(g_tokenizer);
        return 1;
    }

    // Run the event loop
    s_exit_flag = 0;
    while (s_exit_flag == 0) {
        mg_mgr_poll(&mgr, 1000); // Poll with 1 second timeout
    }

    // Cleanup
    printf("Shutting down web server...\n");
    mg_mgr_free(&mgr);
    tinyaiDestroyModel(g_model);
    tinyaiDestroyTokenizer(g_tokenizer);
    g_model     = NULL;
    g_tokenizer = NULL;
    g_interp    = NULL;

    return 0;
}

// Optional: Function to signal server shutdown (e.g., from another thread or signal handler)
void stop_web_server() { s_exit_flag = 1; }
