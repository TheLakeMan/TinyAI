/**
 * TinyAI Tokenizer Header
 * 
 * This header defines the tokenizer API for text processing in TinyAI.
 */

#ifndef TINYAI_TOKENIZER_H
#define TINYAI_TOKENIZER_H

#include <stdint.h>

/* ----------------- Constants ----------------- */

/* Maximum vocabulary size */
#define TINYAI_MAX_VOCAB_SIZE 65536

/* Maximum token length */
#define TINYAI_MAX_TOKEN_LENGTH 256

/* Special token IDs */
#define TINYAI_TOKEN_UNKNOWN 0
#define TINYAI_TOKEN_BOS     1
#define TINYAI_TOKEN_EOS     2
#define TINYAI_TOKEN_PAD     3

/* ----------------- Types ----------------- */

/**
 * Tokenizer structure
 */
typedef struct {
    char *tokens[TINYAI_MAX_VOCAB_SIZE];    /* Token strings */
    uint32_t tokenCount;                    /* Number of tokens in vocabulary */
    uint32_t *frequencies;                  /* Token frequencies (for training) */
    int caseSensitive;                      /* Whether tokenization is case-sensitive */
} TinyAITokenizer;

/* ----------------- API Functions ----------------- */

/**
 * Create a new tokenizer
 * 
 * @return New tokenizer or NULL on error
 */
TinyAITokenizer* tinyaiCreateTokenizer();

/**
 * Destroy a tokenizer
 * 
 * @param tokenizer Tokenizer to destroy
 */
void tinyaiDestroyTokenizer(TinyAITokenizer *tokenizer);

/**
 * Load a vocabulary from a file
 * 
 * @param tokenizer Tokenizer to load into
 * @param path File path
 * @return 0 on success, non-zero on error
 */
int tinyaiLoadVocabulary(TinyAITokenizer *tokenizer, const char *path);

/**
 * Add a token to the vocabulary
 * 
 * @param tokenizer Tokenizer to add to
 * @param token Token string
 * @param frequency Token frequency
 * @return Token ID or -1 on error
 */
int tinyaiAddToken(TinyAITokenizer *tokenizer, const char *token, uint32_t frequency);

/**
 * Get a token ID by string
 * 
 * @param tokenizer Tokenizer to use
 * @param token Token string
 * @return Token ID or TINYAI_TOKEN_UNKNOWN if not found
 */
int tinyaiGetTokenId(const TinyAITokenizer *tokenizer, const char *token);

/**
 * Get a token string by ID
 * 
 * @param tokenizer Tokenizer to use
 * @param id Token ID
 * @return Token string or NULL if not found
 */
const char* tinyaiGetTokenString(const TinyAITokenizer *tokenizer, int id);

/**
 * Encode a text string into token IDs
 * 
 * @param tokenizer Tokenizer to use
 * @param text Input text
 * @param tokens Output token array
 * @param maxTokens Maximum number of tokens to output
 * @return Number of tokens encoded
 */
int tinyaiEncodeText(const TinyAITokenizer *tokenizer, const char *text, 
                   int *tokens, int maxTokens);

/**
 * Decode token IDs into a text string
 * 
 * @param tokenizer Tokenizer to use
 * @param tokens Input token array
 * @param tokenCount Number of tokens to decode
 * @param text Output text buffer
 * @param maxLength Maximum output length
 * @return Length of decoded text
 */
int tinyaiDecodeTokens(const TinyAITokenizer *tokenizer, const int *tokens, 
                     int tokenCount, char *text, int maxLength);

/**
 * Create a minimal BPE tokenizer vocabulary from text corpus
 * 
 * @param tokenizer Tokenizer to use
 * @param corpus Input text corpus
 * @param maxVocabSize Maximum vocabulary size
 * @return 0 on success, non-zero on error
 */
int tinyaiCreateMinimalVocabulary(TinyAITokenizer *tokenizer, 
                                const char *corpus, int maxVocabSize);

/**
 * Save a vocabulary to a file
 * 
 * @param tokenizer Tokenizer to save
 * @param path File path
 * @return 0 on success, non-zero on error
 */
int tinyaiSaveVocabulary(const TinyAITokenizer *tokenizer, const char *path);

#endif /* TINYAI_TOKENIZER_H */
