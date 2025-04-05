/**
 * TinyAI Tokenizer Implementation
 * 
 * This file implements a minimal, memory-efficient tokenizer for TinyAI.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "tokenizer.h"
#include "../../core/memory.h"
#include "../../utils/quantize.h"

/* ----------------- Internal Definitions ----------------- */

/**
 * Trie node structure for efficient token lookup
 */
typedef struct TrieNode {
    int tokenId;       /* -1 if not a token */
    struct TrieNode *children[256];  /* One for each possible byte */
} TrieNode;

/**
 * BPE merge structure for encoding
 */
typedef struct {
    char *pair;       /* Pair of tokens to merge */
    int tokenId;      /* Resulting token ID */
    uint32_t priority; /* Merge priority (lower = higher priority) */
} BPEMerge;

/* ----------------- Static Variables ----------------- */

/* Special token strings */
static const char *SPECIAL_TOKENS[] = {
    "<unk>",  /* TINYAI_TOKEN_UNKNOWN */
    "<bos>",  /* TINYAI_TOKEN_BOS */
    "<eos>",  /* TINYAI_TOKEN_EOS */
    "<pad>",  /* TINYAI_TOKEN_PAD */
};

/* ----------------- Helper Functions ----------------- */

/**
 * Create a trie node
 */
static TrieNode* createTrieNode() {
    TrieNode *node = (TrieNode*)TINYAI_MALLOC(sizeof(TrieNode));
    if (!node) {
        return NULL;
    }
    
    node->tokenId = -1;
    memset(node->children, 0, sizeof(node->children));
    
    return node;
}

/**
 * Free a trie node and all its children
 */
static void freeTrieNode(TrieNode *node) {
    if (!node) {
        return;
    }
    
    for (int i = 0; i < 256; i++) {
        if (node->children[i]) {
            freeTrieNode(node->children[i]);
        }
    }
    
    TINYAI_FREE(node);
}

/**
 * Insert a token into the trie
 */
static void insertTokenToTrie(TrieNode *root, const char *token, int tokenId) {
    if (!root || !token) {
        return;
    }
    
    TrieNode *current = root;
    
    for (const unsigned char *p = (const unsigned char *)token; *p; p++) {
        if (!current->children[*p]) {
            current->children[*p] = createTrieNode();
        }
        current = current->children[*p];
    }
    
    current->tokenId = tokenId;
}

/**
 * Find a token in the trie
 */
static int findTokenInTrie(TrieNode *root, const char *token) {
    if (!root || !token) {
        return -1;
    }
    
    TrieNode *current = root;
    
    for (const unsigned char *p = (const unsigned char *)token; *p; p++) {
        if (!current->children[*p]) {
            return -1;
        }
        current = current->children[*p];
    }
    
    return current->tokenId;
}

/**
 * Simple string hash function for lookup
 */
static uint32_t hashString(const char *str) {
    uint32_t hash = 5381;
    int c;
    
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;  /* hash * 33 + c */
    }
    
    return hash;
}

/**
 * Compare function for BPE merges (used for qsort)
 */
static int compareBPEMerges(const void *a, const void *b) {
    const BPEMerge *mergeA = (const BPEMerge *)a;
    const BPEMerge *mergeB = (const BPEMerge *)b;
    
    return mergeA->priority - mergeB->priority;
}

/* ----------------- Tokenizer Implementation ----------------- */

/**
 * Create a new tokenizer
 */
TinyAITokenizer* tinyaiCreateTokenizer() {
    TinyAITokenizer *tokenizer = (TinyAITokenizer *)TINYAI_MALLOC(sizeof(TinyAITokenizer));
    if (!tokenizer) {
        return NULL;
    }
    
    /* Initialize the tokenizer */
    memset(tokenizer->tokens, 0, sizeof(tokenizer->tokens));
    tokenizer->tokenCount = 0;
    tokenizer->frequencies = (uint32_t *)TINYAI_MALLOC(TINYAI_MAX_VOCAB_SIZE * sizeof(uint32_t));
    if (!tokenizer->frequencies) {
        TINYAI_FREE(tokenizer);
        return NULL;
    }
    
    memset(tokenizer->frequencies, 0, TINYAI_MAX_VOCAB_SIZE * sizeof(uint32_t));
    tokenizer->caseSensitive = 0;
    
    /* Add special tokens */
    for (int i = 0; i < 4; i++) {
        tinyaiAddToken(tokenizer, SPECIAL_TOKENS[i], 0);
    }
    
    return tokenizer;
}

/**
 * Destroy a tokenizer
 */
void tinyaiDestroyTokenizer(TinyAITokenizer *tokenizer) {
    if (!tokenizer) {
        return;
    }
    
    /* Free token strings */
    for (uint32_t i = 0; i < tokenizer->tokenCount; i++) {
        if (tokenizer->tokens[i]) {
            TINYAI_FREE(tokenizer->tokens[i]);
        }
    }
    
    /* Free frequencies */
    if (tokenizer->frequencies) {
        TINYAI_FREE(tokenizer->frequencies);
    }
    
    /* Free the tokenizer */
    TINYAI_FREE(tokenizer);
}

/**
 * Load a vocabulary from a file
 */
int tinyaiLoadVocabulary(TinyAITokenizer *tokenizer, const char *path) {
    if (!tokenizer || !path) {
        return -1;
    }
    
    FILE *file = fopen(path, "r");
    if (!file) {
        return -1;
    }
    
    /* Clear existing vocabulary (except special tokens) */
    for (uint32_t i = 4; i < tokenizer->tokenCount; i++) {
        if (tokenizer->tokens[i]) {
            TINYAI_FREE(tokenizer->tokens[i]);
            tokenizer->tokens[i] = NULL;
        }
    }
    tokenizer->tokenCount = 4;  /* Keep special tokens */
    
    /* Parse the vocabulary file */
    char line[TINYAI_MAX_TOKEN_LENGTH * 2];
    while (fgets(line, sizeof(line), file)) {
        /* Remove trailing newline */
        size_t len = strlen(line);
        if (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }
        if (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }
        
        /* Skip empty lines and comments */
        if (len == 0 || line[0] == '#') {
            continue;
        }
        
        /* Parse token and frequency (if provided) */
        char token[TINYAI_MAX_TOKEN_LENGTH];
        uint32_t frequency = 0;
        
        if (sscanf(line, "%s %u", token, &frequency) >= 1) {
            tinyaiAddToken(tokenizer, token, frequency);
        }
    }
    
    fclose(file);
    return 0;
}

/**
 * Add a token to the vocabulary
 */
int tinyaiAddToken(TinyAITokenizer *tokenizer, const char *token, uint32_t frequency) {
    if (!tokenizer || !token || tokenizer->tokenCount >= TINYAI_MAX_VOCAB_SIZE) {
        return -1;
    }
    
    /* Check if token already exists */
    for (uint32_t i = 0; i < tokenizer->tokenCount; i++) {
        if (tokenizer->tokens[i] && strcmp(tokenizer->tokens[i], token) == 0) {
            /* Update frequency if higher */
            if (frequency > tokenizer->frequencies[i]) {
                tokenizer->frequencies[i] = frequency;
            }
            return i;
        }
    }
    
    /* Add new token */
    int id = tokenizer->tokenCount++;
    tokenizer->tokens[id] = strdup(token);
    if (!tokenizer->tokens[id]) {
        tokenizer->tokenCount--;
        return -1;
    }
    
    tokenizer->frequencies[id] = frequency;
    
    return id;
}

/**
 * Get a token ID by string
 */
int tinyaiGetTokenId(const TinyAITokenizer *tokenizer, const char *token) {
    if (!tokenizer || !token) {
        return TINYAI_TOKEN_UNKNOWN;
    }
    
    /* Check if token exists */
    for (uint32_t i = 0; i < tokenizer->tokenCount; i++) {
        if (tokenizer->tokens[i]) {
            if (tokenizer->caseSensitive) {
                if (strcmp(tokenizer->tokens[i], token) == 0) {
                    return i;
                }
            } else {
#ifdef _WIN32
                if (_stricmp(tokenizer->tokens[i], token) == 0) { // Use _stricmp on Windows
#else
                if (strcasecmp(tokenizer->tokens[i], token) == 0) { // Use strcasecmp elsewhere
#endif
                    return i;
                }
            }
        }
    }
    
    return TINYAI_TOKEN_UNKNOWN;
}

/**
 * Get a token string by ID
 */
const char* tinyaiGetTokenString(const TinyAITokenizer *tokenizer, int id) {
    if (!tokenizer || id < 0 || id >= (int)tokenizer->tokenCount) {
        return NULL;
    }
    
    return tokenizer->tokens[id];
}

/**
 * Tokenize a single word into subwords using BPE
 */
static int tokenizeWord(const TinyAITokenizer *tokenizer, const char *word, 
                      int *tokens, int maxTokens, int *numTokens) {
    if (!tokenizer || !word || !tokens || !numTokens) {
        return -1;
    }
    
    *numTokens = 0;
    
    /* Check if the word is already a token */
    int id = tinyaiGetTokenId(tokenizer, word);
    if (id != TINYAI_TOKEN_UNKNOWN) {
        if (*numTokens < maxTokens) {
            tokens[(*numTokens)++] = id;
        }
        return 0;
    }
    
    /* Split into characters */
    size_t wordLen = strlen(word);
    char *chars[TINYAI_MAX_TOKEN_LENGTH];
    int numChars = 0;
    
    for (size_t i = 0; i < wordLen && numChars < TINYAI_MAX_TOKEN_LENGTH - 1; i++) {
        chars[numChars] = (char*)TINYAI_MALLOC(2);
        if (!chars[numChars]) {
            /* Clean up */
            for (int j = 0; j < numChars; j++) {
                TINYAI_FREE(chars[j]);
            }
            return -1;
        }
        
        chars[numChars][0] = word[i];
        chars[numChars][1] = '\0';
        numChars++;
    }
    
    /* Merge pairs according to BPE rules */
    /* For simplicity, we use a naive approach here */
    int changed;
    do {
        changed = 0;
        
        for (int i = 0; i < numChars - 1; i++) {
            /* Try to merge the pair */
            char merged[TINYAI_MAX_TOKEN_LENGTH];
            snprintf(merged, sizeof(merged), "%s%s", chars[i], chars[i+1]);
            
            id = tinyaiGetTokenId(tokenizer, merged);
            if (id != TINYAI_TOKEN_UNKNOWN) {
                /* Merge the pair */
                TINYAI_FREE(chars[i]);
                TINYAI_FREE(chars[i+1]);
                
                chars[i] = strdup(merged);
                
                /* Shift the remaining characters */
                for (int j = i + 1; j < numChars - 1; j++) {
                    chars[j] = chars[j+1];
                }
                
                numChars--;
                changed = 1;
                break;
            }
        }
    } while (changed && numChars > 1);
    
    /* Add the resulting tokens */
    for (int i = 0; i < numChars && *numTokens < maxTokens; i++) {
        id = tinyaiGetTokenId(tokenizer, chars[i]);
        if (id == TINYAI_TOKEN_UNKNOWN) {
            tokens[(*numTokens)++] = TINYAI_TOKEN_UNKNOWN;
        } else {
            tokens[(*numTokens)++] = id;
        }
    }
    
    /* Clean up */
    for (int i = 0; i < numChars; i++) {
        TINYAI_FREE(chars[i]);
    }
    
    return 0;
}

/**
 * Encode a text string into token IDs
 */
int tinyaiEncodeText(const TinyAITokenizer *tokenizer, const char *text, 
                   int *tokens, int maxTokens) {
    if (!tokenizer || !text || !tokens || maxTokens <= 0) {
        return 0;
    }
    
    int numTokens = 0;
    
    /* Tokenize the text */
    const char *p = text;
    char word[TINYAI_MAX_TOKEN_LENGTH];
    int wordLen = 0;
    
    while (*p && numTokens < maxTokens) {
        if (isalnum((unsigned char)*p) || *p == '\'' || *p == '-') {
            /* Part of a word */
            if (wordLen < TINYAI_MAX_TOKEN_LENGTH - 1) {
                word[wordLen++] = *p;
            }
        } else {
            /* End of a word */
            if (wordLen > 0) {
                word[wordLen] = '\0';
                
                int subtokens[TINYAI_MAX_TOKEN_LENGTH];
                int numSubtokens = 0;
                
                tokenizeWord(tokenizer, word, subtokens, TINYAI_MAX_TOKEN_LENGTH, &numSubtokens);
                
                for (int i = 0; i < numSubtokens && numTokens < maxTokens; i++) {
                    tokens[numTokens++] = subtokens[i];
                }
                
                wordLen = 0;
            }
            
            /* Handle the separator */
            if (!isspace((unsigned char)*p)) {
                /* Treat as a separate token */
                char separator[2] = {*p, '\0'};
                int id = tinyaiGetTokenId(tokenizer, separator);
                
                if (id != TINYAI_TOKEN_UNKNOWN) {
                    tokens[numTokens++] = id;
                } else {
                    tokens[numTokens++] = TINYAI_TOKEN_UNKNOWN;
                }
            }
        }
        
        p++;
    }
    
    /* Handle the last word */
    if (wordLen > 0 && numTokens < maxTokens) {
        word[wordLen] = '\0';
        
        int subtokens[TINYAI_MAX_TOKEN_LENGTH];
        int numSubtokens = 0;
        
        tokenizeWord(tokenizer, word, subtokens, TINYAI_MAX_TOKEN_LENGTH, &numSubtokens);
        
        for (int i = 0; i < numSubtokens && numTokens < maxTokens; i++) {
            tokens[numTokens++] = subtokens[i];
        }
    }
    
    return numTokens;
}

/**
 * Decode token IDs into a text string
 */
int tinyaiDecodeTokens(const TinyAITokenizer *tokenizer, const int *tokens, 
                     int tokenCount, char *text, int maxLength) {
    if (!tokenizer || !tokens || !text || maxLength <= 0) {
        return 0;
    }
    
    int textLen = 0;
    text[0] = '\0';
    
    for (int i = 0; i < tokenCount; i++) {
        const char *token = tinyaiGetTokenString(tokenizer, tokens[i]);
        if (!token) {
            /* Unknown token */
            token = SPECIAL_TOKENS[TINYAI_TOKEN_UNKNOWN];
        }
        
        /* Skip special tokens in the output */
        if (tokens[i] <= TINYAI_TOKEN_PAD) {
            continue;
        }
        
        size_t tokenLen = strlen(token);
        
        /* Check if we need a separator */
        if (textLen > 0 && tokenLen > 0) {
            /* Simple heuristic for natural spacing */
            if (token[0] == '\'' || token[0] == '.' || token[0] == ',' || 
                token[0] == '!' || token[0] == '?' || token[0] == ':' || 
                token[0] == ';') {
                /* No space before punctuation */
            } else {
                /* Add a space */
                if (textLen < maxLength - 1) {
                    text[textLen++] = ' ';
                    text[textLen] = '\0';
                }
            }
        }
        
        /* Add the token */
        if (textLen + tokenLen < maxLength) {
            strcat(text, token);
            textLen += tokenLen;
        } else {
            break;
        }
    }
    
    return textLen;
}

/**
 * Create a minimal BPE tokenizer vocabulary from text corpus
 */
int tinyaiCreateMinimalVocabulary(TinyAITokenizer *tokenizer, 
                                const char *corpus, int maxVocabSize) {
    if (!tokenizer || !corpus || maxVocabSize <= 0) {
        return -1;
    }
    
    /* Reset the tokenizer to just special tokens */
    for (uint32_t i = 4; i < tokenizer->tokenCount; i++) {
        if (tokenizer->tokens[i]) {
            TINYAI_FREE(tokenizer->tokens[i]);
            tokenizer->tokens[i] = NULL;
        }
    }
    tokenizer->tokenCount = 4;  /* Keep special tokens */
    
    /* Step 1: Collect word frequencies */
    /* For simplicity, we'll use a naive approach */
    struct {
        char *word;
        uint32_t count;
    } words[10000]; /* Limit to 10000 words for this simplified implementation */
    int wordCount = 0;
    
    const char *p = corpus;
    char word[TINYAI_MAX_TOKEN_LENGTH];
    int wordLen = 0;
    
    while (*p) {
        if (isalnum((unsigned char)*p) || *p == '\'' || *p == '-') {
            /* Part of a word */
            if (wordLen < TINYAI_MAX_TOKEN_LENGTH - 1) {
                word[wordLen++] = *p;
            }
        } else {
            /* End of a word */
            if (wordLen > 0) {
                word[wordLen] = '\0';
                
                /* Check if word already exists */
                int found = 0;
                for (int i = 0; i < wordCount; i++) {
                    if (strcmp(words[i].word, word) == 0) {
                        words[i].count++;
                        found = 1;
                        break;
                    }
                }
                
                /* Add new word */
                if (!found && wordCount < 10000) {
                    words[wordCount].word = strdup(word);
                    words[wordCount].count = 1;
                    wordCount++;
                }
                
                wordLen = 0;
            }
            
            /* Add punctuation as single-character tokens */
            if (!isspace((unsigned char)*p)) {
                char separator[2] = {*p, '\0'};
                
                /* Check if token already exists */
                int found = 0;
                for (int i = 0; i < wordCount; i++) {
                    if (strcmp(words[i].word, separator) == 0) {
                        words[i].count++;
                        found = 1;
                        break;
                    }
                }
                
                /* Add new token */
                if (!found && wordCount < 10000) {
                    words[wordCount].word = strdup(separator);
                    words[wordCount].count = 1;
                    wordCount++;
                }
            }
        }
        
        p++;
    }
    
    /* Handle the last word */
    if (wordLen > 0) {
        word[wordLen] = '\0';
        
        /* Check if word already exists */
        int found = 0;
        for (int i = 0; i < wordCount; i++) {
            if (strcmp(words[i].word, word) == 0) {
                words[i].count++;
                found = 1;
                break;
            }
        }
        
        /* Add new word */
        if (!found && wordCount < 10000) {
            words[wordCount].word = strdup(word);
            words[wordCount].count = 1;
            wordCount++;
        }
    }
    
    /* Step 2: Add single characters to the vocabulary */
    for (int i = 32; i < 127; i++) {
        if (isalnum(i) || ispunct(i)) {
            char c[2] = {(char)i, '\0'};
            tinyaiAddToken(tokenizer, c, 0);
        }
    }
    
    /* Step 3: Add most frequent words to the vocabulary */
    /* Sort words by frequency */
    for (int i = 0; i < wordCount - 1; i++) {
        for (int j = i + 1; j < wordCount; j++) {
            if (words[j].count > words[i].count) {
                /* Swap */
                char *tempWord = words[i].word;
                uint32_t tempCount = words[i].count;
                words[i].word = words[j].word;
                words[i].count = words[j].count;
                words[j].word = tempWord;
                words[j].count = tempCount;
            }
        }
    }
    
    /* Add words until we reach maxVocabSize */
    for (int i = 0; i < wordCount && tokenizer->tokenCount < maxVocabSize; i++) {
        tinyaiAddToken(tokenizer, words[i].word, words[i].count);
    }
    
    /* Clean up */
    for (int i = 0; i < wordCount; i++) {
        free(words[i].word);
    }
    
    return 0;
}

/**
 * Save a vocabulary to a file
 */
int tinyaiSaveVocabulary(const TinyAITokenizer *tokenizer, const char *path) {
    if (!tokenizer || !path) {
        return -1;
    }
    
    FILE *file = fopen(path, "w");
    if (!file) {
        return -1;
    }
    
    /* Write the vocabulary */
    fprintf(file, "# TinyAI Vocabulary File\n");
    fprintf(file, "# Format: token frequency\n\n");
    
    for (uint32_t i = 0; i < tokenizer->tokenCount; i++) {
        if (tokenizer->tokens[i]) {
            fprintf(file, "%s %u\n", tokenizer->tokens[i], tokenizer->frequencies[i]);
        }
    }
    
    fclose(file);
    return 0;
}
