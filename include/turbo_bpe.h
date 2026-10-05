#ifndef TURBO_BPE_H
#define TURBO_BPE_H

#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>


#define VOCAB_SIZE 5000                   // target vocabulary size: 256 base bytes + learned merges
#define MERGES_THRESHOLD 2               // stop fitting when the best pair occurs fewer times than this
#define MAX_MERGES 500000
#define PAIR_BITS 12
#define PAIR_CAP  (1u << PAIR_BITS)      // 4096, must be a power of 2 (used as a bit mask)
#define BPE_MAGIC "BPE1"                 // first 4 bytes of a model file, bump the digit if the format changes

typedef uint16_t token_t; // 16 bits -> up to 65536 ids in the vocabulary


typedef struct {
    uint32_t key;     // (a << 16) | b
    uint32_t count;   // pair frequency; 0 = empty slot
} PairEntry;

typedef struct {
    uint32_t start;   // where the word begins inside ids[]
    uint32_t len;     // how many tokens it has now (shrinks at every merge)
    uint32_t count;   // how many times the word appears in the corpus
} Word;

/**
 * Merge DAG: token id 256 + i has children merges[i].left and merges[i].right.
 * Ids 0..255 are the leaves (raw bytes). Children are always created before their parent,
 * so they have smaller ids and the graph cannot contain cycles.
 */
typedef struct {
    token_t left;
    token_t right;
} Merge;

typedef struct {
    uint32_t key;   // (a << 16) | b, stessa codifica di PairEntry
    uint32_t id;    // token prodotto dal merge: 256 + rank
} MergeEntry;

typedef struct {
    Merge       merges[MAX_MERGES];
    uint32_t    n_merges;
    MergeEntry* hashtable;
    // later: MergeEntry *rank_map;   (a,b) -> rank
} BPE;

/**
 * One piece (word) of the text to tokenize. Owns no memory: it points into text and out.
 * Its tokens are written starting at out + start: a piece never produces more tokens than bytes,
 * so the slices of different pieces never overlap.
 */
typedef struct {
    uint32_t start;   // offset of the piece in text (and in out)
    uint32_t len;     // bytes of the piece
    uint32_t n_tok;   // tokens produced, filled by tokenize_piece
} Span;



// O(1) average: adds n to the count of pair (a, b) in the open addressing hashmap
void       add_pair(token_t a, token_t b, uint32_t n);

// O(n_ids + PAIR_CAP): recounts every pair and returns the most frequent one (caller must free it)
PairEntry* find_next_pair(Word* words, token_t* ids, size_t n_words);

// Fills words[] and ids[] from the corpus, deduplicating repeated words
void       init(char* corpus[], size_t corpus_len, Word* words, token_t* ids, size_t *unique_words, size_t *n_ids);

// Learns the merges until VOCAB_SIZE, MAX_MERGES or MERGES_THRESHOLD is reached
void       fit(Word* words, token_t* ids, size_t n_words, BPE* bpe_state);

// Encodes text into out (at least strlen(text) tokens), returns the number of tokens written
size_t     tokenize(const char* text, token_t* out, BPE* bpe_state, Span* out_spans);

// Decodes n tokens into a NUL terminated string in out (cap bytes, terminator included)
size_t     detokenize(const token_t* tokens, size_t n, char* out, size_t cap, BPE* bpe_state);

// Maps the file at path and splits it into words (GPT-2 style: a leading space belongs to the word)
// stored in a contiguous pool. Returns the word count, 0 on error (outputs set to NULL)
size_t     parse_training_corpus(char* path, char*** words_out, char** words_pool_out);

// Writes the learned merges to path (magic, n_merges, merges[n_merges]). Returns 0 or -1
int        bpe_save(const BPE* bpe_state, const char* path);

// Reads a model written by bpe_save into a BPE from bpe_init, validating it. Returns 0 or -1
int        bpe_load(BPE* bpe_state, const char* path);

// Allocates an empty model and its merge map on the heap. Returns NULL on error
BPE*       bpe_init(void);

// Releases a model created by bpe_init (NULL is allowed)
void       bpe_free(BPE* bpe_state);

#endif
