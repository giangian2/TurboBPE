#include <fcntl.h> // Per la funzione open e le sue costanti (O_RDONLY, ecc.)
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h> // Per mmap, munmap, madvise, e le costanti (PROT_READ, MAP_PRIVATE, ecc.)
#include <sys/stat.h> // Per fstat e la struttura stat (necessaria per ottenere la dimensione del file)
#include <unistd.h>
#include <unistd.h> // Per close (per chiudere il file descriptor)

#include "turbo_bpe.h"

// static Merge        merges[MAX_MERGES];
static PairEntry pairs[PAIR_CAP];

// int n_merges        = 0;

// Sortable key: freq (high 32 bits) | a (16) | b (16) -> comparing u64 orders by freq, ties broken
// by the pair
static inline uint64_t pair_sort_key(PairEntry e)
{
    return ((uint64_t)e.count << 32) | e.key;
}
static inline token_t pair_key_left(uint64_t k)
{
    return (token_t)(k >> 16);
}
static inline token_t pair_key_right(uint64_t k)
{
    return (token_t)k;
}
static inline uint32_t pair_key_count(uint64_t k)
{
    return (uint32_t)(k >> 32);
}

static inline uint32_t pair_hash(uint32_t key)
{
    return (key * 2654435761u) >> (32 - PAIR_BITS);
}

/**
 * O(1) average: adds n to the count of pair (a, b) in the open addressing hashmap
 */
void add_pair(token_t a, token_t b, uint32_t n)
{
    uint32_t key = ((uint32_t)a << 16) | b;
    uint32_t i   = pair_hash(key);
    while (pairs[i].count != 0 && pairs[i].key != key)
        i = (i + 1) & (PAIR_CAP - 1); // slot taken by another pair: linear probing
    pairs[i].key = key;
    pairs[i].count += n; // n = words[w].count
}

static PairEntry* create_best_pair(token_t a, token_t b, uint32_t n)
{
    PairEntry* best = (PairEntry*)malloc(sizeof(PairEntry));
    if (best == NULL)
    {
        return best;
    }

    best->count = n;
    best->key   = ((uint32_t)a << 16) | b;
    return best;
}

/**
 * O(n_ids + PAIR_CAP): recounts every pair and returns the most frequent one
 */
PairEntry* find_next_pair(Word* words, token_t* ids, size_t n_words)
{
    memset(pairs, 0, sizeof(pairs)); // counts must be recomputed from scratch after every merge

    // 1. aggregate in the hashmap: each pair weighs as much as its word frequency
    for (size_t c = 0; c < n_words; c++)
    {
        for (uint32_t j = 0; j + 1 < words[c].len; j++)
        { // j+1 < len: no underflow when len == 0
            token_t a = ids[words[c].start + j];
            token_t b = ids[words[c].start + j + 1];
            add_pair(a, b, words[c].count);
        }
    }

    // 2. linear scan for the max; comparing u64 keys (freq | a | b) means that on equal freq the
    // pair with higher ids wins
    uint64_t max = 0;
    for (size_t i = 0; i < PAIR_CAP; i++)
    {
        if (pairs[i].count != 0 && pair_sort_key(pairs[i]) > max)
            max = pair_sort_key(pairs[i]);
    }

    if (max == 0)
        return NULL; // no pairs (every word has len < 2)
    return create_best_pair(pair_key_left(max), pair_key_right(max),
                            pair_key_count(max)); // caller must free it
}

/**
 * O(len): replaces every (a, b) in seq with new_id, compacting in place; returns the new length
 */
static uint32_t merge_pair_in_seq(token_t* seq, uint32_t len, token_t a, token_t b, token_t new_id)
{
    uint32_t r = 0, w = 0; // read / write cursors, w <= r so we never overwrite unread tokens
    while (r < len)
    {
        if (r + 1 < len && seq[r] == a && seq[r + 1] == b)
        {
            seq[w++] = new_id;
            r += 2;
        }
        else
        {
            seq[w++] = seq[r++];
        }
    }
    return w;
}

/**
 * O(n^2) in the number of words, because of the linear search for duplicates
 */
void init(char* corpus[], size_t corpus_len, Word* words, token_t* ids, size_t* unique_words,
          size_t* n_ids)
{
    if (!ids || !words)
        return;

    // fill words[] and ids[], deduplicating repeated words
    size_t n_words = 0; // distinct words
    size_t pos     = 0; // first free position in ids[]

    for (size_t c = 0; c < corpus_len; c++)
    {
        const uint8_t* s   = (const uint8_t*)corpus[c];
        size_t         len = strlen(corpus[c]);

        size_t w;
        for (w = 0; w < n_words; w++)
        {
            if (words[w].len != len)
                continue;
            size_t k = 0;
            while (k < len && ids[words[w].start + k] == s[k])
                k++;
            if (k == len)
                break;
        }

        if (w < n_words)
        {
            words[w].count++;
        }
        else
        {
            words[n_words].start = pos;
            words[n_words].len   = len;
            words[n_words].count = 1;
            for (size_t k = 0; k < len; k++)
                ids[pos++] = s[k];
            n_words++;
        }
    }

    *unique_words = n_words;
    *n_ids        = pos;
}

/**
 * Learns the merges: picks the most frequent pair, gives it the next free id, records it in the
 * DAG and rewrites every word with the new token. Stops at VOCAB_SIZE, MAX_MERGES, or when the
 * best pair is rarer than MERGES_THRESHOLD.
 * Words shrink in place: each keeps its own start, leaving unused gaps in ids[] behind it.
 */
void fit(Word* words, token_t* ids, size_t n_words, BPE* bpe_state)
{
    // first free id is 256 + n_merges: 0..255 are reserved for raw bytes, merged tokens start after
    // them
    while (256 + bpe_state->n_merges < VOCAB_SIZE && bpe_state->n_merges < MAX_MERGES)
    {
        PairEntry* best = find_next_pair(words, ids, n_words);
        if (best == NULL)
            break;
        if (best->count < MERGES_THRESHOLD)
        {
            free(best);
            break;
        }

        token_t a      = (token_t)(best->key >> 16);
        token_t b      = (token_t)best->key;
        token_t new_id = (token_t)(256 + bpe_state->n_merges);
        free(best);

        bpe_state->merges[bpe_state->n_merges++] = (Merge){a, b}; // new_id == 256 + (n_merges - 1)

        for (size_t w = 0; w < n_words; w++)
            words[w].len = merge_pair_in_seq(ids + words[w].start, words[w].len, a, b, new_id);
    }
}

// Whitespace that separates the pieces of the text. pretokenize must use the same rule as the
// training parser, otherwise the merges learned by fit do not match the pieces seen by tokenize.
static inline int is_separator(char c)
{
    return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}

static uint32_t tokenize_piece(const BPE* bpe, token_t* piece, uint32_t len)
{
    for (uint32_t m = 0; m < bpe->n_merges && len > 1; m++)     // same order as fit
        len = merge_pair_in_seq(piece, len, bpe->merges[m].left, bpe->merges[m].right,
                                (token_t)(256 + m));
    return len;
}

static size_t pretokenize(const uint8_t* text, size_t len, Span* spans)
{
    size_t n = 0, i = 0;
    while (i < len)
    {
        size_t start = i;
        if (text[i] == ' ' && i + 1 < len && !is_separator(text[i + 1]))
        {
            i++;
            while (i < len && !is_separator(text[i]))
                i++;
        }
        else if (is_separator(text[i]))
        {
            while (i < len && is_separator(text[i]))
            {
                if (text[i] == ' ' && i + 1 < len && !is_separator(text[i + 1]))
                    break;
                i++;
            }
        }
        else
        {
            while (i < len && !is_separator(text[i]))
                i++;
        }
        spans[n++] = (Span){(uint32_t)start, (uint32_t)(i - start), 0};
    }
    return n;
}

/**
 * O(len * n_merges): encodes text into out, which must hold at least strlen(text) tokens, and
 * describes its pieces in spans (at least strlen(text) entries).
 * 1. out gets one token per byte, pretokenize splits the text into spans
 * 2. every piece is merged in place inside its own slice out[start .. start+len)
 * 3. compaction: the slices are moved one after the other, closing the gaps left by the merges
 * Returns the number of tokens, stored in out[0 .. return).
 */
size_t tokenize(const char* text, token_t* out, BPE* bpe_state, Span* out_spans)
{
    const uint8_t* s   = (const uint8_t*)text; // unsigned: bytes >= 128 stay positive
    size_t       len = strlen(text);

    // Copy into out the starting tokens corresponding to the base characters
    for (size_t i = 0; i < len; i++)
        out[i] = s[i];

    size_t n_spans = pretokenize(s, len, out_spans);

    for (size_t i = 0; i < n_spans; i++)
    {
        out_spans[i].n_tok = tokenize_piece(bpe_state, out + out_spans[i].start, out_spans[i].len);
    }

    // compaction: w is where the next piece goes. w <= start always (every piece wrote at most
    // len tokens), so the move only goes backwards and never overwrites a piece not moved yet;
    // memmove because source and destination may overlap
    size_t w = 0;
    for (size_t i = 0; i < n_spans; i++)
    {
        memmove(out + w, out + out_spans[i].start, out_spans[i].n_tok * sizeof(token_t));
        w += out_spans[i].n_tok;
    }
    return w;
}

/**
 * Decodes n tokens into a NUL terminated string in out (cap bytes, terminator included).
 * Returns the string length, or SIZE_MAX if out is too small or a token id is unknown.
 */
size_t detokenize(const token_t* tokens, size_t n, char* out, size_t cap, BPE* bpe_state)
{
}


/**
 * For parsing words from the given file, we need to map it into memory,
 * for performance reasons instead of classic read/write api that doeas too many ctx-switches.
 * We have to improve cache locality!! So we will return an array of char pointers, and each stirng
 * (word) is allocated into an arena (a contiguous memory area) respecting the original ordering.
 *
 * Pieces follow the GPT-2 convention: a single space right before a word belongs to the word,
 * "il fuggiasco e" -> "il", " fuggiasco", " e". This way fit learns merges with the leading
 * space (" e", " il", ...), the same pieces tokenize will see. A word at the start of the file
 * or of a line has no leading space; extra spaces, tabs and newlines are dropped (for training
 * only the frequencies matter, tokenize instead must keep them for the round trip).
 *
 * Returns the number of words. On error or with no words returns 0 and *words_out, *words_pool_out
 * are NULL (free(NULL) is a no-op, the caller can always free them).
 */
size_t parse_training_corpus(char* path, char*** words_out, char** words_pool_out)
{
    *words_out      = NULL;
    *words_pool_out = NULL;

    int corpus_fd = open(path, O_RDONLY);
    if (corpus_fd == -1)
    {
        perror(path);
        return 0;
    }

    struct stat corpus_fs;
    if (fstat(corpus_fd, &corpus_fs) == -1)
    {
        perror(path);
        close(corpus_fd);
        return 0;
    }
    size_t file_size = corpus_fs.st_size;

    if (file_size == 0)
    { // mmap rejects a length of 0
        close(corpus_fd);
        return 0;
    }

    char* corpus_data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, corpus_fd, 0);
    close(corpus_fd); // the mapping stays valid after close
    if (corpus_data == MAP_FAILED)
    {
        perror(path);
        return 0;
    }

    /**
     * Pool size: every word takes its bytes plus the terminator. A word is at least one non
     * separator byte followed by a separator (or EOF), so there are at most (file_size + 1) / 2
     * words, hence at most that many terminators beyond the bytes of the file.
     */
    size_t words_capacity = 1024;
    char** words          = (char**)malloc(sizeof(char*) * words_capacity);
    char*  pool           = (char*)malloc(file_size + (file_size + 1) / 2 + 1);
    if (words == NULL || pool == NULL)
    {
        perror("malloc");
        free(words);
        free(pool);
        munmap(corpus_data, file_size);
        return 0;
    }

    size_t n_words     = 0;
    size_t seek        = 0;
    size_t pool_offset = 0;

    while (seek < file_size)
    {
        while (seek < file_size && is_separator(corpus_data[seek]))
            seek++;
        if (seek >= file_size)
            break;

        // the space right before the word, if any, is part of it
        size_t word_start = (seek > 0 && corpus_data[seek - 1] == ' ') ? seek - 1 : seek;
        while (seek < file_size && !is_separator(corpus_data[seek]))
            seek++;
        size_t word_len = seek - word_start;

        if (n_words == words_capacity)
        {
            words_capacity *= 2;
            char** grown = (char**)realloc(words, sizeof(char*) * words_capacity);
            if (grown == NULL)
            {
                perror("realloc");
                free(words);
                free(pool);
                munmap(corpus_data, file_size);
                return 0;
            }
            words = grown;
        }

        memcpy(&pool[pool_offset], &corpus_data[word_start], word_len);
        words[n_words++] = &pool[pool_offset];
        pool_offset += word_len;
        pool[pool_offset++] = '\0'; // terminator right after the word, then move past it
    }

    munmap(corpus_data, file_size);

    if (n_words == 0)
    { // only separators in the file
        free(words);
        free(pool);
        return 0;
    }

    *words_out      = words;
    *words_pool_out = pool;
    return n_words;
}

/**
 * Writes the learned merges to path: magic, n_merges, then the merges in the order they were
 * learned (the order is the model: token id 256 + i is merges[i]). Only the n_merges used entries
 * are written. Returns 0 on success, -1 on error.
 */
int bpe_save(const BPE* bpe_state, const char* path)
{
    FILE* f = fopen(path, "wb");
    if (f == NULL)
    {
        perror(path);
        return -1;
    }

    uint32_t n  = bpe_state->n_merges;
    int      ok = fwrite(BPE_MAGIC, 1, 4, f) == 4 && fwrite(&n, sizeof(n), 1, f) == 1 &&
                  fwrite(bpe_state->merges, sizeof(Merge), n, f) == n;

    if (fclose(f) != 0)
        ok = 0;
    if (!ok)
    {
        perror(path);
        return -1;
    }
    return 0;
}

/**
 * Reads a model written by bpe_save into bpe_state. Rejects files with a wrong magic, too many
 * merges, or a merge whose children are not older tokens (that would break the DAG walked by
 * detokenize). Returns 0 on success, -1 on error.
 */
int bpe_load(BPE* bpe_state, const char* path)
{
    FILE* f = fopen(path, "rb");
    if (f == NULL)
    {
        perror(path);
        return -1;
    }

    char     magic[4];
    uint32_t n;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, BPE_MAGIC, 4) != 0 ||
        fread(&n, sizeof(n), 1, f) != 1 || n > MAX_MERGES || 256 + n > VOCAB_SIZE ||
        fread(bpe_state->merges, sizeof(Merge), n, f) != n)
    {
        fprintf(stderr, "%s: not a valid BPE model\n", path);
        fclose(f);
        return -1;
    }
    fclose(f);

    for (uint32_t i = 0; i < n; i++)
    {
        if (bpe_state->merges[i].left >= 256 + i || bpe_state->merges[i].right >= 256 + i)
        {
            fprintf(stderr, "%s: merge %u references a token that does not exist yet\n", path, i);
            return -1;
        }
    }
    bpe_state->n_merges = n;
    return 0;
}
