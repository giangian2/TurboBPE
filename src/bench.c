// must come before any #include: with -std=c11 the headers hide POSIX functions such as clock_gettime
#define _POSIX_C_SOURCE 200809L

#include "../include/turbo_bpe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BENCH_RUNS 5   // timed runs per variant, the best one is reported

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/**
 * Reads the file at path and repeats it until the text is at least size bytes long.
 * Returns a NUL terminated string (caller must free it) and its length in *len, NULL on error.
 */
static char* load_text(const char* path, size_t size, size_t* len)
{
    FILE* f = fopen(path, "rb");
    if (f == NULL)
    {
        perror(path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long file_len = ftell(f);
    rewind(f);
    if (file_len <= 0)
    {
        fprintf(stderr, "%s: empty file\n", path);
        fclose(f);
        return NULL;
    }

    size_t copies = size > (size_t)file_len ? (size + (size_t)file_len - 1) / (size_t)file_len : 1;
    char*  text   = (char*)malloc(copies * (size_t)file_len + 1);
    if (text == NULL || fread(text, 1, (size_t)file_len, f) != (size_t)file_len)
    {
        perror("load_text");
        free(text);
        fclose(f);
        return NULL;
    }
    fclose(f);

    for (size_t c = 1; c < copies; c++)
        memcpy(text + c * (size_t)file_len, text, (size_t)file_len);
    *len       = copies * (size_t)file_len;
    text[*len] = '\0';
    return text;
}

/**
 * Best of BENCH_RUNS runs of tokenize, in seconds. With use_cache every run gets a fresh empty
 * cache (created outside the timed region), so the misses of the first words are measured too.
 * Returns a negative value on error.
 */
static double time_tokenize(const char* text, token_t* out, BPE* bpe, Span* spans, int use_cache,
                            size_t* n_tok)
{
    double best = 1e30;
    for (int r = 0; r < BENCH_RUNS; r++)
    {
        CacheEntry* cache = NULL;
        if (use_cache && (cache = bpe_cache_create()) == NULL)
            return -1.0;
        double t = now();
        *n_tok   = tokenize(text, out, bpe, spans, cache);
        t        = now() - t;
        if (t < best)
            best = t;
        bpe_cache_free(cache);
    }
    return best;
}

/**
 * Measures tokenize alone (no file reading, no allocation inside the timed region), without and
 * with the word cache, and checks that both give the same tokens. The text is a repeated file, so
 * caches and branch predictor are as warm as they can get and almost every piece is a cache hit:
 * read the result as an upper bound.
 */
int main(int argc, char** argv)
{
    if (argc != 4)
    {
        fprintf(stderr, "usage: %s <model> <text> <size in bytes>\n", argv[0]);
        return 1;
    }

    size_t   len    = 0;
    char*    text   = load_text(argv[2], strtoull(argv[3], NULL, 10), &len);
    BPE*     bpe    = bpe_init();
    token_t* out    = (token_t*)malloc(len * sizeof(token_t));
    token_t* cached = (token_t*)malloc(len * sizeof(token_t));
    Span*    spans  = (Span*)malloc(len * sizeof(Span));
    int      rc     = 1;
    if (text == NULL || bpe == NULL || out == NULL || cached == NULL || spans == NULL)
        goto cleanup;
    if (bpe_load(bpe, argv[1]) != 0)
        goto cleanup;

    tokenize(text, out, bpe, spans, NULL); // warm up
    size_t n_tok = 0, n_cached = 0;
    double plain = time_tokenize(text, out, bpe, spans, 0, &n_tok);
    double fast  = time_tokenize(text, cached, bpe, spans, 1, &n_cached);
    if (plain < 0 || fast < 0)
    {
        perror("bpe_cache_create");
        goto cleanup;
    }

    printf("%zu merges, %zu bytes -> %zu tokens (%.2f bytes/token)\n", (size_t)bpe->n_merges, len,
           n_tok, (double)len / (double)n_tok);
    printf("no cache:   %8.2f ms, %7.1f MB/s\n", plain * 1e3, (double)len / plain / 1e6);
    printf("word cache: %8.2f ms, %7.1f MB/s  (%.1fx)\n", fast * 1e3, (double)len / fast / 1e6,
           plain / fast);

    if (n_cached != n_tok || memcmp(out, cached, n_tok * sizeof(token_t)) != 0)
    {
        fprintf(stderr, "MISMATCH: the cache changed the tokens\n");
        goto cleanup;
    }
    rc = 0;

cleanup:
    free(spans);
    free(cached);
    free(out);
    bpe_free(bpe);
    free(text);
    return rc;
}
