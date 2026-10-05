// must come before any #include: with -std=c11 the headers hide POSIX functions such as getline
#define _POSIX_C_SOURCE 200809L
// compile the arena functions in this file (and only here), see gian_arena.h
#define GIAN_ARENA_IMPL

#include "../include/turbo_bpe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/gian_arena.h"

static void usage(FILE* out, const char* prog)
{
    fprintf(out,
            "usage:\n"
            "  %s train <corpus> <model>   learn the merges from the corpus file and save them to "
            "model\n"
            "  %s live <model>             read lines from stdin and tokenize them with model\n"
            "  %s help                     show this message\n"
            "\n"
            "examples:\n"
            "  %s train test.txt out/model.bpe\n"
            "  %s live out/model.bpe              type a line + Enter, Ctrl+D to quit\n"
            "  echo \"ciao mondo\" | %s live out/model.bpe\n",
            prog, prog, prog, prog, prog, prog);
}

/**
 * Learns the merges from the corpus file at path and saves them to model_path.
 * Returns 0 on success, 1 on error.
 * We doont need any arena for performance reasons since this phase doesn't run in
 * a production environment.
 */
static int train(char* path, const char* model_path)
{
    char*    text_pool = NULL;
    char**   corpus    = NULL;
    token_t* ids       = NULL;
    Word*    words     = NULL;
    BPE*     bpe       = NULL;
    int      rc        = 1;

    // check the file first: parse_training_corpus returns 0 both on error and on an empty corpus
    FILE* probe = fopen(path, "rb");
    if (probe == NULL)
    {
        perror(path);
        return 1;
    }
    fclose(probe);

    size_t corpus_len = parse_training_corpus(path, &corpus, &text_pool);
    if (corpus_len == 0)
    {
        fprintf(stderr, "%s: the corpus contains no words\n", path);
        goto cleanup;
    }

    size_t raw_size = 0;
    for (size_t c = 0; c < corpus_len; c++)
        raw_size += strlen(corpus[c]);

    bpe   = bpe_init();
    ids   = (token_t*)malloc(sizeof(token_t) * raw_size);
    words = (Word*)malloc(sizeof(Word) * corpus_len);
    if (bpe == NULL || ids == NULL || words == NULL)
    {
        perror("malloc");
        goto cleanup;
    }

    size_t unique_words_count = 0;
    size_t unique_ids         = 0;

    init(corpus, corpus_len, words, ids, &unique_words_count, &unique_ids);

    // free the space taken by repeated words, this saves a lot of memory
    token_t* ids_shrunk   = (token_t*)realloc(ids, sizeof(token_t) * unique_ids);
    Word*    words_shrunk = (Word*)realloc(words, sizeof(Word) * unique_words_count);
    if (ids_shrunk != NULL)
        ids = ids_shrunk; // on failure the old, bigger block is still valid
    if (words_shrunk != NULL)
        words = words_shrunk;

    printf("corpus: %zu words, %zu unique, %zu bytes\n", corpus_len, unique_words_count, raw_size);

    /**
     * Every token is a 16 bit unsigned integer. Ids 0..255 are reserved for the raw bytes; every
     * new token found by the pair frequency method takes the first free id in ascending order.
     * The string of any token can be rebuilt by walking the merge DAG down to the bytes.
     */
    fit(words, ids, unique_words_count, bpe);

    if (bpe_save(bpe, model_path) != 0)
        goto cleanup;
    printf("saved %u merges to %s\n", bpe->n_merges, model_path);
    rc = 0;

cleanup:
    free(ids);
    free(words);
    free(corpus);
    free(text_pool);
    bpe_free(bpe);
    return rc;
}

/**
 * Interactive mode: reads a line from stdin up to Enter and tokenizes it, until EOF (Ctrl+D).
 * Works the same with a pipe or a redirected file, one line at a time.
 * Returns 0 on success, 1 on error.
 */
static int live(const char* model_path)
{
    BPE* bpe = bpe_init();
    if (bpe == NULL)
    {
        perror("malloc");
        return 1;
    }
    if (bpe_load(bpe, model_path) != 0)
    {
        bpe_free(bpe);
        return 1;
    }

    fprintf(stderr, "loaded %u merges from %s\n", bpe->n_merges, model_path);

    Arena*  a    = arena_create(1 << 20); // 1 MB: righe fino a ~70 KB
    if (a == NULL)
    {
        perror("arena_create");
        bpe_free(bpe);
        return 1;
    }
    char*   line = NULL;
    size_t  cap  = 0;
    ssize_t n;

    while ((n = getline(&line, &cap, stdin)) != -1)
    {
        if (n > 0 && line[n - 1] == '\n')
            line[--n] = '\0';
        arena_reset(a); // "libera" tutta la riga precedente in O(1)

        Span*    spans = (Span*)arena_alloc(a, sizeof(Span) * n); // dal più grande
        token_t* out   = (token_t*)arena_alloc(a, sizeof(token_t) * n);
        char*    text  = (char*)arena_alloc(a, n + 1);
        if (!spans || !out || !text)
        {
            fprintf(stderr, "line too long\n");
            continue;
        }
        memcpy(text, line, n + 1);

        size_t nt = tokenize(text, out, bpe, spans);
        printf("%zu bytes -> %zu tokens\n", (size_t)n, nt);

        // after the compaction the tokens of span i follow those of span i-1 in out: walk the
        // spans with a running offset. Every span has at least one token, so stopping when all
        // nt tokens are printed stops exactly after the last span
        size_t t = 0;
        for (size_t i = 0; t < nt; i++)
        {
            printf("  \"");
            for (uint32_t k = 0; k < spans[i].len; k++)
            {
                char c = text[spans[i].start + k];
                if (c == '\t')      printf("\\t");
                else if (c == '\r') printf("\\r");
                else                putchar(c);
            }
            printf("\" ->");
            for (uint32_t k = 0; k < spans[i].n_tok; k++)
                printf(" %u", out[t + k]);
            printf("\n");
            t += spans[i].n_tok;
        }
    }

    free(line);
    arena_free(a);
    bpe_free(bpe);
    return 0;
}

int main(int argc, char* argv[])
{
    const char* prog = argv[0];
    const char* cmd  = argc >= 2 ? argv[1] : NULL;

    if (cmd == NULL)
    {
        usage(stderr, prog);
        return 1;
    }
    if (strcmp(cmd, "help") == 0 || strcmp(cmd, "-h") == 0 || strcmp(cmd, "--help") == 0)
    {
        usage(stdout, prog);
        return 0;
    }
    if (strcmp(cmd, "train") == 0)
    {
        if (argc != 4)
        {
            fprintf(stderr, "train: expected <corpus> <model>\n\n");
            usage(stderr, prog);
            return 1;
        }
        return train(argv[2], argv[3]);
    }
    if (strcmp(cmd, "live") == 0)
    {
        if (argc != 3)
        {
            fprintf(stderr, "live: expected <model>\n\n");
            usage(stderr, prog);
            return 1;
        }
        return live(argv[2]);
    }

    fprintf(stderr, "unknown command: %s\n\n", cmd);
    usage(stderr, prog);
    return 1;
}
