# Tokenize: architecture

> Status: implemented, single threaded, merges replayed in training order (O(len × n_merges)).
> The faster algorithm and the parallel version plug into this same structure, see
> [README.md](README.md#roadmap).

## The key invariant

**A piece of `len` bytes never produces more than `len` tokens.**

Tokenization starts with one token per byte (ids 0..255 are the raw bytes) and every merge
replaces two tokens with one. Sequences only shrink. Everything below relies on this.

## Three levels: text, word, token

| | Works on | Produces |
|---|---|---|
| `tokenize` | the whole text | all the tokens, compacted |
| `Span` | describes one word (piece) | where it is + how many tokens it produced |
| `tokenize_piece` | one word | the tokens of that word |

The unit of work, and later the unit of parallelism, is the **word**. Tokens are the result.

## Memory: no allocation inside tokenize

The caller provides both buffers, sized on the text length:

```c
size_t len = strlen(text);
token_t *out   = malloc(sizeof(token_t) * len);   // tokens of every word
Span    *spans = malloc(sizeof(Span)    * len);   // description of every word (at most len pieces)

size_t n = tokenize(text, out, &model, spans);
```

No allocation per word and no allocation per token. Both buffers can be reused across calls;
the `live` command takes them from an arena that is reset at every line.

`out` is **aligned with the text**: the word starting at byte `start` writes its tokens starting
at `out + start`. Thanks to the invariant, that slice is always big enough, and the slices of
different words never overlap.

## Data structures

```c
typedef struct {
    Merge    merges[MAX_MERGES];
    uint32_t n_merges;
    // later: MergeEntry *rank_map;   (a,b) -> rank
} BPE;

typedef struct {
    uint32_t start;   // offset of the piece in text (and in out)
    uint32_t len;     // bytes of the piece
    uint32_t n_tok;   // filled by tokenize_piece
} Span;
```

- `BPE` is the trained model. **`fit` writes it, `tokenize` only reads it.**
  Training-only state (`pairs`, `ids`) does not belong here.
- A `Span` owns no memory: it is three numbers pointing into `text` and `out`. Input
  (`start`, `len`) and result (`n_tok`) live together, so a span is a self-contained task.

## Functions

```c
static size_t   pretokenize(const uint8_t *text, size_t len, Span *spans);
static uint32_t tokenize_piece(const BPE *bpe, token_t *piece, uint32_t len);
size_t          tokenize(const char *text, token_t *out, BPE *bpe, Span *spans);
```

- **`pretokenize`** splits the text into pieces without copying anything. Every byte ends up in
  exactly one span, in order, so `detokenize(tokenize(x)) == x`. Three kinds of piece:
  - a space followed by a word belongs to the word (GPT-2 style): `" fuggiasco"`;
  - a word with no space before it (start of text, after `\n` or `\t`): `"il"`;
  - leftover whitespace (extra spaces, tabs, newlines) becomes a piece of its own.

  `"la  notte\t buia "` -> `"la"` `" "` `" notte"` `"\t"` `" buia"` `" "`.
- **`tokenize_piece`** is a pure function: it reads the model, works in place on
  `piece[0..len)` (already holding one token per byte) and returns the new length. It replays the
  merges in the order `fit` learned them and stops as soon as one token is left. Later it uses a
  rank hashmap + heap with the same signature.
- **`tokenize`** orchestrates the three phases below.

## The three phases

```c
for (size_t i = 0; i < len; i++)                   // 1. one token per byte, split into spans
    out[i] = s[i];
size_t n_spans = pretokenize(s, len, spans);

for (size_t i = 0; i < n_spans; i++)               // 2. later: one task per block of spans
    spans[i].n_tok = tokenize_piece(bpe, out + spans[i].start, spans[i].len);

size_t w = 0;                                      // 3. compaction: close the gaps
for (size_t i = 0; i < n_spans; i++) {
    memmove(out + w, out + spans[i].start, spans[i].n_tok * sizeof(token_t));
    w += spans[i].n_tok;
}
return w;
```

1. **Split** the text into spans, with `out` holding the byte tokens.
2. **Tokenize** each span inside its own slice of `out`. Independent work, the parallel part.
3. **Compact**: sequential O(n) pass after the barrier. `memmove` is safe because
   `w <= spans[i].start` always.

## Example: `"il fuggiasco"`

Token ids from `out/model.bpe` trained on `test.txt` (they change with the corpus):

```
text:   i   l   ␣   f   u   g   g   i   a   s   c   o
        0   1   2   3   4   5   6   7   8   9  10  11

spans:  [ {0,2} {2,10} ]

out after phase 2:
       [458  l  275 491 105 291  99 111  g   i   a   s ]     n_tok: 1, 6
        └piece0┘└─────────────── piece 1 ───────────────┘
                                         └ stale bytes, ignored ┘

out after phase 3:
       [458 275 491 105 291  99 111  ·   ·   ·   ·   · ]     return 7
```

Each piece shrinks inside its own slice and leaves stale tokens behind it; the compaction reads
only the first `n_tok` of every slice.

## Training uses the same split

`fit` must see the corpus split by the **same** rule as `pretokenize`, otherwise the learned
merges do not match the pieces seen at encode time. Both use `is_separator` and attach a single
leading space to the word. The difference: training drops the whitespace-only pieces (only
frequencies matter there), so no merge ever involves them and they always stay byte tokens.

## Later, without touching this base

| Feature | Where it plugs in |
|---|---|
| coroutines + POSIX thread pool | phase 2: one task per **block** of spans (one per word is too fine grained), barrier before phase 3 |
| hashmap `(a,b) -> rank` | field in `BPE`, built at the end of `fit` and after `bpe_load`, read in `tokenize_piece` |
| heap ordered by rank | inside `tokenize_piece`: one buffer per thread, never one allocation per word |
| word -> tokens cache | the only shared mutable state: per-thread, or protected |
