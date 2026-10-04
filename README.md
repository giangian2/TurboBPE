# TurboBPE

A byte-level BPE (Byte Pair Encoding) tokenizer written from scratch in C11: it learns the
merges from a text corpus, saves them to a compact model file and tokenizes text with them.
The goal is a tokenizer fast enough for training datasets, built step by step and kept readable.

- Ids `0..255` are the raw bytes, so every input can be encoded, with no unknown token.
- Every learned merge `(left, right)` gets the next free id (`256 + i`); the merges form a DAG
  whose leaves are the bytes, so any token can be decoded back to its bytes.
- Text is split into pieces GPT-2 style: a space right before a word belongs to the word
  (`"il fuggiasco"` -> `"il"`, `" fuggiasco"`). Training and tokenization use the same rule.

## Build and run

```sh
make                         # bin/libbpe.a + bin/main, -O0 -g (easy to step through)
make OPT=-O2                 # optimized build, for profiling and measuring

./bin/main train test.txt out/model.bpe     # learn the merges, save the model
./bin/main live out/model.bpe               # type a line + Enter, Ctrl+D to quit
echo "il fuggiasco" | ./bin/main live out/model.bpe
./bin/main help
```

```
12 bytes -> 7 tokens
  "il" -> 458
  " fuggiasco" -> 275 491 105 291 99 111
```

Makefile shortcuts, with `ARGS` passed to `bin/main` (default: `train test.txt out/model.bpe`):

```sh
make run ARGS="live out/model.bpe"
make debug                   # gdb with the layout in gdb/init.gdb
make memcheck                # valgrind --leak-check=full
```

## Layout

| Path | Content |
|---|---|
| `include/turbo_bpe.h` | public API, `BPE` model, `Span`, `Word`, tuning constants |
| `src/bpe.c` | corpus parser, training (`init`, `fit`), `tokenize`, model save/load |
| `src/main.c` | CLI: `train`, `live` |
| `include/gian_arena.h` | linear arena with aligned allocations (used by `live`) |
| `include/hashmap.h` | generic open addressing hashmap (for the rank map) |
| `include/heap.h` | generic binary heap with a user comparator (for the merge queue) |
| `include/arena.h` | vendored arena by Alexey Kutepov (MIT) |
| `README_tokenize.md` | how `tokenize` works: spans, in-place pieces, compaction |

## Model file

```
"BPE1"                      4 bytes, magic (bump the digit when the format changes)
n_merges                    uint32
merges[n_merges]            { uint16 left, uint16 right }, token 256 + i = merges[i]
```

Native endianness. `bpe_load` rejects a wrong magic, too many merges, and any merge whose
children are not older tokens (that would break the DAG).

## Status

| Area | State | Notes |
|---|---|---|
| Corpus parser | ✅ done | `mmap`, words in one contiguous pool, leading-space pieces |
| Training (`init` + `fit`) | ✅ done | dedup of repeated words, most frequent pair merged until `VOCAB_SIZE` / `MERGES_THRESHOLD` |
| Model save / load | ✅ done | validated on load |
| `pretokenize` | ✅ done | every byte in exactly one span, same rule as training |
| `tokenize` | ✅ done | per-piece merge replay in place + compaction, no allocation |
| CLI `train` / `live` | ✅ done | `live` uses one arena reset per line |
| Arena | ✅ done | 8 byte aligned allocations, single-TU implementation |
| `detokenize` | 🚧 stub | declared, not implemented |
| Streaming detokenize | 📋 planned | [see below](#1-streaming-detokenize) |
| Rank hashmap + heap tokenize | 📋 planned | [see below](#2-tokenize-with-a-rank-hashmap--heap) |
| Multithreading with coroutines | 📋 planned | [see below](#3-multithreading-with-a-thread-pool-and-coroutines) |
| Word -> tokens cache | 📋 planned | per-thread, in front of `tokenize_piece` |
| Faster training | 📋 planned | `init` dedup is O(n²), `fit` recounts every pair at each merge |

### Known limitations

- `init` finds duplicate words with a linear search: O(n²) in the number of distinct words.
- `fit` recounts all pairs after every merge: O(n_ids + PAIR_CAP) per merge.
- `tokenize` replays every merge on every piece: O(len × n_merges).
- Whitespace-only pieces are never seen by training, so they always stay byte tokens.
- `VOCAB_SIZE` is a compile-time constant; the model file is in native endianness.
- `tokenize` takes a NUL terminated string, so the text cannot contain `\0` bytes.

## Roadmap

### 1. Streaming detokenize

An LLM produces **one token at a time**, and the text should be shown as it comes. Input does not
need streaming: the prompt is processed in one forward pass and tokenizing it costs microseconds,
so `tokenize` stays a batch function and streaming lives on the decoding side.

**Decoding one token.** `token_decode(id)` walks the merge DAG down to the bytes: ids `< 256` are
the byte itself, otherwise decode `merges[id - 256].left` then `.right`. With an explicit stack
there is no recursion and the depth is bounded. Faster, after `fit` or `bpe_load`: precompute the
bytes of every token once, into one pool plus an `(offset, len)` table, so decoding is a `memcpy`.

**The UTF-8 problem.** A token is a sequence of bytes, not of characters: `è` is `C3 A8`, and the
two bytes can end up in two different tokens. Printing the first token alone shows `�`.

**The streaming decoder** keeps the bytes of an incomplete character between calls:

```c
typedef struct {
    uint8_t pending[4];   // bytes of a UTF-8 character not complete yet
    uint8_t n_pending;
} Detokenizer;

// Appends the bytes of token to the pending ones and writes to out only complete characters.
// Returns the bytes written; the incomplete tail stays in d->pending for the next call.
size_t detok_push(Detokenizer* d, const BPE* bpe, token_t token, char* out, size_t cap);

// End of stream: writes whatever is left (an invalid tail is emitted as is, or as U+FFFD).
size_t detok_flush(Detokenizer* d, char* out, size_t cap);
```

The tail is incomplete when the last lead byte announces more bytes than follow it:
`110xxxxx` needs 1 continuation byte, `1110xxxx` needs 2, `11110xxx` needs 3; continuation bytes
are `10xxxxxx`. At most 3 bytes are ever held back.

`detokenize` (whole sequence) is then `detok_push` over every token followed by `detok_flush`,
so both paths share the same code and must give `detokenize(tokenize(x)) == x`.

### 2. Tokenize with a rank hashmap + heap

**Why.** Today every piece scans **all** the merges, even the ones whose bytes it does not
contain: with ~5000 merges a 6 byte word costs 5000 passes.

**Idea.** The rank of a merge is its index `m` in `merges[]`. Replaying merges in order is the
same as repeatedly applying, among the pairs **present** in the piece, the one with the lowest
rank: every merge before it would have found nothing to do. So only the pairs of the piece are
looked at.

**Rank map.** `(a << 16 | b) -> rank`, built with `hashmap.h` at the end of `fit` and after
`bpe_load` (it is derived data, the file format does not change). Stored in `BPE` as the
`rank_map` field already reserved in the struct.

**Simple version (short pieces), O(k²) for a piece of k tokens:**

```
while k > 1:
    find the adjacent pair with the lowest rank (hashmap lookup per pair)
    if no pair has a rank: stop
    merge_pair_in_seq(piece, k, a, b, 256 + rank)   // all occurrences, left to right
```

**Heap version (long pieces), O(k log k):**

- the piece becomes a doubly linked list over positions (`prev[]`, `next[]`), so a merge is O(1)
  and no token is moved;
- the heap holds candidates `(rank, position)`; `heap.h` is a max-heap with a user comparator,
  so the comparator returns `> 0` when `a` has the **lower** rank (ties: lower position first,
  which reproduces the left-to-right order of `merge_pair_in_seq`);
- pop the best candidate, skip it if stale (the tokens at that position changed since it was
  pushed), merge, push the new pairs it forms with its left and right neighbours;
- finally walk the list and write the tokens back to `piece[0..n_tok)`.

The heap and the list arrays are per-thread scratch, reused with `heap_clear`: never one
allocation per word. Short pieces (the large majority) keep the simple version, which is faster
below a few dozen tokens.

**Correctness check.** Both versions must produce exactly the tokens of the current merge replay
on the whole corpus; the replay stays in the code as the reference.

### 3. Multithreading with a thread pool and coroutines

Spans are independent: span `i` reads only `text[start .. start+len)` and writes only
`out[start .. start+len)`, and the model is read-only. Phase 2 of `tokenize` needs no lock.

**Threads for CPU parallelism.** A pool of POSIX threads, one per core, created once.

- The unit of work is a **block of spans** (for example ~64 KB of text), not a single word: a
  word is far too small to pay for scheduling.
- Each thread owns its scratch (its own arena, heap and linked list buffers, word cache), so
  `ARENA_THREAD_SAFE` is not needed and nothing is shared but the read-only model.
- Compaction after the barrier can be parallel too: each block counts its tokens, an exclusive
  prefix sum over the counts gives each block its destination offset, and every block moves its
  tokens there independently.

**Coroutines for the pipeline.** When the text arrives as a stream (a large file, stdin, a
socket), the work becomes three stages:

```
reader  ──chunks──▶  tokenizer workers  ──blocks of tokens──▶  writer
```

- **reader**: reads a chunk and splits it into spans; the last piece may continue in the next
  chunk, so it is held back until a separator arrives (a space starts the next word, so the
  previous one is complete);
- **workers**: tokenize blocks of spans on the thread pool;
- **writer**: emits blocks **in order** with a `next` index, even if block 5 finishes before
  block 2 (the same reordering idea as TCP).

Coroutines (stackful, e.g. `ucontext`, or a hand-written state machine) let the reader and the
writer suspend while waiting for I/O or for the next block in order, without blocking a thread:
many coroutines are multiplexed on few threads (M:N), threads give the parallelism, coroutines
give the cooperative scheduling between stages.

### 4. Smaller items

- **Word -> tokens cache** in front of `tokenize_piece`: frequent words (`" il"`, `" di"`) are
  tokenized once and then copied. Per-thread, so it needs no lock.
- **Faster training**: a hashmap for the word dedup in `init`; in `fit`, update the pair counts
  incrementally around each merge instead of recounting everything.
- **Learn whitespace too**: let training see the whitespace pieces so that runs like `"\n\n"`
  or indentation get their own tokens.

## License

GPL-3.0, see [LICENSE](LICENSE). `include/arena.h` is MIT, by Alexey Kutepov.
