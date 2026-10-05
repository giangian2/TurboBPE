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
make                         # bin/libbpe.a + bin/libbpe.so + bin/main, -O0 -g (easy to step through)
make OPT=-O2                 # optimized build

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

Benchmark and profile of `tokenize` (`bin/bench` is always built with `-O2 -g`, whatever `OPT` is):

```sh
make bench                   # time tokenize without and with the word cache, check same tokens
make profile                 # callgrind, cost per function (full data in build/callgrind.out)
make bench BENCH_MODEL=out/model2.bpe BENCH_TEXT=test2.txt BENCH_SIZE=10000000
```

```
307 merges, 2001742 bytes -> 662316 tokens (3.02 bytes/token)
no cache:      56.68 ms,    35.3 MB/s
word cache:    18.43 ms,   108.6 MB/s  (3.1x)
```

The text is a small file repeated up to `BENCH_SIZE`: caches are as warm as they get and almost
every piece is a cache hit, so read the numbers as an upper bound. A large, varied corpus gives
the real figure.

### From Python

`bin/libbpe.so` is loaded with `ctypes` by `python/turbo_bpe.py` (rebuild it with `make` after
changing the C code):

```python
import sys; sys.path.insert(0, "python")
from turbo_bpe import Tokenizer

with Tokenizer("out/model.bpe") as tok:
    ids = tok.encode("il fuggiasco")   # array('H', [458, 275, 491, 105, 291, 99, 111])
```

`ctypes` releases the GIL during the C call, so `encode` can be called from many threads at once
(`ThreadPoolExecutor`): the model is shared read-only, every thread gets its own word cache.

## Layout

| Path | Content |
|---|---|
| `include/turbo_bpe.h` | public API, `BPE` model, `Span`, `Word`, tuning constants |
| `src/bpe.c` | corpus parser, training (`init`, `fit`), `tokenize`, model save/load |
| `src/main.c` | CLI: `train`, `live` |
| `src/bench.c` | benchmark of `tokenize`, without and with the word cache (`make bench`, `make profile`) |
| `python/turbo_bpe.py` | `ctypes` wrapper around `bin/libbpe.so`: `Tokenizer(model).encode(text)` |
| `include/gian_arena.h` | linear arena with aligned allocations (used by `live`) |
| `include/hashmap.h` | generic open addressing hashmap (merge map, word cache) |
| `include/heap.h` | generic binary heap with a user comparator (for the merge queue) |
| `README_tokenize.md` | how `tokenize` works: spans, in-place pieces, compaction |

## Model file

```
"BPE1"                      4 bytes, magic (bump the digit when the format changes)
n_merges                    uint32
merges[n_merges]            { uint16 left, uint16 right }, token 256 + i = merges[i]
```

Native endianness. `bpe_load` rejects a wrong magic, too many merges, and any merge whose
children are not older tokens (that would break the DAG).

## Word cache

`tokenize_piece` is the expensive part, and its result depends only on the bytes of the piece:
`" the"` always gives the same tokens. Natural text repeats the same words all the time, so the
first time a piece is seen its tokens are stored, and the next times they are copied.

- **What it is.** A `hashmap.h` table of `CacheEntry`: key = the piece bytes (zero padded) and
  their length, value = the tokens. Pieces longer than `CACHE_MAX_LEN` bytes skip the cache (rare,
  seldom repeated), so every entry has a fixed size and lives entirely in its slot: one lookup,
  one cache line, no second allocation. `CACHE_CAP` slots, about 1.5 MB.
- **Who owns it.** It must remember words across calls, so it cannot live inside `tokenize`: the
  caller creates it with `bpe_cache_create` and passes it to every call (`NULL` = no cache).
  `tokenize` writes to it, so it is **one per thread**; the `BPE` is read-only and shared.
- **When it is full.** At half capacity it is emptied (`hash_table_clear`) and refilled: linear
  probing stays fast, and the frequent words come back after a few lines.
- **Cost.** A hit is one lookup + `memcpy` instead of every merge step of the piece (each with one
  merge map lookup per pair); a miss costs one extra lookup and one insert.

In `live` the cache lives for the whole session, apart from the arena that is reset at every line.

## Status

| Area | State | Notes |
|---|---|---|
| Corpus parser | ✅ done | `mmap`, words in one contiguous pool, leading-space pieces |
| Training (`init` + `fit`) | ✅ done | dedup of repeated words, most frequent pair merged until `VOCAB_SIZE` / `MERGES_THRESHOLD` |
| Model save / load | ✅ done | validated on load |
| `pretokenize` | ✅ done | every byte in exactly one span, same rule as training |
| `tokenize` | ✅ done | per piece: lowest-rank adjacent pair via the merge map, in place + compaction, no allocation |
| Word -> tokens cache | ✅ done | caller-owned, one per thread, [see above](#word-cache) |
| Benchmark / profile | ✅ done | `make bench`, `make profile` |
| CLI `train` / `live` | ✅ done | `live` uses one arena reset per line |
| Arena | ✅ done | 8 byte aligned allocations, single-TU implementation |
| `detokenize` | 🚧 stub | declared, not implemented |
| Streaming detokenize | 📋 planned | [see below](#1-streaming-detokenize) |
| Rank hashmap + heap tokenize | 🚧 partial | merge map done (simple version), heap planned, [see below](#2-tokenize-with-a-rank-hashmap--heap) |
| Thread-safe encode API | 📋 planned | explicit length, `const` model, [see below](#3-production-use-from-python) |
| Python binding | ✅ done | `bin/libbpe.so` + `python/turbo_bpe.py` (ctypes), GIL released, tokens as `array('H')` |
| Training on large corpora | 📋 planned | count unique words instead of keeping every occurrence, [see below](#4-training-on-large-corpora) |

### Known limitations

- `init` finds duplicate words with a linear search: O(n²) in the number of distinct words.
- `fit` recounts all pairs after every merge: O(n_ids + PAIR_CAP) per merge.
- Training keeps every word occurrence in memory, so the corpus size is bounded by RAM.
- `tokenize_piece` looks up every adjacent pair again after each merge: O(len²) merge map lookups
  per piece (cache misses only).
- The merge map has a fixed capacity (`2 * VOCAB_SIZE` slots, no rehash), but `bpe_load` accepts
  up to `MAX_MERGES`: a model with more merges than that would not fit the map.
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

The heap and the list arrays are per-thread scratch (each calling thread has its own), reused
with `heap_clear`: never one allocation per word. Short pieces (the large majority) keep the simple version, which is faster
below a few dozen tokens.

**Correctness check.** Both versions must produce exactly the tokens of the current merge replay
on the whole corpus; the replay stays in the code as the reference.

### 3. Production use from Python

The library is meant to be called from Python: compiled as a `.so`, loaded with `ctypes` (or a
C extension later). There is **no parallelism inside the library**: no thread pool, no
coroutines, no batch API. Each call tokenizes one text on the thread that makes it, and the
concurrency belongs to the caller.

**Why no internal parallelism.**

- A prompt is a few KB and tokenizes in microseconds: splitting it across threads would cost
  more than it saves.
- On Linux a Python `threading.Thread` is a POSIX thread. An inference server that tokenizes
  many requests at once is already calling the library from many threads in parallel.
- Bulk work (tokenizing a whole dataset before training, indexing documents) parallelizes
  across **documents**, which are already independent. The caller does it with its own pool,
  as tiktoken does:

  ```python
  with ThreadPoolExecutor(8) as ex:
      ids = list(ex.map(tok.encode, texts))
  ```

**What the library must guarantee instead.**

- **Reentrant.** Any number of threads may call `encode` at the same time: the model is
  read-only, the buffers belong to the caller, no global state is touched (the static pair
  table in `bpe.c` is used by training only).
- **The GIL is released during the call**, otherwise Python runs one thread at a time.
  `ctypes` does it automatically; a C extension wraps the call in
  `Py_BEGIN_ALLOW_THREADS` / `Py_END_ALLOW_THREADS` and touches no Python object meanwhile.

**API changes this needs.**

```c
// explicit length: a Python bytes object may contain \0; const model: shared by every thread
size_t bpe_encode(const BPE* bpe, const uint8_t* text, size_t len, token_t* out, Span* scratch);
size_t bpe_decode(const BPE* bpe, const token_t* tokens, size_t n, char* out, size_t cap);
```

- Scratch (`out`, `spans`) can come from a per-thread arena (`_Thread_local`), so a call does
  not `malloc`.
- Tokens go back to Python as one `uint16` buffer (a NumPy array or `bytes`), never as a list
  of Python `int` objects: building that list costs more than the tokenization itself.

### 4. Training on large corpora

`mmap` is not the limit: it works with files larger than RAM, the kernel pages them in and out
(`madvise(MADV_SEQUENTIAL)` helps). The limit is that `parse_training_corpus` copies **every
occurrence** of every word into the pool, and `init` deduplicates them in O(n²).

Training only needs the **unique words with their counts**: `fit` already works on `Word`
entries with a `count` field. The number of unique words grows far slower than the corpus, so:

1. **Count in chunks.** Read the corpus a chunk at a time and update a `word -> count` hashmap
   (`hashmap.h`). A word cut at the end of a chunk is carried over to the next one.
2. **Build `words[]` and `ids[]`** from the unique words only, with no O(n²) search.
3. **Run `fit` unchanged**, on data that fits in memory whatever the corpus size.

If even the unique words are too many, the rarest ones are dropped: they barely affect the
merges. Hugging Face tokenizers and SentencePiece train the same way.

### 5. Smaller items

- **Incremental pair counts in `fit`**: update the counts around each merge instead of
  recounting every pair.
- **Learn whitespace too**: let training see the whitespace pieces so that runs like `"\n\n"`
  or indentation get their own tokens.

## License

GPL-3.0, see [LICENSE](LICENSE).
