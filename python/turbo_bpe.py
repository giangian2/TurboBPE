import ctypes
import threading
from array import array
from pathlib import Path

# bin/libbpe.so of this repo, built by make
_DEFAULT_LIB = str(Path(__file__).resolve().parent.parent / "bin" / "libbpe.so")

class Span(ctypes.Structure):
    _fields_ = [("start", ctypes.c_uint32), ("len", ctypes.c_uint32), ("n_tok", ctypes.c_uint32)]

class Tokenizer:
    def __init__(self, model_path, lib_path=_DEFAULT_LIB):
        self._lib = lib = ctypes.CDLL(lib_path)
        lib.bpe_init.restype = ctypes.c_void_p
        lib.bpe_init.argtypes = []
        lib.bpe_free.argtypes = [ctypes.c_void_p]
        lib.bpe_load.restype = ctypes.c_int
        lib.bpe_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.tokenize.restype = ctypes.c_size_t
        lib.tokenize.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint16),
                                 ctypes.c_void_p, ctypes.POINTER(Span), ctypes.c_void_p]
        lib.bpe_cache_create.restype = ctypes.c_void_p
        lib.bpe_cache_create.argtypes = []
        lib.bpe_cache_free.argtypes = [ctypes.c_void_p]

        # the word cache is written by tokenize, so every thread gets its own; the model is shared
        self._local = threading.local()
        self._caches = []          # every cache created, freed by close()
        self._caches_lock = threading.Lock()

        self._bpe = lib.bpe_init()
        if not self._bpe:
            raise MemoryError("bpe_init failed")
        if lib.bpe_load(self._bpe, model_path.encode()) != 0:
            lib.bpe_free(self._bpe)
            self._bpe = None
            raise ValueError(f"cannot load model {model_path!r}")

    def encode(self, text):
        data = text.encode("utf-8")
        if b"\0" in data:
            raise ValueError("text cannot contain NUL bytes (tokenize uses strlen)")
        n = len(data)
        out = (ctypes.c_uint16 * max(n, 1))()      # at most one token per byte
        spans = (Span * max(n, 1))()               # at most one span per byte
        k = self._lib.tokenize(data, out, self._bpe, spans, self._cache())
        return array("H", memoryview(out).cast("B")[: k * 2].cast("H"))

    def _cache(self):
        cache = getattr(self._local, "cache", None)
        if cache is None:
            cache = self._lib.bpe_cache_create()
            if not cache:
                raise MemoryError("bpe_cache_create failed")
            self._local.cache = cache
            with self._caches_lock:
                self._caches.append(cache)
        return cache

    def close(self):
        # call it once no thread is encoding anymore
        for cache in getattr(self, "_caches", []):
            self._lib.bpe_cache_free(cache)
        self._caches = []
        self._local = threading.local()   # drop the per-thread pointers to the freed caches
        if self._bpe:
            self._lib.bpe_free(self._bpe)
            self._bpe = None

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()
    def __del__(self): self.close()

if __name__ == "__main__":
    import sys
    with Tokenizer(sys.argv[1]) as tok:
        print(list(tok.encode("il fuggiasco")))
