#ifndef HEAP_H
#define HEAP_H

/*
 * Max heap generico, header-only, con shadow header (stile stb_ds).
 *
 *   uint64_t* h = heap_create(uint64_t, 64, heap_cmp_u64);
 *   heap_push(h, 42);              // puo' fare realloc: h deve essere una variabile
 *   uint64_t top = heap_peek(h);   // O(1)
 *   uint64_t max = heap_pop(h);    // O(logn)
 *   heap_free(h);
 *
 * peek/pop richiedono heap_len(h) > 0.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define heap__hdr(h)                ((MaxHeapHeader*)(void*)(h) - 1)
#define heap_create(T, cap, cmp)    ((T*)heap__create(sizeof(T), (cap), (cmp)))
#define heap_len(h)                 (heap__hdr(h)->count)
#define heap_clear(h)               (heap__hdr(h)->count = 0)
#define heap_push(h, val)           ((h) = heap__grow((h)), (h)[heap__hdr(h)->count] = (val), \
                                    heap__sift_up((h), heap__hdr(h)->count++))
#define heap_peek(h)                ((h)[0])                                    // O(1), heap non vuoto
#define heap_pop(h)                 (heap__pop(h), (h)[heap__hdr(h)->count])    // O(logn), heap non vuoto
#define heap_free(h)                do { free(heap__hdr(h)); (h) = NULL; } while (0)

typedef int (*heap_cmp_fn)(const void* a, const void* b);   // >0 se a > b

// Shadow header: sta subito prima del primo elemento, l'utente vede solo T*
typedef struct{
	size_t      cap;
	size_t      count;
	size_t      elem_size;
	heap_cmp_fn cmp;
} MaxHeapHeader;

static inline void* heap__create(size_t elem_size, size_t cap, heap_cmp_fn cmp){
	if (cap == 0) cap = 16;
	//Single malloc for contiuous RAM
	MaxHeapHeader* h = (MaxHeapHeader*)malloc(sizeof(MaxHeapHeader) + cap*elem_size);
	if (!h) return NULL;
	h->cap       = cap;
	h->count     = 0;
	h->elem_size = elem_size;
	h->cmp       = cmp;
	return h + 1;
}

// Garantisce spazio per un elemento in piu'; puo' spostare l'heap (realloc)
static inline void* heap__grow(void* data){
	MaxHeapHeader* h = heap__hdr(data);
	if (h->count < h->cap) return data;
	size_t cap = h->cap * 2;
	h = (MaxHeapHeader*)realloc(h, sizeof(MaxHeapHeader) + cap*h->elem_size);
	if (!h) { perror("heap realloc"); exit(1); }
	h->cap = cap;
	return h + 1;
}

static inline void heap__swap(unsigned char* a, unsigned char* b, size_t n){
	while (n--) { unsigned char t = *a; *a++ = *b; *b++ = t; }
}

static inline void heap__sift_up(void* data, size_t i){
	MaxHeapHeader* h  = heap__hdr(data);
	unsigned char* base = (unsigned char*)data;
	size_t es = h->elem_size;
	while (i > 0) {
		size_t p = (i - 1) / 2;
		if (h->cmp(base + i*es, base + p*es) <= 0) break;
		heap__swap(base + i*es, base + p*es, es);
		i = p;
	}
}

static inline void heap__sift_down(void* data, size_t i){
	MaxHeapHeader* h  = heap__hdr(data);
	unsigned char* base = (unsigned char*)data;
	size_t es = h->elem_size, n = h->count;
	for (;;) {
		size_t l = 2*i + 1, r = l + 1, m = i;
		if (l < n && h->cmp(base + l*es, base + m*es) > 0) m = l;
		if (r < n && h->cmp(base + r*es, base + m*es) > 0) m = r;
		if (m == i) break;
		heap__swap(base + i*es, base + m*es, es);
		i = m;
	}
}

// Sposta il max in fondo (indice count dopo il decremento), cosi' heap_pop lo legge tipizzato
static inline void heap__pop(void* data){
	MaxHeapHeader* h = heap__hdr(data);
	unsigned char* base = (unsigned char*)data;
	h->count--;
	heap__swap(base, base + h->count*h->elem_size, h->elem_size);
	heap__sift_down(data, 0);
}

static inline int heap_cmp_u64(const void* a, const void* b){
	uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b;
	return (x > y) - (x < y);
}

#endif // HEAP_H
