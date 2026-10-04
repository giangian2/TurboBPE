#ifndef GIAN_ARENA_H
#define GIAN_ARENA_H

#include <stddef.h>
#include <stdint.h>

#define DEFAULT_ARENA_SIZE 64000

#ifndef ARENA_SIZE
#define ARENA_SIZE DEFAULT_ARENA_SIZE
#endif

// Thread safety is disabled by default (0)
#ifndef ARENA_THREAD_SAFE
#define ARENA_THREAD_SAFE 0
#endif

#if ARENA_THREAD_SAFE
#include <pthread.h>
#endif

/**
 * Every pointer returned by arena_alloc is a multiple of ARENA_ALIGN. Must be a power of 2.
 *
 * Why: every type T may only live at an address that is a multiple of _Alignof(T)
 * (token_t: 2, Span and uint32_t: 4, uint64_t and pointers: 8). Inside a struct the compiler
 * inserts the padding by itself, because it sees all the fields. Between two arena_alloc calls
 * nobody does: the arena only receives a size, not a type, so without rounding the next block
 * starts exactly where the previous one ended. After a 5 byte string, a token_t* would point to
 * an odd address: undefined behaviour in C, slower on x86, SIGBUS on strict CPUs, and a crash
 * with SIMD instructions the compiler may emit at -O2.
 *
 * 8 covers every scalar type. The first block is aligned too: malloc returns 16 byte aligned
 * memory and sizeof(Arena) is a multiple of 8 (checked below), so base = arena + 1 is aligned.
 */
#define ARENA_ALIGN 8

typedef struct {
    size_t capacity;
    size_t size;
    #if ARENA_THREAD_SAFE
    pthread_mutex_t lock;
    #endif
} Arena;

_Static_assert((ARENA_ALIGN & (ARENA_ALIGN - 1)) == 0, "ARENA_ALIGN must be a power of 2");
_Static_assert(sizeof(Arena) % ARENA_ALIGN == 0, "the data after the Arena header must start aligned");

// Prototypes
Arena* arena_create(size_t capacity);
void arena_free(Arena* arena);
uint8_t* arena_alloc(Arena* arena, size_t size);
void arena_reset(Arena* arena);


/**
 * The implementation is compiled only where GIAN_ARENA_IMPL is defined before the include, in
 * exactly one .c file. Every other file sees only the prototypes, otherwise the linker would find
 * the same functions defined twice.
 */
#ifdef GIAN_ARENA_IMPL

#include <stdlib.h>
#include <stdio.h>

Arena* arena_create(size_t capacity) {
    Arena* arena = (Arena*)malloc(sizeof(Arena) + capacity);
    if (arena == NULL) return NULL;
    
    arena->capacity = capacity;
    arena->size     = 0;

    #if ARENA_THREAD_SAFE
    if (pthread_mutex_init(&arena->lock, NULL) != 0) {
        free(arena);
        return NULL;
    }
    #endif

    return arena;
}

void arena_free(Arena* arena) {
    if (arena == NULL) return;

    #if ARENA_THREAD_SAFE
    pthread_mutex_destroy(&arena->lock);
    #endif

    free(arena);
}

uint8_t* arena_alloc(Arena* arena, size_t size) {
    if (arena == NULL) return NULL;

    #if ARENA_THREAD_SAFE
    pthread_mutex_lock(&arena->lock);
    #endif

    // round the offset up to the next multiple of ARENA_ALIGN: adding ALIGN-1 reaches (or passes) it,
    // clearing the low bits truncates back to it. E.g. with 8: 5 -> 12 -> 8, 8 -> 15 -> 8.
    size_t offset = (arena->size + (ARENA_ALIGN - 1)) & ~(size_t)(ARENA_ALIGN - 1);

    if (offset > arena->capacity || size > arena->capacity - offset) {   // no overflow on huge sizes
        #if ARENA_THREAD_SAFE
        pthread_mutex_unlock(&arena->lock);
        #endif
        return NULL; // Out of memory within the arena
    }

    uint8_t *base = (uint8_t*)(arena + 1);
    uint8_t *ptr = base + offset;
    arena->size = offset + size;

    #if ARENA_THREAD_SAFE
    pthread_mutex_unlock(&arena->lock);
    #endif

    return ptr;
}

void arena_reset(Arena* arena) {
    if (arena == NULL) return;

    #if ARENA_THREAD_SAFE
    pthread_mutex_lock(&arena->lock);
    #endif

    arena->size = 0;

    #if ARENA_THREAD_SAFE
    pthread_mutex_unlock(&arena->lock);
    #endif
}

#endif // GIAN_ARENA_IMPL
#endif // GIAN_ARENA_H