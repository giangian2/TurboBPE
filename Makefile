CC      = gcc

# Optimization level, overridable from the command line:
#   make            ordinary build, -O0, easy to step through
#   make OPT=-O2    optimized build, required for profiling and measuring
# Changing OPT invalidates the objects already compiled (see $(OPTSTAMP)
# below), so build/ can never end up holding a mix of the two.
OPT     ?= -O0

CFLAGS  = -std=c11 -Wall -Wextra -g -Iinclude $(OPT)

# library sources (main.c is NOT part of the library)
SRC = src/bpe.c
HDR = $(wildcard include/*.h)
OBJ = $(SRC:src/%.c=build/%.o)
LIB = bin/libbpe.a

# Witness of the optimization level build/ was produced with: the file name
# embeds it, so switching means it does not exist, the rule fires and throws
# the stale objects away before recompiling.
OPTSTAMP = build/.opt$(subst -,,$(OPT))

all: $(LIB) bin/main

# the demo program, linked against the library
bin/main: src/main.c $(LIB) | bin
	$(CC) $(CFLAGS) $< -Lbin -lbpe $(LDLIBS) -o $@

$(LIB): $(OBJ) | bin
	ar rcs $@ $^

build/%.o: src/%.c $(HDR) $(OPTSTAMP) | build
	$(CC) $(CFLAGS) -c $< -o $@

$(OPTSTAMP): | build
	rm -f build/*.o build/.opt* $(LIB)
	touch $@

# arguments passed to bin/main by run/debug/memcheck, e.g. make run ARGS="live out/model.bpe"
ARGS ?= train test.txt out/model.bpe

run: bin/main | out
	./bin/main $(ARGS)

debug: bin/main
	gdb -x gdb/init.gdb --args bin/main $(ARGS)

# output directories, created on demand
bin build out:
	mkdir -p $@

memcheck: bin/main | out
	valgrind --leak-check=full --track-origins=yes bin/main $(ARGS)

clean:
	rm -rf build bin out

.PHONY: all run debug clean memcheck
