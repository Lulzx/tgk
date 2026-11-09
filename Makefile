CC      = cc
CFLAGS  = -std=c99 -O3 -mcpu=native -Wall -Wextra -Wno-unused-parameter -pthread
LDFLAGS = -lm -pthread

SRCS    = main.c geom.c grid.c fft.c field.c rhs.c coll.c imex.c diag.c
LIB_SRCS= geom.c grid.c fft.c field.c rhs.c coll.c imex.c diag.c

.PHONY: all clean check bench

all: tgk tgk_fp64

tgk: $(SRCS) tgk.h simd.h
	$(CC) $(CFLAGS) $(SRCS) -o tgk $(LDFLAGS)

tgk_fp64: $(SRCS) tgk.h simd.h
	$(CC) $(CFLAGS) -DTGK_USE_FP64 $(SRCS) -o tgk_fp64 $(LDFLAGS)

tests/test_suite: tests/test_suite.c $(SRCS) tgk.h simd.h
	$(CC) $(CFLAGS) -DTGK_LIBRARY_ONLY tests/test_suite.c $(SRCS) -o tests/test_suite $(LDFLAGS)

tests/test_suite_fp64: tests/test_suite.c $(SRCS) tgk.h simd.h
	$(CC) $(CFLAGS) -DTGK_USE_FP64 -DTGK_LIBRARY_ONLY tests/test_suite.c $(SRCS) -o tests/test_suite_fp64 $(LDFLAGS)

check: all tests/test_suite tests/test_suite_fp64
	@echo "=== Running Validation & Verification Suite (Mixed Precision FP32/FP64) ==="
	./tests/test_suite
	@echo "=== Running Validation & Verification Suite (Full FP64) ==="
	./tests/test_suite_fp64
	@echo "=== Comparing FP32-mixed vs Full FP64 Agreement (< 0.5% threshold) ==="
	./tests/test_suite --compare-fp64

clean:
	rm -f tgk tgk_fp64 tests/test_suite tests/test_suite_fp64 *.trace.txt *.bin tests/*.bin tests/*.txt
