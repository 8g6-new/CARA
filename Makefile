# Compiler Settings
CC       = gcc
# beyond O2 its found that NAN == NAN (UB)

#_______________________General optimization flags__________________________________
OPTFLAGS+=-O3#                      => Enables aggressive optimizations beyond -O2, including unsafe floating-point optimizations
OPTFLAGS+=-march=native#            => Generates code optimized for the host CPU
OPTFLAGS+=-mtune=native#            => Optimizes code scheduling for the host CPU
OPTFLAGS+=-funroll-loops#           => Unrolls loops to reduce branch overhead
OPTFLAGS+=-fpeel-loops#             => Extracts loop iterations that always execute to optimize performance
OPTFLAGS+=-ftracer#                 => Improves branch prediction and inlining
OPTFLAGS+=-flto#                    => Enables Link-Time Optimization (LTO) for cross-file optimization
OPTFLAGS+=-fuse-linker-plugin#      => Uses a linker plugin for better LTO performance
OPTFLAGS+=-MMD -MP#                 => Generates dependency files for make without including system headers
OPTFLAGS+=-floop-block#             => Optimizes loop memory access patterns for better cache performance
OPTFLAGS+=-floop-interchange#       => Reorders nested loops for better vectorization and locality
OPTFLAGS+=-floop-unroll-and-jam#    => Unrolls outer loops and fuses iterations for better performance
OPTFLAGS+=-fipa-pta#                => Enables interprocedural pointer analysis for optimization
OPTFLAGS+=-fipa-cp#                 => Performs constant propagation across functions
OPTFLAGS+=-fipa-sra#                => Optimizes function arguments and return values for efficiency
OPTFLAGS+=-fipa-icf#                => Merges identical functions to reduce code size
OPTFLAGS+=-fno-unsafe-math-optimizations


#_______________________Says compiler to vectorize loops_________________________________________
VECFLAGS+=-ftree-vectorize#         => Enables automatic vectorization of loops
VECFLAGS+=-ftree-loop-vectorize#    => Enables loop-based vectorization
VECFLAGS+=-fopt-info-vec-optimized# => Outputs details of vectorized loops
# VECFLAGS+=-fopt-info-vec-all#     => Shows ALL vectorization attempts

#_______________________Debugging and safety flags__________________________________
DBGFLAGS+=-Og#                      => Optimizations suitable for debugging
DBGFLAGS+=-fno-omit-frame-pointer#  => Keeps the frame pointer for debugging
DBGFLAGS+=-fno-inline#              => Disables function inlining for better debugging
DBGFLAGS+=-fstack-protector-strong# => Adds stack protection to detect buffer overflows
DBGFLAGS+=-g#                       => Generates debugging information
DBGFLAGS+=-fsanitize=address#       => Enables AddressSanitizer for memory error detection
DBGFLAGS+=-fsanitize=leak#          => Enables leak detection
DBGFLAGS+=-fsanitize=undefined#     => Enables Undefined Behavior Sanitizer (UBSan)

LIBFLAGS  = -DMINIMP3_FLOAT_OUTPUT 
WARNFLAGS = -Wall -Wextra 
LOGFLAGS  = -DLOG_LEVEL=1

INC_DIR   = headers

CFLAGS = -std=c11 $(WARNFLAGS) $(OPTFLAGS) $(VECFLAGS) $(LIBFLAGS) $(LOGFLAGS) -fopenmp -I$(INC_DIR) -I/usr/include/openblas
CFLAGS_DEBUG = $(WARNFLAGS) $(DBGFLAGS) $(LIBFLAGS) $(LOGFLAGS) -fopenmp -I$(INC_DIR) -I/usr/include/openblas -std=c11
LDFLAGS = -lm -lfftw3 -lfftw3f -lsndfile -lpng -fopenmp -lopenblas

# Directory Structure
SRCDIR     = src
SCHEMEDIR  = $(SRCDIR)/libheatmap/colorschemes
BUILDDIR   = build

# Source Files
BASE_SOURCES = main.c \
               $(wildcard $(SRCDIR)/libheatmap/*.c) \
               $(wildcard $(SRCDIR)/png_tools/*.c) \
               $(wildcard $(SRCDIR)/utils/*.c) \
               $(wildcard $(SRCDIR)/audio_tools/*.c)

# Test programs
TEST_ONSET = tests/test_onset.c \
             $(wildcard $(SRCDIR)/libheatmap/*.c) \
             $(wildcard $(SRCDIR)/png_tools/*.c) \
             $(wildcard $(SRCDIR)/utils/*.c) \
             $(wildcard $(SRCDIR)/audio_tools/*.c)

TEST_TEMPO = tests/test_tempo.c \
             $(wildcard $(SRCDIR)/libheatmap/*.c) \
             $(wildcard $(SRCDIR)/png_tools/*.c) \
             $(wildcard $(SRCDIR)/utils/*.c) \
             $(wildcard $(SRCDIR)/audio_tools/*.c)

TEST_BEAT_TRACK = tests/test_beat_track.c \
                  $(wildcard $(SRCDIR)/libheatmap/*.c) \
                  $(wildcard $(SRCDIR)/png_tools/*.c) \
                  $(wildcard $(SRCDIR)/utils/*.c) \
                  $(wildcard $(SRCDIR)/audio_tools/*.c)

# Color Scheme Sources
BUILTIN_DIR    = $(SCHEMEDIR)/builtin
OPENCV_DIR     = $(SCHEMEDIR)/opencv_like
BUILTIN_SOURCES = $(wildcard $(BUILTIN_DIR)/*.c)
OPENCV_SOURCES  = $(wildcard $(OPENCV_DIR)/*.c)

# Object Files mapped to build/
BASE_OBJECTS_BUILTIN = $(patsubst %.c,$(BUILDDIR)/builtin/%.o,$(BASE_SOURCES))
BASE_OBJECTS_OPENCV = $(patsubst %.c,$(BUILDDIR)/opencv/%.o,$(BASE_SOURCES))
BUILTIN_OBJECTS = $(patsubst %.c,$(BUILDDIR)/builtin/%.o,$(BUILTIN_SOURCES))
OPENCV_OBJECTS = $(patsubst %.c,$(BUILDDIR)/opencv/%.o,$(OPENCV_SOURCES))

# Track Last Built Target
LAST_TARGET_FILE = .last_target
ifneq ($(wildcard $(LAST_TARGET_FILE)),)
    LAST_TARGET := $(shell cat $(LAST_TARGET_FILE))
else
    LAST_TARGET := builtin
endif

# Default Target
.PHONY: all clean debug debug_opencv_like debug_builtin test opencv_like builtin run prep_dirs test_all install test_onset


install:
	@if [ -f ./install_libs.sh ]; then \
		echo "Running library installation..."; \
		chmod +x ./install_libs.sh; \
		./install_libs.sh; \
	else \
		echo "Error: install_libs.sh not found in $(shell pwd)!"; \
		exit 1; \
	fi


all: $(LAST_TARGET)

# Create all needed directories
prep_dirs:
	@mkdir -p $(BUILDDIR)/builtin
	@mkdir -p $(BUILDDIR)/opencv
	@mkdir -p $(dir $(BASE_OBJECTS_BUILTIN) $(BASE_OBJECTS_OPENCV) $(BUILTIN_OBJECTS) $(OPENCV_OBJECTS))

# OpenCV Color Scheme Build
opencv_like: 
	@$(MAKE) prep_dirs
	@$(MAKE) $(BASE_OBJECTS_OPENCV) $(OPENCV_OBJECTS)
	$(CC) $(CFLAGS) -DOPENCV_LIKE -o $@ $(BASE_OBJECTS_OPENCV) $(OPENCV_OBJECTS) $(LDFLAGS)
	@echo "opencv_like" > $(LAST_TARGET_FILE)
	@echo "Built with OpenCV-like color scheme"

# Builtin Color Scheme Build
builtin: 
	@$(MAKE) prep_dirs
	@$(MAKE) $(BASE_OBJECTS_BUILTIN) $(BUILTIN_OBJECTS)
	$(CC) $(CFLAGS) -DBUILTIN -o $@ $(BASE_OBJECTS_BUILTIN) $(BUILTIN_OBJECTS) $(LDFLAGS)
	@echo "builtin" > $(LAST_TARGET_FILE)
	@echo "Built with Builtin color scheme"

# Debug Build with OpenCV-like Color Scheme
debug_opencv_like: clean
	@$(MAKE) prep_dirs
	@$(MAKE) CFLAGS="$(CFLAGS_DEBUG) -DOPENCV_LIKE" $(BASE_OBJECTS_OPENCV) $(OPENCV_OBJECTS)
	$(CC) $(CFLAGS_DEBUG) -DOPENCV_LIKE -o opencv_like $(BASE_OBJECTS_OPENCV) $(OPENCV_OBJECTS) $(LDFLAGS)
	@echo "opencv_like" > $(LAST_TARGET_FILE)
	@echo "Debug build with OpenCV-like color scheme"

# Debug Build with Builtin Color Scheme
debug_builtin: clean
	@$(MAKE) prep_dirs
	@$(MAKE) CFLAGS="$(CFLAGS_DEBUG) -DBUILTIN" $(BASE_OBJECTS_BUILTIN) $(BUILTIN_OBJECTS)
	$(CC) $(CFLAGS_DEBUG) -DBUILTIN -o builtin $(BASE_OBJECTS_BUILTIN) $(BUILTIN_OBJECTS) $(LDFLAGS)
	@echo "builtin" > $(LAST_TARGET_FILE)
	@echo "Debug build with Builtin color scheme"

# Compilation Rules
$(BUILDDIR)/builtin/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -DBUILTIN -MMD -MP -c $< -o $@

$(BUILDDIR)/opencv/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -DOPENCV_LIKE -MMD -MP -c $< -o $@

# Include generated dependency files
-include $(BASE_OBJECTS_BUILTIN:.o=.d)
-include $(BASE_OBJECTS_OPENCV:.o=.d)
-include $(BUILTIN_OBJECTS:.o=.d)
-include $(OPENCV_OBJECTS:.o=.d)

# Debug Build
debug: debug_$(LAST_TARGET)
	@echo "Built $(LAST_TARGET) in Debug Mode"
# Run Last Target
run:
	@if [ ! -f "$(LAST_TARGET_FILE)" ]; then \
	  echo "No previous build found. Run 'make' first."; exit 1; \
	fi; \
	LAST_TARGET=$$(cat $(LAST_TARGET_FILE)); \
	if [ ! -x "$$LAST_TARGET" ]; then \
	  echo "Executable '$$LAST_TARGET' not found. Run 'make' first."; exit 1; \
	fi; \
	mkdir -p ./cache/FFT; \
	for f in ./tests/files/*.wav ./tests/files/*.mp3; do \
	  [ -f "$$f" ] || continue; \
	  BASENAME=$$(basename "$$f" | sed 's/\.[^.]*$$//'); \
	  echo "Running $$LAST_TARGET on $$f → outputs/$$BASENAME ..."; \
	  ./$$LAST_TARGET \
	    -i "$$f" \
	    -o "$$BASENAME" \
	    -ws 2048 \
	    -hop 128 \
	    -wf "hann" \
	    -nm 256 \
	    -nfcc 64 \
	    -stft_cs 4 \
	    -fb_cs 6 \
	    -fcc_cs 17 \
	    -fb "mel" \
	    -c "./cache/FFT" \
	    -t 4; \
	done


# Test All Combinations
test_all:
	@if [ -z "$(N)" ]; then \
		echo "Please provide number of combinations: make test_random N=<number>"; exit 1; \
	fi; \
	if [ ! -f "$(LAST_TARGET_FILE)" ]; then \
		echo "No previous build found. Run 'make' first."; exit 1; \
	fi; \
	LAST_TARGET=$$(cat $(LAST_TARGET_FILE)); \
	if [ ! -x "$$LAST_TARGET" ]; then \
		echo "Executable '$$LAST_TARGET' not found. Run 'make' first."; exit 1; \
	fi; \
	mkdir -p ./cache/FFT; \
	mkdir -p ./tests/out; \
	echo "Running $$N random combinations with $$LAST_TARGET..."; \
	WS_LIST="512 1024 2048"; \
	HOP_LIST="64 128 256"; \
	WF_LIST="hann hamming blackman"; \
	FB_LIST="mel bark erb log10 chirp cam"; \
	NM_LIST="32 64 128"; \
	NFCC_LIST="12 24"; \
	MAX_THREADS=$$(nproc); \
	PASS=0; \
	FAIL=0; \
	rm -f run_errors.log; \
	for i in $$(seq 1 $(N)); do \
	    ws=$$(shuf -e $$WS_LIST -n1); \
	    hop=$$(shuf -e $$HOP_LIST -n1); \
	    wf=$$(shuf -e $$WF_LIST -n1); \
	    fb=$$(shuf -e $$FB_LIST -n1); \
	    nm=$$(shuf -e $$NM_LIST -n1); \
	    nfcc=$$(shuf -e $$NFCC_LIST -n1); \
	    th=$$(shuf -i 1-$$MAX_THREADS -n1); \
	    OUT="rand_out_ws$${ws}_hop$${hop}_wf$${wf}_fb$${fb}_nm$${nm}_nfcc$${nfcc}_t$${th}"; \
	    echo -n "Running $$OUT ... "; \
	    if ./$$LAST_TARGET \
	        -i "./tests/files/black_woodpecker.wav" \
	        -o "./tests/out/$$OUT" \
	        -ws $$ws \
	        -hop $$hop \
	        -wf $$wf \
	        -nm $$nm \
	        -nfcc $$nfcc \
	        -stft_cs 4 \
	        -fb_cs 6 \
	        -fcc_cs 17 \
	        -fb $$fb \
	        -c "./cache/FFT" \
	        -t $$th > /dev/null 2>> run_errors.log; then \
	        echo "✅ TEST PASSED"; \
	        PASS=$$((PASS+1)); \
	    else \
	        echo "❌ TEST FAILED"; \
	        FAIL=$$((FAIL+1)); \
	    fi; \
	done; \
	echo "----------------------------------------"; \
	echo "Summary: $$PASS / $$((PASS+FAIL)) tests passed, $$FAIL failed"; \
	if [ $$FAIL -gt 0 ]; then \
	    echo "Check 'run_errors.log' for details on failures."; \
	fi

# Test Onset Detection
test_onset: builtin
	@echo "Building onset detection test..."
	$(CC) $(CFLAGS) -DBUILTIN -o test_onset $(TEST_ONSET) $(LDFLAGS)
	@echo "Running onset detection test..."
	@mkdir -p outputs cache/FFT tests/out
	./test_onset

# Test Tempo Estimation
test_tempo: builtin
	@echo "Building tempo estimation test..."
	$(CC) $(CFLAGS) -DBUILTIN -o test_tempo $(TEST_TEMPO) $(LDFLAGS)
	@echo "Running tempo estimation test..."
	./test_tempo

# Test Beat Tracking
test_beat_track: builtin
	@echo "Building beat tracking test..."
	$(CC) $(CFLAGS) -DBUILTIN -o test_beat_track $(TEST_BEAT_TRACK) $(LDFLAGS)
	@echo "Running beat tracking test..."
	./test_beat_track

# Clean Build
clean:
	rm -rf $(BUILDDIR)
	rm -f builtin opencv_like main test_onset test_tempo test_beat_track $(LAST_TARGET_FILE)
