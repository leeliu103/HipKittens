# NOTE: WE HIGHLY RECOMMEND RUNNING WITH `make -j32` or however many threads your machine has.
# If you want to run lots of tests without parallel compilation, be prepared
# to spend a *long* time compiling. (like, HOURS single-threaded.)

# GPU Selection: lock these tests to RDNA4 (gfx1201 / wave32)
GPU_TARGET=RDNA4
KITTENS_WARP_THREADS=32
COMP_LEVEL=profile

GPU_RUNTIME := HIP

# HIP variables
ROCM_INSTALL_DIR := $(ROCM_PATH)
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include/hip

HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc
HIPFLAGS=-std=c++20 -I$(HIP_INCLUDE_DIR) -I../unit/testing_commons -I../../include -Wall -Wextra -Wl,--allow-multiple-definition

# At what level to run tests?
# Level 1 takes 15 seconds to compile 1,000 tests.
# (Default) Level 2 takes a minute to compile 3,000 tests.
# Level 3 takes 5 minutes to compile 10,000 tests.
# Level 4 takes 15 while to compile 25,000 tests
HIPFLAGS+= -DTEST_INTENSITY=2
# Which tests to run?
# -DTEST_ALL means run all tests.
# You can also specify subsections, e.g. -DTEST_WARP_MEMORY
# Or individual tests, like -DTEST_WARP_MEMORY_VEC_DSMEM. Useful for debugging!
# Enable all warp test coverage by default (override via env)
TEST_DEFINES ?= -DTEST_ALL_WARP
HIPFLAGS+= $(TEST_DEFINES)

ifeq ($(COMP_LEVEL),safe)
HIPFLAGS+= -O0
else ifeq ($(COMP_LEVEL),debug)
HIPFLAGS+= -g
else ifeq ($(COMP_LEVEL),profile)
HIPFLAGS+= -O3 -save-temps=obj
endif

# Compiler flags based on GPU target
ifeq ($(GPU_TARGET),CDNA2)
HIPFLAGS+= -DKITTENS_CDNA2 --offload-arch=gfx90a
else ifeq ($(GPU_TARGET),CDNA3)
HIPFLAGS+= -DKITTENS_CDNA3 --offload-arch=gfx942
else ifeq ($(GPU_TARGET),CDNA4)
HIPFLAGS+= --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS
else ifeq ($(GPU_TARGET),RDNA4)
HIPFLAGS+= --offload-arch=gfx1201 -DHIP_ENABLE_WARP_SYNC_BUILTINS -DKITTENS_RDNA4
ifeq ($(strip $(KITTENS_WARP_THREADS)),)
KITTENS_WARP_THREADS := 32
endif
endif
# -DKITTENS_CDNA4 

ifneq ($(strip $(KITTENS_WARP_THREADS)),)
HIPFLAGS+= -DKITTENS_WARP_THREADS=$(strip $(KITTENS_WARP_THREADS))
endif

# Suppress warnings
HIPFLAGS+= -w
HIPFLAGS+= -Wno-pass-failed

# Target binary name
TARGET=unit_tests
BUILD_DIR=build

.PHONY: all build run clean

# Test source files (unit driver + warp + shared commons)
SKIP_TESTS ?=
TESTS_SRC := unit_tests.cu \
             $(shell find ../unit/testing_commons -name '*.cu') \
             $(shell find warp -name '*.cu')
TESTS_SRC := $(filter-out $(SKIP_TESTS), $(TESTS_SRC))

# Object files
OBJS=$(patsubst %.cu,$(BUILD_DIR)/%.o,$(TESTS_SRC))

# Default target
all: build $(TARGET)

# Create the build directory
build:
	mkdir -p $(BUILD_DIR)

# Rule to compile each CU file
$(BUILD_DIR)/%.o: %.cu
	mkdir -p $(@D)
	$(HIPCXX) $(HIPFLAGS) -c $< -o $@

# Link object files to create the final executable
$(TARGET): $(OBJS)
	$(HIPCXX) $(HIPFLAGS) $^ -o $(TARGET)

# Run target
run: all
	./$(TARGET)

# Clean target
clean:
	rm -rf $(BUILD_DIR) $(TARGET) unit $(TARGET).dSYM
	rm -f *.hipref *.linkinfo *.bc *.ll *.hsaco
