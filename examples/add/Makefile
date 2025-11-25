# Simple HipKittens vector add example. Set OFFLOAD_ARCHS to your GPU's gfx string.

THUNDERKITTENS_ROOT ?= $(abspath ../..)
ROCM_INSTALL_DIR ?= $(ROCM_PATH)
HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

TARGET := hipkittens_add
SRC := main.cpp

CXX_STD ?= c++20
CXXFLAGS ?= -O3 -ffast-math -Wall -Wextra -Wno-unused-parameter

GPU_TARGET ?= CUSTOM

HIPFLAGS := -std=$(CXX_STD) $(CXXFLAGS)
HIPFLAGS += -I$(THUNDERKITTENS_ROOT)/include -I$(ROCM_INSTALL_DIR)/include/hip
HIPFLAGS += -DHIP_ENABLE_WARP_SYNC_BUILTINS

ifeq ($(GPU_TARGET),CDNA2)
HIPFLAGS += -DKITTENS_CDNA2
else ifeq ($(GPU_TARGET),CDNA3)
HIPFLAGS += -DKITTENS_CDNA3
else ifeq ($(GPU_TARGET),CDNA4)
HIPFLAGS += -DKITTENS_CDNA4
endif

ifeq ($(strip $(OFFLOAD_ARCHS)),)
  ifneq ($(strip $(HIP_ARCHS)),)
    OFFLOAD_ARCHS := $(HIP_ARCHS)
  else ifneq ($(strip $(HIP_ARCHITECTURES)),)
    OFFLOAD_ARCHS := $(HIP_ARCHITECTURES)
  else
    OFFLOAD_ARCHS := gfx950
  endif
endif

HIPFLAGS += $(foreach arch,$(OFFLOAD_ARCHS),--offload-arch=$(arch))

ifeq ($(strip $(KITTENS_WARP_THREADS)),)
W32_ARCHS := $(filter gfx10% gfx11% gfx12%,$(OFFLOAD_ARCHS))
W64_ARCHS := $(filter-out gfx10% gfx11% gfx12%,$(OFFLOAD_ARCHS))
ifneq ($(strip $(W32_ARCHS)),)
  ifneq ($(strip $(W64_ARCHS)),)
    $(error OFFLOAD_ARCHS mixes architectures with different native wave sizes. Split the build or set KITTENS_WARP_THREADS explicitly.)
  endif
  HIPFLAGS += -DKITTENS_WARP_THREADS=32
else
  HIPFLAGS += -DKITTENS_WARP_THREADS=64
endif
else
HIPFLAGS += -DKITTENS_WARP_THREADS=$(KITTENS_WARP_THREADS)
endif

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCXX) $< $(HIPFLAGS) -o $@

clean:
	rm -f $(TARGET)
