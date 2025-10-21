GPU_TARGET=CDNA4
COMP_LEVEL=profile

GPU_RUNTIME := HIP

ROCM_INSTALL_DIR := $(ROCM_PATH)
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include/hip

HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc
HIPFLAGS=-std=c++20 -I$(HIP_INCLUDE_DIR) -I../../../include -Wall -Wextra -Wl,--allow-multiple-definition -DKITTENS_CDNA4 --offload-arch=gfx950 -fopenmp

ifeq ($(COMP_LEVEL),safe)
HIPFLAGS+= -O0
else ifeq ($(COMP_LEVEL),debug)
HIPFLAGS+= -g -O0
else ifeq ($(COMP_LEVEL),profile)
HIPFLAGS+= -O3 -save-temps=obj -Rpass-analysis=kernel-resource-usage
endif

HIPFLAGS+= -DFP8_USE_16x128

CXXFLAGS ?= -Wall -Wextra
CXXFLAGS := -w

ICXXFLAGS += $(CXXFLAGS) 

TARGET=matmul
SRC=matmul.cu
BUILD_DIR=build

$(TARGET): $(SRC)
	mkdir -p $(BUILD_DIR)
	$(HIPCXX) $(HIPFLAGS) $(ICXXFLAGS) -c $(SRC) -o $(BUILD_DIR)/matmul.o
	$(HIPCXX) $(HIPFLAGS) $(BUILD_DIR)/matmul.o -o $(TARGET)

LOAD=load
LOAD_SRC=load.cu

$(LOAD): $(LOAD_SRC)
	mkdir -p $(BUILD_DIR)
	$(HIPCXX) $(HIPFLAGS) -c $(LOAD_SRC) -o $(BUILD_DIR)/load.o
	$(HIPCXX) $(HIPFLAGS) $(BUILD_DIR)/load.o -o $(LOAD)

TARGET_2048=matmul_2048
SRC_2048=matmul_2048.cu

$(TARGET_2048): $(SRC_2048)
	mkdir -p $(BUILD_DIR)
	$(HIPCXX) $(HIPFLAGS) $(ICXXFLAGS) -c $(SRC_2048) -o $(BUILD_DIR)/matmul_2048.o
	$(HIPCXX) $(HIPFLAGS) $(BUILD_DIR)/matmul_2048.o -o $(TARGET_2048)

TARGET_1024=matmul_1024
SRC_1024=matmul_1024.cu

$(TARGET_1024): $(SRC_1024)
	mkdir -p $(BUILD_DIR)
	$(HIPCXX) $(HIPFLAGS) $(ICXXFLAGS) -c $(SRC_1024) -o $(BUILD_DIR)/matmul_1024.o
	$(HIPCXX) $(HIPFLAGS) $(BUILD_DIR)/matmul_1024.o -o $(TARGET_1024)

clean:
	rm -rf $(BUILD_DIR) $(TARGET) $(LOAD) $(TARGET_2048) $(TARGET_1024)
