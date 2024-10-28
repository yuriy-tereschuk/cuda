#/*
# * Debian Bookworm
# * CUDA SDK 9.1
# * LLVM Clang 10
# * CUDA Compute sm_30
# * 
# */

CUDA_PATH := /opt/cuda_9.1
CUDA_LIB := /opt/cuda_9.1/lib64
CUDA_GPU_ARCH := sm_30
CUDA_LINK := cudart_static

CXX=/opt/clang-10/bin/clang++ 
CXXFLAGS=-I./inc --cuda-gpu-arch=$(CUDA_GPU_ARCH) --cuda-path=$(CUDA_PATH)
LDFLAGS=-L$(CUDA_LIB) -l$(CUDA_LINK)

TARGET=addvector

all: $(TARGET)

$(TARGET): $(TARGET).cu
	$(CXX) $(TARGET).cu -o $(TARGET) $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -rf $(TARGET)
