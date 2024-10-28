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
CXXFLAGS=--cuda-path=$(CUDA_PATH) --cuda-gpu-arch=$(CUDA_GPU_ARCH) 
LDFLAGS=-L$(CUDA_LIB) -l$(CUDA_LINK)

TARGET=addvector

all: $(TARGET)

thread_sync_prod: thread_sync_prod_test.cu
	$(CXX) thread_sync_prod_test.cu -c -o thread_sync_prod_test.o $(CXXFLAGS)

stride_prod: stride_prod_test.cu
	$(CXX) stride_prod_test.cu -c -o stride_prod_test.o $(CXXFLAGS)

tools: tools.cc 
	$(CXX) tools.cc -c -o tools.o

main: main.cc 
	$(CXX) main.cc -c -o main.o

$(TARGET): main tools thread_sync_prod stride_prod
	$(CXX) main.o tools.o thread_sync_prod_test.o stride_prod_test.o -o $(TARGET) $(LDFLAGS)

clean:
	rm -rf $(TARGET)
	rm -rf main.o tools.o thread_sync_prod_test.o stride_prod_test.o

