
CUDA_PATH := /opt/cuda_9.1
CUDA_LIBS := -L$(CUDA_PATH)/lib64
CUDA_ARTLIB := -lcudart
CUDA_SM := sm_30

MATH_ARMALIB := -larmadillo

CXX = /opt/clang-10/bin/clang++
CXXFLAGS = -I. --cuda-path=$(CUDA_PATH) --cuda-gpu-arch=$(CUDA_SM)
LDFLAGS = -L. $(CUDA_LIBS) $(CUDA_ARTLIB) $(MATH_ARMALIB)

TARGET := app

all: matrix cuda_matrix tools
	$(CXX) matrix.o cuda_matrix.o tools.o -o $(TARGET) $(LDFLAGS)

tools:
	$(CXX) tools.cc -c -o tools.o

cuda_matrix:
	$(CXX) matrix.cu -c -o cuda_matrix.o $(CXXFLAGS)

matrix: tools cuda_matrix
	$(CXX) matrix.cc -c -o matrix.o

clean:
	rm -rf *.o
	rm -rf app
	
