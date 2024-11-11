
CUDA_PATH := /opt/cuda_9.1
CUDA_LIBS := -L$(CUDA_PATH)/lib64
CUDA_ARTLIB := -lcudart
CUDA_SM := sm_30

MATH_ARMALIB := -larmadillo

CXX = /opt/clang-10/bin/clang++
CXXFLAGS = -I. --cuda-path=$(CUDA_PATH) --cuda-gpu-arch=$(CUDA_SM)
LDFLAGS = -L. $(CUDA_LIBS) $(CUDA_ARTLIB) $(MATH_ARMALIB)

TARGET := app

all: matrix cuda_matrix_increment cuda_matrix_transposition cuda_matrix_multiply tools
	$(CXX) matrix.o cuda_matrix_increment.o cuda_matrix_transposition.o cuda_matrix_multiply.o tools.o -o $(TARGET) $(LDFLAGS)

tools:
	$(CXX) tools.cc -c -o tools.o

cuda_matrix_increment:
	$(CXX) increment.cu -c -o cuda_matrix_increment.o $(CXXFLAGS)

cuda_matrix_transposition:
	$(CXX) transposition.cu -c -o cuda_matrix_transposition.o $(CXXFLAGS)

cuda_matrix_multiply:
	$(CXX) multiply.cu -c -o cuda_matrix_multiply.o $(CXXFLAGS)

matrix:
	$(CXX) matrix.cc -c -o matrix.o

clean:
	rm -rf *.o
	rm -rf app
	
