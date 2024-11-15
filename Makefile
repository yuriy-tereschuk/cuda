CUDA_LIBS:=/opt/cuda_9.1/lib64
CUDA_INCLUDE:=/opt/cuda_9.1/include
CUDA_GPU_ARCH:=--cuda-gpu-arch=sm_30

CC=/opt/clang-10/bin/clang++
CFLAGS=-I$(CUDA_INCLUDE) 
LDFLAGS=-L$(CUDA_LIBS) -lcudart_static

TARGET=app

$(TARGET): tools kernel main
	$(CC) tools.o kernel.o main.o -o $(TARGET) $(LDFLAGS)

main:
	$(CC) main.cpp -c -o main.o

tools:
	$(CC) tools.cpp -c -o tools.o

kernel:
	$(CC) kernel.cu -c -o kernel.o $(CFLAGS) $(CUDA_GPU_ARCH)

clean:
	rm -rf $(TARGET)
	rm -rf *.o

