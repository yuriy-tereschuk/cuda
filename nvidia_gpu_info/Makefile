CUDA_ARCH:=sm_32
CUDA_PATH:=/opt/cuda_9.1
CXX:=/opt/clang-10/bin/clang++

CXXFLAGS=-I$(CUDA_PATH)/include
LDFLAGS=-L$(CUDA_PATH)/lib64/stubs -lnvidia-ml

TARGET=app

$(TARGET): main tools nvidia
	$(CXX) main.o tools.o nvidia.o -o $(TARGET) $(LDFLAGS)

main: main.o
	$(CXX) main.cc -c $(CXXFLAGS)

tools: tools.o
	$(CXX) tools.cc -c $(CXXFLAGS)

nvidia: nvidia.o
	$(CXX) nvidia.cc -c $(CXXFLAGS)

clean:
	-@rm -f *.o
	-@rm -f $(TARGET)
