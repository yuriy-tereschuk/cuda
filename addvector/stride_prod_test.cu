#include <iostream>
#include <cstdlib>
//#include <helper_cuda.h>
#include <cuda_runtime.h>
#include "tests.h"
#include "tools.h"

#define THREADS 512

using namespace std;

__global__
void stride_prod(const int* a, const int* b, int* prod, int elements)
{
  __shared__ int temp[THREADS];

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  temp[threadIdx.x] = a[idx] + b[idx];

  __syncthreads();

  int sum = 0;
  for (int i = 0; i < elements; i += gridDim.x * blockDim.x)
  {
    sum += temp[i];
  }
  atomicAdd(prod, sum);
}

void stride_prod_test()
{

  cudaError_t err = cudaSuccess;

  int *h_a, *h_b, *h_prod;
  int *d_a, *d_b, *d_prod;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int elements = 5 << 10;

  h_a = new int[elements];
  h_b = new int[elements];
  h_prod = new int;

  init(h_a, elements);
  init(h_b, elements);
  *h_prod = 0;

  std::cout << "Host prod: " << host_prod(h_a, h_b, elements) << std::endl;

  cudaMalloc((void**) &d_a, sizeof(int) * elements);
  cudaMalloc((void**) &d_b, sizeof(int) * elements);
  cudaMalloc((void**) &d_prod, sizeof(int));

  *h_prod = 0;

  err = cudaMemcpy(d_a, h_a, sizeof(int) * elements, cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
  {
    std::cout << "Error: " << err << std::endl;
    return;
  }

  cudaMemcpy(d_b, h_b, sizeof(int) * elements, cudaMemcpyHostToDevice);
  cudaMemcpy(d_prod, h_prod, sizeof(int), cudaMemcpyHostToDevice);

  int blocks = (elements + THREADS - 1) / THREADS;

  cudaEventRecord(start);
  stride_prod<<<blocks, THREADS>>>(d_a, d_b, d_prod, elements);
  cudaEventRecord(stop);
  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cout << "Cuda last error: " << err << std::endl;
    return;
  }

  err = cudaMemcpy(h_prod, d_prod, sizeof(int), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
  {
    std::cout << "Error prod: " << err << std::endl;
    return;
  }

  std::cout << "Device prod: " << *h_prod << std::endl;

  err = cudaEventSynchronize(stop);
  if (err != cudaSuccess)
  {
    std::cout << "Cuda synchronize error: " << err << std::endl;
    return;
  }
  
  float computationTime = 0;
  cudaEventElapsedTime(&computationTime, start, stop);

  std::cout << "CUDA Computation time: " << computationTime << std::endl;


  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_a);
  cudaFree(d_b);
  delete(h_a);
  delete(h_b);
}
