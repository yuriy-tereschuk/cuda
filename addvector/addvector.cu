#include <iostream>
#include <cstdlib>
//#include <helper_cuda.h>
#include <cuda_runtime.h>

#define THREADS 512

using namespace std;

__global__ 
void thread_sync_prod(const int* a, const int* b, int* prod, int elements)
{
  __shared__ int temp[THREADS];

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  temp[threadIdx.x] = a[idx] + b[idx];

  __syncthreads();

  if (threadIdx.x == 0) {
    int sum = 0;
    for (int i = 0; i < THREADS; i++)
    {
      sum += temp[i];
    }
    atomicAdd(prod, sum);
  }
}

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


void init(int* vec, int elements)
{
  for(int i = 0; i < elements; i++)
  {
    *vec++ = 2;
  }
}

int host_prod(int* a, int* b, int elements)
{
  int prod = a[0] + b[0];
  for ( int i = 1; i < elements; i++)
  {
    prod += a[i] + b[i];
  }

  return prod;
}

int main(int argc, char** argv)
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
    return 2;
  }

  cudaMemcpy(d_b, h_b, sizeof(int) * elements, cudaMemcpyHostToDevice);
  cudaMemcpy(d_prod, h_prod, sizeof(int), cudaMemcpyHostToDevice);

  int blocks = elements / THREADS;

  cudaEventRecord(start);
  thread_sync_prod<<<blocks, THREADS>>>(d_a, d_b, d_prod, elements);
  cudaEventRecord(stop);
  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cout << "Cuda last error: " << err << std::endl;
    return 2;
  }

  err = cudaMemcpy(h_prod, d_prod, sizeof(int), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
  {
    std::cout << "Error prod: " << err << std::endl;
    return 2;
  }

  std::cout << "Device prod: " << *h_prod << std::endl;

  err = cudaEventSynchronize(stop);
  if (err != cudaSuccess)
  {
    std::cout << "Cuda synchronize error: " << err << std::endl;
    return 2;
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

  return EXIT_SUCCESS;
}
