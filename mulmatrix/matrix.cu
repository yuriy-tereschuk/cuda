
#include <iostream>

#include <cuda_runtime.h>
#include "matrix.h"

__global__
void IncrementThreads(const int* a, int* b, int m, int k)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  b[idx] = a[idx] * threadIdx.x;
}

__global__
void IncrementBlocks(const int* a, int* b, int m, int k)
{
  int idx = threadIdx.x + threadIdx.y;

  b[idx] = a[idx] * threadIdx.y;
}

void increment_threads(int* matrix_a, int* matrix_b, int n, int k)
{
  cudaError_t errors;
  int *d_ma, *d_mb;

  int matrix_size = sizeof(int) * n * k;

  errors = cudaMalloc(&d_ma, matrix_size);
  if (errors != cudaSuccess)
  {
    std::cout << "Can't allocate memory on device for matrix A" << std::endl;
    return;
  }

  errors = cudaMalloc(&d_mb, matrix_size);
  if (errors != cudaSuccess)
  {
    std::cout << "Can't allocate memory on device for matrix B" << std::endl;
    return;
  }

  cudaMemcpy(d_ma, matrix_a, matrix_size, cudaMemcpyHostToDevice);

  IncrementThreads<<<1, 16>>>(d_ma, d_mb, n, k);

  errors = cudaDeviceSynchronize();
  if (errors != cudaSuccess)
  {
    std::cout << "Can't start cuda computation!" << std::endl;
  }


  cudaMemcpy(matrix_b, d_mb, matrix_size, cudaMemcpyDeviceToHost);

  cudaFree(d_ma);
  cudaFree(d_mb);
}

void increment_blocks(int* matrix_a, int* matrix_b, int m, int k)
{
  cudaError_t errors;
  int *d_ma, *d_mb;

  int matrix_size = sizeof(int) * m * k;

  errors = cudaMalloc(&d_ma, matrix_size);
  if (errors != cudaSuccess)
  {
    std::cout << "Can't allocate memory on device for matrix A" << std::endl;
    return;
  }

  errors = cudaMalloc(&d_mb, matrix_size);
  if (errors != cudaSuccess)
  {
    std::cout << "Can't allocate memory on device for matrix B" << std::endl;
    return;
  }

  cudaMemcpy(d_ma, matrix_a, matrix_size, cudaMemcpyHostToDevice);

  IncrementBlocks<<<16, 1>>>(d_ma, d_mb, m, k);

  errors = cudaDeviceSynchronize();
  if (errors != cudaSuccess)
  {
    std::cout << "Can't start cuda computation!" << std::endl;
  }


  cudaMemcpy(matrix_b, d_mb, matrix_size, cudaMemcpyDeviceToHost);

  cudaFree(d_ma);
  cudaFree(d_mb);
}
