
#include <iostream>

#include <cuda_runtime.h>
#include "matrix.h"

__global__
void IncrementThreads(const int* a, int* b, int m, int k)
{
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;

  int idx = tx + ty * m;

  b[idx] = a[idx] + threadIdx.x;
}

__global__
void IncrementBlocks(const int* a, int* b, int m, int k)
{
  int tx = blockIdx.y * blockDim.y + threadIdx.y;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;

  int idx = tx + ty * m;

  b[idx] = a[idx] + threadIdx.y;
}

void increment_threads(int* matrix_a, int* matrix_b, int n, int k)
{
  cudaError_t error;
  int *d_ma, *d_mb;

  int matrix_size = sizeof(int) * n * k;

  error = cudaMalloc(&d_ma, matrix_size);
  if (error != cudaSuccess)
  {
    std::cout << "Can't allocate memory on device for matrix A" << std::endl;
    return;
  }

  error = cudaMalloc(&d_mb, matrix_size);
  if (error != cudaSuccess)
  {
    std::cout << "Can't allocate memory on device for matrix B" << std::endl;
    return;
  }

  cudaMemcpy(d_ma, matrix_a, matrix_size, cudaMemcpyHostToDevice);

  int threads = n * k;
  if (threads > 1024)
  {
    threads = 1024;
  }

  IncrementThreads<<<1, threads>>>(d_ma, d_mb, n, k);

  error = cudaDeviceSynchronize();
  if (error != cudaSuccess)
  {
    std::cout << "Can't start cuda computation!" << std::endl;
  }

  cudaMemcpy(matrix_b, d_mb, matrix_size, cudaMemcpyDeviceToHost);

  cudaFree(d_ma);
  cudaFree(d_mb);
}

void increment_blocks(int* matrix_a, int* matrix_b, int m, int k)
{
  cudaError_t error;
  int *d_ma, *d_mb;

  int matrix_size = sizeof(int) * m * k;

  error = cudaMalloc(&d_ma, matrix_size);
  if (error != cudaSuccess)
  {
    std::cout << "Can't allocate memory on device for matrix A" << std::endl;
    return;
  }

  error = cudaMalloc(&d_mb, matrix_size);
  if (error != cudaSuccess)
  {
    std::cout << "Can't allocate memory on device for matrix B" << std::endl;
    return;
  }

  cudaMemcpy(d_ma, matrix_a, matrix_size, cudaMemcpyHostToDevice);

  IncrementBlocks<<<m*k, 1>>>(d_ma, d_mb, m, k);

  error = cudaDeviceSynchronize();
  if (error != cudaSuccess)
  {
    std::cout << "Can't start cuda computation!" << std::endl;
  }


  cudaMemcpy(matrix_b, d_mb, matrix_size, cudaMemcpyDeviceToHost);

  cudaFree(d_ma);
  cudaFree(d_mb);
}
