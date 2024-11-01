
#include <iostream>

#include <cuda_runtime.h>
#include "matrix.h"

__global__
void IncrementThreads(const int* a, int* b, int m, int k)
{
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;

  int idx = tx + ty * m;

  b[idx] = a[idx] * threadIdx.x;
}

__global__
void IncrementBlocks(const int* a, int* b, int m, int k)
{
  int tx = blockIdx.y * blockDim.y + threadIdx.y;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;

  int idx = tx + ty * m;

  b[idx] = a[idx] * threadIdx.y;
}

__global__
void MatrixTransposition(const int* a, const int* b, int* c, int m, int n, int k)
{
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;

  int idx = tx + ty * m;
  int idy = ty + tx * n;

  c[idy] = b[idx];
}

void matrix_transposition(int* matrix_a, int* matrix_b, int* matrix_c, int m, int n, int k)
{
  cudaError_t error;
  int *d_ma, *d_mb, *d_mc;

  cudaMalloc(&d_ma, sizeof(int) * k * m);
  cudaMalloc(&d_mb, sizeof(int) * n * k);
  cudaMalloc(&d_mc, sizeof(int) * m * n);

  error = cudaMemcpy(d_ma, matrix_a, sizeof(int) * k * m, cudaMemcpyHostToDevice);
  if (error != cudaSuccess)
  {
    std::cout << "Can't init device memory for entry A." << std::endl;
    return;
  }
  
  error = cudaMemcpy(d_mb, matrix_b, sizeof(int) * n * k, cudaMemcpyHostToDevice);
  if (error != cudaSuccess)
  {
    std::cout << "Can't init device memory for entry B." << std::endl;
    return;
  }

  dim3 threads(m, n);
  dim3 blocks(m/threads.x, n/threads.y);

  MatrixTransposition<<<blocks, threads>>>(d_ma, d_mb, d_mc, m, n, k);
  
  error = cudaDeviceSynchronize();
  if (error != cudaSuccess)
  {
    std::cout << "Can't start CUDA computations!" << std::endl;
    return;
  }

  error = cudaMemcpy(matrix_c, d_mc, sizeof(int) * m * n, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess)
  {
    std::cout << "Can't copy from device memory." << std::endl;
  }

  cudaFree(d_ma);
  cudaFree(d_mb);
  cudaFree(d_mc);
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

  IncrementThreads<<<1, n*k>>>(d_ma, d_mb, n, k);

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
