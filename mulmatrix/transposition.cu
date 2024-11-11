
#include <iostream>

#include <cuda_runtime.h>
#include "matrix.h"

#define BLOCK_SIZE 32

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

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(n/dimBlock.x, m/dimBlock.y);

  MatrixTransposition<<<dimGrid, dimBlock>>>(d_ma, d_mb, d_mc, m, n, k);
  
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

