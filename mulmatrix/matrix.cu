
#include <iostream>

#include <cuda_runtime.h>
#include "matrix.h"

__global__
void IncrementRow(const int* a, int* b, int m, int k)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  b[idx] = a[idx] * 2;
}

__global__
void IncrementCol(const int* a, int* b, int m, int k)
{
  int idx = threadIdx.x + threadIdx.y;

  b[idx] = a[idx] * 2;
}

void increment_col(int* matrix_a, int* matrix_b, int n, int k)
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

  IncrementCol<<<1, 16>>>(d_ma, d_mb, n, k);

  errors = cudaDeviceSynchronize();
  if (errors != cudaSuccess)
  {
    std::cout << "Can't start cuda computation!" << std::endl;
  }


  cudaMemcpy(matrix_b, d_mb, matrix_size, cudaMemcpyDeviceToHost);

  cudaFree(d_ma);
  cudaFree(d_mb);
}

void increment_row(int* matrix_a, int* matrix_b, int m, int k)
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

  IncrementRow<<<1, 16>>>(d_ma, d_mb, m, k);

  errors = cudaDeviceSynchronize();
  if (errors != cudaSuccess)
  {
    std::cout << "Can't start cuda computation!" << std::endl;
  }


  cudaMemcpy(matrix_b, d_mb, matrix_size, cudaMemcpyDeviceToHost);

  cudaFree(d_ma);
  cudaFree(d_mb);
}
