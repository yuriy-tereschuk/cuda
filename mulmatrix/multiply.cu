
#include <iostream>

#include <cuda_runtime.h>
#include "matrix.h"

#define BLOCK_SIZE 16

typedef struct {
  int height;
  int width;
  int stride;
  int* elements;
} Matrix;

__global__
void MatrixMultiplyLinear(const int* a, const int* b, int* c, int m, int n, int k)
{
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;

  int idx = tx + ty * m;
  int idy = ty + tx * n;

  int value = a[tx * m] * b[ty * n];

  for (int i = 1; i < k; ++i)
  {
    value += a[tx * m + i] * b[ty * n + i];
  }

  c[idx] = value;
}

__device__
Matrix GetSubMatrix(Matrix matrix, int row, int col)
{
  Matrix subMatrix;

  subMatrix.width = BLOCK_SIZE;
  subMatrix.height = BLOCK_SIZE;
  subMatrix.stride = matrix.stride;
  subMatrix.elements = &matrix.elements[matrix.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];

  return subMatrix;
}

__device__
int GetMatrixElement(Matrix matrix, int row, int col)
{
  return matrix.elements[row * matrix.stride + col];
}

__device__
void SetMatrixElement(Matrix matrix, int value, int row, int col)
{
  matrix.elements[row * matrix.stride + col] = value;
}

__global__
void MatrixMultiplySubMatrix(const Matrix a, const Matrix b, Matrix c)
{
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  Matrix subC = GetSubMatrix(c, blockRow, blockCol);
 
  int value = 0;

  int row = threadIdx.y;
  int col = threadIdx.x;

  for (int m = 0; m < a.width / BLOCK_SIZE; ++m)
  {
    Matrix subA = GetSubMatrix(a, blockRow, m);
    Matrix subB = GetSubMatrix(b, m, blockCol);
    
    __shared__ int m_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int m_B[BLOCK_SIZE][BLOCK_SIZE];

    m_A[row][col] = GetMatrixElement(subA, row, col);
    m_B[row][col] = GetMatrixElement(subB, row, col);
    __syncthreads();

    for (int e = 0; e < BLOCK_SIZE; ++e)
    {
      value += m_A[row][e] * m_B[e][col];
    }
    __syncthreads();
  }

  SetMatrixElement(subC, value, row, col);
}

void matrix_multiply_submatrix(int* matrix_a, int* matrix_b, int* matrix_c, int m, int n, int k)
{
  cudaError_t error;
  Matrix d_ma, d_mb, d_mc;

  d_ma.width = d_ma.stride = k;
  d_ma.height = m;
  d_mb.width = d_mb.stride = n;
  d_mb.height = k;
  d_mc.width = d_mc.stride = n;
  d_mc.height = m;

  cudaMalloc(&d_ma.elements, sizeof(int) * k * m);
  cudaMalloc(&d_mb.elements, sizeof(int) * n * k);
  cudaMalloc(&d_mc.elements, sizeof(int) * m * n);

  error = cudaMemcpy(d_ma.elements, matrix_a, sizeof(int) * k * m, cudaMemcpyHostToDevice);
  if (error != cudaSuccess)
  {
    std::cout << "Can't init device memory for entry A." << std::endl;
    return;
  }
  
  error = cudaMemcpy(d_mb.elements, matrix_b, sizeof(int) * n * k, cudaMemcpyHostToDevice);
  if (error != cudaSuccess)
  {
    std::cout << "Can't init device memory for entry B." << std::endl;
    return;
  }

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(n/dimBlock.x, m/dimBlock.y);

  MatrixMultiplySubMatrix<<<dimGrid, dimBlock>>>(d_ma, d_mb, d_mc);
  
  error = cudaDeviceSynchronize();
  if (error != cudaSuccess)
  {
    std::cout << "Can't start CUDA computations!" << std::endl;
    return;
  }

  error = cudaMemcpy(matrix_c, d_mc.elements, sizeof(int) * m * n, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess)
  {
    std::cout << "Can't copy from device memory." << std::endl;
  }

  cudaFree(d_ma.elements);
  cudaFree(d_mb.elements);
  cudaFree(d_mc.elements);
}

void matrix_multiply_linear(int* matrix_a, int* matrix_b, int* matrix_c, int m, int n, int k)
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

  MatrixMultiplyLinear<<<dimGrid, dimBlock>>>(d_ma, d_mb, d_mc, m, n, k);
  
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

