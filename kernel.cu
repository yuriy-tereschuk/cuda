
#include "kernel.h"

#include <device_launch_parameters.h>
#include <cuda_runtime.h>

/*
Функція ядра заповнює нову матрицю у відповідності до блоків-сегментів на 
які поділена матриця вхідних даних.
*/
__global__ 
void MatrixBlockMask(int* c, const int* a, const int* b, int w, int h)
{


	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	c[row * w + col] = blockIdx.y + blockIdx.x + (blockIdx.y * (blockDim.x - 1));
}

/*
Задано розмір блока-сегмента у 4 колонки та 8 рядків. За таких розмірів сегментів,
матриця вхідних даних, що має розміри 16х16, буде сегментована у 8 блоків-сегметів.
*/
void matrix_block_mask(int* c, const int* a, const int *b, int w, int h)
{
	int N = w * h * sizeof(int);
	int *d_a, *d_b, *d_c;

	cudaMalloc(&d_a, N);
	cudaMalloc(&d_b, N);
	cudaMalloc(&d_c, N);

	cudaError_t error;

	cudaMemcpy(d_a, a, N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, N, cudaMemcpyHostToDevice);

	dim3 dimBlock(4, 8);
	dim3 dimGrid(w / dimBlock.x, h / dimBlock.y);

	MatrixBlockMask<<<dimGrid, dimBlock>>> (d_c, d_a, d_b, w , h);
	
	error = cudaDeviceSynchronize();

	cudaMemcpy(c, d_c, N, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}
