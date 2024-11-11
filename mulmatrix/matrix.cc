
#include <iostream>

#include "matrix.h"
#include "tools.h"

int main(int argc, char** argv)
{

  int m=512, k=512, n=512;

  int* matrix_a = new int[m*k];
  int* matrix_b = new int[n*k];
  int* matrix_c = new int[m*n];

  initm(matrix_a, m, k);

//  std::cout << "A:" << std::endl;
//  printm(matrix_a, m, k);

  std::cout << "B: increment threads" << std::endl;
  increment_threads(matrix_a, matrix_b, m, k);
//  printm(matrix_b, m, k);
  
  std::cout << "B: increment blocks" << std::endl;
  increment_blocks(matrix_a, matrix_b, m, k);
//  printm(matrix_b, m, k);

  std::cout << "C: matrix transposition" << std::endl;
  matrix_transposition(matrix_a, matrix_b, matrix_c, m, n, k);
//  printm(matrix_c, m, n);

  std::cout << "A*B=C: matrix multiply (CUDA)" << std::endl;
  matrix_multiply_linear(matrix_a, matrix_b, matrix_c, m, n, k);
//  printm(matrix_c, m, n);

  std::cout << "A*B=C: matrix multiply (Armadillo)" << std::endl;
  matrix_multiply(m, k, n, matrix_a, matrix_b, matrix_c);
//  printm(matrix_c, m, n);

  std::cout << "A*B=C: matrix multiply (CUDA submatrix compute)" << std::endl;
  matrix_multiply_submatrix(matrix_a, matrix_b, matrix_c, m, n, k);
//  printm(matrix_c, m, n);

  return 0;
}
