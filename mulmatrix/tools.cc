
#include  "tools.h"
#include <armadillo>

void initm(int* matrix, int rows, int cols)
{
  for (int i = 0; i < rows * cols; i++)
  {
    *(matrix+i) = 2;
  }
}

void printm(int* matrix, int rows, int cols)
{
  arma::Mat<int> matrix_a(matrix, rows, cols, false, true);
  matrix_a.print();
}

void matrix_multiply(int m, int k, int n, int* matrix_a, int* matrix_b, int* matrix_c)
{
  arma::Mat<int> A(matrix_a, m, k, false, true);
  arma::Mat<int> B(matrix_b, k, n, false, true);
  arma::Mat<int> C(matrix_c, m, n, false, true);

  C = A * B;
}
