
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
