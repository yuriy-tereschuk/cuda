
#include <iostream>

#include "matrix.h"
#include "tools.h"

int main(int argc, char** argv)
{

  int m=4, k=4;

  int* matrix_a = new int[m*k];
  int* matrix_b = new int[m*k];

  initm(matrix_a, m, k);

  std::cout << "A:" << std::endl;
  printm(matrix_a, m, k);

  std::cout << "B: increment threads" << std::endl;
  increment_threads(matrix_a, matrix_b, m, k);
  printm(matrix_b, m, k);
  
  std::cout << "B: increment blocks" << std::endl;
  increment_blocks(matrix_a, matrix_b, m, k);
  printm(matrix_b, m, k);

  return 0;
}
