#include "tools.h"

void init(int* vec, int elements)
{
  for(int i = 0; i < elements; i++)
  {
    *vec++ = 2;
  }
}

int host_prod(int* a, int* b, int elements)
{
  int prod = a[0] + b[0];
  for ( int i = 1; i < elements; i++)
  {
    prod += a[i] + b[i];
  }

  return prod;
}

