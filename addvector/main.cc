#include <iostream>
#include <cstdlib>

#include "tests.h"

using namespace std;

int main(int argc, char** argv)
{
  thread_sync_prod_test();
  stride_prod_test();

  return EXIT_SUCCESS;
}
