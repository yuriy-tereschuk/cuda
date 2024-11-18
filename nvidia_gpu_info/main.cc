
#include "nvidia.h"

#include <stdlib.h>
#include <iostream>

int main(int argc, char** argv)
{

  int gpu_id = 0;

  int result = nvidia_gpu_info(gpu_id);

  return result;
}
