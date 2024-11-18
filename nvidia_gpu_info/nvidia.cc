/* 
 * Nvidia інтерфейс метрики.
 */

#include "nvidia.h"

#include <nvml.h>

#include <stdlib.h>
#include <iostream>

int nvidia_gpu_info(int gpu_id)
{
  int result = EXIT_SUCCESS;
  unsigned int gpus_count;
  nvmlReturn_t nvml_result;

  nvml_result = nvmlInit();
  nvml_result = nvmlDeviceGetCount(&gpus_count);

  std::cout << "GPUs in system: " << gpus_count << std::endl;

  nvmlDevice_t device_handle;
  char name[45];
  int compute_version;

  for (int i = 0; i < gpus_count; i++)
  {
    nvml_result = nvmlDeviceGetHandleByIndex(i, &device_handle);
    nvml_result = nvmlDeviceGetName(device_handle, name, 45);

    std::cout << "Device name: " << name << std::endl;
  }

  unsigned int gpu_units;
  nvmlUnit_t unit_handle;
  nvmlUnitInfo_t unit_info;

  nvml_result = nvmlUnitGetCount(&gpu_units);
  
  std::cout << "GPU units: " << gpu_units << std::endl;

  for (int i = 0; i < gpu_units; i++)
  {
    nvml_result = nvmlUnitGetHandleByIndex(i, &unit_handle);
    nvml_result = nvmlUnitGetUnitInfo(unit_handle, &unit_info);
 
    std::cout << "Unit name: " << unit_info.name << std::endl;

    if (nvml_result != NVML_SUCCESS)
    {
      result = nvml_result;
    }
  }

  nvml_result = nvmlDeviceGetAccountingPids(device_handle, 0, NULL);

  if (nvml_result == NVML_SUCCESS || nvml_result == NVML_ERROR_INSUFFICIENT_SIZE)
  {
    std::cout << "No one process is using GPU: " << name << std::endl;
  }
  if (nvml_result == NVML_ERROR_NOT_SUPPORTED)
  {
    std::cout << "The requested accounting PIDs is not available on target device " << name << std::endl;
  }
  

  nvml_result = nvmlShutdown();

  return result;
}
