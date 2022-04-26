#include <thrust/copy.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cub/cub.cuh>
#include <iomanip>
#include <iostream>
#include <limits>
#include <src/moderngpu/kernel_mergesort.hxx>
#include <vector>

#include "argument_limits.cuh"
#include "cuda_error.cuh"
#include "data_generator.cuh"
#include "device_allocator.cuh"
#include "device_vector.cuh"
#include "host_vector.cuh"
#include "time_durations.cuh"

int main(int argc, char* argv[]) {
  const std::vector<std::string> arguments(argv + 1, argv + argc);

  if (arguments.size() != 3) {
    std::cout << "./local_gpu_sort_algorithm_benchmark <num_elements> <sort_algorithm> <gpu_id>\n";
    exit(1);
  }

  const size_t num_elements = std::stoull(arguments[0]);
  const std::string sort_algorithm = arguments[1];
  const int gpu_id = std::stoi(arguments[2]);

  DataGenerator data_generator(ArgumentLimits::GetDefaultDistributionSeed());
  HostVector<int> elements(num_elements);

  data_generator.ComputeDistribution<int>(&elements[0], num_elements, ArgumentLimits::GetDefaultDistributionType());

  int initial_max_element = *std::max_element(elements.begin(), elements.end());

  cudaSetDevice(gpu_id);
  cudaDeviceSynchronize();

  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  DeviceVector<int> device_vector(num_elements);
  DeviceAllocator device_allocator;
  device_allocator.Malloc(sizeof(int) * num_elements + 128000000);

  thrust::copy(elements.begin(), elements.end(), device_vector.begin());

  if (sort_algorithm == "cub::DeviceRadixSort::SortKeys") {
    TimeDurations::Get()->Tic("local_sort_phase");
    void* temporary_storage = nullptr;
    size_t num_temporary_bytes = 0;
    cub::DeviceRadixSort::SortKeys(
        temporary_storage, num_temporary_bytes, thrust::raw_pointer_cast(device_vector.data()),
        thrust::raw_pointer_cast(device_vector.data()), num_elements, 0, sizeof(int) * 8, stream);

    temporary_storage = device_allocator.allocate(num_temporary_bytes);

    std::cout << num_temporary_bytes << std::endl;

    cub::DeviceRadixSort::SortKeys(
        temporary_storage, num_temporary_bytes, thrust::raw_pointer_cast(device_vector.data()),
        thrust::raw_pointer_cast(device_vector.data()), num_elements, 0, sizeof(int) * 8, stream);
    CheckCudaError(cudaStreamSynchronize(stream));
    TimeDurations::Get()->Toc("local_sort_phase");

  } else if (sort_algorithm == "thrust::sort") {
    TimeDurations::Get()->Tic("local_sort_phase");
    thrust::sort(thrust::cuda::par(device_allocator).on(stream), device_vector.begin(), device_vector.end());
    CheckCudaError(cudaStreamSynchronize(stream));
    TimeDurations::Get()->Toc("local_sort_phase");

  } else if (sort_algorithm == "mgpu::mergesort") {
    TimeDurations::Get()->Tic("local_sort_phase");
    mgpu::standard_context_t context(false, stream);
    mgpu::mergesort(static_cast<int*>(thrust::raw_pointer_cast(device_vector.data())), num_elements,
                    mgpu::less_t<int>(), context);
    cudaStreamSynchronize(stream);
    TimeDurations::Get()->Toc("local_sort_phase");
  }

  thrust::copy(device_vector.begin(), device_vector.end(), elements.begin());

  device_allocator.Free();

  std::cout << num_elements << ",\"" << sort_algorithm << "\"," << gpu_id << "," << std::fixed << std::setprecision(9)
            << TimeDurations::Get()->durations["local_sort_phase"].count() << "\n";

  if (!std::is_sorted(elements.begin(), elements.end()) ||
      *std::max_element(elements.begin(), elements.end()) != initial_max_element) {
    std::cout << "Error: Invalid sort order.\n";
  }

  return 0;
}
