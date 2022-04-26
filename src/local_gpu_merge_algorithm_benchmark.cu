#include <thrust/copy.h>
#include <thrust/sort.h>

#include <cub/cub.cuh>
#include <iomanip>
#include <iostream>
#include <src/moderngpu/kernel_merge.hxx>
#include <vector>

#include "argument_limits.cuh"
#include "data_generator.cuh"
#include "device_allocator.cuh"
#include "device_vector.cuh"
#include "host_vector.cuh"
#include "time_durations.cuh"

int main(int argc, char* argv[]) {
  const std::vector<std::string> arguments(argv + 1, argv + argc);

  if (arguments.size() != 3) {
    std::cout << "./local_gpu_merge_algorithm_benchmark <num_elements> <merge_algorithm> <gpu_id>\n";
    exit(1);
  }

  const size_t num_elements = std::stoull(arguments[0]);
  const std::string merge_algorithm = arguments[1];
  const int gpu_id = std::stoi(arguments[2]);

  HostVector<int> elements_1(num_elements / 2);
  HostVector<int> elements_2(num_elements - (num_elements / 2));
  HostVector<int> sorted_elements(num_elements);

  DataGenerator data_generator(ArgumentLimits::GetDefaultDistributionSeed());
  data_generator.ComputeDistribution<int>(&elements_1[0], elements_1.size(), "sorted");
  data_generator.ComputeDistribution<int>(&elements_2[0], elements_2.size(), "sorted");

  int initial_max_element_1 = elements_1.back();
  int initial_max_element_2 = elements_2.back();

  int initial_max_element = std::max(initial_max_element_1, initial_max_element_2);

  cudaSetDevice(gpu_id);
  cudaDeviceSynchronize();

  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  DeviceVector<int> device_vector_1(num_elements / 2);
  DeviceVector<int> device_vector_2(num_elements - (num_elements / 2));
  DeviceVector<int> sorted_device_vector(num_elements);

  DeviceAllocator device_allocator;
  device_allocator.Malloc(128000000 + num_elements * sizeof(int));

  thrust::copy(elements_1.begin(), elements_1.end(), device_vector_1.begin());
  thrust::copy(elements_2.begin(), elements_2.end(), device_vector_2.begin());

  if (merge_algorithm == "thrust::merge") {
    TimeDurations::Get()->Tic("local_merge_phase");
    thrust::merge(thrust::cuda::par(device_allocator).on(stream), device_vector_1.begin(), device_vector_1.end(),
                  device_vector_2.begin(), device_vector_2.end(), sorted_device_vector.begin());
    cudaStreamSynchronize(stream);
    TimeDurations::Get()->Toc("local_merge_phase");

  } else if (merge_algorithm == "mgpu::merge") {
    TimeDurations::Get()->Tic("local_merge_phase");
    mgpu::standard_context_t context(false, stream);
    mgpu::merge(static_cast<int*>(thrust::raw_pointer_cast(device_vector_1.data())), num_elements / 2,
                static_cast<int*>(thrust::raw_pointer_cast(device_vector_2.data())), num_elements - (num_elements / 2),
                static_cast<int*>(thrust::raw_pointer_cast(sorted_device_vector.data())), mgpu::less_t<int>(), context);
    cudaStreamSynchronize(stream);
    TimeDurations::Get()->Toc("local_merge_phase");
  }

  thrust::copy(sorted_device_vector.begin(), sorted_device_vector.end(), sorted_elements.begin());

  device_allocator.Free();

  std::cout << num_elements << ",\"" << merge_algorithm << "\"," << gpu_id << "," << std::fixed << std::setprecision(9)
            << TimeDurations::Get()->durations["local_merge_phase"].count() << "\n";

  if (!std::is_sorted(sorted_elements.begin(), sorted_elements.end()) ||
      *std::max_element(sorted_elements.begin(), sorted_elements.end()) != initial_max_element) {
    std::cout << "Error: Invalid sort order.\n";
  }

  return 0;
}
