#include <omp.h>

#include <iomanip>
#include <iostream>

#include "argument_limits.cuh"
#include "cuda_error.cuh"
#include "data_generator.cuh"
#include "device_vector.cuh"
#include "host_vector.cuh"
#include "inplace_memcpy.cuh"
#include "time_durations.cuh"

int main(int argc, char* argv[]) {
  const std::vector<std::string> arguments(argv + 1, argv + argc);

  if (arguments.size() != 3) {
    std::cout << "./inplace_memcpy_benchmark <num_bytes> <gpu_id> <block_size>\n";
    exit(1);
  }

  const size_t num_bytes = std::stoull(arguments[0]);
  const std::string arg_gpus = arguments[1];
  const size_t block_size = std::stoull(arguments[2]);

  std::vector<int> gpus;
  if (!ArgumentLimits::ParseGpus(arg_gpus, &gpus)) {
    exit(1);
  }

  const size_t gpu = gpus.front();

  HostVector<uint8_t> elements(num_bytes * 2);

  uint8_t* begin = thrust::raw_pointer_cast(elements.data());
#pragma omp parallel for num_threads(64) schedule(static)
  for (size_t i = 0; i < num_bytes * 2; ++i) {
    *(begin + i) = 33;
  }

  cudaSetDevice(gpu);
  cudaDeviceSynchronize();

  DeviceVector<uint8_t> device_vector;
  device_vector.resize(num_bytes);

  cudaStream_t htod_stream;
  cudaStream_t dtoh_stream;
  cudaStreamCreateWithFlags(&htod_stream, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&dtoh_stream, cudaStreamNonBlocking);

  thrust::fill(device_vector.begin(), device_vector.end(), 42);

  TimeDurations::Get()->Tic("data_transfer");
  InplaceMemcpy(thrust::raw_pointer_cast(elements.data()), thrust::raw_pointer_cast(device_vector.data()),
                thrust::raw_pointer_cast(elements.data() + num_bytes), num_bytes, num_bytes, htod_stream, dtoh_stream,
                block_size);
  TimeDurations::Get()->Toc("data_transfer");

  if (elements[num_bytes] != 42 || elements[(2 * num_bytes) - 1] != 42 || device_vector[0] != 33 ||
      device_vector[num_bytes - 1] != 33) {
    std::cout << "ERROR" << std::endl;
    exit(1);
  }

  cudaSetDevice(gpu);
  device_vector.clear();
  device_vector.shrink_to_fit();
  cudaStreamDestroy(htod_stream);
  cudaStreamDestroy(dtoh_stream);
  cudaDeviceSynchronize();

  std::cout << std::fixed << std::setprecision(9);
  std::cout << num_bytes << "," << gpu << "," << block_size << ","
            << TimeDurations::Get()->durations["data_transfer"].count() << std::endl;

  return 0;
}
