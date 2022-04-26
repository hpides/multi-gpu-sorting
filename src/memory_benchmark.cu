#include <omp.h>

#include <iomanip>
#include <iostream>

#include "cuda_error.cuh"
#include "device_vector.cuh"
#include "time_durations.cuh"

int main(int argc, char* argv[]) {
  const std::vector<std::string> arguments(argv + 1, argv + argc);

  if (arguments.size() != 3) {
    std::cout << "./memory_benchmark <num_bytes> <num_gpus> <execution_type=serial,parallel>" << std::endl;
    exit(1);
  }

  const size_t num_bytes = std::stoull(arguments[0]);
  const size_t num_gpus = std::stoull(arguments[1]);
  const std::string execution_type = arguments[2];

  if (execution_type == "parallel") {
    omp_set_num_threads(num_gpus);
  } else if (execution_type == "serial") {
    omp_set_num_threads(1);
  } else {
    std::cout << "./memory_benchmark <num_bytes> <num_gpus> <execution_type=serial,parallel>" << std::endl;
    exit(1);
  }

  for (size_t i = 0; i < num_gpus; ++i) {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
    TimeDurations::Get()->Tic("memory_allocation_" + std::to_string(i));
  }

  std::vector<DeviceVector<uint8_t>> device_vectors(num_gpus);

  TimeDurations::Get()->Tic("total_memory_allocation");
#pragma omp parallel for
  for (size_t i = 0; i < num_gpus; ++i) {
    TimeDurations::Get()->Tic("memory_allocation_" + std::to_string(i));
    cudaSetDevice(i);
    device_vectors[i].resize(num_bytes);
    TimeDurations::Get()->Toc("memory_allocation_" + std::to_string(i));
  }
  TimeDurations::Get()->Toc("total_memory_allocation");

#pragma omp parallel for
  for (size_t i = 0; i < num_gpus; ++i) {
    cudaSetDevice(i);
    device_vectors[i].clear();
    device_vectors[i].shrink_to_fit();
  }

  std::cout << num_bytes << "," << num_gpus << "," << execution_type << ","
            << "\"" << std::fixed << std::setprecision(9);

  for (size_t i = 0; i < num_gpus; ++i) {
    std::cout << TimeDurations::Get()->durations["memory_allocation_" + std::to_string(i)].count()
              << (i == num_gpus - 1 ? "\"" : "") << ",";
  }

  std::cout << TimeDurations::Get()->durations["total_memory_allocation"].count() << "\n";

  return 0;
}
