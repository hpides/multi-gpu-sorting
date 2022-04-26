#include <omp.h>

#include <iomanip>
#include <iostream>

#include "argument_limits.cuh"
#include "cuda_error.cuh"
#include "device_vector.cuh"
#include "host_vector.cuh"
#include "time_durations.cuh"

int main(int argc, char* argv[]) {
  const std::vector<std::string> arguments(argv + 1, argv + argc);

  if (arguments.size() != 4) {
    std::cout << "./data_transfer_benchmark <num_bytes> <gpus> <num_repetitions> <execution_type=serial,parallel>\n";
    exit(1);
  }

  const size_t num_bytes = std::stoull(arguments[0]);
  const std::string arg_gpus = arguments[1];
  const size_t num_repetitions = std::stoull(arguments[2]);
  const std::string execution_type = arguments[3];

  std::vector<int> gpus;
  if (!ArgumentLimits::ParseGpus(arg_gpus, &gpus)) {
    std::cout << "./data_transfer_benchmark <num_bytes> <gpus> <num_repetitions> <execution_type=serial,parallel>\n";
    exit(1);
  }

  const size_t max_gpu = *std::max_element(gpus.begin(), gpus.end());

  if (execution_type == "serial") {
    omp_set_num_threads(1);
  } else if (execution_type == "parallel") {
    omp_set_num_threads(gpus.size());
  } else {
    std::cout << "./data_transfer_benchmark <num_bytes> <gpus> <num_repetitions> <execution_type=serial,parallel>\n";
    exit(1);
  }

  HostVector<uint8_t> elements(num_bytes * 2 * gpus.size());

  for (size_t i = 0; i < gpus.size(); ++i) {
    cudaSetDevice(gpus[i]);
    cudaDeviceSynchronize();
    for (size_t j = 0; j < gpus.size(); ++j) {
      if (i != j) {
        CheckCudaError(cudaDeviceEnablePeerAccess(gpus[j], 0));
      }
    }
  }

  std::vector<DeviceVector<uint8_t>> device_vectors_primary(max_gpu + 1);
  std::vector<DeviceVector<uint8_t>> device_vectors_secondary(max_gpu + 1);
  std::vector<cudaStream_t> htod_streams(max_gpu + 1);
  std::vector<cudaStream_t> dtoh_streams(max_gpu + 1);
  std::vector<cudaStream_t> streams(max_gpu + 1);

  for (size_t i = 0; i < gpus.size(); ++i) {
    cudaSetDevice(gpus[i]);
    device_vectors_primary[gpus[i]].resize(num_bytes * 2);
    device_vectors_secondary[gpus[i]].resize(num_bytes * 2);
    cudaStreamCreateWithFlags(&streams[gpus[i]], cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&htod_streams[gpus[i]], cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&dtoh_streams[gpus[i]], cudaStreamNonBlocking);

    TimeDurations::Get()->Tic("host_to_device_transfer_" + std::to_string(gpus[i]));
    TimeDurations::Get()->Tic("device_to_host_transfer_" + std::to_string(gpus[i]));
    TimeDurations::Get()->Tic("bidirectional_data_transfer_" + std::to_string(gpus[i]));
  }

  if (execution_type == "parallel") {
    for (size_t i = 0; i < gpus.size() / 2; ++i) {
      const int left_device = gpus[gpus.size() / 2 - i - 1];
      const int right_device = gpus[gpus.size() / 2 + i];

      TimeDurations::Get()->Tic("peer_to_peer_transfer_" + std::to_string(left_device) + "-" +
                                std::to_string(right_device));
    }
  }

  TimeDurations::Get()->Tic("total_host_to_device_transfer");
#pragma omp parallel for
  for (size_t i = 0; i < gpus.size(); ++i) {
    cudaSetDevice(gpus[i]);
    TimeDurations::Get()->Tic("host_to_device_transfer_" + std::to_string(gpus[i]));
    CheckCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(device_vectors_primary[gpus[i]].data()),
                                   thrust::raw_pointer_cast(elements.data()) + (num_bytes * i), num_bytes,
                                   cudaMemcpyHostToDevice, streams[gpus[i]]));
    cudaStreamSynchronize(streams[gpus[i]]);
    TimeDurations::Get()->Toc("host_to_device_transfer_" + std::to_string(gpus[i]));
  }
  TimeDurations::Get()->Toc("total_host_to_device_transfer");

  if (gpus.size() > 1) {
    if (execution_type == "serial") {
      TimeDurations::Get()->Tic("total_peer_to_peer_transfer");
      for (size_t i = 1; i < gpus.size(); ++i) {
        cudaSetDevice(gpus[0]);
        TimeDurations::Get()->Tic("peer_to_peer_transfer_from_" + std::to_string(gpus[0]) + "_to_" +
                                  std::to_string(gpus[i]));
        CheckCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(device_vectors_primary[gpus[i]].data()),
                                       thrust::raw_pointer_cast(device_vectors_primary[gpus[0]].data()), num_bytes,
                                       cudaMemcpyDeviceToDevice, streams[gpus[0]]));

        cudaStreamSynchronize(streams[gpus[0]]);
        TimeDurations::Get()->Toc("peer_to_peer_transfer_from_" + std::to_string(gpus[0]) + "_to_" +
                                  std::to_string(gpus[i]));
      }
      TimeDurations::Get()->Toc("total_peer_to_peer_transfer");

    } else if (execution_type == "parallel") {
      TimeDurations::Get()->Tic("total_peer_to_peer_transfer");
#pragma omp parallel for
      for (size_t i = 0; i < gpus.size() / 2; ++i) {
        const int left_device = gpus[gpus.size() / 2 - i - 1];
        const int right_device = gpus[gpus.size() / 2 + i];

        TimeDurations::Get()->Tic("peer_to_peer_transfer_" + std::to_string(left_device) + "-" +
                                  std::to_string(right_device));
        cudaSetDevice(left_device);
        CheckCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(device_vectors_secondary[left_device].data()),
                                       thrust::raw_pointer_cast(device_vectors_primary[right_device].data()), num_bytes,
                                       cudaMemcpyDeviceToDevice, streams[left_device]));

        cudaSetDevice(right_device);
        CheckCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(device_vectors_secondary[right_device].data()),
                                       thrust::raw_pointer_cast(device_vectors_primary[left_device].data()), num_bytes,
                                       cudaMemcpyDeviceToDevice, streams[right_device]));

        CheckCudaError(cudaStreamSynchronize(streams[left_device]));
        CheckCudaError(cudaStreamSynchronize(streams[right_device]));
        TimeDurations::Get()->Toc("peer_to_peer_transfer_" + std::to_string(left_device) + "-" +
                                  std::to_string(right_device));
      }
      TimeDurations::Get()->Toc("total_peer_to_peer_transfer");
    }
  }

  TimeDurations::Get()->Tic("total_device_to_host_transfer");
#pragma omp parallel for
  for (size_t i = 0; i < gpus.size(); ++i) {
    cudaSetDevice(gpus[i]);
    TimeDurations::Get()->Tic("device_to_host_transfer_" + std::to_string(gpus[i]));
    CheckCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(elements.data()) + (num_bytes * i),
                                   thrust::raw_pointer_cast(device_vectors_secondary[gpus[i]].data()), num_bytes,
                                   cudaMemcpyDeviceToHost, streams[gpus[i]]));
    cudaStreamSynchronize(streams[gpus[i]]);
    TimeDurations::Get()->Toc("device_to_host_transfer_" + std::to_string(gpus[i]));
  }
  TimeDurations::Get()->Toc("total_device_to_host_transfer");

  TimeDurations::Get()->Tic("total_bidirectional_data_transfer");
#pragma omp parallel for
  for (size_t i = 0; i < gpus.size(); ++i) {
    cudaSetDevice(gpus[i]);
    TimeDurations::Get()->Tic("bidirectional_data_transfer_" + std::to_string(gpus[i]));
    for (size_t j = 0; j < num_repetitions; ++j) {
      CheckCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(device_vectors_primary[gpus[i]].data()),
                                     thrust::raw_pointer_cast(elements.data() + (num_bytes * i)), num_bytes,
                                     cudaMemcpyHostToDevice, htod_streams[gpus[i]]));
      CheckCudaError(
          cudaMemcpyAsync(thrust::raw_pointer_cast(elements.data() + (num_bytes * gpus.size()) + (num_bytes * i)),
                          thrust::raw_pointer_cast(device_vectors_primary[gpus[i]].data() + num_bytes), num_bytes,
                          cudaMemcpyDeviceToHost, dtoh_streams[gpus[i]]));

      CheckCudaError(cudaStreamSynchronize(htod_streams[gpus[i]]));
      CheckCudaError(cudaStreamSynchronize(dtoh_streams[gpus[i]]));
    }
    TimeDurations::Get()->Toc("bidirectional_data_transfer_" + std::to_string(gpus[i]));
  }
  TimeDurations::Get()->Toc("total_bidirectional_data_transfer");

  for (size_t i = 0; i < gpus.size(); ++i) {
    cudaSetDevice(gpus[i]);
    device_vectors_primary[gpus[i]].clear();
    device_vectors_primary[gpus[i]].shrink_to_fit();
    device_vectors_secondary[gpus[i]].clear();
    device_vectors_secondary[gpus[i]].shrink_to_fit();
    cudaStreamDestroy(streams[gpus[i]]);
    cudaStreamDestroy(htod_streams[gpus[i]]);
    cudaStreamDestroy(dtoh_streams[gpus[i]]);
    cudaDeviceSynchronize();
  }

  std::cout << std::fixed << std::setprecision(9);
  std::cout << num_bytes << ",\"";
  for (size_t i = 0; i < gpus.size(); ++i) {
    std::cout << gpus[i] << ((i < gpus.size() - 1) ? "," : "\",");
  }

  std::cout << num_repetitions << "," << execution_type << ",\"";
  for (size_t i = 0; i < gpus.size(); ++i) {
    std::cout << TimeDurations::Get()->durations["host_to_device_transfer_" + std::to_string(gpus[i])].count()
              << (i == gpus.size() - 1 ? "\"," : ",");
  }
  std::cout << TimeDurations::Get()->durations["total_host_to_device_transfer"].count() << ",\"";

  if (execution_type == "serial") {
    for (size_t i = 1; i < gpus.size(); ++i) {
      std::cout
          << TimeDurations::Get()
                 ->durations["peer_to_peer_transfer_from_" + std::to_string(gpus[0]) + "_to_" + std::to_string(gpus[i])]
                 .count()
          << (i == gpus.size() - 1 ? "\"," : ",");
    }
  } else if (execution_type == "parallel") {
    for (size_t i = 0; i < gpus.size() / 2; ++i) {
      const int left_device = gpus[gpus.size() / 2 - i - 1];
      const int right_device = gpus[gpus.size() / 2 + i];
      std::cout << TimeDurations::Get()
                       ->durations["peer_to_peer_transfer_" + std::to_string(left_device) + "-" +
                                   std::to_string(right_device)]
                       .count()
                << (right_device == gpus.back() ? "\"" : "") << ",";
    }
  }

  if (gpus.size() == 1) {
    std::cout << "\",";
  }
  std::cout << TimeDurations::Get()->durations["total_peer_to_peer_transfer"].count() << ",";

  std::cout << "\"";
  for (size_t i = 0; i < gpus.size(); ++i) {
    std::cout << TimeDurations::Get()->durations["device_to_host_transfer_" + std::to_string(gpus[i])].count()
              << (i == gpus.size() - 1 ? "\"," : ",");
  }
  std::cout << TimeDurations::Get()->durations["total_device_to_host_transfer"].count() << ",\"";

  for (size_t i = 0; i < gpus.size(); ++i) {
    std::cout << TimeDurations::Get()->durations["bidirectional_data_transfer_" + std::to_string(gpus[i])].count() /
                     num_repetitions
              << (i == gpus.size() - 1 ? "\"," : ",");
  }

  std::cout << TimeDurations::Get()->durations["total_bidirectional_data_transfer"].count() / num_repetitions
            << std::endl;

  return 0;
}
