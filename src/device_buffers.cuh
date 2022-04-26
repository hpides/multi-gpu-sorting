#pragma once

#include <omp.h>

#include <array>
#include <map>
#include <vector>

#include "cuda_error.cuh"
#include "device_allocator.cuh"
#include "device_vector.cuh"
#include "host_allocator.cuh"

template <typename T>
class DeviceBuffers {
 public:
  explicit DeviceBuffers(const std::vector<int>& gpus, size_t buffer_size, size_t num_buffers)
      : buffers_(gpus.size()),
        buffer_flags_(gpus.size(), false),
        device_allocators_(gpus.size()),
        host_allocators_(gpus.size()),
        streams_(gpus.size()),
        partition_count_(gpus.size()),
        partition_size_(buffer_size),
        gpus_(gpus) {
#pragma omp parallel for
    for (size_t i = 0; i < gpus.size(); ++i) {
      CheckCudaError(cudaSetDevice(gpus[i]));

      const size_t add_thrust_overhead = 128000000;

      if (num_buffers == 2) {
        const size_t num_temporary_bytes = buffer_size * sizeof(T) + add_thrust_overhead;
        const size_t adjusted_num_elements = num_temporary_bytes / sizeof(T);

        buffers_[i][0].reserve(adjusted_num_elements);
        buffers_[i][1].reserve(adjusted_num_elements);

        buffers_[i][0].resize(buffer_size);
        buffers_[i][1].resize(buffer_size);

        device_allocators_[i].Malloc(reinterpret_cast<uint8_t*>(thrust::raw_pointer_cast(buffers_[i][1].data())),
                                     reinterpret_cast<uint8_t*>(thrust::raw_pointer_cast(buffers_[i][0].data())),
                                     num_temporary_bytes);
        host_allocators_[i].Malloc(sizeof(size_t));

      } else if (num_buffers == 3) {
        buffers_[i][0].resize(buffer_size);
        buffers_[i][1].resize(buffer_size);

        const size_t num_temporary_bytes = buffer_size * sizeof(T) + add_thrust_overhead;

        device_allocators_[i].Malloc(num_temporary_bytes);
        host_allocators_[i].Malloc(sizeof(size_t));
      }

      CheckCudaError(cudaStreamCreateWithFlags(&streams_[i][0], cudaStreamNonBlocking));
      CheckCudaError(cudaStreamCreateWithFlags(&streams_[i][1], cudaStreamNonBlocking));
      CheckCudaError(cudaStreamCreateWithFlags(&streams_[i][2], cudaStreamNonBlocking));
    }

    for (size_t i = 0; i < gpus.size(); ++i) {
      partitions_[gpus[i]] = static_cast<int>(i);
    }
  }

  DeviceBuffers(const DeviceBuffers& device_buffers) = delete;
  void operator=(const DeviceBuffers& device_buffers) = delete;

  ~DeviceBuffers() {
#pragma omp parallel for
    for (size_t i = 0; i < gpus_.size(); ++i) {
      CheckCudaError(cudaSetDevice(gpus_[i]));

      buffers_[i][0].clear();
      buffers_[i][0].shrink_to_fit();
      buffers_[i][1].clear();
      buffers_[i][1].shrink_to_fit();

      device_allocators_[i].Free();
      host_allocators_[i].Free();

      CheckCudaError(cudaStreamDestroy(streams_[i][0]));
      CheckCudaError(cudaStreamDestroy(streams_[i][1]));
      CheckCudaError(cudaStreamDestroy(streams_[i][2]));
    }
  }

  DeviceVector<T>* AtPrimary(int i) { return &(buffers_[partitions_[i]][buffer_flags_[partitions_[i]]]); }

  DeviceVector<T>* AtSecondary(int i) { return &(buffers_[partitions_[i]][!buffer_flags_[partitions_[i]]]); }

  void Flip(int i) { buffer_flags_[partitions_[i]] = !buffer_flags_[partitions_[i]]; }

  void Resize(size_t buffer_size) {
#pragma omp parallel for
    for (size_t i = 0; i < gpus_.size(); ++i) {
      buffers_[i][0].resize(buffer_size);
      buffers_[i][1].resize(buffer_size);
    }

    partition_size_ = buffer_size;
  }

  DeviceAllocator* GetDeviceAllocator(int i) { return &device_allocators_[partitions_[i]]; }

  HostAllocator* GetHostAllocator(int i) { return &host_allocators_[partitions_[i]]; }

  cudaStream_t* GetPrimaryStream(int i) { return &streams_[partitions_[i]][0]; }

  cudaStream_t* GetSecondaryStream(int i) { return &streams_[partitions_[i]][1]; }

  cudaStream_t* GetTemporaryStream(int i) { return &streams_[partitions_[i]][2]; }

  int GetPartition(int i) { return partitions_[i]; }

  size_t GetPartitionCount() const { return partition_count_; }

  size_t GetPartitionSize() const { return partition_size_; }

 private:
  std::vector<std::array<DeviceVector<T>, 2>> buffers_;
  std::vector<bool> buffer_flags_;

  std::vector<DeviceAllocator> device_allocators_;
  std::vector<HostAllocator> host_allocators_;

  std::vector<std::array<cudaStream_t, 3>> streams_;

  std::map<int, int> partitions_;
  size_t partition_count_;
  size_t partition_size_;

  std::vector<int> gpus_;
};
