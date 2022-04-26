#pragma once

#include <new>

#include "cuda_error.cuh"

struct HostAllocator {
  using value_type = uint8_t;

  void Malloc(size_t num_bytes) {
    CheckCudaError(
        cudaMallocHost(reinterpret_cast<void**>(&begin), sizeof(value_type) * num_bytes, cudaHostAllocMapped));

    capacity = sizeof(value_type) * num_bytes;
  }

  void Free() { CheckCudaError(cudaFreeHost(begin)); }

  value_type* allocate(size_t num_bytes) {
    if (num_bytes > capacity) {
      throw std::bad_alloc();
    }

    return begin;
  }

  void deallocate(value_type* current, size_t num_bytes) {}

  value_type* begin = nullptr;
  size_t capacity = 0;
};
