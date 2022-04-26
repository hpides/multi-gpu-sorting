#pragma once

#include <omp.h>

#include <array>
#include <vector>

#include "host_vector.cuh"

template <typename T>
class HostBuffers {
 public:
  explicit HostBuffers(HostVector<T>* primary_buffer) {
    buffers_[0] = primary_buffer;
    buffers_[1] = new HostVector<T>(primary_buffer->size());
  }

  ~HostBuffers() {
    buffers_[1]->clear();
    buffers_[1]->shrink_to_fit();

    delete buffers_[1];
  }

  T* GetPrimary() { return thrust::raw_pointer_cast(buffers_[0]->data()); }

  T* GetSecondary() { return thrust::raw_pointer_cast(buffers_[1]->data()); }

  size_t GetSize() { return buffers_[0]->size(); }

 private:
  std::array<HostVector<T>*, 2> buffers_;
};
