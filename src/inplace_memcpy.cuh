#pragma once

#include <algorithm>

#include "cuda_error.cuh"

void InplaceMemcpy(void* htod_source, void* dtoh_source, void* dtoh_dest, size_t num_bytes_htod, size_t num_bytes_dtoh,
                   cudaStream_t htod_stream, cudaStream_t dtoh_stream, size_t block_size) {
  if (dtoh_dest == nullptr && htod_source == nullptr) {
    return;
  }

  size_t num_bytes;

  if (dtoh_dest == nullptr) {
    num_bytes = num_bytes_htod;
    block_size = num_bytes;

  } else if (htod_source == nullptr) {
    num_bytes = num_bytes_dtoh;
    block_size = num_bytes;

  } else {
    num_bytes = std::min(num_bytes_htod, num_bytes_dtoh);
    block_size = std::min(block_size, num_bytes);
  }

  size_t bytes_dtoh = 0;

  if (dtoh_dest != nullptr) {
    CheckCudaError(cudaMemcpyAsync(dtoh_dest, dtoh_source, block_size, cudaMemcpyDeviceToHost, dtoh_stream));
    CheckCudaError(cudaStreamSynchronize(dtoh_stream));
  }
  bytes_dtoh += block_size;

  while (bytes_dtoh < num_bytes && dtoh_dest != nullptr && htod_source != nullptr) {
    CheckCudaError(cudaMemcpyAsync(dtoh_source + bytes_dtoh - block_size, htod_source + bytes_dtoh - block_size,
                                   block_size, cudaMemcpyHostToDevice, htod_stream));

    block_size = std::min(block_size, num_bytes - bytes_dtoh);

    CheckCudaError(cudaMemcpyAsync(dtoh_dest + bytes_dtoh, dtoh_source + bytes_dtoh, block_size, cudaMemcpyDeviceToHost,
                                   dtoh_stream));

    CheckCudaError(cudaStreamSynchronize(htod_stream));
    CheckCudaError(cudaStreamSynchronize(dtoh_stream));

    bytes_dtoh += block_size;
  }

  if (htod_source != nullptr) {
    CheckCudaError(cudaMemcpyAsync(dtoh_source + bytes_dtoh - block_size, htod_source + bytes_dtoh - block_size,
                                   block_size, cudaMemcpyHostToDevice, htod_stream));
    CheckCudaError(cudaStreamSynchronize(htod_stream));
  }

  if (num_bytes_htod != num_bytes_dtoh && dtoh_dest != nullptr && htod_source != nullptr) {
    if (num_bytes_htod > num_bytes_dtoh) {
      CheckCudaError(cudaMemcpyAsync(dtoh_source + num_bytes, htod_source + num_bytes, num_bytes_htod - num_bytes,
                                     cudaMemcpyHostToDevice, htod_stream));
      CheckCudaError(cudaStreamSynchronize(htod_stream));

    } else if (num_bytes_dtoh > num_bytes_htod) {
      CheckCudaError(cudaMemcpyAsync(dtoh_dest + num_bytes, dtoh_source + num_bytes, num_bytes_dtoh - num_bytes,
                                     cudaMemcpyDeviceToHost, dtoh_stream));
      CheckCudaError(cudaStreamSynchronize(dtoh_stream));
    }
  }
}
