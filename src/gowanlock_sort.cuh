#pragma once

#include <omp.h>
#include <thrust/sort.h>

#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <parallel/algorithm>
#include <thread>
#include <vector>

#include "cuda_error.cuh"
#include "device_buffers.cuh"
#include "host_buffers.cuh"
#include "inplace_memcpy.cuh"
#include "time_durations.cuh"

struct SyncState {
  std::mutex mutex;

  std::condition_variable new_sorted_chunks;

  size_t num_sorted_chunks = 0;
  std::atomic<size_t> num_merged_chunks = 0;
};

template <typename T>
void MergeChunks(HostBuffers<T>* host_buffers, SyncState* sync_state, size_t chunk_size, size_t last_chunk_size,
                 size_t num_default_sized_chunks, size_t num_chunks_to_merge, size_t merge_group_size,
                 std::vector<std::pair<T*, T*>>* final_pointer_pairs) {
  size_t current_chunk_offset = 0;
  size_t merge_chunk_size = chunk_size;

  while (sync_state->num_merged_chunks < num_chunks_to_merge) {
    std::unique_lock<std::mutex> lock(sync_state->mutex);

    sync_state->new_sorted_chunks.wait(
        lock, [&] { return sync_state->num_sorted_chunks - sync_state->num_merged_chunks >= merge_group_size; });

    sync_state->num_merged_chunks += merge_group_size;

    lock.unlock();

    cudaEvent_t merge_chunks_event;

    if (merge_group_size > 1) {
      cudaEventCreate(&merge_chunks_event);
    }

    std::vector<std::pair<T*, T*>> pointer_pairs;
    pointer_pairs.reserve(merge_group_size);

    size_t num_elements_in_current_merge_group = 0;
    for (size_t i = 0; i < merge_group_size; ++i) {
      if (sync_state->num_merged_chunks - merge_group_size + i >= num_default_sized_chunks) {
        merge_chunk_size = last_chunk_size;
      }

      pointer_pairs.emplace_back(host_buffers->GetPrimary() + current_chunk_offset + (merge_chunk_size * i),
                                 host_buffers->GetPrimary() + current_chunk_offset + (merge_chunk_size * (i + 1)));

      num_elements_in_current_merge_group += merge_chunk_size;
    }

    const size_t merge_position = (sync_state->num_merged_chunks / merge_group_size) - 1;
    (*final_pointer_pairs)[merge_position] = {
        host_buffers->GetSecondary() + current_chunk_offset,
        host_buffers->GetSecondary() + current_chunk_offset + num_elements_in_current_merge_group};

    if (merge_group_size > 1) {
      __gnu_parallel::multiway_merge(
          pointer_pairs.begin(), pointer_pairs.end(), (*final_pointer_pairs)[merge_position].first,
          (*final_pointer_pairs)[merge_position].second - (*final_pointer_pairs)[merge_position].first, std::less<T>());
    }

    current_chunk_offset += num_elements_in_current_merge_group;

    if (merge_group_size > 1) {
      cudaEventDestroy(merge_chunks_event);
    }
  }
}

template <typename T>
void MergeMergeGroups(HostBuffers<T>* host_buffers, std::vector<std::pair<T*, T*>>* final_pointer_pairs) {
  cudaEvent_t final_merge_event;
  cudaEventCreate(&final_merge_event);

  __gnu_parallel::multiway_merge(final_pointer_pairs->begin(), final_pointer_pairs->end(), host_buffers->GetPrimary(),
                                 host_buffers->GetSize(), std::less<T>());

  cudaEventDestroy(final_merge_event);
}

inline std::array<size_t, 3> CalculateLastBufferParameters(size_t num_elements, size_t num_elements_per_chunk_group,
                                                           size_t num_gpus) {
  size_t num_fillers = ((num_elements % num_elements_per_chunk_group) % num_gpus != 0)
                           ? (num_gpus - ((num_elements % num_elements_per_chunk_group) % num_gpus))
                           : 0;

  size_t chunk_size = ((num_elements % num_elements_per_chunk_group) + num_fillers) / num_gpus;

  while (chunk_size < num_fillers) {
    num_gpus /= 2;
    num_fillers = ((num_elements % num_elements_per_chunk_group) % num_gpus != 0)
                      ? (num_gpus - ((num_elements % num_elements_per_chunk_group) % num_gpus))
                      : 0;

    chunk_size = ((num_elements % num_elements_per_chunk_group) + num_fillers) / num_gpus;
  }

  return {num_gpus, num_fillers, chunk_size};
}

void SetupGowanlockSort(const std::vector<int>& gpus, int num_threads) {
  omp_set_num_threads(num_threads);

#pragma omp parallel for num_threads(gpus.size()) schedule(static)
  for (size_t i = 0; i < gpus.size(); ++i) {
    CheckCudaError(cudaSetDevice(gpus[i]));
    CheckCudaError(cudaDeviceSynchronize());
  }
}

template <typename T>
void GowanlockCore2n(HostBuffers<T>* host_buffers, DeviceBuffers<T>* device_buffers,
                     std::vector<std::pair<T*, T*>>* final_pointer_pairs, SyncState* sync_state, size_t chunk_size,
                     size_t last_chunk_size, size_t num_fillers, size_t last_num_fillers, size_t num_chunk_groups,
                     size_t num_elements_per_chunk_group, size_t merge_group_size, size_t num_chunks_to_merge,
                     size_t num_merge_groups_to_merge, size_t max_num_gpus, size_t last_num_gpus,
                     const std::vector<int>& gpus) {
  std::atomic<size_t> last_chunk_merge_counter = 0;
  size_t current_num_gpus = max_num_gpus;

  for (size_t c = 0; c < num_chunk_groups; c++) {
    if (c == num_chunk_groups - 1 && host_buffers->GetSize() % num_elements_per_chunk_group != 0) {
      chunk_size = last_chunk_size;
      num_fillers = last_num_fillers;

      current_num_gpus = last_num_gpus;

      device_buffers->Resize(chunk_size);
    }

#pragma omp parallel for num_threads(max_num_gpus) schedule(static)
    for (size_t i = 0; i < max_num_gpus; ++i) {
      bool is_last_chunk = (i == current_num_gpus - 1);

      auto sync_dtoh_stream_future = std::async(std::launch::async, [&] {
        if (c > 0) {
          CheckCudaError(cudaSetDevice(gpus[i]));

          CheckCudaError(cudaStreamSynchronize(*device_buffers->GetSecondaryStream(gpus[i])));

          std::unique_lock<std::mutex> lock(sync_state->mutex);
          sync_state->num_sorted_chunks++;
          if (sync_state->num_sorted_chunks - sync_state->num_merged_chunks >= merge_group_size) {
            lock.unlock();
            sync_state->new_sorted_chunks.notify_one();
          } else {
            lock.unlock();
          }
        }
      });

      if (i < current_num_gpus) {
        CheckCudaError(cudaSetDevice(gpus[i]));

        CheckCudaError(
            cudaMemcpyAsync(thrust::raw_pointer_cast(device_buffers->AtPrimary(gpus[i])->data()),
                            host_buffers->GetPrimary() + (num_elements_per_chunk_group * c) + (chunk_size * i),
                            (sizeof(T) * chunk_size) - (is_last_chunk ? sizeof(T) * num_fillers : 0),
                            cudaMemcpyHostToDevice, *device_buffers->GetPrimaryStream(gpus[i])));

        CheckCudaError(cudaStreamSynchronize(*device_buffers->GetSecondaryStream(gpus[i])));

        if (is_last_chunk && num_fillers > 0) {
          thrust::fill(thrust::cuda::par(*device_buffers->GetDeviceAllocator(gpus[i]))
                           .on(*device_buffers->GetPrimaryStream(gpus[i])),
                       device_buffers->AtPrimary(gpus[i])->end() - num_fillers,
                       device_buffers->AtPrimary(gpus[i])->end(), std::numeric_limits<T>::max());
        }

        CheckCudaError(cudaStreamSynchronize(*device_buffers->GetPrimaryStream(gpus[i])));

        thrust::sort(thrust::cuda::par(*device_buffers->GetDeviceAllocator(gpus[i]))
                         .on(*device_buffers->GetPrimaryStream(gpus[i])),
                     device_buffers->AtPrimary(gpus[i])->begin(), device_buffers->AtPrimary(gpus[i])->end());

        CheckCudaError(cudaStreamSynchronize(*device_buffers->GetPrimaryStream(gpus[i])));

        sync_dtoh_stream_future.get();

        T* start_pointer = host_buffers->GetPrimary() + num_elements_per_chunk_group * c;

        if (c * max_num_gpus + i >= num_chunks_to_merge) {
          if (final_pointer_pairs->size() > 1) {
            start_pointer = host_buffers->GetSecondary() + num_elements_per_chunk_group * c;
          }

          (*final_pointer_pairs)[num_merge_groups_to_merge + last_chunk_merge_counter++] = {
              start_pointer + (chunk_size * i),
              start_pointer + (chunk_size * (i + 1) - (is_last_chunk ? num_fillers : 0))};
        }

        CheckCudaError(cudaMemcpyAsync(start_pointer + (chunk_size * i),
                                       thrust::raw_pointer_cast(device_buffers->AtPrimary(gpus[i])->data()),
                                       (sizeof(T) * chunk_size) - (is_last_chunk ? sizeof(T) * num_fillers : 0),
                                       cudaMemcpyDeviceToHost, *device_buffers->GetSecondaryStream(gpus[i])));

        device_buffers->Flip(gpus[i]);
        device_buffers->GetDeviceAllocator(gpus[i])->Flip();
      } else {
        sync_dtoh_stream_future.get();
      }
    }
  }

#pragma omp parallel for num_threads(current_num_gpus) schedule(static)
  for (size_t i = 0; i < current_num_gpus; ++i) {
    CheckCudaError(cudaStreamSynchronize(*device_buffers->GetSecondaryStream(gpus[i])));
  }

  std::unique_lock<std::mutex> lock(sync_state->mutex);
  sync_state->num_sorted_chunks += current_num_gpus;
  lock.unlock();
  sync_state->new_sorted_chunks.notify_one();
}

template <typename T>
void GowanlockCore3n(HostBuffers<T>* host_buffers, DeviceBuffers<T>* device_buffers,
                     std::vector<std::pair<T*, T*>>* final_pointer_pairs, SyncState* sync_state, size_t chunk_size,
                     size_t last_chunk_size, size_t num_fillers, size_t last_num_fillers, size_t num_chunk_groups,
                     size_t num_elements_per_chunk_group, size_t merge_group_size, size_t num_chunks_to_merge,
                     size_t num_merge_groups_to_merge, size_t max_num_gpus, size_t last_num_gpus,
                     const std::vector<int>& gpus) {
  std::atomic<size_t> last_chunk_merge_counter = 0;

#pragma omp parallel for num_threads(max_num_gpus) schedule(static)
  for (size_t i = 0; i < max_num_gpus; ++i) {
    CheckCudaError(cudaSetDevice(gpus[i]));

    size_t htod_chunk_size = chunk_size;
    size_t htod_num_fillers = num_fillers;
    size_t htod_num_gpus = max_num_gpus;

    if (num_chunk_groups == 1 && host_buffers->GetSize() % num_elements_per_chunk_group != 0) {
      htod_chunk_size = last_chunk_size;
      htod_num_fillers = last_num_fillers;
      htod_num_gpus = last_num_gpus;
    }

    if (i < htod_num_gpus) {
      bool is_last_chunk = (i == htod_num_gpus - 1);
      CheckCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(device_buffers->AtPrimary(gpus[i])->data()),
                                     host_buffers->GetPrimary() + (htod_chunk_size * i),
                                     (sizeof(T) * htod_chunk_size) - (is_last_chunk ? sizeof(T) * htod_num_fillers : 0),
                                     cudaMemcpyHostToDevice, *device_buffers->GetPrimaryStream(gpus[i])));

      CheckCudaError(cudaStreamSynchronize(*device_buffers->GetPrimaryStream(gpus[i])));

      device_buffers->Flip(gpus[i]);
    }
  }

  for (size_t c = 0; c < num_chunk_groups; c++) {
    size_t htod_chunk_size = chunk_size;
    size_t htod_num_fillers = num_fillers;
    size_t htod_num_gpus = max_num_gpus;

    bool is_last_htod_cg = c + 1 == num_chunk_groups - 1;
    if (is_last_htod_cg && host_buffers->GetSize() % num_elements_per_chunk_group != 0) {
      htod_chunk_size = last_chunk_size;
      htod_num_fillers = last_num_fillers;
      htod_num_gpus = last_num_gpus;
    }

    size_t sort_chunk_size = chunk_size;
    size_t sort_num_fillers = num_fillers;
    size_t sort_num_gpus = max_num_gpus;

    bool is_last_cg_to_sort = c == num_chunk_groups - 1;
    if (is_last_cg_to_sort && host_buffers->GetSize() % num_elements_per_chunk_group != 0) {
      sort_chunk_size = last_chunk_size;
      sort_num_fillers = last_num_fillers;
      sort_num_gpus = last_num_gpus;
    }

#pragma omp parallel for num_threads(max_num_gpus) schedule(static)
    for (size_t i = 0; i < max_num_gpus; ++i) {
      auto sync_inplace_transfer_future = std::async(std::launch::async, [&] {
        CheckCudaError(cudaSetDevice(gpus[i]));

        T* start_pointer = nullptr;
        if (c > 0) {
          start_pointer = host_buffers->GetPrimary() + num_elements_per_chunk_group * (c - 1);

          if ((c - 1) * max_num_gpus + i >= num_chunks_to_merge) {
            if (final_pointer_pairs->size() > 1) {
              start_pointer = host_buffers->GetSecondary() + num_elements_per_chunk_group * (c - 1);
            }

            (*final_pointer_pairs)[num_merge_groups_to_merge + last_chunk_merge_counter++] = {
                start_pointer + (chunk_size * i),
                start_pointer + (chunk_size * (i + 1) - ((i == max_num_gpus - 1) ? num_fillers : 0))};
          }
        }

        size_t dtoh_num_bytes = (sizeof(T) * chunk_size) - ((i == max_num_gpus - 1) ? sizeof(T) * num_fillers : 0);
        size_t htod_num_bytes =
            (sizeof(T) * htod_chunk_size) - ((i == htod_num_gpus - 1) ? sizeof(T) * htod_num_fillers : 0);

        InplaceMemcpy(
            is_last_cg_to_sort || i >= htod_num_gpus
                ? nullptr
                : host_buffers->GetPrimary() + (num_elements_per_chunk_group * (c + 1)) + (htod_chunk_size * i),
            thrust::raw_pointer_cast(device_buffers->AtPrimary(gpus[i])->data()),
            c > 0 ? start_pointer + (chunk_size * i) : nullptr, htod_num_bytes, dtoh_num_bytes,
            *device_buffers->GetPrimaryStream(gpus[i]), *device_buffers->GetSecondaryStream(gpus[i]),
            std::max(1UL, htod_num_bytes / 100));
      });

      CheckCudaError(cudaSetDevice(gpus[i]));

      if (i < sort_num_gpus) {
        if (i == sort_num_gpus - 1 && sort_num_fillers > 0) {
          thrust::fill(thrust::cuda::par(*device_buffers->GetDeviceAllocator(gpus[i]))
                           .on(*device_buffers->GetTemporaryStream(gpus[i])),
                       device_buffers->AtSecondary(gpus[i])->begin() + sort_chunk_size - sort_num_fillers,
                       device_buffers->AtSecondary(gpus[i])->begin() + sort_chunk_size, std::numeric_limits<T>::max());
        }

        CheckCudaError(cudaStreamSynchronize(*device_buffers->GetTemporaryStream(gpus[i])));

        thrust::sort(thrust::cuda::par(*device_buffers->GetDeviceAllocator(gpus[i]))
                         .on(*device_buffers->GetTemporaryStream(gpus[i])),
                     device_buffers->AtSecondary(gpus[i])->begin(),
                     device_buffers->AtSecondary(gpus[i])->begin() + sort_chunk_size);

        CheckCudaError(cudaStreamSynchronize(*device_buffers->GetTemporaryStream(gpus[i])));
      }

      sync_inplace_transfer_future.get();
      device_buffers->Flip(gpus[i]);

      if (c > 0) {
        std::unique_lock<std::mutex> lock(sync_state->mutex);
        sync_state->num_sorted_chunks++;
        if (sync_state->num_sorted_chunks - sync_state->num_merged_chunks >= merge_group_size) {
          lock.unlock();
          sync_state->new_sorted_chunks.notify_one();
        } else {
          lock.unlock();
        }
      }
    }
  }

  size_t c = num_chunk_groups - 1;
#pragma omp parallel for num_threads(last_num_gpus) schedule(static)
  for (size_t i = 0; i < last_num_gpus; ++i) {
    bool is_last_chunk = (i == last_num_gpus - 1);

    CheckCudaError(cudaSetDevice(gpus[i]));

    T* start_pointer = host_buffers->GetPrimary() + num_elements_per_chunk_group * c;

    if (c * max_num_gpus + i >= num_chunks_to_merge) {
      if (final_pointer_pairs->size() > 1) {
        start_pointer = host_buffers->GetSecondary() + num_elements_per_chunk_group * c;
      }

      (*final_pointer_pairs)[num_merge_groups_to_merge + last_chunk_merge_counter++] = {
          start_pointer + (last_chunk_size * i),
          start_pointer + (last_chunk_size * (i + 1) - (is_last_chunk ? last_num_fillers : 0))};
    }

    size_t num_bytes = (sizeof(T) * last_chunk_size) - (is_last_chunk ? sizeof(T) * last_num_fillers : 0);

    CheckCudaError(cudaMemcpyAsync(start_pointer + (last_chunk_size * i),
                                   thrust::raw_pointer_cast(device_buffers->AtPrimary(gpus[i])->data()), num_bytes,
                                   cudaMemcpyDeviceToHost, *device_buffers->GetSecondaryStream(gpus[i])));

    CheckCudaError(cudaStreamSynchronize(*device_buffers->GetSecondaryStream(gpus[i])));
  }

  std::unique_lock<std::mutex> lock(sync_state->mutex);
  sync_state->num_sorted_chunks += last_num_gpus;
  lock.unlock();
  sync_state->new_sorted_chunks.notify_one();
}

template <typename T>
void GowanlockSort(HostVector<T>* elements, size_t chunk_size, size_t merge_group_size, size_t num_buffers,
                   const std::vector<int>& gpus) {
  if (elements->empty()) {
    return;
  }

  if (chunk_size == 0) {
    chunk_size = elements->size() / gpus.size() + ((elements->size() % gpus.size() != 0) ? 1 : 0);
  }

  if (merge_group_size == 0) {
    merge_group_size = gpus.size();
  }

  const size_t num_elements_per_chunk_group = chunk_size * gpus.size();

  const size_t num_chunk_groups =
      elements->size() / num_elements_per_chunk_group + (elements->size() % num_elements_per_chunk_group != 0 ? 1 : 0);

  size_t num_fillers = 0;

  std::array<size_t, 3> buffer_parameters =
      CalculateLastBufferParameters(elements->size(), num_elements_per_chunk_group, gpus.size());
  size_t last_num_gpus = buffer_parameters[0];
  size_t last_num_fillers = buffer_parameters[1];
  size_t last_chunk_size = buffer_parameters[2];

  if (elements->size() % num_elements_per_chunk_group == 0) {
    last_chunk_size = chunk_size;
    last_num_fillers = num_fillers;
  }

  SyncState sync_state;

  const size_t max_num_gpus = gpus.size();

  size_t num_default_sized_chunks = (num_chunk_groups - 1) * gpus.size();
  size_t total_num_chunks = num_default_sized_chunks + last_num_gpus;

  if (merge_group_size > total_num_chunks) {
    merge_group_size = 1;
  }

  size_t num_merge_groups_to_merge =
      total_num_chunks / merge_group_size - (total_num_chunks % merge_group_size == 0 && merge_group_size > 1 ? 1 : 0);

  size_t num_chunks_to_merge = num_merge_groups_to_merge * merge_group_size;

  std::vector<std::pair<T*, T*>> final_pointer_pairs(num_merge_groups_to_merge +
                                                     (total_num_chunks - num_chunks_to_merge));

  TimeDurations::Get()->Tic("memory_allocation");

  DeviceBuffers<T>* device_buffers = new DeviceBuffers<T>(gpus, chunk_size, num_buffers);
  HostBuffers<T>* host_buffers = new HostBuffers(elements);

  TimeDurations::Get()->Toc("memory_allocation");

  TimeDurations::Get()->Tic("sort_phase");

  auto merge_future =
      std::async(std::launch::async, MergeChunks<T>, host_buffers, &sync_state, chunk_size, last_chunk_size,
                 num_default_sized_chunks, num_chunks_to_merge, merge_group_size, &final_pointer_pairs);

  if (num_buffers == 2) {
    GowanlockCore2n<T>(host_buffers, device_buffers, &final_pointer_pairs, &sync_state, chunk_size, last_chunk_size,
                       num_fillers, last_num_fillers, num_chunk_groups, num_elements_per_chunk_group, merge_group_size,
                       num_chunks_to_merge, num_merge_groups_to_merge, max_num_gpus, last_num_gpus, gpus);

  } else if (num_buffers == 3) {
    GowanlockCore3n<T>(host_buffers, device_buffers, &final_pointer_pairs, &sync_state, chunk_size, last_chunk_size,
                       num_fillers, last_num_fillers, num_chunk_groups, num_elements_per_chunk_group, merge_group_size,
                       num_chunks_to_merge, num_merge_groups_to_merge, max_num_gpus, last_num_gpus, gpus);
  }

  merge_future.get();

  if (final_pointer_pairs.size() > 1) {
    MergeMergeGroups<T>(host_buffers, &final_pointer_pairs);
  }

  TimeDurations::Get()->Toc("sort_phase");

  TimeDurations::Get()->Tic("memory_deallocation");
  delete device_buffers;
  delete host_buffers;
  TimeDurations::Get()->Toc("memory_deallocation");
}
