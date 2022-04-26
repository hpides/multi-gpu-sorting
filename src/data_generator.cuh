#pragma once

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <parallel/algorithm>
#include <random>
#include <vector>

class DataGenerator {
 public:
  explicit DataGenerator(uint32_t distribution_seed, int num_threads = kNumThreads)
      : distribution_seed_(distribution_seed), num_threads_(num_threads) {}

  template <typename T>
  void ComputeDistribution(T* begin, size_t num_elements, const std::string& distribution_type) {
    if (distribution_type == "uniform") {
      ComputeUniformDistribution<T>(begin, num_elements);
    } else if (distribution_type == "normal") {
      ComputeNormalDistribution<T>(begin, num_elements);
    } else if (distribution_type == "zero") {
      ComputeZeroDistribution<T>(begin, num_elements);
    } else if (distribution_type == "staggered") {
      ComputeStaggeredDistribution<T>(begin, num_elements);
    } else if (distribution_type == "sorted") {
      ComputeSortedDistribution<T>(begin, num_elements);
    } else if (distribution_type == "reverse-sorted") {
      ComputeReverseSortedDistribution<T>(begin, num_elements);
    } else if (distribution_type == "nearly-sorted") {
      ComputeNearlySortedDistribution<T>(begin, num_elements);
    } else if (distribution_type == "bucket-sorted") {
      ComputeBucketSortedDistribution<T>(begin, num_elements);
    }
  }

 private:
  template <typename T>
  void ComputeUniformDistribution(T* begin, size_t num_elements) {
#pragma omp parallel num_threads(num_threads_)
    {
      std::mt19937 random_generator =
          SeedRandomGenerator(distribution_seed_ + static_cast<size_t>(omp_get_thread_num()));
      std::uniform_real_distribution<double> uniform_dist(0, std::numeric_limits<T>::max());

#pragma omp for schedule(static)
      for (size_t i = 0; i < num_elements; ++i) {
        *(begin + i) = static_cast<T>(uniform_dist(random_generator));
      }
    }
  }

  template <typename T>
  void ComputeNormalDistribution(T* begin, size_t num_elements) {
    const double mean = std::numeric_limits<T>::max() / 2.0;
    const double stddev = mean / 3.0;

#pragma omp parallel num_threads(num_threads_)
    {
      std::mt19937 random_generator =
          SeedRandomGenerator(distribution_seed_ + static_cast<size_t>(omp_get_thread_num()));
      std::normal_distribution<double> normal_dist(mean, stddev);

#pragma omp for schedule(static)
      for (size_t i = 0; i < num_elements; ++i) {
        *(begin + i) = static_cast<T>(std::fabs(normal_dist(random_generator)));
      }
    }
  }

  template <typename T>
  void ComputeZeroDistribution(T* begin, size_t num_elements) {
#pragma omp parallel for num_threads(num_threads_) schedule(static)
    for (size_t i = 0; i < num_elements; ++i) {
      *(begin + i) = 0;
    }
  }

  template <typename T>
  void ComputeStaggeredDistribution(T* begin, size_t num_elements) {
    const size_t num_buckets = 10;
    const size_t bucket_size = num_elements / num_buckets;

#pragma omp parallel num_threads(num_threads_)
    {
      std::mt19937 random_generator =
          SeedRandomGenerator(distribution_seed_ + static_cast<size_t>(omp_get_thread_num()));
      std::uniform_real_distribution<double> rand_range_dist(0, std::numeric_limits<T>::max());

#pragma omp for schedule(static)
      for (size_t i = 0; i < num_elements / bucket_size; ++i) {
        T upper = rand_range_dist(random_generator);
        T lower = num_buckets > upper ? 0 : upper - num_buckets;

        std::uniform_real_distribution<double> uniform_dist(lower, upper);

        for (size_t j = i * bucket_size; j < (i + 1) * bucket_size; ++j) {
          *(begin + j) = static_cast<T>(uniform_dist(random_generator));
        }
      }
    }

    std::mt19937 random_generator = SeedRandomGenerator(distribution_seed_ + num_threads_);
    std::uniform_real_distribution<double> rand_range_dist(0, std::numeric_limits<T>::max());

    T upper = rand_range_dist(random_generator);
    T lower = num_buckets > upper ? 0 : upper - num_buckets;

    std::uniform_real_distribution<double> uniform_dist(lower, upper);

    for (size_t i = num_elements - (num_elements % bucket_size); i < num_elements; ++i) {
      *(begin + i) = static_cast<T>(uniform_dist(random_generator));
    }
  }

  template <typename T>
  void ComputeSortedDistribution(T* begin, size_t num_elements) {
    ComputeUniformDistribution<T>(begin, num_elements);

    __gnu_parallel::sort(begin, begin + num_elements);
  }

  template <typename T>
  void ComputeReverseSortedDistribution(T* begin, size_t num_elements) {
    ComputeUniformDistribution<T>(begin, num_elements);

    __gnu_parallel::sort(begin, begin + num_elements, std::greater<T>());
  }

  template <typename T>
  void ComputeNearlySortedDistribution(T* begin, size_t num_elements) {
    ComputeSortedDistribution<T>(begin, num_elements);

#pragma omp parallel num_threads(num_threads_)
    {
      std::mt19937 random_generator =
          SeedRandomGenerator(distribution_seed_ + static_cast<size_t>(omp_get_thread_num()));

#pragma omp for schedule(static)
      for (size_t i = 0; i < num_elements - 1; ++i) {
        const double mean = 0.0;

        double stddev = *(begin + i + 1) - *(begin + i);

        if (stddev < std::numeric_limits<double>::max() / 2.0) {
          stddev *= 2.0;
        }

        std::normal_distribution<double> normal_dist(mean, stddev);

        T diff = static_cast<T>(std::fabs(normal_dist(random_generator)));

        if (*(begin + i) > std::numeric_limits<T>::max() - diff) {
          *(begin + i) = std::numeric_limits<T>::max();
        } else {
          *(begin + i) += diff;
        }
      }
    }
  }

  template <typename T>
  void ComputeBucketSortedDistribution(T* begin, size_t num_elements) {
    const size_t num_buckets = 10;
    const size_t bucket_size = num_elements / num_buckets;

#pragma omp parallel num_threads(num_threads_)
    {
      std::mt19937 random_generator =
          SeedRandomGenerator(distribution_seed_ + static_cast<size_t>(omp_get_thread_num()));
      std::uniform_real_distribution<double> uniform_dist(0, std::numeric_limits<T>::max());

#pragma omp for schedule(static)
      for (size_t i = 0; i < num_buckets; ++i) {
        for (size_t j = i * bucket_size; j < (i + 1) * bucket_size; ++j) {
          *(begin + j) = static_cast<T>(uniform_dist(random_generator));
        }
      }

#pragma omp for schedule(static)
      for (size_t i = num_elements - num_elements % bucket_size; i < num_elements; ++i) {
        *(begin + i) = static_cast<T>(uniform_dist(random_generator));
      }
    }

#pragma omp parallel for num_threads(num_threads_) schedule(static)
    for (size_t i = 0; i <= num_buckets; ++i) {
      __gnu_parallel::sort(begin + (i * bucket_size), std::min(begin + (i + 1) * bucket_size, begin + num_elements));
    }
  }

  std::mt19937 SeedRandomGenerator(uint32_t distribution_seed) {
    const size_t seeds_bytes = sizeof(std::mt19937::result_type) * std::mt19937::state_size;
    const size_t seeds_length = seeds_bytes / sizeof(std::seed_seq::result_type);

    std::vector<std::seed_seq::result_type> seeds(seeds_length);
    std::generate(seeds.begin(), seeds.end(), [&]() {
      distribution_seed = (distribution_seed << 1) | (distribution_seed >> (-1 & 31));
      return distribution_seed;
    });
    std::seed_seq seed_sequence(seeds.begin(), seeds.end());

    return std::mt19937{seed_sequence};
  }

  const size_t distribution_seed_;
  const int num_threads_;

  inline static constexpr int kNumThreads = 128;
};
