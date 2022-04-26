#include <omp.h>
#include <thrust/copy.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

#include "argument_limits.cuh"
#include "data_generator.cuh"
#include "time_durations.cuh"

int main(int argc, char* argv[]) {
  const std::vector<std::string> arguments(argv + 1, argv + argc);

  if (arguments.size() < 1 || arguments.size() > 2) {
    std::cout << "./local_cpu_multiway_merge_benchmark <sorted_sublist_lengths> (num_threads)\n";
    exit(1);
  }

  std::vector<size_t> sorted_sublist_lengths;
  size_t total_num_elements = 0;
  std::string sublists_string = arguments[0];
  size_t from = 0;
  size_t found = 0;
  while ((found = sublists_string.find(",", from)) != std::string::npos) {
    size_t num_elements = std::stoull(sublists_string.substr(from, found - from));
    sorted_sublist_lengths.emplace_back(num_elements);
    total_num_elements += num_elements;
    from = found + 1;
  }
  size_t num_elements = std::stoull(sublists_string.substr(from));
  sorted_sublist_lengths.emplace_back(num_elements);
  total_num_elements += num_elements;

  const size_t n = sorted_sublist_lengths.size();
  size_t num_threads;

  if (arguments.size() == 2) {
    num_threads = std::stoull(arguments[1]);

  } else {
    num_threads = ArgumentLimits::GetDefaultNumThreads();
  }

  omp_set_num_threads(num_threads);

  DataGenerator data_generator(ArgumentLimits::GetDefaultDistributionSeed());

  std::vector<std::vector<int>> sorted_elements(n);
  for (size_t i = 0; i < n; ++i) {
    sorted_elements[i].resize(sorted_sublist_lengths[i]);
  }

  for (size_t i = 0; i < n; i++) {
    data_generator.ComputeDistribution<int>(&sorted_elements[i][0], sorted_elements[i].size(), "sorted");
  }

  std::vector<int> merged_elements(total_num_elements);

  std::vector<std::pair<int*, int*>> iterator_pairs;
  iterator_pairs.reserve(n);

  for (size_t i = 0; i < n; i++) {
    iterator_pairs.emplace_back(sorted_elements[i].data(), sorted_elements[i].data() + sorted_sublist_lengths[i]);
  }

  TimeDurations::Get()->Tic("merge_phase");

  __gnu_parallel::multiway_merge(iterator_pairs.begin(), iterator_pairs.end(), merged_elements.data(),
                                 total_num_elements, std::less<int>());

  TimeDurations::Get()->Toc("merge_phase");

  std::cout << "\"";
  for (int i = 0; i < n; i++) {
    std::cout << sorted_sublist_lengths[i] << (i < n - 1 ? "," : "\",");
  }
  std::cout << num_threads << "," << std::fixed << std::setprecision(9)
            << TimeDurations::Get()->durations["merge_phase"].count() << "\n";

  if (!std::is_sorted(merged_elements.begin(), merged_elements.end())) {
    std::cout << "Error: Invalid sort order.\n";
  }

  return 0;
}
