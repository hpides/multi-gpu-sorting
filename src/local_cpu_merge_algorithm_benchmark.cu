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

  if (arguments.size() < 2 || arguments.size() > 3) {
    std::cout << "./local_cpu_merge_algorithm_benchmark <num_elements> <merge_algorithm> (num_threads)\n";
    exit(1);
  }

  const size_t num_elements = std::stoull(arguments[0]);
  const std::string merge_algorithm = arguments[1];
  size_t num_threads;

  if (arguments.size() == 3) {
    num_threads = std::stoull(arguments[2]);

  } else {
    num_threads = ArgumentLimits::GetDefaultNumThreads();
  }

  omp_set_num_threads(num_threads);

  DataGenerator data_generator(ArgumentLimits::GetDefaultDistributionSeed());

  std::vector<int> sorted_elements_1(num_elements);
  std::vector<int> sorted_elements_2(num_elements);
  std::vector<int> merged_elements(2 * num_elements);

  data_generator.ComputeDistribution<int>(&sorted_elements_1[0], sorted_elements_1.size(), "sorted");
  data_generator.ComputeDistribution<int>(&sorted_elements_2[0], sorted_elements_2.size(), "sorted");

  if (merge_algorithm == "gnu_parallel::merge") {
    TimeDurations::Get()->Tic("merge_phase");
    __gnu_parallel::merge(sorted_elements_1.begin(), sorted_elements_1.end(), sorted_elements_2.begin(),
                          sorted_elements_2.end(), merged_elements.begin());
    TimeDurations::Get()->Toc("merge_phase");

  } else if (merge_algorithm == "gnu_parallel::multiway_merge") {
    TimeDurations::Get()->Tic("merge_phase");

    std::vector<std::pair<int*, int*>> iterator_pairs;
    iterator_pairs.reserve(2);
    iterator_pairs.emplace_back(sorted_elements_1.data(), sorted_elements_1.data() + num_elements);
    iterator_pairs.emplace_back(sorted_elements_2.data(), sorted_elements_2.data() + num_elements);

    __gnu_parallel::multiway_merge(iterator_pairs.begin(), iterator_pairs.end(), merged_elements.data(),
                                   num_elements * 2, std::less<int>());

    TimeDurations::Get()->Toc("merge_phase");
  }

  std::cout << num_elements << ",\"" << merge_algorithm << "\"," << num_threads << "," << std::fixed
            << std::setprecision(9) << TimeDurations::Get()->durations["merge_phase"].count() << "\n";

  if (!std::is_sorted(merged_elements.begin(), merged_elements.end())) {
    std::cout << "Error: Invalid sort order.\n";
  }

  return 0;
}
