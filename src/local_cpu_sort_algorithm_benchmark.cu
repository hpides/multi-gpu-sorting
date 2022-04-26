#include <algorithm>
#include <boost/sort/sort.hpp>
#include <iomanip>
#include <iostream>
#include <vector>

#include "argument_limits.cuh"
#include "data_generator.cuh"
#include "paradis.h"
#include "time_durations.cuh"

int main(int argc, char* argv[]) {
  const std::vector<std::string> arguments(argv + 1, argv + argc);

  if (arguments.size() < 2 || arguments.size() > 3) {
    std::cout << "./local_cpu_sort_algorithm_comparison_benchmark <num_elements> <sort_algorithm> (num_threads)\n";
    exit(1);
  }

  const size_t num_elements = std::stoull(arguments[0]);
  const std::string sort_algorithm = arguments[1];
  size_t num_threads;

  if (arguments.size() == 3) {
    num_threads = std::stoull(arguments[2]);

  } else {
    num_threads = ArgumentLimits::GetDefaultNumThreads();
  }

  omp_set_num_threads(num_threads);

  DataGenerator data_generator(ArgumentLimits::GetDefaultDistributionSeed());
  std::vector<int> elements(num_elements);

  data_generator.ComputeDistribution<int>(&elements[0], elements.size(), ArgumentLimits::GetDefaultDistributionType());

  if (sort_algorithm == "gnu_parallel::sort") {
    TimeDurations::Get()->Tic("sort_phase");
    __gnu_parallel::sort(elements.begin(), elements.end());
    TimeDurations::Get()->Toc("sort_phase");

  } else if (sort_algorithm == "boost::sort::block_indirect_sort") {
    TimeDurations::Get()->Tic("sort_phase");
    boost::sort::block_indirect_sort(elements.begin(), elements.end(), num_threads);
    TimeDurations::Get()->Toc("sort_phase");
  } else if (sort_algorithm == "paradis::sort") {
    TimeDurations::Get()->Tic("sort_phase");
    paradis::sort<int>(elements.data(), elements.data() + num_elements, num_threads);
    TimeDurations::Get()->Toc("sort_phase");
  }

  std::cout << num_elements << ",\"" << sort_algorithm << "\"," << num_threads << "," << std::fixed
            << std::setprecision(9) << TimeDurations::Get()->durations["sort_phase"].count() << "\n";

  if (!std::is_sorted(elements.begin(), elements.end())) {
    std::cout << "Error: Invalid sort order.\n";
  }

  return 0;
}
