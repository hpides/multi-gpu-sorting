#include <algorithm>
#include <boost/sort/sort.hpp>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "argument_limits.cuh"
#include "data_generator.cuh"
#include "gowanlock_sort.cuh"
#include "host_vector.cuh"
#include "paradis.h"
#include "tanasic_sort.cuh"
#include "time_durations.cuh"

void PrintUsageAndExit() {
  std::cout << "Usage: ./benchmark <num_elements> [options]\n\n"
            << "<num_elements>\t\t\t\t\tThe number of elements " << ArgumentLimits::GetPrintableNumElementsLimits()
            << "\n\n"
            << "--algorithm\t\t<algorithm>\t\tThe algorithm " << ArgumentLimits::GetPrintableAlgorithmLimits() << "\n"
            << "--gpus\t\t\t<gpus>\t\t\tThe visible GPUs " << ArgumentLimits::GetPrintableGpusLimits() << "\n"
            << "--num_threads\t\t<num_threads>\t\tThe number of threads "
            << ArgumentLimits::GetPrintableNumThreadsLimits() << "\n"
            << "--data_type\t\t<data_type>\t\tThe data type " << ArgumentLimits::GetPrintableDataTypeLimits() << "\n"
            << "--distribution_type\t<distribution_type>\tThe distribution type "
            << ArgumentLimits::GetPrintableDistributionTypeLimits().replace(42, 1, "\n\t\t\t\t\t\t") << "\n"
            << "--distribution_seed\t<distribution_seed>\tThe distribution seed "
            << ArgumentLimits::GetPrintableDistributionSeedLimits() << "\n\n"
            << "[--algorithm gowanlock]"
            << "\n"
            << "--chunk_size\t\t<chunk_size>\t\tThe chunk_size " << ArgumentLimits::GetPrintableChunkSizeLimits()
            << "\n"
            << "--merge_group_size\t<merge_group_size>\tThe merge group size "
            << ArgumentLimits::GetPrintableMergeGroupSizeLimits() << "\n"
            << "--num_buffers\t\t<num_buffers>\t\tThe number of device buffers "
            << ArgumentLimits::GetPrintableNumBuffersLimits() << "\n";
  exit(1);
}

void PrintErrorUsageAndExit(const std::string& error) {
  std::cout << "Error: " << error << "\n\n";

  PrintUsageAndExit();
}

bool IsCommandLineOptionPresent(const std::vector<std::string>& arguments, const std::string& option_name) {
  const auto iterator = std::find(arguments.begin(), arguments.end(), option_name);

  return iterator != arguments.end();
}

bool GetCommandLineOptionValue(const std::vector<std::string>& arguments, const std::string& option_name,
                               std::string* option_value) {
  const auto iterator = std::find(arguments.begin(), arguments.end(), option_name);

  if (iterator != arguments.end() && (iterator + 1) != arguments.end()) {
    *option_value = *(iterator + 1);
    return true;
  }

  return false;
}

template <typename T>
bool GenerateAndSortData(size_t num_elements, const std::string& algorithm, const std::vector<int>& gpus,
                         const std::string& data_type, const std::string& distribution_type, uint32_t distribution_seed,
                         int num_threads, size_t chunk_size, size_t merge_group_size, size_t num_buffers) {
  DataGenerator data_generator(distribution_seed);
  HostVector<T> elements(num_elements);

  data_generator.ComputeDistribution<T>(&elements[0], elements.size(), distribution_type);

  if (algorithm == "tanasic") {
    SetupTanasicSort(gpus);

    TanasicSort<T>(&elements, gpus);
  } else if (algorithm == "gowanlock") {
    SetupGowanlockSort(gpus, num_threads);

    GowanlockSort<T>(&elements, chunk_size, merge_group_size, num_buffers, gpus);
  } else if (algorithm == "thrust") {
    cudaSetDevice(gpus.front());
    cudaDeviceSynchronize();

    DeviceVector<T> dev_vector(num_elements);

    TimeDurations::Get()->Tic("sort_phase");
    thrust::copy(elements.begin(), elements.end(), dev_vector.begin());
    thrust::sort(dev_vector.begin(), dev_vector.end());
    thrust::copy(dev_vector.begin(), dev_vector.end(), elements.begin());
    TimeDurations::Get()->Toc("sort_phase");

  } else if (algorithm == "paradis") {
    TimeDurations::Get()->Tic("sort_phase");
    paradis::sort<T>(elements.data(), elements.data() + num_elements, num_threads);
    TimeDurations::Get()->Toc("sort_phase");

  } else if (algorithm == "boost") {
    TimeDurations::Get()->Tic("sort_phase");
    boost::sort::block_indirect_sort(elements.begin(), elements.end(), num_threads);
    TimeDurations::Get()->Toc("sort_phase");

  } else if (algorithm == "gnu_parallel") {
    omp_set_num_threads(num_threads);

    TimeDurations::Get()->Tic("sort_phase");
    __gnu_parallel::sort(elements.begin(), elements.end());
    TimeDurations::Get()->Toc("sort_phase");
  }

  std::cout << num_elements << ",\"" << algorithm << "\",\"";

  for (size_t i = 0; i < gpus.size(); ++i) {
    std::cout << gpus[i] << ((i < gpus.size() - 1) ? "," : "\"");
  }

  std::cout << ",\"" << data_type << "\",\"" << distribution_type << "\"," << distribution_seed << "," << num_threads
            << "," << chunk_size << "," << merge_group_size << "," << num_buffers << ",";

  std::cout << std::fixed << std::setprecision(9);

  std::cout << TimeDurations::Get()->durations["memory_allocation"].count() << ","
            << TimeDurations::Get()->durations["sort_phase"].count() << ","
            << TimeDurations::Get()->durations["memory_deallocation"].count() << ","
            << TimeDurations::Get()->TotalExecutionTime().count() << "\n";

  return std::is_sorted(elements.begin(), elements.end());
}

int main(int argc, char* argv[]) {
  const std::vector<std::string> arguments(argv + 1, argv + argc);

  if (arguments.empty() || arguments.size() % 2 == 0 || IsCommandLineOptionPresent(arguments, "--help")) {
    PrintUsageAndExit();
  }

  std::string option_value;

  size_t num_elements = ArgumentLimits::GetDefaultNumElements();
  if (!ArgumentLimits::ParseNumElements(arguments.front(), &num_elements)) {
    PrintErrorUsageAndExit("Invalid <num_elements> = " + arguments.front());
  }

  std::string algorithm = ArgumentLimits::GetDefaultAlgorithm();
  if (GetCommandLineOptionValue(arguments, "--algorithm", &option_value)) {
    if (!ArgumentLimits::ParseAlgorithm(option_value, &algorithm)) {
      PrintErrorUsageAndExit("Invalid <algorithm> = " + option_value);
    }
  }

  std::vector<int> gpus = ArgumentLimits::GetDefaultGpus();
  if (GetCommandLineOptionValue(arguments, "--gpus", &option_value)) {
    if (!ArgumentLimits::ParseGpus(option_value, &gpus)) {
      PrintErrorUsageAndExit("Invalid <gpus> = " + option_value);
    }
  }

  std::string data_type = ArgumentLimits::GetDefaultDataType();
  if (GetCommandLineOptionValue(arguments, "--data_type", &option_value)) {
    if (!ArgumentLimits::ParseDataType(option_value, &data_type)) {
      PrintErrorUsageAndExit("Invalid <data_type> = " + option_value);
    }
  }

  std::string distribution_type = ArgumentLimits::GetDefaultDistributionType();
  if (GetCommandLineOptionValue(arguments, "--distribution_type", &option_value)) {
    if (!ArgumentLimits::ParseDistributionType(option_value, &distribution_type)) {
      PrintErrorUsageAndExit("Invalid <distribution_type> = " + option_value);
    }
  }

  uint32_t distribution_seed = ArgumentLimits::GetDefaultDistributionSeed();
  if (GetCommandLineOptionValue(arguments, "--distribution_seed", &option_value)) {
    if (!ArgumentLimits::ParseDistributionSeed(option_value, &distribution_seed)) {
      PrintErrorUsageAndExit("Invalid <distribution_seed> = " + option_value);
    }
  }

  int num_threads = ArgumentLimits::GetDefaultNumThreads();
  if (GetCommandLineOptionValue(arguments, "--num_threads", &option_value)) {
    if (!ArgumentLimits::ParseNumThreads(option_value, &num_threads)) {
      PrintErrorUsageAndExit("Invalid <num_threads> = " + option_value);
    }
  }

  size_t chunk_size = ArgumentLimits::GetDefaultChunkSize();
  if (GetCommandLineOptionValue(arguments, "--chunk_size", &option_value)) {
    if (!ArgumentLimits::ParseChunkSize(option_value, &chunk_size)) {
      PrintErrorUsageAndExit("Invalid <chunk_size> = " + option_value);
    }
  }

  size_t merge_group_size = ArgumentLimits::GetDefaultMergeGroupSize();
  if (GetCommandLineOptionValue(arguments, "--merge_group_size", &option_value)) {
    if (!ArgumentLimits::ParseMergeGroupSize(option_value, &merge_group_size)) {
      PrintErrorUsageAndExit("Invalid <merge_group_size> = " + option_value);
    }
  }

  size_t num_buffers = ArgumentLimits::GetDefaultNumBuffers();
  if (GetCommandLineOptionValue(arguments, "--num_buffers", &option_value)) {
    if (!ArgumentLimits::ParseMergeGroupSize(option_value, &num_buffers)) {
      PrintErrorUsageAndExit("Invalid <num_buffers> = " + option_value);
    }
  }

  bool success = false;

  if (data_type == "int") {
    success = GenerateAndSortData<int>(num_elements, algorithm, gpus, data_type, distribution_type, distribution_seed,
                                       num_threads, chunk_size, merge_group_size, num_buffers);
  } else if (data_type == "long") {
    success = GenerateAndSortData<long>(num_elements, algorithm, gpus, data_type, distribution_type, distribution_seed,
                                        num_threads, chunk_size, merge_group_size, num_buffers);
  } else if (data_type == "float") {
    success = GenerateAndSortData<float>(num_elements, algorithm, gpus, data_type, distribution_type, distribution_seed,
                                         num_threads, chunk_size, merge_group_size, num_buffers);
  } else if (data_type == "double") {
    success = GenerateAndSortData<double>(num_elements, algorithm, gpus, data_type, distribution_type,
                                          distribution_seed, num_threads, chunk_size, merge_group_size, num_buffers);
  }

  if (!success) {
    std::cout << "Error: Invalid sort order.\n";
  }

  return 0;
}
