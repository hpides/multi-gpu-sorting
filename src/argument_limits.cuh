#pragma once

#include <omp.h>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "cuda_error.cuh"

class ArgumentLimits {
 public:
  ArgumentLimits() = delete;

  static bool ParseNumElements(const std::string& num_elements_to_parse, size_t* num_elements) {
    if (!std::all_of(num_elements_to_parse.begin(), num_elements_to_parse.end(),
                     [](unsigned char c) { return std::isdigit(c); })) {
      return false;
    }

    *num_elements = std::stoull(num_elements_to_parse);
    return IsValidNumElements(*num_elements);
  }

  static bool ParseAlgorithm(const std::string& algorithm_to_parse, std::string* algorithm) {
    *algorithm = algorithm_to_parse;
    return IsValidAlgorithm(*algorithm);
  }

  static bool ParseGpus(const std::string& gpus_to_parse, std::vector<int>* gpus) {
    std::string trimmed_gpus = gpus_to_parse;
    trimmed_gpus.erase(
        std::remove_if(trimmed_gpus.begin(), trimmed_gpus.end(), [](unsigned char c) { return std::isspace(c); }),
        trimmed_gpus.end());

    if (trimmed_gpus.empty()) {
      return false;
    }

    if (trimmed_gpus.front() == '{' || trimmed_gpus.back() == '}') {
      if (trimmed_gpus.front() == '{' != trimmed_gpus.back() == '}') {
        return false;
      }
      trimmed_gpus = {trimmed_gpus.begin() + 1, trimmed_gpus.end() - 1};
    }

    if (trimmed_gpus.empty()) {
      return false;
    }

    if (!std::all_of(trimmed_gpus.begin(), trimmed_gpus.end(),
                     [](unsigned char c) { return std::isdigit(c) || c == ','; })) {
      return false;
    }

    if (trimmed_gpus.find(",,") != std::string::npos || trimmed_gpus.front() == ',' || trimmed_gpus.back() == ',') {
      return false;
    }

    gpus->clear();
    std::string trimmed_gpu;
    std::stringstream trimmed_stream(trimmed_gpus);
    while (std::getline(trimmed_stream, trimmed_gpu, ',')) {
      gpus->emplace_back(std::stoi(trimmed_gpu));
    }

    return IsValidGpus(*gpus);
  }

  static bool ParseDataType(const std::string& data_type_to_parse, std::string* data_type) {
    *data_type = data_type_to_parse;
    return IsValidDataType(*data_type);
  }

  static bool ParseDistributionType(const std::string& distribution_type_to_parse, std::string* distribution_type) {
    *distribution_type = distribution_type_to_parse;
    return IsValidDistributionType(*distribution_type);
  }

  static bool ParseDistributionSeed(const std::string& distribution_seed_to_parse, uint32_t* distribution_seed) {
    if (!std::all_of(distribution_seed_to_parse.begin(), distribution_seed_to_parse.end(),
                     [](unsigned char c) { return std::isdigit(c); })) {
      return false;
    }

    *distribution_seed = std::stoull(distribution_seed_to_parse);
    return IsValidDistributionSeed(*distribution_seed);
  }

  static bool ParseNumThreads(const std::string& num_threads_to_parse, int* num_threads) {
    if (!std::all_of(num_threads_to_parse.begin(), num_threads_to_parse.end(),
                     [](unsigned char c) { return std::isdigit(c); })) {
      return false;
    }

    *num_threads = std::stoi(num_threads_to_parse);
    return IsValidNumThreads(*num_threads);
  }

  static bool ParseChunkSize(const std::string& chunk_size_to_parse, size_t* chunk_size) {
    if (!std::all_of(chunk_size_to_parse.begin(), chunk_size_to_parse.end(),
                     [](unsigned char c) { return std::isdigit(c); })) {
      return false;
    }

    *chunk_size = std::stoull(chunk_size_to_parse);
    return IsValidChunkSize(*chunk_size);
  }

  static bool ParseMergeGroupSize(const std::string& merge_group_size_to_parse, size_t* merge_group_size) {
    if (!std::all_of(merge_group_size_to_parse.begin(), merge_group_size_to_parse.end(),
                     [](unsigned char c) { return std::isdigit(c); })) {
      return false;
    }

    *merge_group_size = std::stoull(merge_group_size_to_parse);
    return IsValidMergeGroupSize(*merge_group_size);
  }

  static bool ParseNumBuffers(const std::string& num_buffers_to_parse, size_t* num_buffers) {
    if (!std::all_of(num_buffers_to_parse.begin(), num_buffers_to_parse.end(),
                     [](unsigned char c) { return std::isdigit(c); })) {
      return false;
    }

    *num_buffers = std::stoull(num_buffers_to_parse);
    return IsValidMergeGroupSize(*num_buffers);
  }

  static size_t GetDefaultNumElements() { return 0; }

  static std::string GetDefaultAlgorithm() { return kValidAlgorithms.front(); }

  static std::vector<int> GetDefaultGpus() {
    if (kValidGpus.size() & (kValidGpus.size() - 1) == 0) {
      return kValidGpus;
    }

    size_t num_gpus = 1;
    while (num_gpus <= kValidGpus.size()) {
      num_gpus *= 2;
    }
    num_gpus /= 2;

    return {kValidGpus.begin(), kValidGpus.begin() + num_gpus};
  }

  static std::string GetDefaultDataType() { return kValidDataTypes.front(); }

  static std::string GetDefaultDistributionType() { return kValidDistributionTypes.front(); }

  static uint32_t GetDefaultDistributionSeed() { return 2147483647; }

  static int GetDefaultNumThreads() { return kValidNumThreads.second; }

  static size_t GetDefaultChunkSize() { return 0; }

  static size_t GetDefaultMergeGroupSize() { return 1; }

  static size_t GetDefaultNumBuffers() { return 2; }

  static std::string GetPrintableNumElementsLimits() { return LimitsToString(kValidNumElements); }

  static std::string GetPrintableAlgorithmLimits() { return LimitsToString(kValidAlgorithms, GetDefaultAlgorithm()); }

  static std::string GetPrintableGpusLimits() { return LimitsToString(kValidGpus, LimitsToString(GetDefaultGpus())); }

  static std::string GetPrintableDataTypeLimits() { return LimitsToString(kValidDataTypes, GetDefaultDataType()); }

  static std::string GetPrintableDistributionTypeLimits() {
    return LimitsToString(kValidDistributionTypes, GetDefaultDistributionType());
  }

  static std::string GetPrintableDistributionSeedLimits() {
    return LimitsToString(kValidDistributionSeeds, GetDefaultDistributionSeed());
  }

  static std::string GetPrintableNumThreadsLimits() { return LimitsToString(kValidNumThreads, GetDefaultNumThreads()); }

  static std::string GetPrintableChunkSizeLimits() { return LimitsToString(kValidChunkSizes, "auto"); }

  static std::string GetPrintableMergeGroupSizeLimits() { return LimitsToString(kValidMergeGroupSizes, "auto"); }

  static std::string GetPrintableNumBuffersLimits() { return LimitsToString(kValidNumBuffers, GetDefaultNumBuffers()); }

 private:
  static bool IsValidNumElements(size_t num_elements) {
    return num_elements >= kValidNumElements.first && num_elements <= kValidNumElements.second;
  }

  static bool IsValidAlgorithm(const std::string& algorithm) {
    return std::find(kValidAlgorithms.begin(), kValidAlgorithms.end(), algorithm) != kValidAlgorithms.end();
  }

  static bool IsValidGpus(const std::vector<int>& gpus) {
    if (gpus.empty() || gpus.size() & (gpus.size() - 1) != 0) {
      return false;
    }

    if (std::set<int>(gpus.begin(), gpus.end()).size() != gpus.size()) {
      return false;
    }

    for (const auto& gpu : gpus) {
      if (std::find(kValidGpus.begin(), kValidGpus.end(), gpu) == kValidGpus.end()) {
        return false;
      }
    }

    return true;
  }

  static bool IsValidDataType(const std::string& data_type) {
    return std::find(kValidDataTypes.begin(), kValidDataTypes.end(), data_type) != kValidDataTypes.end();
  }

  static bool IsValidDistributionType(const std::string& distribution_type) {
    return std::find(kValidDistributionTypes.begin(), kValidDistributionTypes.end(), distribution_type) !=
           kValidDistributionTypes.end();
  }

  static bool IsValidDistributionSeed(uint32_t distribution_seed) {
    return distribution_seed >= kValidDistributionSeeds.first && distribution_seed <= kValidDistributionSeeds.second;
  }

  static bool IsValidNumThreads(int num_threads) {
    return num_threads >= kValidNumThreads.first && num_threads <= kValidNumThreads.second;
  }

  static bool IsValidChunkSize(size_t chunk_size) {
    return chunk_size >= kValidChunkSizes.first && chunk_size <= kValidChunkSizes.second;
  }

  static bool IsValidMergeGroupSize(size_t merge_group_size) {
    return merge_group_size >= kValidMergeGroupSizes.first && merge_group_size <= kValidMergeGroupSizes.second;
  }

  static bool IsValidNumBuffers(size_t num_buffers) {
    return num_buffers >= kValidNumBuffers.first && num_buffers <= kValidNumBuffers.second;
  }

  template <typename T>
  static std::string LimitsToString(const std::pair<T, T>& limits) {
    std::stringstream stream;
    stream << "[" << limits.first << ", " << limits.second << "]";

    return stream.str();
  }

  template <typename T>
  static std::string LimitsToString(const std::vector<T>& limits) {
    std::stringstream stream;
    stream << "{";
    std::copy(limits.begin(), limits.end(), std::ostream_iterator<T>(stream, ", "));
    stream << (!limits.empty() ? "\b\b" : "") << "}";

    return stream.str();
  }

  template <typename T, typename U>
  static std::string LimitsToString(const std::pair<T, T>& limits, const U& default_limit) {
    std::stringstream stream;
    stream << LimitsToString(limits) << " (= " << default_limit << ")";

    return stream.str();
  }

  template <typename T, typename U>
  static std::string LimitsToString(const std::vector<T>& limits, const U& default_limit) {
    std::stringstream stream;
    stream << LimitsToString(limits) << " (= " << default_limit << ")";

    return stream.str();
  }

  static const std::pair<size_t, size_t> kValidNumElements;
  static const std::vector<std::string> kValidAlgorithms;
  static const std::vector<int> kValidGpus;
  static const std::vector<std::string> kValidDataTypes;
  static const std::vector<std::string> kValidDistributionTypes;
  static const std::pair<uint32_t, uint32_t> kValidDistributionSeeds;
  static const std::pair<int, int> kValidNumThreads;
  static const std::pair<size_t, size_t> kValidChunkSizes;
  static const std::pair<size_t, size_t> kValidMergeGroupSizes;
  static const std::pair<size_t, size_t> kValidNumBuffers;
};

const std::pair<size_t, size_t> ArgumentLimits::kValidNumElements = {0, std::numeric_limits<size_t>::max()};
const std::vector<std::string> ArgumentLimits::kValidAlgorithms = {"tanasic",      "gowanlock", "thrust", "boost",
                                                                   "gnu_parallel", "std",       "paradis"};
const std::vector<int> ArgumentLimits::kValidGpus = []() {
  int cuda_device_count;
  CheckCudaError(cudaGetDeviceCount(&cuda_device_count));

  std::vector<int> valid_gpus(cuda_device_count);
  std::iota(valid_gpus.begin(), valid_gpus.end(), 0);

  return valid_gpus;
}();
const std::vector<std::string> ArgumentLimits::kValidDataTypes = {"int", "long", "float", "double"};
const std::vector<std::string> ArgumentLimits::kValidDistributionTypes = {
    "uniform", "normal", "zero", "staggered", "sorted", "reverse-sorted", "nearly-sorted", "bucket-sorted"};
const std::pair<uint32_t, uint32_t> ArgumentLimits::kValidDistributionSeeds = {0, std::numeric_limits<uint32_t>::max()};
const std::pair<int, int> ArgumentLimits::kValidNumThreads = {1, omp_get_num_procs()};
const std::pair<size_t, size_t> ArgumentLimits::kValidChunkSizes = {0, std::numeric_limits<size_t>::max()};
const std::pair<size_t, size_t> ArgumentLimits::kValidMergeGroupSizes = {0, std::numeric_limits<size_t>::max()};
const std::pair<size_t, size_t> ArgumentLimits::kValidNumBuffers = {2, 3};
