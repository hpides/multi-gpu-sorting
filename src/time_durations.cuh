#pragma once

#include <chrono>
#include <map>
#include <string>

class TimeDurations {
 public:
  static TimeDurations* Get() {
    static TimeDurations instance;

    return &instance;
  }

  TimeDurations(const TimeDurations& time_durations) = delete;

  void operator=(const TimeDurations& time_durations) = delete;

  inline void Tic(std::string key) { begin_time_[key] = std::chrono::high_resolution_clock::now(); }

  inline void Toc(std::string key) { durations[key] = std::chrono::high_resolution_clock::now() - begin_time_.at(key); }

  inline std::chrono::duration<double> TotalExecutionTime() {
    std::chrono::duration<double> total_execution_time(0);
    for (const auto& duration : durations) {
      total_execution_time += duration.second;
    }

    return total_execution_time;
  }

  std::map<std::string, std::chrono::duration<double>> durations;

 private:
  TimeDurations() = default;
  ~TimeDurations() = default;

  std::map<std::string, std::chrono::time_point<std::chrono::high_resolution_clock>> begin_time_;
};
