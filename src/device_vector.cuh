#pragma once

#include <thrust/device_vector.h>

template <typename T>
using DeviceVector = thrust::device_vector<T>;
