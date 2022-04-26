cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(multi-gpu-sorting LANGUAGES CUDA CXX)

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_subdirectory(third_party/thrust)

set(THRUST_REQUIRED_SYSTEMS CUDA OMP)
thrust_create_target(ThrustCUDA HOST OMP DEVICE CUDA)

set(BOOST_INCLUDE_DIRS  "third_party/boost/libs/sort/include/"
                        "third_party/boost/libs/type_traits/include"
                        "third_party/boost/libs/config/include"
                        "third_party/boost/libs/static_assert/include"
                        "third_party/boost/libs/core/include"
                        "third_party/boost/libs/range/include"
                        "third_party/boost/libs/preprocessor/include"
                        "third_party/boost/libs/mpl/include"
                        "third_party/boost/libs/iterator/include")

set(PARADIS_INCLUDE_DIRS "third_party/simple_paradis/src/")

# sort_benchmark
add_executable(sort_benchmark "src/sort_benchmark.cu")
target_link_libraries(sort_benchmark PUBLIC Thrust::Thrust ThrustCUDA)
set_property(TARGET sort_benchmark PROPERTY CUDA_ARCHITECTURES 80 72 70)

target_include_directories(sort_benchmark SYSTEM PRIVATE ${BOOST_INCLUDE_DIRS} ${PARADIS_INCLUDE_DIRS})

# local_cpu_sort_algorithm_benchmark
add_executable(local_cpu_sort_algorithm_benchmark "src/local_cpu_sort_algorithm_benchmark.cu")
target_link_libraries(local_cpu_sort_algorithm_benchmark PUBLIC Thrust::Thrust ThrustCUDA)
set_property(TARGET local_cpu_sort_algorithm_benchmark PROPERTY CUDA_ARCHITECTURES 80 72 70)

target_include_directories(local_cpu_sort_algorithm_benchmark SYSTEM PRIVATE ${BOOST_INCLUDE_DIRS} ${PARADIS_INCLUDE_DIRS})

# local_cpu_merge_algorithm_benchmark
add_executable(local_cpu_merge_algorithm_benchmark "src/local_cpu_merge_algorithm_benchmark.cu")
target_link_libraries(local_cpu_merge_algorithm_benchmark PUBLIC Thrust::Thrust ThrustCUDA)
set_property(TARGET local_cpu_merge_algorithm_benchmark PROPERTY CUDA_ARCHITECTURES 80 72 70)

# local_cpu_multiway_merge_benchmark
add_executable(local_cpu_multiway_merge_benchmark "src/local_cpu_multiway_merge_benchmark.cu")
target_link_libraries(local_cpu_multiway_merge_benchmark PUBLIC Thrust::Thrust ThrustCUDA)
set_property(TARGET local_cpu_multiway_merge_benchmark PROPERTY CUDA_ARCHITECTURES 80 72 70)

# local_gpu_sort_algorithm_benchmark
add_executable(local_gpu_sort_algorithm_benchmark "src/local_gpu_sort_algorithm_benchmark.cu")
target_compile_options(local_gpu_sort_algorithm_benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
target_link_libraries(local_gpu_sort_algorithm_benchmark PUBLIC Thrust::Thrust ThrustCUDA)
set_property(TARGET local_gpu_sort_algorithm_benchmark PROPERTY CUDA_ARCHITECTURES 80 72 70)

target_include_directories(local_gpu_sort_algorithm_benchmark SYSTEM PRIVATE "third_party/moderngpu")

# local_gpu_merge_algorithm_benchmark
add_executable(local_gpu_merge_algorithm_benchmark "src/local_gpu_merge_algorithm_benchmark.cu")
target_compile_options(local_gpu_merge_algorithm_benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
target_link_libraries(local_gpu_merge_algorithm_benchmark PUBLIC Thrust::Thrust ThrustCUDA)
set_property(TARGET local_gpu_merge_algorithm_benchmark PROPERTY CUDA_ARCHITECTURES 80 72 70)

target_include_directories(local_gpu_merge_algorithm_benchmark SYSTEM PRIVATE "third_party/moderngpu")

# data_transfer_benchmark
add_executable(data_transfer_benchmark "src/data_transfer_benchmark.cu")
target_link_libraries(data_transfer_benchmark PUBLIC Thrust::Thrust ThrustCUDA)
set_property(TARGET data_transfer_benchmark PROPERTY CUDA_ARCHITECTURES 80 72 70)

# inplace_memcpy_benchmark
add_executable(inplace_memcpy_benchmark "src/inplace_memcpy_benchmark.cu")
target_link_libraries(inplace_memcpy_benchmark PUBLIC Thrust::Thrust ThrustCUDA)
set_property(TARGET inplace_memcpy_benchmark PROPERTY CUDA_ARCHITECTURES 80 72 70)

# memory_benchmark
add_executable(memory_benchmark "src/memory_benchmark.cu")
target_link_libraries(memory_benchmark PUBLIC Thrust::Thrust ThrustCUDA)
set_property(TARGET memory_benchmark PROPERTY CUDA_ARCHITECTURES 80 72 70)
