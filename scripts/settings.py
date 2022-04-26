import itertools

from typing import List, Union


class Experiment:

    def __init__(self,
                 identifier: str,
                 executable: str,
                 parameters: List[str],
                 arguments: List[Union[float, int, str]],
                 columns: List[str],
                 repetitions: int = 10,
                 profilers: List[str] = None):

        self.identifier = identifier
        self.executable = executable
        self.parameters = parameters
        self.arguments = arguments
        self.columns = columns
        self.repetitions = repetitions
        self.profilers = profilers


def init():
    global executables_path
    executables_path = "../build"

    global experiments_path
    experiments_path = "../experiments"

    global experiments
    experiments = []

    ####################################################################################################################
    # sort_benchmark
    ####################################################################################################################

    executable = "sort_benchmark"
    parameters = ["", "--algorithm", "--gpus", "--data_type", "--distribution_type", "--distribution_seed"]
    columns = [
        "num_elements", "algorithm", "gpus", "data_type", "distribution_type", "distribution_seed", "num_threads",
        "chunk_size", "merge_group_size", "num_buffers", "memory_allocation_duration", "sort_duration",
        "memory_deallocation_duration", "total_duration"
    ]

    ####################################################################################################################
    # num_elements_to_gowanlock_sort_duration_for_gpus
    ####################################################################################################################

    identifier = "num_elements_to_gowanlock_sort_duration_for_gpus"
    arguments = []

    num_elements = [1000, 1000000000, 2000000000, 4000000000]
    algorithm = ["gowanlock"]
    gpus = ["0"]
    data_type = ["int"]
    distribution_type = ["uniform"]
    distribution_seed = [2147483647]

    arguments += list(
        itertools.product(*[num_elements, algorithm, gpus, data_type, distribution_type, distribution_seed]))

    num_elements = [1000, 1000000000, 2000000000, 4000000000, 6000000000, 8000000000]
    algorithm = ["gowanlock"]
    gpus = ["0,1", "1,2", "0,2"]
    data_type = ["int"]
    distribution_type = ["uniform"]
    distribution_seed = [2147483647]

    arguments += list(
        itertools.product(*[num_elements, algorithm, gpus, data_type, distribution_type, distribution_seed]))

    num_elements = [
        1000, 1000000000, 2000000000, 4000000000, 6000000000, 8000000000, 10000000000, 12000000000, 14000000000,
        16000000000
    ]
    algorithm = ["gowanlock"]
    gpus = ["0,1,2,3", "0,2,1,3", "0,2,4,6"]
    data_type = ["int"]
    distribution_type = ["uniform"]
    distribution_seed = [2147483647]

    arguments += list(
        itertools.product(*[num_elements, algorithm, gpus, data_type, distribution_type, distribution_seed]))

    num_elements = [
        1000, 1000000000, 2000000000, 4000000000, 6000000000, 8000000000, 10000000000, 12000000000, 14000000000,
        16000000000, 18000000000, 20000000000, 22000000000, 24000000000, 26000000000, 28000000000, 30000000000,
        32000000000, 34000000000
    ]
    algorithm = ["gowanlock"]
    gpus = ["0,1,2,3,4,5,6,7"]
    data_type = ["int"]
    distribution_type = ["uniform"]
    distribution_seed = [2147483647]

    arguments += list(
        itertools.product(*[num_elements, algorithm, gpus, data_type, distribution_type, distribution_seed]))

    experiments.append(Experiment(identifier, executable, parameters, arguments, columns))

    ####################################################################################################################
    # num_elements_to_tanasic_sort_duration_for_gpus
    ####################################################################################################################

    identifier = "num_elements_to_tanasic_sort_duration_for_gpus"
    arguments = []

    num_elements = [1000, 1000000000, 2000000000, 4000000000]
    algorithm = ["tanasic"]
    gpus = ["0"]
    data_type = ["int"]
    distribution_type = ["uniform"]
    distribution_seed = [2147483647]

    arguments += list(
        itertools.product(*[num_elements, algorithm, gpus, data_type, distribution_type, distribution_seed]))

    num_elements = [1000, 1000000000, 2000000000, 4000000000, 6000000000, 8000000000]
    algorithm = ["tanasic"]
    gpus = ["0,1", "1,2", "0,2"]
    data_type = ["int"]
    distribution_type = ["uniform"]
    distribution_seed = [2147483647]

    arguments += list(
        itertools.product(*[num_elements, algorithm, gpus, data_type, distribution_type, distribution_seed]))

    num_elements = [
        1000, 1000000000, 2000000000, 4000000000, 6000000000, 8000000000, 10000000000, 12000000000, 14000000000,
        16000000000
    ]
    algorithm = ["tanasic"]
    gpus = ["0,1,2,3", "0,2,1,3", "0,2,4,6"]
    data_type = ["int"]
    distribution_type = ["uniform"]
    distribution_seed = [2147483647]

    arguments += list(
        itertools.product(*[num_elements, algorithm, gpus, data_type, distribution_type, distribution_seed]))

    num_elements = [
        1000, 1000000000, 2000000000, 4000000000, 6000000000, 8000000000, 10000000000, 12000000000, 14000000000,
        16000000000, 18000000000, 20000000000, 22000000000, 24000000000, 26000000000, 28000000000, 30000000000,
        32000000000, 34000000000
    ]
    algorithm = ["tanasic"]
    gpus = ["0,1,2,3,4,5,6,7"]
    data_type = ["int"]
    distribution_type = ["uniform"]
    distribution_seed = [2147483647]

    arguments += list(
        itertools.product(*[num_elements, algorithm, gpus, data_type, distribution_type, distribution_seed]))

    experiments.append(Experiment(identifier, executable, parameters, arguments, columns))

    ####################################################################################################################
    # sort_algorithm_to_sort_duration
    ####################################################################################################################

    identifier = "sort_algorithm_to_sort_duration"
    arguments = []

    data_type = ["int"]
    distribution_type = ["uniform"]
    distribution_seed = [2147483647]

    num_elements = [1000000000, 2000000000, 3000000000, 4000000000, 8000000000]
    algorithm = ["gnu_parallel", "paradis", "thrust"]
    gpus = ["0"]

    arguments += list(
        itertools.product(*[num_elements, algorithm, gpus, data_type, distribution_type, distribution_seed]))

    num_elements = [1000000000, 2000000000, 3000000000, 4000000000, 8000000000]
    gpus = ["0", "0,1", "1,2", "0,2", "0,1,2,3", "0,2,1,3", "0,2,4,6", "0,1,2,3,4,5,6,7"]
    algorithm = ["tanasic"]

    arguments += list(
        itertools.product(*[num_elements, algorithm, gpus, data_type, distribution_type, distribution_seed]))

    num_elements = [1000000000, 2000000000, 3000000000, 4000000000, 8000000000]
    gpus = ["0", "0,1", "1,2", "0,2", "0,1,2,3", "0,2,1,3", "0,2,4,6", "0,1,2,3,4,5,6,7"]
    algorithm = ["gowanlock"]

    arguments += list(
        itertools.product(*[num_elements, algorithm, gpus, data_type, distribution_type, distribution_seed]))

    experiments.append(Experiment(identifier, executable, parameters, arguments, columns))

    ####################################################################################################################
    # data_type_to_gowanlock_sort_duration
    ####################################################################################################################

    identifier = "data_type_to_gowanlock_sort_duration"
    arguments = []

    num_elements = [4000000000]
    algorithm = ["gowanlock"]
    gpus = ["0,1", "0,2", "0,1,2,3", "0,2,4,6", "0,1,2,3,4,5,6,7"]
    data_type = ["int", "float"]
    distribution_type = ["uniform"]
    distribution_seed = [2147483647]

    arguments += list(
        itertools.product(*[num_elements, algorithm, gpus, data_type, distribution_type, distribution_seed]))

    num_elements = [2000000000]
    algorithm = ["gowanlock"]
    gpus = ["0,1", "0,2", "0,1,2,3", "0,2,4,6", "0,1,2,3,4,5,6,7"]
    data_type = ["long", "double"]
    distribution_type = ["uniform"]
    distribution_seed = [2147483647]

    arguments += list(
        itertools.product(*[num_elements, algorithm, gpus, data_type, distribution_type, distribution_seed]))

    experiments.append(Experiment(identifier, executable, parameters, arguments, columns))

    ####################################################################################################################
    # data_type_to_tanasic_sort_duration
    ####################################################################################################################

    identifier = "data_type_to_tanasic_sort_duration"
    arguments = []

    num_elements = [4000000000]
    algorithm = ["tanasic"]
    gpus = ["0,1", "0,2", "0,1,2,3", "0,2,4,6", "0,1,2,3,4,5,6,7"]
    data_type = ["int", "float"]
    distribution_type = ["uniform"]
    distribution_seed = [2147483647]

    arguments += list(
        itertools.product(*[num_elements, algorithm, gpus, data_type, distribution_type, distribution_seed]))

    num_elements = [2000000000]
    algorithm = ["tanasic"]
    gpus = ["0,1", "0,2", "0,1,2,3", "0,2,4,6", "0,1,2,3,4,5,6,7"]
    data_type = ["long", "double"]
    distribution_type = ["uniform"]
    distribution_seed = [2147483647]

    arguments += list(
        itertools.product(*[num_elements, algorithm, gpus, data_type, distribution_type, distribution_seed]))

    experiments.append(Experiment(identifier, executable, parameters, arguments, columns))

    ####################################################################################################################
    # distribution_type_to_sort_duration_for_algorithm
    ####################################################################################################################

    identifier = "distribution_type_to_sort_duration_for_algorithm"
    arguments = []

    num_elements = [2000000000]
    algorithm = ["gowanlock", "tanasic"]
    gpus = ["0,1", "0,2", "0,1,2,3", "0,2,4,6", "0,1,2,3,4,5,6,7"]
    data_type = ["int"]
    distribution_type = [
        "uniform", "normal", "zero", "staggered", "sorted", "reverse-sorted", "nearly-sorted", "bucket-sorted"
    ]
    distribution_seed = [2147483647]

    arguments += list(
        itertools.product(*[num_elements, algorithm, gpus, data_type, distribution_type, distribution_seed]))

    experiments.append(Experiment(identifier, executable, parameters, arguments, columns))

    # ####################################################################################################################
    # num_elements_to_sort_duration_profilers_nsys
    # ####################################################################################################################

    identifier = "num_elements_to_sort_duration_profilers_nsys"
    repetitions = 1
    profilers = ["nsys"]
    arguments = []

    num_elements = [1000000000, 2000000000, 4000000000, 8000000000, 16000000000]
    algorithm = ["gowanlock", "tanasic"]
    gpus = ["0,1", "0,2", "0,1,2,3", "0,2,4,6"]
    data_type = ["int"]
    distribution_type = ["uniform"]
    distribution_seed = [2147483647]

    arguments += list(
        itertools.product(*[num_elements, algorithm, gpus, data_type, distribution_type, distribution_seed]))

    experiments.append(
        Experiment(identifier, executable, parameters, arguments, columns, repetitions=repetitions,
                   profilers=profilers))

    ####################################################################################################################
    # gpus_to_tanasic_sort_duration_profilers_nsys
    ####################################################################################################################

    identifier = "gpus_to_tanasic_sort_duration_profilers_nsys"
    repetitions = 1
    profilers = ["nsys"]
    arguments = []

    num_elements = [2000000000]
    algorithm = ["tanasic"]
    gpus = ["0", "0,1", "1,2", "0,2", "0,1,2,3", "0,2,1,3", "0,2,4,6", "0,1,2,3,4,5,6,7"]
    data_type = ["int"]
    distribution_type = ["uniform"]
    distribution_seed = [2147483647]

    arguments += list(
        itertools.product(*[num_elements, algorithm, gpus, data_type, distribution_type, distribution_seed]))

    experiments.append(
        Experiment(identifier, executable, parameters, arguments, columns, repetitions=repetitions,
                   profilers=profilers))

    ####################################################################################################################
    # gpus_to_gowanlock_sort_duration_profilers_nsys
    ####################################################################################################################

    identifier = "gpus_to_gowanlock_sort_duration_profilers_nsys"
    repetitions = 1
    profilers = ["nsys"]
    arguments = []

    num_elements = [2000000000]
    algorithm = ["gowanlock"]
    gpus = ["0", "0,1", "1,2", "0,2", "0,1,2,3", "0,2,1,3", "0,2,4,6", "0,1,2,3,4,5,6,7"]
    data_type = ["int"]
    distribution_type = ["uniform"]
    distribution_seed = [2147483647]

    arguments += list(
        itertools.product(*[num_elements, algorithm, gpus, data_type, distribution_type, distribution_seed]))

    experiments.append(
        Experiment(identifier, executable, parameters, arguments, columns, repetitions=repetitions,
                   profilers=profilers))

    ####################################################################################################################
    # data_transfer_benchmark
    ####################################################################################################################

    executable = "data_transfer_benchmark"
    parameters = ["", "", "", ""]
    columns = [
        "num_bytes", "gpus", "num_repetitions", "execution_type", "h_to_d_durations", "h_to_d_total_duration",
        "p_to_p_durations", "p_to_p_total_duration", "d_to_h_durations", "d_to_h_total_duration",
        "bidirectional_durations", "total_bidirectional_duration"
    ]

    ####################################################################################################################
    # num_bytes_to_h_to_d_total_duration_and_d_to_h_total_duration
    ####################################################################################################################

    identifier = "num_bytes_to_h_to_d_total_duration_and_d_to_h_total_duration"
    arguments = []

    num_bytes = [4000000000]
    gpus = ["0,1,2,3", "0,1,2,3,4,5,6,7"]
    num_repetitions = [5]
    execution_type = ["serial"]

    arguments += list(itertools.product(*[num_bytes, gpus, num_repetitions, execution_type]))

    experiments.append(Experiment(identifier, executable, parameters, arguments, columns))

    ####################################################################################################################
    # num_bytes_to_p_to_p_total_duration
    ####################################################################################################################

    identifier = "num_bytes_to_p_to_p_total_duration"
    arguments = []

    num_bytes = [4000000000]
    gpus = ["0,1,2,3", "0,1,2,3,4,5,6,7"]
    num_repetitions = [1]
    execution_type = ["serial"]

    arguments += list(itertools.product(*[num_bytes, gpus, num_repetitions, execution_type]))

    experiments.append(Experiment(identifier, executable, parameters, arguments, columns))

    ####################################################################################################################
    # gpus_to_h_to_d_total_duration_and_d_to_h_total_duration
    ####################################################################################################################

    identifier = "gpus_to_h_to_d_total_duration_and_d_to_h_total_duration"
    arguments = []

    num_bytes = [4000000000]
    gpus = ["0,1", "1,2", "0,2", "2,3", "4,5", "4,6", "0,1,2,3", "4,5,6,7", "0,2,1,3", "0,2,4,6", "0,1,2,3,4,5,6,7"]
    num_repetitions = [5]
    execution_type = ["parallel"]

    arguments += list(itertools.product(*[num_bytes, gpus, num_repetitions, execution_type]))

    num_bytes = [4000000000]
    gpus = ["0,1,2,3,4,5,6,7"]
    num_repetitions = [5]
    execution_type = ["serial"]

    arguments += list(itertools.product(*[num_bytes, gpus, num_repetitions, execution_type]))

    experiments.append(Experiment(identifier, executable, parameters, arguments, columns))

    ####################################################################################################################
    # gpus_to_p_to_p_total_duration
    ####################################################################################################################

    identifier = "gpus_to_p_to_p_total_duration"
    arguments = []

    num_bytes = [4000000000]
    gpus = ["0,1", "1,2", "0,2", "2,3", "4,5", "4,6", "0,1,2,3", "4,5,6,7", "0,2,1,3", "0,2,4,6", "0,1,2,3,4,5,6,7"]
    num_repetitions = [1]
    execution_type = ["parallel"]

    arguments += list(itertools.product(*[num_bytes, gpus, num_repetitions, execution_type]))

    num_bytes = [4000000000]
    gpus = ["0,1,2,3,4,5,6,7"]
    num_repetitions = [1]
    execution_type = ["serial"]

    arguments += list(itertools.product(*[num_bytes, gpus, num_repetitions, execution_type]))

    experiments.append(Experiment(identifier, executable, parameters, arguments, columns))

    ####################################################################################################################
    # memory_benchmark
    ####################################################################################################################

    executable = "memory_benchmark"
    parameters = ["", "", ""]
    columns = [
        "num_bytes", "num_gpus", "execution_mode", "memory_allocation_durations", "memory_allocation_total_duration"
    ]

    ####################################################################################################################
    # num_gpus_to_memory_allocation_total_duration
    ####################################################################################################################

    identifier = "num_gpus_to_memory_allocation_total_duration"
    arguments = []

    num_bytes = [8000000000]
    num_gpus = [1, 2, 4, 8]
    execution_mode = ["parallel"]

    arguments += list(itertools.product(*[num_bytes, num_gpus, execution_mode]))

    experiments.append(Experiment(identifier, executable, parameters, arguments, columns))

    ####################################################################################################################
    # local_gpu_sort_algorithm_benchmark
    ####################################################################################################################

    executable = "local_gpu_sort_algorithm_benchmark"
    parameters = ["", "", ""]
    columns = ["num_elements", "sort_algorithm", "gpu_id", "local_sort_duration"]

    ####################################################################################################################
    # local_gpu_sort_algorithm_to_sort_duration
    ####################################################################################################################

    identifier = "local_gpu_sort_algorithm_to_sort_duration"
    arguments = []

    num_elements = [1000000000, 2000000000, 3000000000, 4000000000]
    sort_algorithm = ["cub::DeviceRadixSort::SortKeys", "thrust::sort", "mgpu::mergesort"]
    gpu_id = [0]

    arguments += list(itertools.product(*[num_elements, sort_algorithm, gpu_id]))

    experiments.append(Experiment(identifier, executable, parameters, arguments, columns))

    ####################################################################################################################
    # local_gpu_merge_algorithm_benchmark
    ####################################################################################################################

    executable = "local_gpu_merge_algorithm_benchmark"
    parameters = ["", "", ""]
    columns = ["num_elements", "merge_algorithm", "gpu_id", "local_merge_duration"]

    ####################################################################################################################
    # local_gpu_merge_algorithm_to_duration
    ####################################################################################################################

    identifier = "local_gpu_merge_algorithm_to_merge_duration"
    arguments = []

    num_elements = [
        1000000000, 1100000000, 1200000000, 1300000000, 1400000000, 1500000000, 1600000000, 1700000000, 1800000000,
        1900000000, 2000000000, 2100000000, 2200000000, 2300000000, 2400000000, 2500000000, 2600000000, 2700000000,
        2800000000, 2900000000, 3000000000, 3100000000, 3200000000, 3300000000, 3400000000, 3500000000, 3600000000,
        3700000000, 3800000000, 3900000000, 4000000000, 4100000000
    ]
    merge_algorithm = ["thrust::merge", "mgpu::merge"]
    gpu_id = [0]

    arguments += list(itertools.product(*[num_elements, merge_algorithm, gpu_id]))

    experiments.append(Experiment(identifier, executable, parameters, arguments, columns))

    ####################################################################################################################
    # local_cpu_sort_algorithm_benchmark
    ####################################################################################################################

    executable = "local_cpu_sort_algorithm_benchmark"
    parameters = ["", "", ""]
    columns = ["num_elements", "sort_algorithm", "num_threads", "local_sort_duration"]

    ####################################################################################################################
    # local_cpu_sort_algorithm_to_sort_duration
    ####################################################################################################################

    identifier = "local_cpu_sort_algorithm_to_sort_duration"
    arguments = []

    num_elements = [1000000000, 2000000000, 4000000000, 8000000000, 16000000000, 32000000000, 64000000000]
    sort_algorithm = ["gnu_parallel::sort", "boost::sort::block_indirect_sort", "paradis::sort"]

    arguments += list(itertools.product(*[num_elements, sort_algorithm]))

    experiments.append(Experiment(identifier, executable, parameters, arguments, columns))

    ####################################################################################################################
    # local_cpu_merge_algorithm_benchmark
    ####################################################################################################################

    executable = "local_cpu_merge_algorithm_benchmark"
    parameters = ["", "", ""]
    columns = ["num_elements", "merge_algorithm", "num_threads", "local_merge_duration"]

    ####################################################################################################################
    # local_cpu_merge_algorithm_to_merge_duration
    ####################################################################################################################

    identifier = "local_cpu_merge_algorithm_to_merge_duration"
    arguments = []

    num_elements = [1000000000, 2000000000, 4000000000, 8000000000, 16000000000, 32000000000, 64000000000]
    merge_algorithm = ["gnu_parallel::merge", "gnu_parallel::multiway_merge"]

    arguments += list(itertools.product(*[num_elements, merge_algorithm]))

    experiments.append(Experiment(identifier, executable, parameters, arguments, columns))

    ####################################################################################################################
    # local_cpu_multiway_merge_benchmark
    ####################################################################################################################

    executable = "local_cpu_multiway_merge_benchmark"
    parameters = ["", ""]
    columns = ["sorted_sublist_lengths", "num_threads", "merge_duration"]

    ####################################################################################################################
    # number_sorted_sublists_to_multiway_merge_duration
    ####################################################################################################################

    identifier = "number_sorted_sublists_to_multiway_merge_duration"
    arguments = []

    total_num_elements = [2000000000, 8000000000, 32000000000]

    numbers_sublists = [2, 3, 4, 5, 6, 7, 8, 12, 16]

    for num_elements in total_num_elements:

        sorted_sublists = []

        for n in numbers_sublists:

            sorted_sublist_string = ""

            for i in range(n):
                sorted_sublist_string += str(int(num_elements / n)) + ","

            sorted_sublist_string = sorted_sublist_string[:-1]
            sorted_sublists.append(sorted_sublist_string)

        arguments += list(itertools.product(*[sorted_sublists]))

    experiments.append(Experiment(identifier, executable, parameters, arguments, columns))

    ####################################################################################################################
    # large_data_benchmark
    ####################################################################################################################

    executable = "sort_benchmark"
    parameters = ["", "--algorithm", "--gpus", "--chunk_size", "--merge_group_size", "--num_buffers"]
    columns = [
        "num_elements", "algorithm", "gpus", "data_type", "distribution_type", "distribution_seed", "num_threads",
        "chunk_size", "merge_group_size", "num_buffers", "memory_allocation_duration", "sort_duration",
        "memory_deallocation_duration", "total_duration"
    ]

    ####################################################################################################################
    # large_data_to_gowanlock_sort_duration_for_gpus
    ####################################################################################################################

    identifier = "large_data_to_gowanlock_sort_duration_for_gpus"
    arguments = []

    num_elements = [
        1000, 1000000000, 2000000000, 4000000000, 8000000000, 12000000000, 16000000000, 20000000000, 24000000000,
        32000000000, 40000000000, 48000000000, 56000000000, 60000000000, 64000000000
    ]
    algorithm = ["gowanlock"]
    gpus = ["0,1", "0,1,2,3", "0,1,2,3,4,5,6,7"]
    chunk_size = [3000000000, 4125000000]
    merge_group_size = [0, 1]
    num_buffers = [2]

    arguments += list(itertools.product(*[num_elements, algorithm, gpus, chunk_size, merge_group_size, num_buffers]))

    num_elements = [
        1000, 1000000000, 2000000000, 4000000000, 8000000000, 12000000000, 16000000000, 20000000000, 24000000000,
        32000000000, 40000000000, 48000000000, 56000000000, 60000000000, 64000000000
    ]
    algorithm = ["gowanlock"]
    gpus = ["0,1", "0,1,2,3", "0,1,2,3,4,5,6,7"]
    chunk_size = [2000000000, 2750000000]
    merge_group_size = [0, 1]
    num_buffers = [3]

    arguments += list(itertools.product(*[num_elements, algorithm, gpus, chunk_size, merge_group_size, num_buffers]))

    experiments.append(Experiment(identifier, executable, parameters, arguments, columns))

    ####################################################################################################################
    # large_data_to_gowanlock_vs_cpu_sort_duration_for_gpus
    ####################################################################################################################

    identifier = "large_data_to_gowanlock_vs_cpu_sort_duration_for_gpus"
    arguments = []

    num_elements = [
        1000, 1000000000, 2000000000, 4000000000, 8000000000, 12000000000, 16000000000, 20000000000, 24000000000,
        32000000000, 40000000000, 48000000000, 56000000000, 60000000000, 64000000000
    ]
    algorithm = ["gowanlock"]
    gpus = ["0,1", "0,1,2,3", "0,1,2,3,4,5,6,7"]
    chunk_size = [3000000000, 4125000000]
    merge_group_size = [1]
    num_buffers = [2]

    arguments += list(itertools.product(*[num_elements, algorithm, gpus, chunk_size, merge_group_size, num_buffers]))

    num_elements = [
        1000, 1000000000, 2000000000, 4000000000, 8000000000, 12000000000, 16000000000, 20000000000, 24000000000,
        32000000000, 40000000000, 48000000000, 56000000000, 60000000000, 64000000000
    ]
    algorithm = ["gnu_parallel", "paradis"]
    gpus = ["0"]
    chunk_size = [0]
    merge_group_size = [1]
    num_buffers = [2]

    arguments += list(itertools.product(*[num_elements, algorithm, gpus, chunk_size, merge_group_size, num_buffers]))

    experiments.append(Experiment(identifier, executable, parameters, arguments, columns))

    ####################################################################################################################
    # gpus_to_sort_duration_profilers_nsys
    ####################################################################################################################

    executable = "sort_benchmark"
    parameters = [
        "", "--algorithm", "--gpus", "--data_type", "--distribution_type", "--chunk_size", "--merge_group_size",
        "--num_buffers"
    ]
    columns = [
        "num_elements", "algorithm", "gpus", "data_type", "distribution_type", "distribution_seed", "num_threads",
        "chunk_size", "merge_group_size", "num_buffers", "memory_allocation_duration", "sort_duration",
        "memory_deallocation_duration", "total_duration"
    ]

    identifier = "gpus_to_sort_duration_profilers_nsys"
    repetitions = 1
    profilers = ["nsys"]
    arguments = []

    num_elements = [4000000000]
    algorithm = ["gowanlock", "tanasic"]
    gpus = ["0,1", "1,2", "0,2", "0,1,2,3", "0,2,1,3", "0,2,4,6", "0,1,2,3,4,5,6,7"]
    data_type = ["int"]
    distribution_type = ["uniform"]
    chunk_size = [0]
    merge_group_size = [1]
    num_buffers = [2]

    arguments += list(
        itertools.product(
            *[num_elements, algorithm, gpus, data_type, distribution_type, chunk_size, merge_group_size, num_buffers]))

    num_elements = [8000000000]
    algorithm = ["gowanlock", "tanasic"]
    gpus = ["0,1,2,3", "0,2,1,3", "0,2,4,6", "0,1,2,3,4,5,6,7"]
    data_type = ["int"]
    distribution_type = ["uniform"]
    chunk_size = [0]
    merge_group_size = [1]
    num_buffers = [2]

    arguments += list(
        itertools.product(
            *[num_elements, algorithm, gpus, data_type, distribution_type, chunk_size, merge_group_size, num_buffers]))

    num_elements = [2000000000]
    algorithm = ["gowanlock", "tanasic"]
    gpus = ["0,1", "0,2", "0,1,2,3", "0,2,4,6", "0,1,2,3,4,5,6,7"]
    data_type = ["int"]
    distribution_type = [
        "uniform", "normal", "zero", "staggered", "sorted", "reverse-sorted", "nearly-sorted", "bucket-sorted"
    ]
    chunk_size = [0]
    merge_group_size = [1]
    num_buffers = [2]

    arguments += list(
        itertools.product(
            *[num_elements, algorithm, gpus, data_type, distribution_type, chunk_size, merge_group_size, num_buffers]))

    num_elements = [32000000000, 60000000000]
    algorithm = ["gowanlock"]
    gpus = ["0", "0,1", "0,2", "0,1,2,3", "0,2,4,6", "0,1,2,3,4,5,6,7"]
    data_type = ["int"]
    distribution_type = ["uniform"]
    chunk_size = [3000000000, 4125000000]
    merge_group_size = [1, 0]
    num_buffers = [2]

    arguments += list(
        itertools.product(
            *[num_elements, algorithm, gpus, data_type, distribution_type, chunk_size, merge_group_size, num_buffers]))

    num_elements = [32000000000, 60000000000]
    algorithm = ["gowanlock"]
    gpus = ["0", "0,1", "0,2", "0,1,2,3", "0,2,4,6", "0,1,2,3,4,5,6,7"]
    data_type = ["int"]
    distribution_type = ["uniform"]
    chunk_size = [2000000000, 2750000000]
    merge_group_size = [1, 0]
    num_buffers = [3]

    arguments += list(
        itertools.product(
            *[num_elements, algorithm, gpus, data_type, distribution_type, chunk_size, merge_group_size, num_buffers]))

    experiments.append(
        Experiment(identifier, executable, parameters, arguments, columns, repetitions=repetitions,
                   profilers=profilers))

    ####################################################################################################################
    # inplace_memcpy_benchmark
    ####################################################################################################################

    executable = "inplace_memcpy_benchmark"
    parameters = ["", "", ""]
    columns = ["num_bytes", "gpu_id", "block_size", "inplace_data_transfer_duration"]

    ####################################################################################################################
    # block_size_to_inplace_data_transfer_duration
    ####################################################################################################################

    identifier = "block_size_to_inplace_data_transfer_duration"
    arguments = []

    num_bytes = [1000000000, 2000000000, 4000000000, 8000000000]
    gpu_id = [0, 2]

    for b in num_bytes:
        block_size = []

        for i in [1, 10, 100, 1000, 10000, 100000]:
            block_size.append(b / i)

        for i in [2, 4, 8, 16, 32]:
            block_size.append(b / i)

        arguments += list(itertools.product(*[[b], gpu_id, block_size]))

    experiments.append(Experiment(identifier, executable, parameters, arguments, columns))
