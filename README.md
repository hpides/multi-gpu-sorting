# Evaluating Multi-GPU Sorting with Modern Interconnects

This repository contains the source code for our [ACM SIGMOD '22 paper](https://doi.org/10.1145/3514221.3517842).

_GPUs have become a mainstream accelerator for database operations such as sorting. Most GPU sorting algorithms are single-GPU approaches. They neither harness the full computational power nor exploit the high-bandwidth P2P interconnects of modern multi-GPU platforms. The latest NVLink 2.0 and NVLink 3.0-based NVSwitch interconnects promise unparalleled multi-GPU acceleration. So far, multi-GPU sorting has only been evaluated on systems with PCIe 3.0. In this paper, we analyze serial, parallel, and bidirectional data transfer rates to, from, and between multiple GPUs on systems with PCIe 3.0/4.0, NVLink 2.0/3.0, and NVSwitch. We measure up to 35x higher parallel P2P throughput with NVLink 3.0-based NVSwitch over PCIe 3.0. To study GPU-accelerated sorting on today’s hardware, we implement a P2P-based GPU-only (P2P sort) and a heterogeneous (HET sort) multi-GPU sorting algorithm and evaluate them on three modern platforms. We observe speedups over state-of-the-art parallel CPU radix sort of up to 14x for P2P sort and 9x for HET sort. On systems with fast P2P interconnects, P2P sort outperforms HET sort up to 1.65x. Finally, we show that overlapping GPU copy/compute operations does not mitigate the transfer bottleneck when sorting large out-of-core data._

## Table of Contents
Directory | Description
----------|------------
src/ | Sources of the multi-GPU sorting algorithms and benchmarks.
scripts/ | Scripts to run and plot the benchmarks.

## Useful Commands
### Initializing the Project
`git submodule update --init --recursive`

### Building the Project
`./build.sh`

### Running the Benchmarks
`python3 scripts/run_experiments.py`

This creates an `experiments` folder and places the benchmark results into a subfolder, named after the current date/time (e.g., `2022_02_22_23_59_59`).

### Plotting the Graphs
`python3 scripts/plot_experiments.py 2022_02_22_23_59_59`

This creates `.pdf` plots in the folder `2022_02_22_23_59_59`.
