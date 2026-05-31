Peformance Evaluation of FlashAttention-2 for Inference on CPU
=======================
This repository contains implementations of the FlashAttention-2 forward kernel run on CPU. Standard attention baselines are also provided. For causal attention, three scheduling strategies are implemented: static, centralized dynamic, and work-stealing.

The technical report, including algorithm derivation, complexity analysis, and interpretation, is available here:
* [Performance Evaluation of FlashAttention-2 for Inference on CPU (PDF)](./flashattn2-cpu-report.pdf)

Short Abstract
------------
**TL;DR:** We evaluate the performance of FlashAttenion-2 forward kernel for inference on CPU.

FlashAttention has revolutionized transformer training and inference on GPU by minimizing memory I/O, yet its behavior on CPU remains underexplored. We benchmark several implementations of the forward kernel against standard attention baselines and, for masked attention, compare three threading strategies for parallelizing the kernel. We also profile the runtime breakdown of the kernel's algorithmic steps. Results show that FA-2 can outperform standard attention in every tested configuration, yielding up to 7.9x speedup for self-attention and 12.8x for masked attention. We also find three intriguing results: larger block sizes can degrade performance despite lower I/O complexity, the optimal block size is typically small-to-moderate, and online softmax dominates runtime despite its lower complexity than matrix multiplications.

<br>
<p align="center">
  <img src="./images/fa2_figure.png" width="100%">
  <br>
  <em>Figure 1: Computation of the FlashAttention-2 forward kernel.</em>
</p>

Dependencies
------------
- **Python:** numpy, numba, pandas, matplotlib
- **C++:** Eigen3 ≥ 3.3, OpenMP, CMake ≥ 3.16, C++17 compiler

Usage
-----
```bash
cmake -S . -B build && cmake --build build --parallel
bash run.sh
```

Logs go to `outputs/<impl>/runtime.csv`; Plots go to `outputs/plots/`

Each benchmark resumes from existing CSV entries if interrupted.
