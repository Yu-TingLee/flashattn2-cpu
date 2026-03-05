FlashAttention-2 on CPU
=======================
CPU implementations of the FlashAttention-2 forward pass in Python (NumPy, Numba)
and C++ (Eigen3, single- and multi-threaded). Final project for a machine learning course.

Report: https://drive.google.com/file/d/1Ui7b7OmlLXq72F-xsxJRqMYaqPPlKO0-/view?usp=drive_link

```
include/flash_attn/   C++ headers
src/                  C++ implementations
tools/                Benchmark runners (C++ and Python)
flash_attn/           Python package (pure algorithm functions)
data_generation.py    Generates Q/K/V testsets
plot.py               Plots runtime CSVs
run.sh                Full pipeline
```

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
