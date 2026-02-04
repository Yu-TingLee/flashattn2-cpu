FlashAttention-2 on CPU
=======================
This repository contains several implementations of FlashAttention-2 forward pass on CPU in Python (NumPy, Numba), and C++ (Eigen).
Scripts for profiling runtime distributions and naive attention baselines are also included.

This project originated as a final project for a machine learning course.
My report is avaiable at [https://drive.google.com/file/d/1Ui7b7OmlLXq72F-xsxJRqMYaqPPlKO0-/view?usp=sharing](https://drive.google.com/file/d/1Ui7b7OmlLXq72F-xsxJRqMYaqPPlKO0-/view?usp=sharing).

Dependencies
------------
- Python packages: numpy, numba, pandas, matplotlib
- C++ libraries: Eigen3

Key Files
---------
- `data_generation.py`: generates random testsets.
- `naive_attn.py`, `naive_attn.cpp`: Naive attention implementations in Python and C++.
- `flash_attn2.py`, `flash_attn2_jit.py`: FlashAttention-2 forward pass in NumPy and Numba (JIT).
- `flash_attn2.cpp`: FlashAttention-2 forward pass in C++ using Eigen.
- `profile/flash_attn2_profile.*`: Profiling scripts for runtime breakdown of major algorithmic steps.
- `plot.py`: Generates all plots.

Run all the experiments with:
--------

```bash
cmake -S . -B build
cmake --build build
bash run.sh
```

Logs are saved to `outputs/<impl>/runtime.csv`; plots are saved in `outputs/plots/`.
