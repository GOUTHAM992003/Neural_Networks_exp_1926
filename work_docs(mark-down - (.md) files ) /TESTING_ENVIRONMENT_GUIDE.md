# Tensor Testing Environment Guide 

Hello ! Welcome to the testing ground for our **Tensor Library_1926**. 

This environment is built to benchmark our custom C++ Tensor implementation against : **PyTorch**, **NumPy**, and **LibTorch (C++)**.
This document covers the **A-to-Z** of how this works: the math behind the metrics, the operations we test, how to run everything (from single files to the whole suite), and even the tricky technical details we solved.

---

## Operations & Metrics



### 1. Operations Covered
We run a set of **50+ operations** on every framework:
*   **Arithmetic**: `Add`, `Sub`, `Mul`, `Div`
*   **Unary Math**: `Square`, `Sqrt`, `Neg`, `Abs`, `Sign`, `Reciprocal`, `Pow2`
*   **Trigonometry**: `Sin`, `Cos`, `Tan`, `Sinh`, `Cosh`, `Tanh`
*   **Inverse Trig**: `Asin`, `Acos`, `Atan`, `Asinh`, `Acosh`, `Atanh`
*   **Exponents/Logs**: `Exp`, `Log`, `Log2`, `Log10`
*   **Matrix**: `MatMul` (Matrix Multiplication)
*   **Reductions**: `Sum`, `Mean`, `Max`, `Min`, `Var` (Variance), `Std` (Standard Deviation)
*   **Chains**: 15 complex composite operations (e.g., `sin(cos(tan(x)))`) to stress-test operator fusion and chaining efficiency.



### 2. The Math Behind the Metrics
For every operation, we capture these four key stats. Think of them as answering three different questions:

*   **Time (ms)**:
    *   **"How long did it take?"**
    *   We run the operation **50 times** (after 5 warmup runs) and take the mean.

*   **Computational Throughput (GFLOPS)**:
    *   **"How fast are you computing?"** (Math Speed)
    *   Measures raw number crunching power.
    *   FORMULA: `(Total_Elements * FLOPs_per_Ops) / Time_in_Seconds / 1e9`
    *   *Note*: A simple Add is 1 FLOP. `Sin` or `Exp` takes ~5-10 FLOPs. Matrix Mul is `2*R*C`.

*   **Data Throughput (Elements/sec)**:
    *   **"How much data are you processing?"** (Item Speed)
    *   Measures the volume of distinct numbers handled.
    *   FORMULA: `Total_Elements / Time_in_Seconds`

*   **I/O Throughput (Bandwidth GB/s)**:
    *   **"How fast are you moving data?"** (Memory Speed)
    *   **WARNING**: This is NOT the "PCIe Bandwidth" (transfer speed between CPU and GPU).
    *   This is **Effective Memory Bandwidth** (VRAM/RAM speed).
    *   **What it measures**: How fast the kernel read inputs from global memory and wrote outputs back.
    *   **What this is NOT**: This is **NOT** the PCIe transfer speed (CPU ↔ GPU, usually ~16 GB/s).
    *   **What this IS**: This is **Internal VRAM Bandwidth** (GPU Memory ↔ GPU Cores).
        *   We copy inputs to the GPU *before* the timer starts.
        *   Therefore, we measure the raw speed of the GPU's internal GDDR6 memory (e.g., ~360 GB/s on RTX 3060).
    *   **Why it matters**: If we measured PCIe speed, results would cap at ~16 GB/s. By measuring VRAM speed, we can prove our kernels are properly reading/writing at the hardware's maximum capability (300+ GB/s).
    *   **Compute Bound vs Memory Bound**:
        *   **Memory Bound (Add, Copy)**: Limited by how fast VRAM can deliver data. High Bandwidth, Low GFLOPS.
        *   **Compute Bound (Sin, Exp, MatMul)**: Limited by math speed. Lower Bandwidth, High GFLOPS.
    *   **CPU vs GPU**:
        *   **On CPU**: Measures System RAM speed (e.g., DDR4 ~20-50 GB/s).
        *   **On GPU**: Measures GDDR6 VRAM speed (e.g., ~300+ GB/s).
    *   FORMULA: `(Total_Elements * Bytes_Per_Element * Access_Count) / Time_in_Seconds / 1e9`
    *   *Note*: `Access_Count` is estimated (3 for Binary, 2 for Unary, 1 for Reduction).

### 2.1 Reality Check: Derived vs Measured
It is important to understand that in software benchmarks (Python/C++ scripts), the **ONLY** thing we physically measure is **Time (ms)**.

*   **Derived Metrics**: Bandwidth, Throughput, and GFLOPS are **Calculated Estimates**.
    *   **Effective Bandwidth**: Answers *"If I were streaming memory perfectly, how fast does it look like I'm going?"*
    *   **Effective GFLOPS**: Answers *"If I were doing pure math perfectly, how fast does it look like I'm computing?"*

*   **Why use Derived Metrics?**
    *   It gives a universal efficiency score.
    *   **Example**: If your GPU is rated for **300 GB/s**:
        *   If result = **250 GB/s**, your code is **Great** (83% efficiency).
        *   If result = **20 GB/s**, your code is **Terrible**.
    
*   **Real Hardware Counters**: 
    *   To know *exactly* how many bytes moved or how many floating point instructions the GPU executed, you **MUST** use hardware profilers like **NVIDIA Nsight Compute (`ncu`)** or **Intel VTune**. A script cannot see these internal counters.
        *   *Warning*: This is a theoretical "Effective Bandwidth" calculated from the problem definition. It assumes perfect data streaming. For actual hardware utilization and cache hit/miss analysis, use profiling tools like **NVIDIA Nsight Compute (`ncu`)**.

### 3. Example Walkthrough: `C = A + B`
Let's see the math in action for a **1000x1000x10** tensor (10 Million elements) of `float32`.

**Scenario**: We run `Add` on GPU.

1.  **The Setup (Pre-Benchmark)**:
    *   `A` and `B` are already allocated in **GPU VRAM** (Global Memory).
    *   They are **NOT** moving from CPU across PCIe during the timed loop. They are sitting ready in the GPU's high-speed memory.

2.  **The Execution (Timed Loop)**:
    *   The CUDA kernel launches.
    *   **Activity**:
        1.  READ `A[i]` (4 bytes) from VRAM.
        2.  READ `B[i]` (4 bytes) from VRAM.
        3.  COMPUTE `A[i] + B[i]` (1 FLOP).
        4.  WRITE `C[i]` (4 bytes) to VRAM.
    *   **Total Data Moved per Element**: 4 (Read A) + 4 (Read B) + 4 (Write C) = **12 Bytes**.

3.  **The Math**:
    *   **Total Elements**: 10,000,000 (10 Million).
    *   **Total Bytes Moved**: 10M * 12 Bytes = **120 MB** (0.12 GB).
    *   **Total FLOPs**: 10M * 1 = **10 MFLOPs**.
    *   *Hypothetical Time*: Let's say it took **0.0004 seconds** (0.4ms).

4.  **The Metrics We Report**:
    *   **Data Throughput**: `10,000,000 / 0.0004` = **25 Billion Elements/sec**.
    *   **Bandwidth**: `(10M * 12 Bytes) / 0.0004` = **300 GB/s**.
        *   *Interpretation*: This is excellent! We are utilizing the VRAM speed efficiently.
    *   **GFLOPS**: `(10M * 1 FLOP) / 0.0004` = **25 GFLOPS**.
        *   *Interpretation*: Low GFLOPS? Yes, because `Add` is **memory-bound**, not compute-bound. The GPU spends most time waiting for memory, not doing math.

### 4. Numerical Validation (Are we Correct?)
Speed is useless if the math is wrong. We validate correctness by comparing our results against **PyTorch** (the "Ground Truth").

*   **The Workflow**:
    1.  Every benchmark generates a **Values CSV** (e.g., `tensorlib_values.csv`, `pytorch_values.csv`) containing the actual output numbers of every operation.
    2.  Our script automatically matches these files row-by-row.
    3.  It calculates the **Absolute Error**: `|Your_Result - PyTorch_Result|`.
*   **The Output**: Look at `benchmark_results/comparison/precision_comparison.csv`.
*   **Success Criteria**: The error should be negligible (e.g., `< 1e-5` for float32). If it's `0.0`, you have a perfect match.

### 5. Validating the Metrics (Sanity Checks)
How do you trust the speed numbers? Use these rules of thumb:

1.  **The Bandwidth limit**:
    *   Look up your GPU's theoretical limit (e.g., RTX 3060 = **360 GB/s**).
    *   Your `Add`/`Copy` results should be **60-90%** of this limit (e.g., 250-320 GB/s).
    *   *Red Flag*: If you report 500 GB/s on a 360 GB/s card, your timer is wrong (probably measuring cache hits, not VRAM).

2.  **The GFLOPS Reality**:
    *   Do not expect TFLOPS performance for valid `Add` or `Sub`.
    *   *Why*: Simple math is **Memory Bound**. The GPU cores sit idle waiting for data.
    *   *Validation*: Only heavy ops like `MatMul` or `Exp` chains should approach the TFLOPS range (Compute Bound).

---

##  The Architecture (Design Plan)

The main goal is **Fair Comparison** (apples-to-apples).

1.  **Shared  Inputs**: 
    *   **The Flow**: TensorLib generates random inputs -> Saves to `benchmark_results/inputs/benchmark_all_3d_inputs.csv`.
    *   **The Check**: Every other framework (NumPy, PyTorch, LibTorch) tries to read this EXACT file. This ensures they operate on identical numbers.

2.  **Case when u run individually in any one of the libraries**:
    *   **Scenario**: You run PyTorch alone, but the shared input file is missing (maybe you deleted it?).
    *   **Action**: The script won't crash. It catches the error, generates its own random data, and **SAVES** it to `benchmark_results/pytorch/generated_inputs.csv`.
    *   **Result**: You get your benchmark, and you also get a record of the inputs used.

----

##  Execution Guide (Commands)

###  1. The "Complete Suite" (Recommended)
This script does it all: cleans old data, builds C++ binaries, runs CPU/GPU tests for ALL libraries, and generates comparison tables.

*   **Where to run**: From the `scripts/` folder or root.
*   **Command**:
    ```bash
    ./scripts/run_complete_benchmark.sh
    ```

###  2. Running Individual Frameworks
Sometimes you only want to test ONE library.

#### **A. NumPy (CPU)**
*   **Command**: `python3 numpy_tests/numpy_dk_ops.py`
*   **Where to run**: From project root.
*   **Output**: `benchmark_results/numpy/numpy_values.csv`

#### **B. PyTorch**
*   **CPU**: `python3 pytorch_tests/pytorch_fun_tests.py`
*   **GPU**: `python3 pytorch_tests/pytorch_cuda_benchmark.py`
*   **Where to run**: From project root.
*   **Output**: `benchmark_results/pytorch/` or `benchmark_results/pytorch_cuda/`

#### **C. LibTorch (C++)**
You must verify the binary is compiled first.
*   **Compile**: 
    ```bash
    cd libtorch_tests/benchmark
    make cuda_bench_all_3d  # For GPU
    make libtorch_fun_tests # For CPU
    ```
*   **Run (GPU)**:
    ```bash
    ./cuda_bench_all_3d
    ```
*   **Run (CPU)**:
    ```bash
    ./libtorch_fun_tests
    ```
*   **Output logic**: Whether you run this from the `benchmark/` folder OR the project root, the code uses `std::filesystem` (or smart path logic) to ensure CSVs always land in `../../benchmark_results` or `benchmark_results/`. Safe to run from anywhere!

#### **D. Our TensorLib**
*   **Run Complete Benchmark**: `scripts/run_tensorlib_benchmark.sh`
*   **Run Single Snippet/Test**:
    If you want to run a specific test file isolated (e.g., a unit test or scratchpad):
    1.  Go to the implementation directory:
        ```bash
        cd Tensor_Implementations_kota/Tensor_Implementations
        ```
    2.  Use the `run-snippet` target:
        ```bash
        make run-snippet file=Tests/your_test_file.cpp
        ```

---

##  Directory Structure

Here's the map of the testing environment folder `test_env_gau`:

```
test_env_gau/
├── benchmark_results/          <--  (All outputs are here)
│   ├── inputs/                 <-- Shared inputs used by all other test-files.
│   ├── tensorlib/              <-- Our library's results
│   ├── pytorch/                <-- PyTorch CPU results
│   ├── numpy/                  <-- NumPy results
│   ├── pytorch_cuda/           <-- PyTorch GPU results
│   ├── libtorch_cuda/          <-- LibTorch C++ GPU results
│   └── comparison/             <-- Final side-by-side comparison tables
│
├── scripts/                    <-- Automation scripts needed to run things
│   ├── run_complete_benchmark.sh
│   └── compare_all_metrics.py
│
├── Tensor_Implementations/ <-- Our Source Code (TensorLib)
├── pytorch_tests/               <-- Python scripts for PyTorch
├── numpy_tests/                 <-- Python scripts for NumPy
└── libtorch_tests/              <-- C++ code for LibTorch
```

---


If you look at the code/Makefiles, you might see some specific things. Here is why they are there:

1.  **LibTorch CUDA Linking (`-Wl,--no-as-needed`)**:
    *   We had a bug where `cuda_bench_all_3d` compiled fine but crashed with "CUDA not available".
    *   **Reason**: The linker was being "too smart" and dropping the `libtorch_cuda.so` library because it didn't see explicit symbols used.
    *   **Fix**: We added `-Wl,--no-as-needed -ltorch_cuda` to the Makefile. This forces `libtorch_cuda` to be loaded at runtime, properly initializing the GPU backend.

2.  **3D vs 2D Tensors**:
    *   Default shape is `10x10x10` (3D).
    *   **To Test 2D Matrices**: You don't need new code. Just change `D=1` (Depth=1) in the source files. `1x1000x1000` is mathematically identical to `1000x1000`. The metrics remain accurate.

---

##  Code Internals: Helper Functions

You will see some common helper functions and C++ patterns. Here is what those funcs do:

*   **`BENCH_OP(name, expr)` (The Lambda Trick)**:
    *   **Syntax**: It expands to a C++ Lambda `[&](){ ... }`.
    *   **What is a Lambda?**: It's an anonymous function defined inline. The `[&]` captures all local variables (like `a`, `b`) by reference so you can use them inside.
    *   **Why**: It allows us to wrap "Timing Logic" around *any* expression (like `a+b` or `torch::sin(a)`) without writing a separate function for each one.

*   **`auto result = ...`**:
    *   **What is `auto`?**: It tells the C++ compiler: "You figure out the type."
    *   **Why**: Instead of writing `torch::Tensor result = ...`, we write `auto`. It keeps the code clean, especially for complex iterator types or long namespaces.

*   **`get_float_value(tensor, index)`**:
    *   **Purpose**: Extract a single float value from a specific linear index.
    *   **Why**: Used when writing the `values.csv` for correctness verification. We need to be careful—some tensors are `int`, some `double`. This function handles type casting safely so we always write a uniform float format to the CSV.

*   **`toCPU(tensor)`**:
    *   **Purpose**: Safely moves a CUDA tensor to CPU RAM.
    *   **Why**: We CANNOT write to a file directly from GPU memory. We must move the data to CPU first. We typically do this at the very end, just before writing the CSVs, so it doesn't affect the benchmark timing itself.

---

##  Developer Guide: Extending the Suite

Want to add something new?

### 1. Adding a New Operation
If you want to test a new function (e.g., `Sigmoid`):
1.  **Open the benchmark file** for the framework (e.g., `cuda_bench_all_3d.cpp`).
2.  **Add the Bench call**:
    ```cpp
    auto sig_result = BENCH_OP_CUDA("sigmoid", torch::sigmoid(a));
    ```
3.  **Update CSV Writing**:
    *   Add the column header to the `values.csv` writing section.
    *   Add the value writing logic inside the loop: `<< get_float_value(sig_result, idx) << ","`.
4.  **Recompile & Run**: The Timing, Throughput, and GFLOPS CSVs will **automatically** pick up the new key from `all_timings` and include it!

### 2. Adding a New Metric
If you want to measure something new (e.g., "Peak Memory Usage"):
1.  **Update the Data Struct**: Add a `double peak_mem_mb` field to the `OpTiming` struct in the source file.
2.  **Update BENCH_OP**: Modify the macro logic to capture this metric during execution.
3.  **Add Output File**:
    *   Go to the bottom of the file where CSVs are generated.
    *   Add a new `std::ofstream` block (e.g., for `libtorch_cuda_memory.csv`).
    *   Loop through `all_timings` and write the new metric.
4.  **Update Comparison Script**: If you want it in the final side-by-side report, edit `scripts/compare_all_metrics.py` to ingest this new CSV type.

---

##  Support
If you have any weird compilation errors, linker issues, path not found errors, or general confusion about why this architecture exists:

**Please resolve with Tensor&Ops_1926 Team Members.**
