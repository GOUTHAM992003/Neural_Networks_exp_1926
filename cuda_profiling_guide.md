# The Ultimate Guide to CUDA Kernel Profiling

Profiling is the process of analyzing your program's performance to identify bottlenecks and optimization opportunities. In CUDA, where the interaction between the CPU (host) and GPU (device) is intricate, profiling is absolutely essential. We cannot guess what the hardware is doing; we have to measure it.

This guide covers everything you need to know about CUDA profiling, from high-level application tracing to low-level kernel instruction analysis, specifically tailored for engineers building custom tensor and neural network libraries.

---

## 1. Why Do We Profile CUDA Code?

When writing CUDA kernels (like your `add_bias_kernel` or `dispatch_reduction_gpu`), it's very easy to write intuitively "correct" code that runs drastically below the GPU's potential. We profile to answer questions like:

1.  **Am I CPU-bound or GPU-bound?** Is the GPU sitting idle waiting for the CPU to launch kernels (launch overhead), or is the GPU fully occupied?
2.  **Am I Memory-bound or Compute-bound?** Is the kernel taking a long time because it's waiting for data from Global Memory (VRAM), or because it's doing complex math?
3.  **Is my Memory Access Coalesced?** Are adjacent threads reading adjacent memory addresses? (Critical for utilizing the wide memory bus).
4.  **Do I have high Occupancy?** Are there enough active warps on the Streaming Multiprocessor (SM) to hide memory latency?
5.  **Are there PCIe transfer bottlenecks?** Are CPU-to-GPU (`cudaMemcpy`) transfers dominating the runtime?

---

## 2. The Tools of the Trade

NVIDIA provides a suite of powerful profiling tools. Historically, `nvprof` and `Visual Profiler (nvvp)` were used, but **they are now deprecated**. 

You should exclusively use the **NVIDIA Nsight Suite**:

### A. Nsight Systems (`nsys`)
*   **What it is:** A system-wide performance analysis tool. It gives you a macro-level timeline view of CPU and GPU activity.
*   **When to use it:** Always start here! Use it to find out *what* to optimize. It shows CPU threads, CUDA API calls (like `cudaMalloc`, `cudaMemcpy`), kernel launches, and GPU busy/idle times.
*   **Why use it:** To fix launch overheads, overlap data transfers with computation (using CUDA streams), and identify the slowest kernels in the pipeline.

### B. Nsight Compute (`ncu`)
*   **What it is:** An interactive, low-level kernel profiler. It provides granular performance metrics for *a specific kernel*.
*   **When to use it:** After using `nsys` to identify a slow kernel, use `ncu` to find out *why* that specific kernel is slow.
*   **Why use it:** To analyze memory throughput, instruction pipeline stalls, register pressure, occupancy, and cache hit rates at warp/thread level.

---

## 3. How to Use Nsight Systems (`nsys`)

Let's say you have a test executable that runs your reduction operations: `./build/ReductionsTest`.

### Command Line Execution
To profile the entire application and generate a report:

```bash
nsys profile -o my_report --stats=true ./build/ReductionsTest
```

**Key Flags:**
*   `-o my_report`: Outputs the report to `my_report.nsys-rep` (and handles overwriting).
*   `--stats=true`: Prints standard statistical summaries right to your terminal when it finishes.
*   `--trace=cuda,cudnn,cublas`: (Optional but recommended) Explicitly tell it to trace CUDA APIs, plus standard libraries if you use them.

### Analyzing the Report
1.  **Terminal Output (`--stats=true`)**: You will see a table listing all CUDA API calls, memory copies, and kernels. Look at the **"Time (%)"** and **"Total Time (ns)"** columns to find the most expensive operations.
2.  **Visual GUI (`nsight-sys`)**: 
    *   Transfer the `.nsys-rep` file to your local machine (if you are running on a remote server).
    *   Open it with the Nsight Systems GUI application.
    *   **The Timeline View**: This is invaluable. You will see rows for the CPU and GPU.
    *   **What to look for in the GUI:**
        *   **White space on the GPU row:** This means the GPU is doing nothing. Why? Is it waiting for a `cudaMemcpy`? Is the CPU taking too long doing `std::vector` allocations before calling `cudaLaunchKernel`?
        *   **Serial execution:** Are kernels executing one after the other when they could be independent? You might need to utilize concurrent **CUDA Streams** (I noticed your `dispatch_reduction_gpu` takes a `stream`, which is excellent!).
        *   **Stuttered overlapping:** If you have CPU/GPU overlaps, are they actually running in parallel?

---

## 4. How to Use Nsight Compute (`ncu`)

Once `nsys` tells you that `reduce_bias_kernel_optimized` (from your [LinearKernels.cu](file:///home/blu-bridge016/Desktop/master_gau/src/Kernels/cuda/LinearKernels.cu)) is taking up 40% of your total time, you want to dive deep into that specific kernel.

**Warning:** Unconstrained profiling with `ncu` will run the application multiple times (multipass) because it uses hardware counters. It will be very slow.

### Command Line Execution
To profile a specific kernel and get a quick terminal summary:

```bash
ncu -k reduce_bias_kernel_optimized --set full -o bias_kernel_prof ./build/ReductionsTest
```

**Key Flags:**
*   `-k <kernel_regex>`: Only profile kernels matching this regex. This is crucial for speed.
*   `--set full`: Collects all available metrics.
*   `-c 1`: (Optional) Only profile the *first* invocation of the kernel. Useful if it's called in a loop.
*   `-o bias_kernel_prof`: Outputs a `.ncu-rep` file for the GUI.

### Analyzing the Report (The Terminal output)
If you don't use `-o`, `ncu` prints directly to the terminal. It provides "Sections".

**Section 1: GPU Speed Of Light (SOL)**
This is the most important section. It tells you your bottleneck.
*   **Compute (SM) Throughput [X%]**: How much of the math units are you using?
*   **Memory Throughput [Y%]**: How much of the theoretical VRAM bandwidth are you saturating?

*   If Memory > Compute: You are **Memory Bound**. (Most tensor operations like `add`, `bias`, `reduction` are memory bound).
*   If Compute > Memory: You are **Compute Bound**. (Dense matrix multiplications (GEMM) are typically compute-bound).
*   If both are low (e.g., < 20%): You have **Latency Issues** (poor occupancy, stalls).

**Section 2: Launch Statistics**
*   **Block Size / Grid Size**: Analyzes if your launch configuration is efficient (`threads_per_block` in your code).
*   **Registers per Thread**: Highly important. If a thread uses too many registers, the SM cannot schedule many concurrent warps (low occupancy).

### Analyzing the Report (The GUI)
Open the generated `.ncu-rep` file in the Nsight Compute GUI.

*   **The Details Page**: Provides an expert system that *tells you in plain English* what is wrong. It might say: *"This kernel is memory bound. Consider checking for uncoalesced global memory accesses."*
*   **The Source Page (Source Correlation)**: This is the holy grail. 
    *   If you compile with `-lineinfo` flag (e.g., `nvcc -lineinfo ...`), Nsight Compute maps the metrics directly to your C++ source code lines.
    *   You can click on your `input_data[b * bias_size + c]` line and it will tell you exactly how many L1/L2 cache misses occurred on that specific line, and if the access was coalesced or uncoalesced.

---

## 5. Typical Profiling Workflow: A Practical Example

Let's imagine profiling your `cuda_linear_bias_backward` function.

**Step 1: The Macro View (`nsys`)**
```bash
nsys profile -o linear_test ./build/MyTestRunner
```
*Result:* Looking at the stats, you see `reduce_bias_kernel` takes 5ms, while `matmul` takes 2ms. This is suspicious. Bias reduction should usually be faster than a full GEMM.

**Step 2: The Micro View (`ncu`)**
```bash
ncu -k reduce_bias_kernel --set full ./build/MyTestRunner
```
*Result:* The terminal output spits out the SOL (Speed of Light) section.
*   Compute Throughput: 5%
*   Memory Throughput: 15%
Wait, both are low. The profiler gives a warning: **"Uncoalesced Global Memory Access"**.

**Step 3: Source Code Analysis**
You look at your comments in [src/Kernels/cuda/LinearKernels.cu](file:///home/blu-bridge016/Desktop/master_gau/src/Kernels/cuda/LinearKernels.cu):
```cpp
// Stride is `bias_size`. So `grad_output[b * bias_size + c]`. 
// This is strided access. BAD.
```
Your comments already spotted the issue! The threads in a warp are accessing memory locations separated by `bias_size`. This causes multiple memory transactions for a single instruction, destroying bandwidth utilization.

**Step 4: The Fix and Re-profile**
You refactor the code (as you mentioned in your comments, treating it as ColMajor or using a different mapping strategy). You compile and run `ncu` again.
*   *New Result:* Memory Throughput jumps from 15% to 80%.

---

## 6. Key CUDA Performance Concepts to Look For

When reading `ncu` reports, keep these concepts in mind:

### 1. Memory Coalescing
The GPU reads memory in 32-byte, 64-byte, or 128-byte segments. If Thread 0 reads address 0, Thread 1 reads address 4, etc. (contiguous), the hardware satisfies the whole warp with one transaction. If they read scattered addresses, it requires many transactions. `ncu` will explicitly tell you if you have "Sector Misses" or uncoalesced accesses.

### 2. Occupancy
Occupancy is the ratio of active warps on an SM to the maximum number of warps the SM supports. High occupancy is necessary to hide memory latency (when warp A is waiting for data to load, the scheduler instantly switches to warp B to do math).
*   **Limiting factors:** High register usage per thread, too much shared memory allocated per block, or block sizes that don't divide cleanly into the SM limits.

### 3. Shared Memory Bank Conflicts
If you use `__shared__` memory (like in your `reduce_kernel`), it's divided into 32 banks. If multiple threads in a warp access different addresses that map to the *same* bank, the accesses are serialized (bank conflict). `ncu` tracks "Shared Memory Bank Conflicts" specifically.

### 4. Warp Divergence
If threads within a warp take different execution paths in an `if/else` statement, the warp must execute sequentially for both paths (masking out inactive threads). This wastes compute cycles.

---

## 7. How to Setup Your Environment for Profiling

To get the most out of profiling, compile your CUDA code with specific flags:

1.  **Generate Line Information:** Add `-lineinfo` to your `nvcc` flags. This tells the compiler to map PTX/SASS instructions back to your [.cu](file:///home/blu-bridge016/Desktop/master_gau/src/ops/IndexingOps.cu) file, which is crucial for the Source View in `ncu`. 
    *   *Note: Do not use `-G` (device debug). `-G` disables all optimizations and gives you an inaccurate profile.*
2.  **Enable Host Debugging Symbols:** Add `-g` (host debug) so CPU functions are named properly in `nsys`.

Example CMake:
```cmake
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -g -O3")
```

## Summary Checklist

1.  **Macro Profile:** `nsys profile ./app` -> Identify sluggish kernels and CPU/GPU pipeline gaps.
2.  **Micro Profile:** `ncu -k sluggish_kernel ./app` -> Identify Compute vs Memory bottlenecks.
3.  **Analyze & Fix:** Check source correlation for uncoalesced memory, register pressure, and shared memory conflicts.
4.  **Repeat:** Profiling is an iterative loop. Fix the biggest bottleneck, then profile again.
