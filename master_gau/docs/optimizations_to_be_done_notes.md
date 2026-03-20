    # Reduction Ops Optimization Notes

This file tracks all the optimization techniques, architectures, and clever tricks we discover while reverse engineering TensorFlow and PyTorch. We will evaluate these for implementation in our own codebase.

## GPU-Level Optimizations (from nsys/ncu profiling)

### 1. Fix PackedMetaData Slowdown — Remove `cudaStreamSynchronize` + Cache Pinned Buffer
**(File: `src/UnaryOps/cuda/ReductionImplGPU.cu`)**
The `PackedMetaData` constructor does two things that kill the async benefit:
- `cudaStreamSynchronize(stream)` after `cudaMemcpyAsync` — blocks the CPU every call.
  The memcpy and kernel are on the same stream so ordering is already guaranteed; this sync is redundant.
- `device::PinnedCPUAllocator pinned_allocator` created locally per call — `cudaMallocHost` costs
  ~100–200µs. There is no cache for the host-side pinned buffer; it is allocated and freed every reduction.
**Fix:** Remove the `cudaStreamSynchronize`, and persist/reuse the pinned host buffer (same as how
`CachingCUDAAllocator` works on the device side).

### 2. CPU Synchronization Strategy (`cudaDeviceScheduleBlockingSync`)
**(File: library init / device setup)**
Default CUDA behavior spins (`poll`) — the CPU burns 100% on one core just checking "is GPU done yet?"
**Solution:** Call `cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)` once at library init.
This makes the CPU sleep (`sem_timedwait`) and wake only when the GPU signals completion.
**Trade-off:** ~1–5µs extra latency per sync, but near-zero CPU usage during GPU work.
**When to use:** Production/serving mode. Keep spin-wait for low-latency micro-benchmarking.
**Implementation:** Add a config flag like `OwnTensor::set_low_cpu_mode(true)`.

---

## TensorFlow Optimizations

### 1. Front-end / Graph-Level Graph Optimizations
* **Static Axis Resolution (`_ReductionDims` in Python):** When a user calls a reduction without an axis (`axis=None`), meaning "reduce all dimensions", the Python frontend calculates the full list of dimensions `[0, 1, ..., rank-1]` statically if the tensor's rank is known at graph-construction time. It embeds these axes as a `tf.constant` in the graph.
  * **Why it's an optimization:** It prevents the C++ backend from having to dynamically compute the tensor's rank and generate the axis range array during runtime execution. No "empty axis" or "None" concept actually reaches the core C++ reduction kernels; they always receive explicit axes to reduce.

### 2. Architecture & Linkup Overhead
* **TensorFlow (Heavy Linkup, Fast Kernels):** TF builds a graph, passing commands through huge layered abstractions (Python -> GenOps -> Pybind11 -> C++ Ops -> Dispatchers -> Kernels). For eager execution step-by-step, this introduces significant overhead.
* **Our Approach (master_gau) (Light Linkup, Slower Kernels):** Direct C++ function calls to `ReductionImplGPU.cu` dispatcher, then jumping straight to the CUDA generic kernel. We have incredibly low overhead.
* **Takeaway:** Maintain our low-overhead linkup layer, but upgrade our single generic kernel to specialized kernels like TF does to achieve the best of both worlds.

### 3. "Fold to 3D" Pre-Processing (Dimension Folding)
* **TensorFlow's Secret to Avoiding `unravel`:** TF never reduces an N-Dimensional tensor directly. Before calling a GPU kernel, a helper class (`ReductionHelper::Simplify`) collapses adjacent dimensions together. 
  * Any N-D problem is mathematically squished into at most a **3D tensor** (`[planes, rows, cols]`).
  * If you reduce along `axis=1` of a 5D shape `[2, 3, 4, 5, 6]`, TF folds it into 3D: `planes=2, rows=12 (3*4), cols=30 (5*6)` before launching kernels.

### 4. Specialized Kernels vs Generic `unravel/ravel`
* **Our Approach:** One generic `reduce_kernel`. Every thread uses `shape_to_index` (unravel) to divide and modulo its ID into an N-D coordinate, then `index_to_shape` (ravel) to find where to store the answer. This math (integer division/modulo) is very slow on GPUs.
* **TensorFlow's Approach:** **7 specialized kernels** that only work on 1D, 2D, or 3D grids. Since they reshaped to max 3D, threads just do simple `row * cols + col` math instead of expensive N-Dimensional unraveling.

## PyTorch Optimizations

### 1. CPU Vectorization and Pointer Striding (TensorIterator)
* **Our CPU Approach:** Similar to our GPU approach, we do math to calculate the index. We `unravel` a 1D thread ID into N-D coordinates, and then `ravel` them back into the data array using strides. Every thread calculates exactly where it maps to using integer division and modulo (`%` and `/`).
* **PyTorch's CPU Approach:** PyTorch avoids `unravel` entirely on the CPU by using their **`TensorIterator` with Pointer Striding**.
  * Instead of calculating the $index \times stride$ for every single element, they create nested 1D `for` loops that simply **add the stride to the raw memory pointer**. 
  * *Snippet:* `data[0] += strides[0];`
  * **Why it's faster:** Because CPU registers are fast. Instead of doing expensive division/modulo math to find an element, the CPU just incrementally accesses memory addresses (e.g., from `0x1000` to `0x1008`). Pointer addition is significantly faster on a CPU than integer math!
  * **Combine with Vectorization:** Because they stride via pointers, they can easily load multiple elements (like 8 floats) directly into CPU registers using native CPU intrinsics like AVX2 (`Vec::loadu`).
* **Takeaway:** What we have done in our library is incredibly similar to PyTorch's GPU approach. However, if we want to optimize our CPU path to be exactly as fast as PyTorch's, we should implement **memory pointer striding** nested loops instead of `unravel`.

### 2. CPU vs GPU Vectorization (SIMD vs SIMT)
* **CPU Vectorization (SIMD - Single Instruction, Multiple Data):**
  * Modern CPUs have special 256-bit (AVX2) or 512-bit (AVX-512) registers. AVX stands for *Advanced Vector Extensions*.
  * Instead of a CPU core doing `a + b` for one float, it loads 8 floats into a 256-bit register, and does 8 additions in a single clock cycle.
  * *PyTorch Implementation:* PyTorch has deeply integrated CPU vectorization via `aten/src/ATen/cpu/vec/vec.h`. Their CPU iterators load blocks of 8-16 elements (`Vec::loadu`) and issue SIMD math instructions. This makes CPU path enormously faster than naive `for` loops.
* **GPU Vectorization (SIMT - Single Instruction, Multiple Threads):**
  * GPUs don't have AVX registers. Instead, they naturally execute 32 threads at once in a "warp". A warp executing `a + b` is essentially doing 32 float additions per clock cycle natively! So, the computing part is intrinsically "vectorized".
  * **Memory Vectorization (`float4`):** Where GPU vectorization really shines is *memory bandwidth*. Natively, one thread fetches 32-bits (1 float) per clock. But the memory bus allows 128-bits per thread! 
  * By casting a pointer to `float4*` (or PyTorch's `aligned_vector<float, 4>`), a single thread fetches 4 floats in one cycle. A full warp fetches 128 floats in one memory transaction instead of 32.
* **Our Approach vs PyTorch:** Currently, our GPU reduction kernel loads `T input_value = input_data[input_lin_idx];` (1 element per thread). PyTorch has an `input_vectorized_thread_reduce_impl` that detects contiguous arrays, casts to 128-bit types, and loads 4 elements per thread, dramatically increasing memory throughput. We should implement `float4`/`int4` casting for our fast-path kernels.

### 3. SIMD Vectorized Accumulator Type Conversion (CPU Reduction)
* **The Problem:** In our CPU `reduce_kernel` (ReductionImpl.h), we accumulate using a promoted type (e.g., float input accumulates in double). For each element, we do a scalar `static_cast<AccumulatorT>(input_value)` which compiles to one `cvtss2sd` instruction per element. While this is only 1 cycle, when summing millions of elements, the scalar approach leaves the CPU's SIMD units completely idle.
* **PyTorch's Approach:** PyTorch's vectorized reduction kernels (in `aten/src/ATen/cpu/vec/`) use AVX/SSE intrinsics to batch both the type conversion AND the accumulation together:
  ```cpp
  // AVX2: Convert 4 floats to 4 doubles in ONE instruction
  __m256d result = _mm256_cvtps_pd(_mm_loadu_ps(&input[i]));
  // Then accumulate all 4 doubles at once
  acc = _mm256_add_pd(acc, result);
  ```
  This processes 4 elements per instruction instead of 1. The conversion cost per element drops to 0.25 cycles.
* **Our Current Approach:** Scalar loop, one element at a time:
  ```cpp
  for (int64_t i = 0; i < reduced_count; ++i) {
      AccumulatorT val_acc = static_cast<AccumulatorT>(input_value);  // 1 element
      accumulator = op.reduce(accumulator, val_acc);                  // 1 addition
  }
  ```
* **What to implement:** For contiguous reduction slices, detect contiguity and switch to a SIMD path that uses `_mm256_cvtps_pd` + `_mm256_add_pd` (for float->double) or `_mm256_cvtepi32_epi64` + `_mm256_add_epi64` (for int32->int64). This requires the inner reduction dimension to be contiguous in memory (stride = 1). Non-contiguous cases would stay on the scalar path.
* **Expected speedup:** 4x-8x on the inner accumulation loop for contiguous reductions, depending on AVX2 vs AVX-512 support.