# GPU Reduction Overhaul: PyTorch-Style Architecture

## Problem Statement

Our GPU reduction implementation is **catastrophically naive** compared to PyTorch. The CPU side was recently upgraded with `compute_reduction_layout()`, SIMD vectorization, and cascade accumulation — but the GPU side was left untouched. Every GPU reduction currently runs through a slow `unravel_index` (div/mod chain) algorithm with zero layout awareness.

## Current State Analysis

### What We Have (GPU — 4 Kernels)

| Kernel | File | Purpose |
|--------|------|---------|
| `reduce_kernel` | `ReductionKernels.cuh:154` | Value reductions (sum, prod, min, max) |
| `reduce_index_kernel` | `ReductionKernels.cuh:398` | Index ops (argmin, argmax) |
| `reduce_mean_kernel` | `ReductionKernels.cuh:523` | Mean / NanMean |
| `reduce_variance_kernel` | `ReductionKernels.cuh:763` | Variance / NanVariance |

### Critical Problems Identified

> [!CAUTION]
> **Problem 1: No Layout Bifurcation on GPU**
> The CPU has 3 paths (InnerContiguous, OuterContiguous, Generic) via `compute_reduction_layout()`. The GPU has **ZERO** layout awareness. Every case goes through the same slow path.

> [!CAUTION]
> **Problem 2: Naive Unravel/Ravel Index Algorithm**
> Every thread in the GPU kernel does O(ndim) integer divisions per element to convert a flat index to N-D coordinates. For a 4D tensor this means 4 `%` and 4 `/` operations **per element per thread**. PyTorch uses stride-based offset calculation that costs O(1) per element.

> [!CAUTION]
> **Problem 3: 1 Block Per Output Slot**
> `num_blocks = num_slices` (Line 209 of ReductionImplGPU.cu). For a **scalar reduction** (full reduce), this launches exactly 1 block = 256 threads. On a GPU with 82 SMs × 2048 threads = 167,936 threads available, we are using **0.15%** of the GPU.

> [!WARNING]
> **Problem 4: No Vectorized Memory Access**
> No `float4`/`int4` loads. Each thread reads 1 element (4 bytes) per memory transaction. PyTorch reads 4 elements (16 bytes) per transaction — 4x bandwidth utilization.

> [!WARNING]
> **Problem 5: Fixed Launch Config**
> Always `threads_per_block = 256`, no dynamic tuning based on tensor shape, data type, or GPU properties.

> [!WARNING]
> **Problem 6: Massive Code Duplication**
> All 4 kernels copy-paste the same unravel logic, coordinate computation, and shared memory reduction. ~800 lines of duplicated logic.

> [!WARNING]
> **Problem 7: Metadata Overhead**
> Every kernel launch copies dims, strides, axes, reduced_dims arrays to GPU. PyTorch pre-computes offsets on CPU and passes only 2 lightweight `OffsetCalculator` structs.

### What CPU Already Has (That GPU Doesn't)

- `compute_reduction_layout()` → InnerContiguous / OuterContiguous / Generic
- SIMD vectorized inner reduction (AVX2 8-wide float)
- Vertical SIMD outer reduction
- Cascade sum for precision
- Strategy selection (ParallelSlices vs SplitReduction)

### What PyTorch Has (That We're Missing Entirely)

1. **`ReduceConfig` solver** — dynamically computes optimal block/grid dims
2. **`reduction_on_fastest_striding_dimension`** — detects Inner vs Outer
3. **4-stage pipeline**: `thread_reduce` → `block_x_reduce` → `block_y_reduce` → `global_reduce`
4. **Vectorized input loads** (`float4` equivalent)
5. **Multiple accumulators per thread** (`vt0=4` for ILP)
6. **Semaphore-based global reduction** (single kernel launch for massive tensors)
7. **Output vectorization** (multiple outputs per thread when outer-reducing)
8. **Dynamic `split_input` / `split_output`** decisions
9. **One generic kernel** handling ALL ops via functor (not 4 copy-pasted kernels)
10. **`OffsetCalculator`** for stride-based access (no unravel needed)

---

## Proposed Changes

### Phase 1: GPU ReduceConfig Solver (Host-Side)

#### [NEW] `include/ops/helpers/GpuReduceConfig.cuh`

A `ReduceConfig` struct (modeled after PyTorch's) that computes the optimal launch parameters:

```cpp
struct GpuReduceConfig {
    int block_width;        // blockDim.x (warp-aligned)
    int block_height;       // blockDim.y
    int num_threads;        // block_width * block_height
    int ctas_per_output;    // >1 means global reduction needed
    
    int input_mult[3];      // [BLOCK_X, BLOCK_Y, CTA]
    int output_mult[2];     // [BLOCK_X, BLOCK_Y]
    
    bool vectorize_input;
    int output_vec_size;
    
    bool reduction_on_fastest_stride; // Inner vs Outer
    
    // Computed from tensor geometry
    int64_t num_inputs;     // elements per output (reduction size)
    int64_t num_outputs;    // total output elements
    
    bool should_block_x_reduce() const;
    bool should_block_y_reduce() const;
    bool should_global_reduce() const;
    
    dim3 block() const;
    dim3 grid() const;
    int64_t shared_memory_size() const;
    int64_t global_memory_size() const;
};
```

**Key Decisions:**
- `reduction_on_fastest_stride = true` → map `block.x` to reduction dim → warp shuffles
- `reduction_on_fastest_stride = false` → map `block.x` to output dim → coalesced writes
- `values_per_thread >= 16` → split across warps (block_y_reduce)
- `values_per_thread >= 256 && grid < target_grid_size` → split across CTAs (global_reduce)

---

### Phase 2: Unified GPU Reduction Kernel

#### [MODIFY] `include/ops/helpers/ReductionKernels.cuh`

Replace ALL 4 kernels with a single generic `gpu_reduce_kernel` that uses the `ReduceConfig` + functor pattern:

```cpp
template<typename T, typename AccT, typename OutputT, typename OpFunctor, int vt0=4>
__global__ void gpu_reduce_kernel(GpuReduceConfig config, OpFunctor ops, 
                                   const T* input, OutputT* output,
                                   AccT identity, /* stride info */) {
    // Stage 1: Thread-level reduction (vt0 accumulators for ILP)
    AccT value = thread_reduce(input, config, ops, identity);
    
    // Stage 2: Warp-level (block_x) reduction via shuffles
    if (config.should_block_x_reduce())
        value = block_x_reduce(value, shared_memory, ops);
    
    // Stage 3: Block-level (block_y) reduction via shared memory
    if (config.should_block_y_reduce())
        value = block_y_reduce(value, shared_memory, ops);
    
    // Stage 4: Grid-level global reduction via semaphores
    if (config.should_global_reduce())
        value = global_reduce(value, global_buffer, semaphores, ops);
    
    // Write output
    if (config.should_store(output_idx))
        output[output_idx] = ops.project(value);
}
```

**What this replaces:**
- `reduce_kernel` → `gpu_reduce_kernel` with `SumOp/ProdOp/MinOp/MaxOp` functors
- `reduce_index_kernel` → `gpu_reduce_kernel` with `ArgMinOp/ArgMaxOp` functors
- `reduce_mean_kernel` → `gpu_reduce_kernel` with `MeanOp` functor (combine=add, project=divide)
- `reduce_variance_kernel` → kept separate (2-pass architecture requires mean first)

---

### Phase 3: Stride-Based Access (Kill Unravel/Ravel)

#### [NEW] `include/ops/helpers/GpuOffsetCalculator.cuh`

Replace the O(ndim) div/mod chain with PyTorch's `OffsetCalculator`:

```cpp
template<int NARGS, typename index_t = uint32_t>
struct OffsetCalculator {
    int dims;
    index_t sizes[MAX_DIMS];
    index_t strides[NARGS][MAX_DIMS];
    
    __device__ Array<index_t, NARGS> get(index_t linear_idx) const {
        // O(ndim) but computed ONCE per output, not per input element
    }
};
```

**Key difference from current approach:**
- Current: Every input element does `unravel(flat_idx)` → O(ndim) divs
- New: Output offset computed once. Input accessed via `base + i * stride` for contiguous, or via pre-computed calculator for strided.

---

### Phase 4: Dispatcher Refactor

#### [MODIFY] `src/UnaryOps/cuda/ReductionImplGPU.cu`

Replace the 4 separate `dispatch_*_gpu` functions with a unified dispatcher:

```cpp
template<typename T, typename OpFunctor>
Tensor gpu_reduce_dispatch(const Tensor& input, const std::vector<int64_t>& axes, 
                            bool keepdim, OpFunctor ops, cudaStream_t stream) {
    // 1. Compute layout (reuse existing compute_reduction_layout)
    auto layout = compute_reduction_layout(input, axes);
    
    // 2. Build ReduceConfig from layout
    auto config = build_reduce_config(layout, sizeof(AccT));
    
    // 3. Allocate global memory if needed
    if (config.should_global_reduce()) { /* allocate buffer + semaphores */ }
    
    // 4. Launch single kernel
    gpu_reduce_kernel<<<config.grid(), config.block(), config.shared_memory_size(), stream>>>(
        config, ops, input_ptr, output_ptr, identity);
}
```

---

## Open Questions

> [!IMPORTANT]
> **Q1: Variance kernel — keep separate or unify?**
> The variance kernel requires a 2-pass architecture (mean first, then squared deviations). PyTorch handles this differently (Welford's online algorithm). Should we:
> - (A) Keep the 2-pass approach but use the new unified kernel for each pass?
> - (B) Switch to Welford's single-pass algorithm?
> Option A is safer and simpler. Option B is theoretically faster but more complex.

> [!IMPORTANT]
> **Q2: Phased rollout or big-bang?**
> Should we:
> - (A) Replace all 4 kernels at once (risky but cleaner)?
> - (B) Start with `reduce_kernel` only, validate, then do the rest?
> I recommend Option B — start with the value reduction kernel (sum/prod/min/max), validate correctness against existing tests, then migrate mean → index → variance.

> [!IMPORTANT]
> **Q3: Global reduction (semaphore) — include in Phase 1?**
> The semaphore-based global reduction is the most complex part. Should we:
> - (A) Include it from the start (full PyTorch parity)?
> - (B) Defer it and use multi-kernel-launch for large reductions initially?
> I recommend Option B — get the single-block optimizations working first, then add global reduction as Phase 5.

---

## Verification Plan

### Automated Tests
1. Run existing reduction test suite (`make test`) after each phase
2. Add specific tests for:
   - Scalar reduction (full reduce) — validates global reduction path
   - Inner contiguous reduction — validates vectorized loads
   - Outer contiguous reduction — validates coalesced output writes
   - Non-contiguous tensor reduction — validates stride calculator
   - Large tensors (>1M elements) — validates multi-CTA path

### Benchmarks
1. Compare against current implementation: `bench_outer_reduction_strategies.cpp`
2. Compare against PyTorch: `bench_nanmean_pytorch_vs_ours.cpp`
3. New benchmark: sweep across tensor shapes (1D scalar, 2D inner, 2D outer, 3D middle)

### Manual Verification
- Profile with `nsys` to verify memory coalescing and SM utilization
- Check that vectorized loads are actually generated (inspect SASS)
