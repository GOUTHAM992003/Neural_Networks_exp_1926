# 🔬 Reduction Ops Analysis: `master_gau` vs `Tensor-Implementations_gau`

**Sprint**: March 2 – March 27, 2026  
**Goal**: Make reduction ops kernels SOTA

---

## Files Compared

| File | master_gau | Tensor-Implementations_gau |
|------|-----------|---------------------------|
| [ReductionKernels.cuh](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/master_gau/include/ops/helpers/ReductionKernels.cuh) | **1026 lines** | **882 lines** |
| [ReductionImplGPU.cu](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/master_gau/src/UnaryOps/cuda/ReductionImplGPU.cu) | 739 lines | 761 lines |
| [ReductionOps.h](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/master_gau/include/ops/helpers/ReductionOps.h) | Modified | Original |
| ReductionUtils.h | **Identical** | **Identical** |
| ReductionUtils.cpp | **Identical** | **Identical** |
| Reduction.cpp (CPU) | Different | Original |

---

## 🔴 Key Differences Found (3 Major Optimizations in `master_gau`)

### 1. Shared Memory Metadata Caching ⚡
**Location**: `reduce_kernel` in [ReductionKernels.cuh](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/master_gau/include/ops/helpers/ReductionKernels.cuh) (master_gau lines 182–222)

```diff
- // OLD (Tensor-Implementations_gau): Reads from global memory every iteration
- extern __shared__ char shared_mem[];
- AccumulatorType* shared = reinterpret_cast<AccumulatorType*>(shared_mem);

+ // NEW (master_gau): Caches metadata arrays in shared memory
+ extern __shared__ char shared_mem[];
+ int64_t* s_input_strides = reinterpret_cast<int64_t*>(shared_mem);
+ int64_t* s_output_dims   = s_input_strides + ndim;
+ int64_t* s_reduced_dims  = s_output_dims + (...);
+ int64_t* s_normalized_axes = s_reduced_dims + num_reduced_dims;
+ AccumulatorType* shared = reinterpret_cast<AccumulatorType*>(s_normalized_axes + num_axes);
+
+ // Cooperative load by threads
+ if (threadIdx.x < ndim)          s_input_strides[threadIdx.x] = input_strides[threadIdx.x];
+ if (threadIdx.x < output_ndim)   s_output_dims[threadIdx.x] = output_dims[threadIdx.x];
+ if (threadIdx.x < num_reduced_dims) s_reduced_dims[threadIdx.x] = reduced_dims[threadIdx.x];
+ if (threadIdx.x < num_axes)      s_normalized_axes[threadIdx.x] = normalized_axes[threadIdx.x];
+ __syncthreads();
```

**Impact**: Avoids repeated global memory reads for `input_strides`, `output_dims`, `reduced_dims`, `normalized_axes` during the inner loop. Shared memory has ~100x lower latency than global memory.

---

### 2. Base Input Offset Hoisting 🚀
**Location**: `reduce_kernel` in [ReductionKernels.cuh](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/master_gau/include/ops/helpers/ReductionKernels.cuh) (master_gau lines 247–268)

**Old approach** (Tensor-Implementations_gau): For each reduction element `i`, it recomputes the full `full_input_coords[ndim]` array including BOTH reduced and preserved dimension coordinates, then dots them all against strides.

**New approach** (master_gau): Hoists the preserved-dimension offset **outside** the inner loop:

```cpp
// OPTIMIZATION: Computed ONCE per output slice (outside inner loop)
int64_t base_input_offset = 0;
int out_coord_idx = 0;
for (int dim = 0; dim < ndim; ++dim) {
    if (!is_reduced) {
        int64_t coord = rank_preserved ? out_coords[dim] : out_coords[out_coord_idx];
        base_input_offset += coord * s_input_strides[dim];  // Uses shared mem!
        if (!rank_preserved) out_coord_idx++;
    }
}
// Inner loop only needs: base_input_offset + reduced_offset
```

**Impact**: Eliminates redundant coordinate computation for preserved dims on every element — significant for high `reduced_count`.

---

### 3. 1D Reduction Fast Path 🏎️
**Location**: `reduce_kernel` in [ReductionKernels.cuh](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/master_gau/include/ops/helpers/ReductionKernels.cuh) (master_gau lines 270–329)

```cpp
// OPTIMIZATION 2: Fast Path for 1D Reduction
if (num_reduced_dims == 1) {
    // Single dimension reduction - No unraveling needed!
    int64_t reduced_dim_idx = s_normalized_axes[0];
    int64_t stride = s_input_strides[reduced_dim_idx];
    
    for (int64_t i = threadIdx.x; i < reduced_count; i += blockDim.x) {
        int64_t input_lin_idx = base_input_offset + i * stride;  // Direct!
        // ... accumulate
    }
} else {
    // General Case: Multi-dimension reduction
    for (int64_t i = ...) {
        // Unravel + offset calculation (but using shared mem + hoisted base)
    }
}
```

**Impact**: For the **most common case** (reducing a single axis), completely eliminates `unravel` (modulo/divide chain). Replaces with a simple `base + i * stride` linear access. This is a **massive** speedup for inner-dimension reductions.

---

## 🟡 Minor Differences

| Aspect | master_gau | Tensor-Implementations_gau |
|--------|-----------|---------------------------|
| **Namespace** | `OwnTensor::cuda` | `OwnTensor::cuda_1926` |
| **GPU Intrinsics** | Inline in [ReductionKernels.cuh](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/master_gau/include/ops/helpers/ReductionKernels.cuh) (`to_float`, `from_float`, `shfl_down`) | Via separate `GPUIntrinsics.cuh` |
| **Variance kernel template** | `<T, MeanT, OutputT, AccT, VarianceOpType>` (5 params) | `<T, MeanT, OutputT, VarianceOpType>` (4 params; AccT deduced inside) |
| **`#pragma unroll 4`** | Added on outer loop + inner loops | Not present on outer loops |
| **`float4_e2m1` types** | Full support in [ReductionOps.h](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/master_gau/include/ops/helpers/ReductionOps.h) (identity, isnan, etc.) | Not supported |
| **`cudaDeviceSynchronize()`** | Commented out (async) | Active in `dispatch_reduction_gpu` |
| **DeviceArray** | Stores `stream_` member | No stream stored |
| **Mean kernel float4** | Has `float4_e2m1` output conversion branch | Missing |
| **`output_ndim` calc** | Moved **outside** the output loop | Inside the output loop (redundant) |

---

## 🟢 What's Identical

- **[ReductionUtils.h](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/master_gau/include/ops/helpers/ReductionUtils.h)** + **[ReductionUtils.cpp](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/master_gau/src/UnaryOps/cpu/ReductionUtils.cpp)** — Shape calculation utilities are identical
- **Block reduction** logic (warp reduce → shared → final warp) — same algorithm
- **Index reduction kernel** (`reduce_index_kernel`) — nearly identical (only `#pragma unroll` added in master_gau)
- **Mean reduction kernel** — identical algorithm, just `#pragma unroll` annotations
- **Variance kernel** — identical algorithm, only AccT as explicit template param differs

---

## 📊 Performance Impact Summary

| Optimization | Where | Expected Impact |
|---|---|---|
| Shared memory metadata caching | `reduce_kernel` only | **Medium** — reduces global mem latency in inner loop |
| Base offset hoisting | `reduce_kernel` only | **Medium-High** — eliminates redundant per-element work |
| 1D fast path | `reduce_kernel` only | **High** — eliminates unravel for most common case |
| `#pragma unroll 4` | All kernels | **Low-Medium** — compiler hint for loop unrolling |
| `cudaDeviceSynchronize` removed | Dispatcher | **Low** — better async overlap |

> [!IMPORTANT]
> These 3 optimizations are only applied to the **basic `reduce_kernel`** (sum, prod, min, max, etc.) — **NOT** to `reduce_index_kernel`, `reduce_mean_kernel`, or `reduce_variance_kernel`. This is the first opportunity for SOTA work: propagate these optimizations to all kernels.

---

## 🎯 Recommended Sprint Approach

Your plan of **PyTorch → TensorFlow → Our SOTA** is solid. Here's the refined roadmap:

### Phase 1: Research (Days 1–5)
1. **Study PyTorch reduction kernels** — Focus on [ATen/native/cuda/Reduce.cuh](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/Tensor-Implementations_gau/PyTorch-MLP-Text/libtorch/include/ATen/native/cuda/Reduce.cuh) and [reduction_template.cuh](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/Tensor-Implementations_gau/PyTorch-MLP-Text/libtorch/include/ATen/native/cuda/reduction_template.cuh)
   - Key concepts: `ReduceOp`, `gpu_reduce_kernel`, multi-block reductions, vectorized loads
2. **Study TensorFlow reduction kernels** — Focus on `tensorflow/core/kernels/reduction_gpu_kernels.cu.h`
   - Key concepts: Cub-based reductions, column/row specialization

### Phase 2: Apply Missing Optimizations (Days 5–10)
3. Propagate shared mem caching + base offset hoisting + 1D fast path to:
   - `reduce_index_kernel`
   - `reduce_mean_kernel`
   - `reduce_variance_kernel`

### Phase 3: SOTA Techniques (Days 10–25)
4. **Vectorized loads** (`float4`/`int4`) — load 4 elements per thread per iteration
5. **Multi-block reductions** — for very large tensors, use 2-pass global reduction
6. **Contiguous inner-dim specialization** — when reducing the last axis, use coalesced sequential access
7. **Warp-level specialization** — for small reductions (< 32 elements), skip block reduce entirely
8. **Grid-stride loop tuning** — optimize `num_blocks` vs `threads_per_block` for occupancy
