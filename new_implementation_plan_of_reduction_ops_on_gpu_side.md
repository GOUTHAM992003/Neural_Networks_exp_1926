# GPU Reduction Overhaul: PyTorch-Style Unified Architecture

## Goal

Replace our 4 naive GPU reduction kernels with a **single unified kernel** using PyTorch's functor + config solver architecture. Includes Welford single-pass variance for full 4→1 unification.

## Final Decisions

| Decision | Choice | Rationale |
|:---|:---|:---|
| Kernel count | **4 → 1** unified | Eliminates ~800 lines duplication, all optimizations benefit all ops |
| Variance | **Welford single-pass** | 1 data read vs 2. GPU is bandwidth-bound → 2x faster |
| Functor source | **Extend existing `ReductionOps.h`** | Already have 14 ops with `DEVICE_HOST`. Add `combine`/`project`/`warp_shfl_down` |
| Access pattern | **3-tier stride** | Contiguous(95% DL) → single-stride → OffsetCalculator(fallback) |
| Verification | **Step-by-step** | Compile + test after every phase |

---

## Current State (7 Critical Problems)

1. **No layout bifurcation** — GPU has zero layout awareness (CPU has Inner/Outer/Generic)
2. **Naive unravel/ravel** — O(ndim) div/mod per element per thread
3. **1 block per output** — scalar reduction uses 0.15% of GPU
4. **No vectorized loads** — 4 bytes instead of 16 bytes per transaction
5. **Fixed 256 threads** — no dynamic tuning
6. **800 lines copy-paste** — all 4 kernels duplicate same unravel logic
7. **2-pass variance** — reads entire tensor twice from HBM

---

## Proposed Changes (5 Phases)

### Phase 1: Extend Functor Interface in `ReductionOps.h`

#### [MODIFY] [ReductionOps.h](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/master_gau/include/ops/helpers/ReductionOps.h)

Add 3 missing methods to ALL existing op structs:

**For scalar ops** (SumOp, ProductOp, MinOp, MaxOp, NanSum, NanProd, NanMin, NanMax, AllOp, AnyOp):
```cpp
// combine = same as reduce (merging two partial sums = adding them)
DEVICE_HOST AccT combine(const AccT& a, const AccT& b) const { return reduce(a, b); }

// project = identity (output = accumulator as-is)
DEVICE_HOST AccT project(const AccT& a) const { return a; }

// warp_shfl_down = single-value shuffle
__device__ AccT warp_shfl_down(AccT val, int offset) const;
```

**For index ops** (ArgMinOp, ArgMaxOp, NanArgMin, NanArgMax):
```cpp
// combine = same as reduce (comparing two partial argmins)
DEVICE_HOST ValueIndex<T> combine(const ValueIndex<T>& a, const ValueIndex<T>& b) const { return reduce(a, b); }

// project = extract index only
DEVICE_HOST int64_t project(const ValueIndex<T>& a) const { return a.index; }

// warp_shfl_down = shuffle both value and index
__device__ ValueIndex<T> warp_shfl_down(ValueIndex<T> val, int offset) const;
```

**Add `reduce(acc, val, idx)` overload** — Index ops need the element index. Add 3-arg  signature to all ops:
```cpp
// For non-index ops: ignore idx, delegate to 2-arg version
DEVICE_HOST AccT reduce(const AccT& acc, const T& val, int64_t /*idx*/) const { return reduce(acc, val); }

// For index ops: wrap val+idx into ValueIndex, delegate
DEVICE_HOST ValueIndex<T> reduce(const ValueIndex<T>& acc, const T& val, int64_t idx) const {
    return reduce(acc, ValueIndex<T>(val, idx));
}
```

#### [NEW] `WelfordData<T>` struct and `WelfordOps<T>` functor in `ReductionOps.h`

```cpp
template<typename acc_t, typename index_t = int64_t>
struct WelfordData {
    acc_t mean = acc_t(0);
    acc_t m2 = acc_t(0);
    index_t n = 0;
    acc_t nf = acc_t(0);
};

template<typename T>
struct WelfordOps {
    using AccScalar = AccumulatorType<T>;
    using acc_t = WelfordData<AccScalar>;
    AccScalar correction;
    bool take_sqrt;

    DEVICE_HOST acc_t identity() const { return acc_t{}; }

    DEVICE_HOST acc_t reduce(acc_t acc, AccScalar data, int64_t) const {
        index_t new_n = acc.n + 1;
        AccScalar new_nf = static_cast<AccScalar>(new_n);
        AccScalar delta = data - acc.mean;
        AccScalar new_mean = acc.mean + delta / new_nf;
        AccScalar new_delta = data - new_mean;
        return {new_mean, acc.m2 + delta * new_delta, new_n, new_nf};
    }

    DEVICE_HOST acc_t combine(acc_t a, acc_t b) const {
        if (a.nf == 0) return b;
        if (b.nf == 0) return a;
        AccScalar delta = b.mean - a.mean;
        AccScalar new_count = a.nf + b.nf;
        AccScalar nb_over_n = b.nf / new_count;
        return {a.mean + delta * nb_over_n,
                a.m2 + b.m2 + delta * delta * a.nf * nb_over_n,
                -1, new_count};
    }

    DEVICE_HOST AccScalar project(acc_t acc) const {
        AccScalar divisor = acc.nf > correction ? acc.nf - correction : AccScalar(0);
        AccScalar var = acc.m2 / divisor;
        return take_sqrt ? sqrt(var) : var;
    }

    __device__ acc_t warp_shfl_down(acc_t acc, int offset) const {
        return {shfl_down(acc.mean, offset), shfl_down(acc.m2, offset),
                shfl_down(acc.n, offset), shfl_down(acc.nf, offset)};
    }
};
```

#### [NEW] `MeanOps<T>` functor

```cpp
template<typename T>
struct MeanOps {
    using AccT = AccumulatorType<T>;
    AccT factor;  // = 1.0 / reduced_count

    DEVICE_HOST AccT identity() const { return AccT(0); }
    DEVICE_HOST AccT reduce(AccT acc, AccT val, int64_t) const { return acc + val; }
    DEVICE_HOST AccT combine(AccT a, AccT b) const { return a + b; }
    DEVICE_HOST AccT project(AccT a) const { return a * factor; }
    __device__ AccT warp_shfl_down(AccT val, int offset) const;
};
```

> [!IMPORTANT]
> Phase 1 makes NO kernel changes. Only extends `ReductionOps.h`. Must compile cleanly before proceeding.

---

### Phase 2: GpuReduceConfig Solver

#### [NEW] `include/ops/helpers/GpuReduceConfig.cuh`

Host-side configuration solver that replaces the hard-coded `threads=256, blocks=num_slices`.

```cpp
struct GpuReduceConfig {
    int64_t num_inputs;     // elements per output (reduction dimension size)
    int64_t num_outputs;    // total output elements
    int element_size_bytes;

    int block_width;        // blockDim.x (warp-aligned)
    int block_height;       // blockDim.y
    int num_threads;

    int input_mult[3];      // [BLOCK_X, BLOCK_Y, CTA] split multipliers
    int output_mult[2];     // [BLOCK_X, BLOCK_Y]
    int ctas_per_output;    // >1 = global reduce needed

    bool vectorize_input;
    int output_vec_size;
    bool reduction_on_fastest_stride;

    // Decision functions
    bool should_block_x_reduce() const { return input_mult[0] != 0; }
    bool should_block_y_reduce() const { return input_mult[1] != 0; }
    bool should_global_reduce()  const { return ctas_per_output > 1; }

    dim3 block() const;
    dim3 grid() const;
    int64_t shared_memory_size(int acc_size) const;
};
```

The solver logic mirrors PyTorch's `setReduceConfig` (Reduce.cuh:1032-1178):
1. Detect `reduction_on_fastest_stride` from tensor strides
2. Set `block_width` (up to 512) and `block_height` (up to 256/block_width)
3. `split_input` vs `split_output` based on Inner/Outer
4. If `values_per_thread >= 256` and GPU is underutilized → `ctas_per_output > 1`

---

### Phase 3: Stride-Based OffsetCalculator

#### [NEW] `include/ops/helpers/GpuOffsetCalculator.cuh`

Replaces the per-element unravel/ravel with O(1) stride-based access for 95% of cases:

```cpp
template<int NARGS = 1, typename index_t = uint32_t>
struct OffsetCalculator {
    int dims;
    index_t sizes[MAX_DIMS];
    index_t strides[NARGS][MAX_DIMS];

    __device__ Array<index_t, NARGS> get(index_t linear_idx) const {
        // O(ndim) divmod — ONLY used for Generic path (<5% of DL cases)
    }
};
```

The kernel uses a **3-tier access pattern**:
```cpp
// Tier 1: Contiguous (Inner reduction, stride=1) — ~70% of DL
data[i]                                         // Direct index

// Tier 2: Single-stride (Outer reduction) — ~25% of DL
data[i * element_stride]                        // One multiply

// Tier 3: Generic (non-consecutive axes) — ~5% of DL
data[offset_calc.get(i)]                        // O(ndim) divmod fallback
```

---

### Phase 4: Unified GPU Reduction Kernel

#### [MODIFY] `include/ops/helpers/ReductionKernels.cuh`

**Delete** the 4 old kernels. **Replace** with 1 unified kernel:

```cpp
template<typename scalar_t, typename ops_t, typename index_t, typename out_t, int vt0=4>
struct ReduceOp {
    ops_t ops;
    GpuReduceConfig config;
    OffsetCalculator<1, index_t> input_calc;
    OffsetCalculator<1, index_t> output_calc;
    const scalar_t* src;
    out_t* dst;
    void* acc_buf;
    void* cta_buf;
    int* semaphores;
    using arg_t = typename ops_t::acc_t;

    __device__ void run() const {
        extern __shared__ char shared_memory[];

        // Stage 1: Thread-local reduction
        arg_t value = thread_reduce(src);

        // Stage 2: Warp-level X-reduce (register shuffles)
        if (config.should_block_x_reduce())
            value = block_x_reduce(value, shared_memory);

        // Stage 3: Block-level Y-reduce (shared memory)
        if (config.should_block_y_reduce())
            value = block_y_reduce(value, shared_memory);

        // Stage 4: Grid-level global reduce (atomics/semaphores)
        if (config.should_global_reduce())
            value = global_reduce(value, cta_buf, semaphores, shared_memory);

        // Write output via ops.project()
        if (should_store())
            dst[output_idx] = static_cast<out_t>(ops.project(value));
    }
};
```

**What each old kernel maps to:**
| Old Kernel | New Functor | `acc_t` | `project()` output |
|:---|:---|:---|:---|
| `reduce_kernel<SumOp>` | `SumOp<T>` | `AccT` (scalar) | value as-is |
| `reduce_kernel<MinOp>` | `MinOp<T>` | `T` | value as-is |
| `reduce_index_kernel<ArgMinOp>` | `ArgMinOp<T>` | `ValueIndex<T>` | `.index` (int64) |
| `reduce_mean_kernel` | `MeanOps<T>` | `AccT` | `value * (1/count)` |
| `reduce_variance_kernel` | `WelfordOps<T>` | `WelfordData<AccT>` | `m2 / (n - correction)` |

---

### Phase 5: Dispatcher Refactor

#### [MODIFY] `src/UnaryOps/cuda/ReductionImplGPU.cu`

Replace 4 separate `dispatch_*_gpu` functions with ONE:

```cpp
template<typename T, typename ops_t, typename out_t>
Tensor gpu_reduce_dispatch(const Tensor& input, const std::vector<int64_t>& axes,
                            bool keepdim, ops_t ops, cudaStream_t stream) {
    auto layout = compute_reduction_layout(input, axes);
    auto config = build_reduce_config<typename ops_t::acc_t>(layout);

    // Allocate global memory if multi-CTA
    if (config.should_global_reduce()) { /* buffer + semaphores */ }

    // Build offset calculators from strides
    auto input_calc = make_input_calculator(input, axes);
    auto output_calc = make_output_calculator(output);

    // Launch unified kernel
    auto reduce_op = ReduceOp<CudaT, ops_t, uint32_t, OutCudaT>(
        ops, config, input_calc, output_calc, input_ptr, output_ptr, ...);
    launch_reduce_kernel(config, reduce_op);
}
```

**Callers simplified from:**
```cpp
// OLD (4 separate calls):
dispatch_reduction_gpu<float, SumOp>(...)
dispatch_index_reduction_gpu<float, ArgMinOp>(...)
dispatch_mean_gpu<float, SumOp>(...)
dispatch_variance_gpu<float, VarianceOp>(...)
```
**To:**
```cpp
// NEW (1 unified call):
gpu_reduce_dispatch<float>(input, axes, keepdim, SumOp<float>{}, stream);
gpu_reduce_dispatch<float>(input, axes, keepdim, ArgMinOp<float>{}, stream);
gpu_reduce_dispatch<float>(input, axes, keepdim, MeanOps<float>{1.0f/count}, stream);
gpu_reduce_dispatch<float>(input, axes, keepdim, WelfordOps<float>{correction, take_sqrt}, stream);
```

---

## Execution Order

| Step | Phase | What | Verify |
|:---:|:---|:---|:---|
| 1 | Phase 1 | Add `combine`/`project`/`warp_shfl_down` to existing functors | `make` compiles cleanly |
| 2 | Phase 1 | Add `WelfordData`, `WelfordOps`, `MeanOps` | `make` compiles cleanly |
| 3 | Phase 2 | Implement `GpuReduceConfig` | Unit test: config matches expected block/grid for known shapes |
| 4 | Phase 3 | Implement `OffsetCalculator` | Unit test: offset matches unravel for known indices |
| 5 | Phase 4 | Write unified kernel (SumOp only first) | Run `sum` tests, compare output with old kernel |
| 6 | Phase 4 | Wire up MinOp/MaxOp/ProdOp | Run all value reduction tests |
| 7 | Phase 4 | Wire up ArgMinOp/ArgMaxOp | Run all index reduction tests |
| 8 | Phase 4 | Wire up MeanOps | Run mean/nanmean tests |
| 9 | Phase 4 | Wire up WelfordOps | Run variance/std tests |
| 10 | Phase 5 | Refactor dispatcher | Full test suite, benchmark vs old |
| 11 | Cleanup | Delete old 4 kernels + old dispatchers | Final full test |

> [!CAUTION]
> Each step must compile + pass tests before proceeding to the next.

---

## Verification Plan

### Automated Tests
- Existing reduction test suite after each step
- Compare outputs (sum, mean, var, argmin) against PyTorch for 1D/2D/3D/4D tensors
- Edge cases: empty reduction, single-element, scalar output, keepdim=true/false

### Benchmarks
- `bench_nanmean_pytorch_vs_ours.cpp` — must match or beat old implementation
- New benchmark: sweep tensor shapes to verify config solver picks optimal config
