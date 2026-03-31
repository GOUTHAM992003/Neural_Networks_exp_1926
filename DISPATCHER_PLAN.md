# Universal Reduction Dispatcher — FINAL Implementation Plan

## PyTorch Reference (from actual code analysis)

```
parallel_reduce(loop):
  if (numel < GRAIN_SIZE || threads == 1) → serial_for_each
  else if (output.numel() == 1)           → two_pass_reduction (Strategy 2)
  else                                    → parallel_dim_reduction (Strategy 1)

binary_kernel_reduce(iter, ops, init):
  foreach_reduced_elt(sub_iter):          → iterates output elements
    if (numel < GRAIN_SIZE || threads == 1) → sequential
    else → buffer[max_threads] + parallel_for(0, numel, GRAIN_SIZE) + combine

binary_kernel_reduce_lastdim:
  sub_iter.for_each(loop, grain_size)     → parallelizes over output positions ONLY
  BUG: full reduction (1 output) → 1 thread does all work

argmax dispatch:
  if (is_reduce_lastdim) → binary_kernel_reduce_lastdim (fast scalar, no generic math)
  else → binary_kernel_reduce (generic stride math, but has Strategy 2)
```

## PyTorch Bug (CONFIRMED)
File: aten/src/ATen/native/cpu/Reduce.h:290-308
binary_kernel_reduce_lastdim uses sub_iter.for_each() which parallelizes ONLY over
output elements. For full reduction (output.numel()==1), exactly 1 thread runs.
argmax_kernel_impl routes contiguous full reduction to lastdim path.
Result: tensor.argmax() on 1-billion element contiguous tensor = 1 CPU thread.

## Our Current State

### Kernels:
- **cascade_sum_kernel**: Float sum/nansum, SIMD inner, 4-acc ILP outer/generic
- **reduce_kernel**: Int sum, prod, min, max, nanprod, nanmin, nanmax, all, any. SIMD for float/double min/max/prod inner
- **reduce_kernel_index**: Argmax/argmin/nanargmax/nanargmin. No SIMD. Has broken 3-level dispatcher (Level 2 condition wrong)

### What's wrong with current 3-level dispatcher in kernels:
- Level 2 uses `num_slices < max_threads` — should be `num_slices == 1` for full reduction
- Each kernel has its own copy of the dispatcher logic — redundant
- User wants: ONE universal dispatcher + kernels with 2 callable strategy paths

## Benchmark Results (i7-14700K, 28 threads)

```
Size     1T (μs)   2T      4T      8T      14T     28T     Best
1K       1.2       6.0     7.5     11.8    18.2    42.1    1T
10K      4.0       6.3     8.7     10.9    16.1    53.4    1T
50K      35.8      24.7    17.2    15.5    18.0    34.9    8T
100K     37.2      24.4    24.8    18.7    20.1    32.5    8T
500K     183       96      61      54      43      73      14T
1M       367       188     150     100     66      50      28T
10M      3786      1975    1067    708     525     323     28T
```

GRAIN_SIZE = 32768 validated: actual_threads = min(max_threads, reduced_count/GRAIN_SIZE)
- 50K/32768 = 1 thread (bench: 1T wins ✓)
- 500K/32768 = 15 threads (bench: 14T wins ✓)
- 10M/32768 = 305 → 28 threads (bench: 28T wins ✓)

---

## IMPLEMENTATION DESIGN

### Step 1: Strategy Enum + Thread Calculation

```cpp
enum class ReductionStrategy { ParallelSlices, SplitReduction };

// actual_threads: caps thread count based on GRAIN_SIZE
// When actual_threads = 1, both strategies are effectively sequential (no separate path needed)
int actual_threads = std::min(omp_get_max_threads(), std::max(1, (int)(reduced_count / GRAIN_SIZE)));
```

### Step 2: Each Kernel Has 2 Strategy Paths (callable with parameter)

```cpp
Tensor cascade_sum_kernel<ignore_nan, T>(input, axes, shape, ReductionStrategy strategy, int num_threads);
Tensor reduce_kernel<T, OpType, AccT>(input, axes, shape, ReductionStrategy strategy, int num_threads);
Tensor reduce_kernel_index<T, OpType>(input, axes, shape, ReductionStrategy strategy, int num_threads);
```

Inside each kernel:
```cpp
if (strategy == ReductionStrategy::ParallelSlices) {
    // Strategy 1: #pragma omp parallel for num_threads(num_threads) over output slots
    // Each thread handles complete output positions
    // Layout dispatch (Inner/Outer/Generic) happens inside per-position loop
} else {
    // Strategy 2: For each output, split reduction across threads
    // #pragma omp parallel num_threads(num_threads) with thread-local accs
    // Layout dispatch (Inner/Outer/Generic) happens inside per-thread chunk
    // Combine at end
}
```

- cascade_sum + reduce_kernel: SIMD + OpenMP (both strategies)
- reduce_kernel_index: OpenMP only, no SIMD (both strategies)
- When num_threads=1: both strategies degenerate to sequential (no extra path needed)

### Step 3: Universal Dispatcher (in dispatch_reduction)

```cpp
int max_threads = omp_get_max_threads();

// 3 cases
ReductionStrategy strategy;
int actual_threads;

// CASE 1: SMALL TENSOR — not worth threading
// (matches: if (numel < GRAIN_SIZE || threads == 1) → serial_for_each)
if (input.numel() < GRAIN_SIZE || max_threads == 1) {
    actual_threads = 1;
    // Either strategy with 1 thread = effectively sequential. Pick Strategy 1 (simpler path).
    strategy = ReductionStrategy::ParallelSlices;
}
// CASE 2: FULL REDUCTION — always Strategy 2
// (matches: else if (output.numel() == 1) → two_pass_reduction)
else if (num_slices == 1) {
    actual_threads = std::min(max_threads, std::max(1, (int)(reduced_count / GRAIN_SIZE)));
    strategy = ReductionStrategy::SplitReduction;
}
// CASE 3: PARTIAL REDUCTION — choose strategy based on output slots vs threads
// (matches: else → parallel_dim_reduction)
else {
    actual_threads = std::min(max_threads, std::max(1, (int)(reduced_count / GRAIN_SIZE)));
    if (num_slices >= actual_threads) {
        strategy = ReductionStrategy::ParallelSlices;   // enough output slots
    } else {
        strategy = ReductionStrategy::SplitReduction;   // few outputs, split reduction
    }
}

// 3. Route to kernel
if (is_index_op)
    return reduce_kernel_index<T, OpType>(input, axes, shape, strategy, actual_threads);
else if (is_float_sum)
    return cascade_sum_kernel<ignore_nan, T>(input, axes, shape, strategy, actual_threads);
else
    return reduce_kernel<T, OpType, AccT>(input, axes, shape, strategy, actual_threads);
```

### Step 4: Delete Redundant Code
- Remove old 3-level dispatcher from inside each kernel
- Remove old lambda wrappers (reduce_and_store_one)
- Remove old TODO comments about dispatcher

---

## IMPLEMENTATION ORDER

1. Add `ReductionStrategy` enum + GRAIN_SIZE thread capping logic
2. Refactor reduce_kernel_index: remove old 3-level, add 2-path (Strategy 1 / Strategy 2)
3. Refactor cascade_sum_kernel: remove old 3-level, add 2-path
4. Refactor reduce_kernel: remove old 3-level, add 2-path
5. Move strategy selection to dispatch_reduction (universal dispatcher)
6. Add PyTorch bug comment in dispatcher
7. Build + test
8. Clean up redundant .md files

---

## WHAT WE ARE NOT CHANGING
- Vectorized.h (AVX2 engine) — done ✅
- ReductionOps.h (ProductAccumType) — done ✅
- Makefile (AVX2 flags) — done ✅
- Layout detection (InnerContiguous/OuterContiguous/Generic) — done ✅
- SIMD paths in InnerContiguous — done ✅
- Routing logic (which ops go to which kernel) — done ✅
