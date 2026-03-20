# Optimizations and Code-Changes Done in Reductions

## 1. Software / Code Optimizations and Fixes

### 1.1 GPU Caching Allocator Integration
**(File: `src/UnaryOps/cuda/ReductionImpl.cu`)**
Integrated our custom GPU Caching Allocator directly into the reduction kernels. By utilizing pre-cached memory blocks for intermediate operations instead of waiting for standard OS-level allocations via `cudaMalloc`, this drastically reduced kernel launch latency and overall allocation overhead ---> (In PackedMetaData struct , we used caching allocator for intermediate operations) .

### 1.2 Packed Metadata Transmission
**(File: `src/UnaryOps/cuda/ReductionImpl.cu`)**
Replaced individual `Device_Array` objects with a unified `PackedMetaData` object. This reduces the number of kernel arguments and memory transfers by sending all necessary shape, stride, and reduction axes metadata to the GPU in a single packed structure at once rather than one-by-oneover the PCIe bus in a single unified jump.


### 1.3 NaN Fix for Index-Returning Reductions (ArgMin / ArgMax) 
**(File: `include/ops/helpers/ReductionOps.h`)**
Addressed an unpredictable behavior where comparing two NaNs returned an undefined/arbitrary index. We explicitly added a check to correctly halt and propagate the first NaN's index deterministically:
```cpp
if (safe_isnan(a.value) && safe_isnan(b.value)) {
    return (a.index < b.index) ? a : b;
}
```
This ensures that if multiple NaNs are encountered, the function correctly returns the index of the first NaN it saw, maintaining consistent and predictable behavior.

### 1.4 Removed Unreachable Dead Code
Commented out dead code in `ReductionImpl.h` that was previously checking for an impossible state:
**(File: `include/ops/helpers/ReductionImpl.h`, Lines 137-145)**
Removed mathematically redundant runtime error checks. Code validating `reduced_count == 0 && input.numel() > 0` was commented out because `normalize_axes()` handles `numel() == 0` validation prior to this step, making this unreachable dead code.
```cpp
// if (reduced_count == 0 && input.numel() > 0) {
//     throw std::runtime_error("Reduction error: reduced count is zero but input has " + 
//                             std::to_string(input.numel()) + " elements.");
// }
```
This state is unreachable because the `normalize_axes` function already handles and validates the reduced axes from the input shape, correctly bypassing cases where `input.numel() == 0` (e.g., shape `[0, 3]`).

### 1.5 Bitmap Array Lookup inside OpenMP Threads [O(1) Hot-loop Lookup]
**(File: `include/ops/helpers/ReductionImpl.h`)**
Replaced 22 slow $O(N)$ execution path scans via `std::find()` scattered across our $O(num\_slices \times reduced\_count)$ hot-loops with an instantaneous $O(1)$ boolean array read (`reduced_bitmap[dim]`). The bitmap is preemptively built once just before the `#pragma omp parallel for` threading happens. This significantly accelerates CPU execution speeds for heavily chunked tensor operations where loop iterations hit millions.

simply,Replaced the O(N) `std::find(normalized_axes.begin(), ...)` calls inside the innermost parallel CPU loops with an O(1) boolean bitmap array (`bool reduced_bitmap[MAX_DIMS] = {false};`). The bitmap is initialized once before the OpenMP loop, completely removing unnecessary array traversals during every element's coordinate mapping.
Affected `is_reduced` `std::find` replacements natively lie at:
- `reduce_kernel()` Lines: **168, 220, 265, 339**
- `dispatch_mean_kernel()` Lines: **604, 631, 683, 712**
- `dispatch_variance_kernel()` Lines: **945, 970, 1011**

### 1.6 Kahan Summation Removed — Replaced by Double Accumulator
**(File: `include/ops/helpers/ReductionImpl.h`)**

Kahan compensated summation was active on the CPU for any `SumOp<T>` with a floating-point accumulator.
It was gated by:
```cpp
constexpr bool use_kahan = std::is_same_v<OpType<T>, SumOp<T>> &&
                           !std::is_same_v<AccT, ValueIndex<T>> &&
                           (std::is_floating_point_v<AccumulatorT> ||
                            std::is_same_v<AccumulatorT, double>);
```

**Why Kahan was removed:**
- With `float → double` accumulation (change 1.7 below), Kahan is redundant. Double gives lower absolute
  error than Kahan on float.
- Kahan has a serial dependency chain: each step requires the previous `kahan_c` value, preventing
  CPU out-of-order execution and pipelining.
- Kahan **fails equally with naive float** on catastrophic cancellation patterns such as
  `[1e8, 1.0, -1e8] × 1000`. At float precision, `(1e8 + 1.0) = 1e8` (1.0 < ULP at 1e8),
  so the compensation `c = (t - kahan_sum) - y = 0`. Result: 0 instead of 1000.
  Double accumulation fixes this completely because double ULP at 1e8 ≈ 1.5e-8 << 1.0.
- Affected ops: `reduce_sum` and `reduce_mean` on float/float16/bfloat16 tensors.

**What was deleted:**
- The `use_kahan` constexpr (4 lines)
- The entire `if constexpr (use_kahan) { ... } else {` block including the Kahan loop (lines ~249–312)
- The orphaned closing `}` of the old else block

**Baseline precision before removal (Kahan active, float accumulator):**

| Dataset | N | Kahan rel_err | Naive float rel_err | Factor |
|---------|---|--------------|--------------------|----|
| Catastrophic cancellation | 3001 | 0.9990 (FAIL) | 0.9990 (FAIL) | 1× (both wrong) |
| Cross-entropy losses | 65536 | 1.20e-8 | 4.31e-6 | 360× better |
| Gradient tensor | 1048576 | 1.67e-8 | 1.58e-5 | 946× better |
| Batch norm activations | 50176 | 1.56e-7 | 4.92e-5 | 316× better |
| Embedding weights | 786432 | 1.46e-8 | 2.24e-5 | 1534× better |

---

### 1.7 Unified Accumulator System — Two-Parameter Device-Aware Struct
**(File: `include/ops/helpers/ReductionOps.h`, `include/ops/helpers/ReductionKernels.cuh`)**

**Original problems:**
- Single-parameter `AccumulatorTypeSelector<T>` had no device awareness — CPU and GPU used the same
  type, but GPU needs different rules (double is 32× slower on GPU vs CPU).
- `int8_t` was missing entirely — fell through to `type = T`, causing immediate overflow.
- `uint8/16/32/64` mapped to `int64_t` — semantically wrong for values > INT64_MAX (wraps negative).
- `float` had no specialization — accumulated as `float` on CPU (large-N precision loss).
- GPU kernel used a completely separate inline `std::conditional_t<>` chain — diverged silently.

**Changes made:**

1. **Two-parameter struct** — `template<typename T, bool IsGPU = false>`:
   - `IsGPU=false` (default) = CPU path
   - `IsGPU=true` = GPU path
   - Single source of truth for both devices

2. **Partial specializations with `template<bool IsGPU>`** — for types where CPU and GPU agree:
   integers, bool, half, FP4. Full specializations only for `float` which differs between devices.

3. **float CPU → double** — `template<> struct AccumulatorTypeSelector<float, false> { using type = double; };`
   Reduces relative error from ~O(N × 1.2e-7) to ~O(float32_output_eps ≈ 1.2e-7) — bounded by
   float32 OUTPUT precision, not accumulation error.

4. **float GPU → float** — `template<> struct AccumulatorTypeSelector<float, true> { using type = float; };`
   GPU double is 32× slower; float accumulation is acceptable for GPU.

5. **uint8/16/32/64 → uint64_t** — semantically correct for unsigned types.

6. **int8_t added** — `template<bool IsGPU> struct AccumulatorTypeSelector<int8_t, IsGPU> { using type = int64_t; };`

7. **GPU kernel unified** — replaced inline conditional_t chain with one line:
   ```cpp
   using AccumulatorType = detail::AccumulatorType<T, /*IsGPU=*/true>;
   ```
   `/*IsGPU=*/true` is a C block comment label — compiler sees just `true`. Same style as PyTorch's
   `at::acc_type<scalar_t, /*is_cuda=*/true>`.

8. **`G` renamed to `IsGPU`** in partial specialization parameter names (readability only).

---

### 1.8 Int8 Dispatch Gap Fixed
**(Files: `include/core/TensorDispatch.h`, `src/UnaryOps/cuda/ReductionImplGPU.cu`)**

`dispatch_by_dtype` switch was missing `case Dtype::Int8:`, causing ALL reduction ops
(not just sum) to throw `"Unsupported Dtype"` at runtime for any Int8 tensor.

Also missing: explicit GPU template instantiations for `int8_t` in `ReductionImplGPU.cu`
(would have caused linker errors for Int8 tensors on CUDA device).

Both gaps fixed — 9 GPU instantiations added, matching the pattern of `int16_t`.

---

### 1.9 Debug `abi::__cxa_demangle` Call Removed
**(File: `src/UnaryOps/cuda/ReductionImplGPU.cu`, lines 233–237)**

Every GPU reduction call was executing:
```cpp
int status;
std::unique_ptr<char, void(*)(void*)> demangled_name(
    abi::__cxa_demangle(typeid(OpType<T>).name(), nullptr, nullptr, &status),
    std::free
);
```
This is pure CPU string processing in the hot path — allocates a heap string, runs the demangler,
then immediately throws the result away (the variable was never read). Removed entirely.

---

## 2. Hardware Optimizations
*(Reserved for hardware-specific optimizations)*
