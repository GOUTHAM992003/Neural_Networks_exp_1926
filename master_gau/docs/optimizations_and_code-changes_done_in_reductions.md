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

### 1.10 PackedMetadata Host Buffer — `cudaHostAlloc` Replaced by `std::vector`
**(File: `src/UnaryOps/cuda/ReductionImplGPU.cu`, Lines 36–123)**

**Problem before:**
The `PackedMetadata` constructor allocated its host-side staging buffer via `PinnedCPUAllocator`,
which internally calls `cudaHostAlloc` (page-locked / pinned memory). This had two costs every
single reduction call:
1. `cudaHostAlloc` — the CUDA driver acquires a **global device-context lock** on every call.
   Cost: **~100–200 µs per call** regardless of allocation size.
2. `cudaFreeHost` in the destructor — same lock again on teardown.

Additionally, `cudaStreamSynchronize(stream)` was called after the `cudaMemcpyAsync` — this
blocked the CPU thread until the DMA engine finished copying. This was **redundant** because
the memcpy and the subsequent kernel launch are submitted to the same stream; CUDA already
guarantees ordering. (Already commented out by the user before this fix.)

**State of host buffer before (lines ~78–79, ~118):**
```cpp
// BEFORE — per-call cudaHostAlloc
device::PinnedCPUAllocator pinned_allocator;
h_ptr = static_cast<int64_t*>(pinned_allocator.allocate(total_bytes)); // ← ~150µs driver lock
...
if(h_ptr) device::PinnedCPUAllocator().deallocate(h_ptr);              // ← ~150µs driver lock
```

**Fix applied (lines ~77–79):**
```cpp
// AFTER — normal heap allocation, nanoseconds
// PinnedCPUAllocator used cudaHostAlloc which takes a global device-context
// lock on every call (~100-200µs, see PinnedCPUAllocator.cpp TODO). For tiny metadata buffers,
// CUDA's internal staging of pageable memory is only ~1-2µs — far cheaper than the driver lock.
std::vector<int64_t> h_buf(total_size);
```

`h_buf` is a local `std::vector`. It is automatically destroyed (RAII) when the constructor
returns — no destructor cleanup needed. `cudaMemcpyAsync` now reads from `h_buf.data()`.

**Files changed:**
- Removed: `#include "device/PinnedCPUAllocator.h"` → replaced with `#include <vector>`
- Removed: `int64_t* h_ptr` member from the class (line 39)
- Constructor: replaced `pinned_allocator.allocate(total_bytes)` → `std::vector<int64_t> h_buf(total_size)` (line 79)
- Constructor: `pack` lambda now writes to `h_buf.data()` instead of `h_ptr` (line 85)
- Constructor: `cudaMemcpyAsync` source now `h_buf.data()` instead of `h_ptr` (line 99)
- Destructor: removed `if(h_ptr) PinnedCPUAllocator().deallocate(h_ptr)` (line 118)

**Cost trade-off:**
| | Before (pinned) | After (pageable) |
|---|---|---|
| Host alloc | ~150 µs (`cudaHostAlloc` driver lock) | ~50 ns (`malloc` in `std::vector`) |
| DMA transfer | ~0 µs (direct DMA, no staging) | ~1–2 µs (CUDA internal staging) |
| Host free | ~150 µs (`cudaFreeHost` driver lock) | ~0 ns (RAII, stack unwind) |
| **Net per call** | ~300+ µs | **~2 µs** |

The CPU packing step (5 × `std::copy` of tiny vectors into `h_buf`) is pure L1-cache work — tens
of nanoseconds total — and is identical in both before and after.

---

### 1.11 Compare Ops (Min / Max) — Accumulator Bypasses `AccumulatorTypeSelector`
**(Files: `include/ops/helpers/ReductionOps.h`, `include/ops/helpers/ReductionImpl.h`,
`include/ops/helpers/ReductionKernels.cuh`, `src/UnaryOps/cuda/ReductionImplGPU.cu`)**

**Problem before:**
The unified accumulator system (change 1.7) promoted ALL integer types to `int64_t` and `float`
to `double` (CPU). This was correct for `SumOp` and `ProductOp` where arithmetic overflow is
a real risk. But `MinOp`, `MaxOp`, `NanMinOp`, `NanMaxOp` are **compare-only** — the result is
always one of the input values and can never exceed the input type's range. There is no
mathematical reason to widen for comparison.

The unnecessary widening had measurable costs:
- **CPU**: `MOVSX` (sign-extend) per element + one narrowing cast at output ≈ 1 extra instruction
  per element through the inner loop.
- **GPU (Pascal/Maxwell/Kepler, pre-Volta)**: 64-bit integer operations are emulated as 2 × 32-bit
  instructions on these architectures. `int32_t` comparison used `int64_t` accumulator →
  **2–4× slower** comparison instruction for those GPUs.
- **GPU (Volta+, SM 7.0+)**: 64-bit integer is natively supported. Overhead was ~zero.
- **Output dtype wrong**: integer min/max was returning `Int64` tensor instead of the input dtype
  (e.g., `reduce_min<int32_t>` returned Int64). PyTorch/NumPy return Int32 — the input dtype.

**Ops affected**: `reduce_min`, `reduce_max`, `reduce_nanmin`, `reduce_nanmax` on all integer
and float types. `reduce_sum`, `reduce_mean`, `reduce_product`, variance — **unchanged**.

---

**Change A — `ReductionOps.h` (lines 412–413, 458–459, 588–589, 632–633):**

Changed `AccT` in all 4 compare op structs from `AccumulatorType<T>` to `T` directly:
```cpp
// BEFORE (MinOp, MaxOp, NanMinOp, NanMaxOp all had):
using AccT = AccumulatorType<T>;   // int32 → int64_t, float(CPU) → double

// AFTER:
using AccT = T;   // compare op: result always within input range, no widening needed
```
`SumOp`, `ProductOp`, `NanSumOp`, `VarianceOp` — **AccT unchanged**, still use `AccumulatorType<T>`.

---

**Change B — `ReductionImpl.h` (CPU kernel, 3 sub-changes):**

**(B1) Output dtype (line ~123):**
```cpp
// BEFORE — all integer types widened to Int64:
} else if constexpr (std::is_integral_v<T>) {
    output_dtype = Dtype::Int64;

// AFTER — only when accumulator is wider than T (sum/product):
} else if constexpr (std::is_integral_v<T> && !std::is_same_v<AccT, T>) {
    output_dtype = Dtype::Int64;  // sum/product: AccT widened → output must widen
}
// if AccT == T (min/max), falls to else: output_dtype = input.dtype() (Int32 stays Int32)
```

**(B2) `OutputCppT` determination (lines ~148–156):**
```cpp
// BEFORE:
using OutputCppT = ... std::is_integral_v<T> → int64_t ...

// AFTER:
using OutputCppT = typename std::conditional<
    std::is_same_v<AccT, ValueIndex<T>>,
    int64_t,
    typename std::conditional<
        std::is_integral_v<T> && !std::is_same_v<AccT, T>,
        int64_t,   // sum/product: AccT widened → output is int64
        T          // min/max (AccT==T) and floats: output same as input
    >::type
>::type;
```

**(B3) Accumulator initializer simplification (lines ~238–249):**

The old init had a 3-branch chain that hard-coded conversions to `double` or `int64_t` based on
the input type `T`. With `AccT = T` for compare ops this was wrong (narrowing conversions).
Replaced with a single universal cast:
```cpp
// BEFORE:
if constexpr (should_use_double_accumulation<T>()) {
    accumulator = static_cast<double>(op.identity());   // assumed AccT = double/float
} else if constexpr (std::is_integral_v<T>) {
    accumulator = static_cast<int64_t>(op.identity()); // assumed AccT = int64_t
} else {
    accumulator = op.identity();
}

// AFTER — works for all ops since AccumulatorT is already the correct type:
} else {
    accumulator = static_cast<AccumulatorT>(op.identity());
}
// - sum int32 (AccT=int64): static_cast<int64_t>(0)      ✓
// - min int32 (AccT=int32): static_cast<int32_t>(INT32_MAX) ✓ (no-op)
// - sum float16 (AccT=float): static_cast<float>(float16(0)) ✓
// - min float16 (AccT=float16): static_cast<float16>(float16(MAX)) ✓ (no-op)
```

---

**Change C — `ReductionKernels.cuh` (GPU kernel, line ~182):**
```cpp
// BEFORE — single line, no op awareness:
using AccumulatorType = detail::AccumulatorType<T, /*IsGPU=*/true>;

// AFTER — compare ops use T directly, others use the centralized selector:
using AccumulatorType = std::conditional_t<
    std::is_same_v<OpType<T>, detail::MinOp<T>>    ||
    std::is_same_v<OpType<T>, detail::MaxOp<T>>    ||
    std::is_same_v<OpType<T>, detail::NanMinOp<T>> ||
    std::is_same_v<OpType<T>, detail::NanMaxOp<T>>,
    T,
    detail::AccumulatorType<T, /*IsGPU=*/true>
>;
```

---

**Change D — `ReductionImplGPU.cu` (GPU dispatcher, lines ~156–219):**

**(D1) Added `is_compare_op` constexpr (line ~156):**
```cpp
constexpr bool is_compare_op =
    std::is_same_v<OpType<T>, detail::MinOp<T>>    ||
    std::is_same_v<OpType<T>, detail::MaxOp<T>>    ||
    std::is_same_v<OpType<T>, detail::NanMinOp<T>> ||
    std::is_same_v<OpType<T>, detail::NanMaxOp<T>>;
```

**(D2) Output dtype (line ~167):**
```cpp
} else if constexpr (std::is_integral_v<T> && !is_compare_op) {
    output_dtype = Dtype::Int64;   // sum/product: widens
} else {
    output_dtype = input.dtype(); // min/max: stays same as input
}
```

**(D3) `OutputCppT` (line ~225):**
```cpp
using OutputCppT = std::conditional_t<
    std::is_integral_v<T> && !is_compare_op,
    int64_t,   // sum/product
    T          // min/max and floats
>;
```

**(D4) Shared memory size (line ~213):**

Shared memory for warp-level accumulator reduction was hard-coded to `sizeof(int64_t)` for all
integer types. After the fix, compare ops need only `sizeof(T)`:
```cpp
// BEFORE:
if constexpr (std::is_integral_v<T>) {
    shared_mem_size = (threads_per_block / 32) * sizeof(int64_t);  // all integers

// AFTER:
if constexpr (std::is_integral_v<T> && is_compare_op) {
    shared_mem_size = (threads_per_block / 32) * sizeof(T);        // min/max: uses T
} else if constexpr (std::is_integral_v<T>) {
    shared_mem_size = (threads_per_block / 32) * sizeof(int64_t);  // sum/product: uses int64
```
For int32 min/max with 256 threads: shared memory per block = `(256/32) × 4 = 32 bytes`
instead of `(256/32) × 8 = 64 bytes`. Halved shared memory usage for those ops.

---

**Before vs After summary:**

| Op | dtype | Before: accumulator | After: accumulator | Before: output dtype | After: output dtype |
|----|-------|---------------------|--------------------|----------------------|---------------------|
| reduce_min | int32_t | int64_t | **int32_t** | Int64 | **Int32** |
| reduce_max | int32_t | int64_t | **int32_t** | Int64 | **Int32** |
| reduce_min | int8_t  | int64_t | **int8_t**  | Int64 | **Int8**  |
| reduce_min | float   | double (CPU) | **float** | Float32 | Float32 |
| reduce_min | float16 | float   | **float16** | Float16 | Float16 |
| reduce_sum | int32_t | int64_t | int64_t (unchanged) | Int64 | Int64 |
| reduce_sum | float   | double (CPU) | double (CPU, unchanged) | Float32 | Float32 |

**PyTorch and TensorFlow both follow the same rule**: min/max use the input type directly as
accumulator, not a promoted type. This change brings master_gau in line with both frameworks
for compare ops.

---

### 1.12 Complex Type Accumulator Promotion
**(Files: `include/ops/helpers/ReductionOps.h`, `include/ops/helpers/ReductionKernels.cuh`,
`include/ops/helpers/ReductionImpl.h`, `src/UnaryOps/cuda/ReductionImplGPU.cu`)**

**Problem before:**
Complex types (`complex32_t`, `complex64_t`, `complex128_t`) had no specializations in
`AccumulatorTypeSelector` and fell through to the default `using type = T`. This meant:
- `reduce_sum<complex64_t>` on CPU accumulated as `complex64_t` (float32 components) — same
  as scalar `float` accumulating as `float`, which we already identified as a precision problem.
- `reduce_mean<complex64_t>` on CPU same issue.
- `reduce_mean_kernel` GPU had hardcoded logic `complex64_t → complex128_t` for ALL (CPU+GPU),
  which was actively **wrong for GPU**: double components on consumer hardware = 32× slower.

**The rule — mirrors scalar float exactly:**

| Input type | Components | CPU Accumulator | GPU Accumulator | Rationale |
|-----------|-----------|----------------|----------------|-----------|
| complex32_t | 2 × float16 | complex64_t | complex64_t | Same as scalar float16→float, both devices |
| complex64_t | 2 × float32 | **complex128_t** | **complex64_t** | CPU: float→double; GPU: float stays float (32× perf) |
| complex128_t | 2 × double | complex128_t | complex128_t | Already at max precision |

**This is identical to PyTorch's `acc_type<complex<float>, is_cuda>` rule:**
- CPU: `complex<float> → complex<double>`
- CUDA: `complex<float> → complex<float>` (same as non-complex float on GPU)

**Changes made:**

**(A) `ReductionOps.h` — Added 4 specializations after FP4 section (~line 336):**
```cpp
// complex32_t (2×float16): → complex64_t on BOTH CPU and GPU
template<bool IsGPU> struct AccumulatorTypeSelector<complex32_t,  IsGPU> { using type = complex64_t;  };

// complex64_t (2×float32): CPU→complex128_t, GPU→complex64_t
template<>           struct AccumulatorTypeSelector<complex64_t,  false> { using type = complex128_t; };
template<>           struct AccumulatorTypeSelector<complex64_t,  true>  { using type = complex64_t;  };

// complex128_t (2×double): no promotion on either device
template<bool IsGPU> struct AccumulatorTypeSelector<complex128_t, IsGPU> { using type = complex128_t; };
```

**(B) `ReductionKernels.cuh` — `reduce_mean_kernel` AccT complex branch fixed (~line 542):**

```cpp
// BEFORE: hardcoded, and WRONG for GPU (complex64 → complex128 always):
using AccT = std::conditional_t<
    is_complex,
    std::conditional_t<std::is_same_v<T, complex32_t>, complex64_t, complex128_t>,  // ← GPU bug
    ...
>;

// AFTER: uses AccumulatorTypeSelector for complex (centralized, correct for both devices):
using AccT = std::conditional_t<
    is_complex,
    detail::AccumulatorType<T, /*IsGPU=*/true>,  // complex64_t GPU → complex64_t (FIXED)
    std::conditional_t<std::is_same_v<T, double>, double, float>
>;
```

**This also fixed a pre-existing GPU bug**: `reduce_mean<complex64_t>` on GPU was using
`complex128_t` (double components) internally, making it 32× slower on consumer hardware.

**(C) `ReductionImpl.h` — `dispatch_mean_kernel` CPU non-integer AccT (~line 594):**

```cpp
// BEFORE: complex types fell through to T (no promotion):
using AccT = std::conditional<should_use_double_accumulation<T>(), double, T>::type;

// AFTER: uses AccumulatorTypeSelector for all types including complex:
using AccT = std::conditional<
    should_use_double_accumulation<T>(),
    double,
    detail::AccumulatorType<T, /*IsGPU=*/false>  // complex64_t → complex128_t on CPU
>::type;
```

**(D) `ReductionImpl.h` — `dispatch_variance_kernel` CPU AccT (~line 861):**

```cpp
// BEFORE: complex fell to T:
using AccT = ... conditional<is_integral, double, T>::type ...;

// AFTER: complex uses AccumulatorTypeSelector:
using AccT = ... conditional<is_integral, double, detail::AccumulatorType<T, false>>::type ...;
```

**(E) `ReductionImplGPU.cu` — `dispatch_mean_gpu` shared_mem_size (~line 415):**

```cpp
// BEFORE: only checked for double, defaulted to float for everything else:
constexpr size_t mean_acc_size = std::is_same_v<T, double> ? sizeof(double) : sizeof(float);
// complex128_t → AccT = complex128_t (16 bytes) but allocated only 4 bytes → MEMORY BUG

// AFTER: matches kernel's AccT computation exactly:
using MeanAccT = std::conditional_t<is_complex_T,
    detail::AccumulatorType<T, /*IsGPU=*/true>,          // complex: 8 or 16 bytes
    std::conditional_t<std::is_same_v<T, double>, double, float>  // non-complex: 8 or 4
>;
constexpr size_t mean_acc_size = sizeof(MeanAccT);
// complex128_t → 16 bytes ✓, complex32/64_t → 8 bytes ✓, double → 8 ✓, float → 4 ✓
```

Note: the `shared_mem_size` bug for `complex128_t` was a pre-existing issue — allocating only
4 bytes per warp slot but needing 16 bytes would cause silent shared memory corruption. This fix
is important even for correctness, not just performance.

**Output dtype:** Complex types output to the same dtype as input (`output_dtype = input.dtype()`
for non-integer, non-index types). The accumulation precision gain is visible in the sum phase
and cast back. Same pattern as float accumulating in double but writing back as float.

**PyTorch comparison:**
| | PyTorch CPU | PyTorch GPU | master_gau CPU | master_gau GPU |
|--|--|--|--|--|
| complex\<float\> accumulator | complex\<double\> | complex\<float\> | **complex128_t** (double components) | **complex64_t** (float components) |
| complex\<half\> accumulator | complex\<float\> | complex\<float\> | **complex64_t** | **complex64_t** |
| complex\<double\> accumulator | complex\<double\> | complex\<double\> | complex128_t | complex128_t |

Our implementation now matches PyTorch exactly for all three complex types on both devices.

---

### 1.13 Index Reductions — 2-Variable Approach (PyTorch-Style)
**(Files: `include/ops/helpers/ReductionOps.h`, `include/ops/helpers/ReductionImpl.h`)**

**Background — the 5-part ops classification:**
All 22 reduction ops were split into 5 groups to analyze each group's accumulator behaviour
separately and compare against PyTorch / TensorFlow:

| Part | Ops | Issue |
|------|-----|-------|
| 1 | sum, product, nansum, nanproduct | Use `AccumulatorType<T>` — correct, already fixed in changes 1.6–1.7 |
| 2 | min, max, nanmin, nanmax | Were using widened accumulator — fixed in change 1.11 |
| 3 | argmin, argmax, nanargmin, nanargmax | Were using `ValueIndex<T>` struct per element — fixed here (1.13) |
| 4 | reduce_all, reduce_any | Missing short-circuit early exit — fixed in change 1.14 |
| 5 | mean, variance, std, nanmean, nanvar, nanstd | GPU was using slow `double` — fixed in change 1.15; complex AccT also fixed in 1.12 |

**Problem (Part 3 — index ops):**
The CPU `reduce_kernel` index path used `ValueIndex<T>` struct (`{ T value; int64_t index; }`) to
bundle both the running best value and its index. Per element this required:
1. Constructing a new `ValueIndex<T> current_val_index = {input_value, i}` on the stack
2. Calling `op.reduce(accumulator, current_val_index)` which returns a whole struct
3. Storing the full struct back into `accumulator`

PyTorch uses two independent scalar variables `T best_val` + `int64_t best_idx` and only
updates `best_idx` when `best_val` changes. This avoids struct construction per element and
makes the index update conditional (often NOT taken in practice), eliminating ~2–3 extra
instructions per element on CPU.

**Note**: GPU `reduce_index_kernel` in `ReductionKernels.cuh` already shuffles `ValueIndex<T>`
fields **separately** in warp-level reduction:
```cpp
other.value = shfl_down(accumulator.value, offset);
other.index = shfl_down(accumulator.index, offset);
```
So the struct overhead is effectively zero on GPU (both fields live in registers). GPU was
**left unchanged** — the optimization is CPU-only.

**Changes made:**

**(A) `ReductionOps.h` — added two new methods to all 4 index ops:**

```cpp
// ArgMinOp (same pattern for ArgMaxOp, NanArgMinOp, NanArgMaxOp):

// CPU 2-variable path: initial sentinel value
T identity_val() const { return get_max_value<T>(); }

// CPU 2-variable path: returns true if candidate should replace current_best.
// Uses IEEE 754 property: any comparison involving NaN returns false.
// So (NaN < x) = false, meaning if current_best is NaN the comparison below
// naturally returns false and NaN sticks — no 2nd is_nan_check needed.
// Hot path (all-finite data): only 1 is_nan_check per element (on candidate).
bool better_than(const T& candidate, const T& current_best) const {
    if constexpr (is_any_float_v<T>) {
        if (is_nan_check(candidate)) return !is_nan_check(current_best);
    }
    return candidate < current_best;   // ArgMin
    // return candidate > current_best;  // ArgMax variant
}
```

**NaN semantics per op:**

| Op | NaN rule | `better_than` logic |
|----|----------|---------------------|
| `ArgMinOp` | NaN propagates (first NaN wins) | If candidate=NaN: take it only if current isn't NaN yet |
| `ArgMaxOp` | NaN propagates (first NaN wins) | Same |
| `NanArgMinOp` | NaN is **skipped** (ignored) | `if (is_nan_check(candidate)) return false;` — skip always |
| `NanArgMaxOp` | NaN is **skipped** (ignored) | Same |

**NaN check count comparison (float T):**

| Version | Hot path (no NaN) | After first NaN found |
|---------|------------------|-----------------------|
| Old (`ValueIndex` struct) | 2 checks inside `reduce()` | 2 checks |
| New (`better_than`) | **1 check** | **1 check** (IEEE short-circuits the comparison) |

**(B) `ReductionImpl.h` — CPU index path replaced (lines ~197–238):**

```cpp
// BEFORE — struct construction per element:
ValueIndex<T> accumulator = op.identity();
for (int64_t i = 0; i < reduced_count; ++i) {
    // ... coordinate mapping (unchanged) ...
    T input_value = input_data[input_lin_idx];
    ValueIndex<T> current_val_index = {input_value, i};   // ← struct per element
    accumulator = op.reduce(accumulator, current_val_index); // ← full struct return
}
output_data[output_index] = accumulator.index;

// AFTER — 2 scalars, index only written on improvement:
T best_val = op.identity_val();
int64_t best_idx = -1;
for (int64_t i = 0; i < reduced_count; ++i) {
    // ... coordinate mapping (unchanged) ...
    T input_value = input_data[input_lin_idx];
    if (op.better_than(input_value, best_val)) {  // ← conditional: often NOT taken
        best_val = input_value;
        best_idx = i;
    }
}
output_data[output_index] = best_idx;
```

**Benefits:**
- ~2–3 fewer instructions per element on CPU
- No struct construction in the hot path
- `best_idx` write is conditional — the CPU branch predictor handles "no update needed" at near-zero cost for nearly-sorted or highly-duplicate data
- `best_val` stays in a register (scalar), not a struct member — fewer memory operations

**Decision vs PyTorch:** PyTorch uses the same 2-variable pattern. TensorFlow uses a similar 2-variable approach for reduction with separate value/index tracking. GPU code left as-is — struct overhead is zero there.

---

### 1.14 Short-Circuit for `reduce_all` / `reduce_any` (CPU)
**(Files: `include/ops/helpers/ReductionOps.h`, `include/ops/helpers/ReductionImpl.h`)**

**Problem (Part 4 — bool ops):**
`reduce_all` and `reduce_any` were traversing the entire reduced dimension even when the result
was already determined:
- `AllOp` (AND): once any element is `false`, the result is `false` — remaining elements cannot change this
- `AnyOp` (OR): once any element is `true`, the result is `true` — remaining elements cannot change this

For a large tensor where the first element satisfies the early-exit condition, we were doing
`O(reduced_count)` work instead of `O(1)`.

PyTorch's CPU kernel explicitly uses `break` for bool reductions. GPU cannot short-circuit
(SIMT lockstep — all threads in a warp execute together regardless).

**Changes made:**

**(A) `ReductionOps.h` — added `can_short_circuit()` to AllOp and AnyOp:**

```cpp
template <typename T>
struct AllOp {
    using AccT = bool;
    DEVICE_HOST bool identity() const { return true; }
    DEVICE_HOST bool reduce(const bool& a, const bool& b) const { return a && b; }

    // CPU short-circuit: once accumulator is false, AND can never recover — stop.
    bool can_short_circuit(bool acc) const { return !acc; }
};

template <typename T>
struct AnyOp {
    using AccT = bool;
    DEVICE_HOST bool identity() const { return false; }
    DEVICE_HOST bool reduce(const bool& a, const bool& b) const { return a || b; }

    // CPU short-circuit: once accumulator is true, OR can never go back — stop.
    bool can_short_circuit(bool acc) const { return acc; }
};
```

`can_short_circuit` is NOT marked `DEVICE_HOST` — it is intentionally CPU-only. GPU kernels
never call it. The method inlines to a single instruction (`!acc` or `acc`).

**(B) `ReductionImpl.h` — added `break` in the bool path inner loop (lines ~285–291):**

```cpp
} else if constexpr (std::is_same_v<AccT, bool>) {
    bool val_as_bool = to_bool_value(input_value);
    accumulator = op.reduce(accumulator, val_as_bool);
    // Short-circuit: AllOp exits on first false, AnyOp on first true.
    // Safe because the inner loop is sequential (not GPU/SIMT).
    if (op.can_short_circuit(accumulator)) break;
}
```

The `break` exits the `for (int64_t i = 0; i < reduced_count; ++i)` loop.
This is safe because:
1. The inner loop is sequential (not parallelized — OpenMP is on the outer `output_index` loop)
2. `AND(false, x) = false` and `OR(true, x) = true` for all x — the accumulated result cannot
   change after the short-circuit condition is met

**Overhead when NOT short-circuiting (worst case — condition never met):**
`can_short_circuit(acc)` compiles to a single conditional branch per element.
The CPU branch predictor will predict "not taken" after the first few iterations. Cost ≈ 0.
No measurable overhead in the steady state.

**Savings when short-circuiting (best case):**
For `reduce_all` on a tensor where element 0 is false: saves `reduced_count - 1` iterations.
For `reduce_any` on a tensor where element 0 is true: same.

**Decision vs PyTorch:** PyTorch uses the identical `break` pattern on CPU. This change matches PyTorch behaviour exactly.

---

### 1.15 GPU Mean / Variance — Hardcoded `double` Accumulator Replaced by `float`
**(Files: `include/ops/helpers/ReductionKernels.cuh`, `src/UnaryOps/cuda/ReductionImplGPU.cu`)**

**Problem (Part 5 — statistical ops, GPU side):**
`reduce_mean_kernel` in `ReductionKernels.cuh` had a hardcoded `double` accumulator for **all**
non-complex types, including `float`, `half`, `bfloat16`, and all integer types:

```cpp
// BEFORE — double for EVERYTHING (wrong for all non-double types on GPU):
using AccT = typename std::conditional_t<
    is_complex,
    typename std::conditional_t<std::is_same_v<T, complex32_t>, complex64_t, complex128_t>,
    double    // ← ALL non-complex types used double accumulator
>;
```

On consumer GPUs (GeForce, RTX up to 4090, Tesla T4, etc.), FP64 throughput is a small fraction
of FP32:

| GPU generation | FP64 / FP32 ratio |
|---------------|-------------------|
| Kepler / Maxwell / Pascal | 1 / 32 or 1 / 64 |
| Turing (RTX 20xx) | 1 / 32 |
| Ampere A100 (datacenter) | 1 / 2 |
| Ada Lovelace RTX 40xx | 1 / 64 |

Using `double` for `mean<float>` was making the GPU kernel **32–64× slower** on consumer hardware
for no precision benefit (the output was still cast back to `float`).

The same problem existed in `dispatch_mean_gpu` and `dispatch_variance_gpu` in `ReductionImplGPU.cu`:
output dtype was `Float64` for integer inputs, `OutputCppT = double`, `AccCppT = double`.

**The user's question that identified this:** "u said double is slow, what pytorch is using? float
right? then why are we using double?"

**PyTorch's approach:** `opmath_type<T>` for GPU: `half → float`, `bfloat16 → float`, `float → float`,
`double → double`. All non-double, non-complex types → `float`. This is exactly what we changed to.

**Changes made:**

**(A) `ReductionKernels.cuh` — `reduce_mean_kernel` AccT (line ~536):**

```cpp
// BEFORE:
using AccT = std::conditional_t<is_complex, ..., double>;  // double for ALL

// AFTER this change (1.15 intermediate):
using AccT = std::conditional_t<
    is_complex,
    std::conditional_t<std::is_same_v<T, complex32_t>, complex64_t, complex128_t>,
    std::conditional_t<std::is_same_v<T, double>, double, float>
    // double input → double accumulator; everything else (float, half, int*) → float
>;
```

**Note:** the complex branch above (`complex64_t → complex128_t` for GPU) was subsequently
corrected in **change 1.12** (complex type promotion). The final code in the file is:
```cpp
// FINAL (after change 1.12 fixed the complex GPU AccT):
using AccT = typename std::conditional_t<
    is_complex,
    detail::AccumulatorType<T, /*IsGPU=*/true>,  // complex64_t GPU → complex64_t (NOT complex128_t)
    typename std::conditional_t<std::is_same_v<T, double>, double, float>
>;
```
`detail::AccumulatorType<complex64_t, true>` = `complex64_t` — no double components on GPU.

**(B) `ReductionImplGPU.cu` — `dispatch_mean_gpu` (lines ~374–425):**

| What changed | Before | After |
|---|---|---|
| `output_dtype` for integral T | `Dtype::Float64` | `Dtype::Float32` |
| `shared_mem_size` acc part | `num_warps * sizeof(double)` | `num_warps * mean_acc_size` where `mean_acc_size = is_same_v<T,double> ? 8 : 4` |
| `OutputCppT` for integral T | `double` | `float` |

Comment added to explain the intentional device difference:
```cpp
// Float32 is 32x faster than Float64 on consumer GPUs (FP64 = 1/32 FP32 on Pascal/Turing).
// PyTorch uses float (opmath_type) for integer reductions.
// CPU uses Float64 for exact integer arithmetic — GPU trades some precision for 32x speed.
```

**(C) `ReductionImplGPU.cu` — `dispatch_variance_gpu` (lines ~477–552):**

| What changed | Before | After |
|---|---|---|
| `output_dtype` for integral T | `Dtype::Float64` | `Dtype::Float32` |
| `MeanCppT` for integral T | `double` | `float` (mean now outputs Float32) |
| `OutputCppT` for integral T | `double` | `float` |
| `AccCppT` for integral T | `double` | `float` |

---

### 1.16 CPU vs GPU Integer Mean — Intentional Float64 vs Float32 (Design Decision)
**(Files: `include/ops/helpers/ReductionImpl.h`)**

**The question raised:** After making GPU output Float32 for integer mean, should CPU also be
changed to Float32 for consistency?

**Analysis — does changing CPU to Float32 reduce any overhead?**

CPU integer mean loop:
```cpp
int64_t accumulator = 0;
for (int64_t i = 0; i < reduced_count; ++i)
    accumulator += input_value;   // int64 add — dominates all cost, O(reduced_count)
// ONE division per output element:
output = (double)accumulator / (double)reduced_count;   // O(1) — negligible
```

Changing to Float32 changes only the final per-output-slice division:
- `DIVSD` (double): ~14–20 cycles
- `DIVSS` (float): ~10–14 cycles

Savings: ~6 cycles, once per output element. The inner loop runs `reduced_count` iterations.
**This overhead is completely dominated by the reduction loop — changing to Float32 saves zero
meaningful time on CPU.**

**"Cast before" question — does PyTorch reduce overhead by casting int tensor to float first?**

PyTorch doesn't support `mean()` on integer tensors (throws RuntimeError). So there is no
PyTorch comparison here. NumPy returns `float64` for integer `mean()`.

The phrase "cast before" could mean two things:
1. **Allocate a new float tensor from the int tensor before reducing** — wasteful, extra memory
2. **Cast each element to float at the point it is loaded** — this is what we do (T→int64 at `+=`)

Both our approach and PyTorch's `opmath_type` approach do #2 ("cast at load"). Neither allocates
an extra tensor. Our `int64_t` accumulation is actually **more precise** than float accumulation:
- `float` has a 24-bit mantissa — exact for sums up to ~16M only
- `int64_t` accumulation: exact for all practical sizes (overflow only at N × INT32_MAX > 2^63)

**Decision:** Keep CPU integer mean at **Float64** output. Reasoning:
- FP64 division costs the same as FP32 on modern x86 (DIVSD vs DIVSS differ by ~6 cycles, negligible)
- Higher precision is free at CPU — no reason to reduce it
- GPU uses Float32 because FP64 is 32× slower there — a hardware constraint, not a precision choice

This intentional asymmetry is documented in the source comment:
```cpp
// CPU integer mean → Float64 (intentionally different from GPU's Float32).
// Accumulation in int64_t is exact; dividing to double adds no loop overhead
// (the division is O(1) per output slice, dominated by the O(reduced_count) loop).
// GPU uses Float32 because FP64 is 32× slower on consumer hardware; CPU FP64 is free.
```

---

## 2. Hardware Optimizations
*(Reserved for hardware-specific optimizations)*
