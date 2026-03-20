# Accumulator System — Complete Documentation

> **Covers:** what an accumulator is, original state of master_gau, comparison with PyTorch and TensorFlow,
> every change made, the test suite design, bugs found and fixed, and pending work.

---

## Part 1 — What is an accumulator and why do we need one

When you run `reduce_sum` on a tensor of float16 values, the running total cannot be kept in float16.
float16 has a maximum representable value of 65504 and a machine epsilon of about 0.001.
Summing even a few thousand values near 1.0 will overflow or lose precision badly.

The solution is to use a **wider type** for the running total — the "accumulator."
You read each element as float16, cast it up to float32, add it to a float32 accumulator,
and at the end cast the float32 result back to float16 for the output.
This is called **type promotion** for accumulation.

The same problem applies to integers.
If you sum 1000 values of int16, each holding 30000, the true sum is 30,000,000.
int16 can only hold up to 32767. Without promotion to int64, the result wraps around and is wrong.

Every serious tensor library implements this. The difference between libraries is *where* the
accumulator type is decided, *how* it is communicated to the kernel, and *whether* CPU and GPU
use the same rule or different ones.

---

## Part 2 — Original state of master_gau (before changes)

### The CPU accumulator selector (ReductionOps.h)

The original code had a single-parameter template struct in `include/ops/helpers/ReductionOps.h`:

```cpp
// OLD — single parameter, no device awareness
template<typename T>
struct AccumulatorTypeSelector {
    using type = T;   // default: no promotion
};

template<> struct AccumulatorTypeSelector<int16_t>      { using type = int64_t; };
template<> struct AccumulatorTypeSelector<int32_t>      { using type = int64_t; };
template<> struct AccumulatorTypeSelector<int64_t>      { using type = int64_t; };
template<> struct AccumulatorTypeSelector<uint8_t>      { using type = int64_t; };   // BUG: should be uint64_t
template<> struct AccumulatorTypeSelector<uint16_t>     { using type = int64_t; };   // BUG
template<> struct AccumulatorTypeSelector<uint32_t>     { using type = int64_t; };   // BUG
template<> struct AccumulatorTypeSelector<uint64_t>     { using type = int64_t; };   // BUG: int64 can't hold large uint64
template<> struct AccumulatorTypeSelector<float16_t>    { using type = float;   };
template<> struct AccumulatorTypeSelector<bfloat16_t>   { using type = float;   };
// float had NO specialization — accumulated as float (no promotion)
template<> struct AccumulatorTypeSelector<bool>         { using type = int64_t; };
template<> struct AccumulatorTypeSelector<float4_e2m1_t>     { using type = float; };
template<> struct AccumulatorTypeSelector<float4_e2m1_2x_t>  { using type = float; };
#ifdef __CUDACC__
template<> struct AccumulatorTypeSelector<__half>        { using type = float; };
template<> struct AccumulatorTypeSelector<__nv_bfloat16> { using type = float; };
#endif

template<typename T>
using AccumulatorType = typename AccumulatorTypeSelector<T>::type;
```

**Missing types in the original:** `int8_t` had no specialization — it fell through to `type = T`,
meaning int8_t accumulated as int8_t (immediate overflow for any real sum).

### The GPU accumulator (ReductionKernels.cuh) — completely separate

The GPU kernel had its own inline logic, not using the CPU struct at all:

```cpp
// OLD — GPU-local logic, independent of CPU struct
constexpr bool is_half = std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>;
constexpr bool is_integer_sum = std::is_integral_v<T> && std::is_same_v<OpType<T>, SumOp<T>>;
constexpr bool is_integer_product = std::is_integral_v<T> && std::is_same_v<OpType<T>, ProductOp<T>>;

using AccumulatorType = typename std::conditional_t<
    is_integer_sum || is_integer_product,
    int64_t,
    typename std::conditional_t<is_half, float, T>
>;
```

Problems with this GPU logic:
- Only promotes integers for Sum and Product. Max and Min on int32_t used int32_t as accumulator.
  Max and Min don't need overflow protection (they don't add), but it is inconsistent.
- float accumulated as float on GPU (no double promotion, which is correct for GPU perf).
- Completely disconnected from the CPU struct — any change to CPU rules had to be duplicated manually.

### The Kahan summation (ReductionImpl.h) — also active

On CPU, there was Kahan compensated summation active for float-accumulator SumOp paths.
Kahan adds 4 extra FP ops + 1 branch + a serial dependency chain per element.
It was gated by:

```cpp
constexpr bool use_kahan = std::is_same_v<OpType<T>, SumOp<T>> &&
                           !std::is_same_v<AccT, ValueIndex<T>> &&
                           (std::is_floating_point_v<AccumulatorT> ||
                            std::is_same_v<AccumulatorT, double>);
```

### The dispatch gap (TensorDispatch.h)

`dispatch_by_dtype` routes runtime `Dtype` enum values to compile-time type parameters.
The original switch statement was missing `case Dtype::Int8`.
This meant every reduction operation — not just sum — would throw `"Unsupported Dtype"`
for any Int8 tensor, on both CPU and GPU.

---

## Part 3 — Problems identified

| ID | File | Problem |
|----|------|---------|
| P1 | ReductionOps.h | `int8_t` missing — no specialization, accumulated as int8_t (overflows immediately) |
| P2 | ReductionOps.h | `uint8/16/32/64` mapped to `int64_t` — wrong for large unsigned values (uint64 > INT64_MAX wraps negative) |
| P3 | ReductionOps.h | `float` had no specialization — accumulated as float, losing precision on large tensors |
| P4 | ReductionOps.h | Struct is CPU-only with no device parameter — GPU had to duplicate logic separately |
| P5 | ReductionKernels.cuh | GPU accumulator logic was inline, not using the CPU struct — diverged silently over time |
| P6 | ReductionImpl.h | Kahan summation active: 4 extra FP ops + serial dependency per element, blocks pipelining |
| P7 | TensorDispatch.h | `case Dtype::Int8` missing from `dispatch_by_dtype` switch — ALL ops fail on Int8 at runtime |
| P8 | ReductionImplGPU.cu | No explicit template instantiations for `int8_t` GPU kernels — linker error on GPU path |

---

## Part 4 — Changes made (in order)

### Change 1: float → double on CPU (ReductionOps.h)

**Added:**
```cpp
template<> struct AccumulatorTypeSelector<float> { using type = double; };
```

**Why:** Float accumulation on N=1M values gives relative error ~O(N × 1.2e-7) ≈ 0.1.
Double accumulation gives ~O(N × 2.2e-16) ≈ 0. This also eliminates the need for Kahan.
GPU keeps float (double is 32x slower on NVIDIA consumer GPUs).

**Impact:** `reduce_sum`, `reduce_mean` on float tensors now use double internally on CPU.
Result is still written back as float32 (since input dtype = float), so output precision
is bounded by float32's ~1.2e-7 relative error, not the accumulation error.

### Change 2: Kahan summation removed (ReductionImpl.h)

**Removed:** The `use_kahan` constexpr and the entire `if constexpr (use_kahan)` branch.

**Why:**
- With float→double promotion, Kahan is redundant. Double accumulation is more precise anyway.
- Kahan is serially dependent: each iteration needs the previous `kahan_c`, preventing CPU pipelining.
- Kahan actually FAILS on catastrophic cancellation patterns like `[1e8, 1.0, -1e8]×1000`
  because at float precision `(1e8 - 1.0) - 1e8 = 0` (1.0 is below the ULP of 1e8 in float32).
  Double accumulation handles this correctly because double ULP at 1e8 ≈ 1.5e-8 << 1.0.

### Change 3: Unified two-parameter accumulator struct (ReductionOps.h)

**Before:** Single-parameter struct, CPU-only:
```cpp
template<typename T>
struct AccumulatorTypeSelector { using type = T; };
```

**After:** Two-parameter struct with `IsGPU` flag:
```cpp
template<typename T, bool IsGPU = false>
struct AccumulatorTypeSelector { using type = T; };
```

Partial specializations cover all types where CPU and GPU agree:
```cpp
// Same answer for both CPU and GPU — use template<bool IsGPU>
template<bool IsGPU> struct AccumulatorTypeSelector<int8_t,  IsGPU> { using type = int64_t; };
template<bool IsGPU> struct AccumulatorTypeSelector<int16_t, IsGPU> { using type = int64_t; };
template<bool IsGPU> struct AccumulatorTypeSelector<int32_t, IsGPU> { using type = int64_t; };
template<bool IsGPU> struct AccumulatorTypeSelector<int64_t, IsGPU> { using type = int64_t; };
template<bool IsGPU> struct AccumulatorTypeSelector<uint8_t,  IsGPU> { using type = uint64_t; };
template<bool IsGPU> struct AccumulatorTypeSelector<uint16_t, IsGPU> { using type = uint64_t; };
template<bool IsGPU> struct AccumulatorTypeSelector<uint32_t, IsGPU> { using type = uint64_t; };
template<bool IsGPU> struct AccumulatorTypeSelector<uint64_t, IsGPU> { using type = uint64_t; };
template<bool IsGPU> struct AccumulatorTypeSelector<bool,     IsGPU> { using type = int64_t; };
template<bool IsGPU> struct AccumulatorTypeSelector<float16_t,  IsGPU> { using type = float; };
template<bool IsGPU> struct AccumulatorTypeSelector<bfloat16_t, IsGPU> { using type = float; };
```

Full specializations where CPU and GPU differ:
```cpp
// float is the only type that differs between devices
template<> struct AccumulatorTypeSelector<float, false> { using type = double; }; // CPU → double
template<> struct AccumulatorTypeSelector<float, true>  { using type = float;  }; // GPU → float
```

Convenience alias:
```cpp
template<typename T, bool IsGPU = false>
using AccumulatorType = typename AccumulatorTypeSelector<T, IsGPU>::type;
```

Default is CPU (`IsGPU=false`), so existing call sites that omit the second parameter still work.

### Change 4: unsigned integers → uint64_t instead of int64_t

**Before:** `uint8/16/32/64 → int64_t`
**After:** `uint8/16/32/64 → uint64_t`

**Why:** int64_t max is 9.22 × 10^18. uint64_t max is 1.84 × 10^19.
If you have a uint64_t value of say 10^19 and accumulate with int64_t, it wraps negative.
Using uint64_t as accumulator for all unsigned types is semantically correct and has
identical hardware cost (one-clock zero-extension, same as int64_t for unsigned inputs).

### Change 5: GPU kernel unified (ReductionKernels.cuh)

**Before:** Inline `std::conditional_t<>` chain inside each GPU kernel.
**After:** One line using the shared struct:
```cpp
// OLD (commented out):
// using AccumulatorType = typename std::conditional_t<is_integer_sum || is_integer_product,
//     int64_t, typename std::conditional_t<is_half, float, T>>;

// NEW:
using AccumulatorType = detail::AccumulatorType<T, /*IsGPU=*/true>;
```

`/*IsGPU=*/true` is a C block comment used as an inline label — the compiler sees just `true`.
This is the same style PyTorch uses: `at::acc_type<scalar_t, /*is_cuda=*/true>`.

### Change 6: Int8 dispatch gap fixed (TensorDispatch.h)

**Added** `case Dtype::Int8:` to both `dispatch_by_dtype` and `dispatch_by_integer_dtype`:
```cpp
case Dtype::Int8:  return f(typename DtypeToType<Dtype::Int8>::type{});  // ← was missing
case Dtype::Int16: return f(typename DtypeToType<Dtype::Int16>::type{});
```

This was the root cause of ALL reductions failing silently on Int8 tensors.

### Change 7: Int8 GPU kernel instantiations (ReductionImplGPU.cu)

GPU kernel functions are declared in headers but defined in `.cu` files with explicit
template instantiations. `int8_t` was missing from those instantiations.

**Added:**
```cpp
// int8_t (signed char) — Basic operations only
template Tensor dispatch_reduction_gpu<int8_t, SumOp>(...);
template Tensor dispatch_reduction_gpu<int8_t, ProductOp>(...);
template Tensor dispatch_reduction_gpu<int8_t, MinOp>(...);
template Tensor dispatch_reduction_gpu<int8_t, MaxOp>(...);
template Tensor dispatch_index_reduction_gpu<int8_t, ArgMinOp>(...);
template Tensor dispatch_index_reduction_gpu<int8_t, ArgMaxOp>(...);
template Tensor dispatch_mean_gpu<int8_t, SumOp>(...);
template Tensor dispatch_variance_gpu<int8_t, VarianceOp>(...);
template Tensor dispatch_variance_gpu<int8_t, NanVarianceOp>(...);
```

### Change 8: G renamed to IsGPU in partial specializations (ReductionOps.h)

The partial specializations originally used `G` as the boolean parameter name:
```cpp
template<bool G> struct AccumulatorTypeSelector<int8_t, G> { ... }
```
Renamed to `IsGPU` for clarity:
```cpp
template<bool IsGPU> struct AccumulatorTypeSelector<int8_t, IsGPU> { ... }
```
Zero functional change — purely readability.

---

## Part 5 — Comparison with PyTorch and TensorFlow

### PyTorch

PyTorch has one central file: `aten/src/ATen/AccumulateType.h`.
The key design: **parameterized by device type** using a two-bool template or macros.

```cpp
// PyTorch usage in any kernel:
using accscalar_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
```

**PyTorch CPU accumulator rules:**

| Input | CPU Accumulator |
|-------|----------------|
| Half, BFloat16 | float |
| float | double |
| double | double |
| int8_t, uint8_t, int16_t, int32_t, int64_t | int64_t |
| bool | **bool** (not int64_t) |
| complex\<Half\> | complex\<double\> |
| complex\<float\> | complex\<double\> |
| complex\<double\> | complex\<double\> |

**PyTorch CUDA accumulator rules:**

| Input | GPU Accumulator |
|-------|----------------|
| Half, BFloat16 | float |
| float | **float** (different from CPU) |
| double | double |
| int8_t, uint8_t, int16_t, int32_t, int64_t | int64_t |
| bool | **bool** |
| complex\<Half\> | complex\<float\> |
| complex\<float\> | complex\<float\> |
| complex\<double\> | complex\<double\> |

**Key PyTorch differences from master_gau:**
- `bool → bool`, not `bool → int64_t`. PyTorch pre-casts bool tensors to int64 *before* reduction at a higher level.
- Unsigned types other than uint8_t are not supported at all in PyTorch (no uint16/32/64 tensor dtype).
- Complex types get promoted (complex\<float\> → complex\<double\> on CPU).
- Kahan summation is implemented but as a higher-level algorithm, not inside the accumulator struct.

### TensorFlow

TensorFlow has **no centralized accumulator system**.
Each kernel file defines its own local struct independently.

**CPU (bias_op.cc):**
```cpp
template <class T> struct AccumulatorType { typedef T type; }; // default: no promotion
template <> struct AccumulatorType<Eigen::half> { typedef float type; }; // only half
```
That's the entire system. float stays float, integers stay integers.

**GPU (bias_op_gpu.cu.cc):**
```cpp
template <class T> struct AccumulatorType { typedef T type; };
template <> struct AccumulatorType<Eigen::half>        { typedef float type; };
template <> struct AccumulatorType<Eigen::bfloat16>    { typedef float type; }; // one more than CPU
```

**Implications:**
- TF CPU: float32 sums accumulate in float32 — no precision protection for large tensors.
- Each kernel file duplicates this struct independently.
- If a developer writes a new kernel and forgets the struct, it silently accumulates in the input type.
- No overflow protection for integers anywhere.

### master_gau (current, after changes)

| Input | CPU Accumulator | GPU Accumulator |
|-------|----------------|----------------|
| int8_t | int64_t | int64_t |
| int16_t | int64_t | int64_t |
| int32_t | int64_t | int64_t |
| int64_t | int64_t | int64_t |
| uint8_t | uint64_t | uint64_t |
| uint16_t | uint64_t | uint64_t |
| uint32_t | uint64_t | uint64_t |
| uint64_t | uint64_t | uint64_t |
| bool | int64_t | int64_t |
| float16_t | float | float |
| bfloat16_t | float | float |
| **float** | **double** | **float** |
| double | double | double |
| FP4 types | float | float |
| \_\_half (CUDA) | float | float |
| \_\_nv_bfloat16 (CUDA) | float | float |

**How master_gau compares:**

| Feature | PyTorch | TensorFlow | master_gau |
|---------|---------|------------|------------|
| Centralized accumulator | Yes, 1 file | No, per-kernel | Yes, 1 struct (after change) |
| Device-parameterized | Yes (`is_cuda`) | No | Yes (`IsGPU`) |
| float CPU accumulator | double | float | **double** |
| float GPU accumulator | float | float | float |
| half/bf16 accumulator | float | float | float |
| int overflow protection | int64_t | none | int64_t |
| unsigned overflow | not supported | not supported | **uint64_t** |
| bool accumulator | bool | not handled | int64_t |
| complex promotion | complex\<double\> on CPU | none | none (stays same) |
| Kahan summation | Yes, higher-level | No | Removed (double acc replaces it) |

---

## Part 6 — Test suite design

File: `Tests/ReductionsTests/accumulator_type_test.cpp`

### Why these specific tests were chosen

The test is split into 6 sections, each targeting a different failure mode.

**Section 1 — Compile-time type checks (static_assert)**

These run at compile time. If any specialization mapping is wrong, the binary won't compile at all.
They verify every `AccumulatorType<T, IsGPU>` mapping for both CPU and GPU paths.
28 static_asserts covering:
- All signed integers → int64_t on both devices
- All unsigned integers → uint64_t on both devices
- bool → int64_t on both devices
- float16/bfloat16 → float on both devices
- float CPU → double (the critical asymmetric case)
- float GPU → float (must differ from CPU)
- double → double on both devices
- Default `IsGPU=false` gives CPU path

These compile away to zero runtime code. The entire section runs at zero runtime cost.

**Section 2 — Overflow prevention (runtime)**

These prove that the wider accumulator actually prevents wrong results.
The key idea for each test: choose N and value such that N × value exceeds the input type's max,
but fits exactly in the accumulator type. Then verify the result matches the expected exact value.

- `int8_t`: 500 × 100 = 50000. int8 max = 127. Would wrap without int64_t accumulator.
- `uint8_t`: 300 × 200 = 60000. uint8 max = 255. Would wrap without uint64_t accumulator.
- `uint16_t`: 1000 × 65000 = 65,000,000. uint16 max = 65535.
- `int16_t`: 5000 × 30000 = 150,000,000. int16 max = 32767.
- `int32_t`: 3 × 2,000,000,000 = 6,000,000,000. int32 max = 2,147,483,647.
- `uint32_t`: 3 × 3,000,000,000 = 9,000,000,000. uint32 max = 4,294,967,295.

These also verify the `int8_t` dispatch fix — before the fix, these tests crashed before the assert.

**Section 3 — Float precision (runtime)**

Two tests that specifically stress the float→double accumulator change.

*Large N random data:*
1,000,000 random float values in [-1, 1]. With float accumulation, relative error ≈ O(N × eps_f) ≈ 0.1.
With double accumulation, relative error ≈ O(N × eps_d) ≈ 2.2e-10 vs the double reference.
Since output is stored back as float32, the final observable relative error is bounded by
float32 output precision (~1.2e-7). Threshold set to 1e-6 (10× float eps).

*Catastrophic cancellation:*
Pattern `[1e8, 1.0, -1e8] × 1000`. True sum = 1000.0.
- Naive float: `(1e8 + 1.0)` in float32 = 1e8 (1.0 below ULP of 1e8). Net = 0. Error = 1000.
- Kahan float: same failure — `(1e8 - 1.0) - 1e8 = 0` in float32, compensation lost.
- Double accumulator: double ULP at 1e8 ≈ 1.5e-8 << 1.0. Sum = 1000.0 exactly.
This test shows why Kahan was insufficient and double accumulation was the right fix.

**Section 4 — Bool accumulation (runtime)**

Verifies that `bool → int64_t` accumulator:
- `reduce_sum` of N all-true bools = N (count of trues)
- `reduce_sum` of N all-false bools = 0
- `reduce_all` of 100 trues = 1
- `reduce_any` of 100 falses = 0

**Section 5 — Double stays double (runtime)**

Verifies `double → double` (no over-promotion): 1000 × 0.1 = 100.0 with rel_err < 1e-12.

**Section 6 — Half precision (runtime)**

Verifies `float16 → float` accumulator: 1000 × 1.0f16 = 1000. Uses `fill()` instead of `set_data`
because `set_data<float16_t>` is not instantiated in the pre-built library.

### Test results (after all changes)

```
22/22 passed

Section 1: 8/8   — All compile-time type mappings correct
Section 2: 6/6   — All overflow prevention tests pass (including int8_t after dispatch fix)
Section 3: 2/2   — Float precision confirmed, catastrophic cancellation fixed
Section 4: 4/4   — Bool accumulation correct
Section 5: 1/1   — Double precision maintained
Section 6: 1/1   — float16 accumulates in float correctly
```

---

## Part 7 — Bugs found during testing

**Bug B1: int8_t missing from dispatch_by_dtype**
- Location: `include/core/TensorDispatch.h`, `dispatch_by_dtype` switch
- Symptom: ALL reduction ops throw `"Unsupported Dtype"` on Int8 tensors at runtime
- Fix: Added `case Dtype::Int8: return f(typename DtypeToType<Dtype::Int8>::type{});`
- Also fixed in `dispatch_by_integer_dtype`

**Bug B2: int8_t missing from GPU explicit instantiations**
- Location: `src/UnaryOps/cuda/ReductionImplGPU.cu`
- Symptom: Linker error on all GPU reduction functions for int8_t
- Fix: Added 9 explicit template instantiations for int8_t matching the pattern of int16_t

**Bug B3: unsigned types accumulated as int64_t (semantically wrong)**
- Location: `include/ops/helpers/ReductionOps.h`
- Symptom: uint64_t values > INT64_MAX would be cast to negative int64_t before accumulation
- Fix: Changed `uint8/16/32/64 → uint64_t` instead of `int64_t`

**Bug B4: float accumulated as float on CPU (precision loss)**
- Location: `include/ops/helpers/ReductionOps.h`
- Symptom: Large float32 sums had relative error ~O(N × 1.2e-7), visible on N > 100k
- Fix: Added `template<> struct AccumulatorTypeSelector<float, false> { using type = double; };`

---

## Part 8 — Pending work

### Pending: pairwise (tree) reduction to replace sequential loop

Currently the CPU kernel does a sequential scan:
```cpp
for (int64_t i = 0; i < reduced_count; ++i) {
    accumulator = op.reduce(accumulator, cast(input[i]));
}
```

With double accumulation, sequential scan still has O(N × double_eps) error.
Pairwise reduction splits the array recursively into halves, reducing each pair,
which gives O(log2(N) × double_eps) error. For N=1M: sequential = ~2.2e-10 vs pairwise = ~1.4e-14.

Pairwise also enables vectorization: groups of 4–8 accumulators can be updated in parallel
using SIMD (the compiler auto-vectorizes register-independent loops).

After pairwise reduction is implemented, the float CPU accumulator can be reconsidered:
with pairwise + float, error is O(log2(N) × float_eps) ≈ 20 × 1.2e-7 ≈ 2.4e-6 for N=1M.
That may be acceptable, avoiding the float→double conversion cost per element.

### Pending: stride-based traversal to replace ravel/unravel

Current inner loop calls `unravel_index` and `ravel_index` on every element.
These involve integer division and modulo operations (~10–15 integer ops per element).
Replacing with stride-based pointer arithmetic reduces this to 1–2 additions per element.

### Pending: complex type accumulator promotion

Currently complex32_t/64_t/128_t have no accumulator promotion (accumulate as themselves).
PyTorch promotes complex\<float\> → complex\<double\> on CPU. This can be added when needed.

### Pending: IntelliSense / CUDA IDE parser cache

The CUDA IntelliSense parser in VSCode shows false errors on the static_asserts in the test file
because it uses a cached index that still sees the old 1-parameter `AccumulatorType` definition.
This is an IDE cache issue only — GCC compiles and runs all 22 tests correctly.
Fix: reload the window or clear the IntelliSense cache (Ctrl+Shift+P → "C/C++: Reset IntelliSense Database").

---

## Part 9 — File summary

| File | Change |
|------|--------|
| `include/ops/helpers/ReductionOps.h` | Replaced 1-param struct with 2-param `IsGPU` struct; added float→double on CPU; unsigned → uint64_t; int8_t added; renamed `G` → `IsGPU` |
| `include/ops/helpers/ReductionImpl.h` | Removed Kahan summation entirely (use_kahan constexpr + entire if-constexpr block) |
| `include/ops/helpers/ReductionKernels.cuh` | Replaced inline conditional_t chain with `detail::AccumulatorType<T, /*IsGPU=*/true>` |
| `include/core/TensorDispatch.h` | Added `case Dtype::Int8:` to both dispatch switch statements |
| `src/UnaryOps/cuda/ReductionImplGPU.cu` | Added 9 explicit template instantiations for int8_t |
| `Tests/ReductionsTests/accumulator_type_test.cpp` | New file: 22 tests across 6 sections |
| `Tests/ReductionsTests/kahan_precision_test.cpp` | Existing file: precision+timing baseline vs after changes |
| `Tests/ReductionsTests/baseline_before_changes.txt` | Saved precision baseline (Kahan active, float accumulator) |
