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
| P9 | ReductionOps.h + ReductionImpl.h + ReductionKernels.cuh + ReductionImplGPU.cu | `MinOp`/`MaxOp`/`NanMinOp`/`NanMaxOp` unnecessarily widened accumulator: int32→int64, float(CPU)→double. Compare ops cannot overflow; widening added 1 extra instruction/element on CPU and 2–4× slower 64-bit comparison on pre-Volta GPUs. Also caused wrong output dtype: `reduce_min<int32>` was returning Int64 tensor instead of Int32. |
| P10 | ReductionKernels.cuh + ReductionImplGPU.cu | `reduce_mean_kernel` hardcoded `double` accumulator for ALL non-complex types including `float`, `half`, and integer inputs. On consumer GPUs FP64 = 1/32 FP32 throughput → mean on float tensors was 32× slower than necessary. `dispatch_mean_gpu` and `dispatch_variance_gpu` also outputting Float64 for integers when Float32 is correct for GPU. |
| P11 | ReductionImpl.h (CPU index path) | Index reductions (argmin/argmax/nanargmin/nanargmax) on CPU used `ValueIndex<T>` struct per element — allocating `{input_value, i}` on stack per iteration and returning full struct from `op.reduce()`. PyTorch uses two independent scalar variables with conditional index update only when value improves. Our approach had ~2–3 extra instructions per element. Also: NaN check in `better_than` was doing 2 `is_nan_check` calls per element in the hot path; IEEE 754 allows reducing this to 1. |
| P12 | ReductionImpl.h (CPU bool path) | `reduce_all` and `reduce_any` traversed the entire reduced dimension even after the result was determined. Once `AllOp` accumulator = false, no subsequent element can change the result; once `AnyOp` = true, same. Missing early exit added O(reduced_count) unnecessary work. |

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

### Change 9: Compare ops bypass AccumulatorTypeSelector — use input type directly

**(Files: `include/ops/helpers/ReductionOps.h`, `include/ops/helpers/ReductionImpl.h`,
`include/ops/helpers/ReductionKernels.cuh`, `src/UnaryOps/cuda/ReductionImplGPU.cu`)**

**Root cause of the problem:**
The unified accumulator system (Change 3) applied accumulator widening to ALL ops uniformly.
This was correct for `SumOp`/`ProductOp` where arithmetic can overflow.
But `MinOp`, `MaxOp`, `NanMinOp`, `NanMaxOp` are **pure comparison** — the output is always
one of the input values. There is no way for a comparison to overflow or lose precision.
Widening the accumulator for compare ops was unnecessary overhead with no benefit.

**4 sub-changes made:**

**9a — `ReductionOps.h` (lines 412–413, 458–459, 588–589, 632–633):**
Changed `AccT` in all 4 compare op structs:
```cpp
// BEFORE (all 4 compare ops):
using AccT = AccumulatorType<T>;   // int32_t → int64_t, float(CPU) → double

// AFTER (MinOp, MaxOp, NanMinOp, NanMaxOp):
using AccT = T;   // compare op: result always within input range, no widening needed
```
`SumOp`, `ProductOp`, `NanSumOp`, `VarianceOp`, `NanVarianceOp` — AccT **unchanged**.

**9b — `ReductionImpl.h` (CPU path, 3 locations):**

*Output dtype* (line ~123): Changed condition from `is_integral_v<T>` to
`is_integral_v<T> && !is_same_v<AccT, T>`. Now only sum/product widen to Int64;
min/max output matches input dtype (Int32 stays Int32, Int8 stays Int8).

*`OutputCppT`* (line ~148): Same condition change — compare ops map to `T`, not `int64_t`.

*Accumulator initializer* (line ~238): Replaced the 3-branch
`if (should_use_double) ... else if (is_integral) ... else` chain with a single:
```cpp
accumulator = static_cast<AccumulatorT>(op.identity());
```
This is correct for all ops since `AccumulatorT` is now always the right type.
The old branches assumed AccumulatorT was always wider than T; that is no longer true for compare ops.

**9c — `ReductionKernels.cuh` (GPU kernel, line ~182):**
```cpp
// BEFORE:
using AccumulatorType = detail::AccumulatorType<T, /*IsGPU=*/true>;

// AFTER:
using AccumulatorType = std::conditional_t<
    std::is_same_v<OpType<T>, detail::MinOp<T>>    ||
    std::is_same_v<OpType<T>, detail::MaxOp<T>>    ||
    std::is_same_v<OpType<T>, detail::NanMinOp<T>> ||
    std::is_same_v<OpType<T>, detail::NanMaxOp<T>>,
    T,                                      // compare ops: use T directly
    detail::AccumulatorType<T, /*IsGPU=*/true>  // sum/etc: centralized type
>;
```

**9d — `ReductionImplGPU.cu` (GPU dispatcher, lines ~156–219):**

Added `constexpr bool is_compare_op` at line ~156 (same `is_same_v<>` check as 9c).
Used it to guard:
- `output_dtype`: `!is_compare_op` required to widen to Int64
- `OutputCppT`: same condition
- `shared_mem_size`: compare ops allocate `sizeof(T)` per warp slot instead of `sizeof(int64_t)`,
  halving shared memory usage for integer compare ops (e.g., int32 min: 32 bytes → 32 bytes → was 64)

---

### Change 10: Index ops — 2-variable approach on CPU (ReductionOps.h + ReductionImpl.h)

**Context — 5-part ops bifurcation:**
After fixing Part 1 (sum/product) and Part 2 (min/max) accumulator issues, the remaining
three groups were analyzed:
- **Part 3** (argmin/argmax/nanargmin/nanargmax): struct-based accumulation overhead
- **Part 4** (reduce_all/reduce_any): missing short-circuit
- **Part 5** (mean/variance/std/nanmean/nanvar): GPU double accumulator overhead

**Root cause of Part 3 problem:**
The CPU index path used `ValueIndex<T>` struct (`{T value; int64_t index}`) to bundle both
the running best value and its index. Every element required:
1. `ValueIndex<T> current_val_index = {input_value, i}` — struct construction per element
2. `op.reduce(accumulator, current_val_index)` — comparing two structs, returning a struct
3. Storing the result struct back into `accumulator`

GPU (`reduce_index_kernel`) already shuffled struct fields separately:
```cpp
other.value = shfl_down(accumulator.value, offset);
other.index = shfl_down(accumulator.index, offset);
```
So struct overhead on GPU was zero. GPU was left unchanged.

**Added to all 4 index ops in `ReductionOps.h`:**

```cpp
// Example: ArgMinOp<T>

T identity_val() const { return get_max_value<T>(); }  // ArgMax: get_lowest_value<T>()

// 1 NaN check per element in hot path (all-finite data).
// IEEE 754: (NaN < x) = false → if current_best is NaN, comparison returns false naturally.
// So current_best=NaN case needs no extra check — NaN sticks without 2nd is_nan_check.
bool better_than(const T& candidate, const T& current_best) const {
    if constexpr (is_any_float_v<T>) {
        if (is_nan_check(candidate)) return !is_nan_check(current_best);
    }
    return candidate < current_best;   // (> for ArgMax)
}
```

NaN semantics:

| Op | `better_than` for NaN | Effect |
|----|----------------------|--------|
| ArgMinOp | `is_nan_check(candidate)` → take it if current isn't NaN yet | First NaN wins (NaN propagates) |
| ArgMaxOp | Same | First NaN wins |
| NanArgMinOp | `if (is_nan_check(candidate)) return false;` | NaN inputs skipped entirely |
| NanArgMaxOp | Same | NaN inputs skipped entirely |

**Old vs new NaN check count:**

| Code version | Hot path (no NaN) | After first NaN found |
|---|---|---|
| Old reduce() with ValueIndex | 2 checks | 2 checks |
| New better_than() | **1 check** | **1 check** (IEEE property saves the 2nd) |

**Changed in `ReductionImpl.h` (lines ~197–238):**

```cpp
// BEFORE (struct per element):
ValueIndex<T> accumulator = op.identity();
for (int64_t i = 0; i < reduced_count; ++i) {
    T input_value = input_data[input_lin_idx];
    ValueIndex<T> current_val_index = {input_value, i};
    accumulator = op.reduce(accumulator, current_val_index);
}
output_data[output_index] = accumulator.index;

// AFTER (2 scalars, conditional index update only):
T best_val = op.identity_val();
int64_t best_idx = -1;
for (int64_t i = 0; i < reduced_count; ++i) {
    T input_value = input_data[input_lin_idx];
    if (op.better_than(input_value, best_val)) {
        best_val = input_value;
        best_idx = i;
    }
}
output_data[output_index] = best_idx;
```

`identity_val()` and `better_than()` have no `DEVICE_HOST` marker — they are CPU-only helpers
compiled only for host code, not device code. The existing `reduce()` and `identity()` methods
(with `DEVICE_HOST`) remain unchanged for GPU use.

---

### Change 11: Short-circuit for `reduce_all` / `reduce_any` (ReductionOps.h + ReductionImpl.h)

**Root cause of Part 4 problem:**
The inner reduction loop iterated over all `reduced_count` elements unconditionally.
Once `AllOp` accumulates a `false` value, the final result is `false` regardless of remaining
elements (`AND(false, x) = false` for all x). Once `AnyOp` accumulates `true`, the result
is `true`. Continuing the loop is pure wasted work.

**Added to `AllOp` and `AnyOp` in `ReductionOps.h`:**

```cpp
struct AllOp {
    // ... existing methods unchanged ...
    // CPU short-circuit: once false, AND can never recover.
    bool can_short_circuit(bool acc) const { return !acc; }
};

struct AnyOp {
    // ... existing methods unchanged ...
    // CPU short-circuit: once true, OR can never go back.
    bool can_short_circuit(bool acc) const { return acc; }
};
```

`can_short_circuit` is **not** `DEVICE_HOST` — GPU cannot short-circuit (SIMT lockstep means
all threads in a warp execute the same instruction; a `break` in one thread does not stop others).

**Added to the bool path in `ReductionImpl.h` inner loop (line ~291):**

```cpp
} else if constexpr (std::is_same_v<AccT, bool>) {
    bool val_as_bool = to_bool_value(input_value);
    accumulator = op.reduce(accumulator, val_as_bool);
    if (op.can_short_circuit(accumulator)) break;  // ← new
}
```

The `break` exits the inner `for (int64_t i = 0; i < reduced_count; ++i)` loop.
Safe because the inner loop is sequential — OpenMP parallelizes the outer `output_index` loop,
not the inner `reduced_count` loop.

**Overhead in steady state (no early exit):**
`can_short_circuit()` inlines to a single instruction (`!acc` or `acc`).
The conditional branch is predicted not-taken by the CPU branch predictor after the first few
iterations. Measured overhead ≈ 0.

**PyTorch and TensorFlow comparison:**
- PyTorch CPU: uses explicit `break` in the bool reduction kernel. Identical behaviour.
- PyTorch GPU: no short-circuit (SIMT limitation). Same as our GPU.
- TensorFlow: does not implement short-circuit for any reduction.

---

### Change 12: GPU mean kernel — `double` → `float` accumulator (ReductionKernels.cuh + ReductionImplGPU.cu)

**Root cause of Part 5 problem:**
`reduce_mean_kernel` had `AccT = double` hardcoded for every non-complex type — including
`float`, `half`, `bfloat16`, and all integer types. This was wrong for performance on consumer
GPUs where FP64 throughput is 1/32 to 1/64 of FP32.

Example cost: `mean<float>` on a 1M-element tensor on an RTX 3080:
- Before: kernel uses `double` → occupancy reduced (double uses 2 register slots), FP64 execution → ~32× slower
- After: kernel uses `float` → full FP32 occupancy and throughput

PyTorch's rule: `opmath_type<T>` on GPU = `float` for all of {half, bfloat16, float, integers},
`double` for double. We now match this exactly.

**Changed in `ReductionKernels.cuh` (line ~536):**
```cpp
// BEFORE:
using AccT = std::conditional_t<is_complex, ..., double>;

// AFTER:
using AccT = std::conditional_t<
    is_complex,
    std::conditional_t<std::is_same_v<T, complex32_t>, complex64_t, complex128_t>,
    std::conditional_t<std::is_same_v<T, double>, double, float>
>;
// Rationale: FP64 = 1/32 FP32 on consumer GPUs (Turing/Pascal/Maxwell).
// float is the correct choice for mean of float, half, and integer tensors on GPU.
// double → double preserved for double-precision input tensors.
```

**Changed in `ReductionImplGPU.cu` — `dispatch_mean_gpu` (lines ~374–425):**

| Location | Before | After |
|---|---|---|
| `output_dtype` (integral T) | `Dtype::Float64` | `Dtype::Float32` |
| `shared_mem_size` acc part | `num_warps * sizeof(double)` | `num_warps * mean_acc_size` (`sizeof(float)` if T≠double) |
| `OutputCppT` (integral T) | `double` | `float` |

The `shared_mem_size` fix was important: `sizeof(double) = 8` but `sizeof(float) = 4`. Without
this fix, the kernel would have been allocated twice the shared memory it actually needs, reducing
occupancy (number of concurrent blocks per SM).

**Changed in `ReductionImplGPU.cu` — `dispatch_variance_gpu` (lines ~477–552):**

| Location | Before | After |
|---|---|---|
| `output_dtype` (integral T) | `Dtype::Float64` | `Dtype::Float32` |
| `MeanCppT` (integral T) | `double` | `float` (cascade: mean now outputs Float32) |
| `OutputCppT` (integral T) | `double` | `float` |
| `AccCppT` (integral T) | `double` | `float` |

---

### Change 13: CPU integer mean stays Float64 — design decision documented (ReductionImpl.h)

**Question raised:** After GPU changed to Float32 for integer mean/variance, should CPU also
change to Float32 for consistency?

**Analysis of overhead:**
CPU integer mean path accumulates in `int64_t` (exact), then does one division per output element:
- `(double)sum / (double)count` → DIVSD: ~14–20 cycles
- `(float)sum / (float)count` → DIVSS: ~10–14 cycles

Difference: ~6 cycles, **once per output slice** (not per input element). The inner loop runs
`reduced_count` iterations per slice. For reduced_count = 10M, 6 cycles is 0.00000006% of total work.
**No meaningful overhead to save.**

**"Cast before" vs our approach:**
User asked: can we cast the int tensor to float before reducing to reduce overhead (like PyTorch)?
Answer: There is no "cast before the tensor" in PyTorch either. Both approaches cast at element
load time inside the reduction loop. "Cast before" would mean allocating an entirely new float
tensor — wasteful. Our `int64 += int_value` performs the widening in the `+=` instruction itself
(one MOVSX), identical cost to `float += (float)int_value` (one CVTSI2SS + one FADD vs one MOVSX
+ one IADD64). Neither is slower.

Furthermore, `int64_t` accumulation is **strictly more precise** than float accumulation:
- float sum: precision breaks down when sum > 2^24 (~16M). For N×INT32_MAX → sum can reach 2^62.
  float cannot represent this accurately (24-bit mantissa).
- int64_t sum: exact up to 2^63. No precision loss for any realistic integer tensor.

**Decision: CPU keeps Float64 output.** GPU uses Float32 due to a hardware constraint (FP64 = 1/32 FP32).
CPU uses Float64 because it's free. This is intentional and documented asymmetry.

```
CPU integer mean:  int64_t accumulation → (double)sum/count → Float64 output
GPU integer mean:  float   accumulation → (float)sum/count  → Float32 output
Reason: hardware FP64 penalty exists on GPU, not on CPU.
```

NumPy also returns `float64` for `np.mean` on integer arrays — matching our CPU behaviour.
PyTorch throws RuntimeError on integer `mean()` — no comparison possible.

---

## Part 4b — The 5-Part Ops Classification Framework

All 22 reduction ops were systematically divided into 5 groups to analyze accumulator behaviour,
compare against PyTorch/TensorFlow, and decide what changes were needed in each group:

### Part 1 — Arithmetic value-returning ops
`reduce_sum`, `reduce_product`, `reduce_nansum`, `reduce_nanproduct`

These ops **add or multiply** values — they can overflow. Accumulator widening is mandatory.
- int32 sum of N × 10^9 elements exceeds INT32_MAX immediately → must accumulate in int64_t
- float sum of N = 10^6 random values → relative error ~0.1 without double accumulator

**Status after all changes:** Correct. `AccT = AccumulatorType<T>` (widened). CPU float→double,
GPU float→float (matching PyTorch's `opmath_type`). No changes needed in this session.

### Part 2 — Comparison value-returning ops
`reduce_min`, `reduce_max`, `reduce_nanmin`, `reduce_nanmax`

These ops **compare** values — the result is always one of the input values. They can never
overflow or lose precision. Accumulator widening adds overhead with zero benefit.

**Status after Change 9 (session before this one):** Fixed. `AccT = T` in all 4 op structs.
Output dtype matches input dtype. Shared memory halved for integer compare ops on GPU.
Matches PyTorch and NumPy behaviour.

### Part 3 — Index-returning ops
`reduce_argmin`, `reduce_argmax`, `reduce_nanargmin`, `reduce_nanargmax`

These ops find the **index** of the best value. The accumulator must track both the best value
seen so far AND its position.

**Original approach:** `ValueIndex<T>` struct `{T value; int64_t index;}` — constructed per element.
**PyTorch approach:** Two independent scalars `T best_val` + `int64_t best_idx`, index written
only when value strictly improves.

**Status after Change 10 (this session):** Fixed on CPU. GPU unchanged (struct overhead = 0
on GPU since warp shuffle already operates field-by-field). NaN check reduced from 2 to 1
per element using IEEE 754 property.

### Part 4 — Boolean ops
`reduce_all`, `reduce_any`

These ops use `bool` as accumulator type — `AllOp` (AND) and `AnyOp` (OR). The accumulator
type was already correct (`AccT = bool`). The missing feature was **early exit**:
- `AND(false, x) = false` for all x
- `OR(true, x) = true` for all x

Once the result is determined, continuing the inner loop is pure wasted work.

**Status after Change 11 (this session):** Fixed on CPU with `break`. GPU intentionally has no
short-circuit (SIMT lockstep). Matches PyTorch CPU behaviour.

### Part 5 — Statistical ops
`reduce_mean`, `reduce_variance`, `reduce_std`, `reduce_nanmean`, `reduce_nanvar`, `reduce_nanstd`

These ops have both a summation phase (uses Part 1 accumulator) and additional logic:
- **mean**: sum ÷ count
- **variance**: Σ(xi - mean)² ÷ (count - correction)
- **std**: sqrt(variance)

**GPU problem found in this session:** `reduce_mean_kernel` used hardcoded `double` for all
non-complex types. Fixed in Change 12. CPU integer path intentionally stays Float64 (Change 13).

**Pending:** `reduce_std` GPU currently calls sqrt on the float variance result. Correct
but documented as needing review if precision edge cases arise.

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

### master_gau (current, after all changes)

Accumulator type depends on **both the input dtype AND the operation**:

**For sum / product / mean / variance (arithmetic ops — can overflow):**

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

**For min / max / nanmin / nanmax (compare ops — output always within input range):**

| Input | CPU Accumulator | GPU Accumulator | Output dtype |
|-------|----------------|----------------|--------------|
| int8_t | **int8_t** | **int8_t** | Int8 |
| int16_t | **int16_t** | **int16_t** | Int16 |
| int32_t | **int32_t** | **int32_t** | Int32 |
| int64_t | int64_t | int64_t | Int64 |
| uint8_t | **uint8_t** | **uint8_t** | UInt8 |
| uint32_t | **uint32_t** | **uint32_t** | UInt32 |
| float | **float** | **float** | Float32 |
| float16_t | **float16_t** | **float16_t** | Float16 |
| double | double | double | Float64 |

(Bold = changed from the old state where these all used widened types)

**How master_gau compares:**

| Feature | PyTorch | TensorFlow | master_gau |
|---------|---------|------------|------------|
| Centralized accumulator | Yes, 1 file | No, per-kernel | Yes, 1 struct (after change) |
| Device-parameterized | Yes (`is_cuda`) | No | Yes (`IsGPU`) |
| float CPU accumulator | double | float | **double** |
| float GPU accumulator | float | float | float |
| mean GPU accumulator | float | float | **float** (was double, fixed Change 12) |
| half/bf16 accumulator | float | float | float |
| int overflow protection | int64_t | none | int64_t |
| unsigned overflow | not supported | not supported | **uint64_t** |
| bool accumulator | bool | not handled | int64_t |
| bool short-circuit CPU | Yes | No | **Yes** (Change 11) |
| bool short-circuit GPU | No (SIMT) | No | No (SIMT limitation) |
| complex promotion | complex\<double\> on CPU | none | none (stays same) |
| Kahan summation | Yes, higher-level | No | Removed (double acc replaces it) |
| argmin/argmax CPU impl | 2-variable scalars | 2-variable scalars | **2-variable** (was struct, fixed Change 10) |
| argmin/argmax GPU impl | struct (ValueIndex equiv.) | separate fields | struct (ValueIndex, unchanged — zero overhead) |
| int mean CPU output | not supported (throws) | float32 | **Float64** (exact, free on CPU) |
| int mean GPU output | not supported (throws) | float32 | **Float32** (32× faster than float64 on GPU) |

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

**Bug B5: compare ops (min/max) using widened accumulator unnecessarily**
- Location: `include/ops/helpers/ReductionOps.h` — `MinOp`, `MaxOp`, `NanMinOp`, `NanMaxOp`
- Symptom (correctness): `reduce_min<int32>` returned Int64 tensor instead of Int32. Output dtype was wrong — PyTorch returns the same dtype as input for min/max.
- Symptom (performance): 1 extra MOVSX per element on CPU; 2–4× slower comparison instruction on pre-Volta GPUs (int64 emulated as 2 × 32-bit ops on Pascal/Maxwell/Kepler).
- Fix: Changed `using AccT = AccumulatorType<T>` → `using AccT = T` in all 4 compare op structs. Cascaded fixes to output dtype, `OutputCppT`, accumulator init block, GPU kernel accumulator type, GPU dispatcher output dtype, and GPU shared memory size calculation.

**Bug B6: PackedMetadata `cudaHostAlloc` per-call overhead**
- Location: `src/UnaryOps/cuda/ReductionImplGPU.cu`, `PackedMetadata` constructor/destructor
- Symptom: Every reduction call paid ~300+ µs for `cudaHostAlloc` + `cudaFreeHost` driver round-trips regardless of tensor size. The CUDA driver acquires a global device-context lock on each call.
- Fix: Replaced `PinnedCPUAllocator` (pinned buffer) with a local `std::vector<int64_t>` (pageable heap). For tiny metadata buffers, CUDA's internal staging costs ~1–2 µs — far below the driver lock overhead.

**Bug B7: GPU mean kernel hardcoded `double` accumulator for all non-complex types**
- Location: `include/ops/helpers/ReductionKernels.cuh` `reduce_mean_kernel`, line ~536; `src/UnaryOps/cuda/ReductionImplGPU.cu` `dispatch_mean_gpu` and `dispatch_variance_gpu`
- Symptom: `mean<float>` on GPU was 32× slower than necessary. `mean<int32>` was also 32× slower. Output was `Float64` for integer tensors — unintentional, inconsistent with GPU's FP32 preference.
- Root cause: `using AccT = ... double` hardcoded without considering that FP64 = 1/32 FP32 on consumer GPUs (Pascal/Turing/Ada).
- Fix: Changed to `float` for all non-double, non-complex types. `dispatch_mean_gpu` now outputs `Float32` for integers. `dispatch_variance_gpu` cascade-updated to match (MeanCppT, OutputCppT, AccCppT all changed to float for integral T).
- Also fixed: `shared_mem_size` in `dispatch_mean_gpu` was using `sizeof(double)` for AccT allocation — changed to `mean_acc_size` (4 bytes for float, 8 for double) so shared memory is not over-allocated.

**Bug B8: CPU index ops (argmin/argmax) — 2 NaN checks per element in hot path, struct overhead**
- Location: `include/ops/helpers/ReductionOps.h` (ArgMinOp/ArgMaxOp reduce method), `include/ops/helpers/ReductionImpl.h` (CPU index path)
- Symptom: Two `is_nan_check()` calls per element in all code paths (hot path with no NaN data). Also struct `ValueIndex<T>` constructed on stack per element.
- Fix: Added `better_than()` using IEEE 754 property (NaN < x = false eliminates the need for a second check when current_best is NaN). Changed CPU loop from struct-based to 2-variable scalars. Result: 1 NaN check in hot path instead of 2, no struct construction per element.

---

## Part 8 — Pending work

### Completed in this session (were in earlier pending list)

- **Index ops 2-variable approach**: Done (Change 10). CPU path now uses `T best_val` + `int64_t best_idx`.
- **Short-circuit for reduce_all/reduce_any**: Done (Change 11). `break` on first determination.
- **GPU mean double accumulator**: Done (Change 12). Now `float` for all non-double, non-complex types.
- **Integer mean/variance GPU output dtype**: Done (Float32 instead of Float64 on GPU).
- **CPU/GPU consistency analysis**: Done (Change 13). CPU stays Float64 (free precision), GPU Float32 (hardware constraint). Documented.

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

| File | All changes |
|------|-------------|
| `include/ops/helpers/ReductionOps.h` | Replaced 1-param struct with 2-param `IsGPU` struct; float→double on CPU; unsigned→uint64_t; int8_t added; renamed `G`→`IsGPU`; **MinOp/MaxOp/NanMinOp/NanMaxOp AccT = T**; **ArgMinOp/ArgMaxOp/NanArgMinOp/NanArgMaxOp: added `identity_val()` + `better_than()` (CPU 2-variable helpers)**; **AllOp/AnyOp: added `can_short_circuit()`** |
| `include/ops/helpers/ReductionImpl.h` | Removed Kahan summation; output dtype + OutputCppT + accumulator init updated for compare ops; **CPU index path replaced: ValueIndex struct → 2-variable `best_val`/`best_idx`**; **bool path inner loop: added `break` for short-circuit**; **CPU integer mean comment: Float64 output is intentional (free precision, documented asymmetry vs GPU Float32)** |
| `include/ops/helpers/ReductionKernels.cuh` | Replaced inline conditional_t chain with unified struct; compare ops branch to T; **`reduce_mean_kernel` AccT: `double`→`float` for non-double, non-complex types** |
| `include/core/TensorDispatch.h` | Added `case Dtype::Int8:` to both dispatch switch statements |
| `src/UnaryOps/cuda/ReductionImplGPU.cu` | Added 9 int8_t instantiations; removed `__cxa_demangle` debug call; removed `cudaHostAlloc` pinned buffer → `std::vector`; output dtype + OutputCppT + shared_mem_size updated for compare ops; **`dispatch_mean_gpu`: output dtype Float64→Float32 for integers, `OutputCppT` double→float, `shared_mem_size` `sizeof(double)`→`mean_acc_size`**; **`dispatch_variance_gpu`: output dtype Float64→Float32, `MeanCppT` double→float, `OutputCppT` double→float, `AccCppT` double→float for integral T** |
| `Tests/ReductionsTests/accumulator_type_test.cpp` | New file: 22 tests across 6 sections |
| `Tests/ReductionsTests/kahan_precision_test.cpp` | Existing file: precision+timing baseline vs after changes |
| `Tests/ReductionsTests/baseline_before_changes.txt` | Saved precision baseline (Kahan active, float accumulator) |

---

## Part 10 — Summary of all design decisions and why

| Decision | Chosen approach | Alternatives considered | Reason |
|----------|----------------|------------------------|--------|
| float CPU accumulator | `double` | Keep `float`, use Kahan | Double more precise than Kahan, no overhead on CPU |
| float GPU accumulator | `float` | `double` | GPU FP64 = 1/32 FP32 on consumer hardware |
| compare ops AccT | `T` (no widening) | Keep `AccumulatorType<T>` | Compare ops can never overflow; widening gave wrong output dtype |
| unsigned AccT | `uint64_t` | Keep `int64_t` | uint64 values > INT64_MAX would become negative in int64 |
| index ops CPU | 2 scalars (`best_val`/`best_idx`) | Keep `ValueIndex<T>` struct | ~2–3 fewer instructions/element; struct overhead = 0 on GPU |
| index ops GPU | Keep `ValueIndex<T>` unchanged | 2 scalars | Warp shuffle already operates field-by-field; struct overhead is zero |
| NaN check in `better_than` | 1 check (IEEE 754 trick) | 2 checks | `NaN < x = false` (IEEE) means current_best=NaN case handled by comparison itself |
| bool short-circuit | `break` in CPU inner loop | None (full traversal) | Once AND=false or OR=true the result is determined; O(1) branch overhead |
| bool short-circuit GPU | Not implemented | Implement via atomic | SIMT lockstep: all threads in warp execute same instruction; `break` would diverge |
| mean kernel AccT (GPU) | `float` for non-double | `double` | 32× faster on consumer hardware; matches PyTorch `opmath_type` |
| int mean CPU output dtype | `Float64` | `Float32` for consistency | FP64 division is free on CPU (~6 cycle difference, 1 op per output element not per input element) |
| int mean GPU output dtype | `Float32` | `Float64` | FP64 = 32× slower on GPU; CPU's Float64 is intentionally different |
| "cast before" vs "cast at load" | Cast at load (inside reduction loop) | Allocate new float tensor first | No extra memory allocation; int64 accumulation is more precise than float; same cost |
| Kahan summation | Removed | Keep Kahan, add double too | Double accumulation is strictly better for catastrophic cancellation; Kahan has serial dependency chain |
