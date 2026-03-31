# Bifurcation 1: Vectorized Sum-like Operations - Implementation Plan

## **Pattern Confirmation from PyTorch**

### Operations in Bifurcation 1:
1. **sum** (float/double): `cascade_sum` (SPECIAL)
2. **sum** (integers): `binary_kernel_reduce_vec`
3. **nansum** (float/double): `cascade_sum` (SPECIAL)
4. **prod**: `binary_kernel_reduce_vec`
5. **all** (logical AND): `binary_kernel_reduce_vec`
6. **any** (logical OR): `binary_kernel_reduce_vec`

### Key Pattern:
```
Floating-point sum/nansum:
  └─ Use: cascade_sum (custom algorithm with 4-level cascading + vectorization)

Everything else in Bifurcation 1:
  └─ Use: binary_kernel_reduce_vec (generic vectorized kernel)
```

---

## **Cascade Sum Algorithm - Deep Dive**

### Why Cascade Sum?
- **Problem**: Summing massive arrays of floats causes catastrophic cancellation
- **Eigen's Solution**: Deep recursive pairwise tree (slow on CPU due to cache thrashing)
- **PyTorch's Solution**: Shallow 4-level cascade with vectorization (fast + accurate)

### Algorithm Structure (from SumKernel.cpp lines 537-612):

```
cascade_sum(TensorIterator iter):
  1. Fill output with zeros
  2. Call iter.parallel_reduce() with custom lambda

  Inside the lambda:
    ├─ Layout detection:
    │  ├─ If outer reduction needed (output_strides both non-zero)
    │  │  └─ Use basic_loop (fallback for irregular layouts)
    │  │
    │  └─ If true reduction (output_strides[0] == 0):
    │     ├─ Detect layout: Inner-contiguous or Outer-contiguous
    │     │
    │     ├─ Path 1: VECTORIZED INNER SUM (in_strides[0] == sizeof(scalar_t))
    │     │  └─ vectorized_inner_sum<acc_t, VecLoadPolicy, ScalarLoadPolicy, StorePolicy>()
    │     │
    │     ├─ Path 2: VECTORIZED OUTER SUM (in_strides[1] == sizeof(scalar_t))
    │     │  └─ vectorized_outer_sum<acc_t, VecLoadPolicy, ScalarLoadPolicy, StorePolicy>()
    │     │
    │     └─ Path 3: SCALAR (generic fallback)
    │        ├─ scalar_inner_sum or
    │        └─ scalar_outer_sum
```

### Vectorized Inner Sum (lines 434-461):
```cpp
for each output row j:
  1. Load vector-sized chunks using VecLoadPolicy
  2. Call row_sum<vacc_t, VecLoadPolicy>() → returns vacc_t (vectorized accumulator)
  3. Accumulate remaining scalar elements
  4. Combine vector partials + scalar result
  5. Store to output using StorePolicy
```

### Vectorized Outer Sum (lines 476-510):
```cpp
Process in groups of (nrows=4, vec_size) for better cache:
  1. Load 4 vectors at once using multi_row_sum()
  2. Call row_sum() per row
  3. Store results individually

Handle remaining elements:
  ├─ Full vectors (size >= vec_size)
  └─ Scalar tail
```

### Key Components:

**1. Load Policies (different for different cases):**
- `InnerSumCastLoadPolicy<vec_t, vacc_t>`: Load full vector, cast to accumulator type
- `OuterSumCastLoadPolicy<vec_t, vacc_t>`: Load partial vector (vacc_t size), cast
- `NanSumLoadPolicy`: Ignore NaN values during load
- `CastLoadPolicy`: Simple cast without vectorization

**2. Store Policy (CastStoreAccumulate<scalar_t, acc_t>):**
- Cast accumulator (double) back to output type (float)
- Accumulate if output is float32 (cast back and accumulate)

**3. Row Sum Function (lines 413-431):**
```
row_sum():
  1. Process in ILP_FACTOR=4 independent accumulators
  2. multi_row_sum() divides row into chunks, gets 4 partial sums
  3. Handle remainder elements
  4. Combine 4 partials into final result
```

**4. Multi Row Sum (lines 346-410):**
- Uses num_levels = ceil(log4(num_threads)) = 4 levels max
- Creates 4 accumulators per level
- Cascades results up through levels
- Final reduction combines all

---

## **Binary Kernel Reduce Vec - Pattern**

For all other Bifurcation 1 ops (prod, all, any):

```cpp
binary_kernel_reduce_vec(
    iter,
    [=](scalar_t a, scalar_t b) -> scalar_t { return op(a, b); },  // scalar
    [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return vop(a, b); }  // SIMD
);
```

Examples:
- **prod**: `[](a,b){ return a*b; }` + SIMD multiply
- **all**: `[](a,b){ return a && b; }` + SIMD AND
- **any**: `[](a,b){ return a || b; }` + SIMD OR

---

## **Implementation Roadmap for Our Library**

### Phase 1: Infrastructure (Vectorization Engine)
```
Create: VectorizationOps.h
├─ struct VectorizedAdd
├─ struct VectorizedMul
├─ struct VectorizedMin
├─ struct VectorizedMax
├─ struct VectorizedAnd
├─ struct VectorizedOr
└─ Vectorized<T> type wrapper
```

### Phase 2: Load/Store Policies
```
Create: SumKernelPolicies.h
├─ LoadPolicy<T>: Basic load
├─ CastLoadPolicy<scalar_t, acc_t>: Load with cast
├─ InnerSumCastLoadPolicy: Vector load with cast
├─ OuterSumCastLoadPolicy: Partial vector load
├─ NanSumLoadPolicy: Load skipping NaN
└─ StorePolicy: Store with cast back
```

### Phase 3: Core Sum Functions
```
Create: SumKernelCore.h
├─ row_sum<acc_t, LoadPolicy>(): Sum one row with 4-accumulator ILP
├─ multi_row_sum<acc_t, nrows, LoadPolicy>(): Multi-row with cascading
├─ vectorized_inner_sum<acc_t, VecLoadPolicy, ...>()
├─ vectorized_outer_sum<acc_t, VecLoadPolicy, ...>()
├─ scalar_inner_sum<acc_t, LoadPolicy, ...>()
└─ scalar_outer_sum<acc_t, LoadPolicy, ...>()
```

### Phase 4: Cascade Sum Integration
```
Modify: ReductionImpl.h
├─ Add cascade_sum<ignore_nan, scalar_t>() function
├─ Modify sum dispatcher to call cascade_sum for floats
└─ Modify nansum dispatcher to call cascade_sum with ignore_nan=true
```

### Phase 5: Binary Kernel Reduce Vec Integration
```
Modify: ReductionImpl.h
├─ Add binary_kernel_reduce_vec for prod
├─ Add binary_kernel_reduce_vec for all/any
└─ Ensure proper vectorization ops are used
```

---

## **Implementation Order**

1. ✅ **Understand PyTorch code** (DONE)
2. **Vectorization infrastructure** (CPU SIMD wrappers)
3. **Load/Store policies** (type casting, NaN handling)
4. **Core sum helpers** (row_sum, multi_row_sum)
5. **Cascade sum algorithm** (main reduction logic)
6. **Binary kernel reduce vec** (for prod/all/any)
7. **Dispatcher integration** (route to correct kernel)
8. **Testing & validation**

---

## **Expected Results**

After implementation:
- ✅ Float sum/nansum: **cascade_sum** (accurate + fast)
- ✅ Integer sum: **binary_kernel_reduce_vec**
- ✅ Prod/All/Any: **binary_kernel_reduce_vec**
- ✅ Full vectorization on CPU (AVX2/AVX512 ready)
- ✅ Parallel with OpenMP (per-thread cascade)
- ✅ No recursive tree overhead
- ✅ Matches PyTorch accuracy and performance

