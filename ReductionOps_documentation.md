# Reduction Operations — Complete Documentation

---

## 0. What are Reduction Operations?

Reduction operations are tensor operations that **collapse one or more dimensions** of a tensor by aggregating the values along those dimensions using a mathematical combiner (sum, product, min, max, mean, etc.).

**Example:** A tensor of shape `[3, 4, 5]` reduced along axis 1 produces an output of shape `[3, 5]` — the 4 elements along dimension 1 are combined into a single value for each `(i, k)` position.

**Signature Format:**
```cpp
Tensor reduce_X(const Tensor& input, 
                const std::vector<int64_t>& axes = {},     // empty = reduce ALL dims
                bool keepdim = false,                       // retain reduced dims as size-1?
                cudaStream_t stream = 0);                   // GPU stream for async execution
```

### Complete List of Implemented Operations (25 total)

| # | Category | Operation | Identity | Output Type | Description |
|---|----------|-----------|----------|-------------|-------------|
| 1 | Core | `reduce_sum` | 0 | Same / Int64 | Sum of elements |
| 2 | Core | `reduce_product` | 1 | Same / Int64 | Product of elements |
| 3 | Core | `reduce_min` | +MAX | Same | Minimum value |
| 4 | Core | `reduce_max` | -MAX | Same | Maximum value |
| 5 | Core | `reduce_mean` | — | Same / Float64 | Arithmetic mean |
| 6 | NaN-Aware | `reduce_nansum` | 0 | Same | Sum, ignoring NaNs |
| 7 | NaN-Aware | `reduce_nanproduct` | 1 | Same | Product, ignoring NaNs |
| 8 | NaN-Aware | `reduce_nanmin` | +MAX | Same | Min, ignoring NaNs |
| 9 | NaN-Aware | `reduce_nanmax` | -MAX | Same | Max, ignoring NaNs |
| 10 | NaN-Aware | `reduce_nanmean` | — | Same | Mean over non-NaN values only |
| 11 | Index | `reduce_argmin` | — | Int64 | Index of minimum value |
| 12 | Index | `reduce_argmax` | — | Int64 | Index of maximum value |
| 13 | NaN Index | `reduce_nanargmin` | — | Int64 | Index of min, ignoring NaNs |
| 14 | NaN Index | `reduce_nanargmax` | — | Int64 | Index of max, ignoring NaNs |
| 15 | Boolean | `reduce_all` | true | Bool | Logical AND across elements |
| 16 | Boolean | `reduce_any` | false | Bool | Logical OR across elements |
| 17 | Variance | `reduce_var` | — | Same / Float64 | Variance (Bessel's correction) |
| 18 | Variance | `reduce_nanvar` | — | Same / Float64 | Variance, ignoring NaNs |
| 19 | Variance | `reduce_std` | — | Same / Float64 | Standard deviation |
| 20 | Variance | `reduce_nanstd` | — | Same / Float64 | Std dev, ignoring NaNs |
| 21 | Combined | `reduce_var_mean` | — | pair | (Variance, Mean) in single pass |
| 22 | Combined | `reduce_std_mean` | — | pair | (Std dev, Mean) in single pass |
| 23 | Autograd | `autograd::sum` | — | Same | Graph-tracked sum (backward: broadcast) |
| 24 | Autograd | `autograd::mean` | — | Same | Graph-tracked mean (backward: 1/N scale) |

> **Note:** `reduce_median` and `reduce_nanmedian` are declared in the header but **not yet implemented** (deferred to Phase 2 — requires sorting-based approach).

### Supported Data Types

| Type Family | Types | Standard Ops | NaN-Aware Ops | Notes |
|-------------|-------|:---:|:---:|-------|
| Floating Point | `float`, `double` | ✅ | ✅ | Full support |
| Half Precision | `float16_t`, `bfloat16_t` | ✅ | ✅ | Accumulated in `float`/`double` for precision |
| Integer | `int16_t`, `int32_t`, `int64_t`, `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t` | ✅ | ❌ | Output widened to Int64, mean outputs Float64 |
| Boolean | `bool` | ✅ (all/any) | ❌ | Non-bool inputs auto-converted via `to_bool()` |
| Complex | `complex32_t`, `complex64_t`, `complex128_t` | ✅ (sum, product, mean) | ✅ | Min/Max/ArgMin/ArgMax blocked (no ordering) |
| FP4 | `float4_e2m1_t`, `float4_e2m1_2x_t` | ❌ | ❌ | All reductions blocked at compile time |

---

## 1. Algorithm Design — Complete Flow

### 1.1 Forward Pass Flow

The complete journey of a reduction operation from user call to hardware execution:

```
User Code
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Public API (Reduction.h)                               │
│  reduce_sum(input, axes, keepdim, stream)               │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  CPU Implementation (Reduction.cpp)                     │
│  1. normalize_axes(input.shape(), axes)                 │
│  2. DISPATCH_ALL_TYPES(input.dtype()) {                 │
│         dispatch_reduction<T, SumOp>(...)               │
│     }                                                   │
└──────────────────────┬──────────────────────────────────┘
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
┌──────────────────┐  ┌───────────────────────┐
│  CPU Path        │  │  GPU Path             │
│  (ReductionImpl.h)│  │  (ReductionImplGPU.cu)│
│                  │  │                       │
│  reduce_kernel   │  │  dispatch_reduction   │
│  <T, OpType>()   │  │  _gpu<T, OpType>()    │
│                  │  │                       │
│  - OpenMP par.   │  │  - Type conversion    │
│  - Kahan sum     │  │    (float16→__half)   │
│  - Serial inner  │  │  - DeviceArray RAII   │
│    loop          │  │  - Kernel launch      │
└──────────────────┘  └───────────┬───────────┘
                                  │
                                  ▼
                      ┌───────────────────────┐
                      │  CUDA Kernels          │
                      │  (ReductionKernels.cuh)│
                      │                       │
                      │  reduce_kernel<<<>>>   │
                      │  - Block per output    │
                      │  - Warp shuffle        │
                      │  - Shared mem reduce   │
                      └───────────────────────┘
```

### 1.2 Reduction Utilities

These 5 utility functions prepare the metadata needed before any reduction kernel executes.

#### 1.2.1 normalize_axes
```
Input  : input_dims vector (shape of tensor), axes vector (dimensions to reduce)
Output : sorted, unique, positive-index axes vector
```

**What it does:** Converts the user's input set of axes (which might contain duplicates or negative indices) into a clean sorted list of positive indices (0 to N-1).

**Why we use this:**
1. To handle negative indices (e.g., axis=-1 on rank-4 tensor → axis 3)
2. To remove duplicates
3. To sort the axes in ascending order
4. To validate that indices are within correct bounds for the tensor's rank

**Logic:**
1. **Full reduction check:** If input axes vector is empty, the function assumes the user wants to reduce over all dimensions. It fills the set with all indices from 0 to ndim-1.
2. **Negative index handling (Normalization):** Loops through axes; if any axis < 0, adds ndims to convert to positive index. Ex: axis=-1, rank-4 tensor → 4+(-1)=3.
3. **Bounds checking/Validation:** After normalization, checks if index is within valid range [0, ndim-1]. Throws runtime error if not.
4. **Uniqueness and sorting:** Uses `std::set` to automatically handle duplicates and sort axes in ascending order.
5. **Final output:** Elements from the set are copied back into a sorted `std::vector`.

**Design decisions explained:**
- *Why `int64_t` for axes?* — To handle negative indices, support large tensor dimensions, and maintain type consistency with other int64_t variables used throughout the library.
- *Why return by value instead of reference?* — The result is a local variable; returning by reference would create a dangling pointer (the local goes out of scope).

#### 1.2.2 calculate_output_shape
```
Input  : input_dims vector, normalized_axes vector, keepdim boolean flag
Output : output_dims vector (shape of the result tensor)
```

**What it does:** Takes the original shape and the list of axes being reduced, and calculates the shape of the resulting output tensor.

**Why we use this:** The kernel needs to know the exact dimensions of the output before allocating memory for the result tensor.

**Logic:**
1. **Iteration:** Loops through every dimension index (i) of the input shape vector.
2. **Reduction check:** Uses a lambda function `is_reduced` with `std::find` to check if current dimension index is in the normalized_axes list.
3. **Logic branching:**
   - If `i` IS a reduced axis:
     - `keepdim=true` → push size 1 (e.g., `4×3×5` reduce axis 1 → `4×1×5`)
     - `keepdim=false` → skip the dimension entirely (e.g., `4×3×5` reduce axis 1 → `4×5`)
   - If `i` is NOT a reduced axis → simply copy the dimension size to the output shape.
4. **Handle scalar output:** If output shape is empty after the loop, it means all dimensions were reduced to a single scalar → push size 1.

**Design decisions explained:**
- *Why `size_t` for loop variable, not `int`?* — Avoids signed/unsigned comparison dangers. When comparing signed with unsigned, the signed value is implicitly converted to unsigned, which can catastrophically fail for negative numbers (e.g., -1 becomes a huge positive number).
- *Backward loop edge case:* `for(int64_t i = input_dims.size()-1; i>=0; --i)` — if the tensor is empty, `size()-1` wraps to a huge positive value when implicitly converted to unsigned, causing an infinite loop.

#### 1.2.3 calculate_reduced_count
```
Input  : input_dims vector, normalized_axes vector
Output : reduced_count (int64_t) — total number of elements combined per output element
```

**What it does:** Calculates the total number of input elements that will be combined to produce each output element.

**Why we use this:** This count is the divisor for `reduce_mean` and `reduce_nanmean` operations, and also determines the inner loop count in the reduction kernel.

**Logic:**
1. **Full reduction:** If normalized_axes is empty (all dims being reduced), uses `std::accumulate` with `std::multiplies` to compute product of all dimensions in input_dims.
2. **Partial reduction:** Initializes count to 1 and multiplies it by the size of each dimension in normalized_axes.

#### 1.2.4 unravel_index (Successive Division and Modulo Method)
```
Input  : linear_index (offset), shape vector
Output : coordinate vector (e.g., {i, j, k})
```

**What it does:** Converts a flat 1D linear index to multi-dimensional coordinates. Assumes C-order (row-major) layout.

**Why we use this:** Given the linear index of an output element, we need to find its corresponding multi-dimensional coordinates in the tensor.

**Logic (C-order):**
1. **Backward iteration:** Loop from last/innermost dimension to first/outermost.
2. **Modulo:** `coords[i] = temp_index % shape[i]` — gives the index within the current dimension.
3. **Division:** `temp_index /= shape[i]` — updates the index to the next higher dimension.

**Mathematical basis:**
```
C-order offset = C0×(D1×D2) + C1×(D2) + C2×(1)

Step 1: C2 = offset % D2;         next_offset = offset / D2      → gives C0×D1 + C1
Step 2: C1 = next_offset % D1;    next_offset = next_offset / D1  → gives C0
Step 3: C0 = next_offset % D0
```
Where C0,C1,C2 are coordinates and D0,D1,D2 are dimensions.

**Two implementations exist:**
- **Heap version:** `unravel_index()` — returns `std::vector<int64_t>` (allocates memory) → used in non-performance paths.
- **Stack version:** `unravel_index_stack()` — writes to pre-allocated `int64_t[]` buffer → zero allocations, used in the hot kernel loop.

#### 1.2.5 ravel_index (Inverse of Unravel)
```
Input  : coords vector, strides vector
Output : linear_index (int64_t)
```

**What it does:** Converts multi-dimensional coordinates back to a single linear index.
```
linear_index = Σ(coord[i] × stride[i]) for i = 0 to N-1
```
where `stride[i] = product of all dimensions after i`.

**Example:** Coords `{3,1,2}` in tensor shape `{4,5,6}` → `3×30 + 1×6 + 2×1 = 98`.

**Two implementations exist:**
- **Heap version:** `ravel_index()` — takes `std::vector` arguments.
- **Stack version:** `ravel_index_stack()` — takes raw pointer + size → used in kernel hot path.

---

### 1.3 The Reduction Kernel Algorithm

**Core Question:** *"For this specific cell in the output, which values from the input contribute to it?"*

**Analogy:** The reduction kernel is like a person standing in the Output Tensor trying to find all their family members in the Input Tensor.

#### Algorithm Steps:
```
Step 1: Preparatory math
        → reduced_dims, output_shape, reduced_count

Step 2: Outer loop (iterate output tensor, iterations = num_output_elements)
        → For each output element:

Step 3:   Inner loop (iterate reduced dimensions, iterations = reduced_count)
          → For each contributing input element:

Step 4:     Merge (combine outer_unravel coords + inner_unravel coords)
            → Produces full input coordinates

Step 5:     Ravel (full_input_coords, input_strides)
            → Produces the linear index into the input data

Step 6:     Accumulate (input_data[linear_index])
            → Accumulates the value using the operation's reduce() function
```

**In pseudocode:**
```
preparatory_math:
    input_shape, input_strides, reduction_axes
    reduced_dims_shape = [input_shape[ax] for ax in reduction_axes]
    reduced_count = product(reduced_dims_shape)
    output_shape = calculate_output_shape(...)
    outer_loop_count = product(output_shape)

for output_index = 0 to outer_loop_count:
    out_coords = unravel(output_index, output_shape)
    accumulator = op.identity()
    
    for i = 0 to reduced_count:
        slice_coords = unravel(i, reduced_dims_shape)
        
        // Merge: interleave output coords and slice coords
        full_input_coords = []
        for each dim in input:
            if dim is reduced:
                full_input_coords[dim] = slice_coords[next_reduced_idx]
            else:
                full_input_coords[dim] = out_coords[next_output_idx]
        
        linear_index = ravel(full_input_coords, input_strides)
        accumulator = op.reduce(accumulator, input[linear_index])
    
    output[output_index] = accumulator
```

---

### 1.4 Worked Examples (3×3×3 Tensor, values 1–27)

#### Case 1: Partial Reduction over Axis [0] (Planes axis)

**Constants:**
- Input Shape: `[3, 3, 3]`, Input Strides: `[9, 3, 1]`
- Reduction Axis: `[0]`
- Reduced Dims Shape: `[3]`
- Output Shape: `[3, 3]` (9 elements)
- Inner Loop Count: 3

**Trace:**

**Output Index 0:**
- Outer Unravel: `unravel(0, [3,3])` → `{0, 0}` (Row 0, Col 0 of Output)
- Inner Loop:
  - `i=0`: `unravel(0, [3])` → `{0}`. Merge → `{0, 0, 0}`. Ravel: `0×9 + 0×3 + 0×1 = 0`
  - `i=1`: `unravel(1, [3])` → `{1}`. Merge → `{1, 0, 0}`. Ravel: `1×9 + 0×3 + 0×1 = 9`
  - `i=2`: `unravel(2, [3])` → `{2}`. Merge → `{2, 0, 0}`. Ravel: `2×9 + 0×3 + 0×1 = 18`
- **Result:** `Output[0] = Input[0] + Input[9] + Input[18]`

**Output Index 1:**
- Outer Unravel: `unravel(1, [3,3])` → `{0, 1}`
- Inner Loop:
  - `i=0`: Merge → `{0, 0, 1}`. Offset 1.
  - `i=1`: Merge → `{1, 0, 1}`. Offset 10.
  - `i=2`: Merge → `{2, 0, 1}`. Offset 19.
- **Result:** `Output[1] = Input[1] + Input[10] + Input[19]`

**Output Index 3:**
- Outer Unravel: `unravel(3, [3,3])` → `{1, 0}`
- Inner Loop:
  - `i=0`: Merge → `{0, 1, 0}`. Offset 3.
  - `i=1`: Merge → `{1, 1, 0}`. Offset 12.
  - `i=2`: Merge → `{2, 1, 0}`. Offset 21.

**Output Index 4:**
- Outer Unravel: `unravel(4, [3,3])` → `{1, 1}`
- `i=0`: `{0, 1, 1}` → Offset 4. | `i=1`: `{1, 1, 1}` → Offset 13. | `i=2`: `{2, 1, 1}` → Offset 22.

**Output Index 5:**
- Outer Unravel: `unravel(5, [3,3])` → `{1, 2}`
- `i=0`: `{0, 1, 2}` → Offset 5. | `i=1`: `{1, 1, 2}` → Offset 14. | `i=2`: `{2, 1, 2}` → Offset 23.

**Output Index 6:**
- Outer Unravel: `unravel(6, [3,3])` → `{2, 0}`
- `i=0`: `{0, 2, 0}` → Offset 6. | `i=1`: `{1, 2, 0}` → Offset 15. | `i=2`: `{2, 2, 0}` → Offset 24.

**Output Index 7:**
- Outer Unravel: `unravel(7, [3,3])` → `{2, 1}`
- `i=0`: `{0, 2, 1}` → Offset 7. | `i=1`: `{1, 2, 1}` → Offset 16. | `i=2`: `{2, 2, 1}` → Offset 25.

**Output Index 8:**
- Outer Unravel: `unravel(8, [3,3])` → `{2, 2}`
- `i=0`: `{0, 2, 2}` → Offset 8. | `i=1`: `{1, 2, 2}` → Offset 17. | `i=2`: `{2, 2, 2}` → Offset 26.

---

#### Case 2: Partial Reduction over Axes [0, 1] (Collapse Planes and Rows)

**Constants:**
- Reduction Axes: `[0, 1]`
- Reduced Dims Shape: `[3, 3]`
- Output Shape: `[3]` (only Column dimension survives)
- Inner Loop Count: 3×3 = 9

**Output Index 0** (Column 0):
- Outer Unravel: `unravel(0, [3])` → `{0}`
- Inner Loop (i=0 to 8):

| i | inner unravel | Merge | Offset |
|---|--------------|-------|--------|
| 0 | `{0, 0}` | `{0, 0, 0}` | 0 |
| 1 | `{0, 1}` | `{0, 1, 0}` | 3 |
| 2 | `{0, 2}` | `{0, 2, 0}` | 6 |
| 3 | `{1, 0}` | `{1, 0, 0}` | 9 |
| 4 | `{1, 1}` | `{1, 1, 0}` | 12 |
| 5 | `{1, 2}` | `{1, 2, 0}` | 15 |
| 6 | `{2, 0}` | `{2, 0, 0}` | 18 |
| 7 | `{2, 1}` | `{2, 1, 0}` | 21 |
| 8 | `{2, 2}` | `{2, 2, 0}` | 24 |

**Result:** `Output[0] = Sum of Input at offsets (0, 3, 6, 9, 12, 15, 18, 21, 24)`

**Output Index 1** (Column 1):
- Outer Unravel: `unravel(1, [3])` → `{1}`
- All merge results get `{_, _, 1}` as last coordinate.
- Offsets: `(1, 4, 7, 10, 13, 16, 19, 22, 25)`

**Output Index 2** (Column 2):
- Offsets: `(2, 5, 8, 11, 14, 17, 20, 23, 26)`

---

#### Case 3: Full Reduction (Axes [0, 1, 2])

**Constants:**
- Reduced Dims Shape: `[3, 3, 3]`
- Output Shape: `[1]` (scalar)
- Inner Loop Count: 27

**Output Index 0:**
- Outer Unravel: `unravel(0, [1])` → `{ }` (empty coordinates — only one output position exists)
- Inner Loop (i=0 to 26): Since 100% of dimensions are reduced, inner unravel directly gives the full input coordinates.
  - `i=0`: `{0,0,0}` → Offset 0
  - `i=1`: `{0,0,1}` → Offset 1
  - ...
  - `i=9`: `{1,0,0}` → Offset 9
  - ...
  - `i=26`: `{2,2,2}` → Offset 26

**Result:** `Output[0] = Sum of every element (offsets 0 through 26)`

> **Optimization Note:** Because full reduction produces a 1-to-1 mapping (i == offset), the library can detect this case and skip ravel/unravel entirely, using a simple `for(i=0; i<N; i++) accumulator += input[i]` loop.

---

### 1.5 Autograd Integration

The autograd system wraps reduction operations to automatically track gradients for backpropagation.

**Forward (autograd::sum):**
```cpp
Tensor sum(const Tensor& x) {
    return make_unary_op<SumBackward>(
        x,
        [](const Tensor& input) { return reduce_sum(input); },
        x.shape()    // saved for backward: original input shape
    );
}
```

**Forward (autograd::mean):**
```cpp
Tensor mean(const Tensor& x) {
    return make_unary_op<MeanBackward>(
        x,
        [](const Tensor& input) { return reduce_mean(input); },
        x.shape(),   // saved for backward: original input shape
        x.numel()    // saved for backward: total number of elements
    );
}
```

**Backward (SumBackward::apply):**
- `grad_input = ones(input_shape) * grad_output`
- Every element of the input contributed equally to the sum → each gets the full upstream gradient.

**Backward (MeanBackward::apply):**
- `grad_input = ones(input_shape) * (grad_output / numel)`
- Each element's contribution was scaled by `1/N` in the forward pass → the gradient is also scaled by `1/N`.

---

## 2. System Design — System Engineering

### 2.1 File Architecture

| File | Location | Role |
|------|----------|------|
| [`Reduction.h`](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensor_centric_tensorlib/include/ops/UnaryOps/Reduction.h) | `include/ops/UnaryOps/` | **Public API** — declares all `reduce_X()` functions with default args |
| [`Reduction.cpp`](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensor_centric_tensorlib/src/UnaryOps/cpu/Reduction.cpp) | `src/UnaryOps/cpu/` | **CPU implementation** — calls dispatchers, implements variance/std/combined stats |
| [`ReductionUtils.h`](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensor_centric_tensorlib/include/ops/helpers/ReductionUtils.h) | `include/ops/helpers/` | **Utilities** — normalize_axes, output_shape, reduced_count, ravel/unravel |
| [`ReductionOps.h`](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensor_centric_tensorlib/include/ops/helpers/ReductionOps.h) | `include/ops/helpers/` | **Operator structs** — SumOp, MinOp, ArgMaxOp, VarianceOp, etc. + GPU intrinsics |
| [`ReductionImpl.h`](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensor_centric_tensorlib/include/ops/helpers/ReductionImpl.h) | `include/ops/helpers/` | **CPU kernel + dispatcher templates** — reduce_kernel(), dispatch_reduction() |
| [`ReductionKernels.cuh`](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensor_centric_tensorlib/include/ops/helpers/ReductionKernels.cuh) | `include/ops/helpers/` | **CUDA kernels** — reduce_kernel<<<>>>, reduce_index_kernel<<<>>>, reduce_mean_kernel<<<>>>, reduce_variance_kernel<<<>>> |
| [`ReductionImplGPU.cu`](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensor_centric_tensorlib/src/UnaryOps/cuda/ReductionImplGPU.cu) | `src/UnaryOps/cuda/` | **GPU dispatchers** — dispatch_reduction_gpu(), type conversion, explicit template instantiations |
| [`ReductionBackward.h`](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensor_centric_tensorlib/include/autograd/backward/ReductionBackward.h) | `include/autograd/backward/` | **Backward node declarations** — SumBackward, MeanBackward classes |
| [`ReductionBackward.cpp`](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensor_centric_tensorlib/src/autograd/backward/ReductionBackward.cpp) | `src/autograd/backward/` | **Backward implementations** — apply() methods for gradient computation |
| [`ReductionOps.cpp`](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensor_centric_tensorlib/src/autograd/operations/ReductionOps.cpp) | `src/autograd/operations/` | **Autograd forward wrappers** — autograd::sum(), autograd::mean() |

### 2.2 Template Metaprogramming Patterns

**The OpType pattern:** Every reduction operation is defined as a `struct` with a standard interface:
```cpp
template <typename T>
struct SumOp {
    using AccT = AccumulatorType<T>;        // Type used for accumulation
    AccT identity() const;                  // Starting value (0 for sum, 1 for product)
    AccT reduce(const AccT& a, const AccT& b) const;  // Combine two values
};
```

This allows the reduction kernel to be written **once** and instantiated for any operation:
```cpp
template <typename T, template <typename> class OpType>
Tensor reduce_kernel(const Tensor& input, ...);
// Instantiated as: reduce_kernel<float, SumOp>
//                   reduce_kernel<double, MaxOp>
//                   reduce_kernel<float, NanArgMinOp>
// ...etc
```

**The `AccumulatorType<T>` trait:** Selects the safe accumulator type at compile time:
```cpp
template<> struct AccumulatorTypeSelector<float16_t>  { using type = float; };
template<> struct AccumulatorTypeSelector<int32_t>    { using type = int64_t; };
template<> struct AccumulatorTypeSelector<bool>       { using type = int64_t; };
```

### 2.3 CPU/GPU Dual-Path Design

The same `ReductionOps.h` operator structs compile for **both** CPU and GPU using conditional macros:

```cpp
// In operator reduce() methods:
#ifdef __CUDA_ARCH__
    // GPU device code: use CUDA intrinsics (__hadd, __hlt, __hisnan)
    return gpu_add(a, b);
#else
    // CPU host code: use standard C++ operators
    return a + b;
#endif
```

**Type conversion layer (GPU only):**
Custom library types (`float16_t`, `bfloat16_t`) are **bitwise identical** to CUDA native types (`__half`, `__nv_bfloat16`), enabling zero-cost `reinterpret_cast<>` conversion at kernel launch boundaries.

### 2.4 Dispatcher Architecture

The dispatcher function performs **compile-time validation** before routing to the correct kernel:
```
dispatch_reduction<T, OpType>(input, axes, keepdim, stream)
    │
    ├── Compile-time check: Is this a NaN op on a non-float type? → throw error
    ├── Compile-time check: Is this an all/any op on non-bool? → auto-convert to bool
    ├── Compile-time check: Is this FP4? → throw error (not supported)
    │
    ├── Runtime check: Is input on CUDA device?
    │   ├── YES → dispatch_reduction_gpu<T, OpType>(...)
    │   │         (Does CudaNativeType conversion, launches CUDA kernel)
    │   └── NO  → reduce_kernel<T, OpType>(...)
    │             (OpenMP-parallelized CPU kernel)
    │
    └── Special routing for index vs value reductions:
        ├── ArgMin/ArgMax → dispatch_index_reduction_gpu (separate kernel)
        └── All others    → dispatch_reduction_gpu (standard kernel)
```

---

## 3. Numerical Stability

### 3.1 Kahan Summation (CPU, SumOp path)

**Problem:** When summing millions of floating-point numbers, rounding errors accumulate. If you sum `1e8` values of `1.0f` naïvely, you might get `16777216.0` instead of `100000000.0` due to float32 precision limits.

**Solution:** Kahan compensated summation tracks the rounding error in a separate variable `c`:
```cpp
// Kahan summation loop
AccumulatorT kahan_sum = 0;
AccumulatorT kahan_c   = 0;   // compensation for lost low-order bits

for each value:
    AccumulatorT y = value - kahan_c;     // compensate
    AccumulatorT t = kahan_sum + y;       // add
    kahan_c = (t - kahan_sum) - y;        // update error term
    kahan_sum = t;
```

**Overflow/NaN safety:** If `kahan_sum` becomes `inf` or `NaN`, the algorithm falls back to simple accumulation.

### 3.2 Double Accumulation for Half Precision

| Input Type | CPU Accumulator | GPU Accumulator |
|-----------|----------------|----------------|
| `float16_t` | `double` | `float` |
| `bfloat16_t` | `double` | `float` |
| `float` | `float` (with Kahan) | `float` |
| `double` | `double` (with Kahan) | `double` |
| `int16_t`, `int32_t` | `int64_t` | `int64_t` |
| `bool` | `int64_t` | `int64_t` |

**Why:** FP16 has only ~3 decimal digits of precision. Summing even 1000 FP16 values directly would overflow or lose significant precision. By accumulating in float32 (GPU) or float64 (CPU), we preserve full precision during the computation, then convert back to FP16 only for the final result.

### 3.3 NaN Propagation vs NaN Skipping

**Standard operations (SumOp, MinOp, etc.):** NaN values **propagate** — if any input is NaN, the output is NaN. This follows IEEE 754 convention.
```cpp
// SumOp::reduce()
if (is_nan_check(a)) return a;   // NaN propagates
if (is_nan_check(b)) return b;
return a + b;
```

**NaN-aware operations (NanSumOp, NanMinOp, etc.):** NaN values are **skipped** — they are treated as if they don't exist.
```cpp
// NanSumOp::reduce()
if (is_nan_check(a)) return b;   // skip NaN, keep the other value
if (is_nan_check(b)) return a;
return a + b;
```

**NaN-aware mean special handling:** `reduce_nanmean` counts only non-NaN values per output slice and divides by that count (not the total reduced_count). If ALL values in a slice are NaN, the output is `NaN`.

### 3.4 Bessel's Correction for Variance/Std

```
var = Σ(xᵢ - x̄)² / (N - correction)
```

| `correction` | Formula | Name | Use Case |
|:---:|---|---|---|
| 1 (default) | `Σ(xᵢ - x̄)² / (N - 1)` | Sample variance | Estimating population variance from a sample |
| 0 | `Σ(xᵢ - x̄)² / N` | Population variance | When you have the entire population |

The `reduce_var` and `reduce_std` functions accept a `correction` parameter (default=1), consistent with PyTorch's convention.

### 3.5 Variance Two-Pass Algorithm

Rather than computing variance in a single pass (which suffers from catastrophic cancellation), the implementation uses a numerically stable two-pass approach:

```
Pass 1: mean = reduce_mean(input, axes)        // compute mean first
Pass 2: var  = Σ(xᵢ - mean)² / (N - correction)  // compute squared deviations from mean
```

On GPU, this is implemented as two separate kernel launches:
1. `dispatch_mean_gpu()` → produces mean tensor (with `keepdim=true` to preserve broadcasting shape)
2. `reduce_variance_kernel<<<>>>` → reads mean tensor + input tensor → computes squared deviations

---

## 4. Memory Layout

### 4.1 Row-Major (C-order) Storage

Tensors are stored as **contiguous 1D memory arrays** in row-major (C-order) layout. The rightmost/innermost dimension varies fastest.

```
Tensor shape [D0, D1, D2]:
    Strides = [D1×D2, D2, 1]
    
    offset(i, j, k) = i × (D1×D2) + j × D2 + k × 1
```

**Example:** Shape `[4, 5, 6]`:
- Strides: `[30, 6, 1]`
- Element `[3, 1, 2]` → offset = `3×30 + 1×6 + 2 = 98`

### 4.2 Stack-Allocated Coordinate Buffers (Zero-Allocation Kernel)

The reduction kernel uses **fixed-size stack arrays** instead of heap-allocated `std::vector` for all coordinate computations in the hot loop:

```cpp
constexpr size_t MAX_DIMS = 16;   // supports up to 16-dimensional tensors

// Inside the kernel loop (no malloc/new):
int64_t out_coords_buf[MAX_DIMS];
int64_t slice_coords_buf[MAX_DIMS];
int64_t full_input_coords_buf[MAX_DIMS];
```

**Why this matters:**
- `std::vector` allocates on the heap every time → thousands of `malloc`/`free` calls per reduction
- Stack arrays are allocated instantly (just stack pointer adjustment) → near-zero overhead
- Tensors rarely exceed 8 dimensions in practice; 16 is a generous upper bound

### 4.3 GPU Memory Architecture

**Kernel launch configuration:**
```cpp
int threads_per_block = 256;         // 8 warps of 32 threads
int num_blocks = num_slices;         // 1 block per output element
size_t shared_mem_size = (256/32) * sizeof(AccT);  // 8 entries for warp leaders

reduce_kernel<<<num_blocks, threads_per_block, shared_mem_size, stream>>>();
```

**Shared memory:** Dynamically allocated per-block on-chip memory used for the block-level reduction phase (warp leaders write partial sums here).

**GPU coordinate arrays:** Fixed-size `int64_t[10]` arrays (register/local memory on GPU).

### 4.4 DeviceArray RAII Pattern

Tensor metadata (dims, strides, axes) must be copied to GPU device memory before kernel launch:
```cpp
class DeviceArray {
    int64_t* ptr;
    cudaStream_t stream_;
    
    DeviceArray(const std::vector<int64_t>& host_data, cudaStream_t stream) {
        cudaMallocAsync(&ptr, bytes, stream_);
        cudaMemcpyAsync(ptr, host_data.data(), bytes, cudaMemcpyHostToDevice, stream_);
    }
    
    ~DeviceArray() { cudaFreeAsync(ptr, stream_); }   // automatic cleanup
};
```
**Uses stream-ordered allocation (`cudaMallocAsync`)** for efficient async execution.

### 4.5 keepdim Impact on Shape

| Operation | `keepdim=false` | `keepdim=true` |
|-----------|:---:|:---:|
| `[3,4,5]` reduce axis 1 | `[3, 5]` | `[3, 1, 5]` |
| `[3,4,5]` reduce axis [0,2] | `[4]` | `[1, 4, 1]` |
| `[3,4,5]` reduce all | `[1]` (scalar) | `[1, 1, 1]` |

**Why `keepdim=true` matters:** Preserves tensor rank, enabling direct broadcasting with the original tensor. Critical for operations like `x - reduce_mean(x, axis=1, keepdim=true)` (centering).

---

## 5. Pseudocode / Program Logic

### 5.1 CPU Reduction Kernel (Value Reductions)

```
FUNCTION reduce_kernel_cpu(input, normalized_axes, output_shape):
    output = allocate_tensor(output_shape)
    reduced_dims = [input.shape[ax] for ax in normalized_axes]
    reduced_count = product(reduced_dims)
    
    #pragma omp parallel for    ← Each CPU thread handles one output element
    FOR output_index = 0 TO output.numel():
        out_coords = unravel_index_stack(output_index, output_shape)
        
        IF using_kahan_summation:
            kahan_sum = 0, kahan_c = 0
            FOR i = 0 TO reduced_count:
                input_value = lookup_input_value(out_coords, i, ...)
                y = input_value - kahan_c
                t = kahan_sum + y
                kahan_c = (t - kahan_sum) - y
                kahan_sum = t
            output[output_index] = kahan_sum
        ELSE:
            accumulator = op.identity()
            FOR i = 0 TO reduced_count:
                input_value = lookup_input_value(out_coords, i, ...)
                accumulator = op.reduce(accumulator, input_value)
            output[output_index] = accumulator
    
    RETURN output
```

### 5.2 GPU Reduction Kernel

```
KERNEL reduce_kernel_gpu(input_data, output_data, metadata...):
    ← Grid: 1 block per output element
    ← Block: 256 threads per block (8 warps)
    
    shared_memory AccT shared[8]     ← For warp-level partial sums
    
    FOR output_index = blockIdx.x (stride by gridDim.x):    ← Grid-stride loop
        out_coords = unravel(output_index, output_dims)
        accumulator = op.identity()
        
        // PHASE 1: Thread-level accumulation (striped)
        FOR i = threadIdx.x TO reduced_count STEP blockDim.x:
            input_value = lookup_input_value(out_coords, i, ...)
            accumulator = op.reduce(accumulator, input_value)
            // Thread 0 handles elements 0, 256, 512, ...
            // Thread 1 handles elements 1, 257, 513, ...
        
        // PHASE 2: Warp-level reduction (log₂ time)
        FOR offset = 16, 8, 4, 2, 1:
            other = __shfl_down_sync(0xFFFFFFFF, accumulator, offset)
            accumulator = op.reduce(accumulator, other)
        
        // PHASE 3: Block-level reduction
        IF (lane_id == 0):
            shared[warp_id] = accumulator
        __syncthreads()
        
        IF (warp_id == 0):
            accumulator = (threadIdx.x < num_warps) ? shared[lane_id] : op.identity()
            // Second warp reduction
            FOR offset = 16, 8, 4, 2, 1:
                other = __shfl_down_sync(0xFFFFFFFF, accumulator, offset)
                accumulator = op.reduce(accumulator, other)
        
        // PHASE 4: Write result
        IF (threadIdx.x == 0):
            output[output_index] = accumulator
```

### 5.3 GPU Variance Kernel (Two-Pass)

```
STEP 1: Compute mean tensor on GPU
    mean_tensor = dispatch_mean_gpu(input, axes, keepdim=true)
    // keepdim=true so mean has shape compatible for broadcasting

STEP 2: Launch variance kernel
KERNEL reduce_variance_kernel(input_data, mean_data, output_data, ...):
    FOR each output_index:
        mean_val = mean_data[map_output_to_mean_index(output_index)]
        
        // Thread-level accumulation of squared deviations
        sum_sq_dev = 0
        valid_count = 0     (for NaN-aware variant)
        
        FOR i = threadIdx.x TO reduced_count STEP blockDim.x:
            val = input_data[compute_index(...)]
            
            IF NaN-aware AND is_nan(val):
                SKIP
            ELSE:
                diff = val - mean_val
                sum_sq_dev += diff * diff
                valid_count++
        
        // Warp + Block reduction on sum_sq_dev (and valid_count if NaN-aware)
        ...
        
        IF threadIdx.x == 0:
            divisor = NaN-aware ? (valid_count - correction) : (reduced_count - correction)
            output[output_index] = sum_sq_dev / divisor
```

### 5.4 Autograd Backward Logic

```
CLASS SumBackward : public Node
    saved: input_shape_
    
    FUNCTION apply(grads):
        grad_output = grads[0]      // scalar or reduced-shape tensor
        grad_input = ones(input_shape_) * grad_output_value
        RETURN {grad_input}
        // Every input element contributed equally → each gets the full gradient

CLASS MeanBackward : public Node
    saved: input_shape_, numel_
    
    FUNCTION apply(grads):
        grad_output = grads[0]
        scale = grad_output_value / numel_
        grad_input = ones(input_shape_) * scale
        RETURN {grad_input}
        // Each element's contribution was 1/N → gradient is also 1/N
```

---

## 6. Research Material

### 6.1 Standard Algorithms Referenced

| Algorithm | Used In | Reference |
|-----------|---------|-----------|
| **Kahan Summation** | `SumOp` (CPU path) | Kahan, W. (1965). "Pracniques: Further Remarks on Reducing Truncation Errors." |
| **Bessel's Correction** | `reduce_var`, `reduce_std` | Unbiased estimation of population variance from sample data |
| **Warp Shuffle Reduction** | GPU `block_reduce()` | NVIDIA parallel reduction pattern using `__shfl_down_sync()` |
| **Two-Pass Variance** | `reduce_var`, `reduce_nanvar` | Numerically stable variance via separate mean + deviation passes |
| **Ravel/Unravel Index** | Coordinate mapping | Standard C-order (row-major) linearization and inverse |

### 6.2 Comparison with Industry Standards

| Feature | OwnTensor | PyTorch | NumPy |
|---------|-----------|---------|-------|
| Default variance correction | `correction=1` | `correction=1` | `ddof=0` |
| NaN-aware operations | ✅ `reduce_nanX` | ✅ `torch.nanX` | ✅ `np.nanX` |
| `keepdim` parameter | ✅ | ✅ | ✅ (`keepdims`) |
| GPU acceleration | ✅ CUDA kernels | ✅ CUDA kernels | ❌ CPU only |
| FP16 accumulation safety | ✅ auto-promote to float | ✅ auto-promote | N/A |
| Kahan summation | ✅ CPU SumOp | ❌ (uses pairwise) | ✅ (pairwise) |
| Bool type reductions | ✅ `all`/`any` | ✅ | ✅ |
| Complex type support | ✅ (sum, product, mean) | ✅ | ✅ |

### 6.3 CUDA Intrinsics Used

| Intrinsic | Purpose | Used In |
|-----------|---------|---------|
| `__shfl_down_sync(mask, val, delta)` | Warp-level data exchange without shared memory | `warp_reduce()` |
| `__hadd(a, b)` | Native FP16 addition | `gpu_add<__half>()` |
| `__hmul(a, b)` | Native FP16 multiplication | `gpu_mul<__half>()` |
| `__hlt(a, b)` / `__hgt(a, b)` | Native FP16 comparison | `gpu_lt<__half>()` / `gpu_gt<__half>()` |
| `__hisnan(val)` | Native FP16 NaN check | `gpu_isnan<__half>()` |
| `__float2half(f)` / `__half2float(h)` | FP32 ↔ FP16 conversion | `to_float()` / `from_float()` |
| `cudaMallocAsync` / `cudaFreeAsync` | Stream-ordered memory allocation | `DeviceArray` RAII class |

### 6.4 Parallelism Summary

| Component | CPU | GPU |
|-----------|-----|-----|
| **Outer Loop** (output elements) | `#pragma omp parallel for` — one thread per output element | One CUDA block per output element (grid-stride loop for overflow) |
| **Inner Loop** (reduced elements) | Serial `for` loop within each thread | 256 threads divide the work (thread-stride loop) |
| **Accumulation** | Single variable per thread (`accumulator += val`) | Warp shuffle + shared memory → O(log N) block reduction |
| **Synchronization** | None needed (independent output elements) | `__syncthreads()` for shared memory; `__shfl_down_sync` for warp |
