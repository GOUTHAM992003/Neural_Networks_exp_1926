# 2D Tensor Reduction Architecture for All 22 Operations

If you have a 2D tensor (`[Rows, Columns]`), the architecture bifurcates in 3 distinct physically different ways depending on which dimension you reduce:
1.  **Full Reduction** (`dim=(0, 1)`)
2.  **Partial Reduce Columns** (`dim=1`): Reducing the contiguous innermost memory.
3.  **Partial Reduce Rows** (`dim=0`): Reducing the non-contiguous outermost memory, keeping columns intact.

Here is exactly how PyTorch and Eigen route all 22 mathematical operations across these 3 scenarios.

---

## Category A: The Fast-Math Core (6 Ops)
**Operations:** [sum](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#1254-1257), `product`, [max](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#2238-2241), [min](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#878-884), [all](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#2200-2203) (AND), [any](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#2194-2197) (OR)

### Scenario 1: Full Reduction (`dim=(0, 1)`)
*   **PyTorch:** The [coalesce_dimensions()](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/TensorIterator.cpp#638-690) function immediately detects that both Rows and Columns are adjoining memory blocks. It physically flattens the tensor into a 1D `[Rows * Columns]` array and routes perfectly to [vectorized_inner_reduction](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Reduce.h#79-92).
*   **Eigen:** Routes to `FullReducer<ThreadPoolDevice>`.

### Scenario 2: Reduce Columns (`dim=1` - Innermost)
Because calculating across the internal [Columns](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/tensorflow/tensorflow/core/kernels/reduction_gpu_kernels.cu.h#271-338) is physically contiguous in memory, it mimics a 1D array per Row.
*   **PyTorch:** [vectorized_inner_reduction](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Reduce.h#79-92).
    *   **Loop:** `for (row = 0; row < Rows; row++) { RUN_HORIZONTAL_SIMD_ON_ROW_COLS(); }`
*   **Eigen:** `InnerMostDimReducer<true, ...>`.
    *   **Loop:** `for (row = 0; row < Rows; row++) { RUN_PAIRWISE_SIMD_ON_ROW_COLS(); }`

### Scenario 3: Reduce Rows (`dim=0` - Outermost)
Because we are preserving the columns but shrinking the rows, we cannot read horizontally safely. This forces the incredible **Vertical SIMD** architectures.
*   **PyTorch:** [vectorized_outer_reduction](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Reduce.h#93-114).
    *   **Loop:** Reads chunks of columns, but accumulates by moving pointers exactly 1 Row-Stride down into memory.
    *   `acc[0] = vop(acc[0], ptr(row=1, cols0to8))`
    *   `acc[0] = vop(acc[0], ptr(row=2, cols0to8))` ...
    *   Writes directly to RAM (`reduce=false`).
*   **Eigen:** `InnerMostDimPreserver<0, true>`.
    *   **Loop:** Exact same vertical logic. Uses compile-time `<0>` templates to read packets jumping by the Row Stride.

---

## Category B: The NaN-Safe Ops (4 Ops)
**Operations:** [nansum](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#1285-1290), `nanproduct`, `nanmax`, `nanmin`

These perfectly shadow Category A in all 3 scenarios.
*   **Full Reduction:** Flattens to 1D SIMD.
*   **Reduce Cols:** Inner Horizontal SIMD.
*   **Reduce Rows:** Outer Vertical SIMD.

**The catch:** Inside every single SIMD macro, the CPU executes [isnan(val) ? blend_mask : val](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#783-787) alongside the math port additions.

---

## Category C: The Index Trackers (4 Ops)
**Operations:** [argmax](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#1753-1787), [argmin](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#1753-1787), `nanargmax`, `nanargmin`
*(Because an index is tied to a specific tensor position, SIMD is abandoned. Software thread-splitting takes over).*

### Scenario 1: Full Reduction (`dim=(0, 1)`)
*   **PyTorch:** Flattens to 1D, routes to [binary_kernel_reduce_lastdim](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Reduce.h#290-309) (Software Parallel array sweeping).
*   **Eigen:** [FullReducer](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/eigen/unsupported/Eigen/src/Tensor/TensorReduction.h#347-358) -> `<false, false>` Struct tracking.

### Scenario 2: Reduce Columns (`dim=1` - Innermost)
*   **PyTorch:** Because this is perfectly contiguous, it hits the optimized [binary_kernel_reduce_lastdim](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Reduce.h#290-309) where Thread chunks don't have to worry about broken row pointers.
*   **Eigen:** `InnerMostDimReducer<false, false>`.

### Scenario 3: Reduce Rows (`dim=0` - Outermost)
*   **PyTorch:** Because rows are not contiguous, PyTorch drops all optimizations and defaults back to the slowest, safest multi-threader: the standard [binary_kernel_reduce](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Reduce.h#184-249). This safely jumps by memory strides over the [Columns](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/tensorflow/tensorflow/core/kernels/reduction_gpu_kernels.cu.h#271-338) gap for every iteration.
*   **Eigen:** [InnerMostDimPreserver](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/eigen/unsupported/Eigen/src/Tensor/TensorReduction.h#280-288) drops packet execution completely and falls back to a nested scalar loop to carefully maintain the index struct.

---

## Category D: The Simple Composites (2 Ops)
**Operations:** [mean](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#1435-1438), [nanmean](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#1477-1490)

These perfectly shadow Category A and Category B across all 3 scenarios.
The only difference happens at the very finish line:
1.  Calculate `sum_val` perfectly using Scenario 1/2/3 SIMD fast-math engines.
2.  Calculate scalar `return sum_val / N`.

---

## Category E: Welford Stateful Core (6 Ops)
**Operations:** [var](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#2070-2076), [std](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#2084-2088), [var_mean](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#2029-2034), [std_mean](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#2023-2028), `nanvar`, `nanstd`

### Scenarios 1, 2, and 3
Because Welford requires moving 3 distinct variables ([mean](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#1435-1438), `m2`, [count](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensor_centric_tensorlib/include/ops/helpers/ReductionUtils.h#72-78)) cleanly per individual element passed, it is mathematically impossible to cleanly vector-unroll this or execute vertical SIMD safely inside a typical PyTorch hardware loop without destroying the CPU L1 cache with registers.

*   **PyTorch:** Irrespective of whether you are reducing Full, Rows, or Columns, PyTorch forces ALL of these down [binary_kernel_reduce](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Reduce.h#184-249). It does not use [Inner](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/eigen/unsupported/Eigen/src/Tensor/TensorReduction.h#414-424), [Outer](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/eigen/unsupported/Eigen/src/Tensor/TensorReduction.h#426-436), or [_lastdim](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Reduce.h#282-289). It just feeds the raw strides of the tensor straight into standard outer dimension multi-threading [for](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/eigen/unsupported/Eigen/src/Tensor/TensorReduction.h#528-599) loops.
*   **Eigen:** Splits everything into Two Sweeps. 
    *   **Sweep 1:** Runs exactly like Category A (Scenario 1/2/3 dependent) to fetch the purely perfectly vectorized Mean.
    *   **Sweep 2:** Creates an invisible mapping of [tensor(a) - mean](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/TensorIterator.cpp#87-91) and runs Category A *again* to grab the squared sums perfectly vectorized. 

By splitting it into two fast operations instead of one slow Stateful operation, Eigen can reuse the massive power of Vertical SIMD for row reductions, where PyTorch falls back to slow threads!
