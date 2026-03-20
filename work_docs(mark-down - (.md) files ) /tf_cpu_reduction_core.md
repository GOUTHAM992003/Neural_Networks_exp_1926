# TensorFlow CPU Reduction Operations — Architecture analysis

This document strictly breaks down how TensorFlow executes Reduction Operations (like Sum, Max, Min) on the **CPU**.

There are exactly 3 layers to how TensorFlow routes a user's `tf.reduce_sum(tensor, axes)` call down to the CPU hardware.

---

## Layer 1: The Entry Point ([reduction_ops_sum.cc](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensorflow/tensorflow/core/kernels/reduction_ops_sum.cc))

Each mathematical operation has its own C++ file (e.g., [reduction_ops_sum.cc](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensorflow/tensorflow/core/kernels/reduction_ops_sum.cc), [reduction_ops_max.cc](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensorflow/tensorflow/core/kernels/reduction_ops_max.cc)). 

In this file, TensorFlow registers the kernel using C++ Macros. Here is what [Sum](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensorflow/tensorflow/core/kernels/reduction_gpu_kernels.cu.h#46-52) looks like:

```cpp
#define REGISTER_CPU_KERNELS(type)                                             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Sum")                                                              \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<int32>("Tidx"),                                      \
      ReductionOp<CPUDevice, type, int32, Eigen::internal::SumReducer<type>>);
```

**Key Takeaways:**
1. It registers the operation string `"Sum"` to run on `DEVICE_CPU`.
2. It maps the operation to a central template class called **[ReductionOp](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensorflow/tensorflow/core/kernels/reduction_ops_common.h#133-140)**.
3. It passes `Eigen::internal::SumReducer<type>` as the mathematical functor. This functor will be passed all the way down.

---

## Layer 2: The Routing Logic ([reduction_ops_common.h](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensorflow/tensorflow/core/kernels/reduction_ops_common.h))

No matter what operation (Sum, Max, Min) is called, they all funnel into `tensorflow::ReductionOp::Compute(OpKernelContext* ctx)`.

This [Compute()](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensorflow/tensorflow/core/kernels/reduction_ops_common.h#141-247) function executes the following strict pipeline:

### Step 1: `ReductionHelper::Simplify`
Just like the GPU path, the CPU path starts by calling `helper.Simplify(data, axes)`.
This collapses adjacent physical dimensions that share a reduction status. 
*Result:* The shape is converted to at most 3D (Rank 1, 2, or 3). Any higher rank is physically eliminated.

### Step 2: The 6-Way Dispatch
After simplifying, the code checks [ndims()](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensorflow/tensorflow/core/kernels/reduction_ops_common.h#88-90) (1, 2, or 3) and whether the first dimension is being reduced (`reduce_first_axis_`). Based on this, it routes the data differently:

```cpp
// From reduction_ops_common.h (Lines 197-217)

if ((helper.ndims() == 1) && helper.reduce_first_axis()) {
    // 1D shape, reduce the only axis (Full Reduction to Scalar)
    Functor::Reduce(ctx, helper.out(&tmp_out), helper.in(data), constants.kZero, reducer);

} else if ((helper.ndims() == 2) && helper.reduce_first_axis()) {
    // 2D shape, reduce rows (Column-wise reduction)
    Functor::Reduce(ctx, helper.out(&tmp_out), helper.in(data), constants.kZero, reducer);

} else if ((helper.ndims() == 2) && !helper.reduce_first_axis()) {
    // 2D shape, reduce columns (Row-wise reduction)
    Functor::Reduce(ctx, helper.out(&tmp_out), helper.in(data), constants.kOne, reducer);

} else if ((helper.ndims() == 3) && helper.reduce_first_axis()) {
    // 3D shape, reduce dim 0 and 2 
    Functor::Reduce(ctx, helper.out(&tmp_out), helper.in(data), constants.kZeroTwo, reducer);

} else if ((helper.ndims() == 3) && !helper.reduce_first_axis()) {
    // 3D shape, reduce dim 1
    Functor::Reduce(ctx, helper.out(&tmp_out), helper.in(data), constants.kOne, reducer);

} else {
    // Fallback: Transpose all reduced dimensions to the end and treat as 2D Row reduction
}
```

The third argument passed to `Functor::Reduce` is the axis index. `constants.kZero` means "axis 0", `constants.kOne` means "axis 1", and `constants.kZeroTwo` means "axes 0 and 2".

---

## Layer 3: Mathematical Execution ([reduction_ops.h](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensorflow/tensorflow/core/kernels/reduction_ops.h))

All 6 of the branches above end up calling [ReduceEigenImpl(...)](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensorflow/tensorflow/core/kernels/reduction_ops.h#51-59).

Here is the exact implementation of how TensorFlow performs CPU reductions:

```cpp
// From reduction_ops.h (Line 51-58)
template <typename Device, typename OUT_T, typename IN_T,
          typename ReductionAxes, typename Reducer>
struct ReduceEigenImpl {
  void operator()(const Device& d, OUT_T out, IN_T in,
                  const ReductionAxes& reduction_axes, const Reducer& reducer) {
    
    // THE CORE EXECUTION:
    out.device(d) = in.reduce(reduction_axes, reducer);
    
  }
};
```

### The Plot Twist: TensorFlow Doesn't Write Its Own CPU Loops
TensorFlow does **NOT** manually implement CPU [for](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/TensorIterator.h#439-440) loops, AVX vectorization, or OpenMP parallelization inside its own source code for reductions.

The object [in](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/Tensor-Implementations_gau/PyTorch-MLP-Text/libtorch/bin) is an `Eigen::Tensor`. The line `in.reduce(...)` hands complete control over to the **Eigen C++ Library**. 

Eigen is an external third-party matrix math library heavily based on C++ template metaprogramming. When `in.reduce(...)` is compiled, the Eigen library detects compiling flags (like `-mavx2` or `-mavx512`) and automatically generates heavily optimized, multi-threaded SIMD assembly loops specifically for those 1D, 2D, or 3D cases that TensorFlow calculated.

---

## Summary: TensorFlow CPU vs. Our Approach

| Architecture Step | TensorFlow's CPU Approach | Our Library's Current Approach (`master_gau`) |
|-------------------|---------------------------|-----------------------------------------------|
| **1. Shape Parsing** | Collapses to ≤ 3D using [Simplify()](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensorflow/tensorflow/core/kernels/reduction_ops_common.cc#82-159) | Keeps original rank (up to N-dim) |
| **2. Index Math** | Eliminated (Handled natively by 1D/2D/3D shapes) | Calculated iteratively per element using [unravel](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/master_gau_tensorflow/include/ops/helpers/ReductionUtils.h#79-86) / [ravel](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/master_gau_tensorflow/include/ops/helpers/ReductionUtils.h#87-94) |
| **3. Dispatching** | Routes directly to 6 specialized layouts | Routes to 1 generic execution loop |
| **4. Vectorization** | Delegated fully to **Eigen library** at compile time | None (Standard scalar C++ loops) |

To implement the TensorFlow CPU style in our library, we must adopt Steps 1, 2, and 3. Since we do not use Eigen, for Step 4 we will write our own explicit C++ loops for those 6 specialized layouts resulting from the [Simplify()](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensorflow/tensorflow/core/kernels/reduction_ops_common.cc#82-159) step.
