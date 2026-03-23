# 1D Full Reduction Architecture for All 22 Operations

If you have a 1D tensor (an array `[Columns]`), any reduction you run is by definition a **Full Reduction** (since there is only 1 dimension to shrink).

To understand how PyTorch and Eigen/TensorFlow route these operations, we must group the 22 mathematical operations into 5 Core Categories. They literally take different underground paths.

---

## Category A: The Fast-Math Core (6 Ops)
**Operations:** [sum](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#1254-1257), `product`, [max](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#839-845), [min](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#878-884), [all](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#2200-2203) (AND), [any](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#2194-2197) (OR)

Because these don't track state or indexes, they take the absolute fastest SIMD paths available in the libraries.

### **PyTorch Route**
```ascii
[API: sum/max/...] 
   └── binary_kernel_reduce_vec() (Dispatcher)
       └── vectorized_inner_reduction() (Wrapper)
           └── vectorized_reduction(reduce=true) (Hardware Engine)
```
**Underground Loop (`vec[4]` Horizontal SIMD):**
```cpp
// 1. Unrolled Vector block accumulation
for(i = 0; i < N; i += 32) {
    acc[0] = vop_add(acc[0], in[i+0_to_8]);
    acc[1] = vop_add(acc[1], in[i+8_to_16]); ...
}
// 2. Horizontal Merge
out = horizontal_crush(acc[0] + acc[1] + acc[2] + acc[3]);
```

### **Eigen / TensorFlow Route**
```ascii
[API: sum/max/...]
   └── FullReducer<ThreadPoolDevice> (Distributes blocks to threads)
       └── InnerMostDimReducer<true, true> (Pairwise Float) OR <true, false> (Linear int)
```
**Underground Loop (`Packet4f`):**
```cpp
for(i = 0; i < BlockSize; i += 16) {
    paccum0 += packet(i+0_to_4); ...
} // Merged via Pairwise Recursive stack for floats
```

---

## Category B: The NaN-Safe Ops (4 Ops)
**Operations:** [nansum](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#1285-1290), `nanproduct`, `nanmax`, `nanmin`

These follow the **exact same routing paths** as Category A. The only difference is the exact assembly instruction used inside the underground `vop`.

### **PyTorch & Eigen modification**
Before the accumulator instruction executes, the SIMD register loads the array chunk, runs a hardware [isnan()](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#783-787) check mask, and blends mathematically inert values into the vector lanes where NaNs exist.
*   **nansum:** Blends `0.0` over NaNs.
*   **nanproduct:** Blends `1.0` over NaNs.
*   **nanmax:** Blends `-Infinity` over NaNs.
*   **nanmin:** Blends `Infinity` over NaNs.

*No routing is changed. The loop looks perfectly identical to Category A, but the `vop_add` macro is slightly heavier.*

---

## Category C: The Index Trackers (4 Ops)
**Operations:** [argmax](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#1753-1787), [argmin](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#1753-1787), `nanargmax`, `nanargmin`

These operations carry an [index](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/eigen/unsupported/Eigen/src/Tensor/TensorIndexList.h#18-27). This completely fractures SIMD vectorization. They must be routed out of the standard pipeline.

### **PyTorch Route**
```ascii
[API: argmax]
   └── binary_kernel_reduce_lastdim() (Skips vec[] engines completely)
       └── at::parallel_for() (Splits 1D array across cores)
```
**Underground Loop (Software Multithreaded Scalar):**
```cpp
// Thread 0 processes half, Thread 1 processes half
for (i = local_start; i < local_end; i++) {
    if (in[i] > best_val) { // Branch check (or isnan check for nan_argmax)
         best_val = in[i];
         best_idx = i;
    }
}
```

### **Eigen / TensorFlow Route**
```ascii
[API: argmax]
   └── FullReducer<ThreadPoolDevice>
       └── InnerMostDimReducer<false, false> (Forces Vector/Tree flags to FALSE)
```
**Underground Loop (C++ Struct tracking):**
```cpp
struct {float value; int64_t index} acc;
for (i = local_start; i < local_end; i++) {
    if (in[i] > acc.value) { acc = {in[i], i}; }
}
```

---

## Category D: The Simple Composites (2 Ops)
**Operations:** [mean](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#1439-1442), [nanmean](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#1477-1490)

These do not have standalone engine loops. They piggyback off Category A and B.

### **PyTorch Route**
PyTorch just calls [sum](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#1254-1257) (or [nansum](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#1285-1290)) routing down through [vectorized_reduction](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Reduce.h#36-69). After the `acc[0]` registers perfectly crunch out a scalar sum, the top-level API literally just executes: `return _sum_out / N;`.

### **Eigen / TensorFlow Route**
Eigen uses a [MeanReducer](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/tensorflow/tensorflow/core/kernels/reduction_ops.h#35-39) struct inside the normal `InnerMostDimReducer<true, ...>`. It actually tracks the [sum](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#1254-1257) and the [count](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensor_centric_tensorlib/include/ops/helpers/ReductionUtils.h#72-78) together natively.

---

## Category E: Welford Stateful Core (6 Ops)
**Operations:** [var](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#2070-2076), [std](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#2084-2088), [var_mean](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#2046-2054), [std_mean](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/ReduceOps.cpp#2015-2022), `nanvar`, `nanstd`

Calculating Variance requires maintaining three running variables: `[count, mean, M2]`. SIMD registers are incredibly bad at tracking triple-state mathematical pipelines simultaneously without register spilling.

### **PyTorch Route**
```ascii
[API: var]
   └── binary_kernel_reduce() (The software threading fallback)
       └── WelfordOps Functor
```
**Underground Loop (Welford's Algorithm):**
Instead of adding numbers, it runs complex math sequentially.
```cpp
WelfordState acc = {0.0, 0.0, 0.0}; // mean, m2, count
for (i = 0; i < N; i++) {
   acc.count += 1;
   float delta = in[i] - acc.mean;
   acc.mean += delta / acc.count;
   float delta2 = in[i] - acc.mean;
   acc.m2 += delta * delta2;
}
```

### **Eigen / TensorFlow Route**
Rather than writing a massive Welford fallback loop, Eigen often breaks [Var](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensor_centric_tensorlib/include/ops/helpers/ReductionOps.h#464-466) into two **sequential Category A passes**.
1. It runs a full [InnerMostDimReducer](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/eigen/unsupported/Eigen/src/Tensor/TensorReduction.h#159-175) to get the [Mean](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/tensorflow/tensorflow/core/kernels/reduction_ops.h#35-39).
2. It subtracts the mean from the tensor in memory dynamically, and runs a second full [InnerMostDimReducer](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/eigen/unsupported/Eigen/src/Tensor/TensorReduction.h#159-175) (Sum) on the squared differences.
This keeps it inside the fast hardware vectorization templates by splitting the math into two linear sweeps.
