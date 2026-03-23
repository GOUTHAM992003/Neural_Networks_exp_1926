# The [binary_kernel_reduce](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Reduce.h#184-249) Pathways: Bypassing the SIMD Engine

You've asked a very precise architectural question about the relationship between the 5 kernel paths in PyTorch: *Do [binary_kernel_reduce](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Reduce.h#184-249) and [binary_kernel_reduce_lastdim](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Reduce.h#290-309) call [vectorized_inner_reduction](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Reduce.h#79-92), [vectorized_outer_reduction](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Reduce.h#93-114), or [basic_loop](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Loops.h#112-129)? And are they ONLY for Argmax?*

Here is the exact truth based on scanning the PyTorch CPU source code.

---

### 1. Do they call the SIMD engines or [basic_loop](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Loops.h#112-129)?

**No. They completely bypass them. They have their own entirely separate logic.**

If you look at the source code for [binary_kernel_reduce](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Reduce.h#184-249) in [Reduce.h](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Reduce.h) (around line 200), it never calls [vectorized_reduction](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Reduce.h#36-69) and it never calls [basic_loop](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Loops.h#112-129). 

Instead, it implements its own custom structure:
```cpp
    at::parallel_for(0, numel, internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
          auto& acc = buffer[at::get_thread_num()];
          acc = reduction_body(acc, begin, end);
    });
```
Inside that `reduction_body`, it uses `sub_iter.serial_for_each` to run a pure C++ scalar [for](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/eigen/unsupported/Eigen/src/Tensor/TensorReduction.h#594-596)-loop element by element.

**Why are they separate?**
*   `vectorized_inner/outer` are optimized for single-threaded or blocked hardware **SIMD vectorization**.
*   [basic_loop](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Loops.h#112-129) is optimized for **sequential scalar cleanup** (the leftovers of the SIMD vectors).
*   [binary_kernel_reduce](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Reduce.h#184-249) is optimized for **multi-threaded software splitting**. Because it has to divide the memory across 10-20 CPU cores simultaneously, it needs a completely different iterator wrapper (`at::parallel_for`) than what the SIMD engine uses. It abandons hardware vectorization entirely and instead maximizes software multi-threading across cores.

---

### 2. Are they ONLY for Argmax / Argmin?

**No! That is a common misconception.** While Argmax and Argmin are the most famous users of this path (because they carry an index which breaks generic SIMD logic), they are not the only ones.

I just ran a deep `grep_search` across [aten/src/ATen/native/cpu/ReduceOpsKernel.cpp](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp) for [binary_kernel_reduce](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Reduce.h#184-249). Here are the other operations that are forced down this specialized path:

1.  **Lp Norms (`NormZeroOps`, `NormOneOps`, `NormTwoOps`):** Calculating the mathematical norm of a tensor often requires accumulating states in ways that don't neatly fit into standard fast-math SIMD templates (especially L0 and L2 norms which have square roots and conditional limits).
2.  **Absolute Extrema (`AbsMaxOps`, `AbsMinOps`):** These require taking the absolute value of the memory *before* comparing to the maximum. While technically vectorizable, PyTorch often routes these through the multi-threaded scalar fallback because the custom composite logic ([max(acc, abs(current))](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensor_centric_tensorlib/include/dtype/Types.h#476-480)) isn't directly bound to their standard `vec_func_t` engine map.
3.  **Custom / Complex Reductions:** If a PyTorch developer writes a custom reduction operation and *only* provides a scalar C++ functor (they do not provide a `vec_func_t` SIMD equivalent), the engine is forced to route it through [binary_kernel_reduce](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Reduce.h#184-249) to at least get multi-threaded speed.

### Conclusion: The Separation of Concerns

*   If an operation is purely associative, computationally simple, and native to CPU hardware instructions (e.g., [Sum](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/tensorflow/tensorflow/core/kernels/reduction_gpu_kernels.cu.h#46-52), [Product](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/master_gau/include/ops/helpers/ReductionOps.h#337-363), [Max](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/tensorflow/tensorflow/core/kernels/reduction_gpu_kernels.cu.h#1002-1010)), it gets sent to the **Level 1/2/3 SIMD Dispatcher** ([binary_kernel_reduce_vec](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Reduce.h#250-281)) to be obliterated by YMM registers.
*   If an operation requires tracking complex composite state (like an Index, or a specialized Norm limit), it is sent to **[binary_kernel_reduce](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Reduce.h#184-249) or `..._lastdim`**. These paths ignore SIMD instructions entirely, bypass [basic_loop](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Loops.h#112-129), and instead carve the tensor up for raw multi-core Thread processing using heavily customized C++ scalar loops.
