# PyTorch CPU Reduction Architecture: The 5 Kernel Paths

After [reorder_dimensions](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/TensorIterator.cpp#232-309) and [coalesce_dimensions](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/TensorIterator.cpp#638-690) squashes the N-dimensional tensor into a mostly 2D or 1D shape, PyTorch passes the workload to [aten/src/ATen/native/cpu/Reduce.h](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Reduce.h). 

Here, PyTorch selects from **5 specific sub-variants (Kernel Paths)** to execute the reduction. The choice relies heavily on whether the inner dimensions are contiguous and whether the mathematical operation supports SIMD vectorization.

---

### Path 1: [vectorized_inner_reduction](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Reduce.h#79-92) (Horizontal SIMD)

**When it triggers:** 
The reduction is happening strictly on the innermost dimension (`dim 0`), and the memory for that dimension is perfectly contiguous (stride = `sizeof(T)`). 

**How it works:**
1. **Blocked Unrolling:** It processes the inner loop in massive chunks. It loads 4 full SIMD vectors at a time (e.g., on AVX2, this is 4 vectors $\times$ 8 floats = 32 floats per iteration).
2. **Independent Accumulators:** It accumulates into 4 separate vector registers to avoid CPU instruction dependency bottlenecks (breaking the floating-point latency chain).
3. **Horizontal Merge:** Once the block finishes, it mathematically folds the 4 vectors into 1 vector, and then does a "horizontal addition" across the lanes of that final vector to produce a single scalar value.
4. **Remainder Fallback:** If the reduction size isn't perfectly divisible by 32, it passes the remainder to the scalar [basic_loop](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Loops.h#112-129).

---

### Path 2: [vectorized_outer_reduction](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Reduce.h#93-114) (Vertical SIMD)

**When it triggers:**
The reduction is happening on the outer dimension (`dim 1`), but the innermost dimension (`dim 0`) is being preserved and is perfectly contiguous. (e.g., "column-reduce" a row-major matrix).

**How it works:**
1. **Vertical Accumulation:** Because you are preserving the inner columns, you can't flatten horizontally. Instead, it loads a SIMD vector from Row 0, another SIMD vector from Row 1, and applies the vector op (e.g., `_mm256_add_ps`). 
2. **Direct Store:** The magical part of Vertical SIMD is that **no horizontal merge is required**. After adding all the rows together vertically, every single "lane" in the resulting SIMD vector maps directly to an element in the preserved output tensor. It just writes the whole vector block directly to memory.
3. This completely maxes out memory bandwidth because it never has to wait for scalar extraction.

---

### Path 3: [basic_loop](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Loops.h#112-129) (Scalar Fallback)

**When it triggers:**
When the memory strides are fragmented (e.g., strided slices like `tensor[::2, ::2]`), making contiguous vector loading physically impossible, **OR** when the provided operation does not have a vectorized kernel definition (`vec_func_t` is missing).

**How it works:**
A brute-force, standard C++ [for](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/eigen/unsupported/Eigen/src/Tensor/TensorReduction.h#528-599) loop. It manually advances memory pointers by their exact byte strides (`data += stride`) and applies the scalar reduction operation element-by-element. Safe, reliable, but completely misses vector parallelism.

---

### Path 4: [binary_kernel_reduce](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Reduce.h#184-249) (Multi-Threaded Scalar / Index Tracking)

**When it triggers:**
Used for mathematically complex operations that carry "State" and cannot be cleanly mapped to standard SIMD vector registers—most notably **Argmax and Argmin** (which must track both the maximum value *and* the index of that value simultaneously).

**How it works:**
It abandons hardware vectorization entirely and instead maximizes software multi-threading via `at::parallel_for`.
1. It allocates a buffer of scalar accumulators exactly equal to the number of available CPU threads.
2. It divides the 1D/2D iteration space among the threads.
3. Each thread runs a completely scalar [basic_loop](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Loops.h#112-129) (using [serial_for_each](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/TensorIteratorInternal.h#39-70)) to find the local argmax/argmin in its chunk.
4. Once all threads finish, the main thread runs a tiny scalar loop to combine the thread-local accumulators into the final global answer.

---

### Path 5: [binary_kernel_reduce_lastdim](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Reduce.h#290-309) (Last-Dim Specialization)

**When it triggers:**
An optimization explicitly built for `argmax/argmin` operations when the user guarantees they are reducing *exclusively* along the absolute innermost, contiguous dimension.

**How it works:**
Normally, [binary_kernel_reduce](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/pytorch_source/aten/src/ATen/native/cpu/Reduce.h#184-249) blindly splits elements across threads (which might slice a contiguous row in half, complicating index math). 
This optimized variant uses `sub_iter.narrow(0, 0, 1)` to conceptually detach the contiguous innermost dimension. It then parallelizes the threads *exclusively over the outer dimensions*. 
**Result:** Each thread is guaranteed to receive entire, unbroken contiguous rows to process. The inner loop becomes a pure, uninterrupted linear scan, which is significantly faster for index-tracking state machines.
