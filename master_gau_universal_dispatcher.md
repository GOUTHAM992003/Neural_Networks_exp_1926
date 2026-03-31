# Universal Reduction Architecture (`master_gau`)

## 1. Top-Level Decision Engine (`dispatch_reduction`)
The dispatcher calculates the optimal threading strategy and active thread count by evaluating the total workload (`numel`), the shape of the output (`num_slices`), and the size of the reduction dimension (`reduced_count`) against the `GRAIN_SIZE` threshold (32,768 elements).

```mermaid
graph TD
    A[Incoming Tensor Reduction] --> B{Check Total Workload}
    
    %% Case 1
    B -->|numel < GRAIN_SIZE<br>OR max_threads == 1| C[<b>CASE 1: Sequential</b><br>Too small for threading]
    C --> C_Strat[Strategy: ParallelSlices<br>actual_threads: 1]
    
    %% Case 2
    B -->|numel > GRAIN_SIZE| D{Check Output Shape}
    D -->|num_slices == 1<br>Full Reduction| E[<b>CASE 2: Split Reduction</b><br>Massive single output]
    E --> E_Strat[Strategy: SplitReduction<br>actual_threads: min(max, reduced_count/GRAIN_SIZE)]
    
    %% Case 3
    D -->|num_slices > 1<br>Partial Reduction| F[<b>CASE 3: Output vs Threads</b>]
    F --> |num_slices >= actual_threads| G[<b>Strategy: ParallelSlices</b><br>Enough independent outputs]
    F --> |num_slices < actual_threads| H[<b>Strategy: SplitReduction</b><br>Too few outputs for all threads]
    
    G --> I[actual_threads: min(max, reduced_count/GRAIN_SIZE)]
    H --> I
```

---

## 2. Kernel Routing & Operations
Once the strategy and actual thread count are locked in, the dispatcher routes the calculation to one of three specialized kernels based purely on the requested operation.

```mermaid
graph LR
    S[Dispatcher Strategy] --> K{Operation Type}
    
    K -->|argmax, argmin,<br>nanargmax, nanargmin| K1[<b>reduce_kernel_index</b>]
    K -->|sum, nansum<br>float / complex| K2[<b>cascade_sum_kernel</b>]
    K -->|max, min, prod,<br>int-sum, all, any| K3[<b>reduce_kernel</b>]
    
    K1 -.->|Output| Out1[Index Tracker<br>(ValueIndex)]
    K2 -.->|Output| Out2[4-Level Pairwise<br>Accuracy]
    K3 -.->|Output| Out3[Standard Math]
```

---

## 3. Kernel Execution Pipelines
Inside the selected kernel, the execution structurally separates into the chosen OpenMP threaded loop, and naturally drops into the optimal Memory Layout Path.

### Threading Strategies Applied:
*   **`ParallelSlices` Strategy:**
    *   `#pragma omp parallel for` executes over `num_slices`.
    *   **Behavior:** 1 Thread completely calculates 1 (or more) Output Slots. Zero locking overhead.
*   **`SplitReduction` Strategy:**
    *   `#pragma omp parallel` executes over `reduced_count`.
    *   **Behavior:** The reduction dimension is chopped into thread-local chunks. Threads calculate local accumulators and combine them at the end. Utilizes all CPU cores for even a single output slot.

### Low-Level Memory Layout Execution (Inner Loops):
No matter which threading strategy is used, the innermost loop resolves to one of three highly optimized memory strides:
1.  **`InnerContiguous` (Fastest):** Flat array linear traversal. Triggers unrolled AVX2/SIMD (e.g., `_mm256_max_ps`) in value kernels.
2.  **`OuterContiguous` (Great):** Strided column jumps mapping exactly to `pointer + i * row_stride`. Extremely low arithmetic overhead.
3.  **`Generic` (Fallback):** Complete N-dimensional coordinate reconstruction (carry-add loops) using pre-calculated `base_lin_idxs` and array strides.

```mermaid
flowchart TD
    Strategy((Thread Strategy)) --> Memory{Memory Layout Path}
    
    Memory -->|Contiguous Inner| IC[<b>InnerContiguous</b><br>Linear loop: ptr++<br><i>(Activates AVX2 SIMD)</i>]
    Memory -->|Contiguous Outer| OC[<b>OuterContiguous</b><br>Fast jump: ptr + i*stride<br><i>(Zero division overhead)</i>]
    Memory -->|Fragmented| G[<b>Generic</b><br>ND Coordinate Tracker<br><i>(Handles massive strided jumps)</i>]
```
