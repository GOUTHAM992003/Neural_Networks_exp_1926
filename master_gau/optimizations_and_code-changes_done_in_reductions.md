# Optimizations and Code-Changes Done in Reductions

## 1. Software / Code Optimizations and Fixes

### 1.1 GPU Caching Allocator Integration
**(File: `src/UnaryOps/cuda/ReductionImpl.cu`)**
Integrated our custom GPU Caching Allocator directly into the reduction kernels. By utilizing pre-cached memory blocks for intermediate operations instead of waiting for standard OS-level allocations via `cudaMalloc`, this drastically reduced kernel launch latency and overall allocation overhead ---> (In PackedMetaData struct , we used caching allocator for intermediate operations) .

### 1.2 Packed Metadata Transmission
**(File: `src/UnaryOps/cuda/ReductionImpl.cu`)**
Replaced individual `Device_Array` objects with a unified `PackedMetaData` object. This reduces the number of kernel arguments and memory transfers by sending all necessary shape, stride, and reduction axes metadata to the GPU in a single packed structure at once rather than one-by-oneover the PCIe bus in a single unified jump.


### 1.3 NaN Fix for Index-Returning Reductions (ArgMin / ArgMax) 
**(File: `include/ops/helpers/ReductionOps.h`)**
Addressed an unpredictable behavior where comparing two NaNs returned an undefined/arbitrary index. We explicitly added a check to correctly halt and propagate the first NaN's index deterministically:
```cpp
if (safe_isnan(a.value) && safe_isnan(b.value)) {
    return (a.index < b.index) ? a : b;
}
```
This ensures that if multiple NaNs are encountered, the function correctly returns the index of the first NaN it saw, maintaining consistent and predictable behavior.

### 1.4 Removed Unreachable Dead Code
Commented out dead code in `ReductionImpl.h` that was previously checking for an impossible state:
**(File: `include/ops/helpers/ReductionImpl.h`, Lines 137-145)**
Removed mathematically redundant runtime error checks. Code validating `reduced_count == 0 && input.numel() > 0` was commented out because `normalize_axes()` handles `numel() == 0` validation prior to this step, making this unreachable dead code.
```cpp
// if (reduced_count == 0 && input.numel() > 0) {
//     throw std::runtime_error("Reduction error: reduced count is zero but input has " + 
//                             std::to_string(input.numel()) + " elements.");
// }
```
This state is unreachable because the `normalize_axes` function already handles and validates the reduced axes from the input shape, correctly bypassing cases where `input.numel() == 0` (e.g., shape `[0, 3]`).

### 1.5 Bitmap Array Lookup inside OpenMP Threads [O(1) Hot-loop Lookup]
**(File: `include/ops/helpers/ReductionImpl.h`)**
Replaced 22 slow $O(N)$ execution path scans via `std::find()` scattered across our $O(num\_slices \times reduced\_count)$ hot-loops with an instantaneous $O(1)$ boolean array read (`reduced_bitmap[dim]`). The bitmap is preemptively built once just before the `#pragma omp parallel for` threading happens. This significantly accelerates CPU execution speeds for heavily chunked tensor operations where loop iterations hit millions.

simply,Replaced the O(N) `std::find(normalized_axes.begin(), ...)` calls inside the innermost parallel CPU loops with an O(1) boolean bitmap array (`bool reduced_bitmap[MAX_DIMS] = {false};`). The bitmap is initialized once before the OpenMP loop, completely removing unnecessary array traversals during every element's coordinate mapping.
Affected `is_reduced` `std::find` replacements natively lie at:
- `reduce_kernel()` Lines: **168, 220, 265, 339**
- `dispatch_mean_kernel()` Lines: **604, 631, 683, 712**
- `dispatch_variance_kernel()` Lines: **945, 970, 1011**

## 2. Hardware Optimizations
*(Reserved for hardware-specific optimizations)*
