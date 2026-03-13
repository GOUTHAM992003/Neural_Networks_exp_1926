# GPU Kernel Profiling & Optimization Notes

> Findings from `nsys` and `ncu` profiling of our custom reduction kernels vs TensorFlow's implementations.

---

## Optimization Targets (Priority Order)

### 1. ✅ Caching CUDA Allocator (In Progress)
**Problem:** Every reduction call does 5× `cudaMallocAsync` + 5× `cudaFreeAsync` for metadata arrays (`d_input_dims`, `d_input_strides`, `d_output_dims`, `d_normalized_axes`, `d_reduced_dims`).  
**Impact:** Each malloc/free pair costs ~5-10μs. For 5 arrays per reduction, that's 50-100μs of pure overhead.  
**Solution:** `CachingCudaAllocator` — reuse previously freed GPU memory from a pool instead of asking the driver every time.  
**Status:** Integrated into `DeviceArray` in `ReductionImplGPU.cu`.

---

### 2. 🔲 Combine Metadata into Single Transfer
**Problem:** 5 separate `cudaMemcpyAsync` calls to copy dims, strides, axes, etc.  
**Solution:** Pack all metadata into one contiguous host buffer, do ONE `cudaMemcpy`, and use pointer offsets on the GPU side.  
**Expected savings:** ~4 fewer API calls per reduction.

---

### 3. 🔲 Remove Debug `abi::__cxa_demangle` Call
**Problem:** Lines 142-146 in `ReductionImplGPU.cu` call `abi::__cxa_demangle(typeid(OpType<T>).name(), ...)` on every reduction. This is pure CPU string processing that wastes time in the hot path.  
**Solution:** Remove or guard behind `#ifndef NDEBUG`.

---

### 4. 🔲 CPU Synchronization Strategy (`cudaDeviceScheduleBlockingSync`)
**Problem:** Default CUDA behavior uses spin-waiting (`poll`) — the CPU burns 100% on one core just checking "is GPU done yet?"  
**Solution:** Call `cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)` at library init. This makes the CPU sleep (`sem_timedwait`) and wake only when GPU signals completion.  
**Trade-off:** ~1-5μs extra latency per sync, but near-zero CPU usage during GPU work.  
**When to use:** Production/serving mode. Keep spin-wait for low-latency benchmarking.  
**Implementation:** Add a config flag like `OwnTensor::set_low_cpu_mode(true)`.

---

### 5. 🔲 Lazy Function Loading Overhead
**Problem:** First launch of each unique kernel template (`reduce_kernel<SumOp>`, `reduce_kernel<ProductOp>`, etc.) triggers `cuLibraryLoadData` + `Lazy Function Loading`, costing ~20-90ms.  
**Solution:** Add a warmup step at library initialization that launches each kernel template once with dummy data.  
**Note:** This is a one-time cost per process, not per-call.

---

## Profiling Observations

### Thread Layout (6 threads total)
| Thread | Role |
|--------|------|
| `snippet_runner` (main) | Runs our C++ code, fires CUDA API calls via `ioctl` |
| `cuda-EvtHandlr` | CUDA background thread — polls GPU for completion events |
| TBB worker threads (3-4) | Idle during GPU ops, active during CPU parallel ops |

### Key Metrics (27-element float32 tensor)
| Metric | Our Library | TensorFlow |
|--------|------------|------------|
| GPU kernel time | ~3.8 μs per instance | ~1.3 μs (`BlockReduceKernel`) |
| End-to-end (NVTX) | ~18 μs (warmed up) | ~5 μs |
| First-call overhead | ~89 ms (`cuLibraryLoadData`) | ~3.6 ms |

### CUDA API Call Flow (per reduction)
```
ioctl (init) → cudaMalloc ×5 → cudaMemcpy ×5 → cudaLaunchKernel → cudaFree ×5
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              TARGET: Reduce to 1 malloc + 1 memcpy
```
