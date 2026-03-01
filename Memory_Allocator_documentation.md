# Memory Allocator — Complete Documentation
## TensorFlow BFC Allocator Study, Allocator Class Restructuring, Memcpy/Memset Implementation

---

## 0. The Problem — Why Restructuring Was Needed

### What the Old System Looked Like

The original allocator class was a **monolithic mess**:

1. **Everything crammed into one class** — Allocation, deallocation, memcpy, and memset were all mixed together in a single `Allocator` class with no clean separation
2. **Redundant memcpy/memset in every child class** — `CPUAllocator`, `CUDAAllocator` both had copy-pasted synchronous memcpy/memset functions that should never have been there
3. **"Asynchronous" functions that were synchronous** — Functions named `memcpyAsync` were actually calling synchronous APIs internally, just named misleadingly
4. **No pinned memory support** — No function for `cudaMallocHost()` / pinned memory allocation in CPU RAM, which is critical for fast CPU↔GPU transfers
5. **memcpy/memset don't belong in allocators** — An allocator's job is to allocate and deallocate memory. Memory copying and setting are **transfer** and **initialization** tasks that should be separate

### The Restructured Design

I bifurcated the old monolithic allocator into **4 clean, separate responsibilities**:

```
BEFORE (Old Design):                    AFTER (New Design):
┌─────────────────────┐                ┌─────────────────┐
│     Allocator       │                │    Allocator     │ → Allocator.h
│  ├── allocate()     │                │  ├── allocate()  │    (Base class - pure virtual)
│  ├── deallocate()   │                │  └── deallocate()│
│  ├── memcpy()       │                └─────────────────┘
│  ├── memcpyAsync()  │                         │
│  ├── memset()       │                  ┌──────┴──────┐
│  └── memsetAsync()  │                  │             │
│     (all mixed)     │          CPUAllocator    CUDAAllocator
└─────────────────────┘          (malloc/free)   (cudaMalloc/cudaFree)
                                         │             │
                                  ┌──────┘             │
                                  │                    │
                              PinnedCPU         (Future(kathir bro is implementing this): pytorch's
                            Allocator          CUDA  Caching Allocator)
                            (cudaMallocHost/
                             cudaFreeHost)

           ┌──────────────────┐    ┌──────────────────┐
           │  DeviceTransfer  │    │    DeviceSet      │
           │  copy_memory()   │    │   set_memory()    │
           │  (Memcpy tasks)  │    │  (Memset tasks)   │
           │  DeviceTransfer.h│    │   DeviceSet.h     │
           │  DeviceTransfer  │    │   DeviceSet.cpp   │
           │         .cpp     │    │                   │
           └──────────────────┘    └──────────────────┘
```

---

## 1. The 4 Tasks — Bifurcation and API Selection

### 1.1 Task 1: Allocation

| Sub-task | API Used | Why This API |
|----------|----------|-------------|
| **CPU Standard allocation** |  `malloc(size)` | Standard pageable (swappable) memory. Fine for normal CPU tensors |
| **CPU Pinned allocation** | `cudaMallocHost(ptr, size)` | Allocates page-locked (pinned) memory that can't be swapped to disk. Required for fast async CPU→GPU transfers via DMA |
| **GPU allocation** | `cudaMalloc(ptr, size)` | Standard CUDA device memory allocation |
| **GPU allocation ** | ` custom pool | I implemented TensorFlow's BFC caching allocator,but finalised to go with pytorch's PU allocator,as it have more better features compared to tensorflow's one  — reuses freed GPU blocks instead of calling cudaMalloc every time |

**Why we need pinned memory:**
```
CPU→GPU transfer with pageable memory:
  CPU Buffer → [OS copies to staging buffer] → [PCIe DMA to GPU] → GPU Buffer
  (2 copies! Slow!)

CPU→GPU transfer with pinned memory:
  Pinned CPU Buffer → [PCIe DMA directly to GPU] → GPU Buffer
  (1 copy! Fast! DMA can access pinned memory directly)
```

### 1.2 Task 2: Deallocation

| Sub-task | API Used | Why |
|----------|----------|-----|
| **CPU Standard** | `delete[] ptr` / `free(ptr)` | Matches `new[]` / `malloc` |
| **CPU Pinned** | `cudaFreeHost(ptr)` | MUST match `cudaMallocHost`. Using regular `free()` on pinned memory = undefined behavior! |
| **GPU** | `cudaFree(ptr)` | Standard CUDA deallocation. Note: we clear error state with `cudaGetLastError()` after failure because deallocate may be called from destructors (can't throw) |

### 1.3 Task 3: Memcpy (Asynchronous)

Memory copy is handled by `DeviceTransfer.h/.cpp` with the unified `copy_memory()` function:

| Transfer Direction | API Used | Timing (500 MiB) | Why This API |
|-------------------|----------|:-:|-------------|
| **CPU ↔ CPU** | `std::memcpy(dst, src, size)` | ~22 ms | Fastest for host-to-host. No CUDA overhead needed |
| **CPU → GPU** | `cudaMemcpyAsync(dst, src, size, HostToDevice, stream)` | ~30 ms | Asynchronous, stream-aware. With pinned source: bandwidth ~163 GiB/s |
| **GPU → CPU** | `cudaMemcpyAsync(dst, src, size, DeviceToHost, stream)` | ~30 ms | Same async API, direction controlled by `kind` parameter |
| **GPU ↔ GPU** | `cudaMemcpyAsync(dst, src, size, DeviceToDevice, stream)` | ~0.003 ms | Blazing fast — stays entirely on GPU, uses GPU's internal bandwidth |

**The `cudaMemcpyKind` parameter:**
```cpp
enum cudaMemcpyKind {
    cudaMemcpyHostToDevice   = 1,  // CPU → GPU
    cudaMemcpyDeviceToHost   = 2,  // GPU → CPU
    cudaMemcpyDeviceToDevice = 3,  // GPU → GPU
    cudaMemcpyHostToHost     = 0   // CPU → CPU (not used — we use std::memcpy instead)
};
```

### 1.4 Task 4: Memset (Asynchronous)

Memory initialization handled by `DeviceSet.h/.cpp`:

| Device | API Used | Why |
|--------|----------|-----|
| **CPU** | `std::memset(ptr, value, size)` | Standard C, no CUDA overhead |
| **GPU** | `cudaMemsetAsync(ptr, value, size, stream)` | Asynchronous, stream-aware. Doesn't block CPU |

---

## 2. Timing Experiments — API Selection Evidence

Before choosing APIs, I ran timing experiments on multiple NVIDIA APIs to compare performance for each task:

### 2.1 CPU↔CPU Memcpy (500 MiB)

| Source → Dest | API | Time | Conclusion |
|:---:|:---:|:---:|:---:|
| pageable → pageable | `std::memcpy` | ~22 ms | All ~22 ms — type of allocation doesn't matter for CPU↔CPU |
| pageable → pinned | `std::memcpy` | ~22.4 ms | |
| pinned → pageable | `std::memcpy` | ~22 ms | |
| pinned → pinned | `std::memcpy` | ~22.4 ms | |

**Conclusion:** For CPU↔CPU, `std::memcpy` is always the right choice. Pinned vs pageable doesn't affect CPU-side copy speed.

### 2.2 CPU↔GPU Transfer (500 MiB)

| API | Behavior | Time | Notes |
|:---:|:---:|:---:|:---:|
| `cudaMemcpy(dst, src, size, kind)` | Synchronous + 2 copies | ~35 ms | Blocks CPU. If source is pageable: stages through internal pinned buffer |
| `cudaMemcpyAsync` + `cudaStreamSynchronize` | Sync(1 copy) | ~2.6 ms | Same as cudaMemcpy + pinned source (1 DMA copy) |
| `cudaMemcpyAsync` (no sync, pageable src) | Misleadingly "async" | ~35 ms | Actually blocks CPU! CUDA must stage pageable memory |
| `cudaMemcpyAsync` (pinned src) | True async | ~0.003 ms | Returns immediately. DMA runs in background. ~163 GiB/s bandwidth |
| `cudaMemcpyAsync` (pageable dst, D→H) | Sync internally | ~37 ms | Must stage buffer, blocks CPU |

**Critical Insight:** `cudaMemcpyAsync` is only truly asynchronous when the host memory is **pinned**! With pageable memory, it silently falls back to synchronous behavior.

**Final API choices based on experiments:**
- **CPU → GPU:** `cudaMemcpyAsync` with `cudaMemcpyHostToDevice` kind (truly async if source is pinned)
- **GPU → CPU:** `cudaMemcpyAsync` with `cudaMemcpyDeviceToHost` kind
- **GPU → GPU:** `cudaMemcpyAsync` with `cudaMemcpyDeviceToDevice` kind
- **CPU → CPU:** `std::memcpy` (no CUDA needed)

---

## 3. System Design — The Restructured Architecture

### 3.1 File Structure

```
include/device/
├── Allocator.h           ← Abstract base class (pure virtual allocate/deallocate)
├── CPUAllocator.h        ← Standard CPU allocation (new/delete)
├── CUDAAllocator.h       ← GPU allocation (cudaMalloc/cudaFree)
├── AllocatorRegistry.h   ← Dispatcher: Device → correct Allocator
├── DeviceTransfer.h      ← copy_memory() — unified memcpy dispatcher
├── DeviceCore.h          ← CUDA device queries + stream management
└── Device.h              ← Device enum (CPU, CUDA) + DeviceIndex

src/device/
├── CPUAllocator.cpp      ← Implementation
├── CUDAAllocator.cpp     ← Implementation
├── AllocatorRegistry.cpp ← Singleton allocator instances
├── DeviceTransfer.cpp    ← 4-way copy routing implementation
└── DeviceCore.cpp        ← cuda_available(), stream get/set
```

### 3.2 Class Hierarchy

```
Allocator (Abstract Base Class) — Allocator.h
│
│   Pure virtual methods:
│   ├── allocate(bytes) → void*       // The ONLY job of an allocator
│   ├── deallocate(ptr) → void        // The ONLY other job
│   ├── memsetAsync(ptr, val, bytes, stream)    // Still here for backward compat
│   └── memcpyAsync(dst, src, bytes, kind, stream)
│
│   Default implementations (call async + sync):
│   ├── memset(ptr, val, bytes)       // Wrapper: memsetAsync + cudaStreamSynchronize
│   └── memcpy(dst, src, bytes, kind) // Wrapper: memcpyAsync + cudaStreamSynchronize
│
├── CPUAllocator — CPUAllocator.h/.cpp
│   ├── allocate():    new uint8_t[bytes]
│   ├── deallocate():  delete[] static_cast<uint8_t*>(ptr)
│   ├── memsetAsync(): std::memset (ignores stream — CPU has no streams)
│   └── memcpyAsync(): std::memcpy (ignores stream and kind — all host memory)
│
└── CUDAAllocator — CUDAAllocator.h/.cpp
    ├── allocate():    cudaMalloc(&ptr, bytes) with error checking
    ├── deallocate():  cudaFree(ptr) with error clearing (safe for destructors)
    ├── memsetAsync(): cudaMemsetAsync(ptr, value, bytes, stream)
    └── memcpyAsync(): cudaMemcpyAsync(dst, src, bytes, kind, stream)
```

### 3.3 AllocatorRegistry — The Dispatcher

Routes `Device` enum to the correct allocator instance. Uses **file-scope static singletons**:

```cpp
namespace {  // Anonymous namespace = internal linkage
    CPUAllocator  cpu_allocator;   // One global instance
    CUDAAllocator cuda_allocator;  // One global instance
}

Allocator* AllocatorRegistry::get_allocator(Device device) {
    if (device == Device::CPU) return &cpu_allocator;
    else                       return &cuda_allocator;
}
```

**Why singletons?** Allocators hold no per-allocation state (pure strategy pattern). One instance per device type is sufficient.

### 3.4 DeviceTransfer — The Unified Memcpy

`copy_memory()` takes source and destination **device info** and routes to the correct API:

```cpp
void copy_memory(void* dst, Device dst_device,
                 const void* src, Device src_device, size_t bytes) {
    
    if (bytes == 0) return;
    
    // CPU → CPU: use std::memcpy (via CPUAllocator)
    if (dst_device == CPU && src_device == CPU) {
        AllocatorRegistry::get_cpu_allocator()->memcpy(dst, src, bytes, ...);
        return;
    }
    
    // GPU → GPU: use CUDAAllocator's cudaMemcpyAsync(DeviceToDevice)
    if (dst_device == CUDA && src_device == CUDA) {
        AllocatorRegistry::get_cuda_allocator()->memcpy(dst, src, bytes, DeviceToDevice);
        return;
    }
    
    // CPU → GPU: direct cudaMemcpyAsync(HostToDevice, stream)
    if (dst_device == CUDA && src_device == CPU) {
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, getCurrentStream());
        return;
    }
    
    // GPU → CPU: direct cudaMemcpyAsync(DeviceToHost, stream)
    if (dst_device == CPU && src_device == CUDA) {
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, getCurrentStream());
        return;
    }
}
```

### 3.5 Stream Management

Thread-local CUDA stream storage for async operations:

```cpp
namespace OwnTensor::cuda_1926 {
    static thread_local cudaStream_t g_current_stream = 0;  // Default = stream 0
    
    void setCurrentStream(cudaStream_t stream) { g_current_stream = stream; }
    cudaStream_t getCurrentStream()            { return g_current_stream; }
}
```

**Why `thread_local`?** Each CPU thread can work with a different CUDA stream independently. This enables multi-stream parallelism without global locks.

---

## 4. TensorFlow's BFC Allocator — What We Studied

### 4.1 What is BFC?

**Best-Fit with Coalescing** — TensorFlow's GPU memory allocator. Instead of calling `cudaMalloc`/`cudaFree` for every tensor (expensive! ~1ms each), BFC pre-allocates a large GPU memory pool and manages sub-allocations internally.

### 4.2 Why It Exists

```
WITHOUT BFC (naive approach):
  Tensor A = cudaMalloc(100MB)  → ~1 ms
  Tensor B = cudaMalloc(50MB)   → ~1 ms
  cudaFree(A)                   → ~1 ms
  Tensor C = cudaMalloc(80MB)   → ~1 ms  (even though A's freed space fits!)
  Total overhead: ~4 ms of just allocation

WITH BFC:
  Pool = cudaMalloc(2GB)        → ~1 ms (ONCE at startup)
  Tensor A = pool.alloc(100MB)  → ~0.001 ms (pointer arithmetic!)
  Tensor B = pool.alloc(50MB)   → ~0.001 ms
  pool.free(A)                  → ~0.001 ms (mark as free, don't call cudaFree)
  Tensor C = pool.alloc(80MB)   → ~0.001 ms (reuse A's freed block!)
  Total overhead: ~1.004 ms
```

### 4.3 BFC Core Concepts

**Chunks and Bins:**
- **Chunk:** A contiguous block of GPU memory with metadata (size, in_use flag, prev/next pointers for coalescing)
- **Bin:** A size-class bucket (TensorFlow uses powers of 2: 256B, 512B, 1KB, 2KB, ..., 256MB). Each bin holds a free-list of chunks in that size range

**Best-Fit Algorithm:**
```
BFC_ALLOCATE(requested_size):
    1. Round up to next power of 2 (minimum 256 bytes)
    2. Find the smallest bin >= requested_size
    3. Search that bin's free-list for best-fitting chunk
    4. If found: split chunk if much larger than needed → return allocated portion
    5. If not found: try next larger bin
    6. If all bins empty: extend the pool (or call cudaMalloc for more memory)
```

**Coalescing on Free:**
```
BFC_FREE(chunk):
    1. Mark chunk as free
    2. Check if PREVIOUS adjacent chunk is also free → merge them
    3. Check if NEXT adjacent chunk is also free → merge them
    4. Insert merged chunk into appropriate bin
```

**Why "coalescing" matters:** Without it, you get **fragmentation** — many small free chunks that can't serve a large allocation even though total free space is sufficient.

### 4.4 Key BFC Design Decisions

| Design Choice | TensorFlow BFC | Our Implementation Status |
|---------------|----------------|--------------------------|
| Initial pool size | Usually 95% of GPU memory | Future work — currently using direct cudaMalloc |
| Minimum chunk | 256 bytes | To be determined |
| Bin sizes | Powers of 2 | To be determined |
| Thread safety | Mutex-locked | Future: consider lock-free or per-stream pools |
| Splitting policy | Split if leftover >= min chunk size | Standard approach |
| Coalescing | Immediate on free | Optimal for reducing fragmentation |
| Extending pool | Grow in 2× increments | Avoids costly extension waste |
| Stream awareness | NOT stream-aware (TF uses stream 0 mainly) | Our version may be stream-aware |

### 4.5 PyTorch's Approach (Comparison)

PyTorch uses **CUDACachingAllocator** — similar concept but stream-aware:
- Uses "blocks" (similar to TF's chunks) organized by size and stream
- Maintains separate free-lists per CUDA stream
- Supports `cudaMallocAsync` on newer driver versions (CUDA 11.2+)
- Has garbage collection to release blocks back to CUDA

### 4.6 Future Integration Path

```
Current:     Tensor.allocate() → AllocatorRegistry → CUDAAllocator → cudaMalloc (every time)
Future:      Tensor.allocate() → AllocatorRegistry → CachingCUDAAllocator → BFC pool → fast sub-allocation
                                                            └── cudaMalloc only on pool exhaustion
```

The `CachingCUDAAllocator` would inherit from `Allocator` (same interface) — just override `allocate()` and `deallocate()` with the BFC logic. The rest of the system (DeviceTransfer, memset, etc.) stays unchanged.

---

## 5. Numerical Stability

### 5.1 Error Handling in Deallocation

CUDA deallocation can fail silently in destructors. Our design clears the error state:
```cpp
void CUDAAllocator::deallocate(void* ptr) {
    if (ptr) {
        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess) {
            std::cerr << error_msg << std::endl;
            cudaGetLastError();  // CRITICAL: Clear error state!
            // DON'T throw — might be in destructor
        }
    }
}
```

**Why `cudaGetLastError()`?** CUDA uses a sticky error model — once an error occurs, ALL subsequent CUDA calls will fail until the error is cleared. Without this, a single failed deallocation would cascade and crash the entire program.

### 5.2 Zero-byte Transfer Guard

```cpp
if (bytes == 0) return;  // First line of copy_memory()
```
Attempting `cudaMemcpyAsync` with 0 bytes can trigger undefined behavior on some driver versions.

---

## 6. Memory Layout

### 6.1 Pageable vs Pinned Memory

```
PAGEABLE MEMORY (default malloc/new):
┌────────────────────────────────────┐
│         Virtual Address Space      │
│  ┌──────┐  ┌──────┐  ┌──────┐    │
│  │Page 1│  │Page 2│  │Page 3│    │  ← OS can swap ANY page to disk
│  └──────┘  └──────┘  └──────┘    │
│       ↕ swap     ↕ swap          │
│  ┌─────────────────────┐         │
│  │    Disk (Swap)       │         │
│  └─────────────────────┘         │
└────────────────────────────────────┘
  Problem: GPU DMA engine can't access this — must stage through driver buffer!

PINNED MEMORY (cudaMallocHost):
┌────────────────────────────────────┐
│         Physical RAM (locked)      │
│  ┌──────┐  ┌──────┐  ┌──────┐    │
│  │Page 1│  │Page 2│  │Page 3│    │  ← LOCKED — OS CANNOT swap these!
│  └──────┘  └──────┘  └──────┘    │
│       ↑ DMA directly ↑           │
│  ┌─────────────────────┐         │
│  │    GPU (via PCIe)    │         │
│  └─────────────────────┘         │
└────────────────────────────────────┘
  GPU DMA reads/writes directly — no staging buffer needed!
```

### 6.2 PCIe Bandwidth

```
PCIe Gen3 x16: ~16 GiB/s (theoretical)
PCIe Gen4 x16: ~32 GiB/s
PCIe Gen5 x16: ~64 GiB/s

Our measured: ~163 GiB/s (GPU internal), ~16 GiB/s (PCIe, matches Gen3)
Max theoretical bandwidth observed: ~137.4 GiB/s (internal GPU memory bandwidth)
```

---

## 7. Pseudocode

### 7.1 Complete Tensor Allocation Flow

```
FUNCTION create_tensor(shape, dtype, device):
    bytes = product(shape.dims) * dtype_traits[dtype].size
    
    allocator = AllocatorRegistry.get_allocator(device)
    raw_ptr = allocator->allocate(bytes)      // CPU: new[], GPU: cudaMalloc
    
    // Zero-initialize
    IF device == CPU:
        std::memset(raw_ptr, 0, bytes)
    ELSE:
        cudaMemsetAsync(raw_ptr, 0, bytes, getCurrentStream())
    
    RETURN Tensor(raw_ptr, shape, dtype, device)
```

### 7.2 CPU→GPU Transfer

```
FUNCTION copy_cpu_to_gpu(dst_ptr, src_ptr, bytes):
    stream = cuda_1926::getCurrentStream()
    
    // If source is pinned: DMA runs asynchronously in background (~0.003ms return)
    // If source is pageable: CUDA stages internally (~35ms, blocks CPU)
    err = cudaMemcpyAsync(dst_ptr, src_ptr, bytes, cudaMemcpyHostToDevice, stream)
    
    IF err != cudaSuccess:
        THROW "CPU→GPU transfer failed: " + cudaGetErrorString(err)
```

### 7.3 BFC Allocate (Future)

```
FUNCTION bfc_allocate(requested_bytes):
    rounded = round_up_to_power_of_2(requested_bytes)
    bin_index = log2(rounded) - log2(MIN_CHUNK_SIZE)
    
    // Search bins from smallest fitting to largest
    FOR bin = bin_index TO MAX_BIN:
        IF bin.free_list is NOT EMPTY:
            chunk = best_fit(bin.free_list, rounded)
            
            // Split if chunk is much larger
            IF chunk.size - rounded >= MIN_CHUNK_SIZE:
                leftover = split(chunk, rounded)
                insert_to_bin(leftover)
            
            chunk.in_use = true
            RETURN chunk.ptr
    
    // All bins empty — need more GPU memory
    new_pool = cudaMalloc(max(rounded, POOL_GROWTH_SIZE))
    add_to_pool(new_pool)
    RETURN bfc_allocate(requested_bytes)  // Retry with new memory
```

---

## 8. Research Material

### 8.1 References

| Source | What We Studied |
|--------|----------------|
| **TensorFlow `core/common_runtime/bfc_allocator.cc`** | The actual BFC implementation — chunks, bins, best-fit, coalescing |
| **PyTorch `c10/cuda/CUDACachingAllocator.cpp`** | Block-based caching, stream-aware pools, garbage collection |
| **NVIDIA CUDA Runtime API docs** | `cudaMalloc`, `cudaFree`, `cudaMallocHost`, `cudaFreeHost`, `cudaMemcpyAsync`, `cudaMemsetAsync` behavior |
| **NVIDIA Developer Blog - "How to Optimize Data Transfers"** | Pinned vs pageable transfer benchmarks, PCIe bandwidth analysis |
| **Our own timing experiments** | Measured each API on 500 MiB transfers to validate API choices (see Section 2) |

### 8.2 Source Files Reference

| File | Lines | What's Inside |
|------|:---:|------|
| `Allocator.h` | 46 | Abstract base class: virtual allocate/deallocate + default sync wrappers for memset/memcpy |
| `CPUAllocator.h/.cpp` | 18 + 42 | new[]/delete[], std::memcpy, std::memset |
| `CUDAAllocator.h/.cpp` | 19 + 89 | cudaMalloc/cudaFree with error handling, cudaMemcpyAsync/cudaMemsetAsync |
| `AllocatorRegistry.h/.cpp` | 13 + 28 | Static singleton dispatcher: Device → Allocator* |
| `DeviceTransfer.h/.cpp` | 14 + 62 | `copy_memory()` — 4-way routing: CPU↔CPU, CPU→GPU, GPU→CPU, GPU↔GPU |
| `DeviceCore.h/.cpp` | 19 + 58 | cuda_available(), cuda_device_count(), thread_local stream management |
| `Device.h` | 24 | Device enum (CPU, CUDA) + DeviceIndex struct |
