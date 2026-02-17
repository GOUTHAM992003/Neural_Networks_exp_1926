#include "device/CPUAllocator.h"
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new> // for std::bad_alloc
#include <cuda_runtime.h> // Essential for cudaMemcpyAsync

namespace OwnTensor
{
    void* CPUAllocator::allocate(size_t bytes)
    {
        // TODO(Optimization): Switch from malloc to posix_memalign (or aligned_alloc in C++17) 
        // for better alignment suitable for AVX/SIMD instructions.
        //
        // REFERENCE IMPLEMENTATIONS:
        // 
        // 1. PyTorch Approach:
        //    File: c10/core/impl/alloc_cpu.cpp
        //    - Allocation (Line 126): `posix_memalign(&data, c10_compute_alignment(nbytes), nbytes)`
        //         (Uses 64-byte alignment by default for AVX512)
        //    - Deallocation (Line 172): `free(data)`
        //    - Mimalloc Support (Line 112): `mi_malloc_aligned(nbytes, gAlignment)` 
        //      (Microsoft's high-performance, compact allocator useful for small object fragmentation)
        // 
        // 2. TensorFlow Approach:
        //    Definition:
        //    File: tensorflow/third_party/xla/xla/tsl/platform/default/port.cc
        //    - Implementation (Line 295): `posix_memalign(&ptr, alignment, size)` wrapped inside `AlignedMalloc`
        //         (Falls back to malloc if alignment < sizeof(void*))
        //    - Deallocation (Line 317): `free(ptr)` inside `AlignedFree` -> `Free`
        //
        //    Usage:
        //    File: tensorflow/third_party/xla/xla/tsl/framework/cpu_allocator_impl.cc
        //    - Call Site (Line 87): `port::AlignedMalloc(num_bytes, static_cast<std::align_val_t>(alignment))`
        //      inside `CPUAllocator::AllocateRaw`

        void* ptr = std::malloc(bytes); //new throws bad-alloc error on failure,but here for malloc,manually should check whether its nullptr or not.
        if (ptr == nullptr && bytes > 0) {
             throw std::bad_alloc();
        }
        return ptr;
    }

    void CPUAllocator::deallocate(void* ptr)
    {
        std::free(ptr);
    }

    // Asynchronous versions
    void CPUAllocator::memsetAsync(void* ptr, int value, size_t bytes, cudaStream_t stream) 
    {
        // Standard malloc memory cannot be set by cudaMemsetAsync.
        // We must use std::memset (Synchronous on CPU).
        // This is "Fake Async" but required for correctness on non-pinned memory.
        std::memset(ptr, value, bytes);
        (void)stream; // Unused
    }

    void CPUAllocator::memcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) 
    {
        // cudaMemcpyAsync handles Host pointers correctly for transfers!
        // It provides true async for H2D/D2H (if pinned) and correct sync behavior for H2H.
        cudaMemcpyAsync(dst, src, bytes, kind, stream);
    }

    // Synchronous versions
    void CPUAllocator::memset(void* ptr, int value, size_t bytes)
    {
         // Must use std::memset for standard CPU memory.
        std::memset(ptr, value, bytes); 
    }

    void CPUAllocator::memcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) 
    {
        if (kind == cudaMemcpyHostToHost) {
            std::memcpy(dst, src, bytes); // Optimization
        } else {
            // Must use CUDA for D2H/H2D
            cudaMemcpy(dst, src, bytes, kind);
        }
    }

}