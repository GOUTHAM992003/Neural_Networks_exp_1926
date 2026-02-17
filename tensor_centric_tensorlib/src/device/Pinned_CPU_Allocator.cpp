#include "device/Pinned_CPU_Allocator.h"
#include <cuda_runtime.h>
#include <new>

namespace OwnTensor {

// Constructor
Pinned_CPU_Allocator::Pinned_CPU_Allocator(unsigned int flags) 
    : flags_(flags) 
{
}

void* Pinned_CPU_Allocator::allocate(size_t bytes) {
    void* ptr = nullptr;

    // TO-DO(Optimization): High-Throughput Scenarios
    // Current Implementation: cudaHostAlloc()
    // Limitation: The CUDA driver takes a global lock on the device context during allocation.
    // This serializes parallel allocations from multiple threads (e.g., in DataLoaders),
    // causing a bottleneck.
    //
    // Proposed Optimization (as done in PyTorch's allocWithCudaHostRegister):
    // 1. Use standard malloc() (thread-safe, no driver lock).
    // 2. Memset pages (to force OS to pre-fault/assign physical RAM).
    // 3. Call cudaHostRegister() to pin the memory.
    // This pattern avoids the heavy driver allocation lock and scales better with threads.
    cudaError_t err = cudaHostAlloc(&ptr, bytes, flags_);  // Use stored flags
    if (err != cudaSuccess) {
        throw std::bad_alloc();
    }
    return ptr;
}

void Pinned_CPU_Allocator::deallocate(void* ptr) {
    if (ptr) {
        cudaFreeHost(ptr);
    }
}

void Pinned_CPU_Allocator::memsetAsync(void* ptr, int value, size_t bytes, cudaStream_t stream) {
    cudaMemsetAsync(ptr, value, bytes, stream);
}

void Pinned_CPU_Allocator::memcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
    cudaMemcpyAsync(dst, src, bytes, kind, stream);
}

} // namespace OwnTensor
