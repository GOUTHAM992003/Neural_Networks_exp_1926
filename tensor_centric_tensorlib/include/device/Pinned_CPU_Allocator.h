#pragma once
#include "device/Allocator.h"
#include <cuda_runtime.h>

namespace OwnTensor {

class Pinned_CPU_Allocator : public Allocator {
public:
    explicit Pinned_CPU_Allocator(unsigned int flags = 0);

    void* allocate(size_t bytes) override;
    void deallocate(void* ptr) override;

    void memsetAsync(void* ptr, int value, size_t bytes, cudaStream_t stream) override;
    void memcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) override;

private:
    unsigned int flags_;
};

} // namespace OwnTensor