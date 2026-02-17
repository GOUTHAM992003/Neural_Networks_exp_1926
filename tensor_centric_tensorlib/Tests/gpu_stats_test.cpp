#include <iostream>
#include <iomanip>
#include <vector>
#include <cuda_runtime.h>
#include "device/GPUCachingAllocator.h"

using namespace OwnTensor::device;

void print_stats(const AllocatorStats& stats) {
    auto to_mb = [](int64_t bytes) { return bytes / 1024.0 / 1024.0; };
    
    std::cout << "\n=========================================" << std::endl;
    std::cout << "       GPU Memory Allocator Stats        " << std::endl;
    std::cout << "=========================================" << std::endl;
    
    std::cout << std::fixed << std::setprecision(2);
    
    std::cout << " [Usage]" << std::endl;
    std::cout << "  Bytes In Use (Tensor Payload): " << to_mb(stats.bytes_in_use) << " MB" << std::endl;
    std::cout << "  Peak Bytes In Use:             " << to_mb(stats.max_bytes_in_use) << " MB" << std::endl;
    std::cout << "  Allocated Segments (Reserved): " << to_mb(stats.bytes_reserved) << " MB" << std::endl;
    std::cout << "  Peak Reserved:                 " << to_mb(stats.max_bytes_reserved) << " MB" << std::endl;
    std::cout << "  Soft Limit:                    " << to_mb(stats.bytes_limit) << " MB" << std::endl;
    
    std::cout << "\n [Counters]" << std::endl;
    std::cout << "  Num Allocations:               " << stats.num_allocs << std::endl;
    std::cout << "  Num Retries (GC/Growth):       " << stats.num_alloc_retries << std::endl;
    std::cout << "  Num OOMs:                      " << stats.num_ooms << std::endl;
    std::cout << "  Segment Allocs (cudaMalloc):   " << stats.segment_allocs << std::endl;
    std::cout << "  Segment Frees (cudaFree):      " << stats.segment_frees << std::endl;

    std::cout << "\n [Fragmentation]" << std::endl;
    std::cout << "  Largest Free Block:            " << to_mb(stats.largest_free_block_bytes) << " MB" << std::endl;
    std::cout << "  Fragmentation Ratio:           " << stats.fragmentation * 100.0 << " %" << std::endl;
    std::cout << "=========================================\n" << std::endl;
}

int main() {
    std::cout << "Starting GPU Stats Test..." << std::endl;
    
    // Enable Growth for cleaner counting
    GPUCachingAllocator::set_allow_growth(true);
    
    GPUCachingAllocator* allocator = GPUCachingAllocator::instance(0);
    
    // 1. Initial Stats
    std::cout << "\n--- Initial State ---";
    print_stats(allocator->get_stats());
    
    // 2. Allocate 100MB
    std::cout << "\n--- Allocating 100MB ---";
    void* p1 = allocator->allocate(100 * 1024 * 1024);
    print_stats(allocator->get_stats());
    
    // 3. Allocate 50MB
    std::cout << "\n--- Allocating 50MB ---";
    void* p2 = allocator->allocate(50 * 1024 * 1024);
    print_stats(allocator->get_stats());
    
    // 4. Free 100MB (Create Fragmentation)
    std::cout << "\n--- Freeing 100MB (Creating Hole/Fragmentation) ---";
    allocator->deallocate(p1);
    print_stats(allocator->get_stats());

    // 5. Cleanup
    allocator->deallocate(p2);
    
    return 0;
}
