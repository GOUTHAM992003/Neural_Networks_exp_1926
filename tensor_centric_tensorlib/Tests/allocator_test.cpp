#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include "device/AllocatorRegistry.h"
#include "device/GPUCachingAllocator.h"
#include "device/Allocator.h"

// Helper to check CUDA errors
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

using namespace OwnTensor;

void test_cpu_allocator() {
    std::cout << "[Test] CPU Allocator... ";
    Allocator* cpu_alloc = AllocatorRegistry::get_cpu_allocator();
    
    size_t size = 1024;
    void* ptr = cpu_alloc->allocate(size);
    assert(ptr != nullptr);
    
    // Test Memset (Sync)
    cpu_alloc->memset(ptr, 0xAA, size);
    
    unsigned char* bytes = static_cast<unsigned char*>(ptr);
    for(size_t i=0; i<size; i++) {
        if(bytes[i] != 0xAA) {
            std::cerr << "CPU Memset Failed at index " << i << std::endl;
            exit(1);
        }
    }
    
    cpu_alloc->deallocate(ptr);
    std::cout << "Passed." << std::endl;
}

void test_pinned_allocator() {
    std::cout << "[Test] Pinned CPU Allocator... ";
    Allocator* pinned_alloc = AllocatorRegistry::get_pinned_cpu_allocator(Pinned_Flag::Default);
    
    size_t size = 1024 * 1024; // 1MB
    void* ptr = pinned_alloc->allocate(size);
    assert(ptr != nullptr);
    
    // Check if it's actually pinned (rudimentary check using CUDA API)
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err == cudaSuccess) {
         // Valid pinned memory usually returns success and type Host
         assert(attr.type == cudaMemoryTypeHost);
    }
    
    // Test Memset (Async)
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    pinned_alloc->memsetAsync(ptr, 0xBB, size, stream);
    cudaStreamSynchronize(stream);
    
    unsigned char* bytes = static_cast<unsigned char*>(ptr);
    if(bytes[0] != 0xBB) {
         std::cerr << "Pinned MemsetAsync Failed." << std::endl;
         exit(1);
    }
    
    pinned_alloc->deallocate(ptr);
    cudaStreamDestroy(stream);
    std::cout << "Passed." << std::endl;
}

void test_gpu_allocator_features() {
    std::cout << "[Test] GPU Allocator (Greedy & Growth)... ";
    
    // Access static method to Config
    // Note: In real usage, this should be set before first alloc. 
    // Here we assume it wasn't initialized or we can switch it check.
    device::GPUCachingAllocator::set_allow_growth(true); // Enable Growth
    
    Allocator* gpu_alloc = AllocatorRegistry::get_cuda_allocator();
    
    // 1. Alloc Small
    size_t size1 = 1024 * 1024; // 1MB
    void* p1 = gpu_alloc->allocate(size1);
    assert(p1 != nullptr);
    
    // 2. Alloc Large (Trigger Growth if not greedy enough)
    size_t size2 = 50 * 1024 * 1024; // 50MB
    void* p2 = gpu_alloc->allocate(size2);
    assert(p2 != nullptr);
    
    // 3. Memset GPU
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    gpu_alloc->memsetAsync(p1, 0xCC, size1, stream);
    cudaStreamSynchronize(stream);
    
    // 4. Verify GPU Data (Copy back to Host)
    unsigned char* host_buf = new unsigned char[size1];
    
    // Use GPU Allocator's memcpy (D2H)
    gpu_alloc->memcpy(host_buf, p1, size1, cudaMemcpyDeviceToHost);
    
    if(host_buf[0] != 0xCC || host_buf[size1-1] != 0xCC) {
         std::cerr << "GPU Memset/Memcpy Failed." << std::endl;
         exit(1);
    }
    
    // 5. Transfer CPU -> GPU
    std::memset(host_buf, 0xDD, size1);
    AllocatorRegistry::get_cpu_allocator()->memcpyAsync(p1, host_buf, size1, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    
    // Verify again
    gpu_alloc->memcpy(host_buf, p1, size1, cudaMemcpyDeviceToHost);
    if(host_buf[0] != 0xDD) {
         std::cerr << "CPU->GPU Copy Failed." << std::endl;
         exit(1);
    }

    gpu_alloc->deallocate(p1);
    gpu_alloc->deallocate(p2);
    
    delete[] host_buf;
    cudaStreamDestroy(stream);
    std::cout << "Passed." << std::endl;
}

void test_dispatcher() {
    std::cout << "[Test] Registry Dispatcher... ";
    // This tests if get_cuda_allocator() correctly routes to GPUCachingAllocator backend
    Allocator* alloc = AllocatorRegistry::get_cuda_allocator();
    void* ptr = alloc->allocate(512);
    assert(ptr != nullptr);
    
    // Check attributes
    cudaPointerAttributes attr;
    CHECK_CUDA(cudaPointerGetAttributes(&attr, ptr));
    assert(attr.type == cudaMemoryTypeDevice);
    
    alloc->deallocate(ptr);
    std::cout << "Passed." << std::endl;
}

void test_gpu_stats() {
    std::cout << "[Test] GPU Allocator Stats... ";
    
    // Get Instance for Device 0
    device::GPUCachingAllocator* allocator = device::GPUCachingAllocator::instance(0);
    
    // Initial State
    auto stats1 = allocator->get_stats();
    // Note: bytes_reserved might be non-zero due to previous tests or greedy init
    
    size_t alloc_size = 1024 * 1024; // 1MB
    void* ptr = allocator->allocate(alloc_size);
    
    auto stats2 = allocator->get_stats();
    
    // Verify In Use Increased
    // Note: Allocator aligns to 512 bytes. 1MB is aligned.
    // If stats are tracked correctly:
    if (stats2.bytes_in_use <= stats1.bytes_in_use) {
         std::cerr << "Stats Failed: bytes_in_use did not increase. Before: " << stats1.bytes_in_use << " After: " << stats2.bytes_in_use << std::endl;
         exit(1);
    }
    
    if (stats2.num_allocs <= stats1.num_allocs) {
        std::cerr << "Stats Failed: num_allocs did not increase." << std::endl;
        exit(1);
    }

    allocator->deallocate(ptr);
    
    auto stats3 = allocator->get_stats();
    
    // Verify In Use Decreased
    if (stats3.bytes_in_use >= stats2.bytes_in_use) {
        std::cerr << "Stats Failed: bytes_in_use did not decrease after free." << std::endl;
        exit(1);
    }
    
    // Verify Max Bytes Persisted
    if (stats3.max_bytes_in_use < stats2.bytes_in_use) {
         std::cerr << "Stats Failed: max_bytes_in_use not tracking peak." << std::endl;
         exit(1);
    }
    
    std::cout << "Passed. (Peak: " << stats3.max_bytes_in_use / 1024.0 / 1024.0 << " MB)" << std::endl;
    std::cout << "   [Extended Stats]" << std::endl;
    std::cout << "   - Fragmentation: " << stats3.fragmentation * 100.0 << "%" << std::endl;
    std::cout << "   - Largest Free Block: " << stats3.largest_free_block_bytes / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "   - Segment Allocs: " << stats3.segment_allocs << std::endl;
    std::cout << "   - Retries: " << stats3.num_alloc_retries << std::endl;
}

int main() {
    std::cout << "Running Allocator Tests..." << std::endl;
    
    test_cpu_allocator();
    test_pinned_allocator();
    test_gpu_allocator_features();
    test_dispatcher();
    test_gpu_stats(); // New Test
    
    std::cout << "All Allocator Tests Passed Successfully!" << std::endl;
    return 0;
}
