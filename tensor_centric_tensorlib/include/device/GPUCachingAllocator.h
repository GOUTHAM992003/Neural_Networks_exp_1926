#pragma once

#include <iostream>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <set>
#include <cmath>
#include <cuda_runtime.h>
#include "device/Allocator.h"

namespace OwnTensor {
namespace device {

// Pure BFC Constants (TensorFlow defaults)
constexpr size_t kMinBlockSize = 512;          // 512-byte alignment (User requested 512 in edits)
constexpr int kNumBins = 24;                   // 24 Bins (Covers up to ~4GB distinct buckets, larger blocks share the top bucket)

struct Chunk {
    void* ptr;
    size_t size;
    bool allocated;
    Chunk* prev;
    Chunk* next;
    int bin_index;
    cudaStream_t stream; 
    
    Chunk(void* p, size_t s) : ptr(p), size(s), allocated(false), prev(nullptr), next(nullptr), bin_index(-1), stream(0) {}
};

struct ChunkComparator {
    bool operator()(const Chunk* a, const Chunk* b) const {
        if (a->size != b->size) return a->size < b->size;
        return a->ptr < b->ptr;
    }
};

struct Bin {
    std::set<Chunk*, ChunkComparator> free_chunks;
};

struct Block {
    void* ptr;
    size_t size;
};

// TensorFlow-style Allocator Stats
struct AllocatorStats {
    int64_t bytes_in_use;          // Currenly allocated bytes (Chunk Payload)
    int64_t bytes_limit;           // Hard limit (if any)
    int64_t max_bytes_in_use;      // High watermark of bytes_in_use
    int64_t max_alloc_size;        // Largest single allocation
    int64_t bytes_reserved;        // Total bytes obtained from System (cudaMalloc)
    int64_t max_bytes_reserved;    // High watermark of bytes_reserved
    int64_t num_allocs;            // Total allocation calls
    int64_t num_alloc_retries;     // Number of retries (GC/Growth)
    int64_t num_ooms;              // Number of OOM errors
    int64_t segment_allocs;        // Number of cudaMalloc calls
    int64_t segment_frees;         // Number of cudaFree calls
    int64_t largest_free_block_bytes; // Size of largest free chunk
    double fragmentation;          // Fragmentation ratio [0, 1]

    AllocatorStats() 
        : bytes_in_use(0), bytes_limit(0), max_bytes_in_use(0), max_alloc_size(0), 
          bytes_reserved(0), max_bytes_reserved(0), num_allocs(0),
          num_alloc_retries(0), num_ooms(0), segment_allocs(0), segment_frees(0),
          largest_free_block_bytes(0), fragmentation(0.0) {}
};

class GPUCachingAllocator : public Allocator {
public:
    // Multi-GPU Support: instance(device_id)
    static GPUCachingAllocator* instance(int device_id);
    
    static void set_allow_growth(bool allow);
    static bool get_allow_growth();

    // Checklist Item: Garbage Collection Flag
    static void set_garbage_collection(bool enable);
    static bool get_garbage_collection();

    // Interface
    void* allocate(size_t bytes) override;
    void* allocate(size_t bytes, cudaStream_t stream);
    void deallocate(void* ptr) override;
    
    void garbage_collection(); 
    
    void memset(void* ptr, int value, size_t bytes) override;
    void memcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) override;
    void memsetAsync(void* ptr, int value, size_t bytes, cudaStream_t stream) override;
    void memcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) override;

    // Stats
    AllocatorStats get_stats();

private:
    GPUCachingAllocator(int device_id);
    ~GPUCachingAllocator();

    void init_greedy_if_needed();
    void* allocate_internal(size_t aligned_size, cudaStream_t stream);
    bool extend(size_t size, cudaStream_t stream);
    
    // BFC Logic
    int get_bin_index(size_t size);
    void insert_chunk(Chunk* chunk);
    void remove_chunk(Chunk* chunk);
    void split_chunk(Chunk* chunk, size_t size);
    void merge_chunk(Chunk* chunk);

    std::mutex mutex_;
    
    static bool allow_growth_;
    static bool garbage_collection_; // New Flag
    bool initialized_;
    int device_id_; 
    
    // Allocator State
    std::vector<Bin> bins_;
    std::vector<Block*> blocks_;
    std::unordered_map<void*, Chunk*> ptr_to_chunk_map_;
    
    // Growth State
    size_t next_allocation_size_;
    
    // Stats State
    AllocatorStats stats_;
};

} // namespace device
} // namespace OwnTensor
