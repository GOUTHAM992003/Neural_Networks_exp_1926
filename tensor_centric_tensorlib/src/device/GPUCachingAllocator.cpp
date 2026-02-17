#include "device/GPUCachingAllocator.h"
#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

namespace OwnTensor {
namespace device {

bool GPUCachingAllocator::allow_growth_ = false;
bool GPUCachingAllocator::garbage_collection_ = true; // Default Enabled

// --------------------------------------------------------------------------------
// Helper Functions (Proofs)
// --------------------------------------------------------------------------------

// TensorFlow /tensorflow/tensorflow/core/common_runtime/gpu/gpu_device.cc Lines 1156-1199 (MinSystemMemory)
static int64_t MinSystemMemory(int64_t available, int cc_major) {
    if (available < (1LL << 31)) return 225 * 1024 * 1024;
    // Heuristic based on Compute Capability
    if (cc_major <= 6) return 500 * 1024 * 1024;
    if (cc_major <= 7) return 1050 * 1024 * 1024;
    if (cc_major <= 8) return 1536 * 1024 * 1024;
    return 1800 * 1024 * 1024; 
}

// TensorFlow BFC Allocator Logic (Log2)
static int Log2Floor(uint64_t n) {
    if (n == 0) return -1;
    return 63 - __builtin_clzl(n);
}

// --------------------------------------------------------------------------------
// Pure BFC Implementation with Multi-GPU Support
// --------------------------------------------------------------------------------

// Static Instances Map
static std::vector<GPUCachingAllocator*> allocators;
static std::mutex instance_mutex;

GPUCachingAllocator* GPUCachingAllocator::instance(int device_id) {
    std::lock_guard<std::mutex> lock(instance_mutex);
    if (allocators.empty()) {
        int count;
        cudaGetDeviceCount(&count);
        allocators.resize(count, nullptr);
    }
    
    if (device_id < 0 || device_id >= allocators.size()) return nullptr;
    
    if (!allocators[device_id]) {
        allocators[device_id] = new GPUCachingAllocator(device_id);
    }
    return allocators[device_id];
}

GPUCachingAllocator::GPUCachingAllocator(int device_id) 
    : initialized_(false), next_allocation_size_(2 * 1024 * 1024), device_id_(device_id) {
    bins_.resize(kNumBins); 
}

GPUCachingAllocator::~GPUCachingAllocator() {
    // Cleanup chunks/blocks
    for(auto b : blocks_) {
        cudaFree(b->ptr);
        delete b;
    }
}

void GPUCachingAllocator::set_allow_growth(bool allow) {
    allow_growth_ = allow;
}

bool GPUCachingAllocator::get_allow_growth() {
    return allow_growth_;
}

void GPUCachingAllocator::set_garbage_collection(bool enable) {
    garbage_collection_ = enable;
}

bool GPUCachingAllocator::get_garbage_collection() {
    return garbage_collection_;
}

// New: Get Stats
AllocatorStats GPUCachingAllocator::get_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Calculate Largest Free Block + Fragmentation
    int64_t large_free = 0;
    
    // Iterate bins backwards (largest to smallest)
    for (int i = kNumBins - 1; i >= 0; i--) {
        if (!bins_[i].free_chunks.empty()) {
            // Free chunks are sorted by size (ChunkComparator). Last is largest.
             // Actually ChunkComparator sorts size ascending. So last element is largest.
             if (bins_[i].free_chunks.size() > 0) {
                  Chunk* largest = *bins_[i].free_chunks.rbegin();
                  large_free = std::max((size_t)large_free, largest->size);
                  // We found the largest possible because bins are sorted by size ranges.
                  break; 
             }
        }
    }
    stats_.largest_free_block_bytes = large_free;
    
    int64_t total_free = stats_.bytes_reserved - stats_.bytes_in_use;
    if (total_free > 0) {
        stats_.fragmentation = 1.0 - ((double)large_free / (double)total_free);
    } else {
        stats_.fragmentation = 0.0;
    }

    return stats_;
}

int GPUCachingAllocator::get_bin_index(size_t size) {
    // BFC Allocator Mapping
    if (size < 256) return 0;
    int idx = Log2Floor(size) - 8; 
    if (idx < 0) idx = 0;
    if (idx >= kNumBins) idx = kNumBins - 1;
    return idx;
}

void* GPUCachingAllocator::allocate(size_t bytes) {
    return allocate(bytes, 0);
}

void* GPUCachingAllocator::allocate(size_t bytes, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex_); // Thread Safe
    
    // Device Guard Logic
    int prev_device;
    cudaGetDevice(&prev_device);
    if (prev_device != device_id_) cudaSetDevice(device_id_);
    
    if (!initialized_) init_greedy_if_needed();

    // Alignment: 512 Bytes (kMinBlockSize)
    size_t aligned = kMinBlockSize * ((bytes + kMinBlockSize - 1) / kMinBlockSize);
    
    void* ptr = allocate_internal(aligned, stream);
    
    // Retry Logic (Backpedal like TensorFlow)
    if (!ptr) {
        stats_.num_alloc_retries++; // Retry 1
        
        // 1. Try Extension (Growth) (TF AllocateRawInternal logic)
        if (allow_growth_) {
             if (extend(aligned, stream)) {
                 ptr = allocate_internal(aligned, stream);
             }
        }
        
        // 2. Try Garbage Collection + Extension (TF Backpedal logic)
        if (!ptr && garbage_collection_) {
             stats_.num_alloc_retries++; // Retry 2
             garbage_collection();
             
             // After GC, try Extend again (TF logic)
             if (allow_growth_) {
                 if (extend(aligned, stream)) {
                     ptr = allocate_internal(aligned, stream);
                 }
             }
        }
        
        if (!ptr) {
            stats_.num_ooms++;
        }
    }
    
    if (ptr) {
        // Update Stats
        stats_.bytes_in_use += aligned; // Use aligned size as chunk size matches this (mostly)
        // Note: Real chunk size might be larger if we didn't split perfectly.
        // To be precise: We should get the actual chunk->size. 
        // But allocating logic modifies chunk. Let's rely on internal update or approximate here?
        // Better: allocate_internal sets the stats. Ideally.
        // Actually, allocate_internal returns void*. Getting Chunk* from it requires map lookup.
        // Let's do lookup to be accurate.
        auto it = ptr_to_chunk_map_.find(ptr);
        if (it != ptr_to_chunk_map_.end()) {
             size_t real_size = it->second->size;
             // Correction: we likely added aligned above.
             // Actually, I'll move stats update inside allocate_internal if possible, or do it here.
             // Reset calculation:
             stats_.bytes_in_use -= aligned; // Undo approximate
             stats_.bytes_in_use += real_size;
             stats_.max_bytes_in_use = std::max(stats_.max_bytes_in_use, stats_.bytes_in_use);
             stats_.max_alloc_size = std::max(stats_.max_alloc_size, (int64_t)real_size);
             stats_.num_allocs++;
        }
    }

    if (prev_device != device_id_) cudaSetDevice(prev_device); // Restore
    
    return ptr;
}

void* GPUCachingAllocator::allocate_internal(size_t size, cudaStream_t stream) {
    int bin_idx = get_bin_index(size);
    
    for (int i = bin_idx; i < kNumBins; ++i) {
        Bin& bin = bins_[i];
        if (bin.free_chunks.empty()) continue;

        // Optimized Search: O(log N) using lower_bound
        // We search for a "virtual" chunk of the requested size. 
        // Our comparator sorts by size, then ptr. Searching for {size, nullptr}
        // will find the first chunk with size >= requested.
        Chunk key_chunk(nullptr, size);
        auto it = bin.free_chunks.lower_bound(&key_chunk);

        // Iterate from lower_bound just in case (e.g. alignment constraints, though size fits)
        if (it != bin.free_chunks.end()) {
            Chunk* chunk = *it;
            // Best Fit Found (First valid chunk due to sorted set)
            if (chunk->size >= size) {
                // Remove from free list
                bin.free_chunks.erase(it);
                
                chunk->allocated = true;
                chunk->stream = stream; 
                
                // Split Logic
                if (chunk->size >= size + kMinBlockSize) {
                    split_chunk(chunk, size);
                }
                
                return chunk->ptr;
            }
        }
    }
    return nullptr;
}

void GPUCachingAllocator::garbage_collection() {
    // GC: Free totally unused blocks back to OS
    bool freed_any = false;
    for (auto it = blocks_.begin(); it != blocks_.end(); ) {
        Block* b = *it;
        auto chunk_it = ptr_to_chunk_map_.find(b->ptr);
        
        if (chunk_it != ptr_to_chunk_map_.end()) {
             Chunk* c = chunk_it->second;
             // If chunk matches block size and is free -> Entire block is free
             if (!c->allocated && c->size == b->size) {
                 // Stats Update
                 stats_.bytes_reserved -= b->size;
                 
                 remove_chunk(c);
                 ptr_to_chunk_map_.erase(c->ptr);
                 
                 delete c;
                 cudaFree(b->ptr);
                 delete b;
                 
                 stats_.segment_frees++; // Segment Free
                 
                 it = blocks_.erase(it);
                 freed_any = true;
                 continue;
             }
        }
        ++it;
    }
}

void GPUCachingAllocator::split_chunk(Chunk* chunk, size_t size) {
    // Create remaining chunk
    Chunk* remaining = new Chunk((char*)chunk->ptr + size, chunk->size - size);
    remaining->prev = chunk;
    remaining->next = chunk->next;
    
    if (chunk->next) chunk->next->prev = remaining;
    chunk->next = remaining;
    
    chunk->size = size;
    
    // Insert remaining into correct Bin
    ptr_to_chunk_map_[remaining->ptr] = remaining;
    insert_chunk(remaining);
}

bool GPUCachingAllocator::extend(size_t size, cudaStream_t stream) {
    // Doubling Strategy
    size_t alloc_size = next_allocation_size_;
    while (alloc_size < size) {
        alloc_size *= 2;
    }
    
    void* ptr;
    cudaError_t err = cudaMalloc(&ptr, alloc_size);
    if (err != cudaSuccess) {
        return false; 
    }
    
    // Create new Block -> Huge Chunk
    Block* b = new Block();
    b->ptr = ptr;
    b->size = alloc_size;
    blocks_.push_back(b);
    
    Chunk* chunk = new Chunk(ptr, alloc_size);
    chunk->bin_index = get_bin_index(alloc_size);
    ptr_to_chunk_map_[chunk->ptr] = chunk;
    
    insert_chunk(chunk);
    
    // Update next allocation size (Doubling logic)
    next_allocation_size_ = alloc_size * 2;
    
    // Stats Update
    stats_.bytes_reserved += alloc_size;
    stats_.max_bytes_reserved = std::max(stats_.max_bytes_reserved, stats_.bytes_reserved);
    stats_.segment_allocs++; // Segment Alloc
    
    return true;
}

void GPUCachingAllocator::deallocate(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_); // Thread Safe
    if (!ptr) return;
    
    auto it = ptr_to_chunk_map_.find(ptr);
    if (it == ptr_to_chunk_map_.end()) {
        std::cerr << "Deallocate: Invalid pointer or double free at " << ptr << std::endl;
        return;
    }
    
    Chunk* chunk = it->second;
    
    // Stats Update
    if (chunk->allocated) {
        stats_.bytes_in_use -= chunk->size;
    }
    
    chunk->allocated = false;
    
    merge_chunk(chunk);
}

void GPUCachingAllocator::merge_chunk(Chunk* chunk) {
    // Merge Next
    if (chunk->next && !chunk->next->allocated) {
        Chunk* next = chunk->next;
        remove_chunk(next);       
        chunk->size += next->size;
        chunk->next = next->next;
        if (chunk->next) chunk->next->prev = chunk;
        
        ptr_to_chunk_map_.erase(next->ptr);
        delete next;
    }
    
    // Merge Prev
    if (chunk->prev && !chunk->prev->allocated) {
        Chunk* prev = chunk->prev;
        remove_chunk(prev);
        prev->size += chunk->size; 
        prev->next = chunk->next;
        if (prev->next) prev->next->prev = prev;
        
        ptr_to_chunk_map_.erase(chunk->ptr);
        delete chunk;
        chunk = prev;
    }
    
    insert_chunk(chunk);
}

void GPUCachingAllocator::insert_chunk(Chunk* chunk) {
    chunk->allocated = false;
    int idx = get_bin_index(chunk->size);
    chunk->bin_index = idx;
    bins_[idx].free_chunks.insert(chunk);
}

void GPUCachingAllocator::remove_chunk(Chunk* chunk) {
    if (chunk->bin_index == -1) return;
    bins_[chunk->bin_index].free_chunks.erase(chunk);
}

void GPUCachingAllocator::init_greedy_if_needed() {
    initialized_ = true;
    if (allow_growth_) return;
    
    // Greedy Logic
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    stats_.bytes_limit = total; // Set Limit
    
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    size_t reserve = MinSystemMemory(free, prop.major);
    size_t alloc_size = (free > reserve) ? (free - reserve) : 0;
    
    if (alloc_size > 0) {
        void* ptr;
        cudaError_t err = cudaMalloc(&ptr, alloc_size);
        if (err == cudaSuccess) {
            Block* b = new Block{ptr, alloc_size};
            blocks_.push_back(b);
            
            Chunk* c = new Chunk(ptr, alloc_size);
            c->bin_index = get_bin_index(alloc_size);
            ptr_to_chunk_map_[ptr] = c;
            
            insert_chunk(c);
            
            // Stats Update
            stats_.bytes_reserved += alloc_size;
            stats_.max_bytes_reserved = std::max(stats_.max_bytes_reserved, stats_.bytes_reserved);
        } else {
             std::cerr << "Greedy Init Failed Allocating " << alloc_size << " bytes. Fallback to growth." << std::endl;
             allow_growth_ = true; // Fallback
        }
        stats_.segment_allocs++;
    }
}

// Wrappers
void GPUCachingAllocator::memset(void* ptr, int value, size_t bytes) { cudaMemset(ptr, value, bytes); }
void GPUCachingAllocator::memcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) { cudaMemcpy(dst, src, bytes, kind); }
void GPUCachingAllocator::memsetAsync(void* ptr, int value, size_t bytes, cudaStream_t stream) { cudaMemsetAsync(ptr, value, bytes, stream); }
void GPUCachingAllocator::memcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) { cudaMemcpyAsync(dst, src, bytes, kind, stream); }

} // namespace device
} // namespace OwnTensor
