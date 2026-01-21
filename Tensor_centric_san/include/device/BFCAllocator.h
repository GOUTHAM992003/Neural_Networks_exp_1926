/*
This file contains the declarations for the BFC Allocator
*/
#pragma once;
#include <memory>

namespace OwnTensor{
  /// represents memory blocks with metadata for tracking and linking
  struct Chunk{
  public:
    size_t size = 0; // total buffer size
    size_t requested_size = 0; // what is actually requested
    int allocation_id = -1; // unique identifier (-1 when free) 
    void* ptr = nullptr; // the actual memory pointer
    chunkHandle prev;
    chunkHandle next;
    BinNum bin_num =  kInvalidBinNum; // which bin this chunk belongs to
    uint64_t freed_at_count; // (optional timestamp for thread safety)
    // function to check if the chunk is in use
    bool in_use() const { return (allocation_id != -1); }
  };

  class BFCAllocator{
  public:
    // Options structure
    struct Options{
      bool allow_growth = true;
      bool allow_retry_on_failure = true;
      bool garbage_collection = false;
      double fragmentation_fraction = 0;
    };
    // constructor
    BFCAllocator(std::unique_ptr<SubAllocator> sub_allocator, size_t total_memory, const Options& opts);
    // allocation method
    void* AllocateRaw(size_t alignment, size_t num_bytes);
    // deallocation method
    void DeallocateRaw(void* ptr);
    // coalescing logic
    chunkHandle TryToCoalesce(chunkHandle h, bool ignore_freed_at);
  private:
    Options opts_;
    int next_allocation_id_;
    size_t curr_region_allocation_bytes_;
  };
} // endof OwnTensor