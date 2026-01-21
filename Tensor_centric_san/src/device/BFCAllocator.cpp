#include "BFCAllocator.h"
#include "AllocStruct.h"

#include <cstddef>
#include <algorithm>



BFCAllocator::BFCAllocator(std::unique_ptr<SubAllocator> sub_allocator, size_t total_memory, const Options& opts)
  : opts_(ops), next_allocation_id_(1) {
  // set initial allocation size based on allow_growth
  if(opts.allow_growth){
    curr_region_allocation_bytes_ = std::min(total_memory, size_t{2 << 20});
  } else{
    curr_region_allocation_bytes_ = total_memory;
  }

  // initialize bins
  for(BinNum b = 0; b < kNumBins; ++b){
    size_t bin_size = BinNumToSize(b);
    new (BinFromIndex(b)) Bin(this, bin_size);
  }
}

void* BFCAllocator::AllocateRaw(size_t alignment, size_t num_bytes){
  absl::MutexLock lock(&mutex);

  // round up to minimum alignement
  size_t rounded_bytes = RoundedBytes(num_bytes);

  // Find appropriate bin and allocate
  Bin* bin = BinForSize(rounded_bytes);

  // allocation logic here

  return ptr;
}

void BFCAllocator::DeallocateRaw(void* ptr){
  absl::MutexLock lock(&mutex_);

  // find chunk for pointer
  chunkHandle h = region_manager_.get_handle(ptr);
  Chunk* c = chunkFromHandle(h);

  // Mark as free and try to coalesce
  c->allocation_id = -1;
  InsertFreeChunkIntoBin(TryToCoalesce(h, false));
}

chunkHandle BFCAllocator::TryToCoalesce(chunkHandle h, bool ignore_freed_at){
  Chunk* c = chunkFromHandle(h);

  // try to coalesce with previous chunk
  if(c->prev != kInvalidChunkHandle){
    Chunk* prev = chunkFromHandle(c->prev);
    if(!prev->in_use()){
      h = Coalesce(h, c->prev);
      c = chunkFromHanle(h);
    }
  }

  // try to coalesce with the next chunk
  if(c->next != kInvalidChunkHandle){
    Chunk* next = chunkFromHandle(c->next);
    if(!next->in_use()){
      h = Coalesce(h, c->next);
    }
  }
  return h;
}

