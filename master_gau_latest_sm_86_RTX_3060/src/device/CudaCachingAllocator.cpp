#include "device/AllocationTracker.h"
#include "device/Block.h"
#include "device/CachingCudaAllocator.h"
#include "device/BlockPool.h"
#include "device/DeviceCore.h"
#include "device/SizeClass.h"
#include <algorithm>
#include <cstddef>
#include <iostream>

namespace OwnTensor {
CachingCUDAAllocator &CachingCUDAAllocator::instance() {
  static CachingCUDAAllocator allocator;
  return allocator;
}

CachingCUDAAllocator::CachingCUDAAllocator() {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  device_pools_.resize(device_count);
}

CachingCUDAAllocator::~CachingCUDAAllocator() {

  // Removed Cache Clear for Allocator Tests - caused errors while working in
  // tandem with libtorch cache clear

  // Intentionally skip empty_cache() here.
  // At process exit the CUDA context may already be torn down
  // by another library's atexit handler (e.g. libtorch).
  // The OS/driver reclaims all GPU memory on process exit.

  // int device_count = 0;
  // cudaError_t err = cudaGetDeviceCount(&device_count);
  // if (err == cudaSuccess && device_count > 0) {
  //     empty_cache();
  // }
  // empty_cache();
}

Block *CachingCUDAAllocator::find_allocated_block(void *ptr) {
  // Search all device pools for this pointer, locking each device
  // Lock is held per-device to protect the pool lookup
  for (int d = 0; d < (int)device_pools_.size(); ++d) {
    std::lock_guard<std::mutex> lock(*device_pools_[d].mtx);
    for (BlockPool *pool :
         {&device_pools_[d].small_pool, &device_pools_[d].large_pool}) {
      Block *block = pool->find_allocated_block(ptr);
      if (block) {
        return block;
      }
    }
  }
  return nullptr;
}

void CachingCUDAAllocator::recordStream(void *ptr, cudaStream_t stream) {
  // Must hold device lock for the ENTIRE operation (find + insert) to prevent:
  //   1. Use-after-free: another thread deallocating the block between find and insert
  //   2. Data race: concurrent writes to recorded_streams (unordered_set not thread-safe)
  //   3. Orphaned records: concurrent deallocate taking immediate-free path
  for (int d = 0; d < (int)device_pools_.size(); ++d) {
    std::lock_guard<std::mutex> lock(*device_pools_[d].mtx);
    for (BlockPool *pool :
         {&device_pools_[d].small_pool, &device_pools_[d].large_pool}) {
      Block *block = pool->find_allocated_block(ptr);
      if (block) {
        // Skip the allocation stream — already handled by the allocator
        if (block->stream == stream)
          return;
        block->recorded_streams.insert(stream);
        num_record_stream_calls_++;
        return;
      }
    }
  }
}

void *CachingCUDAAllocator::allocate(size_t bytes) {
  cudaStream_t stream = cuda::getCurrentStream();
  return allocate(bytes, stream);
}

void CachingCUDAAllocator::trim_to(size_t target_bytes) {
  for (DevicePools &pools : device_pools_) {
    std::lock_guard<std::mutex> lock(*pools.mtx);
    trim_pool(pools.small_pool, target_bytes / 2);
    trim_pool(pools.large_pool, target_bytes / 2);
  }
}

void CachingCUDAAllocator::trim_pool(BlockPool &pool, size_t target_bytes) {
  // ASSUMES LOCK IS HELD
  //
  // OLD IMPLEMENTATION (freed all blocks including split, needed unlink patch):
  // while (pool.total_cached > target_bytes && !pool.free_blocks.empty()) {
  //   auto it = std::prev(pool.free_blocks.end());
  //   Block *block = *it;
  //   pool.total_cached -= block->size;
  //   pool.free_blocks.erase(it);
  //   {
  //     std::lock_guard<std::mutex> s_lock(stats_mutex_);
  //     total_frees_++;
  //   }
  //   if (block->prev) block->prev->next = block->next;
  //   if (block->next) block->next->prev = block->prev;
  //   cudaStreamSynchronize(block->stream);
  //   cudaFree(block->ptr);
  //   num_cuda_frees_++;
  //   assert(block != nullptr);
  //   delete block;
  // }

  // NEW IMPLEMENTATION — only free root blocks (prev==null && next==null).
  // Split blocks are skipped because their physical CUDA memory can't be
  // partially freed. Search backward from largest for root blocks.
  while (pool.total_cached > target_bytes && !pool.free_blocks.empty()) {

    // Search backward from largest block to find a root block
    Block *block = nullptr;
    auto it = pool.free_blocks.end();
    while (it != pool.free_blocks.begin()) {
      --it;
      if ((*it)->prev == nullptr && (*it)->next == nullptr) {
        block = *it;
        break;
      }
    }

    if (!block) {
      // No root blocks left — only split blocks remain, nothing can be freed
      break;
    }

    pool.total_cached -= block->size;
    pool.free_blocks.erase(it);

    {
      std::lock_guard<std::mutex> s_lock(stats_mutex_);
      total_frees_++;
    }

    // PHYSICAL FREE — root block, safe to cudaFree
    // GPU driver safely handles async frees. No synchronization needed.
    // PyTorch approach: cudaFree can be called without blocking.
    // GPU queues the free and processes it when it's safe.
    cudaFree(block->ptr);
    num_cuda_frees_++;
    pool.total_allocated -= block->size;

    assert(block != nullptr);
    delete block;
  }
}

void *CachingCUDAAllocator::allocate(size_t bytes, cudaStream_t stream) {
  if (bytes == 0)
    return nullptr;

  int device;
  cudaGetDevice(&device);

  size_t alloc_size = SizeClass::round_size(bytes);

  BlockPool &preferred_pool = get_pool(alloc_size, device);
  BlockPool *actual_pool_ptr = &preferred_pool;

  Block *block = nullptr;
  {
    std::lock_guard<std::mutex> lock(*device_pools_[device].mtx);

    // Poll completed events and return deferred blocks to pool
    process_events(device_pools_[device]);

    // Try preferred pool first
    block = preferred_pool.find_free_block(alloc_size, stream);

    // If missed in small pool, try large pool as fallback
    if (!block && SizeClass::is_small(alloc_size)) {
      block =
          device_pools_[device].large_pool.find_free_block(alloc_size, stream);
      if (block) {
        // Pull from large pool!
        actual_pool_ptr = &device_pools_[device].large_pool;
      }
    }

    if (block) {
      {
        std::lock_guard<std::mutex> s_lock(stats_mutex_);
        cache_hits_ += 1;
      }
      actual_pool_ptr->total_cached -= block->size;

      if (block->size >= alloc_size + SizeClass::kSmallSize) {
        block = try_split(block, alloc_size);
      }
    }
  }

  if (block) {
    ensure_stream_safety(block, stream);
  }

  // Cache miss - need fresh CUDA allocation
  if (!block) {
    cache_misses_++;

#ifdef ALLOC_OPT_CAP
    // ---- OPT Task 1: Reserved memory cap ----
    // Before calling cudaMalloc, check if adding alloc_size would exceed the
    // budget. If so, try a merge sweep to reclaim cached blocks, then trim
    // root blocks to free physical CUDA memory back to the driver.
    {
      size_t current_reserved = 0;
      {
        std::lock_guard<std::mutex> lock(*device_pools_[device].mtx);
        for (BlockPool *pool : {&device_pools_[device].small_pool,
                                &device_pools_[device].large_pool}) {
          current_reserved += pool->total_allocated;
        }
      }

      if (current_reserved + alloc_size > max_reserved_bytes_) {
        // Phase A: Trim root blocks to bring reserved under the cap.
        // This cudaFree's physical memory so the new cudaMalloc stays
        // within budget.
        size_t need_to_free = (current_reserved + alloc_size) - max_reserved_bytes_;
        {
          std::lock_guard<std::mutex> lock(*device_pools_[device].mtx);

          // First do a merge sweep so fragments coalesce into root blocks
          for (BlockPool *pool : {&device_pools_[device].small_pool,
                                  &device_pools_[device].large_pool}) {
            std::vector<Block *> to_merge(pool->free_blocks.begin(),
                                          pool->free_blocks.end());
            pool->free_blocks.clear();
            pool->total_cached = 0;

            for (Block *b : to_merge) {
              b = pool->try_block_merge(b);
              BlockPool &final_pool = get_pool(b->size, device);
              final_pool.free_blocks.insert(b);
              final_pool.total_cached += b->size;
            }
          }
          num_cap_merges_++;

          // Now trim root blocks to free physical memory
          size_t target_large = device_pools_[device].large_pool.total_cached > need_to_free
              ? device_pools_[device].large_pool.total_cached - need_to_free
              : 0;
          trim_pool(device_pools_[device].large_pool, target_large);

          // If large pool trim wasn't enough, trim small pool too
          size_t freed_so_far = current_reserved - 0;
          size_t new_reserved = 0;
          for (BlockPool *pool : {&device_pools_[device].small_pool,
                                  &device_pools_[device].large_pool}) {
            new_reserved += pool->total_allocated;
          }
          if (new_reserved + alloc_size > max_reserved_bytes_) {
            size_t still_need = (new_reserved + alloc_size) - max_reserved_bytes_;
            size_t target_small = device_pools_[device].small_pool.total_cached > still_need
                ? device_pools_[device].small_pool.total_cached - still_need
                : 0;
            trim_pool(device_pools_[device].small_pool, target_small);
          }
          num_cap_trims_++;
        }
      }
    }
#endif // ALLOC_OPT_CAP

    block = cuda_alloc(alloc_size, device, stream);
    if (!block) {
      // Log to system stderr directly to bypass potential stream hangs
      fprintf(stderr, "[ALLOCATOR] Hard OOM. Trimming cache...\n");
      num_ooms_++;

      this->trim_to(0);

      block = cuda_alloc(alloc_size, device, stream);
      if (!block) {
        fprintf(stderr, "[ALLOCATOR] FATAL: Still OOM after trim.\n");
        throw std::runtime_error("CUDA OOM");
      }
      num_alloc_retries_++;
    }
  }

  block->allocated = true;
  block->req_size = bytes;
  block->stream = stream;

  size_t current_active = 0;
  size_t current_reserved = 0;
  {
    std::lock_guard<std::mutex> lock(*device_pools_[device].mtx);
    actual_pool_ptr->allocated_blocks[block->ptr] = block;
    actual_pool_ptr->total_active_requested += bytes;
    for (BlockPool *pool : {&device_pools_[device].small_pool,
                            &device_pools_[device].large_pool}) {
      current_active += pool->total_allocated - pool->total_cached;
      current_reserved += pool->total_allocated;
    }
  }

  AllocationTracker::instance().on_alloc(block->ptr, bytes, device);

  {
    std::lock_guard<std::mutex> s_lock(stats_mutex_);
    total_allocs_++;
    peak_active_ = std::max(peak_active_, current_active);
    peak_allocated_ = std::max(peak_allocated_, current_reserved);
    peak_reserved_ = std::max(peak_reserved_, current_reserved);
  }
  return block->ptr;
}

void CachingCUDAAllocator::deallocate(void *ptr) {
  if (!ptr)
    return;

  // Search all device pools for this pointer
  for (int d = 0; d < (int)device_pools_.size(); ++d) {
    std::lock_guard<std::mutex> lock(*device_pools_[d].mtx);
    for (BlockPool *pool :
         {&device_pools_[d].small_pool, &device_pools_[d].large_pool}) {
      auto it = pool->allocated_blocks.find(ptr);
      if (it != pool->allocated_blocks.end()) {
        Block *block = it->second;
        pool->total_active_requested -= block->req_size;
        pool->allocated_blocks.erase(it);

        // Set device context before any CUDA operations related to this block
        device::set_cuda_device(d);
        AllocationTracker::instance().on_free(ptr, d);

        block->allocated = false;

        if (!block->recorded_streams.empty()) {
          // Cross-stream usage — defer freeing until events complete
          insert_events(block, d);
          num_deferred_frees_++;
        } else {
          // Normal path: merge and return to pool immediately
          block = pool->try_block_merge(block);

          BlockPool &final_pool = get_pool(block->size, d);
          final_pool.free_blocks.insert(block);
          final_pool.total_cached += block->size;
        }

        total_frees_++;
        return; // Found and freed!
      }
    }
  }

  // If we get here, the pointer wasn't found in any pool.
  // It could be from a third-party library or an application bug.
  // std::cerr << "Warning: deallocate called on unknown pointer " << ptr <<
  // std::endl;
}

Block *CachingCUDAAllocator::cuda_alloc(size_t size, int device,
                                        cudaStream_t stream) {
  void *ptr = nullptr;
  cudaError_t err = cudaMallocAsync(&ptr, size, stream);
  if (err != cudaSuccess || !ptr) {
    fprintf(stderr, "[ALLOCATOR] cudaMallocAsync(size=%zu) failed: %s\n", size,
            cudaGetErrorString(err));
    return nullptr;
  }

  Block *block = new Block(ptr, size, device, stream);
  num_cuda_mallocs_++;

  BlockPool &pool = get_pool(size, device);
  {
    std::lock_guard<std::mutex> lock(*device_pools_[device].mtx);
    pool.total_allocated += size;
    pool.peak_allocated = std::max(pool.peak_allocated, pool.total_allocated);
  }
  return block;
}

void CachingCUDAAllocator::cuda_free(Block *block) {
  cudaFreeAsync(block->ptr, block->stream);
  num_cuda_frees_++;
  BlockPool &pool = get_pool(block->size, block->device_id);
  {
    std::lock_guard<std::mutex> lock(*device_pools_[block->device_id].mtx);
    pool.total_allocated -= block->size;
  }

  delete block;
}

void CachingCUDAAllocator::cuda_free_locked(Block *block) {
  // ASSUMES LOCK IS HELD
  //
  // OLD IMPLEMENTATION (deleted both split and root blocks, needed unlink patch):
  // if (block->prev) block->prev->next = block->next;
  // if (block->next) block->next->prev = block->prev;
  // if (block->prev == nullptr && block->next == nullptr) {
  //   cudaFreeAsync(block->ptr, block->stream);
  //   num_cuda_frees_++;
  //   BlockPool &pool = get_pool(block->size, block->device_id);
  //   pool.total_allocated -= block->size;
  // }
  // delete block;

  // NEW IMPLEMENTATION — only called on root blocks (fully merged, no neighbors).
  // Split blocks are never passed here; they stay in the pool.
  assert(block->prev == nullptr && block->next == nullptr &&
         "cuda_free_locked called on a split block — only root blocks can be freed");

  cudaFreeAsync(block->ptr, block->stream);
  num_cuda_frees_++;
  BlockPool &pool = get_pool(block->size, block->device_id);
  pool.total_allocated -= block->size;

  delete block;
}

BlockPool &CachingCUDAAllocator::get_pool(size_t size, int device) {
  if (SizeClass::is_small(size)) {
    return device_pools_[device].small_pool;
  } else {
    return device_pools_[device].large_pool;
  }
}

void CachingCUDAAllocator::empty_cache() {
  // =====================================================================
  // OLD IMPLEMENTATION (commented out — deleted ALL free blocks including
  // split blocks, causing misleading stats and previously segfaults)
  // =====================================================================
  // std::vector<Block*> blocks_to_free;
  // for (DevicePools& dev_pools : device_pools_)
  // {
  //      std::lock_guard<std::mutex> lock(*dev_pools.mtx);
  //      for (BlockPool* pool : { &dev_pools.small_pool, &dev_pools.large_pool
  //      })
  //      {
  //         for (Block* block : pool->free_blocks)
  //         {
  //             blocks_to_free.push_back(block);
  //         }
  //         pool->free_blocks.clear();
  //         pool->total_cached = 0;
  //      }
  // }
  // for (Block* block : blocks_to_free)
  // {
  //     if (!block || !block->ptr) continue;
  //     cuda_free_locked(block);
  //     block->ptr = nullptr;
  // }
  // cudaDeviceSynchronize();
  //
  // --- V2: also deleted split blocks via unique_blocks_to_free set ---
  // std::set<Block *> unique_blocks_to_free;
  // for (DevicePools &dev_pools : device_pools_) {
  //   std::lock_guard<std::mutex> lock(*dev_pools.mtx);
  //   for (auto &kv : dev_pools.cuda_events) {
  //     for (auto &pair : kv.second) {
  //       pair.second->event_count--;
  //       if (pair.second->event_count == 0) {
  //         unique_blocks_to_free.insert(pair.second);
  //       }
  //     }
  //   }
  //   dev_pools.cuda_events.clear();
  //   for (BlockPool *pool : {&dev_pools.small_pool, &dev_pools.large_pool}) {
  //     for (Block *block : pool->free_blocks) {
  //       unique_blocks_to_free.insert(block);
  //     }
  //     pool->free_blocks.clear();
  //     pool->total_cached = 0;
  //   }
  // }
  // for (Block *block : unique_blocks_to_free) {
  //   cuda_free_locked(block);
  // }
  // cudaDeviceSynchronize();
  // EventPool::instance().empty_cache();

  // =====================================================================
  // NEW IMPLEMENTATION — PyTorch-style two-phase: drain events, then
  // release only root blocks (prev==null && next==null).
  // Split blocks stay in the pool — their physical CUDA memory can't be
  // partially freed anyway, and they remain reusable for future allocations.
  // =====================================================================

  for (DevicePools &dev_pools : device_pools_) {
    std::lock_guard<std::mutex> lock(*dev_pools.mtx);

    // ----------------------------------------------------------------
    // PHASE 1: Drain all deferred events, merge blocks back into pool.
    // Uses cudaEventSynchronize (blocking) to force all events to
    // complete, then runs the normal merge path so that freed blocks
    // coalesce with neighbors — maximizing root blocks for phase 2.
    // ----------------------------------------------------------------
    for (auto stream_it = dev_pools.cuda_events.begin();
         stream_it != dev_pools.cuda_events.end();) {
      auto &events = stream_it->second;
      while (!events.empty()) {
        // Move event out; unique_ptr destructor returns it to EventPool
        EventPool::Event event = std::move(events.front().first);
        Block *block = events.front().second;
        events.pop_front();

        // Force-wait for this event to complete
        cudaEventSynchronize(*event);

        block->event_count--;
        if (block->event_count == 0) {
          // Normal merge path — same as deallocate() and process_events()
          BlockPool &merge_pool = get_pool(block->size, block->device_id);
          block = merge_pool.try_block_merge(block);

          BlockPool &final_pool = get_pool(block->size, block->device_id);
          final_pool.free_blocks.insert(block);
          final_pool.total_cached += block->size;
        }
      }
      stream_it = dev_pools.cuda_events.erase(stream_it);
    }

    // ----------------------------------------------------------------
    // PHASE 2: Release only root blocks (fully merged, no split neighbors).
    // Split free blocks are left in the pool — they cannot be cudaFree'd
    // because they share a physical cudaMalloc'd region with allocated
    // neighbors. Leaving them preserves correct stats and cache reuse.
    // ----------------------------------------------------------------
    for (BlockPool *pool : {&dev_pools.small_pool, &dev_pools.large_pool}) {
      auto it = pool->free_blocks.begin();
      while (it != pool->free_blocks.end()) {
        Block *block = *it;
        if (block->prev == nullptr && block->next == nullptr) {
          // Root block — safe to cudaFree the entire physical region
          it = pool->free_blocks.erase(it);
          pool->total_cached -= block->size;

          
          cudaFree(block->ptr);
          num_cuda_frees_++;
          pool->total_allocated -= block->size;

          delete block;
        } else {
          // Split block — skip, leave in pool
          ++it;
        }
      }
    }
  }

  cudaDeviceSynchronize();

  // Destroy all pooled events — this is the ONLY code path that calls
  // cudaEventDestroy, matching PyTorch's design.
  EventPool::instance().empty_cache();
}

void CachingCUDAAllocator::ensure_stream_safety(Block *block,
                                                cudaStream_t target_stream) {
  if (block->stream == target_stream) {
    return;
  }

  if (block->stream == 0 || target_stream == 0) {
    cudaStreamSynchronize(block->stream);
  } else {
    EventPool::Event event = EventPool::instance().get(block->device_id);
    cudaEventRecord(*event, block->stream);
    cudaStreamWaitEvent(target_stream, *event, 0);
    // event returns to pool when unique_ptr goes out of scope
  }

  block->stream = target_stream;
}

void CachingCUDAAllocator::insert_events(Block *block, int device) {
  // ASSUMES DEVICE LOCK IS HELD
  // Convert recorded_streams into CUDA events and enqueue for deferred free
  DevicePools &pools = device_pools_[device];
  for (cudaStream_t s : block->recorded_streams) {
    EventPool::Event evt = EventPool::instance().get(device);
    cudaEventRecord(*evt, s);
    block->event_count++;
    pools.cuda_events[s].emplace_back(std::move(evt), block);
  }
  block->recorded_streams.clear();
}

void CachingCUDAAllocator::process_events(DevicePools &pools) {
  // ASSUMES DEVICE LOCK IS HELD
  // Poll per-stream event deques — non-blocking check via cudaEventQuery
  for (auto &kv : pools.cuda_events) {
    std::deque<std::pair<EventPool::Event, Block *>> &events = kv.second;
    while (!events.empty()) {
      cudaEvent_t evt = *events.front().first;
      Block *block = events.front().second;

      cudaError_t err = cudaEventQuery(evt);
      if (err == cudaErrorNotReady) {
        // Not done — skip rest of this stream's deque (preserves ordering)
        cudaGetLastError(); // clear the error state
        break;
      }

      // Event complete — pop_front() triggers the custom deleter on the
      // EventPool::Event unique_ptr, returning the cudaEvent_t to the
      // EventPool instead of calling cudaEventDestroy.
      block->event_count--;
      events.pop_front();

      if (block->event_count == 0) {
        // All events drained — block is safe to return to pool
        BlockPool &merge_pool = get_pool(block->size, block->device_id);
        block = merge_pool.try_block_merge(block);

        BlockPool &final_pool = get_pool(block->size, block->device_id);
        final_pool.free_blocks.insert(block);
        final_pool.total_cached += block->size;
      }
    }
  }
}

// Need to ensure both the blocks after split is the same stream and event
Block *CachingCUDAAllocator::try_split(Block *block, size_t size) {
  size_t remaining = block->size - size;

  if (remaining < SizeClass::kSmallSize) {
    return block; // too small to split }
  }

  num_splits_++; // Track split operation

  void *new_ptr = static_cast<char *>(block->ptr) + size;
  Block *new_block =
      new Block(new_ptr, remaining, block->device_id, block->stream);
  new_block->is_split = true;

  block->size = size;
  block->is_split = true;

  new_block->prev = block;
  new_block->next = block->next;
  if (block->next) {
    block->next->prev = new_block;
  }
  block->next = new_block;

  BlockPool &pool = get_pool(remaining, block->device_id);
  // Already holding device lock from allocate()
  pool.free_blocks.insert(new_block);
  pool.total_cached += remaining;

  return block;
}

// } while (it != pool.blocks.end() && (*it)->expandable_segment_ &&
//              (*it)->stream == p.stream());
//     if (it == pool.blocks.end() || (*it)->stream != p.stream()) {
//       return false;
// }

// Stream-aware: only return a free block that lives on the SAME stream
Block *BlockPool::find_free_block(size_t size, cudaStream_t stream) {
  Block search_key(nullptr, size, 0, nullptr);
  auto it = free_blocks.lower_bound(&search_key);

  // Walk forward from the lower_bound looking for a same-stream match
  while (it != free_blocks.end()) {
    Block *candidate = *it;
    if (candidate->stream == stream) {
      free_blocks.erase(it);
      return candidate;
    }
    ++it;
  }
  return nullptr;
}

Block *BlockPool::find_allocated_block(void *ptr) {
  auto it = allocated_blocks.find(ptr);
  if (it != allocated_blocks.end()) {
    return it->second;
  }
  return nullptr;
}

void BlockPool::return_block(Block *block) {
  free_blocks.insert(block);
  total_cached += block->size;
}

// size_t try_merge_blocks(Block* dst, Block* src, BlockPool& pool) {
// if (!src || src->allocated || src->event_count > 0 ||
//     !src->stream_uses.empty() || dst->mapped != src->mapped) {
//   return 0;
// }

Block *BlockPool::try_block_merge(Block *block) {
  // This is called from CachingCUDAAllocator with the device lock held.
  CachingCUDAAllocator &alloc = CachingCUDAAllocator::instance();

  auto remove_block = [&](Block *b) {
    bool removed = false;
    for (BlockPool *p : {&alloc.device_pools_[b->device_id].small_pool,
                         &alloc.device_pools_[b->device_id].large_pool}) {
      auto it = p->free_blocks.find(b);
      if (it != p->free_blocks.end()) {
        p->free_blocks.erase(it);
        p->total_cached -= b->size;
        removed = true;
        break;
      }
    }
    return removed;
  };

  // Only merge with prev if it is on the SAME stream and has no pending
  // cross-stream work (recorded_streams drained, no outstanding events)
  if (block->prev && !block->prev->allocated &&
      block->prev->stream == block->stream &&
      block->prev->recorded_streams.empty() &&
      block->prev->event_count == 0) {
    Block *prev = block->prev;
    BlockPool &prev_pool = alloc.get_pool(prev->size, prev->device_id);

    prev_pool.free_blocks.erase(prev);
    prev_pool.total_cached -= prev->size;

    prev->size += block->size;
    prev->next = block->next;
    if (block->next) {
      block->next->prev = prev;
    }

    delete block;
    block = prev;
    alloc.num_merges_++; // Track merge operation
  }

  // Only merge with next if it is on the SAME stream and has no pending
  // cross-stream work (recorded_streams drained, no outstanding events)
  if (block->next && !block->next->allocated &&
      block->next->stream == block->stream &&
      block->next->recorded_streams.empty() &&
      block->next->event_count == 0) {
    Block *next = block->next;
    if (remove_block(next)) {
      block->size += next->size;
      block->next = next->next;
      if (next->next) {
        next->next->prev = block;
      }
      delete next;
      alloc.num_merges_++; // Track merge operation
    }
  }

  assert(block != nullptr);
  return block;
}

CachingCUDAAllocator::MemoryStats
CachingCUDAAllocator::get_stats(int device) const {
  MemoryStats stats = {};

  auto add_pool_stats = [&](const BlockPool &pool, bool is_small) {
    size_t pool_active = pool.total_allocated - pool.total_cached;

    stats.active_current += pool_active;
    stats.allocated_current += pool.total_active_requested;
    stats.reserved_current += pool.total_allocated;

    if (is_small) {
      stats.small_pool_allocated += pool.total_allocated;
      stats.small_pool_cached += pool.total_cached;
    } else {
      stats.large_pool_allocated += pool.total_allocated;
      stats.large_pool_cached += pool.total_cached;
    }
  };

  if (device < 0) {
    for (const DevicePools &dev_pools : device_pools_) {
      std::lock_guard<std::mutex> lock(*dev_pools.mtx);
      add_pool_stats(dev_pools.small_pool, true);
      add_pool_stats(dev_pools.large_pool, false);
    }
  } else {
    if (device < (int)device_pools_.size()) {
      const DevicePools &dev_pools = device_pools_[device];
      std::lock_guard<std::mutex> lock(*dev_pools.mtx);
      add_pool_stats(dev_pools.small_pool, true);
      add_pool_stats(dev_pools.large_pool, false);
    }
  }

  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats.num_allocs = total_allocs_;
    stats.num_frees = total_frees_;
    stats.num_cache_hits = cache_hits_;
    stats.num_cache_misses = cache_misses_;
    stats.num_splits = num_splits_;
    stats.num_merges = num_merges_;
    stats.num_cuda_mallocs = num_cuda_mallocs_;
    stats.num_cuda_frees = num_cuda_frees_;
    stats.num_record_stream_calls = num_record_stream_calls_;
    stats.num_deferred_frees = num_deferred_frees_;
    stats.num_ooms = num_ooms_;
    stats.num_alloc_retries = num_alloc_retries_;
    stats.active_peak = peak_active_;
    stats.allocated_peak = peak_active_;
    stats.reserved_peak = peak_reserved_;
  }

  // Legacy compatibility
  stats.allocated = stats.active_current;
  stats.cached = stats.reserved_current;
  stats.peak = stats.allocated_peak;

  return stats;
}

void CachingCUDAAllocator::print_memory_summary() const {
  MemoryStats stats = get_stats();

  auto mb = [](size_t bytes) { return bytes / 1024.0 / 1024.0; };

  std::cerr << "\n==================== OwnTensor - CUDA Caching Allocator "
               "Stats ====================\n";

  std::cerr << "\n--- Memory Usage ---\n";
  std::cerr << "  Active (in use):     " << mb(stats.active_current)
            << " MB (peak: " << mb(stats.active_peak) << " MB)\n";
  std::cerr << "  Allocated (CUDA):    " << mb(stats.allocated_current)
            << " MB (peak: " << mb(stats.allocated_peak) << " MB)\n";
  std::cerr << "  Reserved (cached):   " << mb(stats.reserved_current)
            << " MB (peak: " << mb(stats.reserved_peak) << " MB)\n";

  std::cerr << "\n--- Pool Breakdown ---\n";
  std::cerr << "  Small pool:          " << mb(stats.small_pool_allocated)
            << " MB allocated, " << mb(stats.small_pool_cached)
            << " MB cached\n";
  std::cerr << "  Large pool:          " << mb(stats.large_pool_allocated)
            << " MB allocated, " << mb(stats.large_pool_cached)
            << " MB cached\n";

  std::cerr << "\n--- Allocation Stats ---\n";
  std::cerr << "  Total allocations:   " << stats.num_allocs << "\n";
  std::cerr << "  Total frees:         " << stats.num_frees << "\n";
  std::cerr << "  Cache hits:          " << stats.num_cache_hits << " ("
            << stats.cache_hit_rate() << "%)\n";
  std::cerr << "  Cache misses:        " << stats.num_cache_misses << "\n";

  std::cerr << "\n--- CUDA Driver Calls ---\n";
  std::cerr << "  Malloc calls:        " << stats.num_cuda_mallocs << "\n";
  std::cerr << "  Free calls:          " << stats.num_cuda_frees << "\n";

  std::cerr << "\n--- Cross-Stream Usage ---\n";
  std::cerr << "  recordStream calls:  " << stats.num_record_stream_calls << "\n";
  std::cerr << "  Deferred frees:      " << stats.num_deferred_frees << "\n";

  std::cerr << "\n--- Block Operations ---\n";
  std::cerr << "  Block splits:        " << stats.num_splits << "\n";
  std::cerr << "  Block merges:        " << stats.num_merges << "\n";

  std::cerr << "\n--- OOM Recovery ---\n";
  std::cerr << "  OOM events:          " << stats.num_ooms << "\n";
  std::cerr << "  Successful retries:  " << stats.num_alloc_retries << "\n";

  std::cerr << "\n--- Derived Metrics ---\n";
  std::cerr << "  Fragmentation:       " << stats.fragmentation_ratio()
            << "%\n";

#ifdef ALLOC_OPT_CAP
  std::cerr << "\n--- OPT: Reserved Cap (" << (max_reserved_bytes_ / 1024 / 1024)
            << " MB) ---\n";
  std::cerr << "  Cap merge sweeps:    " << num_cap_merges_ << "\n";
  std::cerr << "  Cap trim triggers:   " << num_cap_trims_ << "\n";
#endif

#ifdef ALLOC_OPT_DEFRAG
  std::cerr << "\n--- OPT: Defragmentation ---\n";
  std::cerr << "  Defrag merges:       " << num_defrag_merges_ << "\n";
#endif

  std::cerr << "==============================================================="
               "========\n\n";
}

std::vector<size_t> CachingCUDAAllocator::get_stats_vector(int device) const {
  MemoryStats stats = get_stats(device);
  std::vector<size_t> stats_vector = {};
  stats_vector.emplace_back(stats.num_allocs);
  stats_vector.emplace_back(stats.num_frees);
  stats_vector.emplace_back(stats.num_ooms);
  stats_vector.emplace_back(stats.num_alloc_retries);
  stats_vector.emplace_back(stats.active_peak);
  stats_vector.emplace_back(stats.active_current);
  stats_vector.emplace_back(stats.allocated_peak);
  stats_vector.emplace_back(stats.allocated_current);

  return stats_vector;
}

#ifdef ALLOC_OPT_DEFRAG
void CachingCUDAAllocator::defragment() {
  // OPT Task 2: Walk every free block in every pool and merge adjacent
  // free blocks that the normal merge-on-free path missed (e.g. because
  // they were freed in a different order or on different streams that
  // have since completed).
  for (int d = 0; d < (int)device_pools_.size(); ++d) {
    std::lock_guard<std::mutex> lock(*device_pools_[d].mtx);

    // First drain any pending events so deferred blocks become mergeable
    process_events(device_pools_[d]);

    for (BlockPool *pool :
         {&device_pools_[d].small_pool, &device_pools_[d].large_pool}) {
      // Snapshot the current free set -- merging will invalidate iterators
      std::vector<Block *> free_snap(pool->free_blocks.begin(),
                                     pool->free_blocks.end());
      pool->free_blocks.clear();
      size_t old_cached = pool->total_cached;
      pool->total_cached = 0;

      for (Block *b : free_snap) {
        size_t old_size = b->size;
        b = pool->try_block_merge(b);
        if (b->size != old_size) {
          num_defrag_merges_++;
        }
        // Re-insert into the correct pool (size may have changed after merge)
        BlockPool &final_pool = get_pool(b->size, d);
        final_pool.free_blocks.insert(b);
        final_pool.total_cached += b->size;
      }
    }
  }
}
#endif // ALLOC_OPT_DEFRAG

} // namespace OwnTensor
