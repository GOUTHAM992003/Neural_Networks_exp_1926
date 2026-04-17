#pragma once

#include "device/Allocator.h"
#include "device/Block.h"
#include "device/BlockPool.h"
#include <cassert>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace OwnTensor {

// ---------------------------------------------------------------------------
// EventPool — per-device pool of reusable cudaEvent_t handles.
// Eliminates repeated cudaEventCreate / cudaEventDestroy driver calls
// in the cross-stream synchronization hot path.
// ---------------------------------------------------------------------------
class EventPool {
public:
  using Event =
      std::unique_ptr<cudaEvent_t, std::function<void(cudaEvent_t *)>>;

  // Intentionally leaked singleton — avoids static-destructor-order issues
  // when the CUDA context may already be torn down at process exit.
  static EventPool &instance() {
    static EventPool *pool = new EventPool();
    return *pool;
  }

  // Acquire an event for `device`. Pops from the per-device pool under a
  // lightweight mutex; only calls cudaEventCreateWithFlags when the pool is
  // empty (cold start). The returned Event's custom deleter pushes the handle
  // back into the pool instead of calling cudaEventDestroy.
  Event get(int device) {
    PerDevicePool &p = pools_[device];

    auto destructor = [this, device](cudaEvent_t *event) {
      std::lock_guard<std::mutex> g(pools_[device].mutex_);
      pools_[device].event_pool_.emplace_back(event);
    };

    {
      std::lock_guard<std::mutex> g(p.mutex_);
      if (!p.event_pool_.empty()) {
        auto *event = p.event_pool_.back().release();
        p.event_pool_.pop_back();
        return Event(event, destructor);
      }
    }

    // Pool empty — allocate a new event with timing disabled (cheaper).
    auto new_ptr = std::make_unique<cudaEvent_t>();
    cudaEventCreateWithFlags(new_ptr.get(), cudaEventDisableTiming);
    return Event(new_ptr.release(), destructor);
  }

  // Destroy all pooled events. Called only from
  // CachingCUDAAllocator::empty_cache() — this is the single code path
  // that actually invokes cudaEventDestroy.
  void empty_cache() {
    for (auto &p : pools_) {
      std::lock_guard<std::mutex> g(p.mutex_);
      for (auto &evt_ptr : p.event_pool_) {
        cudaEventDestroy(*evt_ptr);
      }
      p.event_pool_.clear();
    }
  }

private:
  struct PerDevicePool {
    alignas(64) std::mutex mutex_; // cache-line aligned to prevent false
                                   // sharing between per-device pools
    std::vector<std::unique_ptr<cudaEvent_t>> event_pool_;
  };

  static int query_device_count() {
    int n = 0;
    cudaGetDeviceCount(&n);
    return n;
  }

  // Construct vector with exact size — elements are default-constructed
  // in-place, avoiding std::vector::resize() which requires move semantics
  // (std::mutex is non-movable).
  EventPool() : pools_(query_device_count()) {}

  std::vector<PerDevicePool> pools_;
};

class CachingCUDAAllocator : public Allocator {
  friend class BlockPool;

public:
  static CachingCUDAAllocator &instance();

  void *allocate(size_t bytes) override;
  void deallocate(void *ptr) override;

  void *allocate(size_t bytes, cudaStream_t stream);
  void recordStream(void *ptr, cudaStream_t stream);
  Block *find_allocated_block(void *ptr);
  void empty_cache();

  void trim_to(size_t target_bytes);
  void trim_pool(BlockPool &pool, size_t target_bytes);

#ifdef ALLOC_OPT_CAP
  // OPT Task 1: Reserved memory cap -- triggers merge+trim before exceeding budget
  void set_max_reserved(size_t bytes) { max_reserved_bytes_ = bytes; }
  size_t get_max_reserved() const { return max_reserved_bytes_; }
#endif

#ifdef ALLOC_OPT_DEFRAG
  // OPT Task 2: Defragmentation sweep -- merges all adjacent free blocks
  void defragment();
#endif

  struct MemoryStats {
    // -- Active memory --
    // The actual block sizes currently held by live tensors. These are the
    // rounded-up sizes after size-class quantization, NOT the originally
    // requested sizes. E.g. a 3.1 MB request served by a 4 MB block
    // contributes 4 MB to active_current.
    // Computed as: pool.total_allocated - pool.total_cached (summed over pools).
    size_t active_current;
    size_t active_peak;

    // -- Allocated (requested) memory --
    // The sum of originally requested byte sizes for all live allocations,
    // BEFORE size-class rounding. This is always <= active_current because
    // blocks are rounded up to size classes. The gap (active - allocated)
    // represents internal fragmentation from size-class rounding.
    // Computed as: pool.total_active_requested (summed over pools).
    size_t allocated_current;
    size_t allocated_peak;

    // -- Reserved (cached) memory --
    // Total bytes held by the allocator from CUDA (via cudaMalloc), including
    // both active blocks and free blocks in the cache. This is what shows up
    // on nvidia-smi (plus ~152 MB CUDA context overhead).
    // Computed as: pool.total_allocated (summed over pools).
    // reserved = active + cached_free_blocks.
    size_t reserved_current;
    size_t reserved_peak;

    // -- Allocation counters --
    size_t num_allocs;       // Total allocate() calls from tensors
    size_t num_frees;        // Total deallocate() calls from tensors
    size_t num_cache_hits;   // allocate() served from free block cache
    size_t num_cache_misses; // allocate() required a new cudaMalloc

    // -- Split block tracking --
    size_t num_splits;           // A cached block was split to serve a smaller request
    size_t num_merges;           // Adjacent free blocks were merged back together
    size_t inactive_split_bytes; // Bytes in split-off remainder blocks (not in use)

    // -- Physical CUDA driver calls --
    // Actual cudaMallocAsync/cudaFreeAsync calls to the CUDA driver.
    // Much fewer than num_allocs/num_frees due to caching.
    size_t num_cuda_mallocs;
    size_t num_cuda_frees;

    // -- Cross-stream usage stats --
    size_t num_record_stream_calls; // recordStream() cross-stream tracking inserts
    size_t num_deferred_frees;      // Frees deferred until stream events complete

    // -- OOM and retry tracking --
    size_t num_ooms;          // cudaMalloc failures (before retry)
    size_t num_alloc_retries; // Successful allocations after OOM empty_cache retry

    // -- Pool-specific stats --
    // The allocator uses two pools: small (< kSmallSize) and large (>= kSmallSize).
    // "allocated" = total bytes held from CUDA in that pool (active + cached).
    // "cached" = free bytes in that pool available for reuse.
    size_t small_pool_allocated;
    size_t large_pool_allocated;
    size_t small_pool_cached;
    size_t large_pool_cached;

    // -- Derived metrics --

    // Cache hit rate: % of allocate() calls served from free block cache.
    double cache_hit_rate() const {
      return num_allocs > 0 ? 100.0 * num_cache_hits / num_allocs : 0.0;
    }

    // Internal fragmentation: % of active memory wasted by size-class rounding.
    // fragmentation = (active_current - allocated_current) / active_current.
    // E.g. 23% means 23% of memory held by tensors is padding from rounding.
    double fragmentation_ratio() const {
      if (active_current <= allocated_current || active_current == 0) {
        return 0.0;
      }
      return 100.0 * (double)(active_current - allocated_current) /
             active_current;
    }

    // Legacy compatibility aliases
    size_t allocated = 0; // = active_current
    size_t cached = 0;    // = reserved_current
    size_t peak = 0;      // = allocated_peak
  };

  MemoryStats get_stats(int device = -1) const;
  std::vector<size_t> get_stats_vector(int device) const;
  void print_memory_summary() const;
  BlockPool &get_pool(size_t size, int device);

private:
  CachingCUDAAllocator();
  ~CachingCUDAAllocator();

  struct DevicePools {
    BlockPool small_pool;
    BlockPool large_pool;
    std::unique_ptr<std::mutex>
        mtx; // Global lock for both pools on this device

    // Per-stream deque of (event, block*) pairs awaiting completion
    // One deque per stream avoids head-of-line blocking
    // EventPool::Event is a unique_ptr whose custom deleter returns the
    // cudaEvent_t to the EventPool instead of calling cudaEventDestroy.
    std::unordered_map<cudaStream_t,
                       std::deque<std::pair<EventPool::Event, Block *>>>
        cuda_events;

    DevicePools() : mtx(std::make_unique<std::mutex>()) {}
    DevicePools(DevicePools &&other) noexcept = default;
    DevicePools &operator=(DevicePools &&other) noexcept = default;
  };

  std::deque<DevicePools> device_pools_;

  Block *cuda_alloc(size_t size, int device, cudaStream_t stream);

  void cuda_free(Block *block);
  void cuda_free_locked(Block *block);

  Block *try_split(Block *block, size_t size);

  void ensure_stream_safety(Block *block, cudaStream_t target_stream);

  // Convert recorded_streams into CUDA events and enqueue for deferred free
  void insert_events(Block *block, int device);
  // Poll completed events and return blocks to pool (called from allocate)
  void process_events(DevicePools &pools);

#ifdef ALLOC_OPT_CAP
  // Default cap: 7248 MB (7400 MB GPU - 152 MB context overhead)
  size_t max_reserved_bytes_ = 7400ULL * 1024 * 1024;
  size_t num_cap_merges_ = 0;   // times cap triggered a merge sweep
  size_t num_cap_trims_ = 0;    // times cap triggered a trim
#endif

#ifdef ALLOC_OPT_DEFRAG
  size_t num_defrag_merges_ = 0; // merges performed by defragment()
#endif

  mutable std::mutex stats_mutex_;
  size_t total_allocs_ = 0;
  size_t total_frees_ = 0;
  size_t cache_hits_ = 0;
  size_t cache_misses_ = 0;
  size_t num_splits_ = 0;
  size_t num_merges_ = 0;
  size_t num_ooms_ = 0;
  size_t num_alloc_retries_ = 0;
  size_t num_cuda_mallocs_ = 0;
  size_t num_cuda_frees_ = 0;
  size_t num_record_stream_calls_ = 0;
  size_t num_deferred_frees_ = 0;
  size_t peak_active_ = 0;
  size_t peak_allocated_ = 0;
  size_t peak_reserved_ = 0;
};
} // namespace OwnTensor
