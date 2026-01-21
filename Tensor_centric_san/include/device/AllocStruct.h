/*
This file contains the declaration of the core data structures needed for the 
Best-fit with coalescing allocator similar to the TensorFlow one.
*/

#include <cstddef>
#include <stdint.h>
#include <set>

// represents memory blocks with metadata for tracking and linking
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

  bool in_use() const { return (allocation_id != -1); }
};

// organizes free chunks by size for efficient best fit lookup
struct Bin{
  size_t bin_size = 0; // minimum size for chunks in this bin
  std::set<chunkHandle, chunkComparator> free_cunks;
  Bin(Allocator* allocator, size_t bs);
  // custom comparator for sorting

};

// manages memory regions and provies pointer to chunk mapping
struct AllocateRegion{
  void* memBasePtr;
  size_t size; // don't know exactly what this is?
  
  // handles array maps memory location to chunk

  // supports region extension for growth

};