#include "TensorLib.h"
#include "Allocator.h"

namespace OwnTensor{
  SubAllocator::SubAllocator(const std::vector<Visitor>& alloc_visitors, const std::vector<Visitor>& free_visitors)
    : alloc_visitors_(alloc_visitors), free_visitors_(free_visitors) {}

  void SubAllocator::VisitAlloc(void* ptr, int index, size_t num_bytes){
    for(const auto& v: alloc_visitors_){
      v(ptr, index, num_bytes);
    }
  }

  void SubAllocator::VisitFree(void* ptr, int index, size_t num_bytes){
    for(int i = free_visitors_.size() - 1; i >= 0; --i){
      free_visitors_[i](ptr, index, num_bytes);
    }
  }
} // Endof OwnTensor