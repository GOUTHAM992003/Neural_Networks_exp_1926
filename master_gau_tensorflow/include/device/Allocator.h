#pragma once
#include <cstddef>

namespace OwnTensor
{
    class Allocator 
    {
        public:
            virtual ~Allocator() = default;
            virtual void* allocate(size_t bytes) = 0;
            virtual void deallocate(void* ptr) = 0;
    };
}