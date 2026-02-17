#pragma once
#include "device/Allocator.h"
#include "device/Device.h"
#include "device/Pinned_CPU_Allocator.h"
namespace OwnTensor{
enum class Pinned_Flag{
    None, // Pageable (Regular)
    Default, // Pinned (cudaHostAllocDefault flag) 
    Mapped, // Pinned (cudaHostAllocMapped flag)
    Portable, //Pinned (cudaHostAllocPortable flag)
    WriteCombined //Pinned (cudaHostAllocWriteCombined flag)
};

    class AllocatorRegistry {
    public:
        static Allocator* get_allocator(Device device);
        static Allocator* get_cpu_allocator();
        static Allocator* get_cuda_allocator();
        static Allocator* get_pinned_cpu_allocator(Pinned_Flag pin_ten);
    };

}