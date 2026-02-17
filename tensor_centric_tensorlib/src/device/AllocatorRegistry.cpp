#include "device/AllocatorRegistry.h"
#include "device/CPUAllocator.h"
#include "device/GPUCachingAllocator.h"
#include "device/Pinned_CPU_Allocator.h"
#include <cuda_runtime.h>
#include <iostream>

namespace OwnTensor
{ 
    namespace {
        CPUAllocator cpu_allocator;
        
        // Dispatcher Allocator for Multi-GPU Support
        class DispatcherCUDAAllocator : public Allocator {
        public:
            void* allocate(size_t bytes) override {
                // Allocation goes to the CURRENTLY ACTIVE device
                int dev;
                cudaGetDevice(&dev);
                return device::GPUCachingAllocator::instance(dev)->allocate(bytes);
            }
            
            void deallocate(void* ptr) override {
                if (!ptr) return;
                // Find which device owns this pointer
                cudaPointerAttributes attr;
                cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
                if (err == cudaSuccess && attr.type == cudaMemoryTypeDevice) {
                     device::GPUCachingAllocator::instance(attr.device)->deallocate(ptr);
                } else {
                     // Pointer is not valid device memory (e.g. CPU or Invalid). Ignore to prevent crash.
                }
            }
            
            void memset(void* ptr, int value, size_t bytes) override {
                cudaMemset(ptr, value, bytes);
            }
            void memcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) override {
                cudaMemcpy(dst, src, bytes, kind);
            }
            void memsetAsync(void* ptr, int value, size_t bytes, cudaStream_t stream) override {
                cudaMemsetAsync(ptr, value, bytes, stream);
            }
            void memcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) override {
                cudaMemcpyAsync(dst, src, bytes, kind, stream);
            }
        };

        DispatcherCUDAAllocator cuda_allocator; // The Dispatcher Instance
        
        //Pinned instances
        Pinned_CPU_Allocator pinned_default(cudaHostAllocDefault);
        Pinned_CPU_Allocator pinned_mapped(cudaHostAllocMapped);
        Pinned_CPU_Allocator pinned_portable(cudaHostAllocPortable);
        Pinned_CPU_Allocator pinned_wc(cudaHostAllocWriteCombined);
    }

     Allocator* AllocatorRegistry::get_pinned_cpu_allocator(Pinned_Flag pin_ten){
        switch(pin_ten){
            case Pinned_Flag::Default:  return &pinned_default;
            case Pinned_Flag::Mapped:   return &pinned_mapped;
            case Pinned_Flag::Portable: return &pinned_portable;
            case Pinned_Flag::WriteCombined: return &pinned_wc;
            case Pinned_Flag::None: return &cpu_allocator;
            default: return &pinned_portable; //safe/best pinned fallback
        }
     }

    Allocator* AllocatorRegistry::get_allocator(Device device) {
        if (device == Device::CPU) {
            return &cpu_allocator;
        } else {
            return &cuda_allocator;
        }
    }

    Allocator* AllocatorRegistry::get_cpu_allocator() {
        return &cpu_allocator;
    }

    Allocator* AllocatorRegistry::get_cuda_allocator() {
        return &cuda_allocator;
    }

}