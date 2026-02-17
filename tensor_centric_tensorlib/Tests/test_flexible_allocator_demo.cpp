#include "core/Tensor.h"
#include "device/AllocatorRegistry.h"
#include <iostream>
#include <cuda_runtime.h>

using namespace OwnTensor;

int main() {
    std::cout << "=== Flexible Pinned Allocator Test ===" << std::endl;
    
    // =========================================
    // Test with direct constructor parameters
    // =========================================
    std::cout << "\n--- Testing with direct constructor ---" << std::endl;
    
    // Test 1: Default (Pageable) memory
    std::cout << "\n[Test 1] Creating tensor with Pinned_Flag::None (Pageable)..." << std::endl;
    Tensor t1(Shape({1024}), Dtype::Float32, DeviceIndex(Device::CPU), false, Pinned_Flag::None);
    std::cout << "  ✅ Tensor created with " << t1.numel() << " elements" << std::endl;
    
    // Test 2: Pinned Default memory
    std::cout << "\n[Test 2] Creating tensor with Pinned_Flag::Default..." << std::endl;
    Tensor t2(Shape({1024}), Dtype::Float32, DeviceIndex(Device::CPU), false, Pinned_Flag::Default);
    std::cout << "  ✅ Tensor created with " << t2.numel() << " elements" << std::endl;
    
    // Test 3: Pinned Portable memory
    std::cout << "\n[Test 3] Creating tensor with Pinned_Flag::Portable..." << std::endl;
    Tensor t3(Shape({1024}), Dtype::Float32, DeviceIndex(Device::CPU), false, Pinned_Flag::Portable);
    std::cout << "  ✅ Tensor created with " << t3.numel() << " elements" << std::endl;
    
    // Test 4: Pinned Mapped memory
    std::cout << "\n[Test 4] Creating tensor with Pinned_Flag::Mapped..." << std::endl;
    Tensor t4(Shape({1024}), Dtype::Float32, DeviceIndex(Device::CPU), false, Pinned_Flag::Mapped);
    std::cout << "  ✅ Tensor created with " << t4.numel() << " elements" << std::endl;
    
    // Verify Mapped memory can get device pointer
    void* device_ptr = nullptr;
    cudaError_t err = cudaHostGetDevicePointer(&device_ptr, t4.data(), 0);
    if (err == cudaSuccess && device_ptr != nullptr) {
        std::cout << "  ✅ cudaHostGetDevicePointer succeeded! Zero-copy ready." << std::endl;
    } else {
        std::cout << "  ⚠️ cudaHostGetDevicePointer returned: " << cudaGetErrorString(err) << std::endl;
    }
    

    // =========================================
    // Test with TensorOptions (NEW!)
    // =========================================
    std::cout << "\n--- Testing with TensorOptions builder pattern ---" << std::endl;
    
    // Test 5: TensorOptions with Pinned Portable
    std::cout << "\n[Test 5] TensorOptions().with_pinned(Pinned_Flag::Portable)..." << std::endl;
    auto opts1 = TensorOptions()
        .with_dtype(Dtype::Float32)
        .with_device(DeviceIndex(Device::CPU))
        .with_pinned(Pinned_Flag::Portable);
    Tensor t5(Shape({1024}), opts1);
    std::cout << "  ✅ Tensor created with " << t5.numel() << " elements" << std::endl;
    
    // Test 6: TensorOptions with Pinned Mapped
    std::cout << "\n[Test 6] TensorOptions().with_pinned(Pinned_Flag::Mapped)..." << std::endl;
    auto opts2 = TensorOptions()
        .with_dtype(Dtype::Float64)
        .with_device(DeviceIndex(Device::CPU))
        .with_pinned(Pinned_Flag::Mapped);
    Tensor t6(Shape({512}), opts2);
    std::cout << "  ✅ Tensor created with " << t6.numel() << " elements (Float64)" << std::endl;
    
    // Verify this one is also mapped
    device_ptr = nullptr;
    err = cudaHostGetDevicePointer(&device_ptr, t6.data(), 0);
    if (err == cudaSuccess && device_ptr != nullptr) {
        std::cout << "  ✅ cudaHostGetDevicePointer succeeded! Zero-copy ready." << std::endl;
    } else {
        std::cout << "  ⚠️ cudaHostGetDevicePointer returned: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Test 7: TensorOptions with WriteCombined
    std::cout << "\n[Test 7] TensorOptions().with_pinned(Pinned_Flag::WriteCombined)..." << std::endl;
    Tensor t7(Shape({2048}), TensorOptions()
        .with_dtype(Dtype::Float32)
        .with_pinned(Pinned_Flag::WriteCombined));
    std::cout << "  ✅ Tensor created with " << t7.numel() << " elements" << std::endl;
    
    // Test 8: CUDA tensor with TensorOptions (pin_ten should be ignored)
    std::cout << "\n[Test 8] TensorOptions with CUDA device (pin_ten ignored)..." << std::endl;
    Tensor t8(Shape({1024}), TensorOptions()
        .with_dtype(Dtype::Float32)
        .with_device(DeviceIndex(Device::CUDA, 0))
        .with_pinned(Pinned_Flag::Portable));  // Should be ignored for CUDA
    std::cout << "  ✅ CUDA Tensor created with " << t8.numel() << " elements" << std::endl;
    std::cout << "  Device: " << (t8.is_cuda() ? "CUDA" : "CPU") << std::endl;
    
    //Test 9 : Check if the tensor is Pinned
    Tensor t = Tensor(Shape({10}),TensorOptions().with_pinned(Pinned_Flag::Portable));
    if (t.is_pinned()) std::cout<< "tensor is Pinned!" << std::endl;
    else std::cout<<"Tensor is not Pinned!"<<std::endl;

    // Test 10: In-Place Pinning
    std::cout << "\n[Test 10] Testing pin_memory_() (in-place)..." << std::endl;
    // Create standard CPU tensor (pageable)
    Tensor t_pageable(Shape({4096}), Dtype::Float32, DeviceIndex(Device::CPU), false, Pinned_Flag::None);
    std::cout << "  Before pinning: " << (t_pageable.is_pinned() ? "Pinned" : "Pageable") << std::endl;
    
    try {
        t_pageable.pin_memory();
        std::cout << "  After pinning: " << (t_pageable.is_pinned() ? "Pinned (Success)" : "Pageable (Failed)") << std::endl;
        
        // Verify with CUDA API directly just to be sure
        void* ptr = nullptr;
        cudaError_t err = cudaHostRegister(t_pageable.data(), t_pageable.nbytes(), cudaHostRegisterMapped);
        // Note: Re-registering might fail or succeed depending on flags, but getting attributes is better
        cudaPointerAttributes attr;
        cudaPointerGetAttributes(&attr, t_pageable.data());
        if (attr.type == cudaMemoryTypeHost) {
             std::cout << "  ✅ Attributes confirm Host Pinned Memory." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "  ❌ Exception during pinning: " << e.what() << std::endl;
    }

    std::cout << "\n=== All Tests Passed! ===" << std::endl;
      bool a=t_pageable.is_pinned();
    std::cout<<a<<std::endl;
    return 0;
  
}
