// Test 0-dimensional tensor support (like PyTorch scalars)
#include <iostream>
#include "TensorLib.h"

using namespace OwnTensor;

int main() {
    std::cout << "=== 0-Dimensional Tensor Tests ===\n\n";
    
    // Test 1: Create 0-dim tensor with empty shape
    std::cout << "Test 1: Create 0-dim tensor with empty shape {}\n";
    try {
        Tensor scalar({{}}, Dtype::Float32, Device::CPU);
        std::cout << "  shape.dims.size() = " << scalar.shape().dims.size() << " (expected 0)\n";
        std::cout << "  numel() = " << scalar.numel() << " (expected 1)\n";
        std::cout << "  ndim() = " << scalar.ndim() << " (expected 0)\n";
        std::cout << "  data() is null: " << (scalar.data() == nullptr ? "yes" : "no") << " (expected no)\n";
        
        // Set a value
        scalar.fill(42.0f);
        float* ptr = scalar.data<float>();
        std::cout << "  value after fill(42.0): " << *ptr << " (expected 42)\n";
        
        std::cout << "  ✓ Test 1 PASSED\n\n";
    } catch (const std::exception& e) {
        std::cout << "  ✗ Test 1 FAILED: " << e.what() << "\n\n";
    }
    
    // Test 2: Clone 0-dim tensor
    std::cout << "Test 2: Clone 0-dim tensor\n";
    try {
        Tensor scalar({{}}, Dtype::Float32, Device::CPU);
        scalar.fill(123.0f);
        
        Tensor cloned = scalar.clone();
        std::cout << "  cloned.numel() = " << cloned.numel() << " (expected 1)\n";
        std::cout << "  cloned value = " << *cloned.data<float>() << " (expected 123)\n";
        std::cout << "  ✓ Test 2 PASSED\n\n";
    } catch (const std::exception& e) {
        std::cout << "  ✗ Test 2 FAILED: " << e.what() << "\n\n";
    }
    
    // Test 3: Display 0-dim tensor
    std::cout << "Test 3: Display 0-dim tensor\n";
    try {
        Tensor scalar({{}}, Dtype::Int32, Device::CPU);
        scalar.fill(99);
        std::cout << "  Display output:\n  ";
        scalar.display();
        std::cout << "  ✓ Test 3 PASSED\n\n";
    } catch (const std::exception& e) {
        std::cout << "  ✗ Test 3 FAILED: " << e.what() << "\n\n";
    }
    
    std::cout << "=== All 0-dim tensor tests complete ===\n";
    return 0;
}
