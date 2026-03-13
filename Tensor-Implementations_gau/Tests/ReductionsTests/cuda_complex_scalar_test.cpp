#include <iostream>
#include "TensorLib.h"

using namespace OwnTensor;

int main() {
    std::cout << "=== CUDA Complex Scalar Operations Tests ===" << std::endl;
    
    // Test 1: Complex64 + int scalar on CUDA
    std::cout << "\nTest 1: CUDA Complex64 + int (3)" << std::endl;
    Tensor c64({{2, 2}}, Dtype::Complex64, Device::CUDA);
    
    // Set values via CPU then copy
    Tensor c64_cpu({{2, 2}}, Dtype::Complex64, Device::CPU);
    c64_cpu.data<complex64_t>()[0] = complex64_t(1.0f, 2.0f);
    c64_cpu.data<complex64_t>()[1] = complex64_t(3.0f, 4.0f);
    c64_cpu.data<complex64_t>()[2] = complex64_t(5.0f, 6.0f);
    c64_cpu.data<complex64_t>()[3] = complex64_t(7.0f, 8.0f);
    c64 = c64_cpu.to(Device::CUDA);
    
    std::cout << "Before: (1+2j), (3+4j), (5+6j), (7+8j)" << std::endl;
    
    Tensor result1 = c64 + 3;
    Tensor result1_cpu = result1.to(Device::CPU);
    std::cout << "After + 3: ";
    for (int i = 0; i < 4; ++i) {
        auto v = result1_cpu.data<complex64_t>()[i];
        std::cout << "(" << v.real() << "+" << v.imag() << "j) ";
    }
    std::cout << std::endl;
    // Expected: (4+2j), (6+4j), (8+6j), (10+8j)
    
    std::cout << "\n=== CUDA tests completed! ===" << std::endl;
    return 0;
}
