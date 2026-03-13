#include <iostream>
#include "TensorLib.h"

using namespace OwnTensor;

int main() {
    std::cout << "=== Complex Scalar Operations Tests ===" << std::endl;
    
    // Test 1: Complex64 + int scalar
    std::cout << "\nTest 1: Complex64 + int (3)" << std::endl;
    Tensor c64({{2, 2}}, Dtype::Complex64, Device::CPU);
    c64.data<complex64_t>()[0] = complex64_t(1.0f, 2.0f);
    c64.data<complex64_t>()[1] = complex64_t(3.0f, 4.0f);
    c64.data<complex64_t>()[2] = complex64_t(5.0f, 6.0f);
    c64.data<complex64_t>()[3] = complex64_t(7.0f, 8.0f);
    std::cout << "Before: (1+2j), (3+4j), (5+6j), (7+8j)" << std::endl;
    
    Tensor result1 = c64 + 3;
    std::cout << "After + 3: ";
    for (int i = 0; i < 4; ++i) {
        auto v = result1.data<complex64_t>()[i];
        std::cout << "(" << v.real() << "+" << v.imag() << "j) ";
    }
    std::cout << std::endl;
    // Expected: (4+2j), (6+4j), (8+6j), (10+8j)
    
    // Test 2: Complex64 * float scalar
    std::cout << "\nTest 2: Complex64 * 2.0f" << std::endl;
    Tensor result2 = c64 * 2.0f;
    std::cout << "Result: ";
    for (int i = 0; i < 4; ++i) {
        auto v = result2.data<complex64_t>()[i];
        std::cout << "(" << v.real() << "+" << v.imag() << "j) ";
    }
    std::cout << std::endl;
    // Expected: (2+4j), (6+8j), (10+12j), (14+16j)
    
    // Test 3: Complex64 == 1 (equality check)
    std::cout << "\nTest 3: Complex64 == 1" << std::endl;
    Tensor c64_eq({{4}}, Dtype::Complex64, Device::CPU);
    c64_eq.data<complex64_t>()[0] = complex64_t(1.0f, 0.0f);  // equals 1
    c64_eq.data<complex64_t>()[1] = complex64_t(1.0f, 2.0f);  // not equal
    c64_eq.data<complex64_t>()[2] = complex64_t(2.0f, 0.0f);  // not equal
    c64_eq.data<complex64_t>()[3] = complex64_t(1.0f, 0.0f);  // equals 1
    
    Tensor result3 = c64_eq == 1;
    std::cout << "Result (expect: T F F T): ";
    for (int i = 0; i < 4; ++i) {
        std::cout << (result3.data<bool>()[i] ? "T" : "F") << " ";
    }
    std::cout << std::endl;
    
    // Test 4: Complex64 < scalar should throw
    std::cout << "\nTest 4: Complex64 < 5 (should throw)" << std::endl;
    try {
        Tensor result4 = c64 < 5;
        std::cout << "ERROR: Should have thrown!" << std::endl;
    } catch (const std::runtime_error& e) {
        std::cout << "Correctly threw: " << e.what() << std::endl;
    }
    
    std::cout << "\n=== All tests completed! ===" << std::endl;
    return 0;
}
