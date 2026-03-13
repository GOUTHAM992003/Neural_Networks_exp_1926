#include <iostream>
#include "TensorLib.h"
using namespace OwnTensor;
using namespace std;

int main() {
    cout << "=== Scalar Type Promotion Tests ===" << endl;
    
    // Test 1: Float16 tensor + float32 scalar -> Float16 (tensor wins)
    cout << "\nTest 1: Float16 tensor + float32 scalar" << endl;
    Tensor t1({{2, 2}}, Dtype::Float16, Device::CPU);
    t1.fill(float16_t(10.0f));
    t1.display();
    Tensor r1 = t1 + 3.5f;  // float32 scalar
    cout << "Result dtype: " << get_dtype_name(r1.dtype()) << endl;
    r1.display();
    
    // Test 2: Int32 tensor + float32 scalar -> Float32
    cout << "\nTest 2: Int32 tensor + float32 scalar" << endl;
    Tensor t2({{2, 2}}, Dtype::Int32, Device::CPU);
    t2.fill(10);
    t2.display();
    Tensor r2 = t2 + 3.5f;  // float32 scalar
    cout << "Result dtype: " << get_dtype_name(r2.dtype()) << endl;
    r2.display();
    
    // Test 3: Int32 tensor + int scalar -> Int32 (tensor wins)
    cout << "\nTest 3: Int32 tensor + int scalar" << endl;
    Tensor t3({{2, 2}}, Dtype::Int32, Device::CPU);
    t3.fill(10);
    Tensor r3 = t3 + 5;
    cout << "Result dtype: " << get_dtype_name(r3.dtype()) << endl;
    r3.display();
    
    // Test 4: Bool tensor + int scalar -> Int64
    cout << "\nTest 4: Bool tensor + int scalar" << endl;
    Tensor t4({{2, 2}}, Dtype::Bool, Device::CPU);
    t4.fill(true);
    Tensor r4 = t4 + 5;
    cout << "Result dtype: " << get_dtype_name(r4.dtype()) << endl;
    r4.display();

    cout << "\n=== All tests completed! ===" << endl;
    return 0;
}
