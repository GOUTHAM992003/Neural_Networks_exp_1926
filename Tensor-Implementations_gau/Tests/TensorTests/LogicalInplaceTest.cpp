#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "TensorLib.h"

using namespace OwnTensor;
using namespace std;

void test_logical_and_inplace() {
    cout << "Testing logical_and_..." << endl;
    
    // 1. Bool Tensor
    Tensor a = Tensor::zeros(Shape{{4}}, TensorOptions().with_dtype(Dtype::Bool));
    // a = [0, 0, 0, 0]
    vector<bool> data_a = {true, true, false, false};
    a.set_data(data_a);
    
    Tensor b = Tensor::zeros(Shape{{4}}, TensorOptions().with_dtype(Dtype::Bool));
    vector<bool> data_b = {true, false, true, false};
    b.set_data(data_b);
    
    logical_AND_(a, b);
    
    // Expected: [true, false, false, false]
    auto ptr = a.data<bool>();
    assert(ptr[0] == true);
    assert(ptr[1] == false);
    assert(ptr[2] == false);
    assert(ptr[3] == false);
    
    cout << "  Bool logical_and_ passed." << endl;
    
    // 2. Float Tensor (Non-Bool)
    Tensor f = Tensor::zeros(Shape{{4}}, TensorOptions().with_dtype(Dtype::Float32));
    vector<float> data_f = {1.0f, 0.0f, 2.5f, 0.0f};
    f.set_data(data_f);
    
    // b is still [true, false, true, false]
    // f && b -> [1&&1, 0&&0, 1&&1, 0&&0] -> [1, 0, 1, 0]
    logical_AND_(f, b);
    
    auto f_ptr = f.data<float>();
    assert(f_ptr[0] == 1.0f);
    assert(f_ptr[1] == 0.0f);
    assert(f_ptr[2] == 1.0f);
    assert(f_ptr[3] == 0.0f);
    
    cout << "  Float logical_and_ passed." << endl;
}

void test_logical_or_inplace() {
    cout << "Testing logical_or_..." << endl;
    
    Tensor a = Tensor::zeros(Shape{{4}}, TensorOptions().with_dtype(Dtype::Bool));
    vector<bool> data_a = {true, true, false, false};
    a.set_data(data_a);
    
    Tensor b = Tensor::zeros(Shape{{4}}, TensorOptions().with_dtype(Dtype::Bool));
    vector<bool> data_b = {true, false, true, false};
    b.set_data(data_b);
    
    logical_OR_(a, b);
    
    // Expected: [true, true, true, false]
    auto ptr = a.data<bool>();
    assert(ptr[0] == true);
    assert(ptr[1] == true);
    assert(ptr[2] == true);
    assert(ptr[3] == false);
    
    cout << "  Bool logical_or_ passed." << endl;
}

void test_logical_xor_inplace() {
    cout << "Testing logical_xor_..." << endl;
    
    Tensor a = Tensor::zeros(Shape{{4}}, TensorOptions().with_dtype(Dtype::Bool));
    vector<bool> data_a = {true, true, false, false};
    a.set_data(data_a);
    
    Tensor b = Tensor::zeros(Shape{{4}}, TensorOptions().with_dtype(Dtype::Bool));
    vector<bool> data_b = {true, false, true, false};
    b.set_data(data_b);
    
    logical_XOR_(a, b);
    
    // Expected: [false, true, true, false]
    auto ptr = a.data<bool>();
    assert(ptr[0] == false);
    assert(ptr[1] == true);
    assert(ptr[2] == true);
    assert(ptr[3] == false);
    
    cout << "  Bool logical_xor_ passed." << endl;
}

void test_logical_not_inplace() {
    cout << "Testing logical_not_..." << endl;
    
    Tensor a = Tensor::zeros(Shape{{2}}, TensorOptions().with_dtype(Dtype::Bool));
    vector<bool> data_a = {true, false};
    a.set_data(data_a);
    
    logical_NOT_(a);
    
    auto ptr = a.data<bool>();
    assert(ptr[0] == false);
    assert(ptr[1] == true);
    
    cout << "  Bool logical_not_ passed." << endl;
}

void test_broadcasting() {
    cout << "Testing broadcasting..." << endl;
    
    Tensor a = Tensor::zeros(Shape{{2, 2}}, TensorOptions().with_dtype(Dtype::Bool));
    // [[T, F], [T, F]]
    vector<bool> data_a = {true, false, true, false};
    a.set_data(data_a);
    
    Tensor b = Tensor::zeros(Shape{{2}}, TensorOptions().with_dtype(Dtype::Bool));
    // [T, T]
    vector<bool> data_b = {true, true};
    b.set_data(data_b);
    
    // b broadcasts to [[T, T], [T, T]]
    // a && b -> [[T, F], [T, F]]
    logical_AND_(a, b);
    
    auto ptr = a.data<bool>();
    assert(ptr[0] == true);
    assert(ptr[1] == false);
    assert(ptr[2] == true);
    assert(ptr[3] == false);
    
    cout << "  Broadcasting passed." << endl;
    
    // Test invalid broadcasting
    Tensor c = Tensor::zeros(Shape{{3}}, TensorOptions().with_dtype(Dtype::Bool));
    try {
        logical_AND_(a, c);
        cout << "  FAILED: Should have thrown error for invalid broadcasting." << endl;
        exit(1);
    } catch (const std::exception& e) {
        cout << "  Caught expected error: " << e.what() << endl;
    }
}

int main() {
    try {
        test_logical_and_inplace();
        test_logical_or_inplace();
        test_logical_xor_inplace();
        test_logical_not_inplace();
        test_broadcasting();
        
        cout << "\nAll tests passed!" << endl;
    } catch (const std::exception& e) {
        cerr << "Test failed with exception: " << e.what() << endl;
        return 1;
    }
    return 0;
}
