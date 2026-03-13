#include <iostream>
#include "TensorLib.h"
//#include "core/TensorDataManip.h"  // For bool specialization of set_data
using namespace OwnTensor;
using namespace std;
// #include <iostream>
int main(){

 cout<<"hi"<<endl;
Tensor T({{2,1,3}},Dtype::Bool,Device::CUDA);
//std::vector<bool> data = {false,false,false,false,true,false,true,false,true};
// cout<<"hi"<<endl;
std::vector<bool>data = {true, false, true, false, true, false};
Tensor T6 ({{},Dtype::Int32,Device::CPU});
std::vector<int> data26={9};
std::cout<<"ji"<<endl;
T6.set_data(data26);
std::cout<<"ji"<<endl;
T6.to_cpu().display();
// cout<<"ji"<<endl;
//std::vector data = {complex32_t(100.0,5.0),complex32_t(100,5),complex32_t(100,5),complex32_t(100,5),complex32_t(100,5),complex32_t(100,5)};
//std::vector<float16_t> data = {float16_t(true),float16_t(true),float16_t(true),float16_t(true),float16_t(true),float16_t(true)};
float8_e4m3fn_t x = 4;
float16_t y=1926;
complex128_t b= complex128_t(10,5);
T.set_data(data);   
cout << "\n\n\n" << endl;
T.to_cpu().display();
Tensor T1({{4,3}},Dtype::Float32,Device::CUDA);
std::vector<float> data1 = {100.0,19.26,100.0,100.0,100.0,100.0,100.0,19.26,100.0,100.0,100.0,100.0};
T1.set_data(data1);
Tensor T3=T1.as_type(Dtype::UInt8);
// Note: Using == instead of > because complex numbers are not ordered
//Tensor T2=(T==T3);  // Changed from T>T3 - complex numbers can't use >, <, >=, <=
Tensor T4=where(T>T1,19,26);
T4.to_cpu().display();
Tensor T5=T+b;
T5.to_cpu().display();
//T2.to_cpu().display();
}
//  Tensor T1({{3,2}},Dtype::Float8_E4M3FN,Device::CUDA);
//  std::vector<float8_e4m3fn_t> data1 = {100.0, 79.0, 60.0, 90.0, 10.0, 0.0};
// // T1.fill(complex32_t(40,5));
// T1.set_data(data1);
//  Tensor res = T * T1 ;
//  res.to_cpu().display();

// Tensor res = reduce_max(T);
// res.display(std::cout, 4);
// Tensor Honey = reduce_all(T);
// Tensor Bunty = reduce_any(T);
// // Tensor x=Honey.to_cpu();
// // Tensor y=Bunty.to_cpu();
// Honey.to_cpu().display(cout,2);
// Bunty.to_cpu().display(cout,2);
    // Modulo Test
//     cout << "Testing Modulo:" << endl;
//         Tensor A({{2, 2}}, Dtype::Complex32, Device::CUDA);
//         std::vector<complex32_t> dataA = {complex32_t(10,5), complex32_t(11,5), complex32_t(12,5), complex32_t(13,5)};
//         A.set_data(dataA);

//         Tensor B({{2, 2}}, Dtype::Int32, Device::CUDA);
//         std::vector<int> dataB = {3, 3, 5, 5};
//         B.set_data(dataB);

//         Tensor C = A % B;
//         C.to_cpu().display();
// }