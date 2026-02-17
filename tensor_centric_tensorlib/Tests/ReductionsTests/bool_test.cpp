#include <iostream>
#include "TensorLib.h"
#include "device/GPUCachingAllocator.h"
using namespace OwnTensor;
using namespace std;
// #include <iostream>
int main(){
    device::GPUCachingAllocator::set_allow_growth(true);

 cout<<"hi"<<endl;
Tensor T({{3,2}},Dtype::Complex32,Device::CUDA);
//std::vector<bool> data = {false,false,false,false,true,false,true,false,true};
cout<<"hi"<<endl;
//std::vector<signed char> data = {100,79,60,90,10,0};
cout<<"ji"<<endl;
std::vector data = {complex32_t(100.0,5.0),complex32_t(100.0,5.0),complex32_t(100.0,5.0),complex32_t(100.0,5.0),complex32_t(100.0,5.0),complex32_t(100.0,5.0)};
T.set_data(data);
T.display();

Tensor T1({{2}},Dtype::Complex32,Device::CUDA);
//T1.fill(complex32_t(40.0,5.0));

Tensor res = T + T1 ;
res.display();
std::cin.get();
// Tensor res = reduce_max(T);
// res.display(std::cout, 4);
// Tensor Honey = reduce_all(T);
// Tensor Bunty = reduce_any(T);
// // Tensor x=Honey.to_cpu();
// // Tensor y=Bunty.to_cpu();
// Honey.to_cpu().display(cout,2);
// Bunty.to_cpu().display(cout,2);
}