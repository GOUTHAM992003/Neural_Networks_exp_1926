#include <iostream>
#include<TensorLib.h>
using namespace OwnTensor ;
int main(){
    Tensor t({{3,3}},Dtype::Float32,Device::CPU);
    Tensor t2({{3,3}},Dtype::Float32,Device::CUDA);
    Tensor t4({{3,3}},Dtype::Float32,Device::CPU);
    Tensor t5({{3,3}},Dtype::Float32,Device::CUDA);
std::vector<float> data = {1,2,3,4,5,6,7,8,9};
std::vector<float> data2 = {1.0,2.0,NAN,4.0,5.0,NAN,7.0,8.0,NAN};
t.set_data(data);
t2.set_data(data);
t4.set_data(data2);
t5.set_data(data2);
t.display();
t2.display();
// Tensor t1=reduce_sum(t);
// Tensor t3=reduce_sum(t2);
// Tensor t1 = reduce_product(t);
// Tensor t3 = reduce_product(t2);
// Tensor t1= reduce_max(t);
// Tensor t3= reduce_max(t2);
// Tensor t1=reduce_min(t);
// Tensor t3=reduce_min(t2);
// Tensor t1=reduce_nansum(t4);
// Tensor t3=reduce_nansum(t5);
// Tensor t1=reduce_nanproduct(t4);
// Tensor t3=reduce_nanproduct(t5);
// Tensor t1=reduce_nanmax(t);
// Tensor t3=reduce_nanmax(t2);
// Tensor t1=reduce_nanmin(t4,{1});
// Tensor t3=reduce_nanmin(t5,{1});
Tensor t1=reduce_argmax(t4);
Tensor t3= reduce_argmax(t5);
t1.display();
t3.display();
}