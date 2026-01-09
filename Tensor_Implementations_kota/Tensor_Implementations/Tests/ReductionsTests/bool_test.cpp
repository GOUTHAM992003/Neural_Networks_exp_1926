#include <iostream>
#include "TensorLib.h"
using namespace OwnTensor;
using namespace std;
// #include <iostream>
int main(){

 cout<<"hi"<<endl;
Tensor T({{1,2}},Dtype::Float32,Device::CPU);
//std::vector<bool> data = {false,false,false,false,true,false,true,false,true};
cout<<"hi"<<endl;
//std::vector<signed char> data = {100,79,60,90,10,0};
cout<<"ji"<<endl;
std::vector<float> data = {16777217.0,1.0};
T.set_data(data);
T.display();
Tensor res = reduce_sum(T);
res.display();
Tensor T1 = T+1+1;
//T1.fill(40.0);

//Tensor res = reduce_sum(T);
T1.display();

// Tensor res = reduce_max(T);
// res.display(std::cout, 4);
// Tensor Honey = reduce_all(T);
// Tensor Bunty = reduce_any(T);
// // Tensor x=Honey.to_cpu();
// // Tensor y=Bunty.to_cpu();
// Honey.to_cpu().display(cout,2);
// Bunty.to_cpu().display(cout,2);
}