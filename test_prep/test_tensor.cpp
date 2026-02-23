#include "tensor.h"
#include <iostream>

int main(){
    Tensor t({2,3},Dtype::Float32);
    std::cout<<"Number of elements :"<<t.numel()<<std::endl;
    std::cout<<"Shape of tensor :";
    t.get_shape();

    std::cout<<"Dtype of tensor :"<<t.get_dtype()<<std::endl;
    
}