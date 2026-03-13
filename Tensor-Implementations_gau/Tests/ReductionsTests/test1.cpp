#include <iostream>
#include "TensorLib.h"

using namespace OwnTensor;
using namespace std;

int main(){
    Tensor T({{1,3}},Dtype::UInt8,Device:: CPU);
    //T.fill(true);
    vector<uint8_t> data = {1,2,3};
    T.set_data(data);
    
    //Scalar ops:
    int8_t x = 8;

    Tensor t = T + x;
    std::cout<<"Addition: "<<endl;
    t.display(); 

    Tensor t1 = T - x;
    std::cout<<"Subtraction: "<<endl;
    t1.display(); 

    t1 = T * x;
    std::cout<<"Multiplication "<<endl;
    t1.display(); 
    
    t1 = T / x;
    std::cout<<"Division "<<endl;
    t1.display(); 
    
    // T+=1;
    // T.display();

    // T-=1;
    // T.display();

    // T*=1;
    // T.display();

    // T/=1;
    // T.display();

    // Tensor t2 = logical_AND(T,5.0);
    // t2.display();

     Tensor t3 = logical_OR(T,5.0);
    // t3.display();

    // Tensor t4 = logical_NOT(T);
    // t4.display();

    // Tensor t5 = logical_XOR(T,5.0);
    // t5.display();
    
    return 0;
}