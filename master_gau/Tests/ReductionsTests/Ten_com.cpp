#include "TensorLib.h"
#include <iostream> 
using namespace OwnTensor;
using namespace std;

int main() {
    Tensor x({{300,200,200,200}}, Dtype::Float32, DeviceIndex(Device::CPU));
    x.fill(100.0f);
   // x.set_data({int16_t(150),int16_t(200) , int16_t(250), int16_t(300), int16_t(350), int16_t(400)});
    //cout<<"Tensor X:"<<endl;
    //x.to_cpu().display(std::cout,4);
    //Tensor y({{1,2}}, Dtype::Bool, DeviceIndex(Device::CUDA));
    //y.set_data({bool(300),bool(1)});
    //y.fill(300);
    //cout<<"Tensor Y:"<<endl;
   // y.to_cpu().display(std::cout,4);
    //Comparison
    auto start = std::chrono::high_resolution_clock::now();
    reduce_sum(x,{0,2});
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time taken: " << diff.count() << " seconds" << std::endl;

    //res.to_cpu().display(std::cout,4);

    // res = x -y;
    // res.to_cpu().display(std::cout,4);
    
    // res = x *y;
    // res.to_cpu().display(std::cout,4);
    
    // res = x <=y;
    // res.to_cpu().display(std::cout,4);

    // res = x >y;
    // res.to_cpu().display(std::cout,4);

    // res = x <y;
    // res.to_cpu().display(std::cout,4);

    return 0;
}
