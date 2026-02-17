#include "TensorLib.h"
#include<vector>
using namespace OwnTensor;

int main()
{
    DeviceIndex device = DeviceIndex(Device::CUDA);
    TensorOptions req_grad = TensorOptions().with_req_grad(true).with_device(device).with_dtype(Dtype::Float32);
    Tensor x({{2,3}},req_grad);
    std::vector<float> data = {(1.0f),(2.0f),(3.0f),(4.0f),(5.0f),(6.0f)};
    // std::vector<float16_t> data = {float16_t(1),float16_t(2),float16_t(3),float16_t(4),float16_t(5),float16_t(6)};
    // Tensor x = Tensor::full(Shape{{64, 128}}, TensorOptions().with_device(device).with_dtype(Dtype::Float16).with_req_grad(true), 20.0f);
    // x.display();
    
    // Tensor target = Tensor::zeros(Shape{{64}}, TensorOptions().with_dtype(Dtype::Float16).with_device(device));
    auto gelu_1 = autograd::gelu(x);

    gelu_1.display();
}