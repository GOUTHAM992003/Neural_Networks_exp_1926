// NEW code benchmark — GPU only, gelu forward + backward
// Same overhead as old code: both go through autograd::gelu()
#include "TensorLib.h"
#include "autograd/operations/ActivationOps.h"
#include "device/DeviceCore.h"
#include <iostream>
#include <chrono>
using namespace OwnTensor;

constexpr int WARMUP = 5;
constexpr int N      = 100;

template<typename F>
double bench(F&& fn) {
    for (int i = 0; i < WARMUP; i++) fn();
    cudaDeviceSynchronize();
    auto s = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) fn();
    cudaDeviceSynchronize();
    auto e = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(e - s).count() / N;
}

int main() {
    DeviceIndex dev(Device::CUDA, 0);
    TensorOptions opts = TensorOptions().with_device(dev).with_dtype(Dtype::Float32);

    Tensor x    = Tensor::randn<float>(Shape{{8, 1024, 1536}}, opts, 1337, 1.0f);
    Tensor grad = Tensor::randn<float>(Shape{{8, 1024, 1536}}, opts, 99, 1.0f);

    std::cout << "OwnTensor — GPU GeLU Benchmark (AFTER optimizations)" << std::endl;
    std::cout << "Shape: [8, 1024, 1536]  (" << x.numel() << " elements)" << std::endl;
    std::cout << "Warmup: " << WARMUP << ", Timed: " << N << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    // Forward — through autograd (same overhead as old code)
    std::cout << "autograd::gelu forward (fp32):          "
              << bench([&]{ autograd::gelu(x); }) << " ms" << std::endl;

    // Backward — forward + backward together (same as old code)
    auto bwd = [&]{
        Tensor x2 = x.detach(); x2.set_requires_grad(true);
        Tensor y = autograd::gelu(x2);
        y.backward(&grad);
    };
    std::cout << "autograd::gelu backward (fp32):         "
              << bench(bwd) << " ms  (fwd+bwd)" << std::endl;

    return 0;
}
