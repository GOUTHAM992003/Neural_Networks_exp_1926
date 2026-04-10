#include "TensorLib.h"
#include "device/DeviceCore.h"
#include <iostream>
#include <chrono>
using namespace OwnTensor;

constexpr int WARMUP = 5;
constexpr int N      = 100;

template<typename F>
double bench(F&& fn) {
    for (int i = 0; i < WARMUP; i++) fn();
#ifdef WITH_CUDA
    cudaDeviceSynchronize();
#endif
    auto s = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) fn();
#ifdef WITH_CUDA
    cudaDeviceSynchronize();
#endif
    auto e = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(e - s).count() / N;
}

void run_benchmarks(const char* label, Device dev) {
    TensorOptions opts = TensorOptions().with_device(dev).with_dtype(Dtype::Float32);
    Tensor x    = Tensor::randn<float>(Shape{{8, 1024, 384}}, opts, 1337, 1.0f);
    Tensor bias = Tensor::randn<float>(Shape{{384}}, opts, 42, 1.0f);
    // For backward: need grad_output same shape as x
    Tensor grad = Tensor::randn<float>(Shape{{8, 1024, 384}}, opts, 99, 1.0f);

    std::cout << "\n=== " << label << " ===" << std::endl;
    std::cout << "Shape: [8, 1024, 384]  (" << x.numel() << " elements)" << std::endl;
    std::cout << "Warmup: " << WARMUP << ", Timed runs: " << N << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    // Forward
    std::cout << "gelu_forward (fp32):                    " << bench([&]{ gelu_forward(x); }) << " ms" << std::endl;
    std::cout << "fused_bias_gelu_forward (fp32):         " << bench([&]{ fused_bias_gelu_forward(x, bias); }) << " ms" << std::endl;

    // Backward
    std::cout << "gelu_backward (fp32):                   " << bench([&]{ gelu_backward(grad, x); }) << " ms" << std::endl;
    std::cout << "fused_bias_gelu_backward (fp32):        " << bench([&]{ fused_bias_gelu_backward(grad, x, bias); }) << " ms" << std::endl;
}

int main() {
    std::cout << "OwnTensor Library — GeLU Benchmark (all 4 bifurcations)" << std::endl;

    run_benchmarks("Bifurcation 1+3: CPU Forward+Backward GeLU", Device::CPU);

#ifdef WITH_CUDA
    run_benchmarks("Bifurcation 2+4: GPU Forward+Backward GeLU", Device::CUDA);
#endif

    return 0;
}
