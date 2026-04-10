// NEW optimized code benchmark — exact tensor sizes from gpt2_attn_fixed.cpp training
// B=8, T=1024, n_embd=384, fc_up output = 4*n_embd = 1536
// GeLU applied on [8, 1024, 1536] = 12,582,912 elements
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

void run_tests(const char* label, Device dev) {
    TensorOptions opts = TensorOptions().with_device(dev).with_dtype(Dtype::Float32);

    // Exact training tensor: after fc_up [n_embd → 4*n_embd], shape = [B, T, 4*n_embd]
    Tensor x    = Tensor::randn<float>(Shape{{8, 1024, 1536}}, opts, 1337, 1.0f);
    Tensor bias = Tensor::randn<float>(Shape{{1536}}, opts, 42, 1.0f);
    Tensor grad = Tensor::randn<float>(Shape{{8, 1024, 1536}}, opts, 99, 1.0f);

    std::cout << "\n=== " << label << " ===" << std::endl;
    std::cout << "Shape: [8, 1024, 1536]  (" << x.numel() << " elements) — exact GPT-2 training size" << std::endl;
    std::cout << "Warmup: " << WARMUP << ", Timed runs: " << N << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    // Forward
    std::cout << "gelu_forward (fp32):                    " << bench([&]{ gelu_forward(x); }) << " ms" << std::endl;
    std::cout << "fused_bias_gelu_forward (fp32):         " << bench([&]{ fused_bias_gelu_forward(x, bias); }) << " ms" << std::endl;

    // Backward (separate, not including forward)
    std::cout << "gelu_backward (fp32):                   " << bench([&]{ gelu_backward(grad, x); }) << " ms" << std::endl;
    std::cout << "fused_bias_gelu_backward (fp32):        " << bench([&]{ fused_bias_gelu_backward(grad, x, bias); }) << " ms" << std::endl;
}

int main() {
    std::cout << "OwnTensor — GeLU Training Benchmark (AFTER optimizations)" << std::endl;
    std::cout << "Tensor size from gpt2_attn_fixed.cpp: B=8, T=1024, 4*n_embd=1536" << std::endl;

    run_tests("CPU Forward+Backward", Device::CPU);

#ifdef WITH_CUDA
    run_tests("GPU Forward+Backward", Device::CUDA);
#endif

    return 0;
}
