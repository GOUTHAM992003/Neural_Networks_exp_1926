// BEFORE optimizations baseline — run on colleague's system with OLD code
// Exact tensor sizes from gpt2_attn_fixed.cpp training
// B=8, T=1024, n_embd=384, fc_up output = 4*n_embd = 1536
// GeLU applied on [8, 1024, 1536] = 12,582,912 elements
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

    Tensor x    = Tensor::randn<float>(Shape{{8, 1024, 1536}}, opts, 1337, 1.0f);
    Tensor bias = Tensor::randn<float>(Shape{{1536}}, opts, 42, 1.0f);
    Tensor grad = Tensor::randn<float>(Shape{{8, 1024, 1536}}, opts, 99, 1.0f);

    std::cout << "\n=== " << label << " ===" << std::endl;
    std::cout << "Shape: [8, 1024, 1536]  (" << x.numel() << " elements) — exact GPT-2 training size" << std::endl;
    std::cout << "Warmup: " << WARMUP << ", Timed runs: " << N << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    // Forward — gelu
    std::cout << "autograd::gelu forward (fp32):          " << bench([&]{ autograd::gelu(x); }) << " ms" << std::endl;

    // Forward — fused_bias_gelu (comment out if not available in old code)
    std::cout << "autograd::fused_bias_gelu fwd (fp32):   " << bench([&]{ autograd::fused_bias_gelu(x, bias); }) << " ms" << std::endl;

    // Backward — gelu only (forward + backward together, as autograd requires forward first)
    auto bwd_gelu = [&]{
        Tensor x2 = x.detach(); x2.set_requires_grad(true);
        Tensor y = autograd::gelu(x2);
        y.backward(&grad);
    };
    std::cout << "autograd::gelu backward (fp32):         " << bench(bwd_gelu) << " ms  (fwd+bwd)" << std::endl;

    // Backward — fused_bias_gelu (comment out if not available in old code)
    auto bwd_bias_gelu = [&]{
        Tensor x2 = x.detach(); x2.set_requires_grad(true);
        Tensor y = autograd::fused_bias_gelu(x2, bias);
        y.backward(&grad);
    };
    std::cout << "autograd::fused_bias_gelu bwd (fp32):   " << bench(bwd_bias_gelu) << " ms  (fwd+bwd)" << std::endl;
}

int main() {
    std::cout << "OwnTensor — BEFORE Optimizations Baseline" << std::endl;
    std::cout << "Tensor size from gpt2_attn_fixed.cpp: B=8, T=1024, 4*n_embd=1536" << std::endl;

    run_tests("CPU Forward+Backward (BEFORE)", Device::CPU);

#ifdef WITH_CUDA
    run_tests("GPU Forward+Backward (BEFORE)", Device::CUDA);
#endif

    return 0;
}
