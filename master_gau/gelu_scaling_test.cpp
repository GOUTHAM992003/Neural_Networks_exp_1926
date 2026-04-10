#include "TensorLib.h"
#include "device/DeviceCore.h"
#include <iostream>
#include <chrono>
using namespace OwnTensor;

constexpr int WARMUP = 5;
constexpr int N      = 50;

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

struct TestCase {
    int b, s, h;
    const char* label;
};

int main() {
    std::cout << "OwnTensor — GeLU Scaling Test (CPU vs GPU across tensor sizes)\n" << std::endl;

    TestCase cases[] = {
        {1,   128,  128,  "Tiny    [1, 128, 128]     = 16K   "},
        {1,   512,  512,  "Small   [1, 512, 512]     = 262K  "},
        {8,  1024,  384,  "Medium  [8, 1024, 384]    = 3.1M  "},
        {8,  1024,  768,  "GPT2    [8, 1024, 768]    = 6.3M  "},
        {32, 1024,  768,  "Large   [32, 1024, 768]   = 25.2M "},
        {64, 1024,  768,  "XLarge  [64, 1024, 768]   = 50.3M "},
        {64, 2048, 1024,  "Huge    [64, 2048, 1024]  = 134.2M"},
    };

    std::cout << "┌─────────────────────────────────────────┬──────────────────────────┬──────────────────────────┬──────────────────────────┬──────────────────────────┐" << std::endl;
    std::cout << "│              Tensor Size                 │  gelu_fwd CPU  │ GPU     │  gelu_bwd CPU  │ GPU     │  bias_gelu_fwd CPU│ GPU  │  bias_gelu_bwd CPU│ GPU  │" << std::endl;
    std::cout << "├─────────────────────────────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────┤" << std::endl;

    for (auto& tc : cases) {
        TensorOptions cpu_opts = TensorOptions().with_device(Device::CPU).with_dtype(Dtype::Float32);
        Tensor x_cpu    = Tensor::randn<float>(Shape{{tc.b, tc.s, tc.h}}, cpu_opts, 1337, 1.0f);
        Tensor bias_cpu = Tensor::randn<float>(Shape{{tc.h}}, cpu_opts, 42, 1.0f);
        Tensor grad_cpu = Tensor::randn<float>(Shape{{tc.b, tc.s, tc.h}}, cpu_opts, 99, 1.0f);

        double cpu_fwd  = bench([&]{ gelu_forward(x_cpu); });
        double cpu_bwd  = bench([&]{ gelu_backward(grad_cpu, x_cpu); });
        double cpu_bf   = bench([&]{ fused_bias_gelu_forward(x_cpu, bias_cpu); });
        double cpu_bb   = bench([&]{ fused_bias_gelu_backward(grad_cpu, x_cpu, bias_cpu); });

        double gpu_fwd = 0, gpu_bwd = 0, gpu_bf = 0, gpu_bb = 0;
#ifdef WITH_CUDA
        TensorOptions gpu_opts = TensorOptions().with_device(Device::CUDA).with_dtype(Dtype::Float32);
        Tensor x_gpu    = Tensor::randn<float>(Shape{{tc.b, tc.s, tc.h}}, gpu_opts, 1337, 1.0f);
        Tensor bias_gpu = Tensor::randn<float>(Shape{{tc.b, tc.s, tc.h}}, gpu_opts, 42, 1.0f);
        Tensor grad_gpu = Tensor::randn<float>(Shape{{tc.b, tc.s, tc.h}}, gpu_opts, 99, 1.0f);
        // bias should be [hidden] not [b,s,h] — fix:
        Tensor bias_gpu_h = Tensor::randn<float>(Shape{{tc.h}}, gpu_opts, 42, 1.0f);

        gpu_fwd = bench([&]{ gelu_forward(x_gpu); });
        gpu_bwd = bench([&]{ gelu_backward(grad_gpu, x_gpu); });
        gpu_bf  = bench([&]{ fused_bias_gelu_forward(x_gpu, bias_gpu_h); });
        gpu_bb  = bench([&]{ fused_bias_gelu_backward(grad_gpu, x_gpu, bias_gpu_h); });
#endif

        printf("│ %s │ %7.3f ms │ %7.3f ms │ %7.3f ms │ %7.3f ms │ %7.3f ms │ %7.3f ms │ %7.3f ms │ %7.3f ms │\n",
               tc.label,
               cpu_fwd, gpu_fwd, cpu_bwd, gpu_bwd, cpu_bf, gpu_bf, cpu_bb, gpu_bb);
    }

    std::cout << "└─────────────────────────────────────────┴──────────────────────────┴──────────────────────────┴──────────────────────────┴──────────────────────────┘" << std::endl;

    std::cout << "\nSpeedups (GPU / CPU):" << std::endl;
    std::cout << "─────────────────────────────────────────────────────────────────" << std::endl;

    for (auto& tc : cases) {
        TensorOptions cpu_opts = TensorOptions().with_device(Device::CPU).with_dtype(Dtype::Float32);
        Tensor x_cpu    = Tensor::randn<float>(Shape{{tc.b, tc.s, tc.h}}, cpu_opts, 1337, 1.0f);
        Tensor grad_cpu = Tensor::randn<float>(Shape{{tc.b, tc.s, tc.h}}, cpu_opts, 99, 1.0f);

        double cpu_fwd = bench([&]{ gelu_forward(x_cpu); });

#ifdef WITH_CUDA
        TensorOptions gpu_opts = TensorOptions().with_device(Device::CUDA).with_dtype(Dtype::Float32);
        Tensor x_gpu = Tensor::randn<float>(Shape{{tc.b, tc.s, tc.h}}, gpu_opts, 1337, 1.0f);
        double gpu_fwd = bench([&]{ gelu_forward(x_gpu); });
        printf("  %s  gelu_fwd: CPU=%.3fms  GPU=%.3fms  → GPU is %.1fx faster\n",
               tc.label, cpu_fwd, gpu_fwd, cpu_fwd / gpu_fwd);
#endif
    }

    return 0;
}
