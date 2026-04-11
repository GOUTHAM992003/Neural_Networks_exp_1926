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
    std::cout << "OwnTensor — LayerNorm BEFORE Optimization (old code baseline)\n" << std::endl;
    std::cout << "Uses autograd::layer_norm() which had CPU kernels embedded inside.\n" << std::endl;

    TestCase cases[] = {
        {1,   128,  128,  "Tiny    [1, 128, 128]     = 16K   "},
        {1,   512,  512,  "Small   [1, 512, 512]     = 262K  "},
        {8,  1024,  384,  "Medium  [8, 1024, 384]    = 3.1M  "},
        {8,  1024,  768,  "GPT2    [8, 1024, 768]    = 6.3M  "},
        {32, 1024,  768,  "Large   [32, 1024, 768]   = 25.2M "},
        {64, 1024,  768,  "XLarge  [64, 1024, 768]   = 50.3M "},
        {64, 2048, 1024,  "Huge    [64, 2048, 1024]  = 134.2M"},
    };

    // ── LayerNorm Forward (via autograd::layer_norm) ──
    std::cout << "=== LayerNorm Forward (autograd::layer_norm) ===" << std::endl;
    printf("%-42s  %10s  %10s\n", "Size", "CPU (ms)", "GPU (ms)");
    printf("──────────────────────────────────────────  ──────────  ──────────\n");

    for (auto& tc : cases) {
        TensorOptions cpu_opts = TensorOptions().with_device(Device::CPU).with_dtype(Dtype::Float32);
        Tensor x_cpu     = Tensor::randn<float>(Shape{{tc.b, tc.s, tc.h}}, cpu_opts, 1337, 1.0f);
        Tensor gamma_cpu = Tensor::ones(Shape{{tc.h}}, cpu_opts);
        Tensor beta_cpu  = Tensor::zeros(Shape{{tc.h}}, cpu_opts);

        double cpu_fwd = bench([&]{ autograd::layer_norm(x_cpu, gamma_cpu, beta_cpu, tc.h); });

        double gpu_fwd = 0;
#ifdef WITH_CUDA
        TensorOptions gpu_opts = TensorOptions().with_device(Device::CUDA).with_dtype(Dtype::Float32);
        Tensor x_gpu     = Tensor::randn<float>(Shape{{tc.b, tc.s, tc.h}}, gpu_opts, 1337, 1.0f);
        Tensor gamma_gpu = Tensor::ones(Shape{{tc.h}}, gpu_opts);
        Tensor beta_gpu  = Tensor::zeros(Shape{{tc.h}}, gpu_opts);
        gpu_fwd = bench([&]{ autograd::layer_norm(x_gpu, gamma_gpu, beta_gpu, tc.h); });
#endif
        printf("%-42s  %8.3f ms  %8.3f ms\n", tc.label, cpu_fwd, gpu_fwd);
    }

    // ── LayerNorm Backward (via autograd with requires_grad) ──
    std::cout << "\n=== LayerNorm Backward (autograd forward+backward) ===" << std::endl;
    printf("%-42s  %10s  %10s\n", "Size", "CPU (ms)", "GPU (ms)");
    printf("──────────────────────────────────────────  ──────────  ──────────\n");

    for (auto& tc : cases) {
        TensorOptions cpu_opts = TensorOptions().with_device(Device::CPU).with_dtype(Dtype::Float32).with_req_grad(true);
        Tensor x_cpu     = Tensor::randn<float>(Shape{{tc.b, tc.s, tc.h}}, cpu_opts, 1337, 1.0f);
        Tensor gamma_cpu = Tensor::ones(Shape{{tc.h}}, cpu_opts);
        Tensor beta_cpu  = Tensor::zeros(Shape{{tc.h}}, cpu_opts);

        double cpu_bwd = bench([&]{
            Tensor out = autograd::layer_norm(x_cpu, gamma_cpu, beta_cpu, tc.h);
            Tensor grad = Tensor::randn<float>(out.shape(), TensorOptions().with_device(Device::CPU).with_dtype(Dtype::Float32), 99, 1.0f);
            out.backward(grad);
        });

        double gpu_bwd = 0;
#ifdef WITH_CUDA
        TensorOptions gpu_opts = TensorOptions().with_device(Device::CUDA).with_dtype(Dtype::Float32).with_req_grad(true);
        Tensor x_gpu     = Tensor::randn<float>(Shape{{tc.b, tc.s, tc.h}}, gpu_opts, 1337, 1.0f);
        Tensor gamma_gpu = Tensor::ones(Shape{{tc.h}}, gpu_opts);
        Tensor beta_gpu  = Tensor::zeros(Shape{{tc.h}}, gpu_opts);

        gpu_bwd = bench([&]{
            Tensor out = autograd::layer_norm(x_gpu, gamma_gpu, beta_gpu, tc.h);
            Tensor grad = Tensor::randn<float>(out.shape(), TensorOptions().with_device(Device::CUDA).with_dtype(Dtype::Float32), 99, 1.0f);
            out.backward(grad);
        });
#endif
        printf("%-42s  %8.3f ms  %8.3f ms\n", tc.label, cpu_bwd, gpu_bwd);
    }

    // ── Speedup summary for training size ──
    std::cout << "\n=== Summary — Training Size [8, 1024, 384] ===" << std::endl;
    {
        int b=8, s=1024, h=384;
        TensorOptions cpu_opts = TensorOptions().with_device(Device::CPU).with_dtype(Dtype::Float32);
        Tensor x_cpu = Tensor::randn<float>(Shape{{b,s,h}}, cpu_opts, 1337, 1.0f);
        Tensor g_cpu = Tensor::ones(Shape{{h}}, cpu_opts);
        Tensor b_cpu = Tensor::zeros(Shape{{h}}, cpu_opts);

        double ln_fwd_cpu = bench([&]{ autograd::layer_norm(x_cpu, g_cpu, b_cpu, h); });

        TensorOptions cpu_grad_opts = TensorOptions().with_device(Device::CPU).with_dtype(Dtype::Float32).with_req_grad(true);
        Tensor xg_cpu = Tensor::randn<float>(Shape{{b,s,h}}, cpu_grad_opts, 1337, 1.0f);
        Tensor gg_cpu = Tensor::ones(Shape{{h}}, cpu_grad_opts);
        Tensor bg_cpu = Tensor::zeros(Shape{{h}}, cpu_grad_opts);
        double ln_bwd_cpu = bench([&]{
            Tensor out = autograd::layer_norm(xg_cpu, gg_cpu, bg_cpu, h);
            Tensor grad = Tensor::randn<float>(out.shape(), cpu_opts, 99, 1.0f);
            out.backward(grad);
        });

#ifdef WITH_CUDA
        TensorOptions gpu_opts = TensorOptions().with_device(Device::CUDA).with_dtype(Dtype::Float32);
        Tensor x_gpu = Tensor::randn<float>(Shape{{b,s,h}}, gpu_opts, 1337, 1.0f);
        Tensor g_gpu = Tensor::ones(Shape{{h}}, gpu_opts);
        Tensor b_gpu = Tensor::zeros(Shape{{h}}, gpu_opts);

        double ln_fwd_gpu = bench([&]{ autograd::layer_norm(x_gpu, g_gpu, b_gpu, h); });

        TensorOptions gpu_grad_opts = TensorOptions().with_device(Device::CUDA).with_dtype(Dtype::Float32).with_req_grad(true);
        Tensor xg_gpu = Tensor::randn<float>(Shape{{b,s,h}}, gpu_grad_opts, 1337, 1.0f);
        Tensor gg_gpu = Tensor::ones(Shape{{h}}, gpu_grad_opts);
        Tensor bg_gpu = Tensor::zeros(Shape{{h}}, gpu_grad_opts);
        double ln_bwd_gpu = bench([&]{
            Tensor out = autograd::layer_norm(xg_gpu, gg_gpu, bg_gpu, h);
            Tensor grad = Tensor::randn<float>(out.shape(), gpu_opts, 99, 1.0f);
            out.backward(grad);
        });

        printf("  LN Forward:  CPU=%.3fms  GPU=%.3fms  → GPU %.1fx faster\n", ln_fwd_cpu, ln_fwd_gpu, ln_fwd_cpu/ln_fwd_gpu);
        printf("  LN Backward: CPU=%.3fms  GPU=%.3fms  → GPU %.1fx faster\n", ln_bwd_cpu, ln_bwd_gpu, ln_bwd_cpu/ln_bwd_gpu);
#else
        printf("  LN Forward:  CPU=%.3fms\n", ln_fwd_cpu);
        printf("  LN Backward: CPU=%.3fms\n", ln_bwd_cpu);
#endif
    }

    return 0;
}
