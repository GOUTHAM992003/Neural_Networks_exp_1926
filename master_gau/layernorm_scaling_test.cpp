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
    std::cout << "OwnTensor — LayerNorm + RMSNorm Scaling Test (CPU vs GPU)\n" << std::endl;

    TestCase cases[] = {
        {1,   128,  128,  "Tiny    [1, 128, 128]     = 16K   "},
        {1,   512,  512,  "Small   [1, 512, 512]     = 262K  "},
        {8,  1024,  384,  "Medium  [8, 1024, 384]    = 3.1M  "},  // GPT-2 training size
        {8,  1024,  768,  "GPT2    [8, 1024, 768]    = 6.3M  "},
        {32, 1024,  768,  "Large   [32, 1024, 768]   = 25.2M "},
        {64, 1024,  768,  "XLarge  [64, 1024, 768]   = 50.3M "},
        {64, 2048, 1024,  "Huge    [64, 2048, 1024]  = 134.2M"},
    };

    // ── LayerNorm Forward + Backward ──
    std::cout << "=== LayerNorm Forward ===" << std::endl;
    printf("%-42s  %10s  %10s\n", "Size", "CPU (ms)", "GPU (ms)");
    printf("──────────────────────────────────────────  ──────────  ──────────\n");

    for (auto& tc : cases) {
        TensorOptions cpu_opts = TensorOptions().with_device(Device::CPU).with_dtype(Dtype::Float32);
        Tensor x_cpu     = Tensor::randn<float>(Shape{{tc.b, tc.s, tc.h}}, cpu_opts, 1337, 1.0f);
        Tensor gamma_cpu = Tensor::ones(Shape{{tc.h}}, cpu_opts);
        Tensor beta_cpu  = Tensor::zeros(Shape{{tc.h}}, cpu_opts);

        double cpu_fwd = bench([&]{ layer_norm_forward(x_cpu, gamma_cpu, beta_cpu, tc.h); });

        double gpu_fwd = 0;
#ifdef WITH_CUDA
        TensorOptions gpu_opts = TensorOptions().with_device(Device::CUDA).with_dtype(Dtype::Float32);
        Tensor x_gpu     = Tensor::randn<float>(Shape{{tc.b, tc.s, tc.h}}, gpu_opts, 1337, 1.0f);
        Tensor gamma_gpu = Tensor::ones(Shape{{tc.h}}, gpu_opts);
        Tensor beta_gpu  = Tensor::zeros(Shape{{tc.h}}, gpu_opts);
        gpu_fwd = bench([&]{ layer_norm_forward(x_gpu, gamma_gpu, beta_gpu, tc.h); });
#endif
        printf("%-42s  %8.3f ms  %8.3f ms\n", tc.label, cpu_fwd, gpu_fwd);
    }

    std::cout << "\n=== LayerNorm Backward ===" << std::endl;
    printf("%-42s  %10s  %10s\n", "Size", "CPU (ms)", "GPU (ms)");
    printf("──────────────────────────────────────────  ──────────  ──────────\n");

    for (auto& tc : cases) {
        TensorOptions cpu_opts = TensorOptions().with_device(Device::CPU).with_dtype(Dtype::Float32);
        Tensor x_cpu     = Tensor::randn<float>(Shape{{tc.b, tc.s, tc.h}}, cpu_opts, 1337, 1.0f);
        Tensor gamma_cpu = Tensor::ones(Shape{{tc.h}}, cpu_opts);
        Tensor beta_cpu  = Tensor::zeros(Shape{{tc.h}}, cpu_opts);
        Tensor grad_cpu  = Tensor::randn<float>(Shape{{tc.b, tc.s, tc.h}}, cpu_opts, 99, 1.0f);

        auto fwd = layer_norm_forward(x_cpu, gamma_cpu, beta_cpu, tc.h);
        double cpu_bwd = bench([&]{ layer_norm_backward(grad_cpu, x_cpu, fwd.mean, fwd.rstd, gamma_cpu, tc.h); });

        double gpu_bwd = 0;
#ifdef WITH_CUDA
        TensorOptions gpu_opts = TensorOptions().with_device(Device::CUDA).with_dtype(Dtype::Float32);
        Tensor x_gpu     = Tensor::randn<float>(Shape{{tc.b, tc.s, tc.h}}, gpu_opts, 1337, 1.0f);
        Tensor gamma_gpu = Tensor::ones(Shape{{tc.h}}, gpu_opts);
        Tensor beta_gpu  = Tensor::zeros(Shape{{tc.h}}, gpu_opts);
        Tensor grad_gpu  = Tensor::randn<float>(Shape{{tc.b, tc.s, tc.h}}, gpu_opts, 99, 1.0f);
        auto fwd_gpu = layer_norm_forward(x_gpu, gamma_gpu, beta_gpu, tc.h);
        gpu_bwd = bench([&]{ layer_norm_backward(grad_gpu, x_gpu, fwd_gpu.mean, fwd_gpu.rstd, gamma_gpu, tc.h); });
#endif
        printf("%-42s  %8.3f ms  %8.3f ms\n", tc.label, cpu_bwd, gpu_bwd);
    }

    // ── RMSNorm Forward + Backward ──
    std::cout << "\n=== RMSNorm Forward ===" << std::endl;
    printf("%-42s  %10s  %10s\n", "Size", "CPU (ms)", "GPU (ms)");
    printf("──────────────────────────────────────────  ──────────  ──────────\n");

    for (auto& tc : cases) {
        TensorOptions cpu_opts = TensorOptions().with_device(Device::CPU).with_dtype(Dtype::Float32);
        Tensor x_cpu     = Tensor::randn<float>(Shape{{tc.b, tc.s, tc.h}}, cpu_opts, 1337, 1.0f);
        Tensor gamma_cpu = Tensor::ones(Shape{{tc.h}}, cpu_opts);

        double cpu_fwd = bench([&]{ rms_norm_forward(x_cpu, gamma_cpu, tc.h); });

        double gpu_fwd = 0;
#ifdef WITH_CUDA
        TensorOptions gpu_opts = TensorOptions().with_device(Device::CUDA).with_dtype(Dtype::Float32);
        Tensor x_gpu     = Tensor::randn<float>(Shape{{tc.b, tc.s, tc.h}}, gpu_opts, 1337, 1.0f);
        Tensor gamma_gpu = Tensor::ones(Shape{{tc.h}}, gpu_opts);
        gpu_fwd = bench([&]{ rms_norm_forward(x_gpu, gamma_gpu, tc.h); });
#endif
        printf("%-42s  %8.3f ms  %8.3f ms\n", tc.label, cpu_fwd, gpu_fwd);
    }

    std::cout << "\n=== RMSNorm Backward ===" << std::endl;
    printf("%-42s  %10s  %10s\n", "Size", "CPU (ms)", "GPU (ms)");
    printf("──────────────────────────────────────────  ──────────  ──────────\n");

    for (auto& tc : cases) {
        TensorOptions cpu_opts = TensorOptions().with_device(Device::CPU).with_dtype(Dtype::Float32);
        Tensor x_cpu     = Tensor::randn<float>(Shape{{tc.b, tc.s, tc.h}}, cpu_opts, 1337, 1.0f);
        Tensor gamma_cpu = Tensor::ones(Shape{{tc.h}}, cpu_opts);
        Tensor grad_cpu  = Tensor::randn<float>(Shape{{tc.b, tc.s, tc.h}}, cpu_opts, 99, 1.0f);

        auto fwd = rms_norm_forward(x_cpu, gamma_cpu, tc.h);
        double cpu_bwd = bench([&]{ rms_norm_backward(grad_cpu, x_cpu, fwd.rstd, gamma_cpu, tc.h); });

        double gpu_bwd = 0;
#ifdef WITH_CUDA
        TensorOptions gpu_opts = TensorOptions().with_device(Device::CUDA).with_dtype(Dtype::Float32);
        Tensor x_gpu     = Tensor::randn<float>(Shape{{tc.b, tc.s, tc.h}}, gpu_opts, 1337, 1.0f);
        Tensor gamma_gpu = Tensor::ones(Shape{{tc.h}}, gpu_opts);
        Tensor grad_gpu  = Tensor::randn<float>(Shape{{tc.b, tc.s, tc.h}}, gpu_opts, 99, 1.0f);
        auto fwd_gpu = rms_norm_forward(x_gpu, gamma_gpu, tc.h);
        gpu_bwd = bench([&]{ rms_norm_backward(grad_gpu, x_gpu, fwd_gpu.rstd, gamma_gpu, tc.h); });
#endif
        printf("%-42s  %8.3f ms  %8.3f ms\n", tc.label, cpu_bwd, gpu_bwd);
    }

    // ── Speedup summary for training size ──
    std::cout << "\n=== Speedup Summary (GPU/CPU) — Training Size [8, 1024, 384] ===" << std::endl;
    {
        int b=8, s=1024, h=384;
        TensorOptions cpu_opts = TensorOptions().with_device(Device::CPU).with_dtype(Dtype::Float32);
        Tensor x_cpu = Tensor::randn<float>(Shape{{b,s,h}}, cpu_opts, 1337, 1.0f);
        Tensor g_cpu = Tensor::ones(Shape{{h}}, cpu_opts);
        Tensor b_cpu = Tensor::zeros(Shape{{h}}, cpu_opts);
        Tensor grad_cpu = Tensor::randn<float>(Shape{{b,s,h}}, cpu_opts, 99, 1.0f);

        double ln_fwd_cpu = bench([&]{ layer_norm_forward(x_cpu, g_cpu, b_cpu, h); });
        auto fwd = layer_norm_forward(x_cpu, g_cpu, b_cpu, h);
        double ln_bwd_cpu = bench([&]{ layer_norm_backward(grad_cpu, x_cpu, fwd.mean, fwd.rstd, g_cpu, h); });
        double rms_fwd_cpu = bench([&]{ rms_norm_forward(x_cpu, g_cpu, h); });
        auto rfwd = rms_norm_forward(x_cpu, g_cpu, h);
        double rms_bwd_cpu = bench([&]{ rms_norm_backward(grad_cpu, x_cpu, rfwd.rstd, g_cpu, h); });

#ifdef WITH_CUDA
        TensorOptions gpu_opts = TensorOptions().with_device(Device::CUDA).with_dtype(Dtype::Float32);
        Tensor x_gpu = Tensor::randn<float>(Shape{{b,s,h}}, gpu_opts, 1337, 1.0f);
        Tensor g_gpu = Tensor::ones(Shape{{h}}, gpu_opts);
        Tensor b_gpu = Tensor::zeros(Shape{{h}}, gpu_opts);
        Tensor grad_gpu = Tensor::randn<float>(Shape{{b,s,h}}, gpu_opts, 99, 1.0f);

        double ln_fwd_gpu = bench([&]{ layer_norm_forward(x_gpu, g_gpu, b_gpu, h); });
        auto fwd_g = layer_norm_forward(x_gpu, g_gpu, b_gpu, h);
        double ln_bwd_gpu = bench([&]{ layer_norm_backward(grad_gpu, x_gpu, fwd_g.mean, fwd_g.rstd, g_gpu, h); });
        double rms_fwd_gpu = bench([&]{ rms_norm_forward(x_gpu, g_gpu, h); });
        auto rfwd_g = rms_norm_forward(x_gpu, g_gpu, h);
        double rms_bwd_gpu = bench([&]{ rms_norm_backward(grad_gpu, x_gpu, rfwd_g.rstd, g_gpu, h); });

        printf("  LN  Forward:  CPU=%.3fms  GPU=%.3fms  → %.1fx\n", ln_fwd_cpu, ln_fwd_gpu, ln_fwd_cpu/ln_fwd_gpu);
        printf("  LN  Backward: CPU=%.3fms  GPU=%.3fms  → %.1fx\n", ln_bwd_cpu, ln_bwd_gpu, ln_bwd_cpu/ln_bwd_gpu);
        printf("  RMS Forward:  CPU=%.3fms  GPU=%.3fms  → %.1fx\n", rms_fwd_cpu, rms_fwd_gpu, rms_fwd_cpu/rms_fwd_gpu);
        printf("  RMS Backward: CPU=%.3fms  GPU=%.3fms  → %.1fx\n", rms_bwd_cpu, rms_bwd_gpu, rms_bwd_cpu/rms_bwd_gpu);
#endif
    }

    return 0;
}
