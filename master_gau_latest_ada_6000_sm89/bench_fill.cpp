// Bench: memset vs fill — CPU and GPU zero paths.
// Build:  see tail of this file (one line).
// Run:    CUDA_VISIBLE_DEVICES=6 ./bench_fill
//
// Measures ms/op and GB/s for each method at sizes common to DL training.

#include "core/Tensor.h"
#include "core/TensorDataManip.h"
#include "ops/helpers/FillKernels.h"
#include "device/DeviceCore.h"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace OwnTensor;
using Clock = std::chrono::high_resolution_clock;

static double now_ms() {
    auto t = Clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(t).count();
}

// One size + method row.
struct Row {
    const char* label;
    int64_t n;            // float elements
    double ms_cpu_memset;
    double ms_cpu_fill;
    double ms_gpu_memset;
    double ms_gpu_fill;
};

static void bench_cpu(Row& r, const TensorOptions& opts_cpu, int warm, int iters) {
    Tensor t(Shape{{r.n}}, opts_cpu);
    void*  ptr = t.data();
    size_t nbytes = r.n * sizeof(float);

    for (int i = 0; i < warm; ++i) std::memset(ptr, 0, nbytes);
    double t0 = now_ms();
    for (int i = 0; i < iters; ++i) std::memset(ptr, 0, nbytes);
    r.ms_cpu_memset = (now_ms() - t0) / iters;

    for (int i = 0; i < warm; ++i) t.fill<float>(0.0f);
    t0 = now_ms();
    for (int i = 0; i < iters; ++i) t.fill<float>(0.0f);
    r.ms_cpu_fill = (now_ms() - t0) / iters;
}

static void bench_gpu(Row& r, const TensorOptions& opts_gpu, int warm, int iters) {
    Tensor t(Shape{{r.n}}, opts_gpu);
    void*  ptr    = t.data();
    size_t nbytes = r.n * sizeof(float);
    cudaStream_t s = cuda::getCurrentStream();

    for (int i = 0; i < warm; ++i) cudaMemsetAsync(ptr, 0, nbytes, s);
    cudaStreamSynchronize(s);
    double t0 = now_ms();
    for (int i = 0; i < iters; ++i) cudaMemsetAsync(ptr, 0, nbytes, s);
    cudaStreamSynchronize(s);
    r.ms_gpu_memset = (now_ms() - t0) / iters;

    for (int i = 0; i < warm; ++i)
        cuda::fill_cuda_launch<float>(reinterpret_cast<float*>(ptr), 0.0f, r.n, s);
    cudaStreamSynchronize(s);
    t0 = now_ms();
    for (int i = 0; i < iters; ++i)
        cuda::fill_cuda_launch<float>(reinterpret_cast<float*>(ptr), 0.0f, r.n, s);
    cudaStreamSynchronize(s);
    r.ms_gpu_fill = (now_ms() - t0) / iters;
}

int main() {
    // Sizes span 4 KB -> 1 GB (float). Includes GPT-2-ish weights.
    std::vector<Row> rows = {
        { "1K float (4 KB)",       1'000 },
        { "64K float (256 KB)",   64'000 },
        { "1M float (4 MB)",   1'000'000 },
        { "attention B*T*E (16*1024*768 = 12.5M, 50 MB)", 16LL*1024*768 },
        { "MLP weight (3072*768 = 2.3M, 9 MB)",           3072LL*768 },
        { "embed (50304*768 = 38.6M, 154 MB)",            50304LL*768 },
        { "256M float (1 GB)",          256'000'000 },
    };

    auto cpu = TensorOptions().with_dtype(Dtype::Float32).with_device(Device::CPU);
    auto gpu = TensorOptions().with_dtype(Dtype::Float32).with_device(Device::CUDA);

    const int WARM  = 10;
    const int ITERS = 100;

    printf("%-55s %12s %12s %12s %12s\n",
           "size", "cpu_memset_ms", "cpu_fill_ms", "gpu_memset_ms", "gpu_fill_ms");
    printf("%.*s\n", 120, "-------------------------------------------------------------"
                          "-------------------------------------------------------------");

    for (auto& r : rows) {
        bench_cpu(r, cpu, WARM, ITERS);
        bench_gpu(r, gpu, WARM, ITERS);
        double gb = r.n * sizeof(float) / 1e9;
        printf("%-55s %9.4f ms  %9.4f ms  %9.4f ms  %9.4f ms"
               "   | cpu_memset=%6.1f GB/s  cpu_fill=%6.1f GB/s  "
               "gpu_memset=%6.1f GB/s  gpu_fill=%6.1f GB/s\n",
               r.label,
               r.ms_cpu_memset, r.ms_cpu_fill,
               r.ms_gpu_memset, r.ms_gpu_fill,
               gb / (r.ms_cpu_memset / 1000),
               gb / (r.ms_cpu_fill   / 1000),
               gb / (r.ms_gpu_memset / 1000),
               gb / (r.ms_gpu_fill   / 1000));
    }
    return 0;
}

// Build:
//   make snippet-runner   # or same rule as gpt2_attn_navin.cpp
// or direct:
//   g++ -Iinclude -I/usr/local/cuda-13.0/include -DWITH_CUDA -std=c++2a -O3 -fopenmp \
//       -mavx2 -mfma -mf16c bench_fill.cpp \
//       -Llib -ltensor -L/usr/local/cuda-13.0/lib64 \
//       -lcudart -ltbb -lcurand -lcublas -lcublasLt -lgomp -lnvidia-ml \
//       -Wl,-rpath,'$ORIGIN/lib' -o bench_fill
//
// Or just:  CUDA_VISIBLE_DEVICES=6 make run-snippet FILE=bench_fill.cpp
