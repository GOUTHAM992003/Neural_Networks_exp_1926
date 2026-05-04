#include <iostream>
#include <vector>
#include <chrono>
#include "TensorLib.h"
#include "checkpointing/Checkpointing.h"

using namespace OwnTensor;

void benchmark_staging(size_t n_elements) {
    std::cout << "Benchmarking staging for " << (n_elements * 4 / (1024 * 1024)) << " MB..." << std::endl;
    
    // 1. Create Source on GPU
    Tensor gpu_src = Tensor::randn(Shape{{(int64_t)n_elements}}, TensorOptions().with_device(Device::CUDA));
    
    // --- Paged Staging (Current Method) ---
    auto start_paged = std::chrono::steady_clock::now();
    Tensor cpu_paged = gpu_src.to_cpu();
    auto end_paged = std::chrono::steady_clock::now();
    auto ms_paged = std::chrono::duration_cast<std::chrono::milliseconds>(end_paged - start_paged).count();
    
    // --- Pinned Staging (Optimized Method) ---
    auto start_pinned = std::chrono::steady_clock::now();
    TensorOptions opts;
    opts.device = Device::CPU;
    opts.pinten = Pinned_Flag::Default;
    Tensor cpu_pinned(gpu_src.shape(), opts);
    cpu_pinned.copy_(gpu_src);
    auto end_pinned = std::chrono::steady_clock::now();
    auto ms_pinned = std::chrono::duration_cast<std::chrono::milliseconds>(end_pinned - start_pinned).count();
    
    std::cout << "  Paged Staging (Blocking):  " << ms_paged << " ms" << std::endl;
    std::cout << "  Pinned Staging (Blocking): " << ms_pinned << " ms" << std::endl;
    
    float speedup = (float)ms_paged / ms_pinned;
    std::cout << "  Estimated Speedup: " << speedup << "x" << std::endl;
}

int main() {
    try {
        // Benchmark 1GB (Approx model size)
        benchmark_staging(256 * 1024 * 1024);
        
        // Benchmark 3GB (Approx total state size for 40-layer GPT-2)
        benchmark_staging(768 * 1024 * 1024);
        
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
    }
    return 0;
}
