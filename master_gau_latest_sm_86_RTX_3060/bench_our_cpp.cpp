#include "TensorLib.h"
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
using namespace OwnTensor;
using namespace std;

int main() {
    TensorOptions opts;
    // opts.with_device(DeviceIndex(Device::CUDA, 0)).with_dtype(Dtype::Float32);
    opts = opts.with_device(DeviceIndex(Device::CUDA, 0)).with_dtype(Dtype::Float32);
    // Same tensor size, initialized with random normal values. Seed 1337.
    Tensor x = Tensor::randn<float>(Shape{{8, 1024, 384}}, opts, 1337, 1.0f);
    
    // Warmup
    for (int i = 0; i < 50; ++i) {
        Tensor res = reduce_sum(x, {0});
    }
    
    // Synchronize to ensure warmup finishes before timing
    cudaDeviceSynchronize();
    
    int iterations = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        Tensor res = reduce_sum(x, {0});
    }
    
    // Synchronize to ensure all kernels finish before stopping the timer
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = end - start;
    std::cout << "[master_gau] Average Time over " << iterations << " ops: " 
              << (diff.count() / iterations) << " seconds" << std::endl;

    return 0;
}
