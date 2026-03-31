/**
 * Benchmark: Full Reduction argmax/argmin — Our Library (OwnTensor)
 * =================================================================
 * Tests our Strategy 2 (SplitReduction) fix for full reduction.
 * Compare results with pytorch_vs_ours_argmax_bench.py
 *
 * Same sizes: 10K, 100K, 1M, 10M, 50M, 100M
 * Same dtypes: float32, float64
 * Same operation: full reduction argmax/argmin (no axis = reduce all)
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>
#include <omp.h>

#include "core/Tensor.h"
#include "ops/UnaryOps/Reduction.h"

using namespace OwnTensor;

struct BenchResult {
    double time_us;
    int64_t result_idx;
};

BenchResult bench_argmax(const Tensor& tensor, int warmup = 5, int iters = 50) {
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        auto r = reduce_argmax(tensor);
    }

    std::vector<double> times;
    int64_t result_idx = -1;

    for (int i = 0; i < iters; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = reduce_argmax(tensor);
        auto end = std::chrono::high_resolution_clock::now();

        double us = std::chrono::duration<double, std::micro>(end - start).count();
        times.push_back(us);

        if (i == 0) result_idx = result.data<int64_t>()[0];
    }

    // Remove top/bottom 10% outliers
    std::sort(times.begin(), times.end());
    int trim = std::max(1, (int)times.size() / 10);
    double sum = 0;
    int count = 0;
    for (int i = trim; i < (int)times.size() - trim; ++i) {
        sum += times[i];
        count++;
    }
    return {sum / count, result_idx};
}

BenchResult bench_argmin(const Tensor& tensor, int warmup = 5, int iters = 50) {
    for (int i = 0; i < warmup; ++i) {
        auto r = reduce_argmin(tensor);
    }

    std::vector<double> times;
    int64_t result_idx = -1;

    for (int i = 0; i < iters; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = reduce_argmin(tensor);
        auto end = std::chrono::high_resolution_clock::now();

        double us = std::chrono::duration<double, std::micro>(end - start).count();
        times.push_back(us);

        if (i == 0) result_idx = result.data<int64_t>()[0];
    }

    std::sort(times.begin(), times.end());
    int trim = std::max(1, (int)times.size() / 10);
    double sum = 0;
    int count = 0;
    for (int i = trim; i < (int)times.size() - trim; ++i) {
        sum += times[i];
        count++;
    }
    return {sum / count, result_idx};
}

double bench_argmax_dim(const Tensor& tensor, const std::vector<int64_t>& axes, int warmup = 5, int iters = 50) {
    for (int i = 0; i < warmup; ++i) {
        auto r = reduce_argmax(tensor, axes);
    }

    std::vector<double> times;
    for (int i = 0; i < iters; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = reduce_argmax(tensor, axes);
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    std::sort(times.begin(), times.end());
    int trim = std::max(1, (int)times.size() / 10);
    double sum = 0;
    int count = 0;
    for (int i = trim; i < (int)times.size() - trim; ++i) {
        sum += times[i];
        count++;
    }
    return sum / count;
}

Tensor make_random_tensor(int64_t size, Dtype dtype) {
    Tensor t({Shape({size})}, TensorOptions().with_dtype(dtype));

    // Fill with random data
    std::mt19937 gen(42);  // Fixed seed for reproducibility

    if (dtype == Dtype::Float32) {
        std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
        float* data = t.data<float>();
        for (int64_t i = 0; i < size; ++i) data[i] = dist(gen);
    } else if (dtype == Dtype::Float64) {
        std::uniform_real_distribution<double> dist(-100.0, 100.0);
        double* data = t.data<double>();
        for (int64_t i = 0; i < size; ++i) data[i] = dist(gen);
    }
    return t;
}

Tensor make_random_2d(int64_t rows, int64_t cols, Dtype dtype) {
    Tensor t({Shape({rows, cols})}, TensorOptions().with_dtype(dtype));
    std::mt19937 gen(42);
    if (dtype == Dtype::Float32) {
        std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
        float* data = t.data<float>();
        for (int64_t i = 0; i < rows * cols; ++i) data[i] = dist(gen);
    }
    return t;
}

int main() {
    int max_threads = omp_get_max_threads();

    std::cout << std::string(90, '=') << std::endl;
    std::cout << "BENCHMARK: Full Reduction argmax/argmin — OwnTensor (Our Library)" << std::endl;
    std::cout << std::string(90, '=') << std::endl;
    std::cout << "CPU threads: " << max_threads << std::endl;
    std::cout << "Strategy: Strategy 2 (SplitReduction) for full reduction" << std::endl;
    std::cout << "GRAIN_SIZE: 32768" << std::endl;
    std::cout << std::endl;

    // =========================================================
    // TEST 1: Full Reduction argmax — various sizes
    // =========================================================
    std::cout << std::string(90, '-') << std::endl;
    std::cout << "TEST 1: Full Reduction argmax (reduce all dims) — 1D contiguous tensors" << std::endl;
    std::cout << std::string(90, '-') << std::endl;
    std::cout << std::setw(12) << "Size" << " | "
              << std::setw(10) << "Dtype" << " | "
              << std::setw(14) << "argmax (μs)" << " | "
              << std::setw(14) << "argmin (μs)" << " | "
              << std::setw(14) << "Elements/μs" << std::endl;
    std::cout << std::string(90, '-') << std::endl;

    std::vector<int64_t> sizes = {10000, 100000, 1000000, 10000000, 50000000, 100000000};
    std::vector<std::pair<Dtype, std::string>> dtypes = {
        {Dtype::Float32, "float32"},
        {Dtype::Float64, "float64"}
    };

    for (auto& [dtype, dname] : dtypes) {
        for (auto size : sizes) {
            auto tensor = make_random_tensor(size, dtype);

            auto [t_argmax, idx_max] = bench_argmax(tensor);
            auto [t_argmin, idx_min] = bench_argmin(tensor);
            double throughput = size / t_argmax;

            std::cout << std::setw(12) << size << " | "
                      << std::setw(10) << dname << " | "
                      << std::setw(11) << std::fixed << std::setprecision(1) << t_argmax << "μs | "
                      << std::setw(11) << t_argmin << "μs | "
                      << std::setw(11) << throughput << "M/s" << std::endl;
        }
        std::cout << std::endl;
    }

    // =========================================================
    // TEST 2: Thread count impact (our library should scale!)
    // =========================================================
    std::cout << std::string(90, '-') << std::endl;
    std::cout << "TEST 2: Thread count impact — Full reduction argmax" << std::endl;
    std::cout << std::string(90, '-') << std::endl;

    int64_t test_size = 10000000;
    auto tensor = make_random_tensor(test_size, Dtype::Float32);

    std::cout << "Tensor size: " << test_size << " float32 elements" << std::endl;
    std::cout << std::setw(10) << "Threads" << " | "
              << std::setw(14) << "argmax (μs)" << " | "
              << std::setw(16) << "Speedup vs 1T" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<int> thread_counts = {1, 2, 4, 8, 14, 20, 28};
    double base_time = -1;

    for (int nt : thread_counts) {
        if (nt > max_threads) continue;
        omp_set_num_threads(nt);

        auto [t, idx] = bench_argmax(tensor, 10, 100);
        if (base_time < 0) base_time = t;
        double speedup = base_time / t;

        std::cout << std::setw(10) << nt << " | "
                  << std::setw(11) << std::fixed << std::setprecision(1) << t << "μs | "
                  << std::setw(14) << std::setprecision(2) << speedup << "x" << std::endl;
    }
    omp_set_num_threads(max_threads);
    std::cout << std::endl;

    // =========================================================
    // TEST 3: Full vs Partial reduction
    // =========================================================
    std::cout << std::string(90, '-') << std::endl;
    std::cout << "TEST 3: Full reduction vs Partial reduction (dim=1)" << std::endl;
    std::cout << std::string(90, '-') << std::endl;

    int64_t rows = 100, cols = 1000000;
    auto tensor_2d = make_random_2d(rows, cols, Dtype::Float32);
    auto tensor_flat = make_random_tensor(rows * cols, Dtype::Float32);

    auto [t_full, _] = bench_argmax(tensor_flat);
    double t_partial = bench_argmax_dim(tensor_2d, {1});

    std::cout << "Data: (" << rows << ", " << cols << ") float32 = "
              << rows * cols << " elements" << std::endl;
    std::cout << "Full reduction (argmax all):        " << std::setw(10) << std::setprecision(1) << t_full << " μs" << std::endl;
    std::cout << "Partial reduction (argmax dim=1):   " << std::setw(10) << t_partial << " μs" << std::endl;
    std::cout << "Ratio (full/partial):               " << std::setw(10) << std::setprecision(2) << t_full / t_partial << "x" << std::endl;
    std::cout << std::endl;

    // =========================================================
    // TEST 4: The comparison with PyTorch
    // =========================================================
    std::cout << std::string(90, '-') << std::endl;
    std::cout << "TEST 4: Head-to-head comparison point (50M elements)" << std::endl;
    std::cout << std::string(90, '-') << std::endl;

    auto tensor_50m = make_random_tensor(50000000, Dtype::Float32);
    omp_set_num_threads(max_threads);
    auto [t_ours, idx_ours] = bench_argmax(tensor_50m, 5, 30);

    std::cout << "50M float32 elements, full reduction argmax:" << std::endl;
    std::cout << "Our library (" << max_threads << " threads, Strategy 2): "
              << std::setprecision(1) << t_ours << " μs" << std::endl;
    std::cout << "PyTorch (from Python bench):                  46,581.4 μs (1 thread, lastdim bug)" << std::endl;
    std::cout << "Speedup (PyTorch / Ours):                     "
              << std::setprecision(2) << 46581.4 / t_ours << "x" << std::endl;

    std::cout << std::endl;
    std::cout << std::string(90, '=') << std::endl;
    std::cout << "BENCHMARK COMPLETE" << std::endl;
    std::cout << std::string(90, '=') << std::endl;

    return 0;
}
