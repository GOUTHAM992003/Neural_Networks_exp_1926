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

double bench(auto fn, int warmup = 5, int iters = 50) {
    for (int i = 0; i < warmup; ++i) fn();
    std::vector<double> times;
    for (int i = 0; i < iters; ++i) {
        auto s = std::chrono::high_resolution_clock::now();
        fn();
        auto e = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::micro>(e - s).count());
    }
    std::sort(times.begin(), times.end());
    int trim = std::max(1, (int)times.size() / 10);
    double sum = 0; int cnt = 0;
    for (int i = trim; i < (int)times.size() - trim; ++i) { sum += times[i]; cnt++; }
    return sum / cnt;
}

Tensor make_rand(std::vector<int64_t> shape, Dtype dtype = Dtype::Float32) {
    Tensor t({Shape(shape)}, TensorOptions().with_dtype(dtype));
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    int64_t n = t.numel();
    if (dtype == Dtype::Float32) {
        float* d = t.data<float>();
        for (int64_t i = 0; i < n; ++i) d[i] = dist(gen);
    } else {
        std::uniform_real_distribution<double> dd(-100.0, 100.0);
        double* d = t.data<double>();
        for (int64_t i = 0; i < n; ++i) d[i] = dd(gen);
    }
    return t;
}

int main() {
    std::cout << "OwnTensor, Threads: " << omp_get_max_threads() << "\n\n";

    // FULL REDUCTION
    std::cout << std::string(80, '=') << "\n";
    std::cout << "FULL REDUCTION: argmax() — scalar output\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << std::setw(12) << "Size" << " | " << std::setw(8) << "Dtype"
              << " | " << std::setw(14) << "argmax (μs)" << " | " << std::setw(14) << "argmin (μs)" << "\n";
    std::cout << std::string(60, '-') << "\n";

    for (auto dtype : {Dtype::Float32, Dtype::Float64}) {
        std::string dn = (dtype == Dtype::Float32) ? "f32" : "f64";
        for (auto size : {10000LL, 100000LL, 1000000LL, 10000000LL, 50000000LL, 100000000LL}) {
            auto t = make_rand({size}, dtype);
            double am = bench([&]{ reduce_argmax(t); });
            double ai = bench([&]{ reduce_argmin(t); });
            std::cout << std::setw(12) << size << " | " << std::setw(8) << dn
                      << " | " << std::setw(12) << std::fixed << std::setprecision(1) << am << "μs"
                      << " | " << std::setw(12) << ai << "μs\n";
        }
        std::cout << "\n";
    }

    // PARTIAL REDUCTION — last dim
    std::cout << std::string(80, '=') << "\n";
    std::cout << "PARTIAL REDUCTION: argmax(dim=-1) — reduce last dimension\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << std::setw(20) << "Shape" << " | " << std::setw(8) << "Dtype"
              << " | " << std::setw(14) << "argmax (μs)" << " | " << std::setw(14) << "argmin (μs)" << "\n";
    std::cout << std::string(70, '-') << "\n";

    std::vector<std::pair<int64_t,int64_t>> shapes = {{100,100000},{1000,10000},{10000,1000},{64,1000000},{256,100000}};
    for (auto [r, c] : shapes) {
        auto t = make_rand({r, c});
        double am = bench([&]{ reduce_argmax(t, {1}); });
        double ai = bench([&]{ reduce_argmin(t, {1}); });
        std::string sh = "(" + std::to_string(r) + "," + std::to_string(c) + ")";
        std::cout << std::setw(20) << sh << " | " << std::setw(8) << "f32"
                  << " | " << std::setw(12) << std::fixed << std::setprecision(1) << am << "μs"
                  << " | " << std::setw(12) << ai << "μs\n";
    }
    std::cout << "\n";

    // PARTIAL REDUCTION — first dim (outer)
    std::cout << std::string(80, '=') << "\n";
    std::cout << "PARTIAL REDUCTION: argmax(dim=0) — reduce first dimension (outer)\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << std::setw(20) << "Shape" << " | " << std::setw(8) << "Dtype"
              << " | " << std::setw(14) << "argmax (μs)" << " | " << std::setw(14) << "argmin (μs)" << "\n";
    std::cout << std::string(70, '-') << "\n";

    std::vector<std::pair<int64_t,int64_t>> shapes2 = {{100000,100},{10000,1000},{1000,10000},{1000000,64},{100000,256}};
    for (auto [r, c] : shapes2) {
        auto t = make_rand({r, c});
        double am = bench([&]{ reduce_argmax(t, {0}); });
        double ai = bench([&]{ reduce_argmin(t, {0}); });
        std::string sh = "(" + std::to_string(r) + "," + std::to_string(c) + ")";
        std::cout << std::setw(20) << sh << " | " << std::setw(8) << "f32"
                  << " | " << std::setw(12) << std::fixed << std::setprecision(1) << am << "μs"
                  << " | " << std::setw(12) << ai << "μs\n";
    }
    return 0;
}
