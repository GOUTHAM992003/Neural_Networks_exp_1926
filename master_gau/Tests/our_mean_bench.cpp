// Benchmark: Our reduce_kernel_mean — speed + precision
// Compile: g++ -O3 -mavx2 -mfma -mf16c -fopenmp -std=c++20 -I../include -L../lib -ltensor -o our_mean_bench tests/bench_our_mean.cpp

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include "core/Tensor.h"
#include "ops/UnaryOps/Reduction.h"

using namespace OwnTensor;

template<typename F>
double bench(F fn, int warmup = 5, int iters = 30) {
    for (int i = 0; i < warmup; ++i) fn();
    std::vector<double> times;
    for (int i = 0; i < iters; ++i) {
        auto s = std::chrono::high_resolution_clock::now();
        fn();
        auto e = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::micro>(e - s).count());
    }
    std::sort(times.begin(), times.end());
    return times[iters / 2];
}

// Fill tensor with random data
void fill_randn(Tensor& t, std::mt19937& rng) {
    float* d = t.data<float>();
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int64_t i = 0; i < t.numel(); ++i) d[i] = dist(rng);
}

void fill_randn_double(Tensor& t, std::mt19937& rng) {
    double* d = t.data<double>();
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int64_t i = 0; i < t.numel(); ++i) d[i] = dist(rng);
}

void add_nan(Tensor& t, double pct, std::mt19937& rng) {
    float* d = t.data<float>();
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (int64_t i = 0; i < t.numel(); ++i)
        if (u(rng) < pct) d[i] = std::nanf("");
}

void add_nan_double(Tensor& t, double pct, std::mt19937& rng) {
    double* d = t.data<double>();
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int64_t i = 0; i < t.numel(); ++i)
        if (u(rng) < pct) d[i] = std::nan("");
}

// Compute ground truth mean in fp64
double ground_truth_mean(const float* data, int64_t n) {
    double sum = 0;
    for (int64_t i = 0; i < n; ++i) sum += (double)data[i];
    return sum / n;
}

double ground_truth_nanmean(const float* data, int64_t n) {
    double sum = 0; int64_t count = 0;
    for (int64_t i = 0; i < n; ++i)
        if (!std::isnan(data[i])) { sum += (double)data[i]; ++count; }
    return count > 0 ? sum / count : std::nan("");
}

void print_header(const char* title) {
    printf("\n====================================================================================================\n");
    printf("  %s\n", title);
    printf("====================================================================================================\n");
}

void print_sub(const char* title) {
    printf("\n--- %s ---\n", title);
}

int main() {
    printf("CPU: %d threads (i7-14700K)\n", omp_get_max_threads());
    std::mt19937 rng(42);

    // ================================================================
    print_header("SPEED: Regular Mean — InnerContiguous (reduce last dim)");
    // ================================================================
    struct ShapeTest { int64_t r, c; const char* label; };
    ShapeTest inner_shapes[] = {
        {32, 768, "(32, 768)"},
        {32, 4096, "(32, 4096)"},
        {4096, 768, "(4096, 768)"},
        {256, 4096, "(256, 4096)"},
        {2048, 50176, "(2048, 50176)"},
        {49152, 128, "(49152, 128)"},
        {1000, 100, "(1000, 100)"},
        {1000, 10000, "(1000, 10000)"},
        {1000, 100000, "(1000, 100000)"},
        {10, 1000000, "(10, 1000000)"},
        {1000000, 10, "(1000000, 10)"},
        {100, 10000000, "(100, 10000000)"},
    };

    // float32
    print_sub("float32 mean (reduce last dim)");
    for (auto& s : inner_shapes) {
        if (s.r * s.c > 200000000LL) { printf("  %-20s float32   → SKIP (too large)\n", s.label); continue; }
        Tensor t({Shape({s.r, s.c})}, TensorOptions().with_dtype(Dtype::Float32));
        fill_randn(t, rng);
        double us = bench([&]() { return reduce_mean(t, {1}); });
        printf("  %-20s float32   → %10.0f μs\n", s.label, us);
    }

    // float64
    print_sub("float64 mean (reduce last dim)");
    for (auto& s : inner_shapes) {
        if (s.r * s.c > 200000000LL) { printf("  %-20s float64   → SKIP (too large)\n", s.label); continue; }
        Tensor t({Shape({s.r, s.c})}, TensorOptions().with_dtype(Dtype::Float64));
        fill_randn_double(t, rng);
        double us = bench([&]() { return reduce_mean(t, {1}); });
        printf("  %-20s float64   → %10.0f μs\n", s.label, us);
    }

    // ================================================================
    print_header("SPEED: Regular Mean — OuterContiguous (reduce first dim)");
    // ================================================================
    ShapeTest outer_shapes[] = {
        {1000, 256, "(1000, 256)"},
        {10000, 100, "(10000, 100)"},
        {100, 10000, "(100, 10000)"},
        {50176, 2048, "(50176, 2048)"},
    };
    for (auto& s : outer_shapes) {
        Tensor t({Shape({s.r, s.c})}, TensorOptions().with_dtype(Dtype::Float32));
        fill_randn(t, rng);
        double us = bench([&]() { return reduce_mean(t, {0}); });
        printf("  %-20s float32   → %10.0f μs\n", s.label, us);
    }
    for (auto& s : outer_shapes) {
        Tensor t({Shape({s.r, s.c})}, TensorOptions().with_dtype(Dtype::Float64));
        fill_randn_double(t, rng);
        double us = bench([&]() { return reduce_mean(t, {0}); });
        printf("  %-20s float64   → %10.0f μs\n", s.label, us);
    }

    // ================================================================
    print_header("SPEED: Regular Mean — Generic (reduce mixed dims)");
    // ================================================================
    {
        Tensor t({Shape({100, 200, 50})}, TensorOptions().with_dtype(Dtype::Float32));
        fill_randn(t, rng);
        printf("  (100,200,50) dims=(0,2) → %10.0f μs\n", bench([&]() { return reduce_mean(t, {0, 2}); }));
        printf("  (100,200,50) dims=(0)   → %10.0f μs\n", bench([&]() { return reduce_mean(t, {0}); }));
        printf("  (100,200,50) dims=(1,2) → %10.0f μs\n", bench([&]() { return reduce_mean(t, {1, 2}); }));
    }
    {
        Tensor t({Shape({32, 128, 768})}, TensorOptions().with_dtype(Dtype::Float32));
        fill_randn(t, rng);
        printf("  (32,128,768) dims=(0,2) → %10.0f μs\n", bench([&]() { return reduce_mean(t, {0, 2}); }));
        printf("  (32,128,768) dims=(0)   → %10.0f μs\n", bench([&]() { return reduce_mean(t, {0}); }));
        printf("  (32,128,768) dims=(1,2) → %10.0f μs\n", bench([&]() { return reduce_mean(t, {1, 2}); }));
    }

    // ================================================================
    print_header("SPEED: Regular Mean — Full Reduction");
    // ================================================================
    for (int64_t size : {1000LL, 10000LL, 100000LL, 1000000LL, 10000000LL, 50000000LL}) {
        Tensor t({Shape({size})}, TensorOptions().with_dtype(Dtype::Float32));
        fill_randn(t, rng);
        double us = bench([&]() { return reduce_mean(t); });
        printf("  (%10lld,) float32   → %10.0f μs\n", (long long)size, us);
    }
    for (int64_t size : {1000LL, 10000LL, 100000LL, 1000000LL, 10000000LL, 50000000LL}) {
        Tensor t({Shape({size})}, TensorOptions().with_dtype(Dtype::Float64));
        fill_randn_double(t, rng);
        double us = bench([&]() { return reduce_mean(t); });
        printf("  (%10lld,) float64   → %10.0f μs\n", (long long)size, us);
    }

    // ================================================================
    print_header("SPEED: NanMean — InnerContiguous (10% NaN)");
    // ================================================================
    ShapeTest nanmean_shapes[] = {
        {32, 768, "(32, 768)"},
        {4096, 768, "(4096, 768)"},
        {256, 4096, "(256, 4096)"},
        {2048, 50176, "(2048, 50176)"},
        {1000, 10000, "(1000, 10000)"},
        {10, 1000000, "(10, 1000000)"},
        {1000000, 10, "(1000000, 10)"},
    };
    for (auto& s : nanmean_shapes) {
        if (s.r * s.c > 200000000LL) continue;
        Tensor t({Shape({s.r, s.c})}, TensorOptions().with_dtype(Dtype::Float32));
        fill_randn(t, rng); add_nan(t, 0.1, rng);
        double us = bench([&]() { return reduce_nanmean(t, {1}); });
        printf("  %-20s float32   → %10.0f μs\n", s.label, us);
    }
    for (auto& s : nanmean_shapes) {
        if (s.r * s.c > 200000000LL) continue;
        Tensor t({Shape({s.r, s.c})}, TensorOptions().with_dtype(Dtype::Float64));
        fill_randn_double(t, rng); add_nan_double(t, 0.1, rng);
        double us = bench([&]() { return reduce_nanmean(t, {1}); });
        printf("  %-20s float64   → %10.0f μs\n", s.label, us);
    }

    // ================================================================
    print_header("SPEED: NanMean — Full Reduction (10% NaN)");
    // ================================================================
    for (int64_t size : {10000LL, 100000LL, 1000000LL, 10000000LL, 50000000LL}) {
        Tensor t({Shape({size})}, TensorOptions().with_dtype(Dtype::Float32));
        fill_randn(t, rng); add_nan(t, 0.1, rng);
        double us = bench([&]() { return reduce_nanmean(t); });
        printf("  (%10lld,) float32   → %10.0f μs\n", (long long)size, us);
    }

    // ================================================================
    print_header("SPEED: NanMean — Varying NaN %% (1000, 10000)");
    // ================================================================
    for (double pct : {0.0, 0.01, 0.1, 0.5, 0.9, 0.99}) {
        Tensor t({Shape({1000, 10000})}, TensorOptions().with_dtype(Dtype::Float32));
        fill_randn(t, rng);
        if (pct > 0) add_nan(t, pct, rng);
        double us = bench([&]() { return reduce_nanmean(t, {1}); });
        printf("  %5.1f%% NaN → %10.0f μs\n", pct * 100, us);
    }

    // ================================================================
    print_header("PRECISION: Regular Mean vs fp64 ground truth");
    // ================================================================
    auto precision_test = [&](const char* desc, auto fill_fn, int64_t N) {
        Tensor t({Shape({N})}, TensorOptions().with_dtype(Dtype::Float32));
        float* d = t.data<float>();
        fill_fn(d, N, rng);
        double gt = ground_truth_mean(d, N);
        Tensor result = reduce_mean(t);
        double ours = static_cast<double>(result.data<float>()[0]);
        double rel_err = (gt != 0) ? std::abs(ours - gt) / std::abs(gt) : 0;
        printf("  %-25s N=%10lld  Our rel_err=%.2e  (gt=%.10e, ours=%.10e)\n",
               desc, (long long)N, rel_err, gt, ours);
    };

    auto fill_uniform = [](float* d, int64_t n, std::mt19937& rng) {
        std::uniform_real_distribution<float> u(-1, 1);
        for (int64_t i = 0; i < n; ++i) d[i] = u(rng);
    };
    auto fill_gaussian = [](float* d, int64_t n, std::mt19937& rng) {
        std::normal_distribution<float> u(0, 1);
        for (int64_t i = 0; i < n; ++i) d[i] = u(rng);
    };
    auto fill_large_mean = [](float* d, int64_t n, std::mt19937& rng) {
        std::normal_distribution<float> u(0, 1e-3f);
        for (int64_t i = 0; i < n; ++i) d[i] = 1e6f + u(rng);
    };
    auto fill_mixed = [](float* d, int64_t n, std::mt19937& rng) {
        std::uniform_real_distribution<float> u(1e-6f, 1e6f);
        for (int64_t i = 0; i < n; ++i) d[i] = u(rng);
    };
    auto fill_subnormal = [](float* d, int64_t n, std::mt19937& rng) {
        std::uniform_real_distribution<float> u(1e-45f, 1e-38f);
        for (int64_t i = 0; i < n; ++i) d[i] = u(rng);
    };
    auto fill_near_max = [](float* d, int64_t n, std::mt19937& rng) {
        std::uniform_real_distribution<float> u(1e37f, 3e38f);
        for (int64_t i = 0; i < n; ++i) d[i] = u(rng);
    };

    for (int64_t N : {1000LL, 100000LL, 10000000LL}) {
        precision_test("Uniform [-1,1]", fill_uniform, N);
        precision_test("Gaussian N(0,1)", fill_gaussian, N);
        precision_test("Large mean+tiny var", fill_large_mean, N);
        precision_test("Mixed scale [1e-6,1e6]", fill_mixed, N);
        precision_test("Subnormals", fill_subnormal, N);
        precision_test("Near FLT_MAX", fill_near_max, N);
        printf("\n");
    }

    // ================================================================
    print_header("PRECISION: NanMean (10% NaN)");
    // ================================================================
    for (int64_t N : {1000LL, 100000LL, 10000000LL}) {
        Tensor t({Shape({N})}, TensorOptions().with_dtype(Dtype::Float32));
        fill_randn(t, rng); add_nan(t, 0.1, rng);
        double gt = ground_truth_nanmean(t.data<float>(), N);
        Tensor result = reduce_nanmean(t);
        double ours = static_cast<double>(result.data<float>()[0]);
        double rel_err = (gt != 0) ? std::abs(ours - gt) / std::abs(gt) : 0;
        printf("  N=%10lld  Our rel_err=%.2e\n", (long long)N, rel_err);
    }

    // ================================================================
    print_header("EDGE CASES");
    // ================================================================
    {
        // All NaN
        Tensor t({Shape({100})}, TensorOptions().with_dtype(Dtype::Float32));
        float* d = t.data<float>();
        for (int i = 0; i < 100; ++i) d[i] = std::nanf("");
        Tensor r = reduce_nanmean(t);
        printf("  nanmean([NaN]*100) = %f\n", (double)r.data<float>()[0]);
        Tensor r2 = reduce_mean(t);
        printf("  mean([NaN]*100) = %f\n", (double)r2.data<float>()[0]);
    }
    {
        // Single element
        Tensor t({Shape({1})}, TensorOptions().with_dtype(Dtype::Float32));
        t.data<float>()[0] = 42.0f;
        printf("  mean([42.0]) = %f\n", (double)reduce_mean(t).data<float>()[0]);
        printf("  nanmean([42.0]) = %f\n", (double)reduce_nanmean(t).data<float>()[0]);
    }
    {
        // Mixed NaN
        Tensor t({Shape({5})}, TensorOptions().with_dtype(Dtype::Float32));
        float* d = t.data<float>();
        d[0] = 1; d[1] = std::nanf(""); d[2] = 3; d[3] = std::nanf(""); d[4] = 5;
        printf("  nanmean([1,NaN,3,NaN,5]) = %f (expected 3.0)\n",
               (double)reduce_nanmean(t).data<float>()[0]);
    }
    {
        // Large mean=1 precision
        int64_t N = 100000000;
        Tensor t({Shape({N})}, TensorOptions().with_dtype(Dtype::Float32));
        float* d = t.data<float>();
        for (int64_t i = 0; i < N; ++i) d[i] = 1.0f;
        printf("  mean(ones(100M)) = %.10f (expected 1.0)\n",
               (double)reduce_mean(t).data<float>()[0]);
        d[0] = 1e8f;
        printf("  mean(ones(100M)+[1e8]) = %.10f (expected ~2.0)\n",
               (double)reduce_mean(t).data<float>()[0]);
    }

    printf("\n====================================================================================================\n");
    printf("BENCHMARK COMPLETE\n");
    printf("====================================================================================================\n");
    return 0;
}
