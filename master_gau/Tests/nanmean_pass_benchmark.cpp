/**
 * Benchmark: Single-pass vs Two-pass NaN counting for nanmean
 *
 * Two-pass (current): Pass 1 = nansum, Pass 2 = count non-NaN, then divide
 * Single-pass (fused): One loop that computes both nansum AND non-NaN count
 *
 * Key question: Is the second pass a significant bottleneck?
 * Reductions are memory-bound, so reading data twice = ~2x memory bandwidth.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <omp.h>

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

int main() {
    std::cout << "Threads: " << omp_get_max_threads() << "\n\n";
    std::cout << std::string(85, '=') << "\n";
    std::cout << "BENCHMARK: Single-pass vs Two-pass NaN counting (nanmean)\n";
    std::cout << std::string(85, '=') << "\n\n";

    // NaN density: 10% NaN values
    float nan_ratio = 0.1f;

    std::cout << std::setw(12) << "Size" << " | "
              << std::setw(8) << "NaN%" << " | "
              << std::setw(14) << "Two-pass (μs)" << " | "
              << std::setw(14) << "Single-pass" << " | "
              << std::setw(10) << "Speedup" << "\n";
    std::cout << std::string(70, '-') << "\n";

    for (int64_t size : {10000LL, 100000LL, 1000000LL, 10000000LL, 50000000LL, 100000000LL}) {
        // Create data with 10% NaN
        std::vector<float> data(size);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
        std::uniform_real_distribution<float> nan_dist(0.0f, 1.0f);
        for (int64_t i = 0; i < size; ++i) {
            data[i] = (nan_dist(gen) < nan_ratio) ? std::nanf("") : dist(gen);
        }

        const float* ptr = data.data();
        volatile double result_2pass = 0, result_1pass = 0;
        volatile int64_t count_2pass = 0, count_1pass = 0;

        // TWO-PASS: nansum first, then count non-NaN
        double t_2pass = bench([&]() {
            // Pass 1: nansum
            double sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int64_t i = 0; i < size; ++i) {
                float v = ptr[i];
                if (!std::isnan(v)) sum += v;
            }
            // Pass 2: count non-NaN
            int64_t cnt = 0;
            #pragma omp parallel for reduction(+:cnt)
            for (int64_t i = 0; i < size; ++i) {
                if (!std::isnan(ptr[i])) ++cnt;
            }
            result_2pass = sum / cnt;
            count_2pass = cnt;
        });

        // SINGLE-PASS: nansum AND count simultaneously
        double t_1pass = bench([&]() {
            double sum = 0;
            int64_t cnt = 0;
            #pragma omp parallel for reduction(+:sum,cnt)
            for (int64_t i = 0; i < size; ++i) {
                float v = ptr[i];
                if (!std::isnan(v)) {
                    sum += v;
                    ++cnt;
                }
            }
            result_1pass = sum / cnt;
            count_1pass = cnt;
        });

        double speedup = t_2pass / t_1pass;
        std::cout << std::setw(12) << size << " | "
                  << std::setw(7) << std::fixed << std::setprecision(0) << (nan_ratio * 100) << "%" << " | "
                  << std::setw(12) << std::setprecision(1) << t_2pass << "μs | "
                  << std::setw(12) << t_1pass << "μs | "
                  << std::setw(8) << std::setprecision(2) << speedup << "x\n";
    }

    // Also test with different NaN densities for 10M elements
    std::cout << "\n" << std::string(85, '=') << "\n";
    std::cout << "NaN Density Impact (10M float32 elements)\n";
    std::cout << std::string(85, '=') << "\n\n";

    int64_t size = 10000000;
    for (float nr : {0.0f, 0.01f, 0.1f, 0.5f, 0.9f}) {
        std::vector<float> data(size);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
        std::uniform_real_distribution<float> nan_dist(0.0f, 1.0f);
        for (int64_t i = 0; i < size; ++i)
            data[i] = (nan_dist(gen) < nr) ? std::nanf("") : dist(gen);

        const float* ptr = data.data();
        volatile double r2 = 0, r1 = 0;

        double t_2pass = bench([&]() {
            double sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int64_t i = 0; i < size; ++i) {
                float v = ptr[i]; if (!std::isnan(v)) sum += v;
            }
            int64_t cnt = 0;
            #pragma omp parallel for reduction(+:cnt)
            for (int64_t i = 0; i < size; ++i) {
                if (!std::isnan(ptr[i])) ++cnt;
            }
            r2 = (cnt > 0) ? sum / cnt : 0;
        });

        double t_1pass = bench([&]() {
            double sum = 0; int64_t cnt = 0;
            #pragma omp parallel for reduction(+:sum,cnt)
            for (int64_t i = 0; i < size; ++i) {
                float v = ptr[i];
                if (!std::isnan(v)) { sum += v; ++cnt; }
            }
            r1 = (cnt > 0) ? sum / cnt : 0;
        });

        std::cout << "NaN " << std::setw(3) << std::setprecision(0) << (nr * 100)
                  << "%: Two-pass=" << std::setw(8) << std::setprecision(1) << t_2pass
                  << "μs  Single-pass=" << std::setw(8) << t_1pass
                  << "μs  Speedup=" << std::setprecision(2) << (t_2pass / t_1pass) << "x\n";
    }

    return 0;
}
