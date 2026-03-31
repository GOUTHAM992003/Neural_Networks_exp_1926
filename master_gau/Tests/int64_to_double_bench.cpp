/**
 * Test: int64 → double conversion approaches on AVX2
 * AVX2 has NO native _mm256_cvtepi64_pd (that's AVX-512 only)
 * But there are workarounds!
 */
#include <immintrin.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <random>
#include <cstdint>

double bench(auto fn, int warmup = 5, int iters = 100) {
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
    std::cout << "Benchmark: int64→double conversion + multiply\n\n";

    for (int64_t size : {1000LL, 10000LL, 100000LL, 1000000LL}) {
        std::vector<int64_t> idata(size);
        std::mt19937_64 gen(42);
        for (auto& v : idata) v = gen() % 1000000;
        std::vector<double> out_a(size), out_b(size), out_c(size);
        double recip = 1.0 / 12345.0;

        // A: Pure scalar
        double t_scalar = bench([&]() {
            for (int64_t i = 0; i < size; ++i)
                out_a[i] = static_cast<double>(idata[i]) * recip;
        });

        // B: Extract-convert-multiply (process 4 int64 → 4 double)
        double t_extract = bench([&]() {
            __m256d vrecip = _mm256_set1_pd(recip);
            int64_t i = 0;
            for (; i + 4 <= size; i += 4) {
                // Extract 4 int64, convert to double, multiply
                // No native intrinsic — extract each lane manually
                alignas(32) int64_t tmp[4];
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp),
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&idata[i])));
                __m256d vd = _mm256_set_pd(
                    static_cast<double>(tmp[3]), static_cast<double>(tmp[2]),
                    static_cast<double>(tmp[1]), static_cast<double>(tmp[0]));
                _mm256_storeu_pd(&out_b[i], _mm256_mul_pd(vd, vrecip));
            }
            for (; i < size; ++i) out_b[i] = static_cast<double>(idata[i]) * recip;
        });

        // C: Process 2 halves via SSE (128-bit int64 → double not available either)
        // Actually use scalar with unrolling
        double t_unroll = bench([&]() {
            int64_t i = 0;
            for (; i + 4 <= size; i += 4) {
                out_c[i]   = static_cast<double>(idata[i])   * recip;
                out_c[i+1] = static_cast<double>(idata[i+1]) * recip;
                out_c[i+2] = static_cast<double>(idata[i+2]) * recip;
                out_c[i+3] = static_cast<double>(idata[i+3]) * recip;
            }
            for (; i < size; ++i) out_c[i] = static_cast<double>(idata[i]) * recip;
        });

        std::cout << std::setw(10) << size
                  << " | Scalar: " << std::setw(8) << std::fixed << std::setprecision(1) << t_scalar << "μs"
                  << " | Extract: " << std::setw(8) << t_extract << "μs"
                  << " | Unroll4: " << std::setw(8) << t_unroll << "μs\n";
    }
    return 0;
}
