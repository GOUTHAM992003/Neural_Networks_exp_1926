/**
 * Benchmark: Emulated int64 min/max SIMD vs Scalar
 * AVX2 has NO native _mm256_min_epi64. Must emulate with cmpgt + blend.
 * Question: Is the emulation faster or slower than plain scalar loop?
 */
#include <immintrin.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <random>
#include <climits>

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

// Emulated int64 min using cmpgt + blendv
inline __m256i mm256_min_epi64(__m256i a, __m256i b) {
    __m256i cmp = _mm256_cmpgt_epi64(a, b);  // a > b → all-1s
    return _mm256_blendv_epi8(a, b, cmp);     // select b where a > b (i.e. min)
}

inline __m256i mm256_max_epi64(__m256i a, __m256i b) {
    __m256i cmp = _mm256_cmpgt_epi64(a, b);
    return _mm256_blendv_epi8(b, a, cmp);     // select a where a > b (i.e. max)
}

int main() {
    std::cout << "Benchmark: int64 min/max — Emulated SIMD vs Scalar\n\n";

    for (int64_t size : {100000LL, 1000000LL, 10000000LL, 50000000LL}) {
        std::vector<int64_t> data(size);
        std::mt19937_64 gen(42);
        for (auto& v : data) v = gen() % 1000000 - 500000;

        const int64_t* ptr = data.data();
        volatile int64_t result_s = 0, result_v = 0;

        // Scalar min
        double t_scalar = bench([&]() {
            int64_t best = LLONG_MAX;
            for (int64_t i = 0; i < size; ++i)
                if (ptr[i] < best) best = ptr[i];
            result_s = best;
        });

        // Emulated SIMD min (4-wide)
        double t_simd = bench([&]() {
            __m256i va = _mm256_set1_epi64x(LLONG_MAX);
            __m256i vb = _mm256_set1_epi64x(LLONG_MAX);
            int64_t j = 0;
            for (; j + 8 <= size; j += 8) {
                va = mm256_min_epi64(va, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + j)));
                vb = mm256_min_epi64(vb, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + j + 4)));
            }
            va = mm256_min_epi64(va, vb);
            alignas(32) int64_t lanes[4];
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(lanes), va);
            int64_t best = std::min({lanes[0], lanes[1], lanes[2], lanes[3]});
            for (; j < size; ++j) if (ptr[j] < best) best = ptr[j];
            result_v = best;
        });

        double speedup = t_scalar / t_simd;
        std::cout << std::setw(12) << size << " | Scalar: " << std::setw(8) << std::fixed << std::setprecision(1) << t_scalar
                  << "μs | SIMD: " << std::setw(8) << t_simd
                  << "μs | Speedup: " << std::setprecision(2) << speedup << "x\n";
    }
    return 0;
}
