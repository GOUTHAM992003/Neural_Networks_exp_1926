/**
 * FINAL DEFINITIVE: nanmean — Fused SIMD + scalar div vs Fused SIMD + SIMD div
 *
 * B: Fused SIMD (sum+count in 1 pass) → scalar div per output (inside parallel for)
 * C: Fused SIMD (sum+count in 1 pass) → store sums/counts → SIMD div over outputs
 *
 * Tests: many shapes, NaN densities, edge cases, small/large tensors
 */
#include <immintrin.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <random>
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

inline float hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

int main() {
    omp_set_num_threads(28);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    std::uniform_real_distribution<float> nan_dist(0.0f, 1.0f);

    std::cout << "Threads: " << omp_get_max_threads() << "\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n";
    std::cout << "DEFINITIVE: B (fused+scalar div) vs C (fused+SIMD div)\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n\n";

    // ═══ TEST 1: Various shapes, 10% NaN ═══
    std::cout << "TEST 1: Various shapes (10% NaN)\n";
    std::cout << std::setw(20) << "Shape" << " | "
              << std::setw(14) << "B:scalar div" << " | "
              << std::setw(14) << "C:SIMD div" << " | "
              << std::setw(10) << "Winner" << " | "
              << std::setw(10) << "Ratio" << "\n";
    std::cout << std::string(80, '-') << "\n";

    struct Shape { int64_t rows, cols; };
    std::vector<Shape> shapes = {
        // Few outputs, large reduction (division negligible)
        {1, 1000000}, {10, 100000}, {10, 1000000},
        // Medium outputs
        {100, 1000}, {100, 10000}, {100, 100000},
        {1000, 1000}, {1000, 10000},
        // Many outputs, small reduction (division matters more)
        {10000, 100}, {10000, 1000},
        {100000, 10}, {100000, 100},
        {1000000, 10},
        // Edge: single output (full reduction)
        {1, 10000000},
    };

    for (auto [rows, cols] : shapes) {
        int64_t total = rows * cols;
        std::vector<float> data(total);
        for (auto& v : data) v = (nan_dist(gen) < 0.1f) ? std::nanf("") : dist(gen);
        std::vector<float> out_b(rows), out_c(rows);
        std::vector<float> sums_c(rows), counts_c(rows);

        // B: Fused SIMD + scalar div per output
        double t_b = bench([&]() {
            #pragma omp parallel for
            for (int64_t r = 0; r < rows; ++r) {
                const float* row = &data[r * cols];
                __m256 vsum = _mm256_setzero_ps(), vcnt = _mm256_setzero_ps();
                __m256 zero = _mm256_setzero_ps(), ones = _mm256_set1_ps(1.0f);
                int64_t j = 0;
                for (; j + 8 <= cols; j += 8) {
                    __m256 v = _mm256_loadu_ps(row + j);
                    __m256 m = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
                    vsum = _mm256_add_ps(vsum, _mm256_blendv_ps(v, zero, m));
                    vcnt = _mm256_add_ps(vcnt, _mm256_blendv_ps(ones, zero, m));
                }
                double s = hsum_ps(vsum), c = hsum_ps(vcnt);
                for (; j < cols; ++j)
                    if (!std::isnan(row[j])) { s += row[j]; c += 1.0; }
                out_b[r] = (c > 0) ? static_cast<float>(s / c) : std::nanf("");
            }
        });

        // C: Fused SIMD + store + SIMD div
        double t_c = bench([&]() {
            #pragma omp parallel for
            for (int64_t r = 0; r < rows; ++r) {
                const float* row = &data[r * cols];
                __m256 vsum = _mm256_setzero_ps(), vcnt = _mm256_setzero_ps();
                __m256 zero = _mm256_setzero_ps(), ones = _mm256_set1_ps(1.0f);
                int64_t j = 0;
                for (; j + 8 <= cols; j += 8) {
                    __m256 v = _mm256_loadu_ps(row + j);
                    __m256 m = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
                    vsum = _mm256_add_ps(vsum, _mm256_blendv_ps(v, zero, m));
                    vcnt = _mm256_add_ps(vcnt, _mm256_blendv_ps(ones, zero, m));
                }
                float s = hsum_ps(vsum), c = hsum_ps(vcnt);
                for (; j < cols; ++j)
                    if (!std::isnan(row[j])) { s += row[j]; c += 1.0f; }
                sums_c[r] = s;
                counts_c[r] = c;
            }
            // SIMD divide over outputs
            int64_t r = 0;
            for (; r + 8 <= rows; r += 8)
                _mm256_storeu_ps(&out_c[r], _mm256_div_ps(
                    _mm256_loadu_ps(&sums_c[r]), _mm256_loadu_ps(&counts_c[r])));
            for (; r < rows; ++r)
                out_c[r] = (counts_c[r] > 0) ? sums_c[r] / counts_c[r] : std::nanf("");
        });

        char sh[32]; snprintf(sh, sizeof(sh), "(%ld, %ld)", rows, cols);
        const char* winner = (t_b <= t_c) ? "B" : "C";
        double ratio = t_c / t_b;
        std::cout << std::setw(20) << sh
                  << " | " << std::setw(11) << std::fixed << std::setprecision(1) << t_b << "μs"
                  << " | " << std::setw(11) << t_c << "μs"
                  << " | " << std::setw(10) << winner
                  << " | " << std::setw(8) << std::setprecision(2) << ratio << "x\n";
    }

    // ═══ TEST 2: NaN density sweep ═══
    std::cout << "\nTEST 2: NaN density sweep — (1000, 10000)\n";
    std::cout << std::string(70, '-') << "\n";
    int64_t r2 = 1000, c2 = 10000;
    for (float nr : {0.0f, 0.01f, 0.1f, 0.5f, 0.9f, 0.99f, 1.0f}) {
        std::vector<float> data(r2 * c2);
        for (auto& v : data) v = (nan_dist(gen) < nr) ? std::nanf("") : dist(gen);
        std::vector<float> out(r2), sums(r2), cnts(r2);

        double t_b = bench([&]() {
            #pragma omp parallel for
            for (int64_t r = 0; r < r2; ++r) {
                const float* row = &data[r * c2];
                __m256 vs = _mm256_setzero_ps(), vc = _mm256_setzero_ps();
                __m256 z = _mm256_setzero_ps(), o = _mm256_set1_ps(1.0f);
                int64_t j = 0;
                for (; j + 8 <= c2; j += 8) {
                    __m256 v = _mm256_loadu_ps(row + j);
                    __m256 m = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
                    vs = _mm256_add_ps(vs, _mm256_blendv_ps(v, z, m));
                    vc = _mm256_add_ps(vc, _mm256_blendv_ps(o, z, m));
                }
                double s = hsum_ps(vs), c = hsum_ps(vc);
                for (; j < c2; ++j) if (!std::isnan(row[j])) { s += row[j]; c += 1.0; }
                out[r] = (c > 0) ? static_cast<float>(s / c) : std::nanf("");
            }
        });

        double t_c = bench([&]() {
            #pragma omp parallel for
            for (int64_t r = 0; r < r2; ++r) {
                const float* row = &data[r * c2];
                __m256 vs = _mm256_setzero_ps(), vc = _mm256_setzero_ps();
                __m256 z = _mm256_setzero_ps(), o = _mm256_set1_ps(1.0f);
                int64_t j = 0;
                for (; j + 8 <= c2; j += 8) {
                    __m256 v = _mm256_loadu_ps(row + j);
                    __m256 m = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
                    vs = _mm256_add_ps(vs, _mm256_blendv_ps(v, z, m));
                    vc = _mm256_add_ps(vc, _mm256_blendv_ps(o, z, m));
                }
                float s = hsum_ps(vs), c = hsum_ps(vc);
                for (; j < c2; ++j) if (!std::isnan(row[j])) { s += row[j]; c += 1.0f; }
                sums[r] = s; cnts[r] = c;
            }
            int64_t r = 0;
            for (; r + 8 <= r2; r += 8)
                _mm256_storeu_ps(&out[r], _mm256_div_ps(_mm256_loadu_ps(&sums[r]), _mm256_loadu_ps(&cnts[r])));
            for (; r < r2; ++r)
                out[r] = (cnts[r] > 0) ? sums[r] / cnts[r] : std::nanf("");
        });

        std::cout << "NaN " << std::setw(3) << (int)(nr * 100) << "%"
                  << " | B:" << std::setw(8) << std::setprecision(1) << t_b << "μs"
                  << " | C:" << std::setw(8) << t_c << "μs"
                  << " | " << ((t_b <= t_c) ? "B wins" : "C wins")
                  << " (" << std::setprecision(2) << (t_c/t_b) << "x)\n";
    }

    // ═══ TEST 3: Edge cases ═══
    std::cout << "\nTEST 3: Edge cases\n";
    std::cout << std::string(70, '-') << "\n";

    // All NaN
    {
        std::vector<float> data(1000000, std::nanf(""));
        std::vector<float> out(1000);
        double t = bench([&]() {
            #pragma omp parallel for
            for (int64_t r = 0; r < 1000; ++r) {
                const float* row = &data[r * 1000];
                __m256 vs = _mm256_setzero_ps(), vc = _mm256_setzero_ps();
                __m256 z = _mm256_setzero_ps(), o = _mm256_set1_ps(1.0f);
                for (int64_t j = 0; j + 8 <= 1000; j += 8) {
                    __m256 v = _mm256_loadu_ps(row + j);
                    __m256 m = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
                    vs = _mm256_add_ps(vs, _mm256_blendv_ps(v, z, m));
                    vc = _mm256_add_ps(vc, _mm256_blendv_ps(o, z, m));
                }
                double c = hsum_ps(vc);
                out[r] = (c > 0) ? static_cast<float>(static_cast<double>(hsum_ps(vs)) / c) : std::nanf("");
            }
        });
        std::cout << "All NaN (1000,1000):     " << std::setprecision(1) << t << "μs"
                  << " | result[0]=" << out[0] << " (should be nan)\n";
    }

    // No NaN
    {
        std::vector<float> data(1000000);
        for (auto& v : data) v = dist(gen);
        std::vector<float> out(1000);
        double t = bench([&]() {
            #pragma omp parallel for
            for (int64_t r = 0; r < 1000; ++r) {
                const float* row = &data[r * 1000];
                __m256 vs = _mm256_setzero_ps(), vc = _mm256_setzero_ps();
                __m256 z = _mm256_setzero_ps(), o = _mm256_set1_ps(1.0f);
                for (int64_t j = 0; j + 8 <= 1000; j += 8) {
                    __m256 v = _mm256_loadu_ps(row + j);
                    __m256 m = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
                    vs = _mm256_add_ps(vs, _mm256_blendv_ps(v, z, m));
                    vc = _mm256_add_ps(vc, _mm256_blendv_ps(o, z, m));
                }
                double s = hsum_ps(vs), c = hsum_ps(vc);
                out[r] = static_cast<float>(s / c);
            }
        });
        std::cout << "No NaN (1000,1000):      " << t << "μs"
                  << " | result[0]=" << out[0] << "\n";
    }

    // Single element per output
    {
        std::vector<float> data(100000);
        for (auto& v : data) v = dist(gen);
        data[50000] = std::nanf("");
        std::vector<float> out(100000);
        double t = bench([&]() {
            for (int64_t r = 0; r < 100000; ++r) {
                float v = data[r];
                out[r] = std::isnan(v) ? std::nanf("") : v;
            }
        });
        std::cout << "Single elem (100000,1):  " << t << "μs\n";
    }

    return 0;
}
