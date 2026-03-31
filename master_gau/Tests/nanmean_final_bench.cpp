/**
 * DEFINITIVE: nanmean approaches — Fused SIMD vs everything else
 *
 * A: Fused scalar (current): scalar sum + scalar count + scalar div per output
 * B: Fused SIMD + scalar div: SIMD sum+count in 1 pass, scalar div per output
 * C: Fused SIMD + SIMD div: SIMD sum+count in 1 pass, store, SIMD div over outputs
 * D: PyTorch-style: separate nansum + separate count + SIMD div (multiple passes)
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

    // ═══ TEST 1: Different shapes ═══
    std::cout << "═══════════════════════════════════════════════════════════════════════\n";
    std::cout << "TEST 1: Different shapes (10% NaN)\n";
    std::cout << "═══════════════════════════════════════════════════════════════════════\n\n";
    std::cout << std::setw(18) << "Shape" << " | "
              << std::setw(12) << "A:fused_sc" << " | "
              << std::setw(12) << "B:fused_simd" << " | "
              << std::setw(12) << "C:fused+sdiv" << " | "
              << std::setw(12) << "D:pytorch" << " | Best\n";
    std::cout << std::string(90, '-') << "\n";

    struct Shape { int64_t rows, cols; };
    std::vector<Shape> shapes = {
        {10, 1000}, {10, 100000}, {10, 1000000},
        {100, 1000}, {100, 10000}, {100, 100000}, {100, 1000000},
        {1000, 1000}, {1000, 10000}, {1000, 100000},
        {10000, 100}, {10000, 1000},
        {100000, 100}, {1000000, 10}
    };

    for (auto [rows, cols] : shapes) {
        int64_t total = rows * cols;
        std::vector<float> data(total);
        for (auto& v : data) v = (nan_dist(gen) < 0.1f) ? std::nanf("") : dist(gen);
        std::vector<float> out(rows);

        // A: Fused scalar (current)
        double t_a = bench([&]() {
            #pragma omp parallel for
            for (int64_t r = 0; r < rows; ++r) {
                const float* row = &data[r * cols];
                double s = 0; double c = 0;
                for (int64_t j = 0; j < cols; ++j)
                    if (!std::isnan(row[j])) { s += row[j]; c += 1.0; }
                out[r] = (c > 0) ? (float)(s / c) : std::nanf("");
            }
        });

        // B: Fused SIMD + scalar div
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
                float s = hsum_ps(vsum), c = hsum_ps(vcnt);
                for (; j < cols; ++j)
                    if (!std::isnan(row[j])) { s += row[j]; c += 1.0f; }
                out[r] = (c > 0) ? s / c : std::nanf("");
            }
        });

        // C: Fused SIMD + store + SIMD div over outputs
        std::vector<float> sums(rows), counts(rows);
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
                sums[r] = s; counts[r] = c;
            }
            // SIMD divide over outputs
            int64_t r = 0;
            for (; r + 8 <= rows; r += 8)
                _mm256_storeu_ps(&out[r], _mm256_div_ps(_mm256_loadu_ps(&sums[r]), _mm256_loadu_ps(&counts[r])));
            for (; r < rows; ++r)
                out[r] = (counts[r] > 0) ? sums[r] / counts[r] : std::nanf("");
        });

        // D: PyTorch-style (separate nansum + separate count + div)
        double t_d = bench([&]() {
            // Pass 1: nansum with SIMD mask
            #pragma omp parallel for
            for (int64_t r = 0; r < rows; ++r) {
                const float* row = &data[r * cols];
                __m256 vsum = _mm256_setzero_ps(), zero = _mm256_setzero_ps();
                int64_t j = 0;
                for (; j + 8 <= cols; j += 8) {
                    __m256 v = _mm256_loadu_ps(row + j);
                    vsum = _mm256_add_ps(vsum, _mm256_blendv_ps(v, zero, _mm256_cmp_ps(v, v, _CMP_UNORD_Q)));
                }
                float s = hsum_ps(vsum);
                for (; j < cols; ++j) if (!std::isnan(row[j])) s += row[j];
                sums[r] = s;
            }
            // Pass 2: count non-NaN
            #pragma omp parallel for
            for (int64_t r = 0; r < rows; ++r) {
                const float* row = &data[r * cols];
                __m256 vcnt = _mm256_setzero_ps(), zero = _mm256_setzero_ps(), ones = _mm256_set1_ps(1.0f);
                int64_t j = 0;
                for (; j + 8 <= cols; j += 8) {
                    __m256 v = _mm256_loadu_ps(row + j);
                    vcnt = _mm256_add_ps(vcnt, _mm256_blendv_ps(ones, zero, _mm256_cmp_ps(v, v, _CMP_UNORD_Q)));
                }
                float c = hsum_ps(vcnt);
                for (; j < cols; ++j) if (!std::isnan(row[j])) c += 1.0f;
                counts[r] = c;
            }
            // Pass 3: SIMD divide
            int64_t r = 0;
            for (; r + 8 <= rows; r += 8)
                _mm256_storeu_ps(&out[r], _mm256_div_ps(_mm256_loadu_ps(&sums[r]), _mm256_loadu_ps(&counts[r])));
            for (; r < rows; ++r)
                out[r] = (counts[r] > 0) ? sums[r] / counts[r] : std::nanf("");
        });

        double best = std::min({t_a, t_b, t_c, t_d});
        const char* winner = (best == t_a) ? "A" : (best == t_b) ? "B" : (best == t_c) ? "C" : "D";

        char sh[32]; snprintf(sh, sizeof(sh), "(%ld,%ld)", rows, cols);
        std::cout << std::setw(18) << sh
                  << " | " << std::setw(9) << std::fixed << std::setprecision(1) << t_a << "μs"
                  << " | " << std::setw(9) << t_b << "μs"
                  << " | " << std::setw(9) << t_c << "μs"
                  << " | " << std::setw(9) << t_d << "μs"
                  << " | " << winner << "\n";
    }

    // ═══ TEST 2: Different NaN densities ═══
    std::cout << "\n═══════════════════════════════════════════════════════════════════════\n";
    std::cout << "TEST 2: NaN density impact — shape (1000, 10000)\n";
    std::cout << "═══════════════════════════════════════════════════════════════════════\n\n";

    int64_t rows2 = 1000, cols2 = 10000;
    for (float nr : {0.0f, 0.01f, 0.1f, 0.25f, 0.5f, 0.75f, 0.99f}) {
        std::vector<float> data(rows2 * cols2);
        for (auto& v : data) v = (nan_dist(gen) < nr) ? std::nanf("") : dist(gen);
        std::vector<float> out(rows2), sums(rows2), counts(rows2);

        double t_a = bench([&]() {
            #pragma omp parallel for
            for (int64_t r = 0; r < rows2; ++r) {
                const float* row = &data[r * cols2];
                double s = 0, c = 0;
                for (int64_t j = 0; j < cols2; ++j)
                    if (!std::isnan(row[j])) { s += row[j]; c += 1.0; }
                out[r] = (c > 0) ? (float)(s / c) : std::nanf("");
            }
        });

        double t_b = bench([&]() {
            #pragma omp parallel for
            for (int64_t r = 0; r < rows2; ++r) {
                const float* row = &data[r * cols2];
                __m256 vsum = _mm256_setzero_ps(), vcnt = _mm256_setzero_ps();
                __m256 zero = _mm256_setzero_ps(), ones = _mm256_set1_ps(1.0f);
                int64_t j = 0;
                for (; j + 8 <= cols2; j += 8) {
                    __m256 v = _mm256_loadu_ps(row + j);
                    __m256 m = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
                    vsum = _mm256_add_ps(vsum, _mm256_blendv_ps(v, zero, m));
                    vcnt = _mm256_add_ps(vcnt, _mm256_blendv_ps(ones, zero, m));
                }
                float s = hsum_ps(vsum), c = hsum_ps(vcnt);
                for (; j < cols2; ++j)
                    if (!std::isnan(row[j])) { s += row[j]; c += 1.0f; }
                out[r] = (c > 0) ? s / c : std::nanf("");
            }
        });

        std::cout << "NaN " << std::setw(3) << (int)(nr * 100) << "%"
                  << " | A:scalar=" << std::setw(8) << std::setprecision(1) << t_a << "μs"
                  << " | B:fused_simd=" << std::setw(8) << t_b << "μs"
                  << " | Speedup=" << std::setprecision(2) << (t_a / t_b) << "x\n";
    }

    return 0;
}
