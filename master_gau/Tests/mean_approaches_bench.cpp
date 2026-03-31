/**
 * COMPREHENSIVE: All mean computation approaches
 *
 * For REGULAR MEAN (count is constant):
 *   Approach 1: Two-pass: cascade_sum(SIMD) → fp32 reciprocal-multiply(SIMD)
 *   Approach 2: Two-pass: cascade_sum(SIMD) → fp32 SIMD division
 *   Approach 3: Fused single-pass: sum + divide in one loop (no SIMD for division)
 *   Approach 4: Two-pass: cascade_sum with DOUBLE output → double reciprocal (no round-trip)
 *
 * For NANMEAN (count varies per output):
 *   Approach A: Fused single-pass: nansum + count + divide (1 data read)
 *   Approach B: Two-pass: nansum(SIMD+mask) + count_non_nan → SIMD division (2 data reads)
 *
 * Tests across: 1K, 10K, 100K, 1M, 10M elements
 * Shapes: (100, N/100) reducing dim=1, and (N,) full reduction
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

int main() {
    omp_set_num_threads(28);
    std::cout << "Threads: " << omp_get_max_threads() << "\n\n";

    // ═══════════════════════════════════════════════
    // REGULAR MEAN: Partial reduction (rows, cols) dim=1
    // ═══════════════════════════════════════════════
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "REGULAR MEAN: (rows, cols) reducing dim=1 — float32\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n\n";

    std::cout << std::setw(18) << "Shape" << " | "
              << std::setw(14) << "1:sum+recip" << " | "
              << std::setw(14) << "2:sum+div" << " | "
              << std::setw(14) << "3:fused" << " | "
              << std::setw(14) << "4:dbl_sum+rec" << "\n";
    std::cout << std::string(85, '-') << "\n";

    struct Shape { int64_t rows, cols; };
    std::vector<Shape> shapes = {{100, 1000}, {100, 10000}, {100, 100000}, {100, 1000000},
                                  {1000, 1000}, {1000, 10000}, {10000, 1000}};

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    for (auto [rows, cols] : shapes) {
        int64_t total = rows * cols;
        std::vector<float> data(total);
        for (auto& v : data) v = dist(gen);
        std::vector<float> out1(rows), out2(rows), out3(rows), out4(rows);

        float count_f = static_cast<float>(cols);
        float recip_f = 1.0f / count_f;

        // Approach 1: Two-pass sum(SIMD) + fp32 reciprocal-multiply
        double t1 = bench([&]() {
            #pragma omp parallel for
            for (int64_t r = 0; r < rows; ++r) {
                const float* row = &data[r * cols];
                // SIMD sum (4-acc)
                __m256 va = _mm256_setzero_ps(), vb = _mm256_setzero_ps();
                __m256 vc = _mm256_setzero_ps(), vd = _mm256_setzero_ps();
                int64_t j = 0;
                for (; j + 32 <= cols; j += 32) {
                    va = _mm256_add_ps(va, _mm256_loadu_ps(row + j));
                    vb = _mm256_add_ps(vb, _mm256_loadu_ps(row + j + 8));
                    vc = _mm256_add_ps(vc, _mm256_loadu_ps(row + j + 16));
                    vd = _mm256_add_ps(vd, _mm256_loadu_ps(row + j + 24));
                }
                va = _mm256_add_ps(_mm256_add_ps(va, vb), _mm256_add_ps(vc, vd));
                // Horizontal sum
                __m128 lo = _mm256_castps256_ps128(va);
                __m128 hi = _mm256_extractf128_ps(va, 1);
                lo = _mm_add_ps(lo, hi);
                lo = _mm_hadd_ps(lo, lo);
                lo = _mm_hadd_ps(lo, lo);
                float sum = _mm_cvtss_f32(lo);
                for (; j < cols; ++j) sum += row[j];
                // Reciprocal multiply
                out1[r] = sum * recip_f;
            }
        });

        // Approach 2: Two-pass sum(SIMD) + fp32 division
        double t2 = bench([&]() {
            #pragma omp parallel for
            for (int64_t r = 0; r < rows; ++r) {
                const float* row = &data[r * cols];
                __m256 va = _mm256_setzero_ps(), vb = _mm256_setzero_ps();
                __m256 vc = _mm256_setzero_ps(), vd = _mm256_setzero_ps();
                int64_t j = 0;
                for (; j + 32 <= cols; j += 32) {
                    va = _mm256_add_ps(va, _mm256_loadu_ps(row + j));
                    vb = _mm256_add_ps(vb, _mm256_loadu_ps(row + j + 8));
                    vc = _mm256_add_ps(vc, _mm256_loadu_ps(row + j + 16));
                    vd = _mm256_add_ps(vd, _mm256_loadu_ps(row + j + 24));
                }
                va = _mm256_add_ps(_mm256_add_ps(va, vb), _mm256_add_ps(vc, vd));
                __m128 lo = _mm256_castps256_ps128(va);
                __m128 hi = _mm256_extractf128_ps(va, 1);
                lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
                float sum = _mm_cvtss_f32(lo);
                for (; j < cols; ++j) sum += row[j];
                out2[r] = sum / count_f;
            }
        });

        // Approach 3: Fused single-pass (sum + divide, no SIMD, scalar)
        double t3 = bench([&]() {
            #pragma omp parallel for
            for (int64_t r = 0; r < rows; ++r) {
                const float* row = &data[r * cols];
                double sum = 0;
                for (int64_t j = 0; j < cols; ++j)
                    sum += row[j];
                out3[r] = static_cast<float>(sum / cols);
            }
        });

        // Approach 4: Two-pass with DOUBLE sum (no round-trip) + double reciprocal
        double t4 = bench([&]() {
            double recip_d = 1.0 / static_cast<double>(cols);
            #pragma omp parallel for
            for (int64_t r = 0; r < rows; ++r) {
                const float* row = &data[r * cols];
                // Sum in double (using SIMD cvtps_pd)
                __m256d va = _mm256_setzero_pd(), vb = _mm256_setzero_pd();
                int64_t j = 0;
                for (; j + 8 <= cols; j += 8) {
                    va = _mm256_add_pd(va, _mm256_cvtps_pd(_mm_loadu_ps(row + j)));
                    vb = _mm256_add_pd(vb, _mm256_cvtps_pd(_mm_loadu_ps(row + j + 4)));
                }
                va = _mm256_add_pd(va, vb);
                // Horizontal double sum
                __m128d lo = _mm256_castpd256_pd128(va);
                __m128d hi = _mm256_extractf128_pd(va, 1);
                lo = _mm_add_pd(lo, hi);
                lo = _mm_hadd_pd(lo, lo);
                double sum = _mm_cvtsd_f64(lo);
                for (; j < cols; ++j) sum += row[j];
                // Double reciprocal, then cast to float (no round-trip!)
                out4[r] = static_cast<float>(sum * recip_d);
            }
        });

        char shape_str[32];
        snprintf(shape_str, sizeof(shape_str), "(%ld, %ld)", rows, cols);
        std::cout << std::setw(18) << shape_str
                  << " | " << std::setw(11) << std::fixed << std::setprecision(1) << t1 << "μs"
                  << " | " << std::setw(11) << t2 << "μs"
                  << " | " << std::setw(11) << t3 << "μs"
                  << " | " << std::setw(11) << t4 << "μs\n";
    }

    // ═══════════════════════════════════════════════
    // NANMEAN: Fused vs Two-pass (10% NaN)
    // ═══════════════════════════════════════════════
    std::cout << "\n═══════════════════════════════════════════════════════════════\n";
    std::cout << "NANMEAN: Fused single-pass vs Two-pass — float32, 10% NaN\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n\n";

    std::cout << std::setw(18) << "Shape" << " | "
              << std::setw(16) << "A:Fused 1-pass" << " | "
              << std::setw(16) << "B:Two-pass+div" << " | "
              << std::setw(10) << "Winner" << "\n";
    std::cout << std::string(70, '-') << "\n";

    for (auto [rows, cols] : shapes) {
        int64_t total = rows * cols;
        std::vector<float> data(total);
        std::uniform_real_distribution<float> nan_dist(0.0f, 1.0f);
        for (auto& v : data) v = (nan_dist(gen) < 0.1f) ? std::nanf("") : dist(gen);
        std::vector<float> out_a(rows), out_b(rows);

        // Approach A: Fused single-pass (our current approach)
        double t_a = bench([&]() {
            #pragma omp parallel for
            for (int64_t r = 0; r < rows; ++r) {
                const float* row = &data[r * cols];
                double sum = 0;
                double count = 0;
                for (int64_t j = 0; j < cols; ++j) {
                    if (!std::isnan(row[j])) { sum += row[j]; count += 1.0; }
                }
                out_a[r] = (count > 0) ? static_cast<float>(sum / count) : std::nanf("");
            }
        });

        // Approach B: Two-pass (nansum with SIMD mask + count + SIMD divide)
        double t_b = bench([&]() {
            // Pass 1: nansum (SIMD with NaN masking) + count
            std::vector<float> sums(rows);
            std::vector<float> counts(rows);

            #pragma omp parallel for
            for (int64_t r = 0; r < rows; ++r) {
                const float* row = &data[r * cols];
                __m256 vsum = _mm256_setzero_ps();
                __m256 vcount = _mm256_setzero_ps();
                __m256 zero = _mm256_setzero_ps();
                __m256 ones = _mm256_set1_ps(1.0f);
                int64_t j = 0;
                for (; j + 8 <= cols; j += 8) {
                    __m256 v = _mm256_loadu_ps(row + j);
                    __m256 mask = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);  // NaN → all-1s
                    vsum = _mm256_add_ps(vsum, _mm256_blendv_ps(v, zero, mask));
                    vcount = _mm256_add_ps(vcount, _mm256_blendv_ps(ones, zero, mask));
                }
                // Horizontal reduce
                __m128 lo_s = _mm256_castps256_ps128(vsum), hi_s = _mm256_extractf128_ps(vsum, 1);
                lo_s = _mm_add_ps(lo_s, hi_s); lo_s = _mm_hadd_ps(lo_s, lo_s); lo_s = _mm_hadd_ps(lo_s, lo_s);
                float sum = _mm_cvtss_f32(lo_s);
                __m128 lo_c = _mm256_castps256_ps128(vcount), hi_c = _mm256_extractf128_ps(vcount, 1);
                lo_c = _mm_add_ps(lo_c, hi_c); lo_c = _mm_hadd_ps(lo_c, lo_c); lo_c = _mm_hadd_ps(lo_c, lo_c);
                float cnt = _mm_cvtss_f32(lo_c);
                for (; j < cols; ++j) {
                    if (!std::isnan(row[j])) { sum += row[j]; cnt += 1.0f; }
                }
                sums[r] = sum;
                counts[r] = cnt;
            }

            // Pass 2: SIMD divide
            int64_t r = 0;
            for (; r + 8 <= rows; r += 8) {
                __m256 vs = _mm256_loadu_ps(&sums[r]);
                __m256 vc = _mm256_loadu_ps(&counts[r]);
                _mm256_storeu_ps(&out_b[r], _mm256_div_ps(vs, vc));
            }
            for (; r < rows; ++r)
                out_b[r] = (counts[r] > 0) ? sums[r] / counts[r] : std::nanf("");
        });

        char shape_str[32];
        snprintf(shape_str, sizeof(shape_str), "(%ld, %ld)", rows, cols);
        const char* winner = (t_a < t_b) ? "Fused" : "Two-pass";
        std::cout << std::setw(18) << shape_str
                  << " | " << std::setw(13) << std::fixed << std::setprecision(1) << t_a << "μs"
                  << " | " << std::setw(13) << t_b << "μs"
                  << " | " << std::setw(10) << winner << "\n";
    }

    return 0;
}
