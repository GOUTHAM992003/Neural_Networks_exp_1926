// Benchmark: Two-Pass (optimized sum kernel + SIMD div) vs Fused Single-Pass
// For nanmean: which approach is better?
//
// Two-Pass:  cascade_sum(nansum) [SIMD+cascade+OpenMP] → count_non_nan [SIMD+OpenMP] → SIMD divide
// Fused:     One loop per output: scalar sum + scalar count + scalar div (no cascade, no SIMD sum)

#include <immintrin.h>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>

// ============================================================
// TWO-PASS APPROACH (mimics calling optimized sum kernel + separate count)
// ============================================================

// Pass 1: Optimized SIMD nansum with double accumulator + 4-acc ILP
static void optimized_nansum_pass(const float* input, double* sums, int64_t rows, int64_t cols) {
    #pragma omp parallel for
    for (int64_t r = 0; r < rows; ++r) {
        const float* row = input + r * cols;
        // 4-accumulator ILP (like cascade_sum)
        __m256d acc0 = _mm256_setzero_pd();
        __m256d acc1 = _mm256_setzero_pd();
        __m256d acc2 = _mm256_setzero_pd();
        __m256d acc3 = _mm256_setzero_pd();
        __m256 zero_f = _mm256_setzero_ps();

        int64_t j = 0;
        for (; j + 32 <= cols; j += 32) {
            // Block 0: 8 floats → NaN mask → split to 2×4 doubles
            __m256 v0 = _mm256_loadu_ps(row + j);
            __m256 m0 = _mm256_cmp_ps(v0, v0, _CMP_UNORD_Q);
            v0 = _mm256_blendv_ps(v0, zero_f, m0);
            acc0 = _mm256_add_pd(acc0, _mm256_cvtps_pd(_mm256_castps256_ps128(v0)));
            acc1 = _mm256_add_pd(acc1, _mm256_cvtps_pd(_mm256_extractf128_ps(v0, 1)));

            // Block 1
            __m256 v1 = _mm256_loadu_ps(row + j + 8);
            __m256 m1 = _mm256_cmp_ps(v1, v1, _CMP_UNORD_Q);
            v1 = _mm256_blendv_ps(v1, zero_f, m1);
            acc2 = _mm256_add_pd(acc2, _mm256_cvtps_pd(_mm256_castps256_ps128(v1)));
            acc3 = _mm256_add_pd(acc3, _mm256_cvtps_pd(_mm256_extractf128_ps(v1, 1)));

            // Block 2
            __m256 v2 = _mm256_loadu_ps(row + j + 16);
            __m256 m2 = _mm256_cmp_ps(v2, v2, _CMP_UNORD_Q);
            v2 = _mm256_blendv_ps(v2, zero_f, m2);
            acc0 = _mm256_add_pd(acc0, _mm256_cvtps_pd(_mm256_castps256_ps128(v2)));
            acc1 = _mm256_add_pd(acc1, _mm256_cvtps_pd(_mm256_extractf128_ps(v2, 1)));

            // Block 3
            __m256 v3 = _mm256_loadu_ps(row + j + 24);
            __m256 m3 = _mm256_cmp_ps(v3, v3, _CMP_UNORD_Q);
            v3 = _mm256_blendv_ps(v3, zero_f, m3);
            acc2 = _mm256_add_pd(acc2, _mm256_cvtps_pd(_mm256_castps256_ps128(v3)));
            acc3 = _mm256_add_pd(acc3, _mm256_cvtps_pd(_mm256_extractf128_ps(v3, 1)));
        }
        // 8-wide tail
        for (; j + 8 <= cols; j += 8) {
            __m256 v = _mm256_loadu_ps(row + j);
            __m256 m = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
            v = _mm256_blendv_ps(v, zero_f, m);
            acc0 = _mm256_add_pd(acc0, _mm256_cvtps_pd(_mm256_castps256_ps128(v)));
            acc1 = _mm256_add_pd(acc1, _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1)));
        }
        // Combine 4 accumulators
        __m256d total = _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3));
        double arr[4];
        _mm256_storeu_pd(arr, total);
        double sum = arr[0] + arr[1] + arr[2] + arr[3];
        // Scalar tail
        for (; j < cols; ++j) {
            float v = row[j];
            if (!std::isnan(v)) sum += (double)v;
        }
        sums[r] = sum;
    }
}

// Pass 2: Optimized SIMD non-NaN count
static void optimized_count_pass(const float* input, double* counts, int64_t rows, int64_t cols) {
    #pragma omp parallel for
    for (int64_t r = 0; r < rows; ++r) {
        const float* row = input + r * cols;
        __m256 ones = _mm256_set1_ps(1.0f);
        __m256 zero_f = _mm256_setzero_ps();
        __m256 vcount = _mm256_setzero_ps();

        int64_t j = 0;
        for (; j + 8 <= cols; j += 8) {
            __m256 v = _mm256_loadu_ps(row + j);
            __m256 mask = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
            vcount = _mm256_add_ps(vcount, _mm256_blendv_ps(ones, zero_f, mask));
        }
        // Horizontal sum
        float cnt_arr[8];
        _mm256_storeu_ps(cnt_arr, vcount);
        double count = 0;
        for (int k = 0; k < 8; ++k) count += cnt_arr[k];
        for (; j < cols; ++j)
            if (!std::isnan(row[j])) count += 1.0;
        counts[r] = count;
    }
}

// Pass 3: SIMD division (over output elements)
static void simd_divide(const double* sums, const double* counts, float* output, int64_t n) {
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d vs = _mm256_loadu_pd(sums + i);
        __m256d vc = _mm256_loadu_pd(counts + i);
        __m256d vr = _mm256_div_pd(vs, vc);
        __m128 f4 = _mm256_cvtpd_ps(vr);
        _mm_storeu_ps(output + i, f4);
    }
    for (; i < n; ++i)
        output[i] = (float)(sums[i] / counts[i]);
}

static void two_pass_nanmean(const float* input, float* output, int64_t rows, int64_t cols) {
    std::vector<double> sums(rows), counts(rows);
    optimized_nansum_pass(input, sums.data(), rows, cols);
    optimized_count_pass(input, counts.data(), rows, cols);
    simd_divide(sums.data(), counts.data(), output, rows);
}

// ============================================================
// FUSED SINGLE-PASS (current approach — scalar, no cascade)
// ============================================================

static void fused_scalar_nanmean(const float* input, float* output, int64_t rows, int64_t cols) {
    #pragma omp parallel for
    for (int64_t r = 0; r < rows; ++r) {
        const float* row = input + r * cols;
        double sum = 0.0;
        double count = 0.0;
        for (int64_t j = 0; j < cols; ++j) {
            float v = row[j];
            if (!std::isnan(v)) {
                sum += (double)v;
                count += 1.0;
            }
        }
        output[r] = (count > 0) ? (float)(sum / count) : std::nanf("");
    }
}

// ============================================================
// FUSED SINGLE-PASS + SIMD (branchless NaN mask, double acc)
// ============================================================

static void fused_simd_nanmean(const float* input, float* output, int64_t rows, int64_t cols) {
    #pragma omp parallel for
    for (int64_t r = 0; r < rows; ++r) {
        const float* row = input + r * cols;
        __m256d vsum0 = _mm256_setzero_pd();
        __m256d vsum1 = _mm256_setzero_pd();
        __m256d vcount0 = _mm256_setzero_pd();
        __m256d vcount1 = _mm256_setzero_pd();
        __m256 zero_f = _mm256_setzero_ps();
        __m256 ones_f = _mm256_set1_ps(1.0f);

        int64_t j = 0;
        for (; j + 8 <= cols; j += 8) {
            __m256 v = _mm256_loadu_ps(row + j);
            __m256 mask = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
            __m256 safe_v = _mm256_blendv_ps(v, zero_f, mask);
            __m256 cnt = _mm256_blendv_ps(ones_f, zero_f, mask);
            // Convert to double and accumulate
            vsum0 = _mm256_add_pd(vsum0, _mm256_cvtps_pd(_mm256_castps256_ps128(safe_v)));
            vsum1 = _mm256_add_pd(vsum1, _mm256_cvtps_pd(_mm256_extractf128_ps(safe_v, 1)));
            vcount0 = _mm256_add_pd(vcount0, _mm256_cvtps_pd(_mm256_castps256_ps128(cnt)));
            vcount1 = _mm256_add_pd(vcount1, _mm256_cvtps_pd(_mm256_extractf128_ps(cnt, 1)));
        }
        // Horizontal reduce
        __m256d vtotal_sum = _mm256_add_pd(vsum0, vsum1);
        __m256d vtotal_cnt = _mm256_add_pd(vcount0, vcount1);
        double sarr[4], carr[4];
        _mm256_storeu_pd(sarr, vtotal_sum);
        _mm256_storeu_pd(carr, vtotal_cnt);
        double sum = sarr[0] + sarr[1] + sarr[2] + sarr[3];
        double count = carr[0] + carr[1] + carr[2] + carr[3];
        // Scalar tail
        for (; j < cols; ++j) {
            float v = row[j];
            if (!std::isnan(v)) { sum += (double)v; count += 1.0; }
        }
        output[r] = (count > 0) ? (float)(sum / count) : std::nanf("");
    }
}

// ============================================================
// BENCHMARK HARNESS
// ============================================================

struct BenchResult { double us; const char* name; };

template <typename F>
BenchResult bench(const char* name, F fn, int warmup = 5, int iters = 30) {
    for (int i = 0; i < warmup; ++i) fn();
    std::vector<double> times;
    for (int i = 0; i < iters; ++i) {
        auto s = std::chrono::high_resolution_clock::now();
        fn();
        auto e = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::micro>(e - s).count());
    }
    std::sort(times.begin(), times.end());
    double median = times[iters / 2];
    return {median, name};
}

void fill_with_nan(float* data, int64_t n, double nan_pct, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (int64_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
        if (u(rng) < nan_pct) data[i] = std::nanf("");
    }
}

void run_test(const char* label, int64_t rows, int64_t cols, double nan_pct) {
    int64_t n = rows * cols;
    std::vector<float> input(n);
    std::vector<float> out_2pass(rows), out_fused(rows), out_fused_simd(rows);
    std::mt19937 rng(42);
    fill_with_nan(input.data(), n, nan_pct, rng);

    auto r_2pass = bench("Two-Pass (optimized)", [&]() {
        two_pass_nanmean(input.data(), out_2pass.data(), rows, cols);
    });
    auto r_fused = bench("Fused Scalar", [&]() {
        fused_scalar_nanmean(input.data(), out_fused.data(), rows, cols);
    });
    auto r_fused_simd = bench("Fused SIMD", [&]() {
        fused_simd_nanmean(input.data(), out_fused_simd.data(), rows, cols);
    });

    // Verify correctness
    double max_diff = 0;
    for (int64_t i = 0; i < rows; ++i) {
        if (std::isnan(out_2pass[i]) && std::isnan(out_fused_simd[i])) continue;
        max_diff = std::max(max_diff, (double)std::abs(out_2pass[i] - out_fused_simd[i]));
    }

    double best = std::min({r_2pass.us, r_fused.us, r_fused_simd.us});
    printf("%-45s | %10.0f μs (%4.2fx) | %10.0f μs (%4.2fx) | %10.0f μs (%4.2fx) | diff=%.2e\n",
           label,
           r_2pass.us, r_2pass.us / best,
           r_fused.us, r_fused.us / best,
           r_fused_simd.us, r_fused_simd.us / best,
           max_diff);
}

int main() {
    printf("CPU: %d threads\n", omp_get_max_threads());
    printf("Approaches: Two-Pass = optimized nansum + SIMD count + SIMD div\n");
    printf("            Fused Scalar = one loop, scalar sum+count+div\n");
    printf("            Fused SIMD = one loop, SIMD sum+count, scalar div\n\n");

    printf("%-45s | %18s | %18s | %18s | %s\n",
           "Test Case", "Two-Pass", "Fused Scalar", "Fused SIMD", "Precision");
    printf("%s\n", std::string(130, '-').c_str());

    // === DL-typical shapes ===
    printf("\n--- Deep Learning Typical Shapes (10%% NaN) ---\n");
    run_test("Layer norm: (32, 768)", 32, 768, 0.1);
    run_test("Layer norm: (32, 4096)", 32, 4096, 0.1);
    run_test("Seq layer norm: (4096, 768)", 4096, 768, 0.1);
    run_test("Batch mean: (256, 4096)", 256, 4096, 0.1);
    run_test("Spatial mean: (2048, 50176)", 2048, 50176, 0.1);
    run_test("Attention: (49152, 128)", 49152, 128, 0.1);
    run_test("Feature agg: (4096, 256)", 4096, 256, 0.1);

    // === Varying reduction size ===
    printf("\n--- Varying Reduction Size (1000 outputs, 10%% NaN) ---\n");
    run_test("(1000, 100)", 1000, 100, 0.1);
    run_test("(1000, 1000)", 1000, 1000, 0.1);
    run_test("(1000, 10000)", 1000, 10000, 0.1);
    run_test("(1000, 100000)", 1000, 100000, 0.1);
    run_test("(1000, 1000000)", 1000, 1000000, 0.1);

    // === Varying output count ===
    printf("\n--- Varying Output Count (10000 reduction, 10%% NaN) ---\n");
    run_test("(1, 10000) FULL REDUCTION", 1, 10000, 0.1);
    run_test("(10, 10000)", 10, 10000, 0.1);
    run_test("(100, 10000)", 100, 10000, 0.1);
    run_test("(1000, 10000)", 1000, 10000, 0.1);
    run_test("(10000, 10000)", 10000, 10000, 0.1);
    run_test("(100000, 10000)", 100000, 10000, 0.1);

    // === Varying NaN percentage ===
    printf("\n--- Varying NaN %% (1000, 10000) ---\n");
    run_test("0%% NaN (no NaN at all)", 1000, 10000, 0.0);
    run_test("1%% NaN", 1000, 10000, 0.01);
    run_test("10%% NaN", 1000, 10000, 0.1);
    run_test("50%% NaN", 1000, 10000, 0.5);
    run_test("90%% NaN (mostly NaN)", 1000, 10000, 0.9);
    run_test("99%% NaN", 1000, 10000, 0.99);

    // === Edge cases ===
    printf("\n--- Edge Cases ---\n");
    run_test("Tiny: (8, 16)", 8, 16, 0.1);
    run_test("Tiny: (1, 100)", 1, 100, 0.1);
    run_test("Single col: (100000, 1)", 100000, 1, 0.1);
    run_test("Square: (1000, 1000)", 1000, 1000, 0.1);
    run_test("Wide: (10, 1000000)", 10, 1000000, 0.1);
    run_test("Tall: (1000000, 10)", 1000000, 10, 0.1);
    run_test("Huge: (100, 10000000)", 100, 10000000, 0.1);

    printf("\n");
    return 0;
}
