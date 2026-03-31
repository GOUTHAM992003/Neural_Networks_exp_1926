// ═══════════════════════════════════════════════════════════════════════════
// BENCHMARK: SIMD Division vs Reciprocal-Multiply for Mean Computation
//
// Strategy A: result[i] = sum[i] / N              (SIMD vdivps — PyTorch style)
// Strategy B: inv = 1/N; result[i] = sum[i] * inv (SIMD vmulps — reciprocal)
// Strategy C: inv = 1.0/(double)N; result = (float)(sum * inv)  (f64 reciprocal)
//
// DATASETS: 14 different distributions covering real ML workloads + adversarial
// ═══════════════════════════════════════════════════════════════════════════

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <immintrin.h>
#include <cfloat>
#include <string>

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════

struct Timer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point start;
    void begin() { start = clock::now(); }
    double elapsed_us() const {
        return std::chrono::duration<double, std::micro>(clock::now() - start).count();
    }
};

int32_t ulp_distance_f32(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) return INT32_MAX;
    if (a == b) return 0;
    int32_t ia, ib;
    std::memcpy(&ia, &a, 4);
    std::memcpy(&ib, &b, 4);
    if ((ia < 0) != (ib < 0)) return std::abs(ia) + std::abs(ib);
    return std::abs(ia - ib);
}

int64_t ulp_distance_f64(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return INT64_MAX;
    if (a == b) return 0;
    int64_t ia, ib;
    std::memcpy(&ia, &a, 8);
    std::memcpy(&ib, &b, 8);
    if ((ia < 0) != (ib < 0)) return std::abs(ia) + std::abs(ib);
    return std::abs(ia - ib);
}

// ═══════════════════════════════════════════════════════════════════════════
// STRATEGIES (float32)
// ═══════════════════════════════════════════════════════════════════════════

void mean_division_f32(const float* sums, float* out, int64_t count, float N) {
    __m256 vN = _mm256_set1_ps(N);
    int64_t i = 0;
    for (; i + 8 <= count; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_div_ps(_mm256_loadu_ps(sums + i), vN));
    }
    for (; i < count; ++i) out[i] = sums[i] / N;
}

void mean_reciprocal_f32(const float* sums, float* out, int64_t count, float N) {
    float inv = 1.0f / N;
    __m256 vInv = _mm256_set1_ps(inv);
    int64_t i = 0;
    for (; i + 8 <= count; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(sums + i), vInv));
    }
    for (; i < count; ++i) out[i] = sums[i] * inv;
}

void mean_reciprocal_f32_via_f64(const float* sums, float* out, int64_t count, float N) {
    double inv = 1.0 / static_cast<double>(N);
    __m256d vInv = _mm256_set1_pd(inv);
    int64_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m128 lo4 = _mm_loadu_ps(sums + i);
        __m128 hi4 = _mm_loadu_ps(sums + i + 4);
        __m256d dlo = _mm256_mul_pd(_mm256_cvtps_pd(lo4), vInv);
        __m256d dhi = _mm256_mul_pd(_mm256_cvtps_pd(hi4), vInv);
        _mm256_storeu_ps(out + i, _mm256_set_m128(_mm256_cvtpd_ps(dhi), _mm256_cvtpd_ps(dlo)));
    }
    for (; i < count; ++i)
        out[i] = static_cast<float>(static_cast<double>(sums[i]) * inv);
}

// ═══════════════════════════════════════════════════════════════════════════
// STRATEGIES (float64)
// ═══════════════════════════════════════════════════════════════════════════

void mean_division_f64(const double* sums, double* out, int64_t count, double N) {
    __m256d vN = _mm256_set1_pd(N);
    int64_t i = 0;
    for (; i + 4 <= count; i += 4) {
        _mm256_storeu_pd(out + i, _mm256_div_pd(_mm256_loadu_pd(sums + i), vN));
    }
    for (; i < count; ++i) out[i] = sums[i] / N;
}

void mean_reciprocal_f64(const double* sums, double* out, int64_t count, double N) {
    double inv = 1.0 / N;
    __m256d vInv = _mm256_set1_pd(inv);
    int64_t i = 0;
    for (; i + 4 <= count; i += 4) {
        _mm256_storeu_pd(out + i, _mm256_mul_pd(_mm256_loadu_pd(sums + i), vInv));
    }
    for (; i < count; ++i) out[i] = sums[i] * inv;
}

// ═══════════════════════════════════════════════════════════════════════════
// GROUND TRUTH
// ═══════════════════════════════════════════════════════════════════════════

template <typename T>
void ground_truth(const T* sums, long double* out, int64_t count, long double N) {
    for (int64_t i = 0; i < count; ++i)
        out[i] = static_cast<long double>(sums[i]) / N;
}

// ═══════════════════════════════════════════════════════════════════════════
// DATASET GENERATORS
// ═══════════════════════════════════════════════════════════════════════════

struct Dataset {
    std::string name;
    std::string description;
    std::vector<float> data_f32;
    std::vector<double> data_f64;
    float divisor_f32;
    double divisor_f64;
};

std::vector<Dataset> generate_all_datasets() {
    std::vector<Dataset> datasets;
    std::mt19937 rng(42);
    const int64_t N = 100000;

    // ──────────────────────────────────────────────────────────
    // 1. UNIFORM RANDOM [-1, 1] — standard ML activations
    // ──────────────────────────────────────────────────────────
    {
        Dataset d;
        d.name = "Uniform [-1, 1]";
        d.description = "Standard ML activation outputs (tanh, normalized)";
        d.divisor_f32 = 1000.0f; d.divisor_f64 = 1000.0;
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        d.data_f32.resize(N); d.data_f64.resize(N);
        for (int64_t i = 0; i < N; ++i) {
            d.data_f32[i] = dist(rng);
            d.data_f64[i] = static_cast<double>(d.data_f32[i]);
        }
        datasets.push_back(std::move(d));
    }

    // ──────────────────────────────────────────────────────────
    // 2. NORMAL (Gaussian) — typical neural network gradients
    // ──────────────────────────────────────────────────────────
    {
        Dataset d;
        d.name = "Gaussian N(0,1)";
        d.description = "Standard normal — typical NN gradients/weights";
        d.divisor_f32 = 512.0f; d.divisor_f64 = 512.0;
        std::normal_distribution<float> dist(0.0f, 1.0f);
        d.data_f32.resize(N); d.data_f64.resize(N);
        for (int64_t i = 0; i < N; ++i) {
            d.data_f32[i] = dist(rng);
            d.data_f64[i] = static_cast<double>(d.data_f32[i]);
        }
        datasets.push_back(std::move(d));
    }

    // ──────────────────────────────────────────────────────────
    // 3. LARGE MEAN + TINY VARIANCE — catastrophic cancellation risk
    //    E.g., sensor readings around 1e6 with noise ±0.01
    // ──────────────────────────────────────────────────────────
    {
        Dataset d;
        d.name = "Large Mean + Tiny Var";
        d.description = "1e6 ± 0.01 — catastrophic cancellation stress test";
        d.divisor_f32 = 7.0f; d.divisor_f64 = 7.0;
        std::normal_distribution<float> dist(1e6f, 0.01f);
        d.data_f32.resize(N); d.data_f64.resize(N);
        for (int64_t i = 0; i < N; ++i) {
            d.data_f32[i] = dist(rng);
            d.data_f64[i] = static_cast<double>(d.data_f32[i]);
        }
        datasets.push_back(std::move(d));
    }

    // ──────────────────────────────────────────────────────────
    // 4. EXPONENTIAL — loss values, learning rates (always positive)
    // ──────────────────────────────────────────────────────────
    {
        Dataset d;
        d.name = "Exponential(λ=1)";
        d.description = "Positive skewed — loss values, learning rates";
        d.divisor_f32 = 3.0f; d.divisor_f64 = 3.0;
        std::exponential_distribution<float> dist(1.0f);
        d.data_f32.resize(N); d.data_f64.resize(N);
        for (int64_t i = 0; i < N; ++i) {
            d.data_f32[i] = dist(rng);
            d.data_f64[i] = static_cast<double>(d.data_f32[i]);
        }
        datasets.push_back(std::move(d));
    }

    // ──────────────────────────────────────────────────────────
    // 5. MIXED SCALE — values spanning 1e-6 to 1e6
    //    Simulates batch norm with wildly varying feature scales
    // ──────────────────────────────────────────────────────────
    {
        Dataset d;
        d.name = "Mixed Scale [1e-6, 1e6]";
        d.description = "Wildly varying scales — batch norm nightmare";
        d.divisor_f32 = 13.0f; d.divisor_f64 = 13.0;
        std::uniform_real_distribution<float> exp_dist(-6.0f, 6.0f);
        std::uniform_real_distribution<float> sign_dist(-1.0f, 1.0f);
        d.data_f32.resize(N); d.data_f64.resize(N);
        for (int64_t i = 0; i < N; ++i) {
            float sign = sign_dist(rng) > 0 ? 1.0f : -1.0f;
            d.data_f32[i] = sign * std::pow(10.0f, exp_dist(rng));
            d.data_f64[i] = static_cast<double>(d.data_f32[i]);
        }
        datasets.push_back(std::move(d));
    }

    // ──────────────────────────────────────────────────────────
    // 6. SUBNORMAL NUMBERS — denormalized floats near zero
    //    Tests IEEE 754 edge behavior
    // ──────────────────────────────────────────────────────────
    {
        Dataset d;
        d.name = "Subnormals";
        d.description = "Denormalized floats ~1e-40 — IEEE 754 edge case";
        d.divisor_f32 = 17.0f; d.divisor_f64 = 17.0;
        d.data_f32.resize(N); d.data_f64.resize(N);
        for (int64_t i = 0; i < N; ++i) {
            // Generate subnormal floats: exponent = 0, random mantissa
            uint32_t bits = rng() & 0x007FFFFF; // mantissa only, exp=0, sign=0
            if (rng() % 2) bits |= 0x80000000;  // random sign
            std::memcpy(&d.data_f32[i], &bits, 4);
            d.data_f64[i] = static_cast<double>(d.data_f32[i]);
        }
        datasets.push_back(std::move(d));
    }

    // ──────────────────────────────────────────────────────────
    // 7. NEAR-MAX FLOATS — values near FLT_MAX
    //    Tests overflow boundary behavior
    // ──────────────────────────────────────────────────────────
    {
        Dataset d;
        d.name = "Near FLT_MAX";
        d.description = "Values ~1e38 — overflow boundary stress test";
        d.divisor_f32 = 1000003.0f; d.divisor_f64 = 1000003.0;
        std::uniform_real_distribution<float> dist(FLT_MAX * 0.5f, FLT_MAX * 0.99f);
        d.data_f32.resize(N); d.data_f64.resize(N);
        for (int64_t i = 0; i < N; ++i) {
            d.data_f32[i] = (rng() % 2 ? 1.0f : -1.0f) * dist(rng);
            d.data_f64[i] = static_cast<double>(d.data_f32[i]);
        }
        datasets.push_back(std::move(d));
    }

    // ──────────────────────────────────────────────────────────
    // 8. ALL IDENTICAL VALUES — worst case for cancellation detection
    // ──────────────────────────────────────────────────────────
    {
        Dataset d;
        d.name = "All Identical (π)";
        d.description = "Every element = π — tests perfect cancellation";
        d.divisor_f32 = 7.0f; d.divisor_f64 = 7.0;
        d.data_f32.resize(N, 3.14159265f);
        d.data_f64.resize(N, 3.14159265358979323846);
        datasets.push_back(std::move(d));
    }

    // ──────────────────────────────────────────────────────────
    // 9. ALTERNATING SIGN — +1, -1, +1, -1 (massive cancellation)
    // ──────────────────────────────────────────────────────────
    {
        Dataset d;
        d.name = "Alternating ±1";
        d.description = "Perfect cancellation — sum ≈ 0, each element large";
        d.divisor_f32 = 11.0f; d.divisor_f64 = 11.0;
        d.data_f32.resize(N); d.data_f64.resize(N);
        for (int64_t i = 0; i < N; ++i) {
            d.data_f32[i] = (i % 2 == 0) ? 1.0f : -1.0f;
            d.data_f64[i] = static_cast<double>(d.data_f32[i]);
        }
        datasets.push_back(std::move(d));
    }

    // ──────────────────────────────────────────────────────────
    // 10. SOFTMAX OUTPUT — values in (0, 1) summing to ~1.0
    //     Typical output of softmax layer
    // ──────────────────────────────────────────────────────────
    {
        Dataset d;
        d.name = "Softmax-like [0,1]";
        d.description = "Softmax probabilities — small positive values sum ~1";
        d.divisor_f32 = 1000.0f; d.divisor_f64 = 1000.0;
        std::exponential_distribution<float> dist(10.0f);
        d.data_f32.resize(N); d.data_f64.resize(N);
        float sum = 0;
        for (int64_t i = 0; i < N; ++i) {
            d.data_f32[i] = dist(rng);
            sum += d.data_f32[i];
        }
        // Normalize to sum=1
        for (int64_t i = 0; i < N; ++i) {
            d.data_f32[i] /= sum;
            d.data_f64[i] = static_cast<double>(d.data_f32[i]);
        }
        datasets.push_back(std::move(d));
    }

    // ──────────────────────────────────────────────────────────
    // 11. LOG-NORMAL — financial data, image pixel intensities
    // ──────────────────────────────────────────────────────────
    {
        Dataset d;
        d.name = "Log-Normal";
        d.description = "Financial data / pixel intensities — heavy right tail";
        d.divisor_f32 = 19.0f; d.divisor_f64 = 19.0;
        std::lognormal_distribution<float> dist(0.0f, 1.0f);
        d.data_f32.resize(N); d.data_f64.resize(N);
        for (int64_t i = 0; i < N; ++i) {
            d.data_f32[i] = dist(rng);
            d.data_f64[i] = static_cast<double>(d.data_f32[i]);
        }
        datasets.push_back(std::move(d));
    }

    // ──────────────────────────────────────────────────────────
    // 12. SPARSE (90% zeros) — typical ReLU activation output
    // ──────────────────────────────────────────────────────────
    {
        Dataset d;
        d.name = "Sparse (90% zero)";
        d.description = "ReLU output — mostly zeros with random positive spikes";
        d.divisor_f32 = 23.0f; d.divisor_f64 = 23.0;
        std::normal_distribution<float> dist(0.0f, 1.0f);
        d.data_f32.resize(N, 0.0f); d.data_f64.resize(N, 0.0);
        for (int64_t i = 0; i < N; ++i) {
            if (rng() % 10 == 0) {  // 10% nonzero
                float v = std::abs(dist(rng));
                d.data_f32[i] = v;
                d.data_f64[i] = static_cast<double>(v);
            }
        }
        datasets.push_back(std::move(d));
    }

    // ──────────────────────────────────────────────────────────
    // 13. INTEGER-VALUED FLOATS — quantization, token IDs
    // ──────────────────────────────────────────────────────────
    {
        Dataset d;
        d.name = "Integer-valued floats";
        d.description = "Quantized values {0,1,2,...,255} stored as float";
        d.divisor_f32 = 127.0f; d.divisor_f64 = 127.0;
        d.data_f32.resize(N); d.data_f64.resize(N);
        for (int64_t i = 0; i < N; ++i) {
            d.data_f32[i] = static_cast<float>(rng() % 256);
            d.data_f64[i] = static_cast<double>(d.data_f32[i]);
        }
        datasets.push_back(std::move(d));
    }

    // ──────────────────────────────────────────────────────────
    // 14. ADVERSARIAL — ONE huge value + rest tiny
    //     Tests if single outlier destroys precision
    // ──────────────────────────────────────────────────────────
    {
        Dataset d;
        d.name = "Adversarial outlier";
        d.description = "1 value = 1e30, rest = 1e-10 — outlier stress test";
        d.divisor_f32 = 7.0f; d.divisor_f64 = 7.0;
        d.data_f32.resize(N, 1e-10f);
        d.data_f64.resize(N, 1e-10);
        d.data_f32[0] = 1e30f;
        d.data_f64[0] = 1e30;
        datasets.push_back(std::move(d));
    }

    return datasets;
}

// ═══════════════════════════════════════════════════════════════════════════
// BENCHMARK RUNNERS
// ═══════════════════════════════════════════════════════════════════════════

struct PrecisionStats {
    double avg_ulp;
    int64_t max_ulp;
    int64_t exact_count;
    int64_t total;
};

void print_header(const std::string& name, const std::string& desc, 
                  const std::string& dtype, int64_t count, double N) {
    std::cout << "\n┌──────────────────────────────────────────────────────────────────┐\n";
    std::cout << "│ " << std::left << std::setw(65) << (dtype + ": " + name) << "│\n";
    std::cout << "│ " << std::setw(65) << desc << "│\n";
    std::cout << "│ Elements: " << std::setw(12) << count 
              << "Divisor N: " << std::setw(30) << std::setprecision(6) << N << "│\n";
    std::cout << "├──────────────────────────────────────────────────────────────────┤\n";
}

void print_speed(const std::string& label, double time_us, double baseline_us) {
    std::cout << "│  " << std::left << std::setw(28) << label 
              << std::right << std::setw(8) << std::fixed << std::setprecision(2) << time_us << " μs";
    if (baseline_us > 0) {
        std::cout << "  (" << std::setprecision(1) << baseline_us / time_us << "x)";
    }
    std::cout << "\n";
}

void print_precision(const std::string& label, PrecisionStats s) {
    std::cout << "│  " << std::left << std::setw(28) << label
              << "avg=" << std::setprecision(3) << std::setw(6) << s.avg_ulp << " ULP  "
              << "max=" << std::setw(3) << s.max_ulp << " ULP  "
              << "exact=" << s.exact_count << "/" << s.total << "\n";
}

void run_f32_dataset(const Dataset& ds) {
    const int64_t count = ds.data_f32.size();
    const float N = ds.divisor_f32;
    
    std::vector<float> out_div(count), out_recip(count), out_recip64(count);
    std::vector<long double> out_truth(count);
    
    ground_truth(ds.data_f32.data(), out_truth.data(), count, static_cast<long double>(N));
    
    // Warm up
    mean_division_f32(ds.data_f32.data(), out_div.data(), count, N);
    mean_reciprocal_f32(ds.data_f32.data(), out_recip.data(), count, N);
    mean_reciprocal_f32_via_f64(ds.data_f32.data(), out_recip64.data(), count, N);
    
    // Speed
    constexpr int ITERS = 2000;
    Timer t;
    
    t.begin();
    for (int i = 0; i < ITERS; ++i) {
        mean_division_f32(ds.data_f32.data(), out_div.data(), count, N);
        asm volatile("" ::: "memory");
    }
    double time_div = t.elapsed_us() / ITERS;
    
    t.begin();
    for (int i = 0; i < ITERS; ++i) {
        mean_reciprocal_f32(ds.data_f32.data(), out_recip.data(), count, N);
        asm volatile("" ::: "memory");
    }
    double time_recip = t.elapsed_us() / ITERS;
    
    t.begin();
    for (int i = 0; i < ITERS; ++i) {
        mean_reciprocal_f32_via_f64(ds.data_f32.data(), out_recip64.data(), count, N);
        asm volatile("" ::: "memory");
    }
    double time_recip64 = t.elapsed_us() / ITERS;
    
    // Precision
    auto compute_stats = [&](const float* out) -> PrecisionStats {
        PrecisionStats s{0, 0, 0, count};
        for (int64_t i = 0; i < count; ++i) {
            float truth_f = static_cast<float>(out_truth[i]);
            int32_t ud = ulp_distance_f32(out[i], truth_f);
            s.max_ulp = std::max(s.max_ulp, (int64_t)ud);
            s.avg_ulp += ud;
            if (ud == 0) ++s.exact_count;
        }
        s.avg_ulp /= count;
        return s;
    };
    
    PrecisionStats s_div = compute_stats(out_div.data());
    PrecisionStats s_recip = compute_stats(out_recip.data());
    PrecisionStats s_recip64 = compute_stats(out_recip64.data());
    
    // Check for value mismatches (not just ULP but actual wrong answers)
    int big_errors_recip = 0, big_errors_recip64 = 0;
    double max_rel_err_recip = 0, max_rel_err_recip64 = 0;
    for (int64_t i = 0; i < count; ++i) {
        float truth_f = static_cast<float>(out_truth[i]);
        if (truth_f != 0) {
            double rel_r = std::abs((out_recip[i] - truth_f) / truth_f);
            double rel_r64 = std::abs((out_recip64[i] - truth_f) / truth_f);
            max_rel_err_recip = std::max(max_rel_err_recip, rel_r);
            max_rel_err_recip64 = std::max(max_rel_err_recip64, rel_r64);
            if (rel_r > 1e-5) ++big_errors_recip;
            if (rel_r64 > 1e-5) ++big_errors_recip64;
        }
    }
    
    // Print
    print_header(ds.name, ds.description, "F32", count, N);
    std::cout << "│  SPEED (avg of " << ITERS << " iterations):\n";
    print_speed("A. Division  (vdivps)", time_div, 0);
    print_speed("B. Reciprocal (vmulps)", time_recip, time_div);
    print_speed("C. Recip-f64 (f32→f64)", time_recip64, time_div);
    std::cout << "├──────────────────────────────────────────────────────────────────┤\n";
    std::cout << "│  PRECISION (vs long double):\n";
    print_precision("A. Division:", s_div);
    print_precision("B. Reciprocal:", s_recip);
    print_precision("C. Recip-f64:", s_recip64);
    std::cout << "├──────────────────────────────────────────────────────────────────┤\n";
    std::cout << "│  RELATIVE ERROR:\n";
    std::cout << "│  B. Reciprocal:  max_rel=" << std::scientific << std::setprecision(2) 
              << max_rel_err_recip << "  big_errors(>1e-5)=" << big_errors_recip << "\n";
    std::cout << "│  C. Recip-f64:   max_rel=" << max_rel_err_recip64 
              << "  big_errors(>1e-5)=" << big_errors_recip64 << "\n";
    std::cout << "└──────────────────────────────────────────────────────────────────┘\n";
}

void run_f64_dataset(const Dataset& ds) {
    const int64_t count = ds.data_f64.size();
    const double N = ds.divisor_f64;
    
    std::vector<double> out_div(count), out_recip(count);
    std::vector<long double> out_truth(count);
    
    ground_truth(ds.data_f64.data(), out_truth.data(), count, static_cast<long double>(N));
    
    mean_division_f64(ds.data_f64.data(), out_div.data(), count, N);
    mean_reciprocal_f64(ds.data_f64.data(), out_recip.data(), count, N);
    
    constexpr int ITERS = 2000;
    Timer t;
    
    t.begin();
    for (int i = 0; i < ITERS; ++i) {
        mean_division_f64(ds.data_f64.data(), out_div.data(), count, N);
        asm volatile("" ::: "memory");
    }
    double time_div = t.elapsed_us() / ITERS;
    
    t.begin();
    for (int i = 0; i < ITERS; ++i) {
        mean_reciprocal_f64(ds.data_f64.data(), out_recip.data(), count, N);
        asm volatile("" ::: "memory");
    }
    double time_recip = t.elapsed_us() / ITERS;
    
    auto compute_stats = [&](const double* out) -> PrecisionStats {
        PrecisionStats s{0, 0, 0, count};
        for (int64_t i = 0; i < count; ++i) {
            double truth_d = static_cast<double>(out_truth[i]);
            int64_t ud = ulp_distance_f64(out[i], truth_d);
            s.max_ulp = std::max(s.max_ulp, ud);
            s.avg_ulp += ud;
            if (ud == 0) ++s.exact_count;
        }
        s.avg_ulp /= count;
        return s;
    };
    
    PrecisionStats s_div = compute_stats(out_div.data());
    PrecisionStats s_recip = compute_stats(out_recip.data());
    
    print_header(ds.name, ds.description, "F64", count, N);
    std::cout << "│  SPEED (avg of " << ITERS << " iterations):\n";
    print_speed("A. Division  (vdivpd)", time_div, 0);
    print_speed("B. Reciprocal (vmulpd)", time_recip, time_div);
    std::cout << "├──────────────────────────────────────────────────────────────────┤\n";
    std::cout << "│  PRECISION (vs long double):\n";
    print_precision("A. Division:", s_div);
    print_precision("B. Reciprocal:", s_recip);
    std::cout << "└──────────────────────────────────────────────────────────────────┘\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// DIVISOR SWEEP — test across many different N values for precision
// ═══════════════════════════════════════════════════════════════════════════

void run_divisor_sweep() {
    std::cout << "\n\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  DIVISOR SWEEP: Testing N = 2 to 100000 (precision only)       ║\n";
    std::cout << "║  Dataset: Gaussian N(0,1), 100K elements                       ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  N          │ Div max ULP │ Recip max ULP │ Recip-f64 max ULP  ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════╣\n";
    
    const int64_t count = 100000;
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(count);
    for (auto& v : data) v = dist(rng);
    
    std::vector<float> out_div(count), out_recip(count), out_recip64(count);
    std::vector<long double> out_truth(count);
    
    float divisors[] = {2, 3, 5, 7, 10, 11, 13, 16, 17, 19, 23, 31, 32, 64, 
                        100, 127, 128, 255, 256, 500, 512, 997, 1000, 1024, 
                        4096, 10000, 32768, 65536, 100000};
    
    for (float N : divisors) {
        ground_truth(data.data(), out_truth.data(), count, static_cast<long double>(N));
        mean_division_f32(data.data(), out_div.data(), count, N);
        mean_reciprocal_f32(data.data(), out_recip.data(), count, N);
        mean_reciprocal_f32_via_f64(data.data(), out_recip64.data(), count, N);
        
        int64_t max_div = 0, max_recip = 0, max_recip64 = 0;
        for (int64_t i = 0; i < count; ++i) {
            float tf = static_cast<float>(out_truth[i]);
            max_div = std::max(max_div, (int64_t)ulp_distance_f32(out_div[i], tf));
            max_recip = std::max(max_recip, (int64_t)ulp_distance_f32(out_recip[i], tf));
            max_recip64 = std::max(max_recip64, (int64_t)ulp_distance_f32(out_recip64[i], tf));
        }
        
        std::string n_str = std::to_string(static_cast<int>(N));
        bool is_pow2 = (static_cast<int>(N) & (static_cast<int>(N) - 1)) == 0 && N > 0;
        std::string tag = is_pow2 ? " (2^" + std::to_string(static_cast<int>(std::log2(N))) + ")" : "";
        
        std::cout << "║  " << std::left << std::setw(11) << (n_str + tag)
                  << "│ " << std::setw(11) << max_div
                  << " │ " << std::setw(13) << max_recip
                  << " │ " << std::setw(18) << max_recip64 << " ║\n";
    }
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// SPECIAL: Very small element count (1 to 100)
// Tests overhead dominance regime
// ═══════════════════════════════════════════════════════════════════════════

void run_tiny_sizes() {
    std::cout << "\n\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TINY SIZES: Testing element count = 1, 4, 8, 16, 64, 100     ║\n";
    std::cout << "║  Shows overhead regime where SIMD setup cost dominates         ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════╣\n";
    
    int sizes[] = {1, 4, 8, 16, 32, 64, 100};
    float N = 7.0f;
    
    for (int sz : sizes) {
        std::vector<float> data(sz);
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto& v : data) v = dist(rng);
        
        std::vector<float> out_div(sz), out_recip(sz), out_recip64(sz);
        
        constexpr int ITERS = 100000;
        Timer t;
        
        t.begin();
        for (int i = 0; i < ITERS; ++i) {
            mean_division_f32(data.data(), out_div.data(), sz, N);
            asm volatile("" ::: "memory");
        }
        double time_div = t.elapsed_us() / ITERS;
        
        t.begin();
        for (int i = 0; i < ITERS; ++i) {
            mean_reciprocal_f32(data.data(), out_recip.data(), sz, N);
            asm volatile("" ::: "memory");
        }
        double time_recip = t.elapsed_us() / ITERS;
        
        t.begin();
        for (int i = 0; i < ITERS; ++i) {
            mean_reciprocal_f32_via_f64(data.data(), out_recip64.data(), sz, N);
            asm volatile("" ::: "memory");
        }
        double time_recip64 = t.elapsed_us() / ITERS;
        
        std::cout << "║  N=" << std::left << std::setw(6) << sz
                  << " Div: " << std::right << std::setw(6) << std::fixed << std::setprecision(3) << time_div << "μs"
                  << "  Recip: " << std::setw(6) << time_recip << "μs (" << std::setprecision(1) << time_div/time_recip << "x)"
                  << "  R-f64: " << std::setw(6) << std::setprecision(3) << time_recip64 << "μs (" << std::setprecision(1) << time_div/time_recip64 << "x)"
                  << " ║\n";
    }
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

int main() {
    std::cout << "═══════════════════════════════════════════════════════════════════\n";
    std::cout << "  Division vs Reciprocal-Multiply: Comprehensive Benchmark\n";
    std::cout << "  Compiler: " << __VERSION__ << "\n";
    std::cout << "  SIMD: AVX2 (256-bit)  |  Flags: -O3 -mavx2 -mfma\n";
    std::cout << "  14 datasets × {F32, F64} + divisor sweep + tiny sizes\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n";
    
    auto datasets = generate_all_datasets();
    
    // ── Section 1: All datasets, Float32 ──
    std::cout << "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "  SECTION 1: FLOAT32 — All 14 Datasets\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    
    for (const auto& ds : datasets) {
        run_f32_dataset(ds);
    }
    
    // ── Section 2: All datasets, Float64 ──
    std::cout << "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "  SECTION 2: FLOAT64 — All 14 Datasets\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    
    for (const auto& ds : datasets) {
        run_f64_dataset(ds);
    }
    
    // ── Section 3: Divisor sweep ──
    std::cout << "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "  SECTION 3: DIVISOR SWEEP (N = 2 to 100000)\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    run_divisor_sweep();
    
    // ── Section 4: Tiny sizes ──
    std::cout << "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "  SECTION 4: TINY ELEMENT COUNTS (overhead regime)\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    run_tiny_sizes();
    
    // ── Final summary ──
    std::cout << "\n\n═══════════════════════════════════════════════════════════════════\n";
    std::cout << "  LEGEND:\n";
    std::cout << "  ULP = Units in Last Place (1 ULP = smallest possible float step)\n";
    std::cout << "  0 ULP = bit-exact match with long double ground truth\n";
    std::cout << "  1 ULP = off by 1 mantissa bit (negligible for ML)\n";
    std::cout << "  >2 ULP = potentially concerning precision loss\n";
    std::cout << "  Strategy A = Direct SIMD division (PyTorch approach)\n";
    std::cout << "  Strategy B = Scalar 1/N then SIMD multiply (reciprocal)\n";
    std::cout << "  Strategy C = Double-precision 1/N, multiply in f64, cast back\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n";
    
    return 0;
}
