// ═══════════════════════════════════════════════════════════════════════════
// BENCHMARK: NanMean Division — The REAL test
//
// Previous test was unfair: stored count as int64_t, causing slow extract.
// This test stores count as DOUBLE, so SIMD division works natively.
//
// Also tests: varying output sizes to see WHERE SIMD division matters.
// ═══════════════════════════════════════════════════════════════════════════

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <limits>

struct Timer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point start;
    void begin() { start = clock::now(); }
    double elapsed_us() const {
        return std::chrono::duration<double, std::micro>(clock::now() - start).count();
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// TEST A: Fused single-pass + SCALAR division (count as int64)
// ═══════════════════════════════════════════════════════════════════════════
void nanmean_fused_scalar(const float* data, float* out,
                           int64_t num_slices, int64_t slice_size) {
    for (int64_t o = 0; o < num_slices; ++o) {
        const float* slice = data + o * slice_size;
        double sum = 0.0;
        int64_t count = 0;
        for (int64_t j = 0; j < slice_size; ++j) {
            if (!std::isnan(slice[j])) {
                sum += static_cast<double>(slice[j]);
                ++count;
            }
        }
        out[o] = (count > 0) ? static_cast<float>(sum / static_cast<double>(count)) : std::nanf("");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST B: Fused single-pass with DOUBLE count + SIMD division
// Key insight: accumulate count as double (count += 1.0) instead of int64
// ═══════════════════════════════════════════════════════════════════════════
void nanmean_fused_simd_double_count(const float* data, float* out,
                                      int64_t num_slices, int64_t slice_size) {
    // Phase 1: Accumulate nansum AND count (both as double)
    // We use thread-stack arrays for locality
    std::vector<double> sums(num_slices);
    std::vector<double> counts(num_slices);  // <-- DOUBLE, not int64!
    
    for (int64_t o = 0; o < num_slices; ++o) {
        const float* slice = data + o * slice_size;
        double sum = 0.0;
        double count = 0.0;  // <-- accumulate as double!
        for (int64_t j = 0; j < slice_size; ++j) {
            if (!std::isnan(slice[j])) {
                sum += static_cast<double>(slice[j]);
                count += 1.0;  // <-- double increment
            }
        }
        sums[o] = sum;
        counts[o] = count;
    }
    
    // Phase 2: SIMD division (both operands are double — perfect!)
    __m256d nan_val = _mm256_set1_pd(std::numeric_limits<double>::quiet_NaN());
    __m256d zero = _mm256_setzero_pd();
    int64_t o = 0;
    for (; o + 4 <= num_slices; o += 4) {
        __m256d vs = _mm256_loadu_pd(sums.data() + o);
        __m256d vc = _mm256_loadu_pd(counts.data() + o);
        
        // Check for count == 0 (all NaN slices)
        __m256d mask = _mm256_cmp_pd(vc, zero, _CMP_EQ_OQ);
        vc = _mm256_blendv_pd(vc, _mm256_set1_pd(1.0), mask);
        
        __m256d vr = _mm256_div_pd(vs, vc);
        vr = _mm256_blendv_pd(vr, nan_val, mask);
        
        // Convert double → float (4 doubles → 4 floats)
        __m128 vf = _mm256_cvtpd_ps(vr);
        _mm_storeu_ps(out + o, vf);
    }
    for (; o < num_slices; ++o) {
        out[o] = (counts[o] > 0) ? static_cast<float>(sums[o] / counts[o]) : std::nanf("");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST C: Two separate accum arrays + SIMD (simulates multi-threaded scenario)
// Division on PRE-COMPUTED arrays only (isolates SIMD division cost)
// ═══════════════════════════════════════════════════════════════════════════
void division_only_scalar(const double* sums, const double* counts, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = (counts[i] > 0) ? static_cast<float>(sums[i] / counts[i]) : std::nanf("");
    }
}

void division_only_simd(const double* sums, const double* counts, float* out, int64_t n) {
    __m256d nan_val = _mm256_set1_pd(std::numeric_limits<double>::quiet_NaN());
    __m256d zero = _mm256_setzero_pd();
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d vs = _mm256_loadu_pd(sums + i);
        __m256d vc = _mm256_loadu_pd(counts + i);
        __m256d mask = _mm256_cmp_pd(vc, zero, _CMP_EQ_OQ);
        vc = _mm256_blendv_pd(vc, _mm256_set1_pd(1.0), mask);
        __m256d vr = _mm256_div_pd(vs, vc);
        vr = _mm256_blendv_pd(vr, nan_val, mask);
        _mm_storeu_ps(out + i, _mm256_cvtpd_ps(vr));
    }
    for (; i < n; ++i) {
        out[i] = (counts[i] > 0) ? static_cast<float>(sums[i] / counts[i]) : std::nanf("");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST D: Regular MEAN — Reciprocal vs SIMD division from double acc
// For regular mean, denominator is CONSTANT = N
// ═══════════════════════════════════════════════════════════════════════════
void mean_from_double_acc_reciprocal(const double* acc, float* out, int64_t count, double N) {
    double inv = 1.0 / N;
    __m256d vInv = _mm256_set1_pd(inv);
    int64_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m256d va = _mm256_loadu_pd(acc + i);
        __m256d vr = _mm256_mul_pd(va, vInv);
        _mm_storeu_ps(out + i, _mm256_cvtpd_ps(vr));
    }
    for (; i < count; ++i)
        out[i] = static_cast<float>(acc[i] * inv);
}

void mean_from_double_acc_simd_div(const double* acc, float* out, int64_t count, double N) {
    __m256d vN = _mm256_set1_pd(N);
    int64_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m256d va = _mm256_loadu_pd(acc + i);
        __m256d vr = _mm256_div_pd(va, vN);
        _mm_storeu_ps(out + i, _mm256_cvtpd_ps(vr));
    }
    for (; i < count; ++i)
        out[i] = static_cast<float>(acc[i] / N);
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

int main() {
    std::cout << "═══════════════════════════════════════════════════════════════════\n";
    std::cout << "  NanMean SIMD Division — Fair Comparison (double count)\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n";
    
    // ── TEST 1: Full fused nanmean — scalar vs SIMD-with-double-count ──
    std::cout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 1: Fused NanMean — Scalar vs SIMD (double count)         ║\n";
    std::cout << "║  10% NaN rate, varying slice sizes and output counts            ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════╣\n";
    
    struct Config {
        int64_t slices;
        int64_t slice_size;
        const char* label;
    };
    
    Config configs[] = {
        {10,    10000,  "10 slices × 10K (few outputs, big reduction)"},
        {100,   1000,   "100 slices × 1K (balanced)"},
        {1000,  100,    "1K slices × 100 (many outputs, small reduction)"},
        {10000, 10,     "10K slices × 10 (lots of outputs, tiny reduction)"},
        {100000, 10,    "100K slices × 10 (massive outputs, tiny reduction)"},
        {10000, 100,    "10K slices × 100 (large both)"},
        {1000,  1000,   "1K slices × 1K (big balanced)"},
        {100,   10000,  "100 slices × 10K (classic batch reduction)"},
    };
    
    for (const auto& cfg : configs) {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        int64_t total = cfg.slices * cfg.slice_size;
        
        std::vector<float> data(total);
        for (auto& v : data) {
            v = dist(rng);
            if ((rng() % 10) == 0) v = std::nanf("");
        }
        
        std::vector<float> out_scalar(cfg.slices), out_simd(cfg.slices);
        
        // Warm up
        nanmean_fused_scalar(data.data(), out_scalar.data(), cfg.slices, cfg.slice_size);
        nanmean_fused_simd_double_count(data.data(), out_simd.data(), cfg.slices, cfg.slice_size);
        
        // Check correctness
        int mismatches = 0;
        for (int64_t i = 0; i < cfg.slices; ++i) {
            if (std::isnan(out_scalar[i]) && std::isnan(out_simd[i])) continue;
            if (out_scalar[i] != out_simd[i]) ++mismatches;
        }
        
        int ITERS = std::max(100, static_cast<int>(100000000 / total));
        Timer t;
        
        t.begin();
        for (int i = 0; i < ITERS; ++i) {
            nanmean_fused_scalar(data.data(), out_scalar.data(), cfg.slices, cfg.slice_size);
            asm volatile("" ::: "memory");
        }
        double time_scalar = t.elapsed_us() / ITERS;
        
        t.begin();
        for (int i = 0; i < ITERS; ++i) {
            nanmean_fused_simd_double_count(data.data(), out_simd.data(), cfg.slices, cfg.slice_size);
            asm volatile("" ::: "memory");
        }
        double time_simd = t.elapsed_us() / ITERS;
        
        std::cout << "║  " << std::left << std::setw(55) << cfg.label << "║\n";
        std::cout << "║    Scalar:  " << std::right << std::setw(10) << std::fixed << std::setprecision(1) 
                  << time_scalar << " μs"
                  << "    SIMD: " << std::setw(10) << time_simd << " μs"
                  << "  (" << std::setprecision(2) << time_scalar/time_simd << "x)"
                  << "  err=" << mismatches
                  << std::setw(8) << " ║\n";
    }
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";
    
    // ── TEST 2: Division-only — isolate the SIMD division benefit ──
    std::cout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 2: DIVISION ONLY — Isolate SIMD division speed           ║\n";
    std::cout << "║  Pre-computed sums[] and counts[] (both double), divide only    ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════╣\n";
    
    int64_t div_sizes[] = {100, 1000, 10000, 100000, 1000000};
    
    for (int64_t sz : div_sizes) {
        std::mt19937 rng(42);
        std::normal_distribution<double> sdist(0.0, 100.0);
        std::uniform_int_distribution<int64_t> cdist(1, 1000);
        
        std::vector<double> sums(sz), counts(sz);
        for (int64_t i = 0; i < sz; ++i) {
            sums[i] = sdist(rng);
            counts[i] = static_cast<double>(cdist(rng));
        }
        
        std::vector<float> out_s(sz), out_v(sz);
        
        int ITERS = std::max(1000, static_cast<int>(100000000 / sz));
        Timer t;
        
        t.begin();
        for (int i = 0; i < ITERS; ++i) {
            division_only_scalar(sums.data(), counts.data(), out_s.data(), sz);
            asm volatile("" ::: "memory");
        }
        double time_s = t.elapsed_us() / ITERS;
        
        t.begin();
        for (int i = 0; i < ITERS; ++i) {
            division_only_simd(sums.data(), counts.data(), out_v.data(), sz);
            asm volatile("" ::: "memory");
        }
        double time_v = t.elapsed_us() / ITERS;
        
        int err = 0;
        for (int64_t i = 0; i < sz; ++i) {
            if (out_s[i] != out_v[i]) ++err;
        }
        
        std::cout << "║  " << std::left << std::setw(10) << sz << " elems: "
                  << "Scalar: " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << time_s << "μs"
                  << "  SIMD: " << std::setw(10) << time_v << "μs"
                  << "  (" << std::setprecision(1) << time_s/time_v << "x)"
                  << "  err=" << err
                  << std::setw(8) << " ║\n";
    }
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";
    
    // ── TEST 3: Regular MEAN — Reciprocal vs SIMD division ──
    std::cout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 3: Regular MEAN — Reciprocal-mul vs SIMD-div             ║\n";
    std::cout << "║  Both from double accumulator, constant denominator N           ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════╣\n";
    
    for (int64_t sz : div_sizes) {
        std::mt19937 rng(42);
        std::normal_distribution<double> ddist(0.0, 100.0);
        std::vector<double> acc(sz);
        for (auto& a : acc) a = ddist(rng);
        
        std::vector<float> out_r(sz), out_d(sz);
        double N = 7.0;
        
        int ITERS = std::max(1000, static_cast<int>(100000000 / sz));
        Timer t;
        
        t.begin();
        for (int i = 0; i < ITERS; ++i) {
            mean_from_double_acc_reciprocal(acc.data(), out_r.data(), sz, N);
            asm volatile("" ::: "memory");
        }
        double time_r = t.elapsed_us() / ITERS;
        
        t.begin();
        for (int i = 0; i < ITERS; ++i) {
            mean_from_double_acc_simd_div(acc.data(), out_d.data(), sz, N);
            asm volatile("" ::: "memory");
        }
        double time_d = t.elapsed_us() / ITERS;
        
        // Precision
        int err_r = 0, err_d = 0;
        for (int64_t i = 0; i < sz; ++i) {
            float truth = static_cast<float>(static_cast<long double>(acc[i]) / 7.0L);
            if (out_r[i] != truth) ++err_r;
            if (out_d[i] != truth) ++err_d;
        }
        
        std::cout << "║  " << std::left << std::setw(10) << sz << " elems: "
                  << "Recip: " << std::right << std::setw(8) << std::fixed << std::setprecision(2) << time_r << "μs (err=" << err_r << ")"
                  << "  Div: " << std::setw(8) << time_d << "μs (err=" << err_d << ")"
                  << "  recip " << std::setprecision(1) << time_d/time_r << "x faster"
                  << std::setw(4) << " ║\n";
    }
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";

    // ── Summary ──
    std::cout << "\n═══════════════════════════════════════════════════════════════════\n";
    std::cout << "  KEY INSIGHT:\n";
    std::cout << "  - For nanmean: count stored as DOUBLE (not int64) enables SIMD\n";
    std::cout << "  - Fused pass: sum += val and count += 1.0 (both double)\n";
    std::cout << "  - Division: _mm256_div_pd(sums, counts) → pure SIMD\n";
    std::cout << "  - For mean: reciprocal _mm256_mul_pd faster than _mm256_div_pd\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n";
    
    return 0;
}
