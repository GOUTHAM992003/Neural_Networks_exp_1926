// ═══════════════════════════════════════════════════════════════════════════
// BENCHMARK: Int64 → Double conversion + division
//
// Question: For int64_t sums, should we:
//   A) Scalar: for each element, cast int64→double, multiply by inv (scalar)
//   B) Manual SIMD: emulate _mm256_cvtepi64_pd via bit manipulation + vdivpd
//   C) Scalar division: for each element, cast int64→double, divide by N
//
// Also tests:
//   D) Float32 accumulator: Reciprocal-f64 vs SIMD division for mean
//   E) Double accumulator: Direct divide in double vs cast-down-then-divide
//   F) Fused nanmean: nansum + count in one pass, then SIMD divide
// ═══════════════════════════════════════════════════════════════════════════

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cstring>
#include <immintrin.h>

struct Timer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point start;
    void begin() { start = clock::now(); }
    double elapsed_us() const {
        return std::chrono::duration<double, std::micro>(clock::now() - start).count();
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// TEST 1: Int64 → Double mean division strategies
// ═══════════════════════════════════════════════════════════════════════════

// Strategy A: Scalar cast + reciprocal multiply
void int64_mean_scalar_recip(const int64_t* sums, double* out, int64_t count, int64_t N) {
    double inv = 1.0 / static_cast<double>(N);
    for (int64_t i = 0; i < count; ++i)
        out[i] = static_cast<double>(sums[i]) * inv;
}

// Strategy B: Scalar cast + scalar division
void int64_mean_scalar_div(const int64_t* sums, double* out, int64_t count, int64_t N) {
    double dN = static_cast<double>(N);
    for (int64_t i = 0; i < count; ++i)
        out[i] = static_cast<double>(sums[i]) / dN;
}

// Strategy C: Manual int64→double SIMD emulation + SIMD division
// AVX2 has NO _mm256_cvtepi64_pd, so we emulate it using the
// standard double-precision trick: add/subtract magic constant
inline __m256d manual_cvtepi64_pd(__m256i v) {
    // This trick works for int64 values in range [0, 2^52)
    // For signed: handle sign separately
    // Magic constant: 2^52 + 2^51 = 0x4338000000000000
    const __m256d magic = _mm256_set1_pd(6755399441055744.0); // 2^52 + 2^51
    // Reinterpret int64 bits as double, add magic, subtract magic
    // This ONLY works for unsigned values < 2^52
    // For general int64, we need a different approach
    
    // Simple approach: extract, convert, repack (SLOW but correct)
    alignas(32) int64_t vals[4];
    _mm256_store_si256((__m256i*)vals, v);
    return _mm256_set_pd(
        static_cast<double>(vals[3]),
        static_cast<double>(vals[2]),
        static_cast<double>(vals[1]),
        static_cast<double>(vals[0])
    );
}

void int64_mean_manual_simd(const int64_t* sums, double* out, int64_t count, int64_t N) {
    double dN = static_cast<double>(N);
    __m256d vN = _mm256_set1_pd(dN);
    int64_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m256i vi = _mm256_loadu_si256((const __m256i*)(sums + i));
        __m256d vd = manual_cvtepi64_pd(vi);
        __m256d vr = _mm256_div_pd(vd, vN);
        _mm256_storeu_pd(out + i, vr);
    }
    for (; i < count; ++i)
        out[i] = static_cast<double>(sums[i]) / dN;
}

void test_int64_mean() {
    std::cout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 1: Int64 → Double Mean Division Strategies               ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════╣\n";
    
    int64_t sizes[] = {100, 1000, 10000, 100000};
    int64_t N = 1000;
    
    for (int64_t sz : sizes) {
        std::mt19937_64 rng(42);
        std::uniform_int_distribution<int64_t> dist(-1000000, 1000000);
        std::vector<int64_t> sums(sz);
        for (auto& s : sums) s = dist(rng);
        
        std::vector<double> out_a(sz), out_b(sz), out_c(sz);
        
        // Warm up
        int64_mean_scalar_recip(sums.data(), out_a.data(), sz, N);
        int64_mean_scalar_div(sums.data(), out_b.data(), sz, N);
        int64_mean_manual_simd(sums.data(), out_c.data(), sz, N);
        
        constexpr int ITERS = 10000;
        Timer t;
        
        t.begin();
        for (int i = 0; i < ITERS; ++i) {
            int64_mean_scalar_recip(sums.data(), out_a.data(), sz, N);
            asm volatile("" ::: "memory");
        }
        double time_a = t.elapsed_us() / ITERS;
        
        t.begin();
        for (int i = 0; i < ITERS; ++i) {
            int64_mean_scalar_div(sums.data(), out_b.data(), sz, N);
            asm volatile("" ::: "memory");
        }
        double time_b = t.elapsed_us() / ITERS;
        
        t.begin();
        for (int i = 0; i < ITERS; ++i) {
            int64_mean_manual_simd(sums.data(), out_c.data(), sz, N);
            asm volatile("" ::: "memory");
        }
        double time_c = t.elapsed_us() / ITERS;
        
        // Precision check
        int mismatches_a = 0, mismatches_c = 0;
        for (int64_t i = 0; i < sz; ++i) {
            long double truth = static_cast<long double>(sums[i]) / static_cast<long double>(N);
            if (out_a[i] != static_cast<double>(truth)) ++mismatches_a;
            if (out_c[i] != static_cast<double>(truth)) ++mismatches_c;
        }
        
        std::cout << "║  " << std::left << std::setw(8) << sz << " elements:"
                  << "  Scalar-recip: " << std::right << std::setw(7) << std::fixed << std::setprecision(2) << time_a << "μs"
                  << "  Scalar-div: " << std::setw(7) << time_b << "μs"
                  << "  Manual-SIMD: " << std::setw(7) << time_c << "μs ║\n";
        std::cout << "║           "
                  << "  recip-err: " << mismatches_a << "/" << sz
                  << "                      "
                  << "  simd-err: " << mismatches_c << "/" << sz
                  << std::setw(10) << " ║\n";
    }
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 2: Float accumulators — divide in double vs cast-down-then-divide
//
// Flow A (current wasteful): double_acc → cast to float → store
//                            load float → cast to double → div → float
// Flow B (optimized):        double_acc → div in double → cast to float → store
// ═══════════════════════════════════════════════════════════════════════════

// Flow A: cast down then reciprocal  (simulates current approach)
void mean_castdown_then_recip(const double* acc, float* out, int64_t count, int64_t N) {
    // Step 1: cast double → float (simulating cascade_sum output)
    std::vector<float> temp(count);
    for (int64_t i = 0; i < count; ++i) temp[i] = static_cast<float>(acc[i]);
    
    // Step 2: reciprocal in double, multiply
    double inv = 1.0 / static_cast<double>(N);
    for (int64_t i = 0; i < count; ++i)
        out[i] = static_cast<float>(static_cast<double>(temp[i]) * inv);
}

// Flow B: divide in double directly, then cast to float
void mean_div_in_double(const double* acc, float* out, int64_t count, int64_t N) {
    double inv = 1.0 / static_cast<double>(N);
    __m256d vInv = _mm256_set1_pd(inv);
    int64_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m256d vacc = _mm256_loadu_pd(acc + i);
        __m256d vres = _mm256_mul_pd(vacc, vInv);
        __m128 vf = _mm256_cvtpd_ps(vres);
        _mm_storeu_ps(out + i, vf);
    }
    for (; i < count; ++i)
        out[i] = static_cast<float>(acc[i] * inv);
}

// Flow C: SIMD division in double, then cast (for comparison)
void mean_simd_div_in_double(const double* acc, float* out, int64_t count, int64_t N) {
    double dN = static_cast<double>(N);
    __m256d vN = _mm256_set1_pd(dN);
    int64_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m256d vacc = _mm256_loadu_pd(acc + i);
        __m256d vres = _mm256_div_pd(vacc, vN);
        __m128 vf = _mm256_cvtpd_ps(vres);
        _mm_storeu_ps(out + i, vf);
    }
    for (; i < count; ++i)
        out[i] = static_cast<float>(acc[i] / dN);
}

void test_accumulator_divide() {
    std::cout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 2: Double Accumulator Division Strategies                ║\n";
    std::cout << "║  Question: Divide in double BEFORE casting to float?           ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════╣\n";
    
    int64_t sizes[] = {100, 1000, 10000, 100000};
    int64_t N = 7;  // non-power-of-2 for worst case
    
    for (int64_t sz : sizes) {
        std::mt19937 rng(42);
        std::normal_distribution<double> dist(0.0, 1e6);
        std::vector<double> acc(sz);
        for (auto& a : acc) a = dist(rng);
        
        std::vector<float> out_a(sz), out_b(sz), out_c(sz);
        
        // Reference: long double
        std::vector<float> out_truth(sz);
        for (int64_t i = 0; i < sz; ++i)
            out_truth[i] = static_cast<float>(static_cast<long double>(acc[i]) / static_cast<long double>(N));
        
        // Warm up
        mean_castdown_then_recip(acc.data(), out_a.data(), sz, N);
        mean_div_in_double(acc.data(), out_b.data(), sz, N);
        mean_simd_div_in_double(acc.data(), out_c.data(), sz, N);
        
        constexpr int ITERS = 10000;
        Timer t;
        
        t.begin();
        for (int i = 0; i < ITERS; ++i) {
            mean_castdown_then_recip(acc.data(), out_a.data(), sz, N);
            asm volatile("" ::: "memory");
        }
        double time_a = t.elapsed_us() / ITERS;
        
        t.begin();
        for (int i = 0; i < ITERS; ++i) {
            mean_div_in_double(acc.data(), out_b.data(), sz, N);
            asm volatile("" ::: "memory");
        }
        double time_b = t.elapsed_us() / ITERS;
        
        t.begin();
        for (int i = 0; i < ITERS; ++i) {
            mean_simd_div_in_double(acc.data(), out_c.data(), sz, N);
            asm volatile("" ::: "memory");
        }
        double time_c = t.elapsed_us() / ITERS;
        
        // Precision
        int err_a = 0, err_b = 0, err_c = 0;
        for (int64_t i = 0; i < sz; ++i) {
            if (out_a[i] != out_truth[i]) ++err_a;
            if (out_b[i] != out_truth[i]) ++err_b;
            if (out_c[i] != out_truth[i]) ++err_c;
        }
        
        std::cout << "║  " << std::left << std::setw(8) << sz << "elems:"
                  << "  CastDown+Recip: " << std::right << std::setw(7) << std::fixed << std::setprecision(2) << time_a << "μs"
                  << "  DivInDouble(recip): " << std::setw(7) << time_b << "μs"
                  << "  DivInDouble(div): " << std::setw(7) << time_c << "μs ║\n";
        std::cout << "║           "
                  << "  err: " << err_a << "/" << sz
                  << "          err: " << err_b << "/" << sz
                  << "              err: " << err_c << "/" << sz
                  << std::setw(12) << " ║\n";
    }
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 3: Fused NanMean — single pass nansum + count, then SIMD divide
// Simulates inner loop: accumulate nansum and count simultaneously
// Then divide output[i] = nansum[i] / count[i] via SIMD
// ═══════════════════════════════════════════════════════════════════════════

// Two-pass approach (PyTorch style)
void nanmean_two_pass(const float* data, float* out, int64_t num_slices, int64_t slice_size) {
    for (int64_t o = 0; o < num_slices; ++o) {
        const float* slice = data + o * slice_size;
        double sum = 0.0;
        int64_t count = 0;
        // Pass 1: nansum
        for (int64_t j = 0; j < slice_size; ++j) {
            if (!std::isnan(slice[j])) sum += slice[j];
        }
        // Pass 2: count non-nan
        for (int64_t j = 0; j < slice_size; ++j) {
            if (!std::isnan(slice[j])) ++count;
        }
        out[o] = (count > 0) ? static_cast<float>(sum / static_cast<double>(count)) : std::nanf("");
    }
}

// Fused single-pass (our approach)
void nanmean_fused(const float* data, float* out, int64_t num_slices, int64_t slice_size) {
    for (int64_t o = 0; o < num_slices; ++o) {
        const float* slice = data + o * slice_size;
        double sum = 0.0;
        int64_t count = 0;
        // Single pass: both nansum and count
        for (int64_t j = 0; j < slice_size; ++j) {
            if (!std::isnan(slice[j])) {
                sum += static_cast<double>(slice[j]);
                ++count;
            }
        }
        out[o] = (count > 0) ? static_cast<float>(sum / static_cast<double>(count)) : std::nanf("");
    }
}

// Fused single-pass + SIMD final division
void nanmean_fused_simd_div(const float* data, double* sums, int64_t* counts, float* out,
                             int64_t num_slices, int64_t slice_size) {
    // Phase 1: fused single-pass accumulation
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
        sums[o] = sum;
        counts[o] = count;
    }
    
    // Phase 2: SIMD division (sums_double / counts_double → float output)
    int64_t o = 0;
    for (; o + 4 <= num_slices; o += 4) {
        __m256d vs = _mm256_loadu_pd(sums + o);
        // Convert int64 counts to double (manual since no _mm256_cvtepi64_pd in AVX2)
        __m256d vc = _mm256_set_pd(
            static_cast<double>(counts[o+3]),
            static_cast<double>(counts[o+2]),
            static_cast<double>(counts[o+1]),
            static_cast<double>(counts[o+0])
        );
        // Handle count==0 case: set count to 1 to avoid div-by-zero, result will be NaN
        __m256d zero = _mm256_setzero_pd();
        __m256d mask = _mm256_cmp_pd(vc, zero, _CMP_EQ_OQ);
        vc = _mm256_blendv_pd(vc, _mm256_set1_pd(1.0), mask);
        
        __m256d vr = _mm256_div_pd(vs, vc);
        // Set NaN where count was 0
        __m256d nan_val = _mm256_set1_pd(std::numeric_limits<double>::quiet_NaN());
        vr = _mm256_blendv_pd(vr, nan_val, mask);
        
        __m128 vf = _mm256_cvtpd_ps(vr);
        _mm_storeu_ps(out + o, vf);
    }
    for (; o < num_slices; ++o) {
        out[o] = (counts[o] > 0) ? static_cast<float>(sums[o] / static_cast<double>(counts[o])) : std::nanf("");
    }
}

void test_fused_nanmean() {
    std::cout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 3: Fused NanMean (single-pass vs two-pass + SIMD div)    ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════╣\n";
    
    int64_t slice_size = 1000;
    int64_t num_slices_list[] = {10, 100, 1000, 10000};
    float nan_pct = 0.1f;  // 10% NaN
    
    for (int64_t ns : num_slices_list) {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        std::vector<float> data(ns * slice_size);
        for (auto& v : data) {
            v = dist(rng);
            if (static_cast<float>(rng()) / rng.max() < nan_pct)
                v = std::nanf("");
        }
        
        std::vector<float> out_2pass(ns), out_fused(ns), out_fused_simd(ns);
        std::vector<double> sums_buf(ns);
        std::vector<int64_t> counts_buf(ns);
        
        // Warm up
        nanmean_two_pass(data.data(), out_2pass.data(), ns, slice_size);
        nanmean_fused(data.data(), out_fused.data(), ns, slice_size);
        nanmean_fused_simd_div(data.data(), sums_buf.data(), counts_buf.data(), out_fused_simd.data(), ns, slice_size);
        
        constexpr int ITERS = 1000;
        Timer t;
        
        t.begin();
        for (int i = 0; i < ITERS; ++i) {
            nanmean_two_pass(data.data(), out_2pass.data(), ns, slice_size);
            asm volatile("" ::: "memory");
        }
        double time_2p = t.elapsed_us() / ITERS;
        
        t.begin();
        for (int i = 0; i < ITERS; ++i) {
            nanmean_fused(data.data(), out_fused.data(), ns, slice_size);
            asm volatile("" ::: "memory");
        }
        double time_fused = t.elapsed_us() / ITERS;
        
        t.begin();
        for (int i = 0; i < ITERS; ++i) {
            nanmean_fused_simd_div(data.data(), sums_buf.data(), counts_buf.data(), out_fused_simd.data(), ns, slice_size);
            asm volatile("" ::: "memory");
        }
        double time_fused_simd = t.elapsed_us() / ITERS;
        
        // Check correctness
        int mismatch_fused = 0, mismatch_simd = 0;
        for (int64_t i = 0; i < ns; ++i) {
            if (std::isnan(out_2pass[i]) && std::isnan(out_fused[i])) continue;
            if (out_fused[i] != out_2pass[i]) ++mismatch_fused;
        }
        for (int64_t i = 0; i < ns; ++i) {
            if (std::isnan(out_2pass[i]) && std::isnan(out_fused_simd[i])) continue;
            if (out_fused_simd[i] != out_2pass[i]) ++mismatch_simd;
        }
        
        std::cout << "║  " << std::left << std::setw(6) << ns << "slices × " << slice_size << ":\n";
        std::cout << "║    Two-pass:            " << std::right << std::setw(10) << std::fixed << std::setprecision(1) << time_2p << " μs\n";
        std::cout << "║    Fused single-pass:   " << std::setw(10) << time_fused << " μs  (" 
                  << std::setprecision(1) << time_2p/time_fused << "x)  mismatches: " << mismatch_fused << "\n";
        std::cout << "║    Fused + SIMD div:    " << std::setw(10) << time_fused_simd << " μs  (" 
                  << time_2p/time_fused_simd << "x)  mismatches: " << mismatch_simd << "\n";
    }
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 4: Float32 mean — Reciprocal from double acc vs SIMD div from double acc
// This tests the user's key idea: skip the double→float→double round-trip
// ═══════════════════════════════════════════════════════════════════════════

void test_skip_roundtrip() {
    std::cout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 4: Skip the double→float→double round-trip              ║\n";
    std::cout << "║  Divide directly in double accumulator, then cast to float     ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════╣\n";
    
    int64_t sz = 100000;
    int64_t N_vals[] = {3, 7, 13, 127, 1000, 100000};
    
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(100.0, 50.0);
    std::vector<double> acc(sz);
    for (auto& a : acc) a = dist(rng);
    
    for (int64_t N : N_vals) {
        // Ground truth: long double
        std::vector<float> truth(sz);
        for (int64_t i = 0; i < sz; ++i)
            truth[i] = static_cast<float>(static_cast<long double>(acc[i]) / static_cast<long double>(N));
        
        // Path A: double acc → cast float → cast double → reciprocal multiply → float
        std::vector<float> out_a(sz);
        double inv = 1.0 / static_cast<double>(N);
        for (int64_t i = 0; i < sz; ++i) {
            float temp = static_cast<float>(acc[i]);  // LOSSY cast down
            out_a[i] = static_cast<float>(static_cast<double>(temp) * inv);  // cast back up
        }
        
        // Path B: double acc → reciprocal multiply in double → float (NO round-trip!)
        std::vector<float> out_b(sz);
        for (int64_t i = 0; i < sz; ++i) {
            out_b[i] = static_cast<float>(acc[i] * inv);  // direct in double, cast ONCE
        }
        
        // Path C: double acc → SIMD division in double → float
        std::vector<float> out_c(sz);
        double dN = static_cast<double>(N);
        for (int64_t i = 0; i < sz; ++i) {
            out_c[i] = static_cast<float>(acc[i] / dN);
        }
        
        int err_a = 0, err_b = 0, err_c = 0;
        for (int64_t i = 0; i < sz; ++i) {
            if (out_a[i] != truth[i]) ++err_a;
            if (out_b[i] != truth[i]) ++err_b;
            if (out_c[i] != truth[i]) ++err_c;
        }
        
        std::cout << "║  N=" << std::left << std::setw(8) << N
                  << "  RoundTrip: " << err_a << "/" << sz
                  << "  DirectRecip: " << err_b << "/" << sz
                  << "  DirectDiv: " << err_c << "/" << sz << "\n";
    }
    
    std::cout << "║\n";
    std::cout << "║  RoundTrip  = double→float→double→mul→float (current wasteful)\n";
    std::cout << "║  DirectRecip = double→mul→float (skip round-trip, reciprocal)\n";
    std::cout << "║  DirectDiv  = double→div→float (skip round-trip, division)\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 5: Float16/BFloat16 simulation
// Accumulator is float32, need to divide — what precision do we get?
// ═══════════════════════════════════════════════════════════════════════════

void test_f16_bf16_mean() {
    std::cout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 5: FP16/BF16 Mean — Accumulator is float32               ║\n";
    std::cout << "║  Should we divide in f32 or promote to f64 for division?       ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════╣\n";
    
    int64_t sz = 100000;
    int64_t N = 7;
    
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> acc(sz);  // float32 accumulator (for fp16/bf16 inputs)
    for (auto& a : acc) a = dist(rng);
    
    // Path A: divide in float32 directly (PyTorch CPU approach for bf16/fp16)
    std::vector<float> out_a(sz);
    float inv_f = 1.0f / static_cast<float>(N);
    for (int64_t i = 0; i < sz; ++i)
        out_a[i] = acc[i] * inv_f;
    
    // Path B: promote to double, divide, cast back (our proposed better approach)
    std::vector<float> out_b(sz);
    double inv_d = 1.0 / static_cast<double>(N);
    for (int64_t i = 0; i < sz; ++i)
        out_b[i] = static_cast<float>(static_cast<double>(acc[i]) * inv_d);
    
    // Path C: divide in float32 (direct div, not reciprocal)
    std::vector<float> out_c(sz);
    float fN = static_cast<float>(N);
    for (int64_t i = 0; i < sz; ++i)
        out_c[i] = acc[i] / fN;
    
    // Ground truth: long double
    int err_a = 0, err_b = 0, err_c = 0;
    for (int64_t i = 0; i < sz; ++i) {
        float truth = static_cast<float>(static_cast<long double>(acc[i]) / static_cast<long double>(N));
        if (out_a[i] != truth) ++err_a;
        if (out_b[i] != truth) ++err_b;
        if (out_c[i] != truth) ++err_c;
    }
    
    std::cout << "║  Accumulator: float32, Divisor N=7\n";
    std::cout << "║  A. Reciprocal in f32:    mismatches = " << err_a << "/" << sz << "\n";
    std::cout << "║  B. Reciprocal via f64:   mismatches = " << err_b << "/" << sz << "\n";
    std::cout << "║  C. Division in f32:      mismatches = " << err_c << "/" << sz << "\n";
    
    // Speed comparison
    constexpr int ITERS = 10000;
    Timer t;
    
    t.begin();
    for (int i = 0; i < ITERS; ++i) {
        for (int64_t j = 0; j < sz; ++j) out_a[j] = acc[j] * inv_f;
        asm volatile("" ::: "memory");
    }
    double time_a = t.elapsed_us() / ITERS;
    
    t.begin();
    for (int i = 0; i < ITERS; ++i) {
        for (int64_t j = 0; j < sz; ++j) out_b[j] = static_cast<float>(static_cast<double>(acc[j]) * inv_d);
        asm volatile("" ::: "memory");
    }
    double time_b = t.elapsed_us() / ITERS;
    
    t.begin();
    for (int i = 0; i < ITERS; ++i) {
        for (int64_t j = 0; j < sz; ++j) out_c[j] = acc[j] / fN;
        asm volatile("" ::: "memory");
    }
    double time_c = t.elapsed_us() / ITERS;
    
    std::cout << "║  A. Reciprocal f32:      " << std::fixed << std::setprecision(1) << time_a << " μs\n";
    std::cout << "║  B. Reciprocal via f64:  " << time_b << " μs  (overhead: " << std::setprecision(0) << (time_b/time_a - 1)*100 << "%)\n";
    std::cout << "║  C. Division f32:        " << std::setprecision(1) << time_c << " μs\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";
}

int main() {
    std::cout << "═══════════════════════════════════════════════════════════════════\n";
    std::cout << "  Mean/NanMean Implementation Decision Benchmark\n";
    std::cout << "  Testing: int64 paths, accumulator strategies, fused nanmean\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n";
    
    test_int64_mean();
    test_accumulator_divide();
    test_fused_nanmean();
    test_skip_roundtrip();
    test_f16_bf16_mean();
    
    std::cout << "\n═══════════════════════════════════════════════════════════════════\n";
    std::cout << "  DONE\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n";
    
    return 0;
}
