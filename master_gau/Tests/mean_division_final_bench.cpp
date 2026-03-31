/**
 * DEFINITIVE benchmark: Mean division strategies
 *
 * Strategy A: PyTorch-style — fp32 SIMD division (_mm256_div_ps)
 * Strategy B: Our fp32 reciprocal — 1/N in fp32, then _mm256_mul_ps
 * Strategy C: Our fp64 reciprocal — 1/N in fp64, cast to fp32, then _mm256_mul_ps
 * Strategy D: Full fp64 path — cast fp32→fp64, divide in fp64, cast back to fp32
 *
 * Tests: Speed, Precision (ULP), Edge cases (subnormals, large values, tiny divisors)
 */
#include <immintrin.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <cfloat>
#include <cstring>

double bench(auto fn, int warmup = 10, int iters = 200) {
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

// Count ULP difference between two floats
int ulp_diff(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) return -1;
    if (a == b) return 0;
    int32_t ai, bi;
    memcpy(&ai, &a, 4);
    memcpy(&bi, &b, 4);
    if (ai < 0) ai = 0x80000000 - ai;
    if (bi < 0) bi = 0x80000000 - bi;
    return std::abs(ai - bi);
}

struct PrecisionResult {
    int max_ulp;
    double max_rel_err;
    int64_t total_ulp;
    int count_nonzero_ulp;
};

PrecisionResult measure_precision(const float* result, const float* data, int64_t n, float divisor) {
    PrecisionResult pr{0, 0, 0, 0};
    for (int64_t i = 0; i < n; ++i) {
        // Ground truth: fp64 division
        double exact = static_cast<double>(data[i]) / static_cast<double>(divisor);
        float exact_f = static_cast<float>(exact);
        int ulp = ulp_diff(result[i], exact_f);
        if (ulp > pr.max_ulp) pr.max_ulp = ulp;
        pr.total_ulp += ulp;
        if (ulp > 0) pr.count_nonzero_ulp++;
        double rel = (exact != 0) ? std::abs((double)result[i] - exact) / std::abs(exact) : 0;
        if (rel > pr.max_rel_err) pr.max_rel_err = rel;
    }
    return pr;
}

int main() {
    std::cout << "================================================================\n";
    std::cout << "DEFINITIVE: Mean division strategies — Speed + Precision\n";
    std::cout << "================================================================\n\n";

    // ═══════════════════════════════════════════════════════════
    // PART 1: SPEED across different output sizes
    // ═══════════════════════════════════════════════════════════
    std::cout << "PART 1: SPEED (100K float32 elements, various divisors)\n";
    std::cout << std::string(95, '-') << "\n";
    std::cout << std::setw(12) << "Divisor" << " | "
              << std::setw(14) << "A:fp32 div" << " | "
              << std::setw(14) << "B:fp32 recip" << " | "
              << std::setw(14) << "C:fp64 recip" << " | "
              << std::setw(14) << "D:full fp64" << "\n";
    std::cout << std::string(95, '-') << "\n";

    int64_t N = 100000;
    std::vector<float> data(N);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    for (auto& v : data) v = dist(gen);

    std::vector<float> out_a(N), out_b(N), out_c(N);
    std::vector<double> out_d_dbl(N);
    std::vector<float> out_d(N);

    for (float divisor : {3.0f, 7.0f, 100.0f, 1000.0f, 10000.0f, 100000.0f, 1e7f}) {
        // A: PyTorch-style fp32 SIMD division
        double t_a = bench([&]() {
            __m256 vdiv = _mm256_set1_ps(divisor);
            int64_t i = 0;
            for (; i + 8 <= N; i += 8)
                _mm256_storeu_ps(&out_a[i], _mm256_div_ps(_mm256_loadu_ps(&data[i]), vdiv));
            for (; i < N; ++i) out_a[i] = data[i] / divisor;
        });

        // B: fp32 reciprocal multiply
        double t_b = bench([&]() {
            float recip = 1.0f / divisor;
            __m256 vrecip = _mm256_set1_ps(recip);
            int64_t i = 0;
            for (; i + 8 <= N; i += 8)
                _mm256_storeu_ps(&out_b[i], _mm256_mul_ps(_mm256_loadu_ps(&data[i]), vrecip));
            for (; i < N; ++i) out_b[i] = data[i] * recip;
        });

        // C: fp64 reciprocal cast to fp32, then multiply
        double t_c = bench([&]() {
            float recip = static_cast<float>(1.0 / static_cast<double>(divisor));
            __m256 vrecip = _mm256_set1_ps(recip);
            int64_t i = 0;
            for (; i + 8 <= N; i += 8)
                _mm256_storeu_ps(&out_c[i], _mm256_mul_ps(_mm256_loadu_ps(&data[i]), vrecip));
            for (; i < N; ++i) out_c[i] = data[i] * recip;
        });

        // D: Full fp64 — cast to double, divide, cast back
        double t_d = bench([&]() {
            __m256d vdiv = _mm256_set1_pd(static_cast<double>(divisor));
            int64_t i = 0;
            for (; i + 4 <= N; i += 4) {
                // Load 4 floats, convert to 4 doubles
                __m128 f4 = _mm_loadu_ps(&data[i]);
                __m256d d4 = _mm256_cvtps_pd(f4);
                d4 = _mm256_div_pd(d4, vdiv);
                // Convert back to float and store
                _mm_storeu_ps(&out_d[i], _mm256_cvtpd_ps(d4));
            }
            for (; i < N; ++i) out_d[i] = static_cast<float>(static_cast<double>(data[i]) / static_cast<double>(divisor));
        });

        std::cout << std::setw(12) << std::fixed << std::setprecision(0) << divisor
                  << " | " << std::setw(11) << std::setprecision(1) << t_a << "μs"
                  << " | " << std::setw(11) << t_b << "μs"
                  << " | " << std::setw(11) << t_c << "μs"
                  << " | " << std::setw(11) << t_d << "μs\n";
    }

    // ═══════════════════════════════════════════════════════════
    // PART 2: PRECISION for each strategy
    // ═══════════════════════════════════════════════════════════
    std::cout << "\nPART 2: PRECISION (max ULP error vs fp64 ground truth)\n";
    std::cout << std::string(100, '-') << "\n";
    std::cout << std::setw(12) << "Divisor" << " | "
              << std::setw(16) << "A:fp32 div ULP" << " | "
              << std::setw(16) << "B:fp32 recip" << " | "
              << std::setw(16) << "C:fp64 recip" << " | "
              << std::setw(16) << "D:full fp64" << "\n";
    std::cout << std::string(100, '-') << "\n";

    for (float divisor : {3.0f, 7.0f, 10.0f, 100.0f, 1000.0f, 10000.0f, 99999.0f, 1e7f}) {
        // Compute each strategy
        float recip_f32 = 1.0f / divisor;
        float recip_f64 = static_cast<float>(1.0 / static_cast<double>(divisor));
        __m256 vdiv = _mm256_set1_ps(divisor);
        __m256 vrecip32 = _mm256_set1_ps(recip_f32);
        __m256 vrecip64 = _mm256_set1_ps(recip_f64);
        __m256d vdiv_d = _mm256_set1_pd(static_cast<double>(divisor));

        for (int64_t i = 0; i + 8 <= N; i += 8) {
            __m256 v = _mm256_loadu_ps(&data[i]);
            _mm256_storeu_ps(&out_a[i], _mm256_div_ps(v, vdiv));
            _mm256_storeu_ps(&out_b[i], _mm256_mul_ps(v, vrecip32));
            _mm256_storeu_ps(&out_c[i], _mm256_mul_ps(v, vrecip64));
        }
        for (int64_t i = 0; i + 4 <= N; i += 4) {
            __m128 f4 = _mm_loadu_ps(&data[i]);
            _mm_storeu_ps(&out_d[i], _mm256_cvtpd_ps(_mm256_div_pd(_mm256_cvtps_pd(f4), vdiv_d)));
        }
        // Scalar tails
        for (int64_t i = (N/8)*8; i < N; ++i) {
            out_a[i] = data[i] / divisor;
            out_b[i] = data[i] * recip_f32;
            out_c[i] = data[i] * recip_f64;
            out_d[i] = static_cast<float>(static_cast<double>(data[i]) / static_cast<double>(divisor));
        }

        auto pa = measure_precision(out_a.data(), data.data(), N, divisor);
        auto pb = measure_precision(out_b.data(), data.data(), N, divisor);
        auto pc = measure_precision(out_c.data(), data.data(), N, divisor);
        auto pd = measure_precision(out_d.data(), data.data(), N, divisor);

        std::cout << std::setw(12) << std::fixed << std::setprecision(0) << divisor
                  << " | " << std::setw(8) << pa.max_ulp << " ULP"
                  << " (" << std::setw(3) << pa.count_nonzero_ulp << ")"
                  << " | " << std::setw(8) << pb.max_ulp << " ULP"
                  << " (" << std::setw(5) << pb.count_nonzero_ulp << ")"
                  << " | " << std::setw(8) << pc.max_ulp << " ULP"
                  << " (" << std::setw(5) << pc.count_nonzero_ulp << ")"
                  << " | " << std::setw(8) << pd.max_ulp << " ULP"
                  << " (" << std::setw(3) << pd.count_nonzero_ulp << ")\n";
    }

    // ═══════════════════════════════════════════════════════════
    // PART 3: EDGE CASES
    // ═══════════════════════════════════════════════════════════
    std::cout << "\nPART 3: EDGE CASES precision (max ULP)\n";
    std::cout << std::string(90, '-') << "\n";

    auto test_edge = [&](const char* name, std::vector<float>& edata, float div) {
        int64_t en = edata.size();
        std::vector<float> ea(en), eb(en), ec(en), ed(en);
        float r32 = 1.0f / div;
        float r64 = static_cast<float>(1.0 / static_cast<double>(div));
        for (int64_t i = 0; i < en; ++i) {
            ea[i] = edata[i] / div;
            eb[i] = edata[i] * r32;
            ec[i] = edata[i] * r64;
            ed[i] = static_cast<float>(static_cast<double>(edata[i]) / static_cast<double>(div));
        }
        auto pa = measure_precision(ea.data(), edata.data(), en, div);
        auto pb = measure_precision(eb.data(), edata.data(), en, div);
        auto pc = measure_precision(ec.data(), edata.data(), en, div);
        auto pd = measure_precision(ed.data(), edata.data(), en, div);
        std::cout << std::setw(25) << name
                  << " | A:" << std::setw(2) << pa.max_ulp
                  << " | B:" << std::setw(2) << pb.max_ulp
                  << " | C:" << std::setw(2) << pc.max_ulp
                  << " | D:" << std::setw(2) << pd.max_ulp << "\n";
    };

    // Subnormals
    std::vector<float> subnormals(1000);
    for (int i = 0; i < 1000; ++i) subnormals[i] = FLT_MIN * (float)(i+1) / 2000.0f;
    test_edge("Subnormals /3", subnormals, 3.0f);
    test_edge("Subnormals /7", subnormals, 7.0f);

    // Near FLT_MAX
    std::vector<float> big(1000);
    for (int i = 0; i < 1000; ++i) big[i] = FLT_MAX / (float)(i+2);
    test_edge("Near FLT_MAX /3", big, 3.0f);
    test_edge("Near FLT_MAX /1000", big, 1000.0f);

    // Very small divisor
    std::vector<float> normal(1000);
    std::uniform_real_distribution<float> nd(-100.0f, 100.0f);
    for (auto& v : normal) v = nd(gen);
    test_edge("Normal data /0.001", normal, 0.001f);
    test_edge("Normal data /1e-7", normal, 1e-7f);

    // Power-of-2 divisors (exact in float)
    test_edge("Normal /2", normal, 2.0f);
    test_edge("Normal /8", normal, 8.0f);
    test_edge("Normal /1024", normal, 1024.0f);

    // Typical DL batch sizes
    test_edge("Normal /32 (batch)", normal, 32.0f);
    test_edge("Normal /64 (batch)", normal, 64.0f);
    test_edge("Normal /256 (batch)", normal, 256.0f);
    test_edge("Normal /1000 (seq)", normal, 1000.0f);

    // ═══════════════════════════════════════════════════════════
    // PART 4: fp64 cast overhead measurement
    // ═══════════════════════════════════════════════════════════
    std::cout << "\nPART 4: fp64 CAST OVERHEAD (is promoting to fp64 expensive?)\n";
    std::cout << std::string(80, '-') << "\n";

    for (int64_t sz : {1000LL, 10000LL, 100000LL, 1000000LL}) {
        std::vector<float> d(sz);
        for (auto& v : d) v = dist(gen);
        std::vector<float> o1(sz), o2(sz);
        float div = 1000.0f;

        // fp32 recip only
        double t1 = bench([&]() {
            float recip = static_cast<float>(1.0 / static_cast<double>(div));
            __m256 vr = _mm256_set1_ps(recip);
            int64_t i = 0;
            for (; i + 8 <= sz; i += 8)
                _mm256_storeu_ps(&o1[i], _mm256_mul_ps(_mm256_loadu_ps(&d[i]), vr));
            for (; i < sz; ++i) o1[i] = d[i] * recip;
        });

        // full fp64 path (cast up, divide, cast down)
        double t2 = bench([&]() {
            __m256d vd = _mm256_set1_pd(static_cast<double>(div));
            int64_t i = 0;
            for (; i + 4 <= sz; i += 4) {
                __m256d dd = _mm256_cvtps_pd(_mm_loadu_ps(&d[i]));
                _mm_storeu_ps(&o2[i], _mm256_cvtpd_ps(_mm256_div_pd(dd, vd)));
            }
            for (; i < sz; ++i) o2[i] = static_cast<float>(static_cast<double>(d[i]) / static_cast<double>(div));
        });

        std::cout << std::setw(10) << sz
                  << " | fp64-recip-in-fp32: " << std::setw(8) << std::setprecision(1) << t1 << "μs"
                  << " | full-fp64-path: " << std::setw(8) << t2 << "μs"
                  << " | overhead: " << std::setprecision(2) << (t2/t1) << "x\n";
    }

    return 0;
}
