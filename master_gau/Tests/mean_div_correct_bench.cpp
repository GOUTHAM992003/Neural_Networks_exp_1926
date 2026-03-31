/**
 * CORRECTED benchmark: The user's approach
 *
 * Strategy A: PyTorch — fp32 SIMD division (_mm256_div_ps)
 * Strategy B: fp32 reciprocal — recip in fp32, mul in fp32
 * Strategy E: USER'S APPROACH — cast sum fp32→fp64, recip in fp64, mul in fp64, cast result→fp32
 *             This keeps FULL fp64 precision during the multiply!
 * Strategy D: Full fp64 division — cast fp32→fp64, div in fp64, cast→fp32
 */
#include <immintrin.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
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

int ulp_diff(float a, float b) {
    if (a == b) return 0;
    int32_t ai, bi;
    memcpy(&ai, &a, 4); memcpy(&bi, &b, 4);
    if (ai < 0) ai = 0x80000000 - ai;
    if (bi < 0) bi = 0x80000000 - bi;
    return std::abs(ai - bi);
}

int main() {
    std::cout << "================================================================\n";
    std::cout << "CORRECTED: fp64 reciprocal-multiply (keeping fp64 during multiply)\n";
    std::cout << "================================================================\n\n";

    int64_t N = 100000;
    std::vector<float> data(N);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    for (auto& v : data) v = dist(gen);

    std::vector<float> out_a(N), out_b(N), out_e(N), out_d(N);

    std::cout << "SPEED (100K float32 elements):\n";
    std::cout << std::string(90, '-') << "\n";
    std::cout << std::setw(10) << "Divisor" << " | "
              << std::setw(14) << "A:fp32 div" << " | "
              << std::setw(14) << "B:fp32 recip" << " | "
              << std::setw(18) << "E:fp64 recip-mul" << " | "
              << std::setw(14) << "D:fp64 div" << "\n";
    std::cout << std::string(90, '-') << "\n";

    for (float divisor : {3.0f, 7.0f, 100.0f, 1000.0f, 99999.0f, 1e7f}) {
        // A: PyTorch fp32 SIMD division
        double t_a = bench([&]() {
            __m256 vdiv = _mm256_set1_ps(divisor);
            int64_t i = 0;
            for (; i + 8 <= N; i += 8)
                _mm256_storeu_ps(&out_a[i], _mm256_div_ps(_mm256_loadu_ps(&data[i]), vdiv));
            for (; i < N; ++i) out_a[i] = data[i] / divisor;
        });

        // B: fp32 reciprocal multiply (everything in fp32)
        double t_b = bench([&]() {
            float recip = 1.0f / divisor;
            __m256 vrecip = _mm256_set1_ps(recip);
            int64_t i = 0;
            for (; i + 8 <= N; i += 8)
                _mm256_storeu_ps(&out_b[i], _mm256_mul_ps(_mm256_loadu_ps(&data[i]), vrecip));
            for (; i < N; ++i) out_b[i] = data[i] * recip;
        });

        // E: USER'S APPROACH — cast fp32→fp64, multiply in fp64, cast back to fp32
        // Uses SIMD: load 4 floats → cvtps_pd → 4 doubles → mul → cvtpd_ps → store 4 floats
        double t_e = bench([&]() {
            double recip = 1.0 / static_cast<double>(divisor);  // fp64 reciprocal
            __m256d vrecip = _mm256_set1_pd(recip);
            int64_t i = 0;
            for (; i + 4 <= N; i += 4) {
                __m128 f4 = _mm_loadu_ps(&data[i]);           // Load 4 fp32
                __m256d d4 = _mm256_cvtps_pd(f4);              // Convert to 4 fp64
                d4 = _mm256_mul_pd(d4, vrecip);                // Multiply in fp64!
                _mm_storeu_ps(&out_e[i], _mm256_cvtpd_ps(d4)); // Convert back to fp32
            }
            for (; i < N; ++i)
                out_e[i] = static_cast<float>(static_cast<double>(data[i]) * recip);
        });

        // D: Full fp64 division
        double t_d = bench([&]() {
            __m256d vdiv = _mm256_set1_pd(static_cast<double>(divisor));
            int64_t i = 0;
            for (; i + 4 <= N; i += 4) {
                __m256d d4 = _mm256_cvtps_pd(_mm_loadu_ps(&data[i]));
                _mm_storeu_ps(&out_d[i], _mm256_cvtpd_ps(_mm256_div_pd(d4, vdiv)));
            }
            for (; i < N; ++i)
                out_d[i] = static_cast<float>(static_cast<double>(data[i]) / static_cast<double>(divisor));
        });

        std::cout << std::setw(10) << std::fixed << std::setprecision(0) << divisor
                  << " | " << std::setw(11) << std::setprecision(1) << t_a << "μs"
                  << " | " << std::setw(11) << t_b << "μs"
                  << " | " << std::setw(15) << t_e << "μs"
                  << " | " << std::setw(11) << t_d << "μs\n";
    }

    // PRECISION
    std::cout << "\nPRECISION (max ULP + count of non-zero ULP errors):\n";
    std::cout << std::string(100, '-') << "\n";
    std::cout << std::setw(10) << "Divisor" << " | "
              << std::setw(18) << "A:fp32 div" << " | "
              << std::setw(18) << "B:fp32 recip" << " | "
              << std::setw(18) << "E:fp64 recip-mul" << " | "
              << std::setw(18) << "D:fp64 div" << "\n";
    std::cout << std::string(100, '-') << "\n";

    for (float divisor : {3.0f, 7.0f, 10.0f, 100.0f, 1000.0f, 99999.0f, 1e7f}) {
        // Compute each
        float recip_f32 = 1.0f / divisor;
        double recip_f64 = 1.0 / static_cast<double>(divisor);

        for (int64_t i = 0; i < N; ++i) {
            out_a[i] = data[i] / divisor;
            out_b[i] = data[i] * recip_f32;
            out_e[i] = static_cast<float>(static_cast<double>(data[i]) * recip_f64);
            out_d[i] = static_cast<float>(static_cast<double>(data[i]) / static_cast<double>(divisor));
        }

        // Ground truth: full fp64
        int max_a=0, max_b=0, max_e=0, max_d=0;
        int cnt_a=0, cnt_b=0, cnt_e=0, cnt_d=0;
        for (int64_t i = 0; i < N; ++i) {
            float gt = static_cast<float>(static_cast<double>(data[i]) / static_cast<double>(divisor));
            int ua = ulp_diff(out_a[i], gt); if (ua > max_a) max_a = ua; if (ua > 0) cnt_a++;
            int ub = ulp_diff(out_b[i], gt); if (ub > max_b) max_b = ub; if (ub > 0) cnt_b++;
            int ue = ulp_diff(out_e[i], gt); if (ue > max_e) max_e = ue; if (ue > 0) cnt_e++;
            int ud = ulp_diff(out_d[i], gt); if (ud > max_d) max_d = ud; if (ud > 0) cnt_d++;
        }

        std::cout << std::setw(10) << std::fixed << std::setprecision(0) << divisor
                  << " | " << std::setw(2) << max_a << " ULP (" << std::setw(5) << cnt_a << ")"
                  << " | " << std::setw(2) << max_b << " ULP (" << std::setw(5) << cnt_b << ")"
                  << " | " << std::setw(2) << max_e << " ULP (" << std::setw(5) << cnt_e << ")"
                  << " | " << std::setw(2) << max_d << " ULP (" << std::setw(5) << cnt_d << ")\n";
    }

    // Size scaling
    std::cout << "\nSIZE SCALING (divisor=1000):\n";
    std::cout << std::string(90, '-') << "\n";
    for (int64_t sz : {1000LL, 10000LL, 100000LL, 1000000LL, 10000000LL}) {
        std::vector<float> d(sz), o1(sz), o2(sz);
        for (auto& v : d) v = dist(gen);
        float div = 1000.0f;

        double t_b = bench([&]() {
            float r = 1.0f / div;
            __m256 vr = _mm256_set1_ps(r);
            int64_t i = 0;
            for (; i + 8 <= sz; i += 8) _mm256_storeu_ps(&o1[i], _mm256_mul_ps(_mm256_loadu_ps(&d[i]), vr));
            for (; i < sz; ++i) o1[i] = d[i] * r;
        });

        double t_e = bench([&]() {
            double r = 1.0 / (double)div;
            __m256d vr = _mm256_set1_pd(r);
            int64_t i = 0;
            for (; i + 4 <= sz; i += 4) {
                _mm_storeu_ps(&o2[i], _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(&d[i])), vr)));
            }
            for (; i < sz; ++i) o2[i] = (float)((double)d[i] * r);
        });

        std::cout << std::setw(10) << sz
                  << " | B:fp32 recip: " << std::setw(8) << std::setprecision(1) << t_b << "μs"
                  << " | E:fp64 recip-mul: " << std::setw(8) << t_e << "μs"
                  << " | E/B ratio: " << std::setprecision(2) << (t_e / t_b) << "x\n";
    }

    return 0;
}
