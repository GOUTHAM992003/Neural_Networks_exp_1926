/**
 * Benchmark: Division approaches for mean's output step
 *
 * Approach A: Scalar division (current)
 * Approach B: SIMD division (_mm256_div_ps)
 * Approach C: SIMD multiply-by-reciprocal (_mm256_mul_ps with 1/N)
 * Approach D: SIMD reciprocal approx (_mm256_rcp_ps + Newton-Raphson refinement)
 *
 * Tests: Speed, Precision, and edge cases
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

double bench(auto fn, int warmup = 5, int iters = 100) {
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
    std::cout << "=================================================================\n";
    std::cout << "BENCHMARK: Division approaches for mean output step\n";
    std::cout << "=================================================================\n\n";

    // ─── PART 1: SPEED COMPARISON ───
    std::cout << "PART 1: SPEED (dividing N float elements by a scalar)\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << std::setw(12) << "Size" << " | "
              << std::setw(12) << "Scalar" << " | "
              << std::setw(12) << "SIMD div" << " | "
              << std::setw(12) << "SIMD recip" << " | "
              << std::setw(12) << "SIMD rcp_ps" << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (int64_t size : {1000LL, 10000LL, 100000LL, 1000000LL, 10000000LL}) {
        std::vector<float> data(size);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(1.0f, 1000.0f);
        for (auto& v : data) v = dist(gen);

        float divisor = 12345.0f;
        std::vector<float> out_a(size), out_b(size), out_c(size), out_d(size);

        // A: Scalar division
        double t_scalar = bench([&]() {
            for (int64_t i = 0; i < size; ++i)
                out_a[i] = data[i] / divisor;
        });

        // B: SIMD exact division (_mm256_div_ps)
        double t_simd_div = bench([&]() {
            __m256 vdiv = _mm256_set1_ps(divisor);
            int64_t i = 0;
            for (; i + 8 <= size; i += 8)
                _mm256_storeu_ps(&out_b[i], _mm256_div_ps(_mm256_loadu_ps(&data[i]), vdiv));
            for (; i < size; ++i) out_b[i] = data[i] / divisor;
        });

        // C: SIMD multiply-by-reciprocal
        double t_simd_recip = bench([&]() {
            float recip = 1.0f / divisor;
            __m256 vrecip = _mm256_set1_ps(recip);
            int64_t i = 0;
            for (; i + 8 <= size; i += 8)
                _mm256_storeu_ps(&out_c[i], _mm256_mul_ps(_mm256_loadu_ps(&data[i]), vrecip));
            for (; i < size; ++i) out_c[i] = data[i] * recip;
        });

        // D: SIMD approximate reciprocal (_mm256_rcp_ps + Newton-Raphson)
        double t_simd_rcp = bench([&]() {
            __m256 vdiv = _mm256_set1_ps(divisor);
            __m256 rcp = _mm256_rcp_ps(vdiv);  // ~12-bit approximation
            // Newton-Raphson refinement: rcp = rcp * (2 - div * rcp)
            __m256 two = _mm256_set1_ps(2.0f);
            rcp = _mm256_mul_ps(rcp, _mm256_sub_ps(two, _mm256_mul_ps(vdiv, rcp)));
            int64_t i = 0;
            for (; i + 8 <= size; i += 8)
                _mm256_storeu_ps(&out_d[i], _mm256_mul_ps(_mm256_loadu_ps(&data[i]), rcp));
            for (; i < size; ++i) out_d[i] = data[i] * _mm_cvtss_f32(_mm256_castps256_ps128(rcp));
        });

        std::cout << std::setw(12) << size
                  << " | " << std::setw(9) << std::fixed << std::setprecision(1) << t_scalar << "μs"
                  << " | " << std::setw(9) << t_simd_div << "μs"
                  << " | " << std::setw(9) << t_simd_recip << "μs"
                  << " | " << std::setw(9) << t_simd_rcp << "μs\n";
    }

    // ─── PART 2: PRECISION COMPARISON ───
    std::cout << "\nPART 2: PRECISION (max absolute error vs exact double division)\n";
    std::cout << std::string(80, '-') << "\n";

    int64_t psize = 1000000;
    std::vector<float> pdata(psize);
    std::mt19937 gen2(42);
    std::uniform_real_distribution<float> dist2(0.001f, 1e8f);
    for (auto& v : pdata) v = dist2(gen2);

    // Test with various divisors (small, medium, large, prime)
    for (float div : {3.0f, 7.0f, 100.0f, 12345.0f, 999999.0f, 1e7f}) {
        double max_err_recip = 0, max_err_rcp = 0, max_err_div = 0;
        double sum_err_recip = 0, sum_err_rcp = 0, sum_err_div = 0;

        float recip = 1.0f / div;
        __m256 vdiv = _mm256_set1_ps(div);
        __m256 vrcp = _mm256_rcp_ps(vdiv);
        __m256 two = _mm256_set1_ps(2.0f);
        vrcp = _mm256_mul_ps(vrcp, _mm256_sub_ps(two, _mm256_mul_ps(vdiv, vrcp)));
        alignas(32) float rcp_val[8];
        _mm256_storeu_ps(rcp_val, vrcp);

        for (int64_t i = 0; i < psize; ++i) {
            double exact = static_cast<double>(pdata[i]) / static_cast<double>(div);
            double via_recip = static_cast<double>(pdata[i]) * static_cast<double>(recip);
            double via_rcp = static_cast<double>(pdata[i]) * static_cast<double>(rcp_val[0]);
            double via_div = static_cast<double>(pdata[i] / div);

            double err_recip = std::abs(via_recip - exact);
            double err_rcp = std::abs(via_rcp - exact);
            double err_div = std::abs(via_div - exact);

            max_err_recip = std::max(max_err_recip, err_recip);
            max_err_rcp = std::max(max_err_rcp, err_rcp);
            max_err_div = std::max(max_err_div, err_div);

            sum_err_recip += err_recip;
            sum_err_rcp += err_rcp;
            sum_err_div += err_div;
        }

        std::cout << "Divisor=" << std::setw(10) << std::setprecision(0) << div
                  << " | div max_err=" << std::scientific << std::setprecision(2) << max_err_div
                  << " recip=" << max_err_recip
                  << " rcp_nr=" << max_err_rcp << "\n";
    }

    // ─── PART 3: DOUBLE PRECISION ───
    std::cout << "\nPART 3: DOUBLE precision (same tests)\n";
    std::cout << std::string(80, '-') << "\n";

    for (int64_t size : {100000LL, 1000000LL, 10000000LL}) {
        std::vector<double> ddata(size);
        std::mt19937 gen3(42);
        std::uniform_real_distribution<double> ddist(1.0, 1e8);
        for (auto& v : ddata) v = ddist(gen3);

        double divisor = 12345.0;
        std::vector<double> dout_a(size), dout_b(size), dout_c(size);

        double t_scalar = bench([&]() {
            for (int64_t i = 0; i < size; ++i) dout_a[i] = ddata[i] / divisor;
        });

        double t_simd = bench([&]() {
            __m256d vdiv = _mm256_set1_pd(divisor);
            int64_t i = 0;
            for (; i + 4 <= size; i += 4)
                _mm256_storeu_pd(&dout_b[i], _mm256_div_pd(_mm256_loadu_pd(&ddata[i]), vdiv));
            for (; i < size; ++i) dout_b[i] = ddata[i] / divisor;
        });

        double t_recip = bench([&]() {
            double recip = 1.0 / divisor;
            __m256d vrecip = _mm256_set1_pd(recip);
            int64_t i = 0;
            for (; i + 4 <= size; i += 4)
                _mm256_storeu_pd(&dout_c[i], _mm256_mul_pd(_mm256_loadu_pd(&ddata[i]), vrecip));
            for (; i < size; ++i) dout_c[i] = ddata[i] * recip;
        });

        std::cout << std::setw(12) << size
                  << " | Scalar: " << std::setw(8) << std::fixed << std::setprecision(1) << t_scalar << "μs"
                  << " | SIMD div: " << std::setw(8) << t_simd << "μs"
                  << " | SIMD recip: " << std::setw(8) << t_recip << "μs\n";
    }

    return 0;
}
