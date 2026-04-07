/**
 * VIGOROUS Outer Reduction Strategy Benchmark
 * Tests Strategy 1 (SIMD + idle threads) vs Strategy 2 (all threads + scalar)
 * across ALL combinations of:
 *   - Reduction sizes: 100, 1K, 10K, 100K, 1M, 10M
 *   - Output columns: 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 48, 64, 128, 256, 512, 1024
 *   - Data distributions: uniform, gaussian, large values, subnormals, mixed
 *   - DL-realistic shapes: BatchNorm channels, feature dims, spatial dims
 */
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <algorithm>
#include <iomanip>
#include <immintrin.h>
#include <random>
#include <cmath>
#include <cstring>

const int NT = omp_get_max_threads();

void strat1_simd_idle(const float* in, float* out, int R, int C) {
    // Strategy 1: Parallelize over output columns, vertical SIMD
    // Some threads may sit idle if C < NT
    #pragma omp parallel for num_threads(NT)
    for (int o = 0; o < (C/8)*8; o += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (int r = 0; r < R; ++r)
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(in + r*C + o));
        _mm256_storeu_ps(out + o, acc);
    }
    for (int o = (C/8)*8; o < C; ++o) {
        float s = 0; for (int r = 0; r < R; ++r) s += in[r*C+o]; out[o] = s;
    }
}

void strat2_split_all_threads(const float* in, float* out, int R, int C) {
    // Strategy 2: For each output slot, split reduction rows across ALL threads
    for (int o = 0; o < C; ++o) {
        float thread_acc[128] = {};
        #pragma omp parallel num_threads(NT)
        {
            int t = omp_get_thread_num(), n = omp_get_num_threads();
            int ch = (R+n-1)/n, b = t*ch, e = std::min(b+ch, R);
            float l = 0;
            for (int r = b; r < e; ++r) l += in[r*C + o];
            thread_acc[t] = l;
        }
        float s = 0;
        for (int t = 0; t < NT; ++t) s += thread_acc[t];
        out[o] = s;
    }
}

// Hybrid: Strategy 1 for >= threshold cols, Strategy 2 for < threshold
void strat_hybrid(const float* in, float* out, int R, int C, int threshold) {
    if (C >= threshold) {
        strat1_simd_idle(in, out, R, C);
    } else {
        strat2_split_all_threads(in, out, R, C);
    }
}

double bench(auto fn, int warmup = 3, int iters = 10) {
    for (int i = 0; i < warmup; ++i) fn();
    std::vector<double> t;
    for (int i = 0; i < iters; ++i) {
        auto s = std::chrono::high_resolution_clock::now();
        fn();
        auto e = std::chrono::high_resolution_clock::now();
        t.push_back(std::chrono::duration<double, std::micro>(e - s).count());
    }
    std::sort(t.begin(), t.end());
    return t[iters / 2];
}

int main() {
    printf("==========================================================================\n");
    printf("  VIGOROUS Outer Reduction Strategy Benchmark\n");
    printf("  CPU: %d threads | AVX2 8-wide float\n", NT);
    printf("==========================================================================\n\n");

    std::mt19937 rng(42);

    // ================================================================
    // TEST 1: Full matrix — all reduction sizes × all output cols
    // ================================================================
    printf("TEST 1: Full matrix (uniform random data)\n");
    printf("S1 = Strategy 1 (SIMD + idle threads)\n");
    printf("S2 = Strategy 2 (all threads + scalar per output)\n\n");

    std::vector<int> red_sizes = {100, 1000, 10000, 100000, 1000000};
    std::vector<int> col_sizes = {1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 48, 64, 128, 256, 512};

    for (int R : red_sizes) {
        printf("--- Reduction rows = %d ---\n", R);
        printf("%6s %12s %12s %8s %10s\n", "Cols", "S1(μs)", "S2(μs)", "Winner", "Speedup");
        printf("----------------------------------------------------------\n");

        for (int C : col_sizes) {
            if ((int64_t)R * C > 500000000LL) continue; // skip > 2GB
            std::vector<float> data(R * C), o1(C), o2(C);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (auto& v : data) v = dist(rng);

            double t1 = bench([&]() { strat1_simd_idle(data.data(), o1.data(), R, C); });
            double t2 = bench([&]() { strat2_split_all_threads(data.data(), o2.data(), R, C); });

            // Verify correctness
            float max_diff = 0;
            for (int i = 0; i < C; ++i) max_diff = std::max(max_diff, std::abs(o1[i] - o2[i]));

            const char* winner = (t1 <= t2) ? "S1" : "S2";
            double speedup = (t1 <= t2) ? t2 / t1 : t1 / t2;
            printf("%6d %10.0fμs %10.0fμs %8s %8.1fx  %s\n",
                   C, t1, t2, winner, speedup,
                   max_diff > 1.0f ? "MISMATCH!" : "");
        }
        printf("\n");
    }

    // ================================================================
    // TEST 2: DL-realistic shapes (outer reduction = reduce batch dim)
    // ================================================================
    printf("==========================================================================\n");
    printf("TEST 2: Deep Learning realistic shapes (reduce dim=0 = batch)\n");
    printf("==========================================================================\n\n");

    struct DLShape { const char* name; int batch; int features; };
    DLShape dl_shapes[] = {
        {"BERT-hidden", 32, 768},
        {"GPT-hidden", 16, 4096},
        {"ResNet-features", 64, 2048},
        {"ViT-embed", 32, 768},
        {"Small-batch-big-feat", 4, 4096},
        {"Tiny-batch-huge-feat", 2, 8192},
        {"Single-sample", 1, 1024},
        {"Large-batch-small", 256, 64},
        {"BatchNorm-channels", 32, 256},
        {"Spatial-flat", 32, 3136},  // 56*56
        {"Attention-heads", 32, 12},
        {"Few-classes", 1000, 10},
        {"ImageNet-classes", 64, 1000},
    };

    printf("%-25s %8s %8s %12s %12s %8s %8s\n",
           "Application", "Batch", "Feat", "S1(μs)", "S2(μs)", "Winner", "Speedup");
    printf("--------------------------------------------------------------------------------\n");

    for (auto& s : dl_shapes) {
        int R = s.batch, C = s.features;
        std::vector<float> data(R * C), o1(C), o2(C);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& v : data) v = dist(rng);

        double t1 = bench([&]() { strat1_simd_idle(data.data(), o1.data(), R, C); });
        double t2 = bench([&]() { strat2_split_all_threads(data.data(), o2.data(), R, C); });

        const char* winner = (t1 <= t2) ? "S1" : "S2";
        double speedup = (t1 <= t2) ? t2 / t1 : t1 / t2;
        printf("%-25s %8d %8d %10.0fμs %10.0fμs %8s %7.1fx\n",
               s.name, R, C, t1, t2, winner, speedup);
    }

    // ================================================================
    // TEST 3: Different data distributions (does data pattern affect strategy?)
    // ================================================================
    printf("\n==========================================================================\n");
    printf("TEST 3: Data distribution impact (1M rows × 8 cols)\n");
    printf("==========================================================================\n\n");

    int R3 = 1000000, C3 = 8;
    printf("%-25s %12s %12s %8s\n", "Distribution", "S1(μs)", "S2(μs)", "Winner");
    printf("----------------------------------------------------------\n");

    auto test_dist = [&](const char* name, auto fill_fn) {
        std::vector<float> data(R3 * C3), o1(C3), o2(C3);
        fill_fn(data);
        double t1 = bench([&]() { strat1_simd_idle(data.data(), o1.data(), R3, C3); });
        double t2 = bench([&]() { strat2_split_all_threads(data.data(), o2.data(), R3, C3); });
        printf("%-25s %10.0fμs %10.0fμs %8s\n", name, t1, t2, (t1<=t2)?"S1":"S2");
    };

    test_dist("Uniform [-1, 1]", [&](std::vector<float>& d) {
        std::uniform_real_distribution<float> u(-1, 1); for (auto& v : d) v = u(rng);
    });
    test_dist("Gaussian N(0, 1)", [&](std::vector<float>& d) {
        std::normal_distribution<float> n(0, 1); for (auto& v : d) v = n(rng);
    });
    test_dist("Large values [1e6, 1e7]", [&](std::vector<float>& d) {
        std::uniform_real_distribution<float> u(1e6, 1e7); for (auto& v : d) v = u(rng);
    });
    test_dist("Subnormals [1e-45, 1e-38]", [&](std::vector<float>& d) {
        std::uniform_real_distribution<float> u(1e-45f, 1e-38f); for (auto& v : d) v = u(rng);
    });
    test_dist("All zeros", [&](std::vector<float>& d) {
        std::fill(d.begin(), d.end(), 0.0f);
    });
    test_dist("All ones", [&](std::vector<float>& d) {
        std::fill(d.begin(), d.end(), 1.0f);
    });
    test_dist("10% NaN", [&](std::vector<float>& d) {
        std::uniform_real_distribution<float> u(-1, 1);
        std::uniform_real_distribution<float> p(0, 1);
        for (auto& v : d) v = (p(rng) < 0.1f) ? std::nanf("") : u(rng);
    });
    test_dist("Alternating ±1e30", [&](std::vector<float>& d) {
        for (size_t i = 0; i < d.size(); ++i) d[i] = (i % 2 == 0) ? 1e30f : -1e30f;
    });

    // ================================================================
    // TEST 4: Find optimal crossover threshold
    // ================================================================
    printf("\n==========================================================================\n");
    printf("TEST 4: Optimal crossover threshold (at what C does S1 start winning?)\n");
    printf("==========================================================================\n\n");

    printf("%10s", "Rows\\Cols");
    for (int C = 1; C <= 32; ++C) printf(" %4d", C);
    printf("\n");
    printf("%s\n", std::string(10 + 32 * 5, '-').c_str());

    for (int R : {100, 1000, 10000, 100000, 1000000}) {
        printf("%10d", R);
        for (int C = 1; C <= 32; ++C) {
            if ((int64_t)R * C > 500000000LL) { printf("    -"); continue; }
            std::vector<float> data(R * C), o1(C), o2(C);
            std::uniform_real_distribution<float> dist(-1, 1);
            for (auto& v : data) v = dist(rng);

            double t1 = bench([&]() { strat1_simd_idle(data.data(), o1.data(), R, C); }, 2, 5);
            double t2 = bench([&]() { strat2_split_all_threads(data.data(), o2.data(), R, C); }, 2, 5);

            printf(" %4s", (t1 <= t2) ? "S1" : "S2");
        }
        printf("\n");
    }

    printf("\nBENCHMARK COMPLETE\n");
    return 0;
}
