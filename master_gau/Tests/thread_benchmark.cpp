// thread_benchmark.cpp
// Benchmark: parallel float sum-reduction using OpenMP thread-local accumulators
// Strategy 2: split array across threads, each accumulates locally, then combine.

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <omp.h>
#include <vector>

// ---- tuning knobs ----
static constexpr int WARMUP_ITERS = 10;
static constexpr int BENCH_ITERS  = 100;

static const std::vector<std::size_t> SIZES = {
    1'000,        // 1K
    10'000,       // 10K
    50'000,       // 50K
    100'000,      // 100K
    500'000,      // 500K
    1'000'000,    // 1M
    10'000'000    // 10M
};

static const std::vector<int> THREAD_COUNTS = {1, 2, 4, 8, 14, 28};

// ---- Strategy 2: thread-local accumulators + combine ----
static float reduce_sum(const float* data, std::size_t n, int nthreads) {
    // One accumulator per thread, cache-line padded to avoid false sharing.
    struct alignas(64) PaddedAcc { float val = 0.0f; };
    std::vector<PaddedAcc> accums(nthreads);

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        float local_sum = 0.0f;

        #pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            local_sum += data[i];
        }

        accums[tid].val = local_sum;
    }

    // Combine phase (sequential -- negligible cost).
    float total = 0.0f;
    for (int t = 0; t < nthreads; ++t) {
        total += accums[t].val;
    }
    return total;
}

// ---- pretty-print helpers ----
static const char* size_label(std::size_t n) {
    static char buf[32];
    if (n >= 1'000'000) std::snprintf(buf, sizeof buf, "%zuM", n / 1'000'000);
    else if (n >= 1'000) std::snprintf(buf, sizeof buf, "%zuK", n / 1'000);
    else                  std::snprintf(buf, sizeof buf, "%zu", n);
    return buf;
}

int main() {
    std::printf("======================================================================\n");
    std::printf("  Thread Benchmark: float sum-reduction (Strategy 2)\n");
    std::printf("  Warm-up iterations : %d\n", WARMUP_ITERS);
    std::printf("  Benchmark iterations: %d\n", BENCH_ITERS);
    std::printf("  Available HW threads: %d\n", omp_get_max_threads());
    std::printf("======================================================================\n\n");

    // ---- header row ----
    std::printf("%-10s", "Size");
    for (int t : THREAD_COUNTS) {
        char col[32];
        std::snprintf(col, sizeof col, "%dT (us)", t);
        std::printf("%14s", col);
    }
    std::printf("\n");
    std::printf("%-10s", "--------");
    for (std::size_t i = 0; i < THREAD_COUNTS.size(); ++i)
        std::printf("%14s", "----------");
    std::printf("\n");

    for (std::size_t n : SIZES) {
        // Fill with small positive values so the sum is representable.
        std::vector<float> data(n);
        for (std::size_t i = 0; i < n; ++i)
            data[i] = 1.0f / static_cast<float>(n);   // sum should be ~1.0

        std::printf("%-10s", size_label(n));

        for (int nthreads : THREAD_COUNTS) {
            // Warm up (lets the OS spin up threads, populate caches, etc.)
            for (int w = 0; w < WARMUP_ITERS; ++w) {
                volatile float sink = reduce_sum(data.data(), n, nthreads);
                (void)sink;
            }

            // Timed runs
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int r = 0; r < BENCH_ITERS; ++r) {
                volatile float sink = reduce_sum(data.data(), n, nthreads);
                (void)sink;
            }
            auto t1 = std::chrono::high_resolution_clock::now();

            double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            double avg_us   = total_us / BENCH_ITERS;

            std::printf("%14.2f", avg_us);
        }
        std::printf("\n");
    }

    // ---- speedup table ----
    std::printf("\n\nSpeedup relative to 1-thread:\n\n");
    std::printf("%-10s", "Size");
    for (int t : THREAD_COUNTS) {
        char col[32];
        std::snprintf(col, sizeof col, "%dT", t);
        std::printf("%10s", col);
    }
    std::printf("\n");
    std::printf("%-10s", "--------");
    for (std::size_t i = 0; i < THREAD_COUNTS.size(); ++i)
        std::printf("%10s", "------");
    std::printf("\n");

    for (std::size_t n : SIZES) {
        std::vector<float> data(n);
        for (std::size_t i = 0; i < n; ++i)
            data[i] = 1.0f / static_cast<float>(n);

        std::vector<double> times(THREAD_COUNTS.size());
        for (std::size_t ti = 0; ti < THREAD_COUNTS.size(); ++ti) {
            int nthreads = THREAD_COUNTS[ti];
            for (int w = 0; w < WARMUP_ITERS; ++w) {
                volatile float sink = reduce_sum(data.data(), n, nthreads);
                (void)sink;
            }
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int r = 0; r < BENCH_ITERS; ++r) {
                volatile float sink = reduce_sum(data.data(), n, nthreads);
                (void)sink;
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            times[ti] = std::chrono::duration<double, std::micro>(t1 - t0).count() / BENCH_ITERS;
        }

        std::printf("%-10s", size_label(n));
        double base = times[0];
        for (std::size_t ti = 0; ti < THREAD_COUNTS.size(); ++ti) {
            std::printf("%10.2fx", base / times[ti]);
        }
        std::printf("\n");
    }

    std::printf("\nDone.\n");
    return 0;
}
