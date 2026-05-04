#include <TensorLib.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <chrono>
#include <algorithm>
#include <numeric>

using namespace OwnTensor;
using Clock = std::chrono::high_resolution_clock;
using us    = std::chrono::duration<double, std::micro>;

// ─── timing config ───────────────────────────────────────────────────────────
static constexpr int WARMUP = 5;
static constexpr int ITERS  = 30;

// ─── reference helpers (completely independent of the library) ───────────────

double ref_double_sum(const std::vector<float>& v) {
    double acc = 0.0;
    for (float x : v) acc += static_cast<double>(x);
    return acc;
}

float naive_float_sum(const std::vector<float>& v) {
    float acc = 0.0f;
    for (float x : v) acc += x;
    return acc;
}

// ─── single test runner ──────────────────────────────────────────────────────

void run_test(const std::string& name, const std::vector<float>& data) {

    // ── precision references ──────────────────────────────────────────────
    double ref   = ref_double_sum(data);
    float  naive = naive_float_sum(data);

    // Build tensor once (not counted in timing)
    Tensor t({{(int64_t)data.size()}}, Dtype::Float32, Device::CPU);
    t.set_data(data);

    // ── library warm-up ───────────────────────────────────────────────────
    for (int i = 0; i < WARMUP; ++i) {
        volatile auto r = reduce_sum(t);
        (void)r;
    }

    // ── time library reduce_sum ───────────────────────────────────────────
    std::vector<double> lib_times(ITERS);
    float lib = 0.0f;
    for (int i = 0; i < ITERS; ++i) {
        auto t0 = Clock::now();
        Tensor result = reduce_sum(t);
        auto t1 = Clock::now();
        lib_times[i] = us(t1 - t0).count();
        lib = result.data<float>()[0];
    }

    // ── time naive float loop (raw memory, no library overhead) ──────────
    const float* raw = data.data();
    int64_t      N   = static_cast<int64_t>(data.size());

    volatile float  naive_sink = 0.0f;
    volatile double dbl_sink   = 0.0;

    std::vector<double> naive_times(ITERS);
    for (int i = 0; i < ITERS; ++i) {
        auto  t0 = Clock::now();
        float acc = 0.0f;
        for (int64_t j = 0; j < N; ++j) acc += raw[j];
        auto t1 = Clock::now();
        naive_times[i] = us(t1 - t0).count();
        naive_sink = acc;   // prevent dead-code elimination
    }

    // ── time double reference loop ────────────────────────────────────────
    std::vector<double> dbl_times(ITERS);
    for (int i = 0; i < ITERS; ++i) {
        auto   t0 = Clock::now();
        double acc = 0.0;
        for (int64_t j = 0; j < N; ++j) acc += static_cast<double>(raw[j]);
        auto t1 = Clock::now();
        dbl_times[i] = us(t1 - t0).count();
        dbl_sink = acc;     // prevent dead-code elimination
    }

    // ── statistics: mean and min over ITERS ──────────────────────────────
    auto mean = [](std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };
    auto minv = [](std::vector<double>& v) {
        return *std::min_element(v.begin(), v.end());
    };

    double lib_mean   = mean(lib_times);
    double lib_min    = minv(lib_times);
    double naive_mean = mean(naive_times);
    double naive_min  = minv(naive_times);
    double dbl_mean   = mean(dbl_times);
    double dbl_min    = minv(dbl_times);

    // ── errors ────────────────────────────────────────────────────────────
    double abs_err_lib   = std::fabs((double)lib  - ref);
    double abs_err_naive = std::fabs((double)naive - ref);
    double rel_err_lib   = (ref != 0.0) ? abs_err_lib   / std::fabs(ref) : abs_err_lib;
    double rel_err_naive = (ref != 0.0) ? abs_err_naive / std::fabs(ref) : abs_err_naive;

    // ── print ─────────────────────────────────────────────────────────────
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "\n======================================================\n";
    std::cout << "TEST: " << name << "  (N=" << data.size() << ")\n";

    std::cout << "\n  --- PRECISION ---\n";
    std::cout << "  Double reference   : " << ref              << "\n";
    std::cout << "  Library result     : " << lib              << "\n";
    std::cout << "  Naive float result : " << naive            << "\n";
    std::cout << "  Abs err (lib)      : " << abs_err_lib      << "\n";
    std::cout << "  Abs err (naive)    : " << abs_err_naive    << "\n";
    std::cout << "  Rel err (lib)      : " << rel_err_lib      << "\n";
    std::cout << "  Rel err (naive)    : " << rel_err_naive    << "\n";

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n  --- TIMING (" << ITERS << " runs, microseconds) ---\n";
    std::cout << "                      mean (us)   min (us)\n";
    std::cout << "  Library reduce_sum: " << std::setw(10) << lib_mean
              << "  " << std::setw(8) << lib_min   << "\n";
    std::cout << "  Naive float loop  : " << std::setw(10) << naive_mean
              << "  " << std::setw(8) << naive_min << "\n";
    std::cout << "  Double loop (ref) : " << std::setw(10) << dbl_mean
              << "  " << std::setw(8) << dbl_min   << "\n";
    std::cout << "  Library / Naive   : " << std::setprecision(2)
              << lib_mean / naive_mean << "x  (>1 = lib slower, <1 = lib faster)\n";
    std::cout << "  Library / Double  : "
              << lib_mean / dbl_mean  << "x\n";
}

// ─── dataset builders ────────────────────────────────────────────────────────

std::vector<float> make_cancellation() {
    std::vector<float> v;
    v.reserve(3001);
    for (int i = 0; i < 1000; ++i) {
        v.push_back(1e8f);
        v.push_back(1.0f);
        v.push_back(-1e8f);
    }
    v.push_back(1.0f);
    return v;
}

std::vector<float> make_ce_losses() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.1f, 5.0f);
    std::vector<float> v(65536);
    for (auto& x : v) x = dist(rng);
    return v;
}

std::vector<float> make_gradients() {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1e-4f, 1e-4f);
    std::vector<float> v(1048576);
    for (auto& x : v) x = dist(rng);
    return v;
}

std::vector<float> make_softmax() {
    float val = 1.0f / 512.0f;
    return std::vector<float>(512, val);
}

std::vector<float> make_activations() {
    std::mt19937 rng(999);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(224 * 224);
    for (auto& x : v) x = dist(rng);
    return v;
}

std::vector<float> make_embedding_weights() {
    std::mt19937 rng(7777);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    std::vector<float> v(1024 * 768);
    for (auto& x : v) x = dist(rng);
    return v;
}

// ─── main ────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\n*** PRECISION + TIMING TEST: reduce_sum (float) ***\n";
    std::cout << "Run BEFORE changes (Kahan) and AFTER changes (double acc), compare.\n";
    std::cout << "Warm-up=" << WARMUP << "  Measured=" << ITERS << "\n";

    run_test("1. Catastrophic Cancellation",   make_cancellation());
    run_test("2. Cross-Entropy Losses",         make_ce_losses());
    run_test("3. Gradient Tensor (1M values)",  make_gradients());
    run_test("4. Softmax Output (sum=1.0)",     make_softmax());
    run_test("5. Batch Norm Activations",       make_activations());
    run_test("6. Embedding Weights (786K)",     make_embedding_weights());

    return 0;
}
