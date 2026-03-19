#include <TensorLib.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <string>

using namespace OwnTensor;

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
    // Reference values (independent of library)
    double ref   = ref_double_sum(data);
    float  naive = naive_float_sum(data);

    // Library result  (Kahan path BEFORE changes, double-acc AFTER changes)
    Tensor t({{(int64_t)data.size()}}, Dtype::Float32, Device::CPU);
    t.set_data(data);
    Tensor result = reduce_sum(t);
    float lib = result.data<float>()[0];

    // Errors vs double reference
    double abs_err_lib   = std::fabs((double)lib  - ref);
    double abs_err_naive = std::fabs((double)naive - ref);
    double rel_err_lib   = (ref != 0.0) ? abs_err_lib   / std::fabs(ref) : abs_err_lib;
    double rel_err_naive = (ref != 0.0) ? abs_err_naive / std::fabs(ref) : abs_err_naive;

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "\n======================================================\n";
    std::cout << "TEST: " << name << "  (N=" << data.size() << ")\n";
    std::cout << "------------------------------------------------------\n";
    std::cout << "  Double reference   : " << ref              << "\n";
    std::cout << "  Library result     : " << lib              << "\n";
    std::cout << "  Naive float result : " << naive            << "\n";
    std::cout << "  Abs err (lib)      : " << abs_err_lib      << "\n";
    std::cout << "  Abs err (naive)    : " << abs_err_naive    << "\n";
    std::cout << "  Rel err (lib)      : " << rel_err_lib      << "\n";
    std::cout << "  Rel err (naive)    : " << rel_err_naive    << "\n";
}

// ─── dataset builders ────────────────────────────────────────────────────────

// 1. Catastrophic Cancellation
//    Pattern: [1e8, 1.0, -1e8] x1000 + one trailing 1.0
//    True sum = 1001.0
//    Naive float drops every 1.0 completely → ~0.0
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

// 2. Cross-Entropy Losses (batch of 65536 samples)
//    uniform [0.1, 5.0] — typical per-sample CE loss range
std::vector<float> make_ce_losses() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.1f, 5.0f);
    std::vector<float> v(65536);
    for (auto& x : v) x = dist(rng);
    return v;
}

// 3. Gradient Tensor (1M tiny values)
//    uniform [-1e-4, 1e-4] — typical backprop gradient magnitudes
//    Large N, tiny values — error grows fast with naive float
std::vector<float> make_gradients() {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1e-4f, 1e-4f);
    std::vector<float> v(1048576);
    for (auto& x : v) x = dist(rng);
    return v;
}

// 4. Softmax Output (512 equal values = 1/512)
//    True sum = 1.0 exactly — tests precision for small equal values
std::vector<float> make_softmax() {
    int N = 512;
    float val = 1.0f / static_cast<float>(N);
    return std::vector<float>(N, val);
}

// 5. Batch Norm Activations (224x224 channel = 50176 values)
//    normal(0, 1) — centered activations, true sum ≈ 0
//    Large N near zero: relative error meaningless, absolute error is what matters
std::vector<float> make_activations() {
    std::mt19937 rng(999);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(224 * 224);
    for (auto& x : v) x = dist(rng);
    return v;
}

// 6. Embedding Weights (1024x768 = 786432 values)
//    normal(0, 0.02) — Kaiming/Xavier-style init, large tensor
//    Very large N of small values — max stress test for float precision
std::vector<float> make_embedding_weights() {
    std::mt19937 rng(7777);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    std::vector<float> v(1024 * 768);
    for (auto& x : v) x = dist(rng);
    return v;
}

// ─── main ────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\n*** PRECISION TEST: reduce_sum (float) ***\n";
    std::cout << "Run BEFORE changes (Kahan) and AFTER changes (double acc), compare.\n";

    run_test("1. Catastrophic Cancellation",   make_cancellation());
    run_test("2. Cross-Entropy Losses",         make_ce_losses());
    run_test("3. Gradient Tensor (1M values)",  make_gradients());
    run_test("4. Softmax Output (sum=1.0)",     make_softmax());
    run_test("5. Batch Norm Activations",       make_activations());
    run_test("6. Embedding Weights (786K)",     make_embedding_weights());

    return 0;
}
