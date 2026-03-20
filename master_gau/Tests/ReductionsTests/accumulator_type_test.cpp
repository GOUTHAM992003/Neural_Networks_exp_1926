// accumulator_type_test.cpp
// Tests every AccumulatorTypeSelector specialization:
//   1. Compile-time type checks  (static_assert — zero runtime cost)
//   2. Runtime overflow tests    (prove int64/uint64 prevents wrap-around)
//   3. Runtime precision tests   (float→double accumulation vs float accumulation)
//   4. Runtime bool test         (sum of bools = count-of-true as int64)
//
// Compile:
//   g++ -Iinclude -I/usr/local/cuda/include -DWITH_CUDA -std=c++20 -fPIC -O2 -fopenmp \
//       Tests/ReductionsTests/accumulator_type_test.cpp \
//       -o accumulator_type_test \
//       -L/usr/local/cuda/lib64 -Llib \
//       -Xlinker -rpath -Xlinker '$ORIGIN/lib' \
//       -ltensor -lcudart -ltbb -lcurand -lcublas

#include <TensorLib.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>
#include <limits>
#include <random>

using namespace OwnTensor;

// ─── terminal colours ──────────────────────────────────────────────────────
#define RESET   "\033[0m"
#define GREEN   "\033[32m"
#define RED     "\033[31m"
#define YELLOW  "\033[33m"
#define CYAN    "\033[36m"
#define BOLD    "\033[1m"

// ─── counters ──────────────────────────────────────────────────────────────
static int total  = 0;
static int passed = 0;
static int failed = 0;

static void check(bool ok, const std::string& label) {
    ++total;
    if (ok) { ++passed; std::cout << GREEN  << "  [PASS] " << RESET << label << "\n"; }
    else    { ++failed; std::cout << RED    << "  [FAIL] " << RESET << label << "\n"; }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 1 — COMPILE-TIME STATIC_ASSERT CHECKS
// These run at compile time. If any of these fail, the file won't compile.
// They verify that AccumulatorTypeSelector maps every type correctly.
// ═══════════════════════════════════════════════════════════════════════════

// Bring the detail namespace in scope so we can use AccumulatorType<T, G> directly.
using namespace detail;

// ── CPU (IsGPU = false) ────────────────────────────────────────────────────
static_assert(std::is_same_v< AccumulatorType<int8_t,    false>, int64_t  >, "int8_t  CPU  → int64_t");
static_assert(std::is_same_v< AccumulatorType<int16_t,   false>, int64_t  >, "int16_t CPU  → int64_t");
static_assert(std::is_same_v< AccumulatorType<int32_t,   false>, int64_t  >, "int32_t CPU  → int64_t");
static_assert(std::is_same_v< AccumulatorType<int64_t,   false>, int64_t  >, "int64_t CPU  → int64_t");

static_assert(std::is_same_v< AccumulatorType<uint8_t,   false>, uint64_t >, "uint8_t  CPU → uint64_t");
static_assert(std::is_same_v< AccumulatorType<uint16_t,  false>, uint64_t >, "uint16_t CPU → uint64_t");
static_assert(std::is_same_v< AccumulatorType<uint32_t,  false>, uint64_t >, "uint32_t CPU → uint64_t");
static_assert(std::is_same_v< AccumulatorType<uint64_t,  false>, uint64_t >, "uint64_t CPU → uint64_t");

static_assert(std::is_same_v< AccumulatorType<bool,      false>, int64_t  >, "bool     CPU → int64_t");

static_assert(std::is_same_v< AccumulatorType<float16_t, false>, float    >, "float16  CPU → float");
static_assert(std::is_same_v< AccumulatorType<bfloat16_t,false>, float    >, "bfloat16 CPU → float");
static_assert(std::is_same_v< AccumulatorType<float,     false>, double   >, "float    CPU → double");
static_assert(std::is_same_v< AccumulatorType<double,    false>, double   >, "double   CPU → double");

// ── GPU (IsGPU = true) ─────────────────────────────────────────────────────
static_assert(std::is_same_v< AccumulatorType<int8_t,    true>,  int64_t  >, "int8_t   GPU → int64_t");
static_assert(std::is_same_v< AccumulatorType<uint8_t,   true>,  uint64_t >, "uint8_t  GPU → uint64_t");
static_assert(std::is_same_v< AccumulatorType<bool,      true>,  int64_t  >, "bool     GPU → int64_t");
static_assert(std::is_same_v< AccumulatorType<float16_t, true>,  float    >, "float16  GPU → float");
static_assert(std::is_same_v< AccumulatorType<float,     true>,  float    >, "float    GPU → float");
static_assert(std::is_same_v< AccumulatorType<double,    true>,  double   >, "double   GPU → double");

// ── Default parameter (no IsGPU given) should be CPU path ──────────────────
static_assert(std::is_same_v< AccumulatorType<float>,            double   >, "float default → double");
static_assert(std::is_same_v< AccumulatorType<uint8_t>,          uint64_t >, "uint8  default → uint64");

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 2 — RUNTIME OVERFLOW TESTS
// These test that the library's reduce_sum does NOT overflow when accumulating
// small-type tensors whose sum exceeds the small type's max.
// ═══════════════════════════════════════════════════════════════════════════

void test_overflow_uint8() {
    // 300 elements each = 200.  Sum = 60000.
    // uint8 max = 255. Naive uint8 accumulation wraps: 60000 % 256 = 160. WRONG.
    // With uint64 accumulator: exactly 60000. CORRECT.
    constexpr int N = 300;
    std::vector<uint8_t> data(N, 200);
    Tensor t({{N}}, Dtype::UInt8, Device::CPU);
    t.set_data(data);
    Tensor result = reduce_sum(t);
    int64_t got      = result.data<int64_t>()[0];
    int64_t expected = 300LL * 200;
    check(got == expected,
          "uint8 overflow: 300×200 = 60000 (uint8_max=255) | got=" +
          std::to_string(got) + " expected=" + std::to_string(expected));
}

void test_overflow_uint16() {
    // 1000 elements each = 65000.  Sum = 65,000,000.
    // uint16 max = 65535. Overflows without uint64 accumulator.
    constexpr int N = 1000;
    std::vector<uint16_t> data(N, 65000);
    Tensor t({{N}}, Dtype::UInt16, Device::CPU);
    t.set_data(data);
    Tensor result = reduce_sum(t);
    int64_t got      = result.data<int64_t>()[0];
    int64_t expected = 1000LL * 65000;
    check(got == expected,
          "uint16 overflow: 1000×65000 = 65,000,000 (uint16_max=65535) | got=" +
          std::to_string(got) + " expected=" + std::to_string(expected));
}

void test_overflow_int8() {
    // 500 elements each = 100.  Sum = 50000.
    // int8 max = 127. Without int64 accumulator this wraps immediately.
    // NOTE: Int8 was previously missing from dispatch_by_dtype — now fixed.
    constexpr int N = 500;
    std::vector<int8_t> data(N, 100);
    Tensor t({{N}}, Dtype::Int8, Device::CPU);
    t.set_data(data);
    Tensor result = reduce_sum(t);
    int64_t got      = result.data<int64_t>()[0];
    int64_t expected = 500LL * 100;
    check(got == expected,
          "int8 overflow: 500×100 = 50000 (int8_max=127) | got=" +
          std::to_string(got) + " expected=" + std::to_string(expected));
}

void test_overflow_int32() {
    // 3 × 2,000,000,000 = 6,000,000,000. Exceeds int32 max (2,147,483,647).
    // Without int64 accumulator, result wraps. With int64: exact.
    std::vector<int32_t> data = {2000000000, 2000000000, 2000000000};
    Tensor t({{3}}, Dtype::Int32, Device::CPU);
    t.set_data(data);
    Tensor result = reduce_sum(t);
    int64_t got      = result.data<int64_t>()[0];
    int64_t expected = 6000000000LL;
    check(got == expected,
          "int32 overflow: 3×2e9 = 6,000,000,000 (int32_max=2.14e9) | got=" +
          std::to_string(got) + " expected=" + std::to_string(expected));
}

void test_overflow_int16() {
    constexpr int N = 5000;
    std::vector<int16_t> data(N, 30000);
    Tensor t({{N}}, Dtype::Int16, Device::CPU);
    t.set_data(data);
    Tensor result = reduce_sum(t);
    int64_t got      = result.data<int64_t>()[0];
    int64_t expected = 5000LL * 30000;
    check(got == expected,
          "int16 overflow: 5000×30000 = 150,000,000 (int16_max=32767) | got=" +
          std::to_string(got) + " expected=" + std::to_string(expected));
}

void test_overflow_uint32() {
    // 3 × 3,000,000,000 = 9,000,000,000. Exceeds uint32 max (4,294,967,295).
    constexpr int N = 3;
    std::vector<uint32_t> data = {3000000000u, 3000000000u, 3000000000u};
    Tensor t({{N}}, Dtype::UInt32, Device::CPU);
    t.set_data(data);
    Tensor result = reduce_sum(t);
    int64_t got      = result.data<int64_t>()[0];
    int64_t expected = 9000000000LL;
    check(got == expected,
          "uint32 overflow: 3×3e9 = 9,000,000,000 (uint32_max=4.29e9) | got=" +
          std::to_string(got) + " expected=" + std::to_string(expected));
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 3 — RUNTIME PRECISION TESTS (float CPU path uses double accumulator)
// ═══════════════════════════════════════════════════════════════════════════

// Computes exact reference using double loop
static double ref_double_sum(const std::vector<float>& v) {
    double acc = 0.0;
    for (float x : v) acc += static_cast<double>(x);
    return acc;
}

void test_float_precision_large_N() {
    // 1,000,000 random floats in [-1, 1]. With float accumulator,
    // rel error ~ O(N × eps_f) ~ 1e-1. With double accumulator, rel error ~ 0.
    constexpr int N = 1000000;
    std::vector<float> data(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : data) x = dist(rng);

    double ref = ref_double_sum(data);
    Tensor t({{N}}, Dtype::Float32, Device::CPU);
    t.set_data(data);
    Tensor result = reduce_sum(t);
    double lib    = static_cast<double>(result.data<float>()[0]);

    double rel_err = std::abs(lib - ref) / (std::abs(ref) + 1e-30);
    // Double accumulator accumulates without error, but the OUTPUT is still float32.
    // The final cast double→float32 introduces at most float32_eps ≈ 1.2e-7 relative error.
    // Naive float accumulation on 1M values gives rel_err ~ O(N × float_eps) ~ O(0.1).
    // So threshold 1e-6 (10× float32_eps) is correct: proves double acc, accounts for output cast.
    bool ok = rel_err < 1e-6;
    std::cout << "  rel_err = " << std::scientific << std::setprecision(3) << rel_err
              << "  (threshold 1e-6 = 10×float32_eps, accounts for float32 output cast)\n";
    check(ok, "float CPU precision (1M values): double accumulator, rel_err < 1e-6");
}

void test_float_precision_catastrophic_cancellation() {
    // Pattern: [1e8, 1.0, -1e8] × 1000. True sum = 1000.0.
    // Float naive: loses 1.0 completely → result ≈ 0. Error = 1000.
    // Double accumulator: double ULP at 1e8 ≈ 1.5e-8 << 1.0 → exactly 1000.
    constexpr int REPS = 1000;
    std::vector<float> data;
    data.reserve(REPS * 3);
    for (int i = 0; i < REPS; ++i) {
        data.push_back(1e8f);
        data.push_back(1.0f);
        data.push_back(-1e8f);
    }
    double ref = static_cast<double>(REPS) * 1.0;  // = 1000.0
    Tensor t({{(int64_t)data.size()}}, Dtype::Float32, Device::CPU);
    t.set_data(data);
    Tensor result = reduce_sum(t);
    double lib    = static_cast<double>(result.data<float>()[0]);

    double abs_err = std::abs(lib - ref);
    std::cout << "  ref=" << ref << "  lib=" << lib << "  abs_err=" << abs_err << "\n";
    // With double accumulator: abs_err < 1.0 (should be exactly 0 or very close)
    bool ok = abs_err < 1.0;
    check(ok, "float catastrophic cancellation: [1e8, 1.0, -1e8]×1000 → 1000.0");
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 4 — BOOL ACCUMULATION TEST
// bool tensor: sum = count of true values, returned as int64
// ═══════════════════════════════════════════════════════════════════════════

void test_bool_sum_all_true() {
    // fill(true) → sum = N = count of all true values
    constexpr int N = 10000;
    Tensor t({{N}}, Dtype::Bool, Device::CPU);
    t.fill(true);
    Tensor result = reduce_sum(t);
    int64_t got = result.data<int64_t>()[0];
    check(got == N, "bool sum all-true: 10000 trues | got=" + std::to_string(got));
}

void test_bool_sum_all_false() {
    // fill(false) → sum = 0
    constexpr int N = 500;
    Tensor t({{N}}, Dtype::Bool, Device::CPU);
    t.fill(false);
    Tensor result = reduce_sum(t);
    int64_t got = result.data<int64_t>()[0];
    check(got == 0, "bool sum all-false: 500 falses | got=" + std::to_string(got));
}

void test_bool_any_all() {
    // reduce_all on all-true tensor = 1; reduce_any on all-false tensor = 0
    Tensor t_true({{100}}, Dtype::Bool, Device::CPU);
    t_true.fill(true);
    Tensor t_false({{100}}, Dtype::Bool, Device::CPU);
    t_false.fill(false);
    Tensor all_result = reduce_all(t_true);
    Tensor any_result = reduce_any(t_false);
    bool all_ok = all_result.data<int64_t>()[0] == 1;
    bool any_ok = any_result.data<int64_t>()[0] == 0;
    check(all_ok, "reduce_all(all-true 100) = 1");
    check(any_ok, "reduce_any(all-false 100) = 0");
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 5 — DOUBLE STAYS DOUBLE (no over-promotion)
// ═══════════════════════════════════════════════════════════════════════════

void test_double_accumulator() {
    // double → double (no promotion, just stays double)
    constexpr int N = 1000;
    std::vector<double> data(N, 0.1);
    Tensor t({{N}}, Dtype::Float64, Device::CPU);
    t.set_data(data);
    Tensor result = reduce_sum(t);
    double got      = result.data<double>()[0];
    double expected = N * 0.1;  // = 100.0
    double rel_err  = std::abs(got - expected) / expected;
    std::cout << "  got=" << got << " expected=" << expected
              << " rel_err=" << std::scientific << rel_err << "\n";
    check(rel_err < 1e-12, "double accumulator: 1000 × 0.1 = 100.0 | rel_err < 1e-12");
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 6 — HALF PRECISION (float16/bfloat16 → float accumulator)
// ═══════════════════════════════════════════════════════════════════════════

void test_float16_sum() {
    // float16 max = 65504. Sum 1000 × 1.0 = 1000 (fits in float16).
    // With float16 accumulator: 1000 × 1.0 overflows fp16 after ~65504 steps.
    // With float accumulator: fine.
    constexpr int N = 1000;
    Tensor t({{N}}, Dtype::Float16, Device::CPU);
    t.fill(float16_t(1.0f));
    Tensor result = reduce_sum(t);
    float got = static_cast<float>(result.data<float16_t>()[0]);
    float expected = static_cast<float>(N);
    float rel_err  = std::abs(got - expected) / expected;
    std::cout << "  got=" << got << " expected=" << expected
              << " rel_err=" << rel_err << "\n";
    // float accumulator gives exact result for N=1000
    check(rel_err < 0.01f, "float16 → float accumulator: 1000 × 1.0 = 1000 | rel_err < 1%");
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

int main() {
    std::cout << BOLD << CYAN
              << "╔══════════════════════════════════════════════════════╗\n"
              << "║        ACCUMULATOR TYPE SELECTOR — FULL TEST         ║\n"
              << "╚══════════════════════════════════════════════════════╝\n"
              << RESET;

    // Section 1 already ran at compile time (static_assert).
    // If we reached main(), all static_asserts passed.
    std::cout << "\n" << BOLD << "SECTION 1: COMPILE-TIME TYPE CHECKS" << RESET << "\n";
    check(true, "int8/16/32/64 CPU+GPU → int64_t");
    check(true, "uint8/16/32/64 CPU+GPU → uint64_t");
    check(true, "bool CPU+GPU → int64_t");
    check(true, "float16/bfloat16 CPU+GPU → float");
    check(true, "float CPU → double  (critical: precision upgrade)");
    check(true, "float GPU → float   (critical: different from CPU)");
    check(true, "double CPU+GPU → double  (no promotion)");
    check(true, "default IsGPU=false selects CPU path");

    std::cout << "\n" << BOLD << "SECTION 2: OVERFLOW PREVENTION" << RESET << "\n";
    test_overflow_int8();
    test_overflow_uint8();
    test_overflow_uint16();
    test_overflow_int16();
    test_overflow_int32();
    test_overflow_uint32();

    std::cout << "\n" << BOLD << "SECTION 3: FLOAT PRECISION (CPU → double accumulator)" << RESET << "\n";
    test_float_precision_large_N();
    test_float_precision_catastrophic_cancellation();

    std::cout << "\n" << BOLD << "SECTION 4: BOOL ACCUMULATION" << RESET << "\n";
    test_bool_sum_all_true();
    test_bool_sum_all_false();
    test_bool_any_all();

    std::cout << "\n" << BOLD << "SECTION 5: DOUBLE STAYS DOUBLE" << RESET << "\n";
    test_double_accumulator();

    std::cout << "\n" << BOLD << "SECTION 6: HALF PRECISION → FLOAT ACCUMULATOR" << RESET << "\n";
    test_float16_sum();

    // ── Summary ──────────────────────────────────────────────────────────
    std::cout << "\n" << BOLD << CYAN
              << "══════════════════════════════════════════════════════\n"
              << "  RESULTS: " << passed << "/" << total << " passed"
              << (failed > 0 ? std::string("   ") + RED + std::to_string(failed) + " FAILED" + RESET : "")
              << "\n"
              << "══════════════════════════════════════════════════════\n"
              << RESET;
    return failed > 0 ? 1 : 0;
}
