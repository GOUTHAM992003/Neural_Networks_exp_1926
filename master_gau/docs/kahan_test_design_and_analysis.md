[text](../../accumulator_system_analysis.md)# Kahan Summation — Test Design, Metrics, and Full Baseline Analysis

**File:** `Tests/ReductionsTests/kahan_precision_test.cpp`
**Purpose:** Capture BEFORE and AFTER snapshots of `reduce_sum` precision and performance
**Change being tested:** Remove Kahan compensated summation → promote `float` accumulator → `double`

---

## 1. Why This Test Exists

### The Problem We Are Solving

Inside `master_gau`'s `reduce_kernel`, `SumOp<float>` used **Kahan compensated summation** — a well-known error-correction algorithm that adds a running compensation variable `c` to recover bits lost during floating-point addition.

The Kahan loop looked like this (simplified):

```cpp
AccumulatorT kahan_c   = 0;
AccumulatorT kahan_sum = init_val;

for (int64_t i = 0; i < reduced_count; ++i) {
    AccumulatorT val = static_cast<AccumulatorT>(input[i]);
    if (std::isinf(kahan_sum) || std::isnan(kahan_sum)) {
        kahan_sum += val;
    } else {
        AccumulatorT y = val - kahan_c;      // compensated value
        AccumulatorT t = kahan_sum + y;      // new sum (loses low bits)
        kahan_c = (t - kahan_sum) - y;       // recover lost bits
        kahan_sum = t;
    }
}
```

**The proposed replacement:** Delete Kahan entirely. Instead, add a single specialization in `AccumulatorTypeSelector<float>` so float tensors accumulate in `double` internally:

```cpp
template<> struct AccumulatorTypeSelector<float> { using type = double; };
```

**Why this is better than Kahan:**
- Double accumulator eliminates rounding error at source (ULP of double at 1e8 ≈ 1.5e-8, which is smaller than `float` 1.0 → error-free addition)
- No extra 4 FP ops + branch per element
- No serial dependency chain (Kahan's `kahan_c` update must wait for `t` → blocks CPU pipelining and SIMD vectorization)
- Fixes catastrophic cancellation cases that Kahan cannot handle (explained in Dataset 1 below)

**Before making any code changes**, we needed a full numeric and performance baseline to prove the change is strictly better — not just theoretically, but with actual library numbers on realistic deep-learning workloads.

---

## 2. Test Script Architecture

### File: `kahan_precision_test.cpp`

The script is self-contained (no test framework dependency). It links only against `libtensor.so` + standard library.

### Structure

```
main()
  ├─ run_test("1. Catastrophic Cancellation",  make_cancellation())
  ├─ run_test("2. Cross-Entropy Losses",        make_ce_losses())
  ├─ run_test("3. Gradient Tensor (1M)",        make_gradients())
  ├─ run_test("4. Softmax Output",              make_softmax())
  ├─ run_test("5. Batch Norm Activations",      make_activations())
  └─ run_test("6. Embedding Weights (786K)",    make_embedding_weights())
```

### `run_test()` — What Happens Per Dataset

For each dataset:

1. **Build ground truth** — `ref_double_sum()`: a raw C++ loop that casts every `float` to `double` before adding, completely independent of the library
2. **Build naive baseline** — `naive_float_sum()`: a raw C++ loop, float accumulator, zero compensation
3. **Build tensor** — wrap data into a `Tensor` object (not counted in timing)
4. **Warm-up** — call `reduce_sum(t)` 5 times (fills CPU caches, JIT/branch predictor warmup)
5. **Time library** — 30 measured calls to `reduce_sum(t)`, record each duration with `std::chrono::high_resolution_clock`
6. **Time naive loop** — 30 measured iterations of the raw float loop; use `volatile float naive_sink` to prevent GCC dead-code elimination
7. **Time double loop** — 30 measured iterations of the raw double reference loop; use `volatile double dbl_sink`
8. **Compute errors** — absolute and relative error of library and naive vs. double reference
9. **Print** — precision table + timing table with mean and min over 30 runs

### Compile Command

```bash
cd master_gau
g++ -Iinclude -I/usr/local/cuda/include -DWITH_CUDA -std=c++20 -fPIC -O2 -fopenmp \
    Tests/ReductionsTests/kahan_precision_test.cpp \
    -o kahan_precision_test \
    -L/usr/local/cuda/lib64 -Llib \
    -Xlinker -rpath -Xlinker '$ORIGIN/lib' \
    -ltensor -lcudart -ltbb -lcurand -lcublas
```

No full library rebuild needed — `libtensor.so` already exists.

---

## 3. The Six Datasets — What, Why, and How

We chose 6 datasets that collectively cover the full range of floating-point stress cases you encounter in real deep-learning workloads. Each one stresses a different property of a summation algorithm.

---

### Dataset 1 — Catastrophic Cancellation (N = 3001)

```cpp
std::vector<float> make_cancellation() {
    for (int i = 0; i < 1000; ++i) {
        v.push_back(1e8f);   // +100,000,000
        v.push_back(1.0f);   //  +1
        v.push_back(-1e8f);  // -100,000,000
    }
    v.push_back(1.0f);       // final +1
    // True sum = 1001.0
}
```

**What it models:**
This is a pure stress test for accumulation error. The pattern `[+1e8, 1, -1e8]` repeats 1000 times. The true sum is 1001.0. Every triple contributes exactly 1.0 to the total.

**Why this dataset exposes algorithm limits:**
This is the hardest possible case for float summation. The key fact is:

> `float` has ULP (unit in the last place) at 1e8 ≈ **8.0**

Which means `1e8 + 1.0 = 1e8` in float — the `1.0` is **entirely lost** in rounding. Then `1e8 - 1e8 = 0` exactly. So the net contribution of each triple is `0.0` in float, not `1.0`. The accumulated sum is `0.0 + 1.0 = 1.0` (the last element), not `1001.0`.

**Why Kahan also fails here:**
This is the key discovery from our analysis. Kahan is designed to recover bits lost when adding a *small* number to a *large* accumulated sum. But here, the *individual inputs* are magnitude-swinging, not the accumulator. When `kahan_sum = 1.0` and the next value is `1e8`:
- `y = 1e8 - kahan_c ≈ 1e8`
- `t = 1.0 + 1e8 = 1e8` (float, 1.0 is lost)
- `kahan_c = (1e8 - 1.0) - 1e8 = -1e8 - 1e8 = 0` in float — **compensation also lost**

Kahan's correction only works when the accumulated sum is large and a small correction is being added. It cannot compensate for the addition of a large number that completely buries the current sum. **Kahan and naive float produce identical error on this pattern.**

**Why double accumulator fixes it:**
Double ULP at 1e8 ≈ **1.5e-8**, which is far smaller than `1.0`. So `1e8 + 1.0` is representable in double with no rounding. The cancellation is handled exactly.

**Result:**
| Method | Result | Error |
|--------|--------|-------|
| True (double) | 1001.0 | — |
| Library (Kahan) | 1.0 | 1000.0 (rel 99.9%) |
| Naive float | 1.0 | 1000.0 (rel 99.9%) |

This is the strongest argument for the float→double change: Kahan provides **zero benefit** here and double **completely fixes** it.

---

### Dataset 2 — Cross-Entropy Losses (N = 65,536)

```cpp
std::uniform_real_distribution<float> dist(0.1f, 5.0f);
std::vector<float> v(65536);
```

**What it models:**
Cross-entropy loss values in a classification network. `N = 65536` = batch of 64 samples × 1024-class softmax, or a batch of 65536 tokens in a language model. Values in `[0.1, 5.0]` match real `-log(p)` loss values from a stable, not-yet-converged model.

**Why this stresses precision:**
65536 positive values summing to ~167195 is a classic "monotone accumulation" problem. Every addition lands in roughly the same magnitude band (0–5 gets added to a growing sum 0→167195), so the relative error of the working accumulator grows as `O(N × ε)` where ε = float machine epsilon = 1.2e-7.

**Why we chose N=65536:**
This is a power-of-2 that commonly appears in attention heads, batch sizes × vocab sizes, and feature maps. Large enough to show measurable error without hitting memory limits.

**Result:**
| Method | Rel Error |
|--------|-----------|
| Library (Kahan) | 1.20e-8 |
| Naive float | 4.31e-6 |
| **Kahan improvement** | **360×** |

---

### Dataset 3 — Gradient Tensor, 1 Million Values (N = 1,048,576)

```cpp
std::uniform_real_distribution<float> dist(-1e-4f, 1e-4f);
std::vector<float> v(1048576);
```

**What it models:**
Gradient values at the end of a backward pass in a large model. Gradients are typically very small (often clipped to `[-1e-4, 1e-4]` or smaller), both positive and negative, and nearly cancel each other out. The true sum is nearly zero (~0.118).

**Why this stresses precision:**
This is the worst-case for standard float summation because:
1. N is 1 million — the maximum error term is `N × ε ≈ 0.12` (float's maximum relative error = 100% of the true value)
2. Positive and negative values cancel, so the true sum is small. Accumulated rounding error can be **comparable to the true result**
3. This simulates `reduce_sum(gradient_tensor)` calls that happen millions of times during training

**Why we chose N=1M:**
GPT-2-small has 117M parameters. A single layer's gradient tensor is easily 1M+ floats. This is a "production size" test.

**Result:**
| Method | Rel Error |
|--------|-----------|
| Library (Kahan) | 1.67e-8 |
| Naive float | 1.58e-5 |
| **Kahan improvement** | **946×** |

---

### Dataset 4 — Softmax Output, All Equal Probabilities (N = 512)

```cpp
float val = 1.0f / 512.0f;
return std::vector<float>(512, val);
```

**What it models:**
A uniform softmax output — all probabilities are equal (1/512). The sum should be exactly 1.0. This appears in:
- Temperature softmax with very high temperature (approaches uniform)
- Sanity checks / normalization verification
- Attention weight normalization after softmax

**Why this is a control/baseline test:**
512 × (1/512) = 1.0 exactly, and 1/512 = 0.001953125 which **is exactly representable in float** (it's a power of two: 2⁻⁹). Every addition is exact. All methods — Kahan, naive float, and double — produce identical results.

**Why include a test where everything is perfect:**
To verify the algorithm doesn't *introduce* error for easy cases. If a method fails here, something is fundamentally broken. This is the "no regression" baseline.

**Result:**
| Method | Rel Error |
|--------|-----------|
| All | 0.0 |

All methods exact. Confirms correct behavior on trivial input.

---

### Dataset 5 — Batch Normalization Activations (N = 50,176)

```cpp
std::normal_distribution<float> dist(0.0f, 1.0f);
std::vector<float> v(224 * 224);
```

**What it models:**
`224 × 224 = 50176` is the exact spatial size of a standard ImageNet input feature map. Batch normalization computes `mean = reduce_sum(activations) / N` over this spatial extent. Values drawn from `N(0,1)` represent post-ReLU or pre-activation distributions.

**Why this stresses precision:**
Mixed positive and negative values (Gaussian, zero mean), large N. The true sum is close to zero by symmetry (double says 8.68), but individual values range roughly `[-3, +3]` causing moderate cancellation. This is "medium difficulty" — harder than monotone sums but easier than Dataset 1.

**Why N=50176 specifically:**
This is not a round number — it's `224²`, a real DL shape. We deliberately avoided only power-of-2 sizes to make the test realistic. Most tensors in practice are not perfectly aligned sizes.

**Result:**
| Method | Rel Error |
|--------|-----------|
| Library (Kahan) | 1.56e-7 |
| Naive float | 4.92e-5 |
| **Kahan improvement** | **316×** |

---

### Dataset 6 — Embedding Weight Matrix (N = 786,432)

```cpp
std::normal_distribution<float> dist(0.0f, 0.02f);
std::vector<float> v(1024 * 768);
```

**What it models:**
`1024 × 768 = 786432` is the exact weight matrix size for the embedding layer of BERT-base (vocab_size=30522 is larger, but token embedding dim=768, and this size is used in attention projection matrices). Standard deviation of 0.02 matches the typical initialization scale (truncated normal with std=0.02 is standard for transformer weights).

**Why this stresses precision:**
Very large N with very small values (std=0.02, so most values in `[-0.06, +0.06]`). Nearly-zero mean means massive cancellation. True sum is tiny (~22.0) compared to the magnitude of individual partial sums that can reach thousands before cancellation brings them back down.

**Why 786K specifically:**
This tests the "very large tensor" performance regime. GPU kernels typically hit this size in `nn.Linear` weight reduction. We need to know if library overhead scales reasonably with N at production model scale.

**Result:**
| Method | Rel Error |
|--------|-----------|
| Library (Kahan) | 1.46e-8 |
| Naive float | 2.24e-5 |
| **Kahan improvement** | **1534×** |

---

## 4. Metrics — What We Measured, Why, and How

### 4.1 Precision Metrics

#### Ground Truth: Double Reference Sum

```cpp
double ref_double_sum(const std::vector<float>& v) {
    double acc = 0.0;
    for (float x : v) acc += static_cast<double>(x);
    return acc;
}
```

**Why double is the ground truth:**
We cannot use an external math library's answer as truth — that would introduce another variable. Instead, we use the mathematical fact that: double precision (53-bit mantissa) accumulating float values (23-bit mantissa) has ULP errors that are ~10,000× smaller than float accumulation. For practical N (up to 10M), the double sum is indistinguishable from the true mathematical sum.

Formally: max absolute error of double sum = `N × ε_double × max|x_i|`. For N=1M, `ε_double=2.2e-16`, `max|x_i|=1e-4`: error ≤ `1e6 × 2.2e-16 × 1e-4 = 2.2e-14` — effectively zero for our purposes.

**Why not use 80-bit long double:**
Would require special build flags and is not portable to GPU code. Double is the universal standard in numerical libraries for this role.

#### Absolute Error

```cpp
double abs_err_lib = std::fabs((double)lib - ref);
```

Raw magnitude of deviation from ground truth. Useful when you need to know the actual units of error (e.g., "the loss sum was off by 0.7 — that's a meaningful difference per-sample if divided by N=65536").

#### Relative Error

```cpp
double rel_err_lib = (ref != 0.0) ? abs_err_lib / std::fabs(ref) : abs_err_lib;
```

Error as a fraction of the true result. This is the primary comparison metric because it's scale-independent. A relative error of 1e-7 means "7 significant digits correct", regardless of whether the sum is 1.0 or 167195.

**Why both:**
- Absolute error is important when the true result is near zero (rel error explodes)
- Relative error is important for comparing across datasets of different magnitudes
- We guard against division by zero: if `ref == 0`, we use abs error as rel error

---

### 4.2 Timing Metrics

#### Warmup Runs (5 iterations, not measured)

```cpp
static constexpr int WARMUP = 5;
for (int i = 0; i < WARMUP; ++i) {
    volatile auto r = reduce_sum(t);
    (void)r;
}
```

**Why warm up:**
- First call allocates internal buffers (temporary workspace for reduction)
- CPU branch predictor learns the control flow inside `reduce_kernel`
- L1/L2 cache fills with the input tensor data (most subsequent calls are cache-hot)
- Without warmup, the first timing sample would be a cold-cache outlier and skew the mean

The `volatile` on `r` prevents the compiler from skipping the call. Even though we discard the result, the actual computation must happen.

#### Measured Runs (30 iterations)

```cpp
static constexpr int ITERS = 30;
std::vector<double> lib_times(ITERS);
for (int i = 0; i < ITERS; ++i) {
    auto t0 = Clock::now();
    Tensor result = reduce_sum(t);
    auto t1 = Clock::now();
    lib_times[i] = us(t1 - t0).count();
    lib = result.data<float>()[0];  // force result to be used
}
```

**Why 30 iterations:**
- Enough to compute a stable mean and observe the minimum
- Not so many that OS scheduling effects compound (Linux time-slice = 1–4ms; for 30 × ~3ms = 90ms total, we stay under one typical OS scheduling window per test)
- Standard in microbenchmarking: enough to average away jitter without spending minutes

**Clock choice — `std::chrono::high_resolution_clock`:**
Nanosecond resolution on Linux (backed by `clock_gettime(CLOCK_REALTIME_COARSE)` or TSC depending on platform). Sufficient for timing operations in the 10µs–10ms range. Finer than this would require `rdtsc` directly.

#### Mean and Min

```cpp
auto mean = [](std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
};
auto minv = [](std::vector<double>& v) {
    return *std::min_element(v.begin(), v.end());
};
```

**Why mean:**
Average performance represents the steady-state throughput you'd see in production (where the operation runs repeatedly). Captures the "typical" behavior.

**Why min:**
The minimum over many runs represents the **best-case hardware performance** — a single clean run with no OS interruptions, no cache misses, no context switches. This is the closest measurement to "what the CPU can actually do" stripped of OS noise. In computer architecture benchmarking, min is often more meaningful than mean for comparing algorithms because it removes external variance. If algorithm A has a lower min than algorithm B, A is fundamentally faster.

#### Preventing Dead-Code Elimination (the `volatile` trick)

```cpp
volatile float  naive_sink = 0.0f;
volatile double dbl_sink   = 0.0;

// in naive loop:
for (int64_t j = 0; j < N; ++j) acc += raw[j];
naive_sink = acc;   // force the loop to actually run
```

**Why this matters:**
GCC `-O2` will eliminate any computation whose result is not "observable" (written to memory, returned, or used in I/O). A raw accumulation loop that assigns to a local variable `acc` with no further use is completely removed. Our first run of the test showed naive/double loops taking `0.01µs` for N=1M — clearly optimized away.

**The fix:** Write the final result to a `volatile` variable. `volatile` tells GCC "this write must happen, with observable side effects." GCC cannot prove the volatile store is unobservable, so it keeps the entire computation chain that produces the value.

This is the same technique used by Google Benchmark's `DoNotOptimize<T>()`.

#### Library Timing — No Dead-Code Issue

The library timing doesn't need this trick because `reduce_sum(t)` returns a `Tensor` object (heap allocation, destructor, reference counting). GCC cannot possibly eliminate it. And we access `result.data<float>()[0]` — another pointer dereference with external side effects.

#### Ratio Reporting

```
Library / Naive  : 7.55x  (>1 = lib slower, <1 = lib faster)
Library / Double : 7.45x
```

These ratios quantify the **library overhead** compared to a raw loop. They don't measure algorithmic efficiency — they measure the full cost of:
- Function call dispatch
- Tensor metadata checking
- Index calculation (unravel/ravel)
- Loop branching
- Output type conversion
- Any synchronization overhead

After removing Kahan, this ratio should decrease because the per-element work in the library is cheaper.

---

## 5. Full BEFORE Baseline Results

*Captured with Kahan active (before any code changes). Run date: 2026-03-18.*

### Precision

| Dataset | N | Lib Rel Error | Naive Rel Error | Kahan Improvement |
|---------|---|---------------|-----------------|-------------------|
| 1. Catastrophic Cancellation | 3,001 | **0.9990 (FAIL)** | 0.9990 (FAIL) | 1× (equal — both fail) |
| 2. Cross-Entropy Losses | 65,536 | 1.20e-8 | 4.31e-6 | **360×** |
| 3. Gradient Tensor (1M) | 1,048,576 | 1.67e-8 | 1.58e-5 | **946×** |
| 4. Softmax Uniform | 512 | 0.0 | 0.0 | Equal (trivial case) |
| 5. Batch Norm Activations | 50,176 | 1.56e-7 | 4.92e-5 | **316×** |
| 6. Embedding Weights | 786,432 | 1.46e-8 | 2.24e-5 | **1534×** |

**Key finding:** Kahan dramatically outperforms naive float on all normal DL workloads. But on catastrophic cancellation patterns (large magnitude swings), it provides zero benefit — identical error to naive float. **The double accumulator change will fix ALL of these**, including Dataset 1 which Kahan cannot fix.

### Library Timing (30-run mean, BEFORE changes)

| Dataset | N | Lib Mean (µs) | Lib Min (µs) | vs Naive | vs Double |
|---------|---|---------------|--------------|----------|-----------|
| 1. Catastrophic Cancellation | 3,001 | 30.76 | 10.85 | 32.6× | ~32× |
| 2. Cross-Entropy Losses | 65,536 | 184.32 | 175.79 | 7.55× | 7.45× |
| 3. Gradient Tensor (1M) | 1,048,576 | 2,887.22 | 2,803.24 | 7.38× | 7.45× |
| 4. Softmax Uniform | 512 | 17.85 | 9.75 | 63.3× | — |
| 5. Batch Norm Activations | 50,176 | 143.03 | 135.96 | 7.57× | — |
| 6. Embedding Weights | 786,432 | 2,166.30 | 2,068.74 | 3.97× | 6.23× |

**Key finding:** Library is 7–8× slower than a raw naive float loop for medium/large N. For very small N (512, 3001), overhead dominates and ratio is 30–63×. This overhead is from index calculation (unravel/ravel in `reduce_kernel`), not from Kahan itself — so removing Kahan alone won't close this gap. That gap is Phase 2 work (stride-based traversal).

---

## 6. What the AFTER Run Should Show

After making the two code changes:
1. `AccumulatorTypeSelector<float> → double`
2. Kahan branch deleted from `reduce_kernel`

Expected AFTER results:

| Dataset | Expected Rel Error (AFTER) | Reason |
|---------|---------------------------|--------|
| 1. Catastrophic Cancellation | **< 1e-12 (near zero)** | Double handles magnitude swings exactly |
| 2. Cross-Entropy Losses | **< 1e-10** (equal or better than Kahan) | Double accumulation is mathematically superior |
| 3. Gradient Tensor (1M) | **< 1e-10** | Same reason |
| 4. Softmax Uniform | 0.0 | Trivial case, unchanged |
| 5. Batch Norm Activations | **< 1e-10** | Same reason |
| 6. Embedding Weights | **< 1e-10** | Same reason |

Timing: Library should be **faster** because:
- 4 Kahan FP ops per element removed
- Branch on `isinf/isnan` removed (branch predictor no longer stressed)
- The serial dependency `kahan_c → y → t → kahan_c` chain removed, allowing CPU out-of-order execution to pipeline the accumulation
- The double accumulation itself is **not slower** on modern CPUs because double and float have the same latency on x86 FPU/SSE (both 4–5 cycles for fadd)

Expected timing improvement: 10–30% reduction in library mean time for large N datasets.

---

## 7. Design Decisions Explained

### Why Not Use Google Benchmark / Catch2?

We chose a standalone script over a framework for two reasons:
1. No framework is installed in the repo — adding a dependency just for this test is not worth it
2. We only need two statistical moments (mean, min) over 30 iterations — trivial to compute manually

The script produces the exact same information a benchmark framework would, without the setup overhead.

### Why Fix the Seed on Random Generators?

```cpp
std::mt19937 rng(42);  // fixed seed
```

Every run of the test generates **identical data** regardless of when it's run. This ensures:
- BEFORE and AFTER runs are operating on the same inputs
- Results are reproducible for other developers or CI
- No "lucky" or "unlucky" random draws inflate/deflate the error

### Why Both Mean and Min Timing?

- **Mean** = what the user experiences on average (production throughput)
- **Min** = theoretical hardware limit (algorithm speed without OS noise)

If mean decreases after changes but min stays same → we removed overhead but the kernel is unchanged.
If both decrease → the kernel itself is faster.

### Why Not Measure Median or Stddev?

For this microbenchmark with 30 runs at warm cache state, the distribution is typically bimodal: a tight cluster at the fast value, and occasional outliers from OS preemption. Median would be nearly identical to mean. Stddev would just quantify OS jitter, which is not what we're measuring. Min is more informative than any of these for algorithm comparison.

### Why 5 Warmup Runs Specifically?

- 1 warmup: buffer allocation happens, but cache may not be warm
- 3 warmup: usually sufficient, but TLB and branch predictor may still be cold
- 5 warmup: experimentally, most microbenchmarks stabilize within 3–5 repetitions; 5 is a safe margin
- 10+ warmup: unnecessary for a simple reduction with no JIT compilation

---

## 8. Summary Table — Test Design at a Glance

| Aspect | Decision | Reason |
|--------|----------|--------|
| Number of datasets | 6 | Cover all DL stress patterns without being exhaustive |
| Ground truth | Double reference loop | Mathematically provable accuracy, no external dependency |
| Precision metric | Relative error | Scale-independent, comparable across datasets |
| Timing tool | `std::chrono::high_resolution_clock` | nanosecond resolution, standard C++, cross-platform |
| Warmup | 5 runs | Cache/predictor warmup without over-spending time |
| Measured iterations | 30 | Stable statistics, under one OS scheduling window |
| Reported statistics | Mean + Min | Mean = average perf, Min = peak hardware performance |
| Dead-code prevention | `volatile` sink | Prevents GCC from eliminating raw loops at `-O2` |
| RNG seed | Fixed (42, 123, 999, etc.) | Reproducible, identical BEFORE/AFTER comparison |
| Framework | None (standalone) | No dependency overhead, fits in existing test directory |
