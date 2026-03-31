// ═══════════════════════════════════════════════════════════════════════════
// COMPREHENSIVE BENCHMARK: sum / nansum / mean / nanmean
// master_gau (Our Library)
//
// Mirrors Tests/pytorch_sum_mean_bench.py exactly — same shapes, dtypes,
// reduction dims, sizes, DL apps, and NaN% sweep.
//
// Build & Run:
//   make run-snippet FILE=Tests/our_sum_mean_bench.cpp
// ═══════════════════════════════════════════════════════════════════════════

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>
#include <algorithm>
#include <string>
#include <omp.h>
#include "core/Tensor.h"
#include "ops/UnaryOps/Reduction.h"

using namespace OwnTensor;

// ─── Benchmark helper: median of sorted timings ─────────────────
template<typename F>
double bench(F fn, int warmup=10, int iters=50) {
    for (int i = 0; i < warmup; ++i) fn();
    std::vector<double> t;
    for (int i = 0; i < iters; ++i) {
        auto s = std::chrono::high_resolution_clock::now();
        fn();
        auto e = std::chrono::high_resolution_clock::now();
        t.push_back(std::chrono::duration<double, std::micro>(e - s).count());
    }
    std::sort(t.begin(), t.end());
    return t[iters / 2];  // median
}

// ─── Data fill helpers ──────────────────────────────────────────
void fill_f32(Tensor& t, std::mt19937& r) {
    float* d = t.data<float>();
    std::normal_distribution<float> n(0, 1);
    for (size_t i = 0; i < t.numel(); ++i) d[i] = n(r);
}
void fill_f64(Tensor& t, std::mt19937& r) {
    double* d = t.data<double>();
    std::normal_distribution<double> n(0, 1);
    for (size_t i = 0; i < t.numel(); ++i) d[i] = n(r);
}
void fill_i32(Tensor& t, std::mt19937& r) {
    int32_t* d = t.data<int32_t>();
    std::uniform_int_distribution<int32_t> u(-100, 100);
    for (size_t i = 0; i < t.numel(); ++i) d[i] = u(r);
}
void fill_i64(Tensor& t, std::mt19937& r) {
    int64_t* d = t.data<int64_t>();
    std::uniform_int_distribution<int64_t> u(-100, 100);
    for (size_t i = 0; i < t.numel(); ++i) d[i] = u(r);
}
void inject_nans_f32(Tensor& t, double pct, std::mt19937& r) {
    float* d = t.data<float>();
    std::uniform_real_distribution<float> u(0, 1);
    for (size_t i = 0; i < t.numel(); ++i) {
        if (u(r) < pct) d[i] = std::nanf("");
    }
}
void inject_nans_f64(Tensor& t, double pct, std::mt19937& r) {
    double* d = t.data<double>();
    std::uniform_real_distribution<double> u(0, 1);
    for (size_t i = 0; i < t.numel(); ++i) {
        if (u(r) < pct) d[i] = std::nan("");
    }
}

std::string fmt_shape(const std::vector<int64_t>& s) {
    std::string r = "(";
    for (size_t i = 0; i < s.size(); ++i) { if (i) r += ","; r += std::to_string(s[i]); }
    return r + ")";
}
std::string fmt_dims(const std::vector<int64_t>& d) {
    if (d.empty()) return "None";
    std::string r = "(";
    for (size_t i = 0; i < d.size(); ++i) { if (i) r += ","; r += std::to_string(d[i]); }
    return r + ")";
}

#define W 110
#define SEP "=============================================================================================================="

int main() {
    printf("%s\n", SEP);
    printf("  OUR LIBRARY BENCHMARK: sum / nansum / mean / nanmean\n");
    printf("  CPU: %d threads\n", omp_get_max_threads());
    printf("%s\n", SEP);
    std::mt19937 rng(42);

    // ═══════════════════════════════════════════════════════════════
    // SECTION 1: INNERCONTIGUOUS (reduce last dim)
    // ═══════════════════════════════════════════════════════════════
    {
        printf("\n%.*s\n  SECTION 1: INNERCONTIGUOUS (reduce last dim)\n%.*s\n", W, SEP, W, SEP);
        struct SC { int64_t r, c; const char* lbl; };
        SC shapes[] = {
            {8,16,"tiny"},{32,64,"tiny"},{10,100,"small"},
            {32,768,"LayerNorm"},{32,4096,"large-LN"},{256,4096,"batch-feat"},
            {4096,768,"Seq-LN"},{1000,10000,"medium"},
            {2048,50176,"spatial"},{49152,128,"many-out"},
            {10,1000000,"wide"},{1000000,10,"tall"},
        };
        printf("%22s %12s %6s | %10s %10s %10s %10s\n",
               "Shape","Label","Dtype","sum","nansum","mean","nanmean");
        printf("%s\n", std::string(95, '-').c_str());

        for (auto& s : shapes) {
            if (s.r * s.c > 200000000LL) continue;
            // fp32
            {
                Tensor t({Shape({s.r, s.c})}, TensorOptions().with_dtype(Dtype::Float32));
                fill_f32(t, rng);
                Tensor tn({Shape({s.r, s.c})}, TensorOptions().with_dtype(Dtype::Float32));
                fill_f32(tn, rng); inject_nans_f32(tn, 0.1, rng);

                double t_sum = bench([&](){return reduce_sum(t, {-1});});
                double t_nsum = bench([&](){return reduce_nansum(tn, {-1});});
                double t_mean = bench([&](){return reduce_mean(t, {-1});});
                double t_nmean = bench([&](){return reduce_nanmean(tn, {-1});});
                printf("  %20s %12s %6s | %9.0fμ %9.0fμ %9.0fμ %9.0fμ\n",
                       fmt_shape({s.r,s.c}).c_str(), s.lbl, "fp32",
                       t_sum, t_nsum, t_mean, t_nmean);
            }
            // fp64
            {
                Tensor t({Shape({s.r, s.c})}, TensorOptions().with_dtype(Dtype::Float64));
                fill_f64(t, rng);
                Tensor tn({Shape({s.r, s.c})}, TensorOptions().with_dtype(Dtype::Float64));
                fill_f64(tn, rng); inject_nans_f64(tn, 0.1, rng);

                double t_sum = bench([&](){return reduce_sum(t, {-1});});
                double t_nsum = bench([&](){return reduce_nansum(tn, {-1});});
                double t_mean = bench([&](){return reduce_mean(t, {-1});});
                double t_nmean = bench([&](){return reduce_nanmean(tn, {-1});});
                printf("  %20s %12s %6s | %9.0fμ %9.0fμ %9.0fμ %9.0fμ\n",
                       fmt_shape({s.r,s.c}).c_str(), s.lbl, "fp64",
                       t_sum, t_nsum, t_mean, t_nmean);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // SECTION 2: OUTERCONTIGUOUS (reduce first dim)
    // ═══════════════════════════════════════════════════════════════
    {
        printf("\n%.*s\n  SECTION 2: OUTERCONTIGUOUS (reduce first dim)\n%.*s\n", W, SEP, W, SEP);
        struct SC { int64_t r, c; const char* lbl; };
        SC shapes[] = {
            {16,64,"tiny"},{100,256,"small"},{1000,256,"medium"},
            {10000,100,"tall"},{100,10000,"wide"},{50176,2048,"spatial"},
        };
        printf("%22s %12s %6s | %10s %10s %10s %10s\n",
               "Shape","Label","Dtype","sum","nansum","mean","nanmean");
        printf("%s\n", std::string(95, '-').c_str());

        for (auto& s : shapes) {
            if (s.r * s.c > 200000000LL) continue;
            for (auto dtype : {Dtype::Float32, Dtype::Float64}) {
                const char* dt = dtype == Dtype::Float32 ? "fp32" : "fp64";
                Tensor t({Shape({s.r, s.c})}, TensorOptions().with_dtype(dtype));
                Tensor tn({Shape({s.r, s.c})}, TensorOptions().with_dtype(dtype));
                if (dtype == Dtype::Float32) { fill_f32(t, rng); fill_f32(tn, rng); inject_nans_f32(tn, 0.1, rng); }
                else { fill_f64(t, rng); fill_f64(tn, rng); inject_nans_f64(tn, 0.1, rng); }

                double t_sum = bench([&](){return reduce_sum(t, {0});});
                double t_nsum = bench([&](){return reduce_nansum(tn, {0});});
                double t_mean = bench([&](){return reduce_mean(t, {0});});
                double t_nmean = bench([&](){return reduce_nanmean(tn, {0});});
                printf("  %20s %12s %6s | %9.0fμ %9.0fμ %9.0fμ %9.0fμ\n",
                       fmt_shape({s.r,s.c}).c_str(), s.lbl, dt,
                       t_sum, t_nsum, t_mean, t_nmean);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // SECTION 3: GENERIC (reduce mixed/non-contiguous dims)
    // ═══════════════════════════════════════════════════════════════
    {
        printf("\n%.*s\n  SECTION 3: GENERIC (reduce mixed/non-contiguous dims)\n%.*s\n", W, SEP, W, SEP);
        struct GC { std::vector<int64_t> shape; std::vector<int64_t> dims; const char* lbl; };
        GC configs[] = {
            {{100,200,50},{0,2},"3D-XZ"},
            {{100,200,50},{0},"3D-X"},
            {{100,200,50},{1,2},"3D-YZ"},
            {{32,128,768},{0,2},"transformer"},
            {{32,128,768},{1,2},"batch"},
            {{16,64,32,32},{0,2},"4D-XH"},
            {{16,64,32,32},{2,3},"4D-spatial"},
        };
        printf("%22s %12s %12s %6s | %10s %10s %10s\n",
               "Shape","Dims","Label","Dtype","sum","mean","nanmean");
        printf("%s\n", std::string(95, '-').c_str());

        for (auto& c : configs) {
            for (auto dtype : {Dtype::Float32, Dtype::Float64}) {
                const char* dt = dtype == Dtype::Float32 ? "fp32" : "fp64";
                Tensor t({Shape(c.shape)}, TensorOptions().with_dtype(dtype));
                Tensor tn({Shape(c.shape)}, TensorOptions().with_dtype(dtype));
                if (dtype == Dtype::Float32) { fill_f32(t, rng); fill_f32(tn, rng); inject_nans_f32(tn, 0.1, rng); }
                else { fill_f64(t, rng); fill_f64(tn, rng); inject_nans_f64(tn, 0.1, rng); }

                double t_sum = bench([&](){return reduce_sum(t, c.dims);});
                double t_mean = bench([&](){return reduce_mean(t, c.dims);});
                double t_nmean = bench([&](){return reduce_nanmean(tn, c.dims);});
                printf("  %20s %12s %12s %6s | %9.0fμ %9.0fμ %9.0fμ\n",
                       fmt_shape(c.shape).c_str(), fmt_dims(c.dims).c_str(), c.lbl, dt,
                       t_sum, t_mean, t_nmean);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // SECTION 4: FULL REDUCTION (all dims)
    // ═══════════════════════════════════════════════════════════════
    {
        printf("\n%.*s\n  SECTION 4: FULL REDUCTION (reduce all dims)\n%.*s\n", W, SEP, W, SEP);
        printf("%12s %6s | %10s %10s %10s %10s\n", "Size", "Dtype", "sum", "nansum", "mean", "nanmean");
        printf("%s\n", std::string(70, '-').c_str());

        for (int64_t sz : {100LL, 1000LL, 10000LL, 100000LL, 1000000LL, 10000000LL, 50000000LL}) {
            for (auto dtype : {Dtype::Float32, Dtype::Float64}) {
                const char* dt = dtype == Dtype::Float32 ? "fp32" : "fp64";
                Tensor t({Shape({sz})}, TensorOptions().with_dtype(dtype));
                Tensor tn({Shape({sz})}, TensorOptions().with_dtype(dtype));
                if (dtype == Dtype::Float32) { fill_f32(t, rng); fill_f32(tn, rng); inject_nans_f32(tn, 0.1, rng); }
                else { fill_f64(t, rng); fill_f64(tn, rng); inject_nans_f64(tn, 0.1, rng); }

                double t_sum = bench([&](){return reduce_sum(t);});
                double t_nsum = bench([&](){return reduce_nansum(tn);});
                double t_mean = bench([&](){return reduce_mean(t);});
                double t_nmean = bench([&](){return reduce_nanmean(tn);});
                printf("  %10lld %6s | %9.0fμ %9.0fμ %9.0fμ %9.0fμ\n",
                       (long long)sz, dt, t_sum, t_nsum, t_mean, t_nmean);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // SECTION 5: DL APPLICATION SHAPES (fp32)
    // ═══════════════════════════════════════════════════════════════
    {
        printf("\n%.*s\n  SECTION 5: DEEP LEARNING APPLICATION SHAPES (fp32)\n%.*s\n", W, SEP, W, SEP);
        struct DL { const char* name; std::vector<int64_t> shape; std::vector<int64_t> dims; };
        DL apps[] = {
            {"LayerNorm-small", {32,768}, {-1}},
            {"LayerNorm-large", {32,4096}, {-1}},
            {"BatchNorm-2D", {32,256,14,14}, {0,2,3}},
            {"BatchNorm-1D", {32,256,100}, {0,2}},
            {"Attention-QK", {32,12,128,128}, {-1}},
            {"Attn-head-mean", {32,12,128,128}, {1}},
            {"Seq-LayerNorm", {32,512,768}, {-1}},
            {"Feature-mean", {64,2048}, {0}},
            {"Spatial-pool", {32,512,7,7}, {2,3}},
            {"Global-avg-pool", {32,2048,7,7}, {2,3}},
            {"Loss-reduction", {32,10000}, {-1}},
            {"Loss-full", {32,10000}, {}},
            {"Token-mean", {32,512,768}, {1}},
            {"Channel-mean", {32,256,56,56}, {1}},
            {"Embedding-mean", {32,128,300}, {-1}},
            {"BERT-pool", {16,512,1024}, {1}},
            {"ViT-patch", {32,197,768}, {1}},
            {"ResNet-feat", {64,2048}, {-1}},
        };
        printf("%22s %25s %12s | %10s %10s %10s\n",
               "Application","Shape","Dim","sum","mean","nanmean");
        printf("%s\n", std::string(100, '-').c_str());

        for (auto& a : apps) {
            Tensor t({Shape(a.shape)}, TensorOptions().with_dtype(Dtype::Float32));
            fill_f32(t, rng);
            Tensor tn({Shape(a.shape)}, TensorOptions().with_dtype(Dtype::Float32));
            fill_f32(tn, rng); inject_nans_f32(tn, 0.1, rng);

            double t_sum, t_mean, t_nmean;
            if (a.dims.empty()) {
                t_sum = bench([&](){return reduce_sum(t);});
                t_mean = bench([&](){return reduce_mean(t);});
                t_nmean = bench([&](){return reduce_nanmean(tn);});
            } else {
                t_sum = bench([&](){return reduce_sum(t, a.dims);});
                t_mean = bench([&](){return reduce_mean(t, a.dims);});
                t_nmean = bench([&](){return reduce_nanmean(tn, a.dims);});
            }
            printf("  %20s %25s %12s | %9.0fμ %9.0fμ %9.0fμ\n",
                   a.name, fmt_shape(a.shape).c_str(), fmt_dims(a.dims).c_str(),
                   t_sum, t_mean, t_nmean);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // SECTION 6: INTEGER DTYPES (sum only — we support int mean too!)
    // ═══════════════════════════════════════════════════════════════
    {
        printf("\n%.*s\n  SECTION 6: INTEGER DTYPES (sum + mean)\n%.*s\n", W, SEP, W, SEP);
        printf("%12s %8s | %10s %10s %10s\n", "Size", "Dtype", "sum-full", "sum-last", "mean-full");
        printf("%s\n", std::string(60, '-').c_str());

        for (int64_t sz : {10000LL, 100000LL, 1000000LL, 10000000LL}) {
            // int32
            {
                Tensor t({Shape({sz})}, TensorOptions().with_dtype(Dtype::Int32));
                fill_i32(t, rng);
                int64_t rows = (sz / 100 > 1) ? (sz / 100) : 1;
                Tensor t2({Shape({rows, 100})}, TensorOptions().with_dtype(Dtype::Int32));
                fill_i32(t2, rng);

                double t_full = bench([&](){return reduce_sum(t);});
                double t_last = bench([&](){return reduce_sum(t2, {-1});});
                double t_mean = bench([&](){return reduce_mean(t);});
                printf("  %10lld %8s | %9.0fμ %9.0fμ %9.0fμ\n",
                       (long long)sz, "int32", t_full, t_last, t_mean);
            }
            // int64
            {
                Tensor t({Shape({sz})}, TensorOptions().with_dtype(Dtype::Int64));
                fill_i64(t, rng);
                int64_t rows = (sz / 100 > 1) ? (sz / 100) : 1;
                Tensor t2({Shape({rows, 100})}, TensorOptions().with_dtype(Dtype::Int64));
                fill_i64(t2, rng);

                double t_full = bench([&](){return reduce_sum(t);});
                double t_last = bench([&](){return reduce_sum(t2, {-1});});
                double t_mean = bench([&](){return reduce_mean(t);});
                printf("  %10lld %8s | %9.0fμ %9.0fμ %9.0fμ\n",
                       (long long)sz, "int64", t_full, t_last, t_mean);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // SECTION 7: NaN% SWEEP (nanmean sensitivity)
    // ═══════════════════════════════════════════════════════════════
    {
        printf("\n%.*s\n  SECTION 7: NaN PERCENTAGE SWEEP — nanmean (1000,10000) fp32\n%.*s\n", W, SEP, W, SEP);
        printf("%8s | %10s %10s\n", "NaN%", "nanmean", "nansum");
        printf("%s\n", std::string(35, '-').c_str());

        for (double pct : {0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99}) {
            Tensor t({Shape({1000, 10000})}, TensorOptions().with_dtype(Dtype::Float32));
            fill_f32(t, rng);
            inject_nans_f32(t, pct, rng);
            double t_nm = bench([&](){return reduce_nanmean(t, {-1});});
            double t_ns = bench([&](){return reduce_nansum(t, {-1});});
            printf("  %5.1f%% | %9.0fμ %9.0fμ\n", pct*100, t_nm, t_ns);
        }
    }

    printf("\n%s\n  OUR LIBRARY BENCHMARK COMPLETE\n%s\n", SEP, SEP);
    return 0;
}
