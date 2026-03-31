#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>
#include <omp.h>
#include "core/Tensor.h"
#include "ops/UnaryOps/Reduction.h"
using namespace OwnTensor;

template<typename F>
double bench(F fn, int w=10, int it=50) {
    for(int i=0;i<w;++i)fn();
    std::vector<double>t;
    for(int i=0;i<it;++i){auto s=std::chrono::high_resolution_clock::now();fn();
    auto e=std::chrono::high_resolution_clock::now();
    t.push_back(std::chrono::duration<double,std::micro>(e-s).count());}
    std::sort(t.begin(),t.end());return t[it/2];
}

int main() {
    std::mt19937 rng(42);
    printf("=== SUM vs MEAN timing breakdown ===\n\n");
    
    struct S { int64_t r,c; const char* lbl; };
    S shapes[] = {
        {32,768,"(32,768)"}, {256,4096,"(256,4096)"}, {4096,768,"(4096,768)"},
        {49152,128,"(49152,128)"}, {1000,10000,"(1000,10000)"}, {2048,50176,"(2048,50176)"},
    };
    
    printf("%15s %10s %10s %10s %10s\n", "Shape", "sum(μs)", "mean(μs)", "diff(μs)", "overhead%");
    printf("-------------------------------------------------------------------\n");
    
    for (auto& s : shapes) {
        if (s.r*s.c > 200000000LL) { printf("%15s SKIP\n", s.lbl); continue; }
        
        Tensor t({Shape({s.r, s.c})}, TensorOptions().with_dtype(Dtype::Float32));
        float* d = t.data<float>();
        std::normal_distribution<float> n(0,1);
        for(size_t i=0;i<t.numel();++i) d[i]=n(rng);
        
        double sum_us = bench([&]() { return reduce_sum(t, {1}); });
        double mean_us = bench([&]() { return reduce_mean(t, {1}); });
        double diff = mean_us - sum_us;
        double pct = (diff / sum_us) * 100;
        
        printf("%15s %10.0f %10.0f %10.0f %9.1f%%\n", s.lbl, sum_us, mean_us, diff, pct);
    }
    
    printf("\n=== If mean ≈ sum, then the sum part is the bottleneck ===\n");
    printf("=== If mean >> sum, then division/overhead is the bottleneck ===\n");
    return 0;
}
