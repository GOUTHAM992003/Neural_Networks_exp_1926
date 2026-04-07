#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <algorithm>
#include <iomanip>
#include <immintrin.h>
#include <random>

void strat1(const float* in, float* out, int R, int C, int nt) {
    #pragma omp parallel for num_threads(nt)
    for (int o = 0; o < (C/8)*8; o += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (int r = 0; r < R; ++r)
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(in + r*C + o));
        _mm256_storeu_ps(out + o, acc);
    }
    #pragma omp parallel for num_threads(nt)
    for (int o = (C/8)*8; o < C; ++o) {
        float s = 0; for (int r = 0; r < R; ++r) s += in[r*C+o]; out[o] = s;
    }
}

void strat2(const float* in, float* out, int R, int C, int nt) {
    for (int o = 0; o < C; ++o) {
        std::vector<float> ta(nt, 0.0f);
        #pragma omp parallel num_threads(nt)
        {
            int t=omp_get_thread_num(), n=omp_get_num_threads();
            int ch=(R+n-1)/n, b=t*ch, e=std::min(b+ch,R);
            float l=0; for(int r=b;r<e;++r) l+=in[r*C+o]; ta[t]=l;
        }
        float s=0; for(int t=0;t<nt;++t) s+=ta[t]; out[o]=s;
    }
}

double bench(auto fn, int w=3, int it=10) {
    for(int i=0;i<w;++i) fn();
    std::vector<double> t;
    for(int i=0;i<it;++i) {
        auto s=std::chrono::high_resolution_clock::now(); fn();
        auto e=std::chrono::high_resolution_clock::now();
        t.push_back(std::chrono::duration<double,std::milli>(e-s).count());
    }
    std::sort(t.begin(),t.end()); return t[it/2];
}

int main() {
    const int NT = 28;
    std::mt19937 rng(42);
    
    printf("Outer Reduction: Strategy 1 (SIMD+idle) vs Strategy 2 (scalar+all threads)\n");
    printf("Threads: %d\n\n", NT);
    
    // Test across different reduction sizes
    for (int R : {1000, 10000, 100000, 1000000}) {
        printf("=== %d reduction rows ===\n", R);
        printf("%6s %12s %12s %10s %8s\n", "Cols", "S1(ms)", "S2(ms)", "Winner", "Ratio");
        printf("--------------------------------------------------------------\n");
        
        for (int C : {1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 64, 128, 256}) {
            std::vector<float> data(R*C), out1(C), out2(C);
            std::uniform_real_distribution<float> dist(-1,1);
            for(auto&v:data) v=dist(rng);
            
            double t1 = bench([&](){ strat1(data.data(),out1.data(),R,C,NT); });
            double t2 = bench([&](){ strat2(data.data(),out2.data(),R,C,NT); });
            
            const char* winner = (t1 < t2) ? "S1" : "S2";
            double ratio = (t1 < t2) ? t2/t1 : t1/t2;
            printf("%6d %10.3fms %10.3fms %10s %7.2fx\n", C, t1, t2, winner, ratio);
        }
        printf("\n");
    }
    return 0;
}
