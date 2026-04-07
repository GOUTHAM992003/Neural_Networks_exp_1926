// Dummy typedef to bypass the header error introduced by the new GPU signatures
struct CUstream_st; typedef struct CUstream_st* cudaStream_t;

#include <chrono>
#include <cstdio>
#include <vector>
#include <algorithm>
#include "core/Tensor.h"
#include "ops/UnaryOps/Reduction.h"

using namespace OwnTensor;

int main() {
    const int NUM_ROWS = 1000000;
    
    printf("Actual master_gau Outer Reduction Benchmark: [1000000, C] -> [1, C]\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("%-15s %-20s\n", "Output Slots", "master_gau Actual (ms)");
    printf("--------------------------------------------------------------------------------\n");

    for (int c = 28; c >= 1; --c) {
        Tensor t({Shape({NUM_ROWS, c})}, TensorOptions().with_dtype(Dtype::Float32));
        float* d = t.data<float>();
        for(int i=0; i<NUM_ROWS*c; ++i) d[i] = 1.0f;

        // Warmup
        for (int i = 0; i < 5; ++i) {
            Tensor out = reduce_sum(t, {0});
        }
        
        int iters = 10;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            Tensor out = reduce_sum(t, {0});
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
        printf("%-15d %-20.2f\n", c, time_ms);
    }
    return 0;
}
