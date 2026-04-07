#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <numeric>
#include <iomanip>
#include <immintrin.h>

// Simulated Strategy 1: PyTorch (ParallelSlices + Vertical SIMD)
// If output_slots < threads, threads sit idle. But it uses AVX2 for vertical sums.
void strategy_1_pytorch(const float* input, float* output, int num_rows, int num_cols, int max_threads) {
    // Zero output
    for(int o = 0; o < num_cols; ++o) output[o] = 0.0f;

    // Parallelize over output slots. 
    // If num_cols (e.g. 15) < max_threads (e.g. 28), only 15 threads work. 13 sit idle.
    #pragma omp parallel for num_threads(max_threads)
    for (int o = 0; o < (num_cols / 8) * 8; o += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (int r = 0; r < num_rows; ++r) {
            __m256 v = _mm256_loadu_ps(input + r * num_cols + o);
            acc = _mm256_add_ps(acc, v);
        }
        _mm256_storeu_ps(output + o, acc);
    }

    // Leftover columns handled sequentially per column
    #pragma omp parallel for num_threads(max_threads)
    for (int o = (num_cols / 8) * 8; o < num_cols; ++o) {
        float sum = 0.0f;
        for (int r = 0; r < num_rows; ++r) {
            sum += input[r * num_cols + o];
        }
        output[o] = sum;
    }
}

// Simulated Strategy 2: master_gau fallback (Split Reduction)
// All threads are active regardless of output_slots. Breaking Vertical SIMD.
void strategy_2_master_gau(const float* input, float* output, int num_rows, int num_cols, int max_threads) {
    for (int o = 0; o < num_cols; ++o) {
        // Output slot 'o' is being cooperatively calculated by all threads!
        std::vector<float> thread_accs(max_threads, 0.0f);

        #pragma omp parallel num_threads(max_threads)
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            
            // Chunk the rows
            int chunk = (num_rows + nt - 1) / nt;
            int begin = tid * chunk;
            int end = std::min(begin + chunk, num_rows);
            
            float local = 0.0f;
            // Scalar fallback loop with massive cache jumps (num_cols jump)
            for (int r = begin; r < end; ++r) {
                local += input[r * num_cols + o];
            }
            thread_accs[tid] = local;
        }

        float final_acc = 0.0f;
        for (int t = 0; t < max_threads; ++t) {
            final_acc += thread_accs[t];
        }
        output[o] = final_acc;
    }
}

int main() {
    const int NUM_ROWS = 1000000;
    const int MAX_THREADS = 28;
    
    std::cout << "Outer Reduction Benchmark: [1000000, C] -> [1, C]" << std::endl;
    std::cout << "Matrix size: " << NUM_ROWS << " rows. Threads available: " << MAX_THREADS << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(15) << "Output Slots"
              << std::setw(20) << "PyTorch (AVX+Idle)" 
              << std::setw(25) << "master_gau (Scalar+Thrashing)" 
              << "Speedup" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (int num_cols = MAX_THREADS; num_cols >= 1; --num_cols) {
        std::vector<float> input(NUM_ROWS * num_cols, 1.0f);
        std::vector<float> out_pytorch(num_cols, 0.0f);
        std::vector<float> out_master(num_cols, 0.0f);

        // Warmup
        strategy_1_pytorch(input.data(), out_pytorch.data(), NUM_ROWS, num_cols, MAX_THREADS);
        strategy_2_master_gau(input.data(), out_master.data(), NUM_ROWS, num_cols, MAX_THREADS);

        // Benchmark PyTorch (Strategy 1)
        auto t1 = std::chrono::high_resolution_clock::now();
        int iters = 10;
        for (int i = 0; i < iters; ++i) {
            strategy_1_pytorch(input.data(), out_pytorch.data(), NUM_ROWS, num_cols, MAX_THREADS);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        double time_pytorch = std::chrono::duration<double, std::milli>(t2 - t1).count() / iters;

        // Benchmark master_gau (Strategy 2)
        auto t3 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            strategy_2_master_gau(input.data(), out_master.data(), NUM_ROWS, num_cols, MAX_THREADS);
        }
        auto t4 = std::chrono::high_resolution_clock::now();
        double time_master = std::chrono::duration<double, std::milli>(t4 - t3).count() / iters;

        double speedup = time_master / time_pytorch;

        std::cout << std::left << std::setw(15) << num_cols
                  << std::setw(20) << time_pytorch
                  << std::setw(25) << time_master
                  << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    }

    return 0;
}
