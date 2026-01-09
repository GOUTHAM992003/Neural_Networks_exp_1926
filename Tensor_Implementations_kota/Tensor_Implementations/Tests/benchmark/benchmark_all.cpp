#include "TensorLib.h"
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

using namespace OwnTensor;

// Timing helper structure
struct OpTiming {
    std::string name;
    std::vector<double> times_ms;
    double mean_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double std_ms = 0.0;
};

// Global timing and result storage
std::map<std::string, OpTiming> all_timings;
std::map<std::string, Tensor> all_results;
const int NUM_RUNS = 50;
const int WARMUP_RUNS = 5;

// Benchmark macro
#define BENCH_OP(op_name, expr) \
    [&]() { \
        for (int _w = 0; _w < WARMUP_RUNS; _w++) { auto _tmp = expr; } \
        OpTiming timing; \
        timing.name = std::string(op_name); \
        for (int _i = 0; _i < NUM_RUNS; _i++) { \
            auto _start = std::chrono::high_resolution_clock::now(); \
            auto _result = expr; \
            auto _end = std::chrono::high_resolution_clock::now(); \
            timing.times_ms.push_back(std::chrono::duration<double, std::milli>(_end - _start).count()); \
        } \
        double _sum = 0.0; \
        timing.min_ms = timing.times_ms[0]; \
        timing.max_ms = timing.times_ms[0]; \
        for (double t : timing.times_ms) { \
            _sum += t; \
            timing.min_ms = std::min(timing.min_ms, t); \
            timing.max_ms = std::max(timing.max_ms, t); \
        } \
        timing.mean_ms = _sum / timing.times_ms.size(); \
        double _sq_sum = 0.0; \
        for (double t : timing.times_ms) { \
            _sq_sum += (t - timing.mean_ms) * (t - timing.mean_ms); \
        } \
        timing.std_ms = std::sqrt(_sq_sum / timing.times_ms.size()); \
        all_timings[std::string(op_name)] = timing; \
        auto _final_res = expr; \
        all_results[std::string(op_name)] = _final_res; \
        return _final_res; \
    }()


// Helper to get float value from tensor safely
float get_float_value(const Tensor& t, int64_t idx) {
    switch (t.dtype()) {
        case Dtype::Float32: return t.data<float>()[idx];
        case Dtype::Float64: return static_cast<float>(t.data<double>()[idx]);
        case Dtype::Int32:   return static_cast<float>(t.data<int32_t>()[idx]);
        case Dtype::Int64:   return static_cast<float>(t.data<int64_t>()[idx]);
        default:             return 0.0f;
    }
}

// Helper to read CSV inputs manually
std::pair<Tensor, Tensor> read_inputs_from_csv(const std::string& filename, int64_t D, int64_t R, int64_t C, const TensorOptions& opts) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open input CSV: " + filename);
    
    std::string line;
    // Skip header
    while (std::getline(file, line)) {
        if (line.find("index") != std::string::npos) break;
    }
    
    int64_t size = D * R * C;
    std::vector<float> data_a;
    std::vector<float> data_b;
    data_a.reserve(size);
    data_b.reserve(size);
    
    int64_t idx = 0;
    while (std::getline(file, line) && idx < size) {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(ss, token, ',')) tokens.push_back(token);
        
        if (tokens.size() >= 6) {
            data_a.push_back(std::stof(tokens[4]));
            data_b.push_back(std::stof(tokens[5]));
            idx++;
        }
    }
    
    if (idx != size) {
        throw std::runtime_error("CSV element count (" + std::to_string(idx) + 
                               ") does not match target size (" + std::to_string(size) + ")");
    }
    
    Tensor t_a = Tensor::zeros({{D, R, C}}, opts);
    t_a.set_data(data_a.data(), size);
    
    Tensor t_b = Tensor::zeros({{D, R, C}}, opts);
    t_b.set_data(data_b.data(), size);
    return {t_a, t_b};
}

int main(int argc, char* argv[]) {
    std::cout << "╔════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  TensorLib 3D Comprehensive Benchmark (CPU)    ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════╝\n" << std::endl;
    
    std::string input_csv_path = "/home/blu-bridge016/Downloads/test_env_gau/benchmark_results/inputs/benchmark_all_3d_inputs.csv";
    if (argc > 1) input_csv_path = argv[1];

    TensorOptions opts;
    opts.dtype = Dtype::Float32;
    opts.device = DeviceIndex(Device::CPU);
    
    const int64_t D = 10;
    const int64_t R = 10;
    const int64_t C = 10;
    const int64_t size = D * R * C;
    
    std::cout << "Reading inputs from: " << input_csv_path << std::endl;
    std::cout << "Target dimensions: [" << D << ", " << R << ", " << C << "]" << std::endl;

    Tensor a, b;
    try {
        auto inputs = read_inputs_from_csv(input_csv_path, D, R, C, opts);
        a = inputs.first;
        b = inputs.second;
        std::cout << " Inputs loaded successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << " Error reading inputs: " << e.what() << std::endl;
        std::cerr << "   Generating random inputs with seed 1926..." << std::endl;
        
        // Set seed for reproducibility (same as Python benchmarks)
        srand(1926);
        
        a = Tensor::rand({{D, R, C}}, opts);
        b = Tensor::rand({{D, R, C}}, opts);
        
        // Save ALL generated inputs so other libraries can use the exact same data
        std::cout << "   Saving generated inputs to: " << input_csv_path << std::endl;
        std::ofstream save_inputs(input_csv_path);
        if (save_inputs.is_open()) {
            save_inputs << std::fixed << std::setprecision(12);
            save_inputs << "index,depth,row,col,input_a,input_b\n";
            // Save ALL elements (no limit for input file)
            for (int64_t i = 0; i < size; ++i) {
                int64_t d = i / (R * C);
                int64_t r = (i / C) % R;
                int64_t c = i % C;
                save_inputs << i << "," << d << "," << r << "," << c << ","
                           << get_float_value(a, i) << "," << get_float_value(b, i) << "\n";
            }
            save_inputs.close();
            std::cout << "   ✓ Saved " << size << " input rows for other libraries" << std::endl;
        }
    }
    
    Tensor a_pos = a + 0.1f;
    Tensor b_pos = b + 0.1f;
    
    std::cout << "Computing operations with timing (50 runs + 5 warmup)..." << std::endl;
    
    // Element-wise binary operations
    Tensor add_result = BENCH_OP("add", a + b);
    Tensor sub_result = BENCH_OP("sub", a - b);
    Tensor mul_result = BENCH_OP("mul", a * b);
    Tensor div_result = BENCH_OP("div", a / (b + 0.1f));
    
    // Unary arithmetic operations
    Tensor square_result = BENCH_OP("square", (square(a,0)));
    Tensor sqrt_result = BENCH_OP("sqrt", (sqrt(a_pos,0)));
    Tensor neg_result = BENCH_OP("neg", (neg(a,0)));
    Tensor abs_result = BENCH_OP("abs", (abs(a,0)));
    Tensor sign_result = BENCH_OP("sign", (sign(a,0)));
    Tensor reciprocal_result = BENCH_OP("reciprocal", (reciprocal(a_pos,0)));
    Tensor pow2_result = BENCH_OP("pow2", (pow(a, 2.0f,0)));
    
    // Trigonometric operations
    Tensor sin_result = BENCH_OP("sin", (sin(a,0)));
    Tensor cos_result = BENCH_OP("cos", (cos(a,0)));
    Tensor tan_result = BENCH_OP("tan", (tan(a,0)));
    Tensor sinh_result = BENCH_OP("sinh",(sinh(a,0)));
    Tensor cosh_result = BENCH_OP("cosh",(cosh(a,0)));
    Tensor tanh_result = BENCH_OP("tanh",(tanh(a,0)));
    Tensor asin_result = BENCH_OP("asin",(asin(a,0)));
    Tensor acos_result = BENCH_OP("acos",(acos(a,0)));
    Tensor atan_result = BENCH_OP("atan",(atan(a,0)));
    Tensor asinh_result = BENCH_OP("asinh",(asinh(a,0)));
    Tensor acosh_result = BENCH_OP("acosh",(acosh(a,0)));
    Tensor atanh_result = BENCH_OP("atanh",(atanh(a,0)));

    // Exponential and logarithmic operations
    Tensor exp_result = BENCH_OP("exp", (exp(a,0)));
    Tensor log_result = BENCH_OP("log", (log(a_pos,0)));
    Tensor log2_result = BENCH_OP("log2", (log2(a_pos,0)));
    Tensor log10_result = BENCH_OP("log10", (log10(a_pos,0)));
    
    // Matmul
    Tensor matmul_result = BENCH_OP("matmul", (matmul(a,b)));

    // Reductions
    Tensor sum_all = BENCH_OP("sum_all", reduce_sum(a_pos));
    Tensor mean_all = BENCH_OP("mean_all", reduce_mean(a_pos));
    Tensor max_all = BENCH_OP("max_all", reduce_max(a_pos));
    Tensor min_all = BENCH_OP("min_all", reduce_min(a_pos));
    Tensor var_all = BENCH_OP("var_all", reduce_var(a_pos));
    Tensor std_all = BENCH_OP("std_all", reduce_std(a_pos));
    
    std::cout << "✓ Completed " << all_timings.size() << " operations" << std::endl;

    // Save performance metric CSVs
    std::cout << "\nGenerating performance metric CSVs..." << std::endl;
    
    // 1. Timings CSV
    std::ofstream timing_csv("../../benchmark_results/tensorlib/tensorlib_timings.csv");
    timing_csv << std::fixed << std::setprecision(10);
    timing_csv << "operation,mean_ms,min_ms,max_ms,std_ms\n";
    for (const auto& [name, timing] : all_timings) {
        timing_csv << name << "," << timing.mean_ms << "," << timing.min_ms << ","
                   << timing.max_ms << "," << timing.std_ms << "\n";
    }
    timing_csv.close();
    std::cout << "✓ Timings saved to: tensorlib_timings.csv" << std::endl;
    
    // 2. Throughput CSV
    std::ofstream throughput_csv("../../benchmark_results/tensorlib/tensorlib_throughput.csv");
    throughput_csv << std::fixed << std::setprecision(2);
    throughput_csv << "operation,throughput_elem_per_sec\n";
    for (const auto& [name, timing] : all_timings) {
        double throughput = size / (timing.mean_ms / 1000.0);
        throughput_csv << name << "," << throughput << "\n";
    }
    throughput_csv.close();
    std::cout << "✓ Throughput saved to: tensorlib_throughput.csv" << std::endl;
    
    // 3. Bandwidth CSV
    std::ofstream bandwidth_csv("../../benchmark_results/tensorlib/tensorlib_bandwidth.csv");
    bandwidth_csv << std::fixed << std::setprecision(2);
    bandwidth_csv << "operation,memory_bandwidth_gb_per_sec\n";
    for (const auto& [name, timing] : all_timings) {
        double bandwidth = (size * 12) / (timing.mean_ms / 1000.0) / 1e9;
        bandwidth_csv << name << "," << bandwidth << "\n";
    }
    bandwidth_csv.close();
    std::cout << "✓ Bandwidth saved to: tensorlib_bandwidth.csv" << std::endl;
    
    // 4. FLOPS CSV
    std::ofstream flops_csv("../../benchmark_results/tensorlib/tensorlib_flops.csv");
    flops_csv << std::fixed << std::setprecision(4);
    flops_csv << "operation,gflops\n";
    for (const auto& [name, timing] : all_timings) {
        int flops_per_elem = 1;
        if (name.find("sin") != std::string::npos || name.find("cos") != std::string::npos ||
            name.find("tan") != std::string::npos || name.find("exp") != std::string::npos ||
            name.find("log") != std::string::npos) {
            flops_per_elem = 5;
        } else if (name.find("matmul") != std::string::npos) {
            flops_per_elem = 2 * C;
        }
        double gflops = (size * flops_per_elem) / (timing.mean_ms / 1000.0) / 1e9;
        flops_csv << name << "," << gflops << "\n";
    }
    flops_csv.close();
    std::cout << "✓ FLOPS saved to: tensorlib_flops.csv" << std::endl;
    
  

    // 5. Values CSV (for precision comparison)
    std::ofstream values_csv("../../benchmark_results/tensorlib/tensorlib_values.csv");
    values_csv << std::fixed << std::setprecision(6);
    
    // Header
    values_csv << "Idx";
    for (const auto& [name, _] : all_results) {
        values_csv << "," << name;
    }
    values_csv << "\n";
    
    // Rows (limited to 1000 for efficiency)
    int64_t num_export = std::min(size, (int64_t)1000);
    for (int64_t i = 0; i < num_export; ++i) {
        values_csv << i;
        for (const auto& [name, tensor] : all_results) {
             values_csv << "," << get_float_value(tensor, i);
        }
        values_csv << "\n";
    }
    values_csv.close();
    std::cout << "✓ Values saved to: tensorlib_values.csv" << std::endl;
    
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "TensorLib Benchmark Summary\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "  Total operations: " << all_timings.size() << "\n";
    std::cout << "  Shape: [" << D << "," << R << "," << C << "]\n";
    std::cout << std::string(60, '=') << "\n\n";
    
    return 0;
}