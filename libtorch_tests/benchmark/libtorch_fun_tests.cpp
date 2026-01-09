#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <map>

// Timing helper structure
struct OpTiming {
    std::string name;
    std::vector<double> times_ms;
    double mean_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double std_ms = 0.0;
};

// Global timing storage
std::map<std::string, OpTiming> all_timings;
const int NUM_RUNS = 50;
const int WARMUP_RUNS = 5;

// Benchmark macro - wraps operation with timing
#define BENCH_OP(op_name, expr) \
    [&]() { \
        /* Warmup */ \
        for (int _w = 0; _w < WARMUP_RUNS; _w++) { auto _tmp = expr; } \
        \
        /* Actual timing */ \
        OpTiming timing; \
        timing.name = op_name; \
        for (int _i = 0; _i < NUM_RUNS; _i++) { \
            auto _start = std::chrono::high_resolution_clock::now(); \
            auto _result = expr; \
            auto _end = std::chrono::high_resolution_clock::now(); \
            double _ms = std::chrono::duration<double, std::milli>(_end - _start).count(); \
            timing.times_ms.push_back(_ms); \
        } \
        \
        /* Calculate statistics */ \
        double _sum = 0.0; \
        timing.min_ms = timing.times_ms[0]; \
        timing.max_ms = timing.times_ms[0]; \
        for (double t : timing.times_ms) { \
            _sum += t; \
            timing.min_ms = std::min(timing.min_ms, t); \
            timing.max_ms = std::max(timing.max_ms, t); \
        } \
        timing.mean_ms = _sum / timing.times_ms.size(); \
        \
        double _sq_sum = 0.0; \
        for (double t : timing.times_ms) { \
            _sq_sum += (t - timing.mean_ms) * (t - timing.mean_ms); \
        } \
        timing.std_ms = std::sqrt(_sq_sum / timing.times_ms.size()); \
        \
        all_timings[op_name] = timing; \
        return expr; \
    }()

float get_float_value(const torch::Tensor& t, int64_t idx) {
    auto cpu = t.to(torch::kCPU).contiguous();
    switch (cpu.scalar_type()) {
        case torch::kFloat32:
            return cpu.data_ptr<float>()[idx];
        case torch::kFloat64:
            return static_cast<float>(cpu.data_ptr<double>()[idx]);
        case torch::kInt32:
            return static_cast<float>(cpu.data_ptr<int32_t>()[idx]);
        case torch::kInt64:
            return static_cast<float>(cpu.data_ptr<int64_t>()[idx]);
        default:
            return 0.0f;
    }
}

// Read inputs from CSV file
std::pair<torch::Tensor, torch::Tensor> read_inputs_from_csv(const std::string& filename, 
                                                               int64_t D, int64_t R, int64_t C) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open input CSV: " + filename);
    }

    std::string line;
    
    // Skip comment lines and header
    while (std::getline(file, line)) {
        if (line[0] != '#' && line.find("index") != std::string::npos) {
            break;  // Found header, next line is data
        }
    }

    int64_t size = D * R * C;
    std::vector<float> data_a(size);
    std::vector<float> data_b(size);

    int64_t idx = 0;
    while (std::getline(file, line) && idx < size) {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }
        
        if (tokens.size() >= 6) {
            // tokens: index, depth, row, col, input_a, input_b
            data_a[idx] = std::stof(tokens[4]);
            data_b[idx] = std::stof(tokens[5]);
            idx++;
        }
    }

    file.close();

    if (idx != size) {
        throw std::runtime_error("Expected " + std::to_string(size) + 
                                " values but read " + std::to_string(idx));
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor a = torch::from_blob(data_a.data(), {D, R, C}, options).clone();
    torch::Tensor b = torch::from_blob(data_b.data(), {D, R, C}, options).clone();

    return {a, b};
}

int main(int argc, char* argv[]) {
   
    std::string input_csv = "/home/blu-bridge016/Downloads/test_env_gau/benchmark_results/inputs/benchmark_all_3d_inputs.csv";
    
    // Allow overriding input file from command line
    if (argc > 1) {
        input_csv = argv[1];
    }

    const int64_t D = 10;
    const int64_t R = 10;
    const int64_t C = 10;
    const int64_t size = D * R * C;

    torch::Tensor a, b;
    
    std::cout << "Using LARGE tensors: [" << D << ", " << R << ", " << C << "] -> " << size << " elements\n";
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    
    // Read from CSV
    try {
        std::cout << "Reading inputs from " << input_csv << "...\n";
        auto inputs = read_inputs_from_csv(input_csv, D, R, C);
        a = inputs.first;
        b = inputs.second;
        std::cout << "✓ Inputs loaded from CSV\n";
    } catch (const std::exception& e) {
        std::cerr << "⚠ Error reading inputs: " << e.what() << "\n";
        std::cerr << "   Falling back to random generation...\n";
        torch::manual_seed(1926);
        a = torch::randn({D, R, C}, options);
        b = torch::randn({D, R, C}, options);
        std::cout << "✓ Generated random tensors [" << D << ", " << R << ", " << C << "]\n";
        
        // Save generated inputs
        std::string gen_path = "/home/blu-bridge016/Downloads/test_env_gau/benchmark_results/libtorch/generated_inputs.csv";
        std::cout << "   Saving generated inputs to " << gen_path << "...\n";
        std::ofstream input_file(gen_path);
        input_file << std::fixed << std::setprecision(12);
        input_file << "index,depth,row,col,input_a,input_b\n";
        
        auto a_cpu = a.to(torch::kCPU);
        auto b_cpu = b.to(torch::kCPU);
        
        for(int64_t d=0; d<D; d++)
            for(int64_t r=0; r<R; r++)
                for(int64_t c=0; c<C; c++){
                    int64_t idx = d*(R*C)+r*C+c;
                    input_file << idx << "," << d << "," << r << "," << c << ","
                               << get_float_value(a_cpu, idx) << ","
                               << get_float_value(b_cpu, idx) << "\n";
                }
        input_file.close();
        std::cout << "   ✓ Inputs saved.\n";
    }
    
    auto a_pos = torch::abs(a) + 0.1;
    auto b_pos = torch::abs(b) + 0.1;

    // Output file streams
    std::cout << "Computing operations with timing (50 runs + 5 warmup)...\n";

    auto add_result = BENCH_OP("add", a + b);
    auto sub_result = BENCH_OP("sub", a - b);
    auto mul_result = BENCH_OP("mul", a * b);
    auto div_result = BENCH_OP("div", a / (b + 0.1f));

    auto square_result = BENCH_OP("square", a * a);
    auto sqrt_result = BENCH_OP("sqrt", torch::sqrt(a_pos));
    auto neg_result = BENCH_OP("neg", -a);
    auto abs_result = BENCH_OP("abs", torch::abs(a));
    auto sign_result = BENCH_OP("sign", torch::sign(a));
    auto reciprocal_result = BENCH_OP("reciprocal", torch::reciprocal(a_pos));
    auto pow2_result = BENCH_OP("pow2", torch::pow(a, 2));

    auto sin_result = BENCH_OP("sin", torch::sin(a));
    auto cos_result = BENCH_OP("cos", torch::cos(a));
    auto tan_result = BENCH_OP("tan", torch::tan(a));
    auto sinh_result = BENCH_OP("sinh", torch::sinh(a));
    auto cosh_result = BENCH_OP("cosh", torch::cosh(a));
    auto tanh_result = BENCH_OP("tanh", torch::tanh(a));
    auto asin_result = BENCH_OP("asin", torch::asin(a));
    auto acos_result = BENCH_OP("acos", torch::acos(a));
    auto atan_result = BENCH_OP("atan", torch::atan(a));
    auto asinh_result = BENCH_OP("asinh", torch::asinh(a));
    auto acosh_result = BENCH_OP("acosh", torch::acosh(a_pos));
    auto atanh_result = BENCH_OP("atanh", torch::atanh(a * 0.9));  // keep domain safe

    auto exp_result = BENCH_OP("exp", torch::exp(a));
    auto log_result = BENCH_OP("log", torch::log(a_pos));
    auto log2_result = BENCH_OP("log2", torch::log2(a_pos));
    auto log10_result = BENCH_OP("log10", torch::log10(a_pos));

    auto matmul_result = BENCH_OP("matmul", torch::matmul(a, b));

    // ----------- REDUCTION OPERATIONS -----------
    std::cout << "Computing reduction operations...\n";
    
    // Full tensor reductions
    auto sum_all = BENCH_OP("sum_all", torch::sum(a_pos));
    auto mean_all = BENCH_OP("mean_all", torch::mean(a_pos));
    auto max_all = BENCH_OP("max_all", torch::max(a_pos));
    auto min_all = BENCH_OP("min_all", torch::min(a_pos));
    auto var_all = BENCH_OP("var_all", torch::var(a_pos, /*unbiased=*/true));
    auto std_all = BENCH_OP("std_all", torch::std(a_pos, /*unbiased=*/true));
    
    // Get scalar values
    float sum_val = sum_all.item<float>();
    float mean_val = mean_all.item<float>();
    float max_val = max_all.item<float>();
    float min_val = min_all.item<float>();
    float var_val = var_all.item<float>();
    float std_val = std_all.item<float>();

    // ----------- SCALAR OPERATIONS -----------
    std::cout << "Computing scalar operations...\n";
    
    auto scalar_add = BENCH_OP("scalar_add", a + 2.5f);
    auto scalar_mul = BENCH_OP("scalar_mul", a * 3.0f);
    auto scalar_div = BENCH_OP("scalar_div", a / 2.0f);
    auto reverse_sub = BENCH_OP("reverse_sub", 5.0f - a);

    std::cout << "Computing chain operations...\n";

    auto chain1  = BENCH_OP("chain1", torch::sin(torch::cos(torch::sqrt(square_result))));
    auto chain2  = BENCH_OP("chain2", torch::exp(torch::log(torch::log2(torch::log10(a_pos)))));
    auto chain3  = BENCH_OP("chain3", torch::sin(torch::cos(torch::tan(matmul_result))));
    auto chain4  = BENCH_OP("chain4", torch::pow(torch::log(torch::tan(matmul_result + 0.5f)), 2));
    auto chain5  = BENCH_OP("chain5", torch::tanh(torch::sin(torch::exp(a))));
    auto chain6  = BENCH_OP("chain6", torch::log(torch::exp(torch::sqrt(a_pos))));
    auto chain7  = BENCH_OP("chain7", torch::cos(torch::sin(torch::tanh(torch::log(a_pos)))));
    auto chain8  = BENCH_OP("chain8", torch::sqrt(torch::pow(torch::exp(torch::log(a_pos)), 2)));
    auto chain9  = BENCH_OP("chain9", torch::log(torch::reciprocal(torch::sqrt(torch::abs(torch::sin(a) + 0.01f)))));
    auto chain10 = BENCH_OP("chain10", torch::atan(torch::sinh(torch::tan(torch::cos(torch::sqrt(a_pos))))));
    auto chain11 = BENCH_OP("chain11", torch::sin(a + b));
    auto chain12 = BENCH_OP("chain12", torch::log(a_pos + b_pos));
    auto chain13 = BENCH_OP("chain13", torch::tanh(torch::exp(a) + torch::log(b_pos)));
    auto chain14 = BENCH_OP("chain14", torch::sin(torch::cos(torch::tan(torch::exp(torch::log(torch::log10(a_pos)))))));
    auto chain15 = BENCH_OP("chain15", torch::exp(torch::sin(torch::cos(torch::tanh(torch::exp(torch::log(a_pos)))))));

    std::cout << "✓ Completed " << all_timings.size() << " operations\n";

    std::cout << "Writing outputs to CSV...\n";

    std::ofstream csv("/home/blu-bridge016/Downloads/test_env_gau/benchmark_results/libtorch/libtorch_res.csv");
    csv << std::fixed << std::setprecision(6);

    csv << "index,depth,row,col,"
        << "input_a,input_b,"
        << "add,sub,mul,div,"
        << "square,sqrt,neg,abs,sign,reciprocal,pow2,"
        << "sin,cos,tan,sinh,cosh,tanh,asin,acos,atan,asinh,acosh,atanh,"
        << "exp,log,log2,log10,"
        << "matmul,"
        << "scalar_add,scalar_mul,scalar_div,reverse_sub,"
        << "sum_all,mean_all,max_all,min_all,var_all,std_all,"
        << "chain1,chain2,chain3,chain4,chain5,chain6,chain7,chain8,"
        << "chain9,chain10,chain11,chain12,chain13,chain14,chain15\n";

    for (int64_t d = 0; d < D; d++) {
        for (int64_t r = 0; r < R; r++) {
            for (int64_t c = 0; c < C; c++) {

                int64_t idx = d * (R * C) + r * C + c;
                if (idx >= 1000) goto skip_csv; // Optimized saving

                csv << idx << "," << d << "," << r << "," << c << ",";

                csv << get_float_value(a, idx) << ","
                    << get_float_value(b, idx) << ",";

                csv << get_float_value(add_result, idx) << ","
                    << get_float_value(sub_result, idx) << ","
                    << get_float_value(mul_result, idx) << ","
                    << get_float_value(div_result, idx) << ",";

                csv << get_float_value(square_result, idx) << ","
                    << get_float_value(sqrt_result, idx) << ","
                    << get_float_value(neg_result, idx) << ","
                    << get_float_value(abs_result, idx) << ","
                    << get_float_value(sign_result, idx) << ","
                    << get_float_value(reciprocal_result, idx) << ","
                    << get_float_value(pow2_result, idx) << ",";

                csv << get_float_value(sin_result, idx) << ","
                    << get_float_value(cos_result, idx) << ","
                    << get_float_value(tan_result, idx) << ","
                    << get_float_value(sinh_result, idx) << ","
                    << get_float_value(cosh_result, idx) << ","
                    << get_float_value(tanh_result, idx) << ","
                    << get_float_value(asin_result, idx) << ","
                    << get_float_value(acos_result, idx) << ","
                    << get_float_value(atan_result, idx) << ","
                    << get_float_value(asinh_result, idx) << ","
                    << get_float_value(acosh_result, idx) << ","
                    << get_float_value(atanh_result, idx) << ",";

                csv << get_float_value(exp_result, idx) << ","
                    << get_float_value(log_result, idx) << ","
                    << get_float_value(log2_result, idx) << ","
                    << get_float_value(log10_result, idx) << ",";

                csv << get_float_value(matmul_result, idx) << ",";

                // Scalar operations
                csv << get_float_value(scalar_add, idx) << ","
                    << get_float_value(scalar_mul, idx) << ","
                    << get_float_value(scalar_div, idx) << ","
                    << get_float_value(reverse_sub, idx) << ",";
                
                // Reduction operations
                csv << sum_val << ","
                    << mean_val << ","
                    << max_val << ","
                    << min_val << ","
                    << var_val << ","
                    << std_val << ",";


                csv << get_float_value(chain1, idx) << ","
                    << get_float_value(chain2, idx) << ","
                    << get_float_value(chain3, idx) << ","
                    << get_float_value(chain4, idx) << ","
                    << get_float_value(chain5, idx) << ","
                    << get_float_value(chain6, idx) << ","
                    << get_float_value(chain7, idx) << ","
                    << get_float_value(chain8, idx) << ","
                    << get_float_value(chain9, idx) << ","
                    << get_float_value(chain10, idx) << ","
                    << get_float_value(chain11, idx) << ","
                    << get_float_value(chain12, idx) << ","
                    << get_float_value(chain13, idx) << ","
                    << get_float_value(chain14, idx) << ","
                    << get_float_value(chain15, idx)
                    << "\n";
            }
        }
    }

skip_csv:
    csv.close();

    std::cout << "\n✓ Values CSV generated: libtorch_res.csv\n";
    
    // Save performance metric CSVs
    std::cout << "\nGenerating performance metric CSVs...\n";
    
    // 1. Timings CSV
    std::ofstream timing_csv("/home/blu-bridge016/Downloads/test_env_gau/benchmark_results/libtorch/libtorch_timings.csv");
    timing_csv << std::fixed << std::setprecision(10);
    timing_csv << "operation,mean_ms,min_ms,max_ms,std_ms\n";
    for (const auto& [name, timing] : all_timings) {
        timing_csv << name << "," << timing.mean_ms << "," << timing.min_ms << ","
                   << timing.max_ms << "," << timing.std_ms << "\n";
    }
    timing_csv.close();
    std::cout << "✓ Timings saved to: libtorch_timings.csv\n";
    
    // 2. Throughput CSV
    std::ofstream throughput_csv("/home/blu-bridge016/Downloads/test_env_gau/benchmark_results/libtorch/libtorch_throughput.csv");
    throughput_csv << std::fixed << std::setprecision(2);
    throughput_csv << "operation,throughput_elem_per_sec\n";
    for (const auto& [name, timing] : all_timings) {
        double throughput = size / (timing.mean_ms / 1000.0);
        throughput_csv << name << "," << throughput << "\n";
    }
    throughput_csv.close();
    std::cout << "✓ Throughput saved to: libtorch_throughput.csv\n";
    
    // 3. Bandwidth CSV
    std::ofstream bandwidth_csv("/home/blu-bridge016/Downloads/test_env_gau/benchmark_results/libtorch/libtorch_bandwidth.csv");
    bandwidth_csv << std::fixed << std::setprecision(2);
    bandwidth_csv << "operation,memory_bandwidth_gb_per_sec\n";
    for (const auto& [name, timing] : all_timings) {
        double bandwidth = (size * 12) / (timing.mean_ms / 1000.0) / 1e9;
        bandwidth_csv << name << "," << bandwidth << "\n";
    }
    bandwidth_csv.close();
    std::cout << "✓ Bandwidth saved to: libtorch_bandwidth.csv\n";
    
    // 4. FLOPS CSV  
    std::ofstream flops_csv("/home/blu-bridge016/Downloads/test_env_gau/benchmark_results/libtorch/libtorch_flops.csv");
    flops_csv << std::fixed << std::setprecision(4);
    flops_csv << "operation,gflops\n";
    for (const auto& [name, timing] : all_timings) {
        int flops_per_elem = 1;  // Default
        if (name.find("sin") != std::string::npos || name.find("cos") != std::string::npos ||
            name.find("tan") != std::string::npos || name.find("exp") != std::string::npos ||
            name.find("log") != std::string::npos) {
            flops_per_elem = 5;
        } else if (name.find("matmul") != std::string::npos) {
            flops_per_elem = 2 * C;
        } else if (name.find("chain") != std::string::npos) {
            flops_per_elem = 10;
        }
        double gflops = (size * flops_per_elem) / (timing.mean_ms / 1000.0) / 1e9;
        flops_csv << name << "," << gflops << "\n";
    }
    flops_csv.close();
    std::cout << "✓ FLOPS saved to: libtorch_flops.csv\n";

    
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "LibTorch Benchmark Summary\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "  Total operations: " << all_timings.size() << "\n";
    std::cout << "  Shape: [" << D << "," << R << "," << C << "]\n";
    std::cout << std::string(60, '=') << "\n\n";

    return 0;
}
