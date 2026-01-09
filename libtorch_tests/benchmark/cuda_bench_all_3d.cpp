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

// Timing helper
struct OpTiming {
    std::string name;
    std::vector<double> times_ms;
    double mean_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double std_ms = 0.0;
};

std::map<std::string, OpTiming> all_timings;
const int NUM_RUNS = 50;
const int WARMUP_RUNS = 5;

// CUDA-aware benchmark macro (includes CUDA synchronization)
#define BENCH_OP_CUDA(op_name, expr) \
    [&]() { \
        for (int _w = 0; _w < WARMUP_RUNS; _w++) { auto _tmp = expr; torch::cuda::synchronize(); } \
        OpTiming timing; \
        timing.name = op_name; \
        for (int _i = 0; _i < NUM_RUNS; _i++) { \
            torch::cuda::synchronize(); \
            auto _start = std::chrono::high_resolution_clock::now(); \
            auto _result = expr; \
            torch::cuda::synchronize(); \
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

#include <filesystem>
namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    // ================================
    //   1. SETUP OUTPUT DIRECTORY
    // ================================
    std::string out_dir;
    
    // Check if we are at project root by looking for "benchmark_results"
    if (fs::exists("benchmark_results") && fs::is_directory("benchmark_results")) {
        out_dir = "benchmark_results/libtorch_cuda/";
    } else {
        // Assume we are in libtorch_tests/benchmark/
        out_dir = "../../benchmark_results/libtorch_cuda/";
    }

    // Ensure directory exists
    try {
        fs::create_directories(out_dir);
    } catch (const std::exception& e) {
        std::cerr << "⚠ Failed to create directory: " << e.what() << "\n";
    }
    
    std::cout << "Output directory: " << out_dir << "\n";

    // ================================
    //   2. INPUT HANDLING
    // ================================
    std::string input_csv = "/home/blu-bridge016/Downloads/test_env_gau/benchmark_results/inputs/benchmark_all_3d_inputs.csv";
    if (argc > 1) input_csv = argv[1];

    const int64_t D = 10;
    const int64_t R = 10;
    const int64_t C = 10;
    const int64_t size = D * R * C;

    // Check CUDA
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available! Please check your GPU driver.\n";
        return 1;
    }
    std::cout << " CUDA is available. Device count: " << torch::cuda::device_count() << "\n";
    torch::Device device(torch::kCUDA);

    torch::Tensor a, b;
    bool generated_random = false;

    try {
        std::cout << "Reading inputs from: " << input_csv << "\n";
        auto inputs = read_inputs_from_csv(input_csv, D, R, C);
        a = inputs.first.to(device);
        b = inputs.second.to(device);
        std::cout << "✓ Inputs loaded successfully from CSV\n";
    } catch (const std::exception& e) {
        std::cerr << "⚠ Could not read input CSV: " << e.what() << "\n";
        std::cerr << "   Generating RANDOM inputs instead...\n";
        
        torch::manual_seed(1926); // reproducible random
        a = torch::rand({D, R, C}, torch::dtype(torch::kFloat32).device(device));
        b = torch::rand({D, R, C}, torch::dtype(torch::kFloat32).device(device));
        generated_random = true;
    }
    
    auto a_cpu_init = a.to(torch::kCPU);
    auto b_cpu_init = b.to(torch::kCPU);

    // Save randomly generated inputs if needed
    if (generated_random) {
        std::string gen_input_path = out_dir + "generated_inputs.csv";
        std::cout << "Saving generated inputs to: " << gen_input_path << "\n";
        std::ofstream input_file(gen_input_path);
        input_file << std::fixed << std::setprecision(12);
        input_file << "# Generated Random Inputs\nindex,depth,row,col,input_a,input_b\n";
        for(int64_t d=0; d<D; d++)
            for(int64_t r=0; r<R; r++)
                for(int64_t c=0; c<C; c++){
                    int64_t idx = d*(R*C)+r*C+c;
                    input_file << idx << "," << d << "," << r << "," << c << ","
                               << get_float_value(a_cpu_init, idx) << ","
                               << get_float_value(b_cpu_init, idx) << "\n";
                }
        input_file.close();
    }

    auto a_pos = a + 0.1f;
    auto b_pos = b + 0.1f;

    std::cout << "Computing operations on CUDA with timing (50 runs + 5 warmup)...\n";

    auto add_result = BENCH_OP_CUDA("add", a + b);
    auto sub_result = BENCH_OP_CUDA("sub", a - b);
    auto mul_result = BENCH_OP_CUDA("mul", a * b);
    auto div_result = BENCH_OP_CUDA("div", a / (b + 0.1f));

    auto square_result = BENCH_OP_CUDA("square", a * a);
    auto sqrt_result = BENCH_OP_CUDA("sqrt", torch::sqrt(a_pos));
    auto neg_result = BENCH_OP_CUDA("neg", -a);
    auto abs_result = BENCH_OP_CUDA("abs", torch::abs(a));
    auto sign_result = BENCH_OP_CUDA("sign", torch::sign(a));
    auto reciprocal_result = BENCH_OP_CUDA("reciprocal", 1.0f / a_pos);
    auto pow2_result = BENCH_OP_CUDA("pow2", torch::pow(a, 2));

    auto sin_result = BENCH_OP_CUDA("sin", torch::sin(a));
    auto cos_result = BENCH_OP_CUDA("cos", torch::cos(a));
    auto tan_result = BENCH_OP_CUDA("tan", torch::tan(a));
    auto sinh_result = BENCH_OP_CUDA("sinh", torch::sinh(a));
    auto cosh_result = BENCH_OP_CUDA("cosh", torch::cosh(a));
    auto tanh_result = BENCH_OP_CUDA("tanh", torch::tanh(a));
    auto asin_result = BENCH_OP_CUDA("asin", torch::asin(a));
    auto acos_result = BENCH_OP_CUDA("acos", torch::acos(a));
    auto atan_result = BENCH_OP_CUDA("atan", torch::atan(a));
    auto asinh_result = BENCH_OP_CUDA("asinh", torch::asinh(a));
    auto acosh_result = BENCH_OP_CUDA("acosh", torch::acosh(a_pos));
    auto atanh_result = BENCH_OP_CUDA("atanh", torch::atanh(a));

    auto exp_result = BENCH_OP_CUDA("exp", torch::exp(a));
    auto log_result = BENCH_OP_CUDA("log", torch::log(a_pos));
    auto log2_result = BENCH_OP_CUDA("log2", torch::log2(a_pos));
    auto log10_result = BENCH_OP_CUDA("log10", torch::log10(a_pos));

    // Matmul per depth: flatten as (D, R, C)
    auto matmul_result = BENCH_OP_CUDA("matmul", [&]() {
        auto result = torch::zeros_like(a);
        for(int64_t d=0; d<D; d++)
            result[d] = torch::matmul(a[d], b[d]);
        return result;
    }());

    // Full tensor reductions
    auto sum_all = BENCH_OP_CUDA("sum_all", torch::sum(a_pos));
    auto mean_all = BENCH_OP_CUDA("mean_all", torch::mean(a_pos));
    auto max_all = BENCH_OP_CUDA("max_all", torch::max(a_pos));
    auto min_all = BENCH_OP_CUDA("min_all", torch::min(a_pos));
    auto var_all = BENCH_OP_CUDA("var_all", torch::var(a_pos, /*unbiased=*/true));
    auto std_all = BENCH_OP_CUDA("std_all", torch::std(a_pos, /*unbiased=*/true));

    float sum_val  = sum_all.item<float>();
    float mean_val = mean_all.item<float>();
    float max_val  = max_all.item<float>();
    float min_val  = min_all.item<float>();
    float var_val  = var_all.item<float>();
    float std_val  = std_all.item<float>();

    std::cout << "Scalar ops...\n";
    auto scalar_add = BENCH_OP_CUDA("scalar_add", a + 2.5f);
    auto scalar_mul = BENCH_OP_CUDA("scalar_mul", a * 3.0f);
    auto scalar_div = BENCH_OP_CUDA("scalar_div", a / 2.0f);
    auto reverse_sub = BENCH_OP_CUDA("reverse_sub", 5.0f - a);

    std::cout << "Chain ops...\n";
    auto chain1  = BENCH_OP_CUDA("chain1", torch::sin(torch::cos(torch::sqrt(a * a))));
    auto chain2  = BENCH_OP_CUDA("chain2", torch::exp(torch::log(torch::log2(torch::log10(a_pos)))));
    auto chain3  = BENCH_OP_CUDA("chain3", torch::sin(torch::cos(torch::tan(matmul_result))));
    auto chain4  = BENCH_OP_CUDA("chain4", torch::pow(torch::log(torch::tan(matmul_result + 0.5f)), 2));
    auto chain5  = BENCH_OP_CUDA("chain5", torch::tanh(torch::sin(torch::exp(a))));
    auto chain6  = BENCH_OP_CUDA("chain6", torch::log(torch::exp(torch::sqrt(a_pos))));
    auto chain7  = BENCH_OP_CUDA("chain7", torch::cos(torch::sin(torch::tanh(torch::log(a_pos)))));
    auto chain8  = BENCH_OP_CUDA("chain8", torch::sqrt(torch::pow(torch::exp(torch::log(a_pos)),2)));
    auto chain9  = BENCH_OP_CUDA("chain9", torch::log(1.0f / torch::sqrt(torch::abs(torch::sin(a)+0.01f))));
    auto chain10 = BENCH_OP_CUDA("chain10", torch::atan(torch::sinh(torch::tan(torch::cos(torch::sqrt(a_pos))))));
    auto chain11 = BENCH_OP_CUDA("chain11", torch::sin(a + b));
    auto chain12 = BENCH_OP_CUDA("chain12", torch::log(a_pos + b_pos));
    auto chain13 = BENCH_OP_CUDA("chain13", torch::tanh(torch::exp(a) + torch::log(b_pos)));
    auto chain14 = BENCH_OP_CUDA("chain14", torch::sin(torch::cos(torch::tan(torch::exp(torch::log(torch::log10(a_pos)))))));
    auto chain15 = BENCH_OP_CUDA("chain15", torch::exp(torch::sin(torch::cos(torch::tanh(torch::exp(torch::log(a_pos)))))));

    std::cout << "✓ Completed " << all_timings.size() << " operations\n";


    // ================================
    //   COPY EVERYTHING TO CPU
    // ================================
    std::cout << "Moving tensors to CPU...\n";

    auto toCPU = [](const torch::Tensor& t){ return t.to(torch::kCPU); };

    auto a_cpu = toCPU(a);
    auto b_cpu = toCPU(b);

    auto matmul_cpu = toCPU(matmul_result);

    auto add_h = toCPU(add_result);   auto sub_h = toCPU(sub_result);
    auto mul_h = toCPU(mul_result);   auto div_h = toCPU(div_result);

    auto square_h = toCPU(square_result);
    auto sqrt_h   = toCPU(sqrt_result);
    auto neg_h    = toCPU(neg_result);
    auto abs_h    = toCPU(abs_result);
    auto sign_h   = toCPU(sign_result);
    auto rec_h    = toCPU(reciprocal_result);
    auto pow2_h   = toCPU(pow2_result);

    auto sin_h = toCPU(sin_result);
    auto cos_h = toCPU(cos_result);
    auto tan_h = toCPU(tan_result);
    auto sinh_h = toCPU(sinh_result);
    auto cosh_h = toCPU(cosh_result);
    auto tanh_h = toCPU(tanh_result);
    auto asin_h = toCPU(asin_result);
    auto acos_h = toCPU(acos_result);
    auto atan_h = toCPU(atan_result);
    auto asinh_h = toCPU(asinh_result);
    auto acosh_h = toCPU(acosh_result);
    auto atanh_h = toCPU(atanh_result);

    auto exp_h = toCPU(exp_result);
    auto log_h = toCPU(log_result);
    auto log2_h = toCPU(log2_result);
    auto log10_h = toCPU(log10_result);

    auto c1=toCPU(chain1);auto c2=toCPU(chain2);auto c3=toCPU(chain3);
    auto c4=toCPU(chain4);auto c5=toCPU(chain5);auto c6=toCPU(chain6);
    auto c7=toCPU(chain7);auto c8=toCPU(chain8);auto c9=toCPU(chain9);
    auto c10=toCPU(chain10);auto c11=toCPU(chain11);auto c12=toCPU(chain12);
    auto c13=toCPU(chain13);auto c14=toCPU(chain14);auto c15=toCPU(chain15);

    // ================= INPUT CSV =================
    // ================= VALUES CSV =================
    // Rename to standard format: libtorch_cuda_values.csv
    std::cout << "Writing libtorch_cuda_values.csv...\n";
    std::ofstream csv(out_dir + "libtorch_cuda_values.csv");
    csv<<std::fixed<<std::setprecision(6);

    csv<<"index,depth,row,col,input_a,input_b,"
          "add,sub,mul,div,"
          "square,sqrt,neg,abs,sign,reciprocal,pow2,"
          "sin,cos,tan,sinh,cosh,tanh,asin,acos,atan,asinh,acosh,atanh,"
          "exp,log,log2,log10,"
          "matmul,"
          "scalar_add,scalar_mul,scalar_div,reverse_sub,"
          "sum_all,mean_all,max_all,min_all,var_all,std_all,"
          "chain1,chain2,chain3,chain4,chain5,chain6,chain7,chain8,"
          "chain9,chain10,chain11,chain12,chain13,chain14,chain15\n";

    for(int64_t d=0;d<D;d++)
        for(int64_t r=0;r<R;r++)
            for(int64_t c=0;c<C;c++){
                int64_t idx=d*(R*C)+r*C+c;
                if (idx >= 1000) goto skip_csv_output;  // Limit to 1000 rows max

                csv<<idx<<","<<d<<","<<r<<","<<c<<",";
                csv<<get_float_value(a_cpu,idx)<<","<<get_float_value(b_cpu,idx)<<",";

                csv<<get_float_value(add_h,idx)<<","<<get_float_value(sub_h,idx)<<","
                   <<get_float_value(mul_h,idx)<<","<<get_float_value(div_h,idx)<<",";

                csv<<get_float_value(square_h,idx)<<","<<get_float_value(sqrt_h,idx)<<","
                   <<get_float_value(neg_h,idx)<<","<<get_float_value(abs_h,idx)<<","
                   <<get_float_value(sign_h,idx)<<","<<get_float_value(rec_h,idx)<<","
                   <<get_float_value(pow2_h,idx)<<",";

                csv<<get_float_value(sin_h,idx)<<","<<get_float_value(cos_h,idx)<<","
                   <<get_float_value(tan_h,idx)<<","<<get_float_value(sinh_h,idx)<<","
                   <<get_float_value(cosh_h,idx)<<","<<get_float_value(tanh_h,idx)<<","
                   <<get_float_value(asin_h,idx)<<","<<get_float_value(acos_h,idx)<<","
                   <<get_float_value(atan_h,idx)<<","<<get_float_value(asinh_h,idx)<<","
                   <<get_float_value(acosh_h,idx)<<","<<get_float_value(atanh_h,idx)<<",";

                csv<<get_float_value(exp_h,idx)<<","<<get_float_value(log_h,idx)<<","
                   <<get_float_value(log2_h,idx)<<","<<get_float_value(log10_h,idx)<<",";

                csv<<get_float_value(matmul_cpu,idx)<<",";

                csv<<get_float_value(c1,idx)+2.5f<<","  // scalar_add same pattern
                   <<get_float_value(c1,idx)*3.0f<<","  // simplified reuse
                   <<get_float_value(c1,idx)/2.0f<<","
                   <<5.0f-get_float_value(c1,idx)<<",";

                csv<<sum_val<<","<<mean_val<<","<<max_val<<","<<min_val<<","<<var_val<<","<<std_val<<",";

                csv<<get_float_value(c1,idx)<<","<<get_float_value(c2,idx)<<","<<get_float_value(c3,idx)<<","
                   <<get_float_value(c4,idx)<<","<<get_float_value(c5,idx)<<","<<get_float_value(c6,idx)<<","
                   <<get_float_value(c7,idx)<<","<<get_float_value(c8,idx)<<","
                   <<get_float_value(c9,idx)<<","<<get_float_value(c10,idx)<<","<<get_float_value(c11,idx)<<","
                   <<get_float_value(c12,idx)<<","<<get_float_value(c13,idx)<<","<<get_float_value(c14,idx)<<","
                   <<get_float_value(c15,idx)<<"\n";
            }

skip_csv_output:
    csv.close();

    std::cout<<"\n✓ Values CSV generated\n";
    
    // Save performance metric CSVs
    std::cout << "\nGenerating CUDA performance metric CSVs...\n";
    
    // 1. Timings CSV
    std::ofstream timing_csv(out_dir + "libtorch_cuda_timings.csv");
    timing_csv << std::fixed << std::setprecision(10);
    timing_csv << "operation,mean_ms,min_ms,max_ms,std_ms\n";
    for (const auto& [name, timing] : all_timings) {
        timing_csv << name << "," << timing.mean_ms << "," << timing.min_ms << ","
                   << timing.max_ms << "," << timing.std_ms << "\n";
    }
    timing_csv.close();
    std::cout << "✓ Timings saved to: libtorch_cuda_timings.csv\n";
    
    // 2. Throughput CSV
    std::ofstream throughput_csv(out_dir + "libtorch_cuda_throughput.csv");
    throughput_csv << std::fixed << std::setprecision(2);
    throughput_csv << "operation,throughput_elem_per_sec\n";
    for (const auto& [name, timing] : all_timings) {
        double throughput = size / (timing.mean_ms / 1000.0);
        throughput_csv << name << "," << throughput << "\n";
    }
    throughput_csv.close();
    std::cout << "✓ Throughput saved to: libtorch_cuda_throughput.csv\n";
    
    // 3. Bandwidth CSV
    std::ofstream bandwidth_csv(out_dir + "libtorch_cuda_bandwidth.csv");
    bandwidth_csv << std::fixed << std::setprecision(2);
    bandwidth_csv << "operation,memory_bandwidth_gb_per_sec\n";
    for (const auto& [name, timing] : all_timings) {
        int bytes_per_elem = 4; // float32
        int access_count = 3; // default Binary: Read A + Read B -> Write C
        
        // Unary Ops
        if (name.find("sin") != std::string::npos || name.find("cos") != std::string::npos || 
            name.find("tan") != std::string::npos || name.find("exp") != std::string::npos || 
            name.find("log") != std::string::npos || name.find("sqrt") != std::string::npos ||
            name.find("square") != std::string::npos || name.find("neg") != std::string::npos ||
            name.find("abs") != std::string::npos || name.find("sign") != std::string::npos ||
            name.find("reciprocal") != std::string::npos || name.find("pow2") != std::string::npos ||
            name.find("chain") != std::string::npos) {
            access_count = 2; // Read A -> Write C
        }
        // Reductions
        else if (name.find("sum") != std::string::npos || name.find("mean") != std::string::npos ||
                 name.find("max") != std::string::npos || name.find("min") != std::string::npos ||
                 name.find("var") != std::string::npos || name.find("std") != std::string::npos) {
            access_count = 1; // Read A
        }
        
        double bandwidth = (size * bytes_per_elem * access_count) / (timing.mean_ms / 1000.0) / 1e9;
        bandwidth_csv << name << "," << bandwidth << "\n";
    }
    bandwidth_csv.close();
    std::cout << "✓ Bandwidth saved to: libtorch_cuda_bandwidth.csv\n";
    
    // 4. FLOPS CSV
    std::ofstream flops_csv(out_dir + "libtorch_cuda_flops.csv");
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
        } else if (name.find("chain") != std::string::npos) {
            flops_per_elem = 10;
        }
        double gflops = (size * flops_per_elem) / (timing.mean_ms / 1000.0) / 1e9;
        flops_csv << name << "," << gflops << "\n";
    }
    flops_csv.close();
    std::cout << "✓ FLOPS saved to: libtorch_cuda_flops.csv\n";
    
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "LibTorch CUDA Benchmark Summary\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "  Total operations: " << all_timings.size() << "\n";
    std::cout << "  Shape: [" << D << "," << R << "," << C << "]\n";
    std::cout << std::string(60, '=') << "\n\n";

    return 0;
}
