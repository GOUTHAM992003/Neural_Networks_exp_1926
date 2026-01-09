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

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

using namespace OwnTensor;

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

// Helper to time operation
template<typename Func>
Tensor time_op(const std::string& name, Func func) {
    #ifdef WITH_CUDA
    cudaDeviceSynchronize();
    #endif
    
    // Warmup
    for (int w = 0; w < WARMUP_RUNS; w++) {
        Tensor tmp = func();
        #ifdef WITH_CUDA
        cudaDeviceSynchronize();
        #endif
    }
    
    // Actual timing
    OpTiming timing;
    timing.name = name;
    Tensor result;
    
    for (int i = 0; i < NUM_RUNS; i++) {
        #ifdef WITH_CUDA
        cudaDeviceSynchronize();
        #endif
        
        auto start = std::chrono::high_resolution_clock::now();
        result = func();
        
        #ifdef WITH_CUDA
        cudaDeviceSynchronize();
        #endif
        
        auto end = std::chrono::high_resolution_clock::now();
        timing.times_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    
    // Calculate stats
    double sum = 0.0;
    timing.min_ms = timing.times_ms[0];
    timing.max_ms = timing.times_ms[0];
    for (double t : timing.times_ms) {
        sum += t;
        timing.min_ms = std::min(timing.min_ms, t);
        timing.max_ms = std::max(timing.max_ms, t);
    }
    timing.mean_ms = sum / timing.times_ms.size();
    
    double sq_sum = 0.0;
    for (double t : timing.times_ms) {
        sq_sum += (t - timing.mean_ms) * (t - timing.mean_ms);
    }
    timing.std_ms = std::sqrt(sq_sum / timing.times_ms.size());
    
    all_timings[name] = timing;
    return result;
}

float get_float_value(const Tensor& t, int64_t idx) {
    switch (t.dtype()) {
        case Dtype::Float32: return t.data<float>()[idx];
        case Dtype::Float64: return static_cast<float>(t.data<double>()[idx]);
        case Dtype::Int32: return static_cast<float>(t.data<int32_t>()[idx]);
        case Dtype::Int64: return static_cast<float>(t.data<int64_t>()[idx]);
        default: return 0.0f;
    }
}

int main() {
    std::cout << "╔════════════════════════════════════════════════╗\n";
    std::cout << "║  TensorLib CUDA Benchmark with Metrics        ║\n";
    std::cout << "╚════════════════════════════════════════════════╝\n\n";
    
#ifndef WITH_CUDA
    std::cerr << " Error: This benchmark requires CUDA support!\n";
    std::cerr << "   Compile with WITH_CUDA=1\n";
    return 1;
#endif
    
    TensorOptions opts;
    opts.dtype = Dtype::Float32;
    opts.device = DeviceIndex(Device::CUDA, 0);
    
    // Updated for cache-busting large tensor test (27M elements)
    const int64_t D = 10;
    const int64_t R = 10;
    const int64_t C = 10;
    const int64_t size = D * R * C;
    
    // Helper to read CSV lines manualy since we don't have the utility header here easily
    auto read_csv_inputs = [&](const std::string& filename) -> std::pair<Tensor, Tensor> {
        std::ifstream file(filename);
        if (!file.is_open()) throw std::runtime_error("Cannot open input CSV: " + filename);
        
        std::string line;
        // Skip header
        while (std::getline(file, line)) {
            if (line.find("index") != std::string::npos) break;
        }
        
        int64_t idx = 0;
        std::vector<float> data_a;
        std::vector<float> data_b;
        data_a.reserve(size);
        data_b.reserve(size);
        
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
        
        // Use Tensor constructor directly (like in test code)
        Tensor t_a({{D, R, C}}, Dtype::Float32, Device::CUDA);
        t_a.set_data(data_a);
        
        Tensor t_b({{D, R, C}}, Dtype::Float32, Device::CUDA);
        t_b.set_data(data_b);
        
        return {t_a, t_b};
    };

    Tensor a, b;
    std::string input_csv_path = "/home/blu-bridge016/Downloads/test_env_gau/benchmark_results/inputs/benchmark_all_3d_inputs.csv";
    
    try {
        std::cout << "Reading inputs from " << input_csv_path << "...\n";
        auto inputs = read_csv_inputs(input_csv_path);
        
        a = inputs.first;
        b = inputs.second;
        std::cout << "✓ Inputs loaded and moved to CUDA\n";
    } catch (const std::exception& e) {
        std::cerr << "⚠ Error reading inputs: " << e.what() << "\n";
        std::cerr << "   Falling back to random generation on CUDA...\n";
        a = Tensor::rand({{D, R, C}}, opts);
        b = Tensor::rand({{D, R, C}}, opts);
        std::cout << "✓ Generated random CUDA tensors\n";
    }
    
    Tensor a_pos = a + 0.1f;
    Tensor b_pos = b + 0.1f;
    
    std::cout << "Running TensorLib CUDA benchmark (50 runs + 5 warmup)...\n";
    
    // Element-wise operations
    Tensor add_result = time_op("add", [&]() { return a + b; });
    Tensor sub_result = time_op("sub", [&]() { return a - b; });
    Tensor mul_result = time_op("mul", [&]() { return a * b; });
    Tensor div_result = time_op("div", [&]() { return a / (b + 0.1f); });
    
    // Unary operations
    Tensor square_result = time_op("square", [&]() { return square(a, 0); });
    Tensor sqrt_result = time_op("sqrt", [&]() { return sqrt(a_pos, 0); });
    Tensor neg_result = time_op("neg", [&]() { return neg(a, 0); });
    Tensor abs_result = time_op("abs", [&]() { return abs(a, 0); });
    Tensor sign_result = time_op("sign", [&]() { return sign(a, 0); });
    Tensor reciprocal_result = time_op("reciprocal", [&]() { return reciprocal(a_pos, 0); });
    Tensor pow2_result = time_op("pow2", [&]() { return pow(a, 2.0f, 0); });
    
    // Trig operations
    Tensor sin_result = time_op("sin", [&]() { return sin(a, 0); });
    Tensor cos_result = time_op("cos", [&]() { return cos(a, 0); });
    Tensor tan_result = time_op("tan", [&]() { return tan(a, 0); });
    Tensor sinh_result = time_op("sinh", [&]() { return sinh(a, 0); });
    Tensor cosh_result = time_op("cosh", [&]() { return cosh(a, 0); });
    Tensor tanh_result = time_op("tanh", [&]() { return tanh(a, 0); });
    Tensor asin_result = time_op("asin", [&]() { return asin(a, 0); });
    Tensor acos_result = time_op("acos", [&]() { return acos(a, 0); });
    Tensor atan_result = time_op("atan", [&]() { return atan(a, 0); });
    Tensor asinh_result = time_op("asinh", [&]() { return asinh(a, 0); });
    Tensor acosh_result = time_op("acosh", [&]() { return acosh(a_pos, 0); });
    Tensor atanh_result = time_op("atanh", [&]() { return atanh(a, 0); });
    
    // Exp/log operations
    Tensor exp_result = time_op("exp", [&]() { return exp(a, 0); });
    Tensor log_result = time_op("log", [&]() { return log(a_pos, 0); });
    Tensor log2_result = time_op("log2", [&]() { return log2(a_pos, 0); });
    Tensor log10_result = time_op("log10", [&]() { return log10(a_pos, 0); });
    
    // Matmul
    Tensor matmul_result = time_op("matmul", [&]() { return matmul(a, b); });
    
    // Scalar operations
    Tensor scalar_add = time_op("scalar_add", [&]() { return a + 2.5f; });
    Tensor scalar_mul = time_op("scalar_mul", [&]() { return a * 3.0f; });
    Tensor scalar_div = time_op("scalar_div", [&]() { return a / 2.0f; });
    Tensor reverse_sub = time_op("reverse_sub", [&]() { return 5.0f - a; });
    
    // Chain operations (matching PyTorch)
    Tensor chain1 = time_op("chain1", [&]() { return sin(cos(sqrt(square(a, 0), 0)), 0); });
    Tensor chain2 = time_op("chain2", [&]() { return exp(log(log2(log10(a_pos, 0), 0), 0), 0); });
    Tensor chain3 = time_op("chain3", [&]() { return sin(cos(tan(matmul_result, 0), 0), 0); });
    Tensor chain4 = time_op("chain4", [&]() { return pow(log(tan(matmul_result + 0.5f, 0), 0), 2.0f, 0); });
    Tensor chain5 = time_op("chain5", [&]() { return tanh(sin(exp(a, 0), 0), 0); });
    Tensor chain6 = time_op("chain6", [&]() { return log(exp(sqrt(a_pos, 0), 0), 0); });
    Tensor chain7 = time_op("chain7", [&]() { return cos(sin(tanh(log(a_pos, 0), 0), 0), 0); });
    Tensor chain8 = time_op("chain8", [&]() { return sqrt(pow(exp(log(a_pos, 0), 0), 2.0f, 0), 0); });
    Tensor chain9 = time_op("chain9", [&]() { return log(reciprocal(sqrt(abs(sin(a, 0) + 0.01f, 0), 0), 0), 0); });
    Tensor chain10 = time_op("chain10", [&]() { return atan(sinh(tan(cos(sqrt(a_pos, 0), 0), 0), 0), 0); });
    Tensor chain11 = time_op("chain11", [&]() { return sin(a + b, 0); });
    Tensor chain12 = time_op("chain12", [&]() { return log(a_pos + b_pos, 0); });
    Tensor chain13 = time_op("chain13", [&]() { return tanh(exp(a, 0) + log(b_pos, 0), 0); });
    Tensor chain14 = time_op("chain14", [&]() { return sin(cos(tan(exp(log(log10(a_pos, 0), 0), 0), 0), 0), 0); });
    Tensor chain15 = time_op("chain15", [&]() { return exp(sin(cos(tanh(exp(log(a_pos, 0), 0), 0), 0), 0), 0); });
    
    // Reduction operations (matching PyTorch)
    Tensor sum_all = time_op("sum_all", [&]() { return reduce_sum(a_pos, {}, false); });
    Tensor mean_all = time_op("mean_all", [&]() { return reduce_mean(a_pos, {}, false); });
    Tensor max_all = time_op("max_all", [&]() { return reduce_max(a_pos, {}, false); });
    Tensor min_all = time_op("min_all", [&]() { return reduce_min(a_pos, {}, false); });
    Tensor var_all = time_op("var_all", [&](){ return reduce_var(a_pos, {}, false, 0); });
    Tensor std_all = time_op("std_all", [&](){ return reduce_std(a_pos, {}, false, 0); });

    // Reduction operations (matching PyTorch)

    std::cout << "✓ Completed " << all_timings.size() << " operations\n";
    
    // Move to CPU
    std::cout << "Moving results to CPU...\n";
    DeviceIndex cpu_dev(Device::CPU);
    
    auto a_cpu = a.to(cpu_dev);
    auto b_cpu = b.to(cpu_dev);
    auto add_cpu = add_result.to(cpu_dev);
    auto sub_cpu = sub_result.to(cpu_dev);
    auto mul_cpu = mul_result.to(cpu_dev);
    auto div_cpu = div_result.to(cpu_dev);
    auto square_cpu = square_result.to(cpu_dev);
    auto sqrt_cpu = sqrt_result.to(cpu_dev);
    auto neg_cpu = neg_result.to(cpu_dev);
    auto abs_cpu = abs_result.to(cpu_dev);
    auto sign_cpu = sign_result.to(cpu_dev);
    auto reciprocal_cpu = reciprocal_result.to(cpu_dev);
    auto pow2_cpu = pow2_result.to(cpu_dev);
    auto sin_cpu = sin_result.to(cpu_dev);
    auto cos_cpu = cos_result.to(cpu_dev);
    auto tan_cpu = tan_result.to(cpu_dev);
    auto sinh_cpu = sinh_result.to(cpu_dev);
    auto cosh_cpu = cosh_result.to(cpu_dev);
    auto tanh_cpu = tanh_result.to(cpu_dev);
    auto asin_cpu = asin_result.to(cpu_dev);
    auto acos_cpu = acos_result.to(cpu_dev);
    auto atan_cpu = atan_result.to(cpu_dev);
    auto asinh_cpu = asinh_result.to(cpu_dev);
    auto acosh_cpu = acosh_result.to(cpu_dev);
    auto atanh_cpu = atanh_result.to(cpu_dev);
    auto exp_cpu = exp_result.to(cpu_dev);
    auto log_cpu = log_result.to(cpu_dev);
    auto log2_cpu = log2_result.to(cpu_dev);
    auto log10_cpu = log10_result.to(cpu_dev);
    auto matmul_cpu = matmul_result.to(cpu_dev);
    
    // Save values CSV
    std::cout << "Writing values CSV...\n";
    std::ofstream csv("../../benchmark_results/tensorlib_cuda/tensorlib_cuda_values.csv");
    csv << std::fixed << std::setprecision(12);
    
    csv << "index,row,col,input_a,input_b,add,sub,mul,div,square,sqrt,neg,abs,sign,reciprocal,pow2,"
        << "sin,cos,tan,sinh,cosh,tanh,asin,acos,atan,asinh,acosh,atanh,exp,log,log2,log10,matmul\n";
    
    for (int64_t idx = 0; idx < size; idx++) {
        if (idx >= 1000) break; // Optimized to avoid massive CSV saving
        
        int64_t d = idx / (R * C);
        int64_t r = (idx / C) % R;
        int64_t c = idx % C;
        
        csv << idx << "," << d << "," << r << "," << c << ","
            << get_float_value(a_cpu, idx) << "," << get_float_value(b_cpu, idx) << ","
            << get_float_value(add_cpu, idx) << "," << get_float_value(sub_cpu, idx) << ","
            << get_float_value(mul_cpu, idx) << "," << get_float_value(div_cpu, idx) << ","
            << get_float_value(square_cpu, idx) << "," << get_float_value(sqrt_cpu, idx) << ","
            << get_float_value(neg_cpu, idx) << "," << get_float_value(abs_cpu, idx) << ","
            << get_float_value(sign_cpu, idx) << "," << get_float_value(reciprocal_cpu, idx) << ","
            << get_float_value(pow2_cpu, idx) << ","
            << get_float_value(sin_cpu, idx) << "," << get_float_value(cos_cpu, idx) << ","
            << get_float_value(tan_cpu, idx) << "," << get_float_value(sinh_cpu, idx) << ","
            << get_float_value(cosh_cpu, idx) << "," << get_float_value(tanh_cpu, idx) << ","
            << get_float_value(asin_cpu, idx) << "," << get_float_value(acos_cpu, idx) << ","
            << get_float_value(atan_cpu, idx) << "," << get_float_value(asinh_cpu, idx) << ","
            << get_float_value(acosh_cpu, idx) << "," << get_float_value(atanh_cpu, idx) << ","
            << get_float_value(exp_cpu, idx) << "," << get_float_value(log_cpu, idx) << ","
            << get_float_value(log2_cpu, idx) << "," << get_float_value(log10_cpu, idx) << ","
            << get_float_value(matmul_cpu, idx) << "\n";
    }
    csv.close();
    std::cout << "✓ Values saved\n";
    
    // Metric CSVs
    std::cout << "\nGenerating metric CSVs...\n";
    
    // 1. Timings
    std::ofstream timing_csv("../../benchmark_results/tensorlib_cuda/tensorlib_cuda_timings.csv");
    timing_csv << std::fixed << std::setprecision(10);
    timing_csv << "operation,mean_ms,min_ms,max_ms,std_ms\n";
    for (const auto& [name, timing] : all_timings) {
        timing_csv << name << "," << timing.mean_ms << "," << timing.min_ms << ","
                   << timing.max_ms << "," << timing.std_ms << "\n";
    }
    timing_csv.close();
    std::cout << "✓ Timings saved\n";
    
    // 2. Throughput
    std::ofstream throughput_csv("../../benchmark_results/tensorlib_cuda/tensorlib_cuda_throughput.csv");
    throughput_csv << std::fixed << std::setprecision(2);
    throughput_csv << "operation,throughput_elem_per_sec\n";
    for (const auto& [name, timing] : all_timings) {
        double throughput = size / (timing.mean_ms / 1000.0);
        throughput_csv << name << "," << throughput << "\n";
    }
    throughput_csv.close();
    std::cout << "✓ Throughput saved\n";
    
    // 3. Bandwidth
    std::ofstream bandwidth_csv("../../benchmark_results/tensorlib_cuda/tensorlib_cuda_bandwidth.csv");
    bandwidth_csv << std::fixed << std::setprecision(2);
    bandwidth_csv << "operation,memory_bandwidth_gb_per_sec\n";
    for (const auto& [name, timing] : all_timings) {
        int bytes_per_elem = 4; // float32
        int access_count = 3; // default Binary
        
        // Unary Ops
        if (name.find("sin") != std::string::npos || name.find("cos") != std::string::npos || 
            name.find("tan") != std::string::npos || name.find("exp") != std::string::npos || 
            name.find("log") != std::string::npos || name.find("sqrt") != std::string::npos ||
            name.find("square") != std::string::npos || name.find("neg") != std::string::npos ||
            name.find("abs") != std::string::npos || name.find("sign") != std::string::npos ||
            name.find("reciprocal") != std::string::npos || name.find("pow2") != std::string::npos ||
            name.find("chain") != std::string::npos) {
            access_count = 2;
        }
        // Reductions
        else if (name.find("sum") != std::string::npos || name.find("mean") != std::string::npos ||
                 name.find("max") != std::string::npos || name.find("min") != std::string::npos ||
                 name.find("var") != std::string::npos || name.find("std") != std::string::npos) {
            access_count = 1;
        }

        double bandwidth = (size * bytes_per_elem * access_count) / (timing.mean_ms / 1000.0) / 1e9;
        bandwidth_csv << name << "," << bandwidth << "\n";
    }
    bandwidth_csv.close();
    std::cout << "✓ Bandwidth saved\n";
    
    // 4. FLOPS
    std::ofstream flops_csv("../../benchmark_results/tensorlib_cuda/tensorlib_cuda_flops.csv");
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
            flops_per_elem = 10;  // Chain operations have multiple composed ops
        }
        double gflops = (size * flops_per_elem) / (timing.mean_ms / 1000.0) / 1e9;
        flops_csv << name << "," << gflops << "\n";
    }
    flops_csv.close();
    std::cout << "✓ FLOPS saved\n";
    
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "TensorLib CUDA Benchmark Summary\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "  Total operations: " << all_timings.size() << "\n";
    std::cout << "  Shape: [" << D << "," << R << "," << C << "]\n";
    std::cout << std::string(60, '=') << "\n\n";
    
    return 0;
}
