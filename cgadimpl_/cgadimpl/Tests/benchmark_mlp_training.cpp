#include "ad/ag_all.hpp"
#include "ad/optimizer/optim.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <random>

using namespace ag;
using namespace OwnTensor;
namespace fs = std::filesystem;

// --- Helpher for Directory Creation ---
std::string ensure_benchmark_dir(std::string library_name) {
    std::vector<std::string> search_paths = {
        ".", "..", "../..", "../../.."
    };
    std::string root_path = ".";
    for(const auto& p : search_paths) {
        if(fs::exists(p + "/benchmark_results")) {
            root_path = p;
            break;
        }
    }
    std::string base = root_path + "/benchmark_results/" + library_name;
    if (!fs::exists(base)) fs::create_directories(base);
    return base;
}

Tensor rand_tensor(Shape shape, float min = -0.1f, float max = 0.1f) {
    Tensor t(shape);
    float* data = t.data<float>();
    size_t size = t.numel();
    static std::mt19937 gen(1926);
    std::uniform_real_distribution<float> dist(min, max);
    for(size_t i=0; i<size; ++i) data[i] = dist(gen);
    return t;
}

int main() {
    try {
        std::cout << "Starting MLP Training Benchmark (CGADIMPL)..." << std::endl;

        // Config
        int B = 64; 
        int In = 128;
        int H1 = 256;
        int H2 = 256;
        int Out = 10;
        int max_iters = 100;
        
        // Data
        Value X = make_tensor(rand_tensor(Shape({B, In})), "X");
        Value Y = make_tensor(rand_tensor(Shape({B, Out})), "Y");

        // Weights
        Tensor t_W1 = rand_tensor(Shape({In, H1})); t_W1.set_requires_grad(true);
        Tensor t_b1 = rand_tensor(Shape({1, H1}));  t_b1.set_requires_grad(true);
        Value W1 = make_tensor(t_W1, "W1");
        Value b1 = make_tensor(t_b1, "b1");

        Tensor t_W2 = rand_tensor(Shape({H1, H2})); t_W2.set_requires_grad(true);
        Tensor t_b2 = rand_tensor(Shape({1, H2}));  t_b2.set_requires_grad(true);
        Value W2 = make_tensor(t_W2, "W2");
        Value b2 = make_tensor(t_b2, "b2");

        Tensor t_W3 = rand_tensor(Shape({H2, Out})); t_W3.set_requires_grad(true);
        Tensor t_b3 = rand_tensor(Shape({1, Out}));  t_b3.set_requires_grad(true);
        Value W3 = make_tensor(t_W3, "W3");
        Value b3 = make_tensor(t_b3, "b3");

        std::vector<Value> params = {W1, b1, W2, b2, W3, b3};
        ag::SGDOptimizer optim(params, 0.01f);

        // Warmup
        for(int i=0; i<5; ++i) {
            optim.zero_grad();
            Value l1 = ag::relu(ag::matmul(X, W1) + b1);
            Value l2 = ag::relu(ag::matmul(l1, W2) + b2);
            Value logits = ag::matmul(l2, W3) + b3;
            Value loss = ag::mse_loss(logits, Y);
            backward(loss);
            optim.step();
        }

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < max_iters; ++iter) {
            optim.zero_grad();
            
            // Forward
            Value l1 = ag::relu(ag::matmul(X, W1) + b1);
            Value l2 = ag::relu(ag::matmul(l1, W2) + b2);
            Value logits = ag::matmul(l2, W3) + b3;
            
            // Loss
            Value loss = ag::mse_loss(logits, Y);
            
            // Backward
            backward(loss);
            
            // Update
            optim.step();
        }

        auto end = std::chrono::high_resolution_clock::now();
        double duration_sec = std::chrono::duration<double>(end - start).count();
        double avg_ms = (duration_sec * 1000.0) / max_iters;

        // Metrics
        long long weights_bytes = (In*H1 + H1 + H1*H2 + H2 + H2*Out + Out) * 4;
        long long inputs_bytes = (B*In) * 4;
        long long acts_bytes = (B*H1 + B*H2 + B*Out) * 4; // Approx
        long long total_data_per_step = weights_bytes + inputs_bytes + acts_bytes; // Read
        // Write: Grads (same size as weights + acts).
        // Total RW approximation for Bandwidth:
        long long total_rw_bytes = (weights_bytes + acts_bytes) * 2 + inputs_bytes; // Read+Write
        
        double bandwidth_gbps = (total_rw_bytes * max_iters) / duration_sec / 1e9;

        // Save
        std::string out_dir = ensure_benchmark_dir("tensorlib");
        std::string csv_path = out_dir + "/mlp_training_metrics.csv";
        std::ofstream file(csv_path);
        file << "Metric,Value,Unit\n";
        file << "Time," << avg_ms << ",ms\n";
        file << "Bandwidth," << bandwidth_gbps << ",GB/s\n";
        
        std::cout << "Training Benchmark Finished.\n";
        std::cout << "  Avg Time: " << avg_ms << " ms\n";
        std::cout << "  Bandwidth: " << bandwidth_gbps << " GB/s\n";
        std::cout << "  Result: " << csv_path << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Training Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
