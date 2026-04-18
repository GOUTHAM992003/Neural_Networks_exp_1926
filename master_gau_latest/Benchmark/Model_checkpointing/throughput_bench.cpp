#include <iostream>
#include <chrono>
#include <filesystem>
#include <vector>
#include <iomanip>
#include "TensorLib.h"
#include <nvtx3/nvToolsExt.h>
#include "nn/NN.h"
#include "nn/optimizer/Optim.h"
#include "checkpointing/Checkpointing.h"
#include "autograd/AutogradOps.h"

using namespace OwnTensor;
namespace fs = std::filesystem;

/**
 * @brief Simple model for benchmarking.
 * This model will have a large number of parameters to simulate a real model.
 */
class LargeModel : public nn::Module {
public:
    LargeModel(int64_t num_params, DeviceIndex device) {
        // We create a single large linear layer to hold the parameters.
        // n_embd * m_embd = num_params
        // Let's use a square-ish shape.
        int64_t dim = static_cast<int64_t>(std::sqrt(num_params));
        linear = std::make_shared<nn::Linear>(dim, dim, true);
        linear->to(device);
        register_module(linear.get());
    }

    Tensor forward(const Tensor& x) override {
        return linear->forward(x);
    }

    std::shared_ptr<nn::Linear> linear;
};

void run_benchmark(int64_t target_size_mb, bool use_async) {
    std::cout << "\n=== Benchmarking " << (use_async ? "Asynchronous" : "Synchronous") 
              << " Checkpointing (" << target_size_mb << " MB) ===" << std::endl;

    DeviceIndex device(Device::CUDA, 0);
    
    // Each float is 4 bytes. We need 3 tensors (params, m, v).
    // So 12 bytes per parameter.
    int64_t num_params = (target_size_mb * 1024 * 1024 ) / 12;
    
    std::cout << "Initializing model and optimizer with ~" << num_params << " parameters..." << std::endl;
    LargeModel model(num_params, device);
    
    auto params = model.parameters();
    nn::AdamW optimizer(params, 1e-3);
    
    // Trigger optimizer state initialization by doing a dummy step
    int64_t in_features = model.linear->weight.shape().dims[1];
    Tensor x = Tensor::randn<float>(Shape{{1, in_features}}, TensorOptions().with_device(device));
    Tensor y = model.forward(x);
    Tensor loss = autograd::sum(y);
    loss.backward();
    optimizer.step();
    
    std::string base_dir = "benchmark_checkpoints";
    if (fs::exists(base_dir)) fs::remove_all(base_dir);
    fs::create_directories(base_dir);

    CheckpointManager manager(base_dir, "bench", 1, 0, false, use_async);
    
    // Measure Capture Time (blocking part of save)
    nvtxRangePush("CheckpointSnapshot");
    auto t_start = std::chrono::high_resolution_clock::now();
    manager.save(0, model, optimizer, loss.to_cpu().data<float>()[0]);
    auto t_mid = std::chrono::high_resolution_clock::now();
    nvtxRangePop();

    double capture_ms = std::chrono::duration<double, std::milli>(t_mid - t_start).count();
    double background_ms = 0;

    if (use_async) {
        std::cout << "Capture complete (" << capture_ms << " ms). Async save in progress..." << std::endl;
        nvtxRangePush("CheckpointBackgroundIO");
        manager.wait_for_completion();
        auto t_end = std::chrono::high_resolution_clock::now();
        nvtxRangePop();
        background_ms = std::chrono::duration<double, std::milli>(t_end - t_mid).count();
    } else {
        std::cout << "Synchronous save complete." << std::endl;
    }
    
    double total_ms = capture_ms + background_ms;
    double total_sec = total_ms / 1000.0;
    
    // Find the saved file to get accurate size
    std::string ckpt_path = base_dir + "/bench_step_0.ckpt";
    if (!fs::exists(ckpt_path)) {
        std::cerr << "Error: Checkpoint file not found at " << ckpt_path << std::endl;
        return;
    }
    
    uint64_t file_size = fs::file_size(ckpt_path);
    double size_mb = static_cast<double>(file_size) / (1024.0 * 1024.0);
    double throughput_mb_s = (size_mb / total_sec);
    
    auto format_time = [](double ms) -> std::string {
        double sec = ms / 1000.0;
        if (sec < 60) {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << sec << " s";
            return ss.str();
        } else {
            double min = sec / 60.0;
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << min << " min";
            return ss.str();
        }
    };

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Results:" << std::endl;
    std::cout << "  File Size:     " << size_mb << " MB" << std::endl;
    std::cout << "  Capture Time:  " << capture_ms << " ms" << std::endl;
    if (use_async) {
        std::cout << "  I/O Time:      " << background_ms << " ms" << std::endl;
    }
    std::cout << "  Total Time:    " << format_time(total_ms) << " (" << total_ms << " ms)" << std::endl;
    std::cout << "  Throughput:    " << throughput_mb_s << " MB/s (" << (throughput_mb_s / 1024.0) << " GB/s)" << std::endl;
    
    // Cleanup
    fs::remove_all(base_dir);
}

int main() {
    try {
        // Run variations
        run_benchmark(5120, false); // 256MB Sync
        run_benchmark(5120, true);  // 256MB Async
        
        run_benchmark(2048, false); // 1GB Sync
        run_benchmark(2048, true);  // 1GB Async

    } bbcatch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
