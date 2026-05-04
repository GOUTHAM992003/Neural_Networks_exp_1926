#ifndef BENCHMARK_HARNESS_H
#define BENCHMARK_HARNESS_H

#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <nvml.h>
#include <string>

namespace OwnTensor {
namespace Benchmarking {

class GpuManager {
public:
    GpuManager() : initialized_(false) {
        nvmlReturn_t result = nvmlInit();
        if (result == NVML_SUCCESS) {
            initialized_ = true;
            nvmlDeviceGetHandleByIndex(0, &device_);
        } else {
            std::cerr << "NVML Warning: Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
        }
    }

    ~GpuManager() {
        if (initialized_) {
            resetClocks();
            nvmlShutdown();
        }
    }

    bool lockClocks(unsigned int gpu_mhz, unsigned int mem_mhz) {
        if (!initialized_) return false;

        nvmlReturn_t res_gpu = nvmlDeviceSetGpuLockedClocks(device_, gpu_mhz, gpu_mhz);
        nvmlReturn_t res_mem = nvmlDeviceSetMemoryLockedClocks(device_, mem_mhz, mem_mhz);

        if (res_gpu != NVML_SUCCESS || res_mem != NVML_SUCCESS) {
            std::cerr << "NVML Warning: Failed to lock clocks. (Likely permission denied)\n";
            if (res_gpu != NVML_SUCCESS) std::cerr << "  GPU Clock Error: " << nvmlErrorString(res_gpu) << "\n";
            if (res_mem != NVML_SUCCESS) std::cerr << "  Mem Clock Error: " << nvmlErrorString(res_mem) << "\n";
            return false;
        }

        std::cout << "NVML: Locked Clocks -> GPU: " << gpu_mhz << " MHz, MEM: " << mem_mhz << " MHz\n";
        return true;
    }

    void resetClocks() {
        if (!initialized_) return;
        nvmlDeviceResetGpuLockedClocks(device_);
        nvmlDeviceResetMemoryLockedClocks(device_);
        std::cout << "NVML: Reset Clocks to default.\n";
    }

    unsigned int getTemperature() {
        if (!initialized_) return 0;
        unsigned int temp = 0;
        nvmlDeviceGetTemperature(device_, NVML_TEMPERATURE_GPU, &temp);
        return temp;
    }

    std::string getDeviceName() {
        if (!initialized_) return "N/A";
        char name[NVML_DEVICE_NAME_BUFFER_SIZE];
        nvmlDeviceGetName(device_, name, NVML_DEVICE_NAME_BUFFER_SIZE);
        return std::string(name);
    }

    bool isInitialized() const { return initialized_; }

private:
    bool initialized_;
    nvmlDevice_t device_;
};

struct BenchmarkResult {
    double mean_ms;
    double median_ms;
    double stddev_ms;
    double min_ms;
    double max_ms;
    double conf_interval_95_ms;
    size_t iterations;
};

class BenchmarkHarness {
public:
    BenchmarkHarness(int warmups = 100, int iterations = 1000000)
        : warmups_(warmups), iterations_(iterations) {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        
        //* allocation is happending but where exactly is the flushing of L2 happening
        //* why are we not using cudaCtxResetPersistingL2Cache() api for clearing the L2 buffer
        // Allocate L2 flush buffer (typically GPU L2 is 4-80MB, 128MB is safe for most)
        size_t l2_size = 128 * 1024 * 1024; 
        cudaMalloc(&l2_flush_buffer_, l2_size);
    }

    void lock_clocks(unsigned int gpu_mhz, unsigned int mem_mhz) {
        gpu_.lockClocks(gpu_mhz, mem_mhz);
    }

    ~BenchmarkHarness() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
        cudaFree(l2_flush_buffer_);
    }

    template<typename Func>
    BenchmarkResult run(Func&& kernel_func, bool flush_l2 = true) {
        unsigned int start_temp = gpu_.getTemperature();

        // Warmup
        for (int i = 0; i < warmups_; ++i) {
            kernel_func();
        }
        cudaDeviceSynchronize();

        std::vector<float> times;
        times.reserve(iterations_);

        for (int i = 0; i < iterations_; ++i) {
            if (flush_l2) {
                // Flush L2 by writing to a large buffer
                //* we should be using the api here!
                cudaMemset(l2_flush_buffer_, 0, 128 * 1024 * 1024);
            }

            cudaEventRecord(start_);
            kernel_func();
            cudaEventRecord(stop_);
            cudaEventSynchronize(stop_);

            float ms = 0;
            cudaEventElapsedTime(&ms, start_, stop_);
            times.push_back(ms);
        }

        BenchmarkResult res = calculate_stats(times);
        unsigned int end_temp = gpu_.getTemperature();
        
        std::cout << "  (Temp: " << start_temp << "C -> " << end_temp << "C)\n";

        return res;
    }

    static void print_result(const std::string& name, const BenchmarkResult& res, double throughput_gb = -1.0) {
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Benchmark: " << name << "\n";
        std::cout << "  Mean:     " << res.mean_ms << " ms\n";
        std::cout << "  Median:   " << res.median_ms << " ms\n";
        std::cout << "  StdDev:   " << res.stddev_ms << " ms (" << (res.stddev_ms / res.mean_ms * 100.0) << "%)\n";
        std::cout << "  95% CI:   +/- " << res.conf_interval_95_ms << " ms\n";
        //* confusing
        if (throughput_gb > 0) {
            double gb_per_sec = throughput_gb / (res.mean_ms / 1000.0);
            std::cout << "  Throughput: " << gb_per_sec << " GB/s\n";
        }
        std::cout << "------------------------------------------------\n";
    }

private:
    int warmups_;
    int iterations_;
    cudaEvent_t start_, stop_;
    void* l2_flush_buffer_;
    GpuManager gpu_;

    BenchmarkResult calculate_stats(std::vector<float>& times) {
        size_t n = times.size();
        double sum = std::accumulate(times.begin(), times.end(), 0.0);
        double mean = sum / n;

        std::sort(times.begin(), times.end());
        double median = (n % 2 == 0) ? (times[n/2 - 1] + times[n/2]) / 2.0 : times[n/2]; //* beacuse starts from zero

        double sq_sum = 0;
        for (float t : times) sq_sum += (t - mean) * (t - mean);
        double stddev = std::sqrt(sq_sum / (n - 1));

        // 95% confidence interval (z-score for 95% is approx 1.96)
        double ci = 1.96 * (stddev / std::sqrt(n));

        return {mean, median, stddev, (double)times.front(), (double)times.back(), ci, n};
    }
};

} // namespace Benchmarking
} // namespace OwnTensor

#endif
