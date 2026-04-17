#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/ops/_fused_adamw.h>
#include <ATen/ops/_softmax_backward_data.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>

#include "TensorLib.h"
#include "BenchmarkHarness.h"
#include "ops/helpers/LayerNormKernels.h"
#include "ops/helpers/MultiTensorKernels.h"
#include "ops/helpers/ActivationKernels.h"
#include "ops/helpers/EmbeddingKernels.h"
#include "ops/helpers/LossKernels.h"
#include "ops/helpers/FusedKernels.h"
#include "ops/TensorOps.h"

using namespace OwnTensor::Benchmarking;

void bench_adam(int num_tensors, int64_t tensor_size, BenchmarkHarness& harness) {
    std::cout << "\n========================================================" << std::endl;
    std::cout << "Multi-Tensor AdamW: " << num_tensors << " tensors x " << tensor_size << " elements" << std::endl;
    std::cout << "========================================================" << std::endl;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    float lr = 1e-3f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, wd = 0.01f;
    int step = 100;
    float bc1 = 1.0f - std::pow(beta1, step);
    float bc2 = 1.0f - std::pow(beta2, step);

    std::vector<at::Tensor> params_t, grads_t, ms_t, vs_t;
    std::vector<at::Tensor> max_exp_avg_sqs_t; 
    std::vector<at::Tensor> steps_t;

    std::vector<at::Tensor> params_custom, ms_custom, vs_custom;

    for (int i = 0; i < num_tensors; ++i) {
        params_t.push_back(torch::randn({tensor_size}, opts));
        grads_t.push_back(torch::randn({tensor_size}, opts));
        ms_t.push_back(torch::zeros({tensor_size}, opts));
        vs_t.push_back(torch::zeros({tensor_size}, opts));
        max_exp_avg_sqs_t.push_back(torch::zeros({tensor_size}, opts));
        steps_t.push_back(torch::tensor((float)step, opts)); 

        params_custom.push_back(params_t[i].clone());
        ms_custom.push_back(ms_t[i].clone());
        vs_custom.push_back(vs_t[i].clone());
    }

    std::vector<OwnTensor::cuda::TensorInfo> param_info, grad_info, m_info, v_info;
    for (int i = 0; i < num_tensors; ++i) {
        param_info.push_back({params_custom[i].data_ptr<float>(), tensor_size});
        grad_info.push_back({grads_t[i].data_ptr<float>(), tensor_size});
        m_info.push_back({ms_custom[i].data_ptr<float>(), tensor_size});
        v_info.push_back({vs_custom[i].data_ptr<float>(), tensor_size});
    }

    auto custom_kernel = [&]() {
        OwnTensor::cuda::multi_tensor_adam_cuda(
            param_info, grad_info, m_info, v_info,
            lr, beta1, beta2, eps, wd, bc1, bc2
        );
    };

    nvtxRangePush("Custom_MultiTensor_Adam");
    auto custom_res = harness.run(custom_kernel, true);
    nvtxRangePop();

    std::vector<at::Tensor> pt_params, pt_ms, pt_vs, pt_max_sq;
    for (int i = 0; i < num_tensors; ++i) {
        pt_params.push_back(params_t[i].clone());
        pt_ms.push_back(ms_t[i].clone());
        pt_vs.push_back(vs_t[i].clone());
        pt_max_sq.push_back(max_exp_avg_sqs_t[i].clone());
    }

    auto pytorch_kernel = [&]() {
        at::_fused_adamw_(
            pt_params, grads_t, pt_ms, pt_vs, pt_max_sq, steps_t,
            lr, beta1, beta2, wd, eps, false, false
        );
    };

    nvtxRangePush("PyTorch_Fused_AdamW");
    auto pytorch_res = harness.run(pytorch_kernel, true);
    nvtxRangePop();

    BenchmarkHarness::print_result("Custom MultiTensor Adam", custom_res);
    BenchmarkHarness::print_result("PyTorch Fused AdamW", pytorch_res);
}


// ============================================================================
// Benchmark 2: Multi-Tensor Grad Norm
// ============================================================================
void bench_grad_norm(int num_tensors, int64_t tensor_size, BenchmarkHarness& harness) {
    std::cout << "\n========================================================" << std::endl;
    std::cout << "Multi-Tensor Grad Norm: " << num_tensors << " tensors x " << tensor_size << std::endl;
    std::cout << "========================================================" << std::endl;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    std::vector<at::Tensor> grads;
    std::vector<OwnTensor::cuda::TensorInfo> info;
    float norm_custom = 0.0f;
    float* d_norm;
    cudaMalloc(&d_norm, sizeof(float));

    for (int i = 0; i < num_tensors; ++i) {
        grads.push_back(torch::randn({tensor_size}, opts));
        info.push_back({grads[i].data_ptr<float>(), tensor_size});
    }

    auto custom_kernel = [&]() {
        cudaMemset(d_norm, 0, sizeof(float));
        OwnTensor::cuda::multi_tensor_grad_norm_cuda(info, d_norm);
    };

    nvtxRangePush("Custom_GradNorm");
    auto custom_res = harness.run(custom_kernel, true);
    nvtxRangePop();

    auto pytorch_kernel = [&]() {
        float norm = 0.0f;
        for (auto& g : grads) norm += g.norm().item<float>() * g.norm().item<float>();
        norm = std::sqrt(norm);
    };

    nvtxRangePush("PyTorch_GradNorm");
    auto pytorch_res = harness.run(pytorch_kernel, true);
    nvtxRangePop();

    BenchmarkHarness::print_result("Custom Grad Norm", custom_res);
    BenchmarkHarness::print_result("PyTorch Grad Norm", pytorch_res);
    
    cudaFree(d_norm);
}


// ============================================================================
// Benchmark 3: Embedding Forward + Backward
// ============================================================================
void bench_embedding(int64_t V, int64_t C, int64_t N, BenchmarkHarness& harness) {
    std::cout << "\n========================================================" << std::endl;
    std::cout << "Embedding: V=" << V << "  C=" << C << "  N=" << N << std::endl;
    std::cout << "========================================================" << std::endl;

    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    int padding_idx = -1;  // no padding for benchmark

    // ---------- shared data ---------------------------------------------------
    // Random indices in [0, V)
    auto indices_i64 = torch::randint(0, V, {N}, torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
    // uint16 copy for our kernel
    auto indices_u16 = indices_i64.to(torch::kShort).to(torch::kCUDA);

    // Weight table – contiguous, 16-byte aligned by default from CUDA malloc
    auto weight = torch::randn({V, C}, opts_f);

    // ---------- pre-allocate outputs for our kernel ----------------------------
    auto output_custom  = torch::empty({N, C}, opts_f);
    auto grad_weight_custom = torch::zeros({V, C}, opts_f);
    // A fake upstream gradient for the backward pass
    auto grad_output = torch::randn({N, C}, opts_f);

    // ===================== FORWARD ====================

    // --- Custom forward ---
    auto custom_fwd = [&]() {
        OwnTensor::cuda::embedding_forward_cuda(
            reinterpret_cast<const uint16_t*>(indices_u16.data_ptr<int16_t>()),
            weight.data_ptr<float>(),
            output_custom.data_ptr<float>(),
            N, C, V, padding_idx,
            /*weight_stride_row=*/C,
            /*weight_stride_col=*/1
        );
    };

    nvtxRangePush("Custom_Embedding_Fwd");
    auto custom_fwd_res = harness.run(custom_fwd, true);
    nvtxRangePop();

    // --- PyTorch forward ---
    auto pytorch_fwd = [&]() {
        auto out = torch::embedding(weight, indices_i64);
    };

    nvtxRangePush("PyTorch_Embedding_Fwd");
    auto pytorch_fwd_res = harness.run(pytorch_fwd, true);
    nvtxRangePop();

    // --- Forward correctness check ---
    {
        custom_fwd();
        cudaDeviceSynchronize();
        auto pt_out = torch::embedding(weight, indices_i64);
        float max_err = (output_custom - pt_out).abs().max().item<float>();
        std::cout << "  Forward max |err|: " << max_err << "\n";
    }

    BenchmarkHarness::print_result("Custom  Embedding Fwd", custom_fwd_res);
    BenchmarkHarness::print_result("PyTorch Embedding Fwd", pytorch_fwd_res);

    // Throughput: read weight row (C floats) + write output row (C floats) per index
    double fwd_bytes = (double)N * C * sizeof(float) * 2.0;
    double fwd_gb = fwd_bytes / 1e9;
    std::cout << "  Custom  BW: " << fwd_gb / (custom_fwd_res.mean_ms / 1000.0) << " GB/s\n";
    std::cout << "  PyTorch BW: " << fwd_gb / (pytorch_fwd_res.mean_ms / 1000.0) << " GB/s\n";

    // ===================== BACKWARD ====================

    // --- Custom backward ---
    auto custom_bwd = [&]() {
        // Zero grad_weight before each iteration
        cudaMemset(grad_weight_custom.data_ptr<float>(), 0, V * C * sizeof(float));
        OwnTensor::cuda::embedding_backward_cuda(
            reinterpret_cast<const uint16_t*>(indices_u16.data_ptr<int16_t>()),
            grad_output.data_ptr<float>(),
            grad_weight_custom.data_ptr<float>(),
            N, C, V, padding_idx,
            /*grad_weight_stride_row=*/C,
            /*grad_weight_stride_col=*/1
        );
    };

    nvtxRangePush("Custom_Embedding_Bwd");
    auto custom_bwd_res = harness.run(custom_bwd, true);
    nvtxRangePop();

    // --- PyTorch backward ---
    // Use torch::embedding_backward (the ATen-level function)
    auto pytorch_bwd = [&]() {
        auto gw = torch::zeros({V, C}, opts_f);
        gw.index_add_(0, indices_i64, grad_output);
    };

    nvtxRangePush("PyTorch_Embedding_Bwd");
    auto pytorch_bwd_res = harness.run(pytorch_bwd, true);
    nvtxRangePop();

    // --- Backward correctness check ---
    {
        cudaMemset(grad_weight_custom.data_ptr<float>(), 0, V * C * sizeof(float));
        custom_bwd();
        cudaDeviceSynchronize();
        auto pt_gw = torch::zeros({V, C}, opts_f);
        pt_gw.index_add_(0, indices_i64, grad_output);
        float max_err = (grad_weight_custom - pt_gw).abs().max().item<float>();
        std::cout << "  Backward max |err|: " << max_err << "\n";
    }

    BenchmarkHarness::print_result("Custom  Embedding Bwd", custom_bwd_res);
    BenchmarkHarness::print_result("PyTorch Embedding Bwd", pytorch_bwd_res);

    // Throughput: read grad_output (N*C) + scatter into grad_weight
    double bwd_bytes = (double)N * C * sizeof(float) * 2.0;
    double bwd_gb = bwd_bytes / 1e9;
    std::cout << "  Custom  BW: " << bwd_gb / (custom_bwd_res.mean_ms / 1000.0) << " GB/s\n";
    std::cout << "  PyTorch BW: " << bwd_gb / (pytorch_bwd_res.mean_ms / 1000.0) << " GB/s\n";

    // ===================== SUMMARY ====================
    std::cout << "\n--- Embedding Speedup Summary ---\n";
    std::cout << "  Forward  : " << pytorch_fwd_res.mean_ms / custom_fwd_res.mean_ms << "x (custom vs PyTorch)\n";
    std::cout << "  Backward : " << pytorch_bwd_res.mean_ms / custom_bwd_res.mean_ms << "x (custom vs PyTorch)\n";
    std::cout << "------------------------------------------------\n";
}


// ============================================================================
// Benchmark 4: Fused Tril Softmax Backward
// ============================================================================
void bench_fused_tril_softmax_backward(int64_t B, int64_t H, int64_t T,
                                        BenchmarkHarness& harness) {
    // B = batch size, H = num heads, T = sequence length (cols)
    int64_t rows = B * H * T;   // total rows = B * H * T (each attention row)
    int64_t cols = T;

    std::cout << "\n========================================================" << std::endl;
    std::cout << "Fused Tril Softmax Backward: B=" << B << " H=" << H
              << " T=" << T << " (rows=" << rows << " cols=" << cols << ")" << std::endl;
    std::cout << "========================================================" << std::endl;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    // Simulate softmax output: values in [0,1] that sum to 1 per row
    // Use actual softmax of random data to get realistic distribution
    auto logits = torch::randn({rows, cols}, opts);
    // Apply causal mask (tril) then softmax — this is what the forward pass produces
    auto mask = torch::ones({T, T}, opts).tril(0);
    // Broadcast mask across batch: logits is [rows, cols], mask row = row % T
    auto output = torch::softmax(
        logits.masked_fill(mask.repeat({B * H, 1}).eq(0), -1e9f), -1);

    auto grad_output = torch::randn({rows, cols}, opts);

    // Pre-allocate for custom kernel
    auto grad_input_custom = torch::empty({rows, cols}, opts);

    // ---- Custom kernel ----
    auto custom_kernel = [&]() {
        OwnTensor::cuda::fused_tril_softmax_backward_cuda(
            grad_output.data_ptr<float>(),
            output.data_ptr<float>(),
            grad_input_custom.data_ptr<float>(),
            rows, cols
        );
    };

    nvtxRangePush("Custom_FusedTrilSoftmax_Bwd");
    auto custom_res = harness.run(custom_kernel, true);
    nvtxRangePop();

    // ---- PyTorch fused softmax backward ----
    // _softmax_backward_data is PyTorch's internal fused kernel (single launch),
    // NOT the decomposed mul->sum->sub->mul sequence which is artificially slow.
    // dim=-1 (last dim), input_dtype=kFloat32
    auto pytorch_kernel = [&]() {
        auto gi = at::_softmax_backward_data(grad_output, output, -1, at::kFloat);
        (void)gi.data_ptr<float>();
    };

    nvtxRangePush("PyTorch_Softmax_Bwd");
    auto pytorch_res = harness.run(pytorch_kernel, true);
    nvtxRangePop();

    // ---- Correctness check ----
    {
        custom_kernel();
        cudaDeviceSynchronize();
        auto pt_grad = at::_softmax_backward_data(grad_output, output, -1, at::kFloat);
        float max_err = (grad_input_custom - pt_grad).abs().max().item<float>();
        std::cout << "  Max |err| vs PyTorch: " << max_err << "\n";
    }

    BenchmarkHarness::print_result("Custom  Fused Bwd", custom_res);
    BenchmarkHarness::print_result("PyTorch Softmax Bwd", pytorch_res);

    // Throughput: read output (rows*cols) + read grad_output (rows*cols) + write grad_input (rows*cols)
    double total_bytes = 3.0 * (double)rows * cols * sizeof(float);
    double total_gb = total_bytes / 1e9;
    std::cout << "  Custom  BW: " << total_gb / (custom_res.mean_ms / 1000.0) << " GB/s\n";
    std::cout << "  PyTorch BW: " << total_gb / (pytorch_res.mean_ms / 1000.0) << " GB/s\n";

    std::cout << "\n--- Speedup: " << pytorch_res.mean_ms / custom_res.mean_ms
              << "x (custom vs PyTorch) ---\n";
    std::cout << "------------------------------------------------\n";
}


int main() {
    cudaFree(0);

    BenchmarkHarness harness(20, 100);

    // =============== Adam ===============
    std::cout << "================================================================" << std::endl;
    std::cout << " AdamW Benchmark" << std::endl;
    std::cout << "================================================================" << std::endl;
    bench_adam(50, 1024 * 1024, harness);


    // =============== Embedding ===============
    std::cout << "\n================================================================" << std::endl;
    std::cout << " Embedding Benchmark" << std::endl;
    std::cout << "================================================================" << std::endl;

    // GPT-2 style: V=50257, C=768, seq=4096
    bench_embedding(50257, 768, 4096, harness);

    // Larger model: V=50257, C=1024, seq=8192
    bench_embedding(50257, 1024, 8192, harness);

    // Small / high-contention: V=1000, C=256, seq=16384
    bench_embedding(1000, 256, 16384, harness);


    // =============== Fused Tril Softmax Backward ===============
    std::cout << "\n================================================================" << std::endl;
    std::cout << " Fused Tril Softmax Backward Benchmark" << std::endl;
    std::cout << "================================================================" << std::endl;

    // GPT-2 small: B=8, H=12, T=128
    bench_fused_tril_softmax_backward(8, 12, 128, harness);

    // GPT-2 medium: B=4, H=12, T=512
    bench_fused_tril_softmax_backward(4, 12, 512, harness);

    // GPT-2 full: B=2, H=12, T=1024
    bench_fused_tril_softmax_backward(2, 12, 1024, harness);

    // Large model: B=1, H=16, T=2048
    bench_fused_tril_softmax_backward(1, 16, 2048, harness);


    std::cout << "\n All benchmarks complete." << std::endl;
    return 0;
}
