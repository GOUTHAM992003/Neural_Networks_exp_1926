#include "autograd/operations/NormalizationOps.h"
#include "autograd/ops_template.h"
#include "autograd/backward/NormalizationBackward.h"
#include "autograd/AutogradContext.h"
#include "autograd/Variable.h" // For make_variable
#include "ops/helpers/LayerNormKernels.h"
#include "dtype/DtypeTraits.h" // For is_same_type in TensorDataManip.h
#include "dtype/Types.h"
#include "core/TensorDataManip.h" // For data access
#include "device/DeviceCore.h"

namespace OwnTensor {
namespace autograd {

Tensor layer_norm(
    const Tensor& input, 
    const Tensor& weight, 
    const Tensor& bias, 
    int normalized_shape, 
    float eps)
{
    GraphRecordMode::record_forward("NORMALIZATION: layer_norm");
    // 1. Prepare Output Tensors
    Shape x_shape = input.shape();
    Tensor output = Tensor(x_shape, input.opts());
    
    // Mean and Rstd are (N,)
    // Assuming normalized_shape corresponds to the last dimension size.
    // Total rows N = numel / normalized_shape
    int64_t total_ele = input.numel();
    int64_t cols = normalized_shape;
    int64_t rows = total_ele / cols;
    
    // Check shape validity
    if (x_shape.dims.back() != cols) {
        throw std::runtime_error("LayerNorm: Last dimension of input must match normalized_shape");
    }
    
    // mean/rstd always float32 (statistics accumulate in fp32)
    TensorOptions stat_opts = TensorOptions()
        .with_dtype(Dtype::Float32)
        .with_device(input.device())
        .with_req_grad(false);
    Tensor mean = Tensor(Shape{{rows}}, stat_opts);
    Tensor rstd = Tensor(Shape{{rows}}, stat_opts);

    // 2. Dispatch
    if (input.device().is_cuda()) {
        device::set_cuda_device(input.device().index);

        if (input.dtype() == Dtype::Float16) {
            const float16_t* gamma_ptr = (weight.is_valid()) ? weight.data<float16_t>() : nullptr;
            const float16_t* beta_ptr = (bias.is_valid()) ? bias.data<float16_t>() : nullptr;
            cuda::layer_norm_forward_cuda(
                input.data<float16_t>(), gamma_ptr, beta_ptr,
                output.data<float16_t>(), mean.data<float>(), rstd.data<float>(),
                rows, cols, eps);
        } else if (input.dtype() == Dtype::Bfloat16) {
            const bfloat16_t* gamma_ptr = (weight.is_valid()) ? weight.data<bfloat16_t>() : nullptr;
            const bfloat16_t* beta_ptr = (bias.is_valid()) ? bias.data<bfloat16_t>() : nullptr;
            cuda::layer_norm_forward_cuda(
                input.data<bfloat16_t>(), gamma_ptr, beta_ptr,
                output.data<bfloat16_t>(), mean.data<float>(), rstd.data<float>(),
                rows, cols, eps);
        } else {
            const float* gamma_ptr = (weight.is_valid()) ? weight.data<float>() : nullptr;
            const float* beta_ptr = (bias.is_valid()) ? bias.data<float>() : nullptr;
            cuda::layer_norm_forward_cuda(
                input.data<float>(), gamma_ptr, beta_ptr,
                output.data<float>(), mean.data<float>(), rstd.data<float>(),
                rows, cols, eps);
        }
    } else {
        // CUDA Bridge: Connection to run on CUDA if available
#ifdef WITH_CUDA
        try {
            // Move to CUDA (Device 0)
            DeviceIndex gpu_dev(Device::CUDA, 0);
            Tensor x_cu = input.to(gpu_dev);
            Tensor w_cu = (weight.is_valid()) ? weight.to(gpu_dev) : Tensor();
            Tensor b_cu = (bias.is_valid()) ? bias.to(gpu_dev) : Tensor();
            
            // Execute on GPU
            Tensor out_cu = layer_norm(x_cu, w_cu, b_cu, normalized_shape, eps);
            
            // Move back to original device (CPU)
            return out_cu.to(input.device());
        } catch (...) {
            // Fallback to CPU execution if CUDA fails
        }
#endif
        // CPU Fallback (OpenMP) — templated lambda, always computes in float
        auto cpu_layer_norm_forward = [&](auto* x_ptr, auto* y_ptr, auto* gamma_ptr, auto* beta_ptr) {
            using T = std::remove_const_t<std::remove_pointer_t<decltype(x_ptr)>>;
            float* mean_ptr = mean.data<float>();
            float* rstd_ptr = rstd.data<float>();

            #pragma omp parallel for
            for (int64_t i = 0; i < rows; ++i) {
                const T* row_x = x_ptr + i * cols;
                T* row_y = y_ptr + i * cols;

                float sum = 0.0f;
                for (int64_t j = 0; j < cols; ++j) sum += static_cast<float>(row_x[j]);
                float mu = sum / cols;
                mean_ptr[i] = mu;

                float sum_sq = 0.0f;
                for (int64_t j = 0; j < cols; ++j) {
                    float diff = static_cast<float>(row_x[j]) - mu;
                    sum_sq += diff * diff;
                }
                float var = sum_sq / cols;
                float rs = 1.0f / std::sqrt(var + eps);
                rstd_ptr[i] = rs;

                for (int64_t j = 0; j < cols; ++j) {
                    float val = (static_cast<float>(row_x[j]) - mu) * rs;
                    float g = gamma_ptr ? static_cast<float>(gamma_ptr[j]) : 1.0f;
                    float b = beta_ptr ? static_cast<float>(beta_ptr[j]) : 0.0f;
                    row_y[j] = static_cast<T>(val * g + b);
                }
            }
        };

        if (input.dtype() == Dtype::Float16) {
            const float16_t* gp = weight.is_valid() ? weight.data<float16_t>() : nullptr;
            const float16_t* bp = bias.is_valid() ? bias.data<float16_t>() : nullptr;
            cpu_layer_norm_forward(input.data<float16_t>(), output.data<float16_t>(), gp, bp);
        } else if (input.dtype() == Dtype::Bfloat16) {
            const bfloat16_t* gp = weight.is_valid() ? weight.data<bfloat16_t>() : nullptr;
            const bfloat16_t* bp = bias.is_valid() ? bias.data<bfloat16_t>() : nullptr;
            cpu_layer_norm_forward(input.data<bfloat16_t>(), output.data<bfloat16_t>(), gp, bp);
        } else {
            const float* gp = weight.is_valid() ? weight.data<float>() : nullptr;
            const float* bp = bias.is_valid() ? bias.data<float>() : nullptr;
            cpu_layer_norm_forward(input.data<float>(), output.data<float>(), gp, bp);
        }
    }
    
    // Construct Autograd Graph
    if (GradMode::is_enabled() && (input.requires_grad() || (weight.is_valid() && weight.requires_grad()) || (bias.is_valid() && bias.requires_grad()))) {
        
        auto grad_fn = std::make_shared<LayerNormBackward>(
            input, mean, rstd, weight, normalized_shape, eps
        );
        
        if (input.requires_grad()) {
            grad_fn->set_next_edge(0, get_grad_edge(input));
        }

        if (weight.is_valid() && weight.requires_grad()) {
            grad_fn->set_next_edge(1, get_grad_edge(weight));
        }

        if (bias.is_valid() && bias.requires_grad()) {
            grad_fn->set_next_edge(2, get_grad_edge(bias));
        }
        
        output.set_grad_fn(grad_fn);
        output.set_requires_grad(true);
    }

    if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(output.shape(), output.dtype());
    return output;
}

} // namespace autograd
} // namespace OwnTensor