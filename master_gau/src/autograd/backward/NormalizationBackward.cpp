#include "autograd/backward/NormalizationBackward.h"
#include "autograd/operations/NormalizationOps.h"
#include "ops/helpers/LayerNormKernels.h"
#include "dtype/DtypeTraits.h"
#include "dtype/Types.h"
#include "core/TensorDataManip.h"
#include "device/DeviceCore.h"

namespace OwnTensor {
namespace autograd {

std::vector<Tensor> LayerNormBackward::apply(std::vector<Tensor>&& grads) {
    // 1. Unpack
    Tensor grad_output = grads[0];
    Tensor input = input_.unpack(shared_from_this()); // Version check
    Tensor mean = mean_.unpack(shared_from_this());
    Tensor rstd = rstd_.unpack(shared_from_this());
    
    Tensor weight;
    if (weight_.defined()) weight = weight_.unpack(shared_from_this());
    
    int64_t total_ele = input.numel();
    int64_t cols = normalized_shape_;
    int64_t rows = total_ele / cols;
    
    // 2. Output Gradients
    Tensor grad_input = Tensor::zeros(input.shape(), input.opts());
    Tensor grad_weight, grad_bias;
    
    // grad_weight/grad_bias always float32 — weight grads accumulate in fp32
    TensorOptions weight_grad_opts = TensorOptions()
        .with_dtype(Dtype::Float32)
        .with_device(input.device());

    if (weight.is_valid()) {
        grad_weight = Tensor::zeros(weight.shape(), weight_grad_opts);
        grad_bias = Tensor::zeros(weight.shape(), weight_grad_opts);
    } else {
        grad_weight = Tensor::zeros(Shape{{cols}}, weight_grad_opts);
        grad_bias = Tensor::zeros(Shape{{cols}}, weight_grad_opts);
    }
    
    // ---- CPU fallback (templated, always accumulates in float) ----
    auto cpu_layer_norm_backward = [&](auto* gy_ptr, auto* x_ptr, auto* gamma_ptr, auto* gx_ptr) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(gy_ptr)>>;
        const float* mean_ptr = mean.data<float>();
        const float* rstd_ptr = rstd.data<float>();
        float* gw_ptr = grad_weight.data<float>();
        float* gb_ptr = grad_bias.data<float>();

        std::vector<float> gw_acc(cols, 0.0f);
        std::vector<float> gb_acc(cols, 0.0f);

        for (int64_t i = 0; i < rows; ++i) {
            const T* row_gy = gy_ptr + i * cols;
            const T* row_x = x_ptr + i * cols;
            float mu = mean_ptr[i];
            float rs = rstd_ptr[i];

            for (int64_t j = 0; j < cols; ++j) {
                float val = (static_cast<float>(row_x[j]) - mu) * rs;
                float gy = static_cast<float>(row_gy[j]);
                gw_acc[j] += gy * val;
                gb_acc[j] += gy;
            }
        }

        for (int64_t j = 0; j < cols; ++j) {
            gw_ptr[j] = gw_acc[j];
            gb_ptr[j] = gb_acc[j];
        }

        #pragma omp parallel for
        for (int64_t i = 0; i < rows; ++i) {
            const T* row_gy = gy_ptr + i * cols;
            const T* row_x = x_ptr + i * cols;
            T* row_gx = gx_ptr + i * cols;
            float mu = mean_ptr[i];
            float rs = rstd_ptr[i];

            float sum1 = 0.0f;
            float sum2 = 0.0f;

            for (int64_t j = 0; j < cols; ++j) {
                float g = (gamma_ptr) ? static_cast<float>(gamma_ptr[j]) : 1.0f;
                float gy = static_cast<float>(row_gy[j]);
                float val = (static_cast<float>(row_x[j]) - mu) * rs;
                sum1 += gy * g;
                sum2 += gy * g * val;
            }

            for (int64_t j = 0; j < cols; ++j) {
                float g = (gamma_ptr) ? static_cast<float>(gamma_ptr[j]) : 1.0f;
                float gy = static_cast<float>(row_gy[j]);
                float val = (static_cast<float>(row_x[j]) - mu) * rs;
                row_gx[j] = static_cast<T>(rs * (gy * g - (sum1 + val * sum2) / cols));
            }
        }
    };

    // 3. Dispatch
    if (grad_output.device().is_cuda()) {
        // grad_gamma/grad_beta always float (weight grads accumulate in fp32)
        float* grad_gamma_ptr = grad_weight.data<float>();
        float* grad_beta_ptr = grad_bias.data<float>();

        if (grad_output.dtype() == Dtype::Float32) {
            float* gamma_ptr = (weight.is_valid()) ? weight.data<float>() : nullptr;
            cuda::layer_norm_backward_cuda(
                grad_output.data<float>(), input.data<float>(), mean.data<float>(), rstd.data<float>(), gamma_ptr,
                grad_input.data<float>(), grad_gamma_ptr, grad_beta_ptr, rows, cols);
        } else if (grad_output.dtype() == Dtype::Float16) {
            float16_t* gamma_ptr = (weight.is_valid()) ? weight.data<float16_t>() : nullptr;
            cuda::layer_norm_backward_cuda(
                grad_output.data<float16_t>(), input.data<float16_t>(), mean.data<float>(), rstd.data<float>(), gamma_ptr,
                grad_input.data<float16_t>(), grad_gamma_ptr, grad_beta_ptr, rows, cols);
        } else if (grad_output.dtype() == Dtype::Bfloat16) {
            bfloat16_t* gamma_ptr = (weight.is_valid()) ? weight.data<bfloat16_t>() : nullptr;
            cuda::layer_norm_backward_cuda(
                grad_output.data<bfloat16_t>(), input.data<bfloat16_t>(), mean.data<float>(), rstd.data<float>(), gamma_ptr,
                grad_input.data<bfloat16_t>(), grad_gamma_ptr, grad_beta_ptr, rows, cols);
        }
    } else {
        // CPU path
        if (grad_output.dtype() == Dtype::Float32) {
            const float* gamma_ptr = (weight.is_valid()) ? weight.data<float>() : nullptr;
            cpu_layer_norm_backward(grad_output.data<float>(), input.data<float>(), gamma_ptr, grad_input.data<float>());
        } else if (grad_output.dtype() == Dtype::Float16) {
            const float16_t* gamma_ptr = (weight.is_valid()) ? weight.data<float16_t>() : nullptr;
            cpu_layer_norm_backward(grad_output.data<float16_t>(), input.data<float16_t>(), gamma_ptr, grad_input.data<float16_t>());
        } else if (grad_output.dtype() == Dtype::Bfloat16) {
            const bfloat16_t* gamma_ptr = (weight.is_valid()) ? weight.data<bfloat16_t>() : nullptr;
            cpu_layer_norm_backward(grad_output.data<bfloat16_t>(), input.data<bfloat16_t>(), gamma_ptr, grad_input.data<bfloat16_t>());
        }
    }
    
    return {grad_input, grad_weight, grad_bias};
}

} // namespace autograd
} // namespace OwnTensor
