#include "nn/optimizer/Optim.h"
#include "ops/TensorOps.h"
#include "ops/ScalarOps.h"
#include "ops/UnaryOps/Arithmetics.h"
#include "ops/UnaryOps/Reduction.h"
#include <cmath>
#include <iostream>
#include "ops/helpers/AdamKernels.h"
#include "ops/helpers/GradNormKernels.h"
#include "ops/helpers/MultiTensorKernels.h"
#include <cuda_runtime.h>
#include "device/DeviceCore.h"
namespace OwnTensor {
namespace nn {

Optimizer::Optimizer(const std::vector<Tensor>& params) : params_(params) {
    for (const auto& p : params_) {
        void* key = p.unsafeGetTensorImpl();
        if (p.requires_grad() && p.dtype() != Dtype::Float32) {
            master_params_[key] = p.as_type(Dtype::Float32);
        }
    }
}

void Optimizer::zero_grad() {
    for (auto& p : params_) {
        if (p.requires_grad()) {
            p.fill_grad(0.0f);
        }
    }
}

Tensor* Optimizer::get_master_weight(const Tensor& v) {
    void* key = v.unsafeGetTensorImpl();
    auto it = master_params_.find(key);
    if (it != master_params_.end()) {
        return &it->second;
    }
    return nullptr;
}

SGDOptimizer::SGDOptimizer(const std::vector<Tensor>& params, float learning_rate)
    : Optimizer(params), learning_rate_(learning_rate) {}

void SGDOptimizer::step() {
    for (auto& p : params_) {
        if (!p.requires_grad()) continue;

        void* key = p.unsafeGetTensorImpl();
        Tensor grad_f32 = p.grad_view();
        if (grad_f32.dtype() != Dtype::Float32) {
            grad_f32 = grad_f32.as_type(Dtype::Float32);
        }

        auto it_master = master_params_.find(key);
        if (it_master != master_params_.end()) {
            Tensor& master_p = it_master->second;
            master_p += -learning_rate_ * grad_f32;
            p.copy_(master_p.as_type(p.dtype()));
        } else {
            // p is already Float32 or we update it directly
            // Note: If p is not Float32 and no master param, we might lose precision.
            // But the constructor ensures master_params for non-Float32.
            p += -learning_rate_ * grad_f32;
        }
    }
}

// **************************************************************************************
// ======================== Adam Optimizer ==============================================
// **************************************************************************************

Adam::Adam(std::vector<Tensor*> params, 
           float lr, 
           float beta1, 
           float beta2,
           float eps,
           float weight_decay)
    : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), 
      weight_decay_(weight_decay), step_count_(0),
      params_(std::move(params)), initialized_(false) {}

void Adam::step() {
    step_count_++;
    
    // Lazy initialization of momentum buffers
    if (!initialized_) {
        m_.reserve(params_.size());
        v_.reserve(params_.size());
        for (auto* param : params_) {
            TensorOptions opts = TensorOptions()
                .with_dtype(param->dtype())
                .with_device(param->device());
            m_.push_back(Tensor::zeros(param->shape(), opts));
            v_.push_back(Tensor::zeros(param->shape(), opts));
        }
        initialized_ = true;
    }
    
    // Bias correction factors
    float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(step_count_));
    float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(step_count_));
    
    for (size_t i = 0; i < params_.size(); ++i) {
        Tensor* param = params_[i];
        if (!param->requires_grad() || !param->has_grad()) {
            continue;
        }
        Tensor grad;
        try{
            grad=param->grad_view();
        } catch(...){
            continue;
        }
        int64_t numel = param->numel();
        
        // CPU implementation with direct data manipulation
        if (param->device().is_cpu()) {
            float* param_data = param->data<float>();
            const float* grad_data = param->grad<float>();
            float* m_data = m_[i].data<float>();
            float* v_data = v_[i].data<float>();
            
            for (int64_t j = 0; j < numel; ++j) {
                float g = grad_data[j];
                
                // Apply weight decay (AdamW style)
                if (weight_decay_ > 0.0f) {
                    g += weight_decay_ * param_data[j];
                }
                
                // Update first moment: m = beta1 * m + (1 - beta1) * g
                m_data[j] = beta1_ * m_data[j] + (1.0f - beta1_) * g;
                
                // Update second moment: v = beta2 * v + (1 - beta2) * g^2
                v_data[j] = beta2_ * v_data[j] + (1.0f - beta2_) * g * g;
                
                // Bias-corrected estimates
                float m_hat = m_data[j] / bias_correction1;
                float v_hat = v_data[j] / bias_correction2;
                
                // Update parameter
                param_data[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
            }
        } else {
            // CUDA: Use fused CUDA kernel for maximum performance
            cuda::fused_adam_cuda(
                param->data<float>(),
                param->grad<float>(),
                m_[i].data<float>(),
                v_[i].data<float>(),
                numel,
                lr_,
                beta1_,
                beta2_,
                eps_,
                weight_decay_,
                bias_correction1,
                bias_correction2
            );
        }
    }
}

void Adam::zero_grad() {
    for (auto* param : params_) {
        // Try to zero gradient - use try-catch for CUDA compatibility
        try {
            param->zero_grad();
        } catch (...) {
            // No gradient to zero
        }
    }
}

} // namespace nn
 
float clip_grad_norm_(std::vector<Tensor>& params, float max_norm, float norm_type, bool error_if_nonfinite, float* out_norm_async) {
    // FAST GPU-based gradient clipping using multi-tensor fused kernels
    static float* s_d_norm = nullptr;
    static float* s_d_clip_coef = nullptr;
    
    std::vector<cuda::TensorInfo> cuda_tensors;
    bool is_cuda = false;
    bool is_inf_norm = std::isinf(norm_type);
    
    for (auto param : params) {
        if (!param.requires_grad() || !param.has_grad()) continue;
        if (param.device().is_cuda()) {
            is_cuda = true;
            Tensor grad = param.grad_view();
            if (grad.dtype() == Dtype::Float32) {
                cuda_tensors.push_back({grad.data<float>(), static_cast<int64_t>(grad.numel())});
            }
        }
    }

    if (is_cuda && !cuda_tensors.empty()) {
        if (!s_d_norm) {
            cudaMalloc(&s_d_norm, sizeof(float));
            cudaMalloc(&s_d_clip_coef, sizeof(float));
        }
        
        cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
        cudaMemsetAsync(s_d_norm, 0, sizeof(float), stream);
        
        if (std::isinf(norm_type)) {
             for (const auto& t : cuda_tensors) {
                 cuda::grad_norm_inf_cuda(t.ptr, s_d_norm, t.numel);
             }
        } else {
            cuda::multi_tensor_grad_norm_cuda(cuda_tensors, s_d_norm);
        }
        
        cuda::compute_clip_coef_cuda(s_d_norm, s_d_clip_coef, max_norm, std::isinf(norm_type));
        cuda::multi_tensor_scale_cuda(cuda_tensors, s_d_clip_coef);
        
        if (out_norm_async) {
            cudaMemcpyAsync(out_norm_async, s_d_norm, sizeof(float), cudaMemcpyDeviceToHost, stream);
            return 0.0f; // Caller responsible for reading out_norm_async later
        } else {
            float total_norm;
            cudaMemcpy(&total_norm, s_d_norm, sizeof(float), cudaMemcpyDeviceToHost);
            
            if (error_if_nonfinite && (std::isnan(total_norm) || std::isinf(total_norm))) {
                 throw std::runtime_error("The total norm of gradients is non-finite.");
            }
            
            return total_norm;
        }
    } else {
        // CPU PATH: Simple loop
        float total_norm_sq = 0.0f;
        float total_norm_inf = 0.0f;
        
        for (auto param : params) {
            if (!param.requires_grad() || !param.has_grad()) continue;
            
            try {
                Tensor grad = param.grad_view();
                const float* data = grad.data<float>();
                int64_t n = grad.numel();
                
                if (is_inf_norm) {
                    for (int64_t i = 0; i < n; ++i) {
                        float val = std::abs(data[i]);
                        if (val > total_norm_inf) total_norm_inf = val;
                    }
                } else {
                    for (int64_t i = 0; i < n; ++i) {
                        total_norm_sq += data[i] * data[i];
                    }
                }
            } catch (...) {
                continue;
            }
        }
        
        float total_norm;
        if (is_inf_norm) {
            total_norm = total_norm_inf;
        } else {
            total_norm = std::sqrt(total_norm_sq);
        }

        if (error_if_nonfinite && (std::isnan(total_norm) || std::isinf(total_norm))) {
             throw std::runtime_error("The total norm of gradients from `parameters` is non-finite, so it cannot be clipped. To disable this error and scale the gradients by the non-finite norm anyway, set `error_if_nonfinite=False`");
        }
        
        float clip_coef = max_norm / (total_norm + 1e-6f);
        if (clip_coef < 1.0f) {
            for (auto param : params) {
                if (!param.requires_grad() || !param.has_grad()) continue;
                
                try {
                    Tensor grad = param.grad_view();
                    float* data = grad.data<float>();
                    int64_t n = grad.numel();
                    for (int64_t i = 0; i < n; ++i) {
                        data[i] *= clip_coef;
                    }
                } catch (...) {
                    continue;
                }
            }
        }
        
        return total_norm;
    }
}


namespace nn {

// *********************************************************************************************
// ============================ AdamW Optimizer ================================================
// *********************************************************************************************

AdamW::AdamW(const std::vector<Tensor>& params, float alpha, float beta1, float beta2, float epsilon, float weight_decay)
   : Optimizer(params), alpha_(alpha), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), weight_decay_(weight_decay), t_(0) {
  
   for (const auto& p : params_) {
       if (p.requires_grad()) {
           void* key = p.unsafeGetTensorImpl();
           TensorOptions opts_f32 = TensorOptions().with_dtype(Dtype::Float32).with_device(p.device());
           m_[key] = Tensor::zeros(p.shape(), opts_f32);
           v_[key] = Tensor::zeros(p.shape(), opts_f32);
       }
   }
}

void AdamW::step() {
   t_++;
   float bias_corr1 = 1.0f - std::pow(beta1_, t_);
   float bias_corr2 = 1.0f - std::pow(beta2_, t_);
   
   std::vector<cuda::TensorInfo> cuda_params, cuda_grads, cuda_ms, cuda_vs;
   
   for (auto& p : params_) {
       if (!p.requires_grad()) continue;

       void* key = p.unsafeGetTensorImpl();
       Tensor grad = p.grad_view();
       
       if (p.device().is_cuda() && grad.dtype() == Dtype::Float32) {
           cuda_params.push_back({p.data<float>(), static_cast<int64_t>(p.numel())});
           cuda_grads.push_back({grad.data<float>(), static_cast<int64_t>(grad.numel())});
           cuda_ms.push_back({m_[key].data<float>(), static_cast<int64_t>(m_[key].numel())});
           cuda_vs.push_back({v_[key].data<float>(), static_cast<int64_t>(v_[key].numel())});
       } else {
           // Fallback for CPU or other types
           Tensor& m = m_[key];
           Tensor& v = v_[key];
           m *= beta1_;
           m += (1.0f - beta1_) * grad;
           v *= beta2_;
           v += (1.0f - beta2_) * OwnTensor::square(grad);
           
           float alpha_eff = alpha_ * std::sqrt(bias_corr2) / bias_corr1;
           Tensor update = (alpha_eff * m / (OwnTensor::sqrt(v) + epsilon_ * std::sqrt(bias_corr2)))
                           + (alpha_ * weight_decay_ * p);
           p -= update;
       }
   }

   if (!cuda_params.empty()) {
       cuda::multi_tensor_adam_cuda(
           cuda_params, cuda_grads, cuda_ms, cuda_vs,
           alpha_, beta1_, beta2_, epsilon_, weight_decay_,
           bias_corr1, bias_corr2
       );
   }
}

} // namespace nn
} // namespace OwnTensor
