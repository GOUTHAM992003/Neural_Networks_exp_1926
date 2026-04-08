#include "autograd/backward/ActivationBackward.h"
#include "ops/TensorOps.h"
#include "ops/ScalarOps.h"
#include "ops/helpers/ConditionalOps.h"
#include "ops/UnaryOps/Trigonometry.h"
#include "ops/UnaryOps/Reduction.h"
#include "ops/helpers/ActivationKernels.h"
#include "dtype/Types.h"
#include "device/DeviceCore.h"
#include <stdexcept>
#include <cmath>

namespace OwnTensor {
namespace autograd {

// ============================================================================
// ReluBackward
// ============================================================================

ReluBackward::ReluBackward(const Tensor& input)
    : Node(1), saved_input_(input) {}

std::vector<Tensor> ReluBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("ReluBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // grad_input = grad_output * (input > 0)
    Tensor grad_input;
    if (grad_output.device().is_cuda()) {
        grad_input = Tensor(saved_input_.shape(), grad_output.opts());
        device::set_cuda_device(grad_output.device().index);
        if (grad_output.dtype() == Dtype::Float32) {
            cuda::relu_backward_cuda(grad_output.data<float>(), saved_input_.data<float>(), grad_input.data<float>(), grad_input.numel());
        } else if (grad_output.dtype() == Dtype::Float16) {
            cuda::relu_backward_cuda(grad_output.data<float16_t>(), saved_input_.data<float16_t>(), grad_input.data<float16_t>(), grad_input.numel());
        } else if (grad_output.dtype() == Dtype::Bfloat16) {
            cuda::relu_backward_cuda(grad_output.data<bfloat16_t>(), saved_input_.data<bfloat16_t>(), grad_input.data<bfloat16_t>(), grad_input.numel());
        } else {
            Tensor mask = saved_input_ > 0.0f;
            grad_input = grad_output * mask;
        }
    } else {
        Tensor mask = saved_input_ > 0.0f;
        grad_input = grad_output * mask;
    }
    
    return {grad_input};
}

// ============================================================================
// GeLUBackward
// ============================================================================

GeLUBackward::GeLUBackward(const Tensor& input)
    : Node(1), saved_input_(input) {}

std::vector<Tensor> GeLUBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("GeLUBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    const Tensor& x = saved_input_;
    
    // Use fused CUDA kernel for GPU tensors (much faster)
    if (x.device().is_cuda() && (x.dtype() == Dtype::Float32 || x.dtype() == Dtype::Float16 || x.dtype() == Dtype::Bfloat16)) {
        Tensor grad_input(x.shape(), TensorOptions()
            .with_dtype(x.dtype())
            .with_device(x.device()));

        OwnTensor::device::set_cuda_device(x.device().index);
        if (x.dtype() == Dtype::Float32) {
            cuda::fused_gelu_backward_cuda(grad_output.data<float>(), x.data<float>(), grad_input.data<float>(), x.numel());
        } else if (x.dtype() == Dtype::Float16) {
            cuda::fused_gelu_backward_cuda(grad_output.data<float16_t>(), x.data<float16_t>(), grad_input.data<float16_t>(), x.numel());
        } else {
            cuda::fused_gelu_backward_cuda(grad_output.data<bfloat16_t>(), x.data<bfloat16_t>(), grad_input.data<bfloat16_t>(), x.numel());
        }

        return {grad_input};
    }

    // Fallback to tensor ops for CPU or unsupported dtype
    // GeLU derivative:
    // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // Let u = sqrt(2/pi) * (x + 0.044715 * x^3)
    // gelu'(x) = 0.5 * (1 + tanh(u)) + 0.5 * x * sech^2(u) * sqrt(2/pi) * (1 + 3*0.044715*x^2)
    
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    const float c = 0.044715f;
    
    Tensor x_sq = x * x;
    Tensor x_cubed = x_sq * x;
    Tensor u = sqrt_2_over_pi * (x + c * x_cubed);
    Tensor tanh_u = tanh(u);
    
    // sech^2(u) = 1 - tanh^2(u)
    Tensor sech2_u = 1.0f - tanh_u * tanh_u;
    
    // du/dx = sqrt(2/pi) * (1 + 3*c*x^2)
    Tensor du_dx = sqrt_2_over_pi * (1.0f + 3.0f * c * x_sq);
    
    // gelu'(x) = 0.5 * (1 + tanh(u)) + 0.5 * x * sech^2(u) * du/dx
    Tensor grad_x = 0.5f * (1.0f + tanh_u) + 0.5f * x * sech2_u * du_dx;
    
    return {grad_output * grad_x};
}

// ============================================================================
// SigmoidBackward
// ============================================================================

SigmoidBackward::SigmoidBackward(const Tensor& output)
    : Node(1), saved_output_(output) {}

std::vector<Tensor> SigmoidBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("SigmoidBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // grad_x = grad_out * sigmoid(x) * (1 - sigmoid(x))
    // We saved sigmoid(x) as saved_output_
    Tensor grad_x;
    if (grad_output.device().is_cuda()) {
        grad_x = Tensor(saved_output_.shape(), grad_output.opts());
        device::set_cuda_device(grad_output.device().index);
        if (grad_output.dtype() == Dtype::Float32) {
            cuda::sigmoid_backward_cuda(grad_output.data<float>(), saved_output_.data<float>(), grad_x.data<float>(), grad_x.numel());
        } else if (grad_output.dtype() == Dtype::Float16) {
            cuda::sigmoid_backward_cuda(grad_output.data<float16_t>(), saved_output_.data<float16_t>(), grad_x.data<float16_t>(), grad_x.numel());
        } else if (grad_output.dtype() == Dtype::Bfloat16) {
            cuda::sigmoid_backward_cuda(grad_output.data<bfloat16_t>(), saved_output_.data<bfloat16_t>(), grad_x.data<bfloat16_t>(), grad_x.numel());
        } else {
            grad_x = grad_output * saved_output_ * (1.0f - saved_output_);
        }
    } else {
        grad_x = grad_output * saved_output_ * (1.0f - saved_output_);
    }
    
    return {grad_x};
}

// ============================================================================
// SoftmaxBackward
// ============================================================================

SoftmaxBackward::SoftmaxBackward(const Tensor& output, int64_t dim)
    : Node(1), saved_output_(output), dim_(dim) {}

std::vector<Tensor> SoftmaxBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("SoftmaxBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    const Tensor& s = saved_output_;  // softmax output
    
    // Softmax backward: grad_x = s * (grad_out - sum(grad_out * s, dim))
    Tensor grad_x;
    int64_t ndim = s.ndim();
    int64_t d = dim_ < 0 ? dim_ + ndim : dim_;
    
    if (grad_output.device().is_cuda() && d == ndim - 1) {
        grad_x = Tensor(s.shape(), grad_output.opts());
        int64_t cols = s.shape().dims.back();
        int64_t rows = s.numel() / cols;

        OwnTensor::device::set_cuda_device(grad_output.device().index);
        if (grad_output.dtype() == Dtype::Float32) {
            cuda::softmax_backward_cuda(grad_output.data<float>(), s.data<float>(), grad_x.data<float>(), rows, cols);
        } else if (grad_output.dtype() == Dtype::Float16) {
            cuda::softmax_backward_cuda(grad_output.data<float16_t>(), s.data<float16_t>(), grad_x.data<float16_t>(), rows, cols);
        } else if (grad_output.dtype() == Dtype::Bfloat16) {
            cuda::softmax_backward_cuda(grad_output.data<bfloat16_t>(), s.data<bfloat16_t>(), grad_x.data<bfloat16_t>(), rows, cols);
        } else {
            Tensor gs = grad_output * s;
            Tensor sum_gs = reduce_sum(gs, {dim_}, true);
            grad_x = s * (grad_output - sum_gs);
        }
    } else {
        Tensor gs = grad_output * s;
        Tensor sum_gs = reduce_sum(gs, {dim_}, true);
        grad_x = s * (grad_output - sum_gs);
    }
    
    return {grad_x};
}

// ============================================================================
// DropoutBackward
// ============================================================================

DropoutBackward::DropoutBackward(const Tensor& mask, float scale)
    : Node(1), saved_mask_(mask), scale_(scale) {}

std::vector<Tensor> DropoutBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("DropoutBackward: no gradients provided");
    }

    const Tensor& grad_output = grads[0];

    // grad_input = grad_output * mask * scale
    Tensor grad_input = grad_output * saved_mask_ * scale_;

    return {grad_input};
}

// ============================================================================
// SwiGLUBackward
// ============================================================================
SwiGLUBackward::SwiGLUBackward(const Tensor& input) : Node(1), saved_input_(input) {}

std::vector<Tensor> SwiGLUBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("SwiGLU Backward: no gradient provided");
    }

    const Tensor& grad_output = grads[0];
    const Tensor& x = saved_input_;

    int64_t last = x.shape().dims.back();
    int64_t hidden = last / 2;
    int64_t rows = x.numel() / last;

    Tensor grad_input(x.shape(), x.opts());
    if (x.dtype() == Dtype::Float32) {
        cuda::swiglu_backward_cuda(grad_output.data<float>(), x.data<float>(), grad_input.data<float>(), rows, hidden);
    } else if (x.dtype() == Dtype::Float16) {
        cuda::swiglu_backward_cuda(grad_output.data<float16_t>(), x.data<float16_t>(), grad_input.data<float16_t>(), rows, hidden);
    } else if (x.dtype() == Dtype::Bfloat16) {
        cuda::swiglu_backward_cuda(grad_output.data<bfloat16_t>(), x.data<bfloat16_t>(), grad_input.data<bfloat16_t>(), rows, hidden);
    }

    return {grad_input};
}


// ============================================================================
// FusedBiasGeLUBackward
// ============================================================================

FusedBiasGeLUBackward::FusedBiasGeLUBackward(const Tensor& input, const Tensor& bias)
    : Node(2), saved_input_(input), saved_bias_(bias) {}

std::vector<Tensor> FusedBiasGeLUBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("FusedBiasGeLUBackward: no gradients provided");
    }

    const Tensor& grad_output = grads[0];
    const Tensor& input = saved_input_;
    const Tensor& bias = saved_bias_;

    int64_t hidden_dim = input.shape().dims.back();
    int64_t batch_size = input.numel() / hidden_dim;

    Tensor grad_input(input.shape(), input.opts());
    Tensor grad_bias = Tensor::zeros(bias.shape(), bias.opts());

    device::set_cuda_device(input.device().index);
    cuda::fused_bias_gelu_backward_cuda(
        grad_output.data<float>(), input.data<float>(), bias.data<float>(),
        grad_input.data<float>(), grad_bias.data<float>(),
        batch_size, hidden_dim);

    return {grad_input, grad_bias};
}

} // namespace autograd
} // namespace OwnTensor
