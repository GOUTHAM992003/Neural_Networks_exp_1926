#include "autograd/operations/ActivationOps.h"
#include "autograd/backward/ActivationBackward.h"
#include "device/DeviceCore.h"
#include "autograd/backward/TrilBackward.h"
#include "autograd/ops_template.h"
#include "device/CachingCudaAllocator.h"
#include "ops/ScalarOps.h"
#include "ops/TensorOps.h"
#include "ops/UnaryOps/Exponents.h"
#include "ops/UnaryOps/Reduction.h"
#include "ops/UnaryOps/Trigonometry.h"
#include "ops/helpers/ActivationKernels.h"
#include "ops/FusedKernels.cuh"
#include "ops/helpers/ConditionalOps.h"
#include "dtype/CudaTraits.h"
#include "utils/Profiler.h"
#include <cmath>
#include <cstdint>

namespace OwnTensor {
namespace autograd {

Tensor relu(const Tensor &x) {
    GraphRecordMode::record_forward("ACTIVATION: ReLU");
  if (x.device().is_cuda() && x.dtype() == Dtype::Float32) {

         Tensor output(x.shape(), TensorOptions().with_dtype(x.dtype()).with_device(x.device()));
         {
             AUTO_PROFILE_CUDA("Forward::ReLU_Forward");
             cuda::relu_forward_cuda(x.data<float>(), output.data<float>(), x.numel());
         }
         
          if (GradMode::is_enabled() && x.requires_grad()) {
             auto grad_fn = std::make_shared<ReluBackward>(x);
             grad_fn->set_next_edge(0, get_grad_edge(x));
             output.set_grad_fn(grad_fn);
             output.set_requires_grad(true);
         }
         if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(output.shape(), output.dtype());
         return output;
    }

  return make_unary_op<ReluBackward>(
      x,
      [](const Tensor &input) {
        Tensor zero =
            Tensor::zeros(input.shape(), TensorOptions()
                                             .with_dtype(input.dtype())
                                             .with_device(input.device()));
        return where(input > zero, input, zero);
      },
      x); // Pass x to ReluBackward constructor
}

Tensor gelu(const Tensor &x) {
    GraphRecordMode::record_forward("ACTIVATION: GeLU");
  // Use fused CUDA kernel for GPU tensors (6x faster)
  if (!is_float(x.dtype()))
  {
    throw std::runtime_error("Only float dtypes supported!");
  }

  if (x.device().is_cuda() ) {
    Tensor output(x.shape(), TensorOptions().with_dtype(x.dtype()).with_device(
                                 x.device()));

    {
      AUTO_PROFILE_CUDA("Forward::GeLU_Forward");
      switch (x.dtype()) {
        case Dtype::Float32:
          cuda::fused_gelu_cuda(x.data<float>(), output.data<float>(), x.numel());
          break;
        case Dtype::Float16: {
          using CudaF16 = detail::CudaNativeType<float16_t>;
          cuda::fused_gelu_cuda(
              reinterpret_cast<const CudaF16*>(x.data<float16_t>()),
              reinterpret_cast<CudaF16*>(output.data<float16_t>()),
              x.numel());
          break;
        }
        case Dtype::Bfloat16: {
          using CudaBF16 = detail::CudaNativeType<bfloat16_t>;
          cuda::fused_gelu_cuda(
              reinterpret_cast<const CudaBF16*>(x.data<bfloat16_t>()),
              reinterpret_cast<CudaBF16*>(output.data<bfloat16_t>()),
              x.numel());
          break;
        }
        default:
          throw std::runtime_error("GeLU CUDA: unsupported dtype");
      }
    }

    // Set up autograd if needed
    if (GradMode::is_enabled() && x.requires_grad()) {
            auto grad_fn = std::make_shared<GeLUBackward>(x);
            grad_fn->set_next_edge(0, get_grad_edge(x));
            output.set_grad_fn(grad_fn);
            output.set_requires_grad(true);
        }
    if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(output.shape(), output.dtype());
    return output;
  }

  // TODO: CPU fallback
  // Fallback to tensor ops for CPU or non-float32
  return make_unary_op<GeLUBackward>(
      x,
      [](const Tensor &input) {
        const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
        Tensor half_x = 0.5f * input;
        Tensor x_cubed = input * input * input;
        Tensor tanh_inp = sqrt_2_over_pi * (input + 0.044715f * x_cubed);
        Tensor inner_output = 1.0f + tanh(tanh_inp);
        return half_x * inner_output;
      },
      x);
}

Tensor sigmoid(const Tensor &x) {
    GraphRecordMode::record_forward("ACTIVATION: Sigmoid");
  if (!is_float(x.dtype())) {
    throw std::runtime_error("Only float dtypes supported!");
  }

  if (x.device().is_cuda()) {
    Tensor output(x.shape(), TensorOptions().with_dtype(x.dtype()).with_device(
                                 x.device()));
    {
      AUTO_PROFILE_CUDA("Forward::Sigmoid_Forward");
      switch (x.dtype()) {
        case Dtype::Float32:
          cuda::sigmoid_forward_cuda(x.data<float>(), output.data<float>(),
                                     x.numel());
          break;
        case Dtype::Float16: {
          using CudaF16 = detail::CudaNativeType<float16_t>;
          cuda::sigmoid_forward_cuda(
              reinterpret_cast<const CudaF16*>(x.data<float16_t>()),
              reinterpret_cast<CudaF16*>(output.data<float16_t>()),
              x.numel());
          break;
        }
        case Dtype::Bfloat16: {
          using CudaBF16 = detail::CudaNativeType<bfloat16_t>;
          cuda::sigmoid_forward_cuda(
              reinterpret_cast<const CudaBF16*>(x.data<bfloat16_t>()),
              reinterpret_cast<CudaBF16*>(output.data<bfloat16_t>()),
              x.numel());
          break;
        }
        default:
          throw std::runtime_error("Sigmoid CUDA: unsupported dtype");
      }
    }

    if (GradMode::is_enabled() && x.requires_grad()) {
             auto grad_fn = std::make_shared<SigmoidBackward>(output.detach());
             grad_fn->set_next_edge(0, get_grad_edge(x));
             output.set_grad_fn(grad_fn);
             output.set_requires_grad(true);
         }
    if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(output.shape(), output.dtype());
    return output;
  }

    // Compute forward and save output for backward
    Tensor exp_input = exp(x);
    Tensor denom = 1.0f + exp_input;
    Tensor output = exp_input / denom;
    
    // Build graph if needed
    if (GradMode::is_enabled() && GradMode::is_enabled() && x.requires_grad()) {
        auto grad_fn = std::make_shared<SigmoidBackward>(output.detach());
        grad_fn->set_next_edge(0, get_grad_edge(x));
        output.set_grad_fn(grad_fn);
        output.set_requires_grad(true);
    }
      if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(output.shape(), output.dtype());
    return output;
}

Tensor softmax(const Tensor& x, int64_t dim) {
    GraphRecordMode::record_forward("ACTIVATION: Softmax");
    int64_t ndim = x.ndim();
    if (dim < 0) dim += ndim;
    if (!is_float(x.dtype()))
  {
    throw std::runtime_error("Only float dtypes supported!");
  }
    // ── Fusion: tril + softmax → fused_tril_softmax ──────────────
    // If the input was produced by tril (last-dim softmax, CUDA float),
    // call the fused kernel on the original pre-tril input directly.
    if (x.device().is_cuda() && is_float(x.dtype()) && dim == ndim - 1) {
         auto grad_fn_node = x.grad_fn();
         if (grad_fn_node) {
             auto* tril_node = dynamic_cast<TrilBackward*>(grad_fn_node.get());
             if (tril_node && tril_node->has_saved_input()) {
                 Tensor original_input = tril_node->saved_input();
                 Tensor& input_mut = const_cast<Tensor&>(original_input);
                 return fused_tril_softmax(input_mut,
                                           tril_node->diagonal(),
                                           tril_node->value());
             }
         }
    }

    if (x.device().is_cuda() && is_float(x.dtype()) && dim == ndim - 1) {
         Tensor output(x.shape(), TensorOptions().with_dtype(x.dtype()).with_device(x.device()));

         int64_t cols = x.shape().dims.back();
         int64_t rows = x.numel() / cols;
         {
             AUTO_PROFILE_CUDA("Forward::Softmax_Forward");
             switch (x.dtype()) {
               case Dtype::Float32:
                 cuda::softmax_forward_cuda(x.data<float>(), output.data<float>(), rows, cols);
                 break;
               case Dtype::Float16: {
                 using CudaF16 = detail::CudaNativeType<float16_t>;
                 cuda::softmax_forward_cuda_typed(
                     reinterpret_cast<const CudaF16*>(x.data<float16_t>()),
                     reinterpret_cast<CudaF16*>(output.data<float16_t>()),
                     rows, cols);
                 break;
               }
               case Dtype::Bfloat16: {
                 using CudaBF16 = detail::CudaNativeType<bfloat16_t>;
                 cuda::softmax_forward_cuda_typed(
                     reinterpret_cast<const CudaBF16*>(x.data<bfloat16_t>()),
                     reinterpret_cast<CudaBF16*>(output.data<bfloat16_t>()),
                     rows, cols);
                 break;
               }
               default:
                 throw std::runtime_error("Softmax CUDA: unsupported dtype");
             }
         }

         if (GradMode::is_enabled() && x.requires_grad()) {
             auto grad_fn = std::make_shared<SoftmaxBackward>(output.detach(), dim);
             grad_fn->set_next_edge(0, get_grad_edge(x));
             output.set_grad_fn(grad_fn);
             output.set_requires_grad(true);
         }
         if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(output.shape(), output.dtype());
         return output;
    }

  // Forward: exp(x - max(x)) / sum(exp(x - max(x)))
  Tensor max_val = reduce_max(x, {dim}, true);
  Tensor shifted = x - max_val;
  Tensor exp_x = exp(shifted);
  Tensor sum_exp = reduce_sum(exp_x, {dim}, true);
  Tensor output = exp_x / sum_exp;

  // Build graph if needed
  if (x.requires_grad()) {
    auto grad_fn = std::make_shared<SoftmaxBackward>(output.detach(), dim);
    grad_fn->set_next_edge(0, get_grad_edge(x));
    output.set_grad_fn(grad_fn);
    output.set_requires_grad(true);
  }
  if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(output.shape(), output.dtype());
  return output;
}

Tensor fused_tril_softmax(const Tensor& x, int64_t diagonal, double value) {
    GraphRecordMode::record_forward("ACTIVATION: FusedTrilSoftmax");
    Tensor& x_mut = const_cast<Tensor&>(x);
    Tensor result = OwnTensor::fused_tril_softmax(x_mut, diagonal, value);
    if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(result.shape(), result.dtype());
    return result;
}

Tensor dropout(const Tensor& x, float p, bool training) {
    GraphRecordMode::record_forward("ACTIVATION: Dropout");

    // During inference, just pass through
    if (!training || p == 0.0f) {
        if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(x.shape(), x.dtype());
        return x;
    }

    if (!is_float(x.dtype())) {
        throw std::runtime_error("Only float dtypes supported!");
    }

    // Use TensorOps dropout (handles both CPU and CUDA, returns {output, mask})
    auto [output, mask] = OwnTensor::dropout(x, p);

    float scale = 1.0f / (1.0f - p);

    // Set up autograd
    if (GradMode::is_enabled() && x.requires_grad()) {
        auto grad_fn = std::make_shared<DropoutBackward>(mask.detach(), scale);
        grad_fn->set_next_edge(0, get_grad_edge(x));
        output.set_grad_fn(grad_fn);
        output.set_requires_grad(true);
    }
    if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(output.shape(), output.dtype());
    return output;
}

Tensor swiglu(const Tensor &x) {
  GraphRecordMode::record_forward("ACTIVATION: SwiGLU");

  int64_t last = x.shape().dims.back();
  if (last % 2 != 0) {
    throw std::runtime_error("SwiGLU expects last dim divisible by 2");
  }
  std::cout << "Swiglu public API" << std::endl;

  int64_t hidden = last / 2;
  Shape out_shape = x.shape();
  out_shape.dims.back() = hidden;

  if (!is_float(x.dtype())) {
    throw std::runtime_error("Only float dtypes supported!");
  }

  if (x.device().is_cuda()) {
    Tensor output(out_shape, x.opts());

    int64_t rows = x.numel() / last;
    {
      AUTO_PROFILE_CUDA("Forward::SwiGLU_Forward");
      switch (x.dtype()) {
        case Dtype::Float32:
          cuda::swiglu_forward_cuda(x.data<float>(), output.data<float>(), rows,
                                    hidden);
          break;
        case Dtype::Float16: {
          using CudaF16 = detail::CudaNativeType<float16_t>;
          cuda::swiglu_forward_cuda(
              reinterpret_cast<const CudaF16*>(x.data<float16_t>()),
              reinterpret_cast<CudaF16*>(output.data<float16_t>()),
              rows, hidden);
          break;
        }
        case Dtype::Bfloat16: {
          using CudaBF16 = detail::CudaNativeType<bfloat16_t>;
          cuda::swiglu_forward_cuda(
              reinterpret_cast<const CudaBF16*>(x.data<bfloat16_t>()),
              reinterpret_cast<CudaBF16*>(output.data<bfloat16_t>()),
              rows, hidden);
          break;
        }
        default:
          throw std::runtime_error("SwiGLU CUDA: unsupported dtype");
      }
    }

    if (GradMode::is_enabled() && x.requires_grad()) {
      auto grad_fn = std::make_shared<SwiGLUBackward>(x);
      Tensor &x_mut = const_cast<Tensor &>(x);
      grad_fn->set_next_edge(0, get_grad_edge(x_mut));
      output.set_grad_fn(grad_fn);
      output.set_requires_grad(true);
    }
    if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(output.shape(), output.dtype());
    return output;
  }

  throw std::runtime_error("CPU Fallback is not implemented atm...");
}

Tensor fused_bias_gelu(const Tensor& input, const Tensor& bias) {
    GraphRecordMode::record_forward("ACTIVATION: FusedBiasGeLU");
    if (!input.device().is_cuda() || input.dtype() != Dtype::Float32) {
        throw std::runtime_error("fused_bias_gelu: only CUDA float32 supported");
    }

    int64_t hidden_dim = input.shape().dims.back();
    int64_t batch_size = input.numel() / hidden_dim;

    Tensor output(input.shape(), input.opts());
    {
        AUTO_PROFILE_CUDA("Forward::FusedBiasGeLU_Forward");
        cuda::fused_bias_gelu_cuda(input.data<float>(), bias.data<float>(),
                                   output.data<float>(), batch_size, hidden_dim);
    }

    if (GradMode::is_enabled() && (input.requires_grad() || bias.requires_grad())) {
        auto grad_fn = std::make_shared<FusedBiasGeLUBackward>(input, bias);
        Tensor& input_mut = const_cast<Tensor&>(input);
        Tensor& bias_mut  = const_cast<Tensor&>(bias);
        grad_fn->set_next_edge(0, get_grad_edge(input_mut));
        grad_fn->set_next_edge(1, get_grad_edge(bias_mut));
        output.set_grad_fn(grad_fn);
        output.set_requires_grad(true);
    }
    if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(output.shape(), output.dtype());
    return output;
}

} // namespace autograd
} // namespace OwnTensor
