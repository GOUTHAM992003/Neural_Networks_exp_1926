
// #include "autograd/backward/AttentionBackward.h"
// #include "ops/helpers/AttentionKernels.h"
// #include <stdexcept>

// namespace OwnTensor {
// namespace autograd {

// MemEfficientAttentionBackward::MemEfficientAttentionBackward(
//     const Tensor& query, const Tensor& key, const Tensor& value,
//     const Tensor& output, const Tensor& lse,
//     int64_t B, int64_t nh, int64_t T, int64_t hd,
//     bool is_causal)
//     : Node(3),  // 3 inputs: Q, K, V
//       saved_query_(query), saved_key_(key), saved_value_(value),
//       saved_output_(output), saved_lse_(lse),
//       B_(B), nh_(nh), T_(T), hd_(hd), is_causal_(is_causal) {}

// std::vector<Tensor> MemEfficientAttentionBackward::apply(std::vector<Tensor>&& grads) {
//     if (grads.empty()) {
//         throw std::runtime_error("MemEfficientAttentionBackward: no gradients provided");
//     }

//     const Tensor& grad_output_raw = grads[0];

//     if (!grad_output_raw.device().is_cuda() || grad_output_raw.dtype() != Dtype::Float32) {
//         throw std::runtime_error(
//             "MemEfficientAttentionBackward: only CUDA float32 tensors are supported");
//     }

//     // Kernel requires contiguous layout
//     Tensor grad_output = grad_output_raw.is_contiguous()
//         ? grad_output_raw : grad_output_raw.contiguous();

//     auto opts = TensorOptions().with_dtype(Dtype::Float32).with_device(grad_output.device());

//     // Allocate gradient tensors: same shape as Q, K, V = (B, nh, T, hd)
//     Shape qkv_shape({{B_, nh_, T_, hd_}});
//     Tensor grad_query = Tensor::zeros(qkv_shape, opts);  // zeroed — atomicAdd target
//     Tensor grad_key   = Tensor::empty(qkv_shape, opts);
//     Tensor grad_value = Tensor::empty(qkv_shape, opts);

//     // D buffer for precomputed dot(dO, O) — (B*nh, T)
//     Shape d_shape({{B_ * nh_, T_}});
//     Tensor D_buf = Tensor::empty(d_shape, opts);

//     // Single unified backward launch: precompute D + KV-tile-centric dQ/dK/dV
//     cuda::mem_efficient_attn_backward(
//         saved_query_.data<float>(),
//         saved_key_.data<float>(),
//         saved_value_.data<float>(),
//         saved_output_.data<float>(),
//         grad_output.data<float>(),
//         saved_lse_.data<float>(),
//         grad_query.data<float>(),
//         grad_key.data<float>(),
//         grad_value.data<float>(),
//         D_buf.data<float>(),
//         B_, nh_, T_, hd_, is_causal_);

//     return {grad_query, grad_key, grad_value};
// }

// } // namespace autograd
// } // namespace OwnTensor

#include "autograd/backward/AttentionBackward.h"
#include "ops/helpers/AttentionKernels.h"
#include <stdexcept>

namespace OwnTensor {
namespace autograd {

MemEfficientAttentionBackward::MemEfficientAttentionBackward(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& output, const Tensor& lse,
    int64_t B, int64_t nh, int64_t T, int64_t hd,
    bool is_causal)
    : Node(3),  // 3 inputs: Q, K, V
      saved_query_(query), saved_key_(key), saved_value_(value),
      saved_output_(output), saved_lse_(lse),
      B_(B), nh_(nh), T_(T), hd_(hd), is_causal_(is_causal) {}

std::vector<Tensor> MemEfficientAttentionBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("MemEfficientAttentionBackward: no gradients provided");
    }

    const Tensor& grad_output_raw = grads[0];

    if (!grad_output_raw.device().is_cuda() || grad_output_raw.dtype() != Dtype::Float32) {
        throw std::runtime_error(
            "MemEfficientAttentionBackward: only CUDA float32 tensors are supported");
    }

    // Kernel requires contiguous layout
    Tensor grad_output = grad_output_raw.is_contiguous()
        ? grad_output_raw : grad_output_raw.contiguous();

    Tensor q_c = saved_query_.is_contiguous() ? saved_query_ : saved_query_.contiguous();
    Tensor k_c = saved_key_.is_contiguous()   ? saved_key_   : saved_key_.contiguous();
    Tensor v_c = saved_value_.is_contiguous() ? saved_value_ : saved_value_.contiguous();
    Tensor o_c = saved_output_.is_contiguous() ? saved_output_ : saved_output_.contiguous();
    Tensor lse_c = saved_lse_.is_contiguous() ? saved_lse_ : saved_lse_.contiguous();

    auto opts = TensorOptions().with_dtype(Dtype::Float32).with_device(grad_output.device());

    // Allocate gradient tensors: same shape as Q, K, V = (B, nh, T, hd)
    Shape qkv_shape({{B_, nh_, T_, hd_}});
    Tensor grad_query = Tensor::zeros(qkv_shape, opts);  // zeroed — atomicAdd target
    Tensor grad_key   = Tensor::empty(qkv_shape, opts);
    Tensor grad_value = Tensor::empty(qkv_shape, opts);

    // D buffer for precomputed dot(dO, O) — (B*nh, T)
    Shape d_shape({{B_ * nh_, T_}});
    Tensor D_buf = Tensor::empty(d_shape, opts);

    // Single unified backward launch: precompute D + KV-tile-centric dQ/dK/dV
    cuda::mem_efficient_attn_backward(
        q_c.data<float>(),
        k_c.data<float>(),
        v_c.data<float>(),
        o_c.data<float>(),
        grad_output.data<float>(),
        lse_c.data<float>(),
        grad_query.data<float>(),
        grad_key.data<float>(),
        grad_value.data<float>(),
        D_buf.data<float>(),
        B_, nh_, T_, hd_, is_causal_);

    return {grad_query, grad_key, grad_value};
}

} // namespace autograd
} // namespace OwnTensor