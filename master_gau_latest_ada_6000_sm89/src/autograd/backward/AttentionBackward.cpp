
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

    // Stride-aware path: only the head-dim (last axis) must be unit-strided;
    // all other strides are forwarded to the kernel and used for indexing.
    // This eliminates the 6 x 50 MB .contiguous() copies that used to happen here.
    const Tensor& grad_output = grad_output_raw;

    auto opts = TensorOptions().with_dtype(Dtype::Float32).with_device(grad_output.device());

    // Allocate gradient tensors: same shape as Q, K, V = (B, nh, T, hd).
    // Allocate D buffer as [B, nh, T] (3D) so its stride-B/stride-H are well-defined.
    // dQ/dK/dV zero-init (when needed for atomicAdd) is done inside the kernel launcher.
    Shape qkv_shape({{B_, nh_, T_, hd_}});
    Tensor grad_query = Tensor::empty(qkv_shape, opts);
    Tensor grad_key   = Tensor::empty(qkv_shape, opts);
    Tensor grad_value = Tensor::empty(qkv_shape, opts);

    Shape d_shape({{B_, nh_, T_}});
    Tensor D_buf = Tensor::empty(d_shape, opts);

    // Strides — shape is [B, nh, T, HD], so (strideB, strideH, strideM) = (s0, s1, s2).
    const auto& q_s  = saved_query_.stride().strides;
    const auto& k_s  = saved_key_.stride().strides;
    const auto& v_s  = saved_value_.stride().strides;
    const auto& o_s  = saved_output_.stride().strides;
    const auto& do_s = grad_output.stride().strides;
    const auto& dq_s = grad_query.stride().strides;
    const auto& dk_s = grad_key.stride().strides;
    const auto& dv_s = grad_value.stride().strides;
    const auto& lse_s = saved_lse_.stride().strides;  // [B, nh, T]
    const auto& d_s   = D_buf.stride().strides;        // [B, nh, T]

    // Single unified backward launch: precompute D + KV-tile-centric dQ/dK/dV
    cuda::mem_efficient_attn_backward(
        saved_query_.data<float>(),  q_s[0], q_s[2], q_s[1],
        saved_key_.data<float>(),    k_s[0], k_s[2], k_s[1],
        saved_value_.data<float>(),  v_s[0], v_s[2], v_s[1],
        saved_output_.data<float>(), o_s[0], o_s[2], o_s[1],
        grad_output.data<float>(),   do_s[0], do_s[2], do_s[1],
        saved_lse_.data<float>(),    lse_s[0], lse_s[1],
        grad_query.data<float>(),    dq_s[0], dq_s[2], dq_s[1],
        grad_key.data<float>(),      dk_s[0], dk_s[2], dk_s[1],
        grad_value.data<float>(),    dv_s[0], dv_s[2], dv_s[1],
        D_buf.data<float>(),         d_s[0], d_s[1],
        B_, nh_, T_, hd_, is_causal_);

    return {grad_query, grad_key, grad_value};
}

} // namespace autograd
} // namespace OwnTensor