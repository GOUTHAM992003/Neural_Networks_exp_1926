#pragma once
#include <cstdint>
#include "dtype/Types.h"

namespace OwnTensor {
namespace cuda {

// =============================================================================
// CUDA Activation Kernel Declarations
//
// All activation forward + backward CUDA launchers.
// Defined in: src/Kernels/cuda/ActivationKernels.cu
//
// Pattern:
//   - Forward kernels are templated (fp32/fp16/bf16) where supported
//   - Backward kernels have explicit overloads per dtype
//   - All compute in fp32 internally (upcast for fp16/bf16)
//   - Forward fp32 uses float4 vectorization where available
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// 1. ReLU
//    Forward:  (x + |x|) * 0.5  (NaN-propagating)    — fp32, fp16, bf16
//    Backward: grad * (x > 0)                         — fp32, fp16, bf16
// ─────────────────────────────────────────────────────────────────────────────

// Forward
void relu_forward_cuda(const float *input, float *output, int64_t numel);
void relu_forward_cuda(const float16_t *input, float16_t *output, int64_t numel);
void relu_forward_cuda(const bfloat16_t *input, bfloat16_t *output, int64_t numel);

// Backward
void relu_backward_cuda(const float *grad_output, const float *input, float *grad_input, int64_t numel);
void relu_backward_cuda(const float16_t *grad_output, const float16_t *input, float16_t *grad_input, int64_t numel);
void relu_backward_cuda(const bfloat16_t *grad_output, const bfloat16_t *input, bfloat16_t *grad_input, int64_t numel);

// ─────────────────────────────────────────────────────────────────────────────
// 2. GeLU (fused, tanh approximation)
//    Forward:  0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//              — fp32 (scalar + float4 vectorized), fp16, bf16
//    Backward: chain rule with sech^2                 — fp32, fp16, bf16
// ─────────────────────────────────────────────────────────────────────────────

// Forward (templated: instantiated for float, __half, __nv_bfloat16)
// For fp32 with numel >= 1024 && numel % 4 == 0: uses float4 vectorized kernel
// Otherwise: scalar kernel with #pragma unroll 4
template <typename T>
void fused_gelu_cuda(const T *input, T *output, int64_t numel);

// Backward
void fused_gelu_backward_cuda(const float *grad_output, const float *input, float *grad_input, int64_t numel);
void fused_gelu_backward_cuda(const float16_t *grad_output, const float16_t *input, float16_t *grad_input, int64_t numel);
void fused_gelu_backward_cuda(const bfloat16_t *grad_output, const bfloat16_t *input, bfloat16_t *grad_input, int64_t numel);

// ─────────────────────────────────────────────────────────────────────────────
// 3. Sigmoid
//    Forward:  1 / (1 + exp(-x))                     — fp32, fp16, bf16
//    Backward: grad * sigmoid(x) * (1 - sigmoid(x))  — fp32, fp16, bf16
//              (saves output, not input)
// ─────────────────────────────────────────────────────────────────────────────

// Forward (templated: instantiated for float, __half, __nv_bfloat16)
template <typename T>
void sigmoid_forward_cuda(const T *input, T *output, int64_t numel);

// Backward
void sigmoid_backward_cuda(const float *grad_output, const float *output, float *grad_input, int64_t numel);
void sigmoid_backward_cuda(const float16_t *grad_output, const float16_t *output, float16_t *grad_input, int64_t numel);
void sigmoid_backward_cuda(const bfloat16_t *grad_output, const bfloat16_t *output, bfloat16_t *grad_input, int64_t numel);

// ─────────────────────────────────────────────────────────────────────────────
// 4. Softmax (along last dimension)
//    Forward:  exp(x - max) / sum(exp(x - max))       — fp32, fp16, bf16
//    Backward: s * (grad - sum(grad * s, dim))         — fp32, fp16, bf16
// ─────────────────────────────────────────────────────────────────────────────

// Forward — fp32 (shared-memory reduction kernel)
void softmax_forward_cuda(const float *input, float *output, int64_t rows, int64_t cols);

// Forward — fp16/bf16 (templated mixed-precision: compute in fp32, store as T)
template <typename T>
void softmax_forward_cuda_typed(const T *input, T *output, int64_t rows, int64_t cols);

// Forward — online softmax variant (single-pass, fp32 only)
void softmax_forward_cuda_online(const float *input, float *output, int64_t rows, int64_t cols);
void softmaxonline_forward_cuda(const float *input, float *output, int64_t rows, int64_t cols);

// Backward
void softmax_backward_cuda(const float *grad_output, const float *output, float *grad_input, int64_t rows, int64_t cols);
void softmax_backward_cuda(const float16_t *grad_output, const float16_t *output, float16_t *grad_input, int64_t rows, int64_t cols);
void softmax_backward_cuda(const bfloat16_t *grad_output, const bfloat16_t *output, bfloat16_t *grad_input, int64_t rows, int64_t cols);

// ─────────────────────────────────────────────────────────────────────────────
// 5. SwiGLU
//    Forward:  swish(A) * B  where input = [A | B]    — fp32, fp16, bf16
//    Backward: dA = grad * B * (sig + A*sig*(1-sig))
//              dB = grad * swish(A)                    — fp32, fp16, bf16
// ─────────────────────────────────────────────────────────────────────────────

// Forward (templated: instantiated for float, __half, __nv_bfloat16)
template <typename T>
void swiglu_forward_cuda(const T *input, T *output, int64_t rows, int64_t hidden);

// Backward
void swiglu_backward_cuda(const float *grad_out, const float *input, float *grad_input, int64_t rows, int64_t hidden);
void swiglu_backward_cuda(const float16_t *grad_out, const float16_t *input, float16_t *grad_input, int64_t rows, int64_t hidden);
void swiglu_backward_cuda(const bfloat16_t *grad_out, const bfloat16_t *input, bfloat16_t *grad_input, int64_t rows, int64_t hidden);

// ─────────────────────────────────────────────────────────────────────────────
// 6. Fused Bias + GeLU
//    Forward:  gelu(input + bias)                     — fp32, fp16, bf16
//              Uses 2D grid (no modulo), shared memory for bias, float4/half2
//    Backward: two-pass design (fp32 only for now):
//              Pass 1: element-parallel grad_input (coalesced, no atomics)
//              Pass 2: column-per-block grad_bias (shared-mem reduction)
// ─────────────────────────────────────────────────────────────────────────────

// Forward
void fused_bias_gelu_cuda(const float *input, const float *bias, float *output,
                          int64_t batch_size, int64_t hidden_dim);
void fused_bias_gelu_cuda(const float16_t *input, const float16_t *bias, float16_t *output,
                          int64_t batch_size, int64_t hidden_dim);
void fused_bias_gelu_cuda(const bfloat16_t *input, const bfloat16_t *bias, bfloat16_t *output,
                          int64_t batch_size, int64_t hidden_dim);

// Backward (fp32 only for now — fp16/bf16 to be added with mixed precision)
void fused_bias_gelu_backward_cuda(const float *grad_output, const float *input, const float *bias,
                                   float *grad_input, float *grad_bias,
                                   int64_t batch_size, int64_t hidden_dim);

} // namespace cuda
} // namespace OwnTensor
