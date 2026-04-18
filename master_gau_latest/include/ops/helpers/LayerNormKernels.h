#pragma once
#include <cuda_runtime.h>
#include "dtype/Types.h"

namespace OwnTensor {
namespace cuda {

// Forward LayerNorm — mean/rstd always float (statistics accumulate in fp32)
// Float32
void layer_norm_forward_cuda(
    const float* x, const float* gamma, const float* beta,
    float* y, float* mean, float* rstd,
    int rows, int cols, float eps);

// Float16
void layer_norm_forward_cuda(
    const float16_t* x, const float16_t* gamma, const float16_t* beta,
    float16_t* y, float* mean, float* rstd,
    int rows, int cols, float eps);

// Bfloat16
void layer_norm_forward_cuda(
    const bfloat16_t* x, const bfloat16_t* gamma, const bfloat16_t* beta,
    bfloat16_t* y, float* mean, float* rstd,
    int rows, int cols, float eps);

// Backward LayerNorm — grad_gamma/grad_beta always float (weight grads accumulate in fp32)
// Float32
void layer_norm_backward_cuda(
    const float* grad_y, const float* x, const float* mean, const float* rstd, const float* gamma,
    float* grad_x, float* grad_gamma, float* grad_beta, int rows, int cols);

// Float16
void layer_norm_backward_cuda(
    const float16_t* grad_y, const float16_t* x, const float* mean, const float* rstd, const float16_t* gamma,
    float16_t* grad_x, float* grad_gamma, float* grad_beta, int rows, int cols);

// Bfloat16
void layer_norm_backward_cuda(
    const bfloat16_t* grad_y, const bfloat16_t* x, const float* mean, const float* rstd, const bfloat16_t* gamma,
    bfloat16_t* grad_x, float* grad_gamma, float* grad_beta, int rows, int cols);

} // namespace cuda
} // namespace OwnTensor
