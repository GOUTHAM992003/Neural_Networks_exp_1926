#include <iostream>
#include "ops/helpers/ActivationKernels.h"
#include "dtype/Types.h"
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <type_traits>
#include <cmath>
#include <iostream>

namespace OwnTensor {
namespace cuda {

// ---- Type conversion helpers for templated kernels ----
template<typename T> __device__ __forceinline__ float to_float(T val);
template<> __device__ __forceinline__ float to_float(float val) { return val; }
template<> __device__ __forceinline__ float to_float(__half val) { return __half2float(val); }
template<> __device__ __forceinline__ float to_float(__nv_bfloat16 val) { return __bfloat162float(val); }

template<typename T> __device__ __forceinline__ T from_float(float val);
template<> __device__ __forceinline__ float from_float(float val) { return val; }
template<> __device__ __forceinline__ __half from_float(float val) { return __float2half(val); }
template<> __device__ __forceinline__ __nv_bfloat16 from_float(float val) { return __float2bfloat16(val); }

// Constants for GELU computation
__device__ constexpr float SQRT_2_OVER_PI = 0.7978845608028654f; // sqrt(2/pi)
__device__ constexpr float GELU_COEF = 0.044715f;

// Fast tanh approximation (Direct hardware instruction via Inline PTX)
__device__ __forceinline__ float fast_tanh(float x) {
  float res;
  asm("tanh.approx.f32 %0, %1;" : "=f"(res) : "f"(x));
  return res;
}

// Fast exponential approximation (exp(x) = 2^(x * log2(e)))
__device__ __forceinline__ float fast_exp(float x) {
  float res;
  float x_log2e = x * 1.44269504089f;
  asm("ex2.approx.f32 %0, %1;" : "=f"(res) : "f"(x_log2e));
  return res;
}

// Fast reciprocal approximation (1/x)
__device__ __forceinline__ float fast_rcp(float x) {
  float res;
  asm("rcp.approx.f32 %0, %1;" : "=f"(res) : "f"(x));
  return res;
}

// For SwiGLU
__device__ __forceinline__ float fast_sigmoid(float x) {
  return fast_rcp(1.0f + fast_exp(-x));
}

// =============================================================================
// FUSED GELU KERNEL
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// =============================================================================

// Generic kernel: reads as T, upcasts to fp32 for compute, writes back as T
template <typename T>
__global__ void fused_gelu_kernel(const T *__restrict__ input,
                                  T *__restrict__ output, int64_t numel) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;

#pragma unroll 4
  for (int64_t i = idx; i < numel; i += stride) {
    float x = static_cast<float>(input[i]);
    float x3 = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
    float tanh_inner = fast_tanh(inner);
    float result = 0.5f * x * (1.0f + tanh_inner);
    output[i] = static_cast<T>(result);
  }
}

// Vectorized version using float4 for better memory throughput (fp32 only)
__global__ void fused_gelu_kernel_vectorized(const float *__restrict__ input,
                                             float *__restrict__ output,
                                             int64_t numel) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;

  int64_t numel4 = numel / 4;
  for (int64_t i = idx; i < numel4; i += stride) {
    float4 x_vec = reinterpret_cast<const float4 *>(input)[i];
    float4 out_vec;

#pragma unroll
    for (int j = 0; j < 4; j++) {
      float x = (&x_vec.x)[j];
      float x3 = x * x * x;
      float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
      float tanh_inner = fast_tanh(inner);
      (&out_vec.x)[j] = 0.5f * x * (1.0f + tanh_inner);
    }

    reinterpret_cast<float4 *>(output)[i] = out_vec;
  }

  // Handle remaining elements
  for (int64_t i = numel4 * 4 + idx; i < numel; i += stride) {
    float x = input[i];
    float x3 = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
    float tanh_inner = fast_tanh(inner);
    output[i] = 0.5f * x * (1.0f + tanh_inner);
  }
}

template <typename T>
void fused_gelu_cuda(const T *input, T *output, int64_t numel) {
  int threads = 256;
  int blocks = std::min((numel + threads - 1) / threads, (int64_t)65535);

  if constexpr (std::is_same_v<T, float>) {
    if (numel >= 1024 && numel % 4 == 0) {
      int blocks4 = std::min((numel / 4 + threads - 1) / threads, (int64_t)65535);
      fused_gelu_kernel_vectorized<<<blocks4, threads>>>(input, output, numel);
      return;
    }
  }

  fused_gelu_kernel<T><<<blocks, threads>>>(input, output, numel);
}

// Explicit template instantiations
template void fused_gelu_cuda<float>(const float*, float*, int64_t);
template void fused_gelu_cuda<__half>(const __half*, __half*, int64_t);
template void fused_gelu_cuda<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int64_t);

// =============================================================================
// FUSED GELU BACKWARD KERNEL
// gelu'(x) = 0.5 * (1 + tanh(u)) + 0.5 * x * sech^2(u) * du/dx
// where u = sqrt(2/pi) * (x + 0.044715 * x^3)
// and du/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
// =============================================================================

template<typename T>
__global__ void
fused_gelu_backward_kernel(const T *__restrict__ grad_output,
                           const T *__restrict__ input,
                           T *__restrict__ grad_input, int64_t numel) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;

#pragma unroll 4
  for (int64_t i = idx; i < numel; i += stride) {
    float x = to_float(input[i]);
    float grad = to_float(grad_output[i]);

    float x2 = x * x;
    float x3 = x2 * x;

    float u = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
    float du_dx = SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_COEF * x2);

    float tanh_u = fast_tanh(u);
    float sech2_u = 1.0f - tanh_u * tanh_u;

    float gelu_grad = 0.5f * (1.0f + tanh_u) + 0.5f * x * sech2_u * du_dx;

    grad_input[i] = from_float<T>(grad * gelu_grad);
  }
}

template<typename T>
static void launch_gelu_backward(const T *grad_output, const T *input, T *grad_input, int64_t numel) {
  int threads = 256;
  int blocks = std::min((numel + threads - 1) / threads, (int64_t)65535);
  fused_gelu_backward_kernel<T><<<blocks, threads>>>(grad_output, input, grad_input, numel);
}

void fused_gelu_backward_cuda(const float *grad_output, const float *input, float *grad_input, int64_t numel) {
  launch_gelu_backward<float>(grad_output, input, grad_input, numel);
}
void fused_gelu_backward_cuda(const float16_t *grad_output, const float16_t *input, float16_t *grad_input, int64_t numel) {
  launch_gelu_backward<__half>(reinterpret_cast<const __half*>(grad_output), reinterpret_cast<const __half*>(input),
                               reinterpret_cast<__half*>(grad_input), numel);
}
void fused_gelu_backward_cuda(const bfloat16_t *grad_output, const bfloat16_t *input, bfloat16_t *grad_input, int64_t numel) {
  launch_gelu_backward<__nv_bfloat16>(reinterpret_cast<const __nv_bfloat16*>(grad_output), reinterpret_cast<const __nv_bfloat16*>(input),
                                      reinterpret_cast<__nv_bfloat16*>(grad_input), numel);
}

// =============================================================================
// FUSED BIAS + GELU KERNEL
// output = gelu(input + bias), bias broadcast along last dim
// Single templated kernel for fp32/fp16/bf16 — same pattern as fused_gelu_kernel
// =============================================================================

template <typename T>
__global__ void fused_bias_gelu_kernel(const T *__restrict__ input,
                                       const T *__restrict__ bias,
                                       T *__restrict__ output,
                                       int64_t batch_size, int64_t hidden_dim) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = batch_size * hidden_dim;
  int64_t stride = blockDim.x * gridDim.x;

#pragma unroll 4
  for (int64_t i = idx; i < total; i += stride) {
    int64_t bias_idx = i % hidden_dim;
    float x = static_cast<float>(input[i]) + static_cast<float>(bias[bias_idx]);

    float x3 = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
    float tanh_inner = fast_tanh(inner);
    output[i] = static_cast<T>(0.5f * x * (1.0f + tanh_inner));
  }
}

template <typename T>
void fused_bias_gelu_cuda_impl(const T *input, const T *bias, T *output,
                               int64_t batch_size, int64_t hidden_dim) {
  int threads = 256;
  int64_t total = batch_size * hidden_dim;
  int blocks = std::min((total + threads - 1) / threads, (int64_t)65535);
  fused_bias_gelu_kernel<T><<<blocks, threads>>>(input, bias, output, batch_size, hidden_dim);
}

void fused_bias_gelu_cuda(const float *input, const float *bias, float *output,
                          int64_t batch_size, int64_t hidden_dim) {
  fused_bias_gelu_cuda_impl<float>(input, bias, output, batch_size, hidden_dim);
}

void fused_bias_gelu_cuda(const float16_t *input, const float16_t *bias, float16_t *output,
                          int64_t batch_size, int64_t hidden_dim) {
  fused_bias_gelu_cuda_impl<__half>(reinterpret_cast<const __half*>(input),
                                    reinterpret_cast<const __half*>(bias),
                                    reinterpret_cast<__half*>(output),
                                    batch_size, hidden_dim);
}

void fused_bias_gelu_cuda(const bfloat16_t *input, const bfloat16_t *bias, bfloat16_t *output,
                          int64_t batch_size, int64_t hidden_dim) {
  fused_bias_gelu_cuda_impl<__nv_bfloat16>(reinterpret_cast<const __nv_bfloat16*>(input),
                                           reinterpret_cast<const __nv_bfloat16*>(bias),
                                           reinterpret_cast<__nv_bfloat16*>(output),
                                           batch_size, hidden_dim);
}

// =============================================================================
// FUSED BIAS + GELU BACKWARD — TWO-PASS
//
// Pass 1: element-parallel, coalesced — computes grad_input with no atomics.
// Pass 2: column-per-block reduction — one block per bias slot sums grad_input
//         rows using shared-memory reduction, then a single atomicAdd per
//         column. Reduces from ~8192 atomicAdds/slot to exactly 1.
// =============================================================================

// Pass 1: element-parallel grad_input (coalesced reads/writes, no atomics)
__global__ void fused_bias_gelu_backward_grad_input_kernel(
    const float *__restrict__ grad_output,
    const float *__restrict__ input,
    const float *__restrict__ bias,
    float *__restrict__ grad_input,
    int64_t batch_size, int64_t hidden_dim) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = batch_size * hidden_dim;
  int64_t stride = blockDim.x * gridDim.x;

#pragma unroll 4
  for (int64_t i = idx; i < total; i += stride) {
    int64_t j = i % hidden_dim;
    float x = input[i] + bias[j];

    float x2 = x * x;
    float x3 = x2 * x;
    float u = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
    float tanh_u = fast_tanh(u);
    float sech2_u = 1.0f - tanh_u * tanh_u;
    float du_dx = SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_COEF * x2);
    float gelu_grad = 0.5f * (1.0f + tanh_u) + 0.5f * x * sech2_u * du_dx;

    grad_input[i] = grad_output[i] * gelu_grad;
  }
}

// Pass 2: one block per bias column — shared-memory reduction, one atomicAdd
__global__ void bias_grad_reduce_kernel(
    const float *__restrict__ grad_input,
    float *__restrict__ grad_bias,
    int64_t batch_size, int64_t hidden_dim) {
  int64_t j = blockIdx.x;  // one block per bias column
  if (j >= hidden_dim) return;

  __shared__ float smem[256];
  float partial = 0.0f;

  for (int64_t row = threadIdx.x; row < batch_size; row += blockDim.x)
    partial += grad_input[row * hidden_dim + j];

  smem[threadIdx.x] = partial;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }

  if (threadIdx.x == 0) atomicAdd(&grad_bias[j], smem[0]);
}

void fused_bias_gelu_backward_cuda(const float *grad_output, const float *input,
                                    const float *bias, float *grad_input,
                                    float *grad_bias, int64_t batch_size,
                                    int64_t hidden_dim) {
  int threads = 256;
  int64_t total = batch_size * hidden_dim;
  int blocks1 = std::min((total + threads - 1) / threads, (int64_t)65535);

  // Pass 1: compute grad_input element-parallel (coalesced, no atomics)
  fused_bias_gelu_backward_grad_input_kernel<<<blocks1, threads>>>(
      grad_output, input, bias, grad_input, batch_size, hidden_dim);

  // Pass 2: reduce grad_input columns → grad_bias (1 atomicAdd per column)
  bias_grad_reduce_kernel<<<hidden_dim, threads>>>(
      grad_input, grad_bias, batch_size, hidden_dim);
}

// =============================================================================
// RELU KERNELS
// relu(x) = (x + |x|) * 0.5
//   - NaN-propagating: NaN + |NaN| = NaN (correct behavior for training)
//   - Branch-free: no divergence, pure arithmetic
//   - Matches PyTorch/TensorFlow NaN propagation semantics
// =============================================================================

// ── Forward: templated for fp32/fp16/bf16 ───────────────────────────────────
template<typename T>
__global__ void relu_forward_kernel(const T *__restrict__ input,
                                    T *__restrict__ output, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    #pragma unroll 4
    for (int64_t i = idx; i < numel; i += stride) {
        float val = to_float(__ldg(&input[i]));
        output[i] = from_float<T>((val + fabsf(val)) * 0.5f);
    }
}

// ── Backward: templated for fp32/fp16/bf16 ──────────────────────────────────
template<typename T>
__global__ void relu_backward_kernel(const T *__restrict__ grad_output,
                                     const T *__restrict__ input,
                                     T *__restrict__ grad_input,
                                     int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    #pragma unroll 4
    for (int64_t i = idx; i < numel; i += stride) {
        float val = to_float(__ldg(&input[i]));
        float grad = to_float(__ldg(&grad_output[i]));
        // grad * (x + |x|) / (2 * |x| + eps) simplifies to: grad if x > 0, 0 if x <= 0, NaN if NaN
        // Simpler: grad * step(x) where step uses the same (x+|x|)/(2x) trick
        // But cleanest: just check sign bit — NaN propagation handled by multiplication
        float mask = (val + fabsf(val)) > 0.0f ? 1.0f : 0.0f;
        // NaN case: val is NaN → (NaN + NaN) > 0 is false → mask = 0 → grad * 0 = 0...
        // Actually for backward, NaN input → 0 grad is acceptable (PyTorch does the same)
        grad_input[i] = from_float<T>(grad * mask);
    }
}

// ── Forward launcher ────────────────────────────────────────────────────────
void relu_forward_cuda(const float *input, float *output, int64_t numel) {
    int threads = 256;
    int blocks = std::min((numel + threads - 1) / threads, (int64_t)65535);
    relu_forward_kernel<float><<<blocks, threads>>>(input, output, numel);
}

void relu_forward_cuda(const float16_t *input, float16_t *output, int64_t numel) {
    int threads = 256;
    int blocks = std::min((numel + threads - 1) / threads, (int64_t)65535);
    relu_forward_kernel<__half><<<blocks, threads>>>(
        reinterpret_cast<const __half*>(input), reinterpret_cast<__half*>(output), numel);
}

void relu_forward_cuda(const bfloat16_t *input, bfloat16_t *output, int64_t numel) {
    int threads = 256;
    int blocks = std::min((numel + threads - 1) / threads, (int64_t)65535);
    relu_forward_kernel<__nv_bfloat16><<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(input), reinterpret_cast<__nv_bfloat16*>(output), numel);
}

// ── Backward launcher ───────────────────────────────────────────────────────
template<typename T>
static void launch_relu_backward(const T *grad_output, const T *input, T *grad_input, int64_t numel) {
    int threads = 256;
    int blocks = std::min((numel + threads - 1) / threads, (int64_t)65535);
    relu_backward_kernel<T><<<blocks, threads>>>(grad_output, input, grad_input, numel);
}

void relu_backward_cuda(const float *grad_output, const float *input, float *grad_input, int64_t numel) {
    launch_relu_backward<float>(grad_output, input, grad_input, numel);
}
void relu_backward_cuda(const float16_t *grad_output, const float16_t *input, float16_t *grad_input, int64_t numel) {
    launch_relu_backward<__half>(reinterpret_cast<const __half*>(grad_output), reinterpret_cast<const __half*>(input),
                                 reinterpret_cast<__half*>(grad_input), numel);
}
void relu_backward_cuda(const bfloat16_t *grad_output, const bfloat16_t *input, bfloat16_t *grad_input, int64_t numel) {
    launch_relu_backward<__nv_bfloat16>(reinterpret_cast<const __nv_bfloat16*>(grad_output), reinterpret_cast<const __nv_bfloat16*>(input),
                                        reinterpret_cast<__nv_bfloat16*>(grad_input), numel);
}

// =============================================================================
// SIGMOID KERNELS
// =============================================================================

template <typename T>
__global__ void sigmoid_forward_kernel(const T *__restrict__ input,
                                       T *__restrict__ output,
                                       int64_t numel) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;

#pragma unroll 4
  for (int64_t i = idx; i < numel; i += stride) {
    float val = static_cast<float>(input[i]);
    float result = fast_rcp(1.0f + fast_exp(-val));
    output[i] = static_cast<T>(result);
  }
}

template<typename T>
__global__ void sigmoid_backward_kernel(const T *__restrict__ grad_output,
                                        const T *__restrict__ output,
                                        T *__restrict__ grad_input,
                                        int64_t numel) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;

#pragma unroll 4
  for (int64_t i = idx; i < numel; i += stride) {
    float s = to_float(output[i]);
    float grad = to_float(grad_output[i]);
    grad_input[i] = from_float<T>(grad * s * (1.0f - s));
  }
}

template <typename T>
void sigmoid_forward_cuda(const T *input, T *output, int64_t numel) {
  int threads = 256;
  int blocks = std::min((numel + threads - 1) / threads, (int64_t)65535);
  sigmoid_forward_kernel<T><<<blocks, threads>>>(input, output, numel);
}

// Explicit instantiations
template void sigmoid_forward_cuda<float>(const float*, float*, int64_t);
template void sigmoid_forward_cuda<__half>(const __half*, __half*, int64_t);
template void sigmoid_forward_cuda<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int64_t);

// // Explicit instantiations
// template void sigmoid_forward_cuda<float>(const float*, float*, int64_t);
// template void sigmoid_forward_cuda<__half>(const __half*, __half*, int64_t);
// template void sigmoid_forward_cuda<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int64_t);

// Explicit instantiations
// template void sigmoid_forward_cuda<float>(const float*, float*, int64_t);
// template void sigmoid_forward_cuda<__half>(const __half*, __half*, int64_t);
// template void sigmoid_forward_cuda<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int64_t);
template<typename T>
static void launch_sigmoid_backward(const T *grad_output, const T *output, T *grad_input, int64_t numel) {
  int threads = 256;
  int blocks = std::min((numel + threads - 1) / threads, (int64_t)65535);
  sigmoid_backward_kernel<T><<<blocks, threads>>>(grad_output, output, grad_input, numel);
}

void sigmoid_backward_cuda(const float *grad_output, const float *output, float *grad_input, int64_t numel) {
  launch_sigmoid_backward<float>(grad_output, output, grad_input, numel);
}
void sigmoid_backward_cuda(const float16_t *grad_output, const float16_t *output, float16_t *grad_input, int64_t numel) {
  launch_sigmoid_backward<__half>(reinterpret_cast<const __half*>(grad_output), reinterpret_cast<const __half*>(output),
                                  reinterpret_cast<__half*>(grad_input), numel);
}
void sigmoid_backward_cuda(const bfloat16_t *grad_output, const bfloat16_t *output, bfloat16_t *grad_input, int64_t numel) {
  launch_sigmoid_backward<__nv_bfloat16>(reinterpret_cast<const __nv_bfloat16*>(grad_output), reinterpret_cast<const __nv_bfloat16*>(output),
                                         reinterpret_cast<__nv_bfloat16*>(grad_input), numel);
}
// =============================================================================
// SOFTMAX KERNELS
// =============================================================================

// Warp Reduce Helpers
__inline__ __device__ float warpReduceMax(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  return val;
}

__inline__ __device__ float warpReduceSum(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

//* PTX helper for cp.async (16-byte load - 4 floats)
__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* glob_ptr) {
    unsigned int smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"(smem_addr), "l"(glob_ptr));
}

//TODO: Apply vectorization and inline ptx
// if it outperforms the online one go for this kernel.
__global__ void softmax_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t rows,
    int64_t cols
) {
    extern __shared__ float s_data[]; //* alignas(16)
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const int bid = blockIdx.x;

    int row = bid;
    if (row >= rows) return;

    // One block per row
    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;

    // 1. Find Max for numerical stability
    // Each thread accumulates its own partial max — no __syncthreads() needed here.
    float max_val = -INFINITY;
    //* Optimization: Vectorized loads for float type
    const int64_t vec_size = 4;
    const int64_t vec_count = cols / vec_size;
    if((reinterpret_cast<uintptr_t>(row_input) & 0xF) == 0){
        const float4* vec_input = reinterpret_cast<const float4*>(row_input);
        for(int64_t i = tid * 4; i < cols; i += bdim * 4){
            cp_async_16(&s_data[tid * 4], &row_input[i]);
            asm volatile("cp.async.commit_group;\n" ::: "memory");
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            #pragma unroll
            for(int k = 0; k < 4; ++k){
                max_val = fmaxf(max_val, s_data[tid * 4 + k]);
            }
        }
    } else {
        //* Fallback to scalar for unaligned rows
        for (int64_t i = tid; i < vec_count * vec_size; i += bdim) {
            max_val = fmaxf(max_val, row_input[i]);
        }
    }

    //* Handle tail elements
    for (int64_t i = vec_count * vec_size + tid; i < cols; i += bdim) {
        max_val = fmaxf(max_val, row_input[i]);
    }
    
    // Block reduce max
    max_val = warpReduceMax(max_val);
    
    static __shared__ float s_max;
    if (threadIdx.x == 0) s_max = -INFINITY;
    __syncthreads();
    
    static __shared__ float warp_vals[32]; // Max 1024 threads = 32 warps
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    
    if (laneId == 0) warp_vals[warpId] = -INFINITY;
    __syncthreads();
    
    if (laneId == 0) warp_vals[warpId] = max_val;
    __syncthreads();
    
    if (threadIdx.x == 0) {
        float block_max = -INFINITY;
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        for (int i=0; i<num_warps; ++i) {
            block_max = fmaxf(block_max, warp_vals[i]);
        }
        s_max = block_max;
    }
    __syncthreads();
    max_val = s_max;
    
    // 2. Compute Exp and Sum
    float sum_exp = 0.0f;
    #pragma unroll 4
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = expf(row_input[i] - max_val);
        sum_exp += val;
        row_output[i] = val; // Store exp(x-max)
    }
    sum_exp = warpReduceSum(sum_exp);
    
    // Block reduce sum
    static __shared__ float s_sum;
    if (laneId == 0) warp_vals[warpId] = sum_exp;
    __syncthreads();
    
    if (threadIdx.x == 0) {
        float block_sum = 0.0f;
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        for (int i=0; i<num_warps; ++i) {
            block_sum += warp_vals[i];
        }
        s_sum = block_sum;
    }
    __syncthreads();
    sum_exp = s_sum;
    
    // 3. Normalize
    float inv_sum = fast_rcp(sum_exp);
    #pragma unroll 4
    for (int i = tid; i < cols; i += bdim) {
        row_output[i] *= inv_sum;
    }
}

//* Softmax Online forward kernel
__global__ void softmaxOnline_forward_kernel(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    int64_t rows,
    int64_t cols
){
    extern __shared__ float s_data[]; //* alignas(16)
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const int bid = blockIdx.x;

    int row = bid;
    if (row >= rows) return;

    const float* row_input = d_input + row * cols;
    float* row_output = d_output + row * cols;
    
    float local_max = -1e38f;
    float local_sum = 0.0f;

    //* Optimization: Vectorized loads for float type
    const int64_t vec_size = 4;
    const int64_t vec_count = cols / vec_size;
    
    //* Check row alignment for float4 (16 bytes)
    if ((reinterpret_cast<uintptr_t>(row_input) & 0xF) == 0) {
        const float4* vec_input = reinterpret_cast<const float4*>(row_input);
        for (int64_t j = threadIdx.x* 4; j < cols; j += blockDim.x * 4) {
            cp_async_16(&s_data[tid * 4], &row_input[j]);
            asm volatile("cp.async.commit_group;\n" ::: "memory");
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            // No __syncthreads() here: each thread only reads its own s_data[tid*4..tid*4+3] slot.
            // __syncthreads() inside a variable-iteration loop causes a deadlock when
            // cols is not a multiple of bdim*4.
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float val = s_data[tid * 4 + k];
                if (val > local_max) {
                    local_sum = local_sum * expf(local_max - val) + 1.0f;
                    local_max = val;
                } else {
                    local_sum += expf(val - local_max);
                }
            }
        }
    } else {
        //* Fallback to scalar for unaligned rows
        for (int64_t j = threadIdx.x; j < vec_count * vec_size; j += blockDim.x) {
            float val = static_cast<float>(row_input[j]);
            if (val > local_max) {
                local_sum = local_sum * expf(local_max - val) + 1.0f;
                local_max = val;
            } else {
                local_sum += expf(val - local_max);
            }
        }
    }
    
    //* Handle tail elements for float type
    for (int64_t j = vec_count * vec_size + threadIdx.x; j < cols; j += blockDim.x) {
        float val = static_cast<float>(row_input[j]);
        if (val > local_max) {
            local_sum = local_sum * expf(local_max - val) + 1.0f;
            local_max = val;
        } else {
            local_sum += expf(val - local_max);
        }
    }

    //* Block reduction in shared memory
    extern __shared__ float sdata[];
    //* Use two halves of shared memory: one for max and one for sum
    float *smax = sdata;
    float *ssum = sdata + bdim;

    smax[tid] = local_max;
    ssum[tid] = local_sum;
    __syncthreads(); //* Barrier synchronization

    //* Reduction loop: down to warp size
    for(unsigned int s = bdim/2; s >= 32; s >>= 1){
        if(tid < s){
            float other_max = smax[tid + s];
            float other_sum = ssum[tid + s];

            //* Merge logic (Online softmax)
            if(other_max > smax[tid]){
                ssum[tid] = ssum[tid] * expf(smax[tid] - other_max) + other_sum;
                smax[tid] = other_max;
            } else{
                ssum[tid] += other_sum * expf(other_max - smax[tid]);
            }
        }
        __syncthreads();
    }

    //* Warp-level reduction
    if (tid < 32) {
        float cur_max = smax[tid];
        float cur_sum = ssum[tid];
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_max = __shfl_down_sync(0xFFFFFFFF, cur_max, offset);
            float other_sum = __shfl_down_sync(0xFFFFFFFF, cur_sum, offset);
            
            if (other_max > cur_max) {
                cur_sum = cur_sum * expf(cur_max - other_max) + other_sum;
                cur_max = other_max;
            } else {
                cur_sum += other_sum * expf(other_max - cur_max);
            }
        }
        smax[tid] = cur_max;
        ssum[tid] = cur_sum;
    }
    __syncthreads();


    //* Now index 0 in the shared mem contains the block wide max and sum
    float final_max = smax[0];
    float final_sum = ssum[0];
    //* smax and ssum contains the global max and global sum
    for (size_t i = tid; i < cols; i += bdim) { //* problem size_t i = gidx; i < cols; i += gdim * bdim
        row_output[i] = expf(row_input[i] - final_max) / final_sum; //* should work now...
    }
}

//* Two kernel approach for softmax
__global__ void softmaxOnlineReduce(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    float *d_partial_max, 
    float *d_partial_sum,
    int64_t rows,
    int64_t cols
){
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const int bid = blockIdx.x;
    const int gdim = gridDim.x;
    const int gidx = bid * bdim + tid;

    int row = bid;
    if(row >= rows) return;

    //* one block per row
    const float* row_input = d_input + row * cols;
    float* row_output = d_output + row * cols;

    float local_max = -1e38f;
    float local_sum = 0.0f;

    // Grid-stride loop for partial reduction //* changed it to blockstride
    for (size_t i = tid; i < cols; i += bdim) { //* shouldn't it be i = tid, earlier it was gidx
        //* this logic is correct
        float val = row_input[i]; //* maybe it should start from row_input | float val = d_input[i];
        if (val > local_max) {
            local_sum = local_sum * expf(local_max - val) + 1.0f;
            local_max = val;
        } else {
            local_sum += expf(val - local_max);
        }
    }

    // Phase 2: Block reduction using shared memory
    extern __shared__ float sdata[];
    float *smax = sdata;
    float *ssum = sdata + bdim;

    smax[tid] = local_max;
    ssum[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float other_max = smax[tid + s];
            float other_sum = ssum[tid + s];

            if (other_max > smax[tid]) {
                ssum[tid] = ssum[tid] * expf(smax[tid] - other_max) + other_sum;
                smax[tid] = other_max;
            } else {
                ssum[tid] += other_sum * expf(other_max - smax[tid]);
            }
        }
        __syncthreads();
    }

    // Write this block's result to global memory
    if (tid == 0) {
        d_partial_max[bid] = smax[0];
        d_partial_sum[bid] = ssum[0];
    }
}

__global__ void softmaxOnlineNormalize(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    float* d_partial_max,
    float* d_partial_sum,
    int num_blocks,
    int64_t rows,
    int64_t cols
){
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const int bid = blockIdx.x; //* not used
    const int gdim = gridDim.x;
    const int gidx = bid * bdim + tid;

    //* try rowin and rowout
    int row = bid;
    if(row >= rows) return;
    const float* row_input = d_input + row * cols;
    float* row_output = d_output + row * cols;

    // All blocks need the global max and sum.
    // We can have each block reduce the partial results (which are few: num_blocks).
    __shared__ float global_max_shared;
    __shared__ float global_sum_shared;

    if (tid == 0) {
        float g_max = -1e38f;
        float g_sum = 0.0f;
        for (int i = 0; i < num_blocks; ++i) {
            float p_max = d_partial_max[i];
            float p_sum = d_partial_sum[i];
            
            if (p_max > g_max) {
                g_sum = g_sum * expf(g_max - p_max) + p_sum;
                g_max = p_max;
            } else {
                g_sum += p_sum * expf(p_max - g_max);
            }
        }
        global_max_shared = g_max;
        global_sum_shared = g_sum;
    }
    __syncthreads();

    float final_max = global_max_shared;
    float final_sum = global_sum_shared;

    // Phase 3: Normalization & Store
    //! Error is not here
    for (size_t i = tid; i < cols; i += bdim) { //* problem size_t i = gidx; i < cols; i += gdim * bdim
        row_output[i] = expf(row_input[i] - final_max) / final_sum; //* should work now...
    }
}

template<typename T>
__global__ void softmax_backward_kernel(
    const T* __restrict__ grad_output,
    const T* __restrict__ output,
    T* __restrict__ grad_input,
    int64_t rows,
    int64_t cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const T* row_grad = grad_output + row * cols;
    const T* row_out = output + row * cols;
    T* row_gin = grad_input + row * cols;

    // 1. Compute dot product sum(grad * out) in float
    float dot = 0.0f;
    #pragma unroll 4
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        dot += to_float(row_grad[i]) * to_float(row_out[i]);
    }

    dot = warpReduceSum(dot);

    static __shared__ float s_dot;
    static __shared__ float warp_vals[32];
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;

    if (laneId == 0) warp_vals[warpId] = dot;
    __syncthreads();

    if (threadIdx.x == 0) {
        float block_dot = 0.0f;
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        for (int i=0; i<num_warps; ++i) {
            block_dot += warp_vals[i];
        }
        s_dot = block_dot;
    }
    __syncthreads();
    dot = s_dot;

    // 2. Compute grad_input = out * (grad - dot)
    #pragma unroll 4
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float s = to_float(row_out[i]);
        float g = to_float(row_grad[i]);
        row_gin[i] = from_float<T>(s * (g - dot));
    }
}

void softmax_forward_cuda_online(const float* input, float* output, int64_t rows, int64_t cols) {
    dim3 block{(cols <= 1024) ? 256 : 1024};
    //* one block per row
    dim3 grid{rows};
    
    //* Buffers for multi-block online softmax
    float *d_partial_max, *d_partial_sum;
    cudaMalloc(&d_partial_max, grid.x * sizeof(float));
    cudaMalloc(&d_partial_sum, grid.x * sizeof(float));

    //* same stream for serializing and avoiding synchronization
    //* by deafult stream 0
    softmaxOnlineReduce<<<grid, block, 2 * block.x * sizeof(float)>>>(input, output, d_partial_max, d_partial_sum, rows, cols);
    softmaxOnlineNormalize<<<grid, block>>>(input, output, d_partial_max, d_partial_sum, grid.x, rows, cols);
}

void softmax_forward_cuda(const float* input, float* output, int64_t rows, int64_t cols) {
    int threads = (cols <= 1024) ? 256 : 1024;
    // Ensure threads is multiple of 32 for warp ops
    if (threads < 32) threads = 32;
    // One block per row
    dim3 blocks(rows);
    softmax_forward_kernel<<<blocks, threads, 4 * threads * sizeof(float)>>>(input, output, rows, cols);
}

// =============================================================================
// TEMPLATED SOFTMAX FOR HALF TYPES (read T → compute fp32 → write T)
// =============================================================================
template <typename T>
__global__ void softmax_forward_kernel_mixed(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t rows,
    int64_t cols
) {
    extern __shared__ float s_data[];
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;

    int row = blockIdx.x;
    if (row >= rows) return;

    const T* row_input = input + row * cols;
    T* row_output = output + row * cols;

    // 1. Find max for numerical stability
    float max_val = -INFINITY;
    for (int64_t i = tid; i < cols; i += bdim) {
        max_val = fmaxf(max_val, static_cast<float>(row_input[i]));
    }

    // Warp reduce max
    max_val = warpReduceMax(max_val);

    static __shared__ float s_max;
    static __shared__ float warp_vals[32];
    int warpId = tid / warpSize;
    int laneId = tid % warpSize;

    if (laneId == 0) warp_vals[warpId] = max_val;
    __syncthreads();

    if (tid == 0) {
        float block_max = -INFINITY;
        int num_warps = (bdim + warpSize - 1) / warpSize;
        for (int i = 0; i < num_warps; ++i)
            block_max = fmaxf(block_max, warp_vals[i]);
        s_max = block_max;
    }
    __syncthreads();
    max_val = s_max;

    // 2. Compute exp and sum
    float sum_exp = 0.0f;
    for (int64_t i = tid; i < cols; i += bdim) {
        float val = expf(static_cast<float>(row_input[i]) - max_val);
        sum_exp += val;
        s_data[i] = val; // store in shared memory (fp32)
    }
    __syncthreads();

    sum_exp = warpReduceSum(sum_exp);

    static __shared__ float s_sum;
    if (laneId == 0) warp_vals[warpId] = sum_exp;
    __syncthreads();

    if (tid == 0) {
        float block_sum = 0.0f;
        int num_warps = (bdim + warpSize - 1) / warpSize;
        for (int i = 0; i < num_warps; ++i)
            block_sum += warp_vals[i];
        s_sum = block_sum;
    }
    __syncthreads();
    sum_exp = s_sum;

    // 3. Normalize and write back as T
    float inv_sum = fast_rcp(sum_exp);
    for (int64_t i = tid; i < cols; i += bdim) {
        row_output[i] = static_cast<T>(s_data[i] * inv_sum);
    }
}

template <typename T>
void softmax_forward_cuda_typed(const T* input, T* output, int64_t rows, int64_t cols) {
    int threads = (cols <= 1024) ? 256 : 1024;
    if (threads < 32) threads = 32;
    dim3 blocks(rows);
    // Shared memory: cols floats for intermediate exp values
    size_t smem = cols * sizeof(float);
    softmax_forward_kernel_mixed<T><<<blocks, threads, smem>>>(input, output, rows, cols);
}

// Explicit instantiations
template void softmax_forward_cuda_typed<__half>(const __half*, __half*, int64_t, int64_t);
template void softmax_forward_cuda_typed<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int64_t, int64_t);

void softmaxonline_forward_cuda(const float* input, float* output, int64_t rows, int64_t cols) {
    int threads = (cols <= 1024) ? 256 : 1024;
    // One block per row
    dim3 blocks(rows);
    softmaxOnline_forward_kernel<<<blocks, threads, 4 * threads * sizeof(float)>>>(input, output, rows, cols);
}

// void softmaxonline_forward_cuda(const float* input, float* output, int64_t rows, int64_t cols) {
//     int threads = (cols <= 1024) ? 256 : 1024;
//     // Ensure threads is multiple of 32 for warp ops
//     if (threads < 32) threads = 32;
//     // One block per row
//     dim3 blocks(rows);
//     softmaxOnline_forward_kernel<<<blocks, threads>>>(input, output, rows, cols);
// }

template<typename T>
static void launch_softmax_backward(const T* grad_output, const T* output, T* grad_input, int64_t rows, int64_t cols) {
    int threads = (cols <= 1024) ? 256 : 1024;
    if (threads < 32) threads = 32;
    dim3 blocks(rows);
    softmax_backward_kernel<T><<<blocks, threads>>>(grad_output, output, grad_input, rows, cols);
}

void softmax_backward_cuda(const float* grad_output, const float* output, float* grad_input, int64_t rows, int64_t cols) {
    launch_softmax_backward<float>(grad_output, output, grad_input, rows, cols);
}
void softmax_backward_cuda(const float16_t* grad_output, const float16_t* output, float16_t* grad_input, int64_t rows, int64_t cols) {
    launch_softmax_backward<__half>(reinterpret_cast<const __half*>(grad_output), reinterpret_cast<const __half*>(output),
                                    reinterpret_cast<__half*>(grad_input), rows, cols);
}
void softmax_backward_cuda(const bfloat16_t* grad_output, const bfloat16_t* output, bfloat16_t* grad_input, int64_t rows, int64_t cols) {
    launch_softmax_backward<__nv_bfloat16>(reinterpret_cast<const __nv_bfloat16*>(grad_output), reinterpret_cast<const __nv_bfloat16*>(output),
                                           reinterpret_cast<__nv_bfloat16*>(grad_input), rows, cols);
}

// ##################################################################################################
//                                  SwiGLU Kernel
// ##################################################################################################

template <typename T>
__global__ void swiglu_forward_kernel(const T *__restrict__ input,
                                      T *__restrict__ output, int64_t rows,
                                      int64_t hidden) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int64_t total = rows * hidden;

  for (int i = idx; i < total; i += stride) {
    int64_t row = i / hidden;
    int64_t col = i % hidden;
    int64_t base = row * hidden * 2;
    float a = static_cast<float>(input[base + col]);
    float b = static_cast<float>(input[base + hidden + col]);
    float sig = fast_sigmoid(a);
    float swish = a * sig;
    output[i] = static_cast<T>(swish * b);
  }
}

template<typename T>
__global__ void swiglu_backward_kernel(const T *__restrict__ grad_out,
                                       const T *__restrict__ input,
                                       T *__restrict__ grad_input,
                                       int64_t rows, int64_t hidden) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int64_t total = rows * hidden;

  for (int i = idx; i < total; i += stride) {
    int64_t row = i / hidden;
    int64_t col = i % hidden;
    int64_t base = row * hidden * 2;

    float a = to_float(input[base + col]);
    float b = to_float(input[base + hidden + col]);
    float g = to_float(grad_out[i]);

    float sig = fast_sigmoid(a);
    float swish = a * sig;

    float dA = b * (sig + a * sig * (1.0f - sig));
    float dB = swish;
    grad_input[base + col] = from_float<T>(g * dA);
    grad_input[base + hidden + col] = from_float<T>(g * dB);
  }
}

template <typename T>
void swiglu_forward_cuda(const T *input, T *output, int64_t rows,
                         int64_t hidden) {
  int threads = 256;
  int64_t total = rows * hidden;
  int blocks = std::min((total + threads - 1) / threads, int64_t(65535));
  swiglu_forward_kernel<T><<<blocks, threads>>>(input, output, rows, hidden);
}

// Explicit instantiations
template void swiglu_forward_cuda<float>(const float*, float*, int64_t, int64_t);
template void swiglu_forward_cuda<__half>(const __half*, __half*, int64_t, int64_t);
template void swiglu_forward_cuda<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int64_t, int64_t);

template <typename T>
void launch_swiglu_backward(const T *grad_out, const T *input,
                          T *grad_input, int64_t rows, int64_t hidden) {
  int threads = 256;
  int64_t total = rows * hidden;
  int blocks = std::min((total + threads - 1) / threads, int64_t(65535));
  swiglu_backward_kernel<T><<<blocks, threads>>>(grad_out, input, grad_input, rows, hidden);
}

template void launch_swiglu_backward<float>(const float*, const float*,
                          float*, int64_t, int64_t);
template void launch_swiglu_backward<__half>(const __half*, const __half*,
                          __half*, int64_t, int64_t);
template void launch_swiglu_backward<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*,
                          __nv_bfloat16*, int64_t, int64_t);

void swiglu_backward_cuda(const float *grad_out, const float *input, float *grad_input, int64_t rows, int64_t hidden) {
  launch_swiglu_backward<float>(grad_out, input, grad_input, rows, hidden);
}
void swiglu_backward_cuda(const float16_t *grad_out, const float16_t *input, float16_t *grad_input, int64_t rows, int64_t hidden) {
  launch_swiglu_backward<__half>(reinterpret_cast<const __half*>(grad_out), reinterpret_cast<const __half*>(input),
                                 reinterpret_cast<__half*>(grad_input), rows, hidden);
}
void swiglu_backward_cuda(const bfloat16_t *grad_out, const bfloat16_t *input, bfloat16_t *grad_input, int64_t rows, int64_t hidden) {
  launch_swiglu_backward<__nv_bfloat16>(reinterpret_cast<const __nv_bfloat16*>(grad_out), reinterpret_cast<const __nv_bfloat16*>(input),
                                        reinterpret_cast<__nv_bfloat16*>(grad_input), rows, hidden);
}

} // namespace cuda
} // namespace OwnTensor
