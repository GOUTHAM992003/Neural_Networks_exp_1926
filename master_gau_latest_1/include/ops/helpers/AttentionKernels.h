#pragma once
#include <cstdint>

namespace OwnTensor {
namespace cuda {

/**
* @brief Compute log-sum-exp per row of attention scores.
*
* Read-only kernel  256)
* @param is_causal     whether to apply lower-triangular causal mask
* @param dropout_p     dropout probability in [0, 1); pass 0 to disable
* @param dropout_mask  pre-generated mask tensor of shape [B*nh, T, T] where
*                      each element is either 0 (dropped) or 1/(1-dropout_p)
*                      (kept and rescaled); pass nullptr when dropout_p == 0
*/
void mem_efficient_attn_forward(
   const float* query,
   const float* key,
   const float* value,
   float* output,
   float* lse,
   int64_t B, int64_t nh, int64_t T, int64_t hd,
   bool is_causal,
  float dropout_p = 0.0f,
  const float* dropout_mask = nullptr
);

void mem_efficient_attn_forward_tc(
   const float* query,
   const float* key,
   const float* value,
   float* output,
   float* lse,
   int64_t B, int64_t nh, int64_t T, int64_t hd,
   bool is_causal,
  float dropout_p = 0.0f,
  const float* dropout_mask = nullptr
);
/**
* @brief Unified memory-efficient attention backward pass for dQ, dK, dV.
*
* Architecture (PyTorch CUTLASS FMHA style):
*   1. Precompute D[i] = dot(dO[i], O[i]) in a lightweight kernel (one warp per row)
*   2. Single KV-tile-centric kernel computes dQ, dK, dV together:
*      - Each block owns one KV tile, sweeps all Q rows
*      - Phase A: warp-parallel score recompute + dQ accumulation in registers
*      - Phase B: cooperative dK/dV reduction (no per-element atomics)
*      - dQ written via atomicAdd; dK/dV written once from shared memory
*
* @param D_buf    (B*nh, T) scratch buffer for precomputed D values (pre-allocated)
*/
void mem_efficient_attn_backward(
   const float* query,
   const float* key,
   const float* value,
   const float* output,
   const float* grad_output,
   const float* lse,
   float* grad_query,
   float* grad_key,
   float* grad_value,
   float* D_buf,
   int64_t B, int64_t nh, int64_t T, int64_t hd,
   bool is_causal
);

} // namespace cuda
} // namespace OwnTensor