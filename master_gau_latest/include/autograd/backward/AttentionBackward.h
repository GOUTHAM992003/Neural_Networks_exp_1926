#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Backward function for memory-efficient scaled dot-product attention.
 *
 * Forward:
 *   O = softmax(Q @ K^T / sqrt(hd)) @ V
 *   (computed without materializing the full T×T attention matrix)
 *
 * Backward:
 *   Recomputes attention weights block-by-block from saved Q, K, V, and LSE.
 *   Uses separate kernels for dQ and dK/dV.
 *
 *   dS[i,j] = dO[i] @ V[j]^T
 *   D[i]    = dO[i] . O[i]
 *   dP[i,j] = p[i,j] * (dS[i,j] - D[i])       where p[i,j] = exp(s[i,j] - LSE[i])
 *   dQ[i]   = scale * sum_j dP[i,j] * K[j]
 *   dK[j]   = scale * sum_i dP[i,j] * Q[i]
 *   dV[j]   = sum_i p[i,j] * dO[i]
 *
 * Saves: Q, K, V (detached), O (detached), LSE (log-sum-exp per row)
 * Has 3 inputs: Q (edge 0), K (edge 1), V (edge 2)
 */
class MemEfficientAttentionBackward : public Node {
private:
    Tensor saved_query_;
    Tensor saved_key_;
    Tensor saved_value_;
    Tensor saved_output_;
    Tensor saved_lse_;
    int64_t B_, nh_, T_, hd_;
    bool is_causal_;

public:
    MemEfficientAttentionBackward(
        const Tensor& query, const Tensor& key, const Tensor& value,
        const Tensor& output, const Tensor& lse,
        int64_t B, int64_t nh, int64_t T, int64_t hd,
        bool is_causal);

    const char* name() const override { return "MemEfficientAttentionBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
    void release_saved_variables() override {
        saved_query_  = Tensor();
        saved_key_    = Tensor();
        saved_value_  = Tensor();
        saved_output_ = Tensor();
        saved_lse_    = Tensor();
    }
};

} // namespace autograd
} // namespace OwnTensor