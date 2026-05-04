
// #include "autograd/operations/AttentionOps.h"
// #include "autograd/operations/MatrixOps.h"
// #include "autograd/operations/ActivationOps.h"
// #include "autograd/operations/ReshapeOps.h"
// #include "autograd/operations/BinaryOps.h"
// #include "autograd/ops_template.h"
// #include "checkpointing/GradMode.h"
// #include "utils/Profiler.h"

// #ifdef WITH_CUDA
// #include "ops/helpers/AttentionKernels.h"
// #include "autograd/backward/AttentionBackward.h"
// #endif

// #include <cmath>
// #include <limits>
// #include <stdexcept>

// namespace OwnTensor {
// namespace autograd {

// // ============================================================================
// // Math Backend: composes existing autograd ops
// // ============================================================================

// static Tensor sdpa_math(
//     const Tensor& query,
//     const Tensor& key,
//     const Tensor& value,
//     bool is_causal)
// {
//     // std::cout<<"hi"<<std::endl;
//     // query, key, value: (B, nh, T, hd)
//     if(query.device() != key.device() && query.device()!= value.device())
//     {
    
//         throw std::runtime_error(
//             "Device Mismatch, Q, K, V do not reside in the same device");
    
//     }
//     int64_t hd = query.shape().dims[3];
//     float scale = 1.0f / std::sqrt(static_cast<float>(hd));
//     // std::cout<<scale<<std::endl;
//     // Scale query
//     Tensor scale_t = Tensor::full(Shape{{1}},
//         TensorOptions().with_dtype(query.dtype()).with_device(query.device()),
//         scale);
//     Tensor q_scaled = autograd::mul(query, scale_t);

//     // Q @ K^T -> (B, nh, T, T)
//     Tensor k_t = autograd::transpose(key, -2, -1);
//     Tensor attn_weights = autograd::matmul(q_scaled, k_t);

//     // Apply causal mask + softmax
//     Tensor attn_probs;
//     if (is_causal) {
//         float neg_inf = -std::numeric_limits<float>::infinity();
//         attn_probs = autograd::fused_tril_softmax(attn_weights, 0, neg_inf);
//     } else {
//         attn_probs = autograd::softmax(attn_weights, -1);
//     }

//     // Attn @ V -> (B, nh, T, hd)
//     return autograd::matmul(attn_probs, value);
// }

// // ============================================================================
// // Memory-Efficient Backend (requires CUDA)
// // ============================================================================

// #ifdef WITH_CUDA
// static Tensor sdpa_memory_efficient(
//     const Tensor& query,
//     const Tensor& key,
//     const Tensor& value,
//     bool is_causal)
// {
//     if(query.device() != key.device() && query.device()!= value.device())
//     {
//         throw std::runtime_error("Device Mismatch, Q, K, V do not reside in the same device");
//     }
//     if (!query.device().is_cuda() || query.dtype() != Dtype::Float32) {
//         throw std::runtime_error(
//             "Memory-efficient attention requires CUDA float32 tensors");
//     }

//     int64_t B  = query.shape().dims[0];
//     int64_t nh = query.shape().dims[1];
//     int64_t T  = query.shape().dims[2];
//     int64_t hd = query.shape().dims[3];

//     //* print the dimensions
//     // printf("The dimensions of Q, K, V, O are: {%d, %d, %d, %d}\n", B, nh, T, hd);
//     // exit(1);
//     //* Kernel requires contiguous (B*nh, T, hd) layout — make contiguous if needed
//     // Tensor q_contig = query.is_contiguous() ? query : query.contiguous();
//     // Tensor k_contig = key.is_contiguous()   ? key   : key.contiguous();
//     // Tensor v_contig = value.is_contiguous() ? value : value.contiguous();

//     auto opts = TensorOptions().with_dtype(Dtype::Float32).with_device(query.device());

//     // Allocate output and LSE — NO T×T allocation anywhere
//     Tensor output = Tensor::empty(Shape{{B, nh, T, hd}}, opts);
//     Tensor lse    = Tensor::empty(Shape{{B, nh, T}}, opts);

//     // Single fused kernel: Q @ K^T → scale → mask → online softmax → @ V
//     // Never materializes the T×T attention matrix.
//     cuda::mem_efficient_attn_forward(
//         query.data<float>(), key.data<float>(), value.data<float>(),
//         output.data<float>(), lse.data<float>(),
//         B, nh, T, hd, is_causal);

//     // Build autograd graph with memory-efficient backward
//     if (GradMode::is_enabled() &&
//         (query.requires_grad() || key.requires_grad() || value.requires_grad()))
//     {
//         // std::cout<<"inside grad mode"<<std::endl;
//         auto grad_fn = std::make_shared<MemEfficientAttentionBackward>(
//             query.detach(), key.detach(), value.detach(),
//             output.detach(), lse.detach(),
//             B, nh, T, hd, is_causal);

//         Tensor& q_mut = const_cast<Tensor&>(query);
//         Tensor& k_mut = const_cast<Tensor&>(key);
//         Tensor& v_mut = const_cast<Tensor&>(value);

//         if (query.requires_grad()) {
//             grad_fn->set_next_edge(0, get_grad_edge(q_mut));
//         }
//         if (key.requires_grad()) {
//             grad_fn->set_next_edge(1, get_grad_edge(k_mut));
//         }
//         if (value.requires_grad()) {
//             grad_fn->set_next_edge(2, get_grad_edge(v_mut));
//         }

//         output.set_grad_fn(grad_fn);
//         output.set_requires_grad(true);
//     }

//     return output;
// }
// #endif

// // ============================================================================
// // Public Dispatch
// // ============================================================================

// Tensor scaled_dot_product_attention(
//     const Tensor& query,
//     const Tensor& key,
//     const Tensor& value,
//     bool is_causal,
//     SDPBackend backend)
// {
//     GraphRecordMode::record_forward("ATTENTION: scaled_dot_product_attention");

//     switch (backend) {
//         case SDPBackend::Math:
//             // std::cout << "Using Math Backend" << std::endl;
//             return sdpa_math(query, key, value, is_causal);

//         case SDPBackend::MemoryEfficient:
// #ifdef WITH_CUDA
//             // std::cout << "Using Memory-Efficient Attention Backend" << std::endl;
//             return sdpa_memory_efficient(query, key, value, is_causal);
// #else
//             throw std::runtime_error(
//                 "Memory-efficient attention requires CUDA. "
//                 "Falling back to Math backend.");
//             //std::cout << "Using Math Backend   _____1 " << std::endl;
//             return sdpa_math(query, key, value, is_causal);
// #endif

//         default:
//             throw std::runtime_error("Unknown SDPBackend");
//     }
// }

// } // namespace autograd
// } // namespace OwnTensor

#include "autograd/operations/AttentionOps.h"
#include "autograd/operations/MatrixOps.h"
#include "autograd/operations/ActivationOps.h"
#include "autograd/operations/ReshapeOps.h"
#include "autograd/operations/BinaryOps.h"
#include "autograd/ops_template.h"
#include "checkpointing/GradMode.h"
#include "utils/Profiler.h"

#ifdef WITH_CUDA
#include "ops/helpers/AttentionKernels.h"
#include "autograd/backward/AttentionBackward.h"
#endif
#include "ops/Kernels.h"
#include "autograd/backward/FusedAttnSoftmaxMatmulBackward.h"
#include "device/AllocationTracker.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace OwnTensor {
namespace autograd {

// ============================================================================
// Math Backend: composes existing autograd ops
// ============================================================================

static Tensor sdpa_math(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    bool is_causal)
{
    // std::cout<<"hi"<<std::endl;
    // query, key, value: (B, nh, T, hd)
    if(query.device() != key.device() && query.device()!= value.device())
    {
    
        throw std::runtime_error(
            "Device Mismatch, Q, K, V do not reside in the same device");
    
    }
    int64_t hd = query.shape().dims[3];
    float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    // std::cout<<scale<<std::endl;
    // Scale query
    Tensor scale_t = Tensor::full(Shape{{1}},
        TensorOptions().with_dtype(query.dtype()).with_device(query.device()),
        scale);
    Tensor q_scaled = autograd::mul(query, scale_t);

    // Q @ K^T -> (B, nh, T, T)
    Tensor k_t = autograd::transpose(key, -2, -1);
    Tensor attn_weights = autograd::matmul(q_scaled, k_t);

    // Apply causal mask + softmax
    Tensor attn_probs;
    if (is_causal) {
        float neg_inf = -std::numeric_limits<float>::infinity();
        attn_probs = autograd::fused_tril_softmax(attn_weights, 0, neg_inf);
    } else {
        attn_probs = autograd::softmax(attn_weights, -1);
    }

    // Attn @ V -> (B, nh, T, hd)
    return autograd::matmul(attn_probs, value);
}

// ============================================================================
// Memory-Efficient Backend (requires CUDA)
// ============================================================================

#ifdef WITH_CUDA
static Tensor sdpa_memory_efficient(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    bool is_causal)
{
    if(query.device() != key.device() && query.device()!= value.device())
    {
        throw std::runtime_error("Device Mismatch, Q, K, V do not reside in the same device");
    }
    if (!query.device().is_cuda() || query.dtype() != Dtype::Float32) {
        throw std::runtime_error(
            "Memory-efficient attention requires CUDA float32 tensors");
    }

    int64_t B  = query.shape().dims[0];
    int64_t nh = query.shape().dims[1];
    int64_t T  = query.shape().dims[2];
    int64_t hd = query.shape().dims[3];

    // Stride-aware path (matches PyTorch's _efficient_attention_forward):
    // kernel reads Q/K/V with per-tensor strides, so no .contiguous() copy
    // is needed for transposed views of a [B, T, H, HD] tensor. The only
    // requirement is that the last dim (HeadDim) is contiguous.
    const auto& q_strides = query.stride().strides;
    const auto& k_strides = key.stride().strides;
    const auto& v_strides = value.stride().strides;

    auto opts = TensorOptions().with_dtype(Dtype::Float32).with_device(query.device());

    // Allocate output and LSE as contiguous [B, nh, T, hd] / [B, nh, T]
    Tensor output = Tensor::empty(Shape{{B, nh, T, hd}}, opts);
    Tensor lse    = Tensor::empty(Shape{{B, nh, T}}, opts);
    const auto& o_strides = output.stride().strides;
    const auto& lse_strides = lse.stride().strides;

    // Single fused kernel: Q @ K^T → scale → mask → online softmax → @ V
    // Never materializes the T×T attention matrix.
    cuda::mem_efficient_attn_forward_tc(
        query.data<float>(), q_strides[0], q_strides[2], q_strides[1],
        key.data<float>(),   k_strides[0], k_strides[2], k_strides[1],
        value.data<float>(), v_strides[0], v_strides[2], v_strides[1],
        output.data<float>(), o_strides[0], o_strides[2], o_strides[1],
        lse.data<float>(),    lse_strides[0], lse_strides[1],
        B, nh, T, hd, is_causal, 0.0f /*dropout_p*/, nullptr /*dropout_mask*/);

    // Build autograd graph with memory-efficient backward
    if (GradMode::is_enabled() &&
        (query.requires_grad() || key.requires_grad() || value.requires_grad()))
    {
        // Save the ORIGINAL (possibly strided) views — backward also handles strides.
        auto grad_fn = std::make_shared<MemEfficientAttentionBackward>(
            query.detach(), key.detach(), value.detach(),
            output.detach(), lse.detach(),
            B, nh, T, hd, is_causal);

        Tensor& q_mut = const_cast<Tensor&>(query);
        Tensor& k_mut = const_cast<Tensor&>(key);
        Tensor& v_mut = const_cast<Tensor&>(value);

        if (query.requires_grad()) {
            grad_fn->set_next_edge(0, get_grad_edge(q_mut));
        }
        if (key.requires_grad()) {
            grad_fn->set_next_edge(1, get_grad_edge(k_mut));
        }
        if (value.requires_grad()) {
            grad_fn->set_next_edge(2, get_grad_edge(v_mut));
        }

        output.set_grad_fn(grad_fn);
        output.set_requires_grad(true);
    }

    return output;
}
#endif

// ============================================================================
// Public Dispatch
// ============================================================================

Tensor scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    bool is_causal,float dropout_p,
    SDPBackend backend)
{
    GraphRecordMode::record_forward("ATTENTION: scaled_dot_product_attention");
    TRACK_ALLOC_SCOPE("L157:autograd::scaled_dot_product_attention");
    switch (backend) {
        case SDPBackend::Math:
            // std::cout << "Using Math Backend" << std::endl;
            return sdpa_math(query, key, value, is_causal);

        case SDPBackend::MemoryEfficient:
#ifdef WITH_CUDA
            // std::cout << "Using Memory-Efficient Attention Backend" << std::endl;
            return sdpa_memory_efficient(query, key, value, is_causal);
#else
            throw std::runtime_error(
                "Memory-efficient attention requires CUDA. "
                "Falling back to Math backend.");
            //std::cout << "Using Math Backend   _____1 " << std::endl;
            return sdpa_math(query, key, value, is_causal);
#endif

        default:
            throw std::runtime_error("Unknown SDPBackend");
    }
}


// ============================================================================
// Fused tril_softmax + matmul (dedup attn_probs storage)
// ============================================================================

Tensor fused_attn_softmax_matmul(
    const Tensor& attn_weights,
    const Tensor& v,
    int64_t diagonal,
    double fill_value)
{
    GraphRecordMode::record_forward("ATTENTION: fused_attn_softmax_matmul");

    // Forward: two raw ops, no autograd nodes attached
    Tensor attn_probs;
    Tensor attn_out;
    {
        NoGradGuard no_grad;
        attn_probs = autograd::fused_tril_softmax(attn_weights, diagonal, fill_value);
        attn_out = OwnTensor::matmul(attn_probs, v);
    }

    // Wire up the single fused backward node
    if (GradMode::is_enabled() &&
        (attn_weights.requires_grad() || v.requires_grad()))
    {
        auto grad_fn = std::make_shared<FusedAttnSoftmaxMatmulBackward>(
            attn_probs.detach(), v.detach());

        if (attn_weights.requires_grad()) {
            grad_fn->set_next_edge(0, get_grad_edge(attn_weights));
        }
        if (v.requires_grad()) {
            grad_fn->set_next_edge(1, get_grad_edge(v));
        }

        attn_out.set_grad_fn(grad_fn);
        attn_out.set_requires_grad(true);
    }

    return attn_out;
}

} // namespace autograd
} // namespace OwnTensor