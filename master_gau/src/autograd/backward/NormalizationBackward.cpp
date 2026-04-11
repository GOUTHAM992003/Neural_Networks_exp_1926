#include "autograd/backward/NormalizationBackward.h"
#include "ops/UnaryOps/Normalizations.h"
#include <stdexcept>

namespace OwnTensor {
namespace autograd {

// ============================================================================
// LayerNormBackward — thin wrapper, delegates to layer_norm_backward()
// ============================================================================
std::vector<Tensor> LayerNormBackward::apply(std::vector<Tensor>&& grads) {
    Tensor grad_output = grads[0];
    Tensor input  = input_.unpack(shared_from_this());
    Tensor mean   = mean_.unpack(shared_from_this());
    Tensor rstd   = rstd_.unpack(shared_from_this());
    Tensor weight;
    if (weight_.defined()) weight = weight_.unpack(shared_from_this());

    auto result = layer_norm_backward(
        grad_output, input, mean, rstd, weight, normalized_shape_, eps_);

    return {result.grad_input, result.grad_weight, result.grad_bias};
}

// ============================================================================
// RMSNormBackward — thin wrapper, delegates to rms_norm_backward()
// ============================================================================
std::vector<Tensor> RMSNormBackward::apply(std::vector<Tensor>&& grads) {
    Tensor grad_output = grads[0];
    Tensor input = input_.unpack(shared_from_this());
    Tensor rstd  = rstd_.unpack(shared_from_this());
    Tensor weight;
    if (weight_.defined()) weight = weight_.unpack(shared_from_this());

    auto result = rms_norm_backward(
        grad_output, input, rstd, weight, normalized_shape_, eps_);

    return {result.grad_input, result.grad_weight};
}

} // namespace autograd
} // namespace OwnTensor
