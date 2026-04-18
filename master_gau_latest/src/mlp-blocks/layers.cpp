#include "mlp/layers.h"
#include "autograd/operations/ActivationOps.h"

namespace OwnTensor
{
    namespace mlp_forward
    {
        Tensor linear(const Tensor& input, const Tensor& weights, const Tensor& bias)
        {
            Tensor weights_t = weights.t();
            Tensor weighted_sum = OwnTensor::matmul(input, weights_t);
            Tensor output = weighted_sum + bias;
            return output;
        }

        Tensor flatten(const Tensor& input)
        {
            const std::vector<int64_t>& dims = input.shape().dims;
            if (dims.empty())
            {
                return input;
            }

            int64_t batch_size = dims[0];
            int64_t total_features = 1;

            for (size_t i = 1; i < dims.size(); ++i)
            {
                total_features *= dims[i];
            }

            return input.reshape({ {batch_size, total_features} });

        }

        Tensor dropout(const Tensor& input, float p) {
            return autograd::dropout(input, p, true);
        }
    }
}