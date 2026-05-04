#include "nn/NN.h"
#include "autograd/ops_template.h"
#include "autograd/AutogradOps.h"
#include "checkpointing/GradMode.h"
#include "core/Tensor.h"
#include <iostream>

using namespace OwnTensor;

class MHA : public nn::Module {
public:
    nn::Linear c_attn;
    nn::Linear c_proj;
    int64_t n_heads;
    int64_t head_dim;

    MHA(int64_t n_embd, int64_t n_heads_) 
        : c_attn(n_embd, 3 * n_embd, true), 
          c_proj(n_embd, n_embd, true),
          n_heads(n_heads_), head_dim(n_embd / n_heads_) {
        register_module(c_attn);
        register_module(c_proj);
    }

    Tensor forward(const Tensor& x) override {
        int64_t B = x.shape().dims[0];
        int64_t T = x.shape().dims[1];
        int64_t C = x.shape().dims[2];

        Tensor qkv = c_attn.forward(x);
        std::vector<Tensor> inp = qkv.make_shards_inplace_axis(3, 2);
        Tensor q = inp[0];
        Tensor k = inp[1];
        Tensor v = inp[2];

        q = autograd::transpose(autograd::reshape(q, Shape({{B, T, n_heads, head_dim}})), 1, 2);
        k = autograd::transpose(autograd::reshape(k, Shape({{B, T, n_heads, head_dim}})), 1, 2);
        v = autograd::transpose(autograd::reshape(v, Shape({{B, T, n_heads, head_dim}})), 1, 2);

        Tensor attn_out = autograd::scaled_dot_product_attention(
                q, k, v, true, 0.0f, autograd::SDPBackend::MemoryEfficient);

        Tensor merged = autograd::reshape(
                            autograd::transpose(attn_out, 1, 2),
                            Shape({{B, T, C}}));

        return c_proj.forward(merged);
    }
};

int main() {
    DeviceIndex device(Device::CUDA, 0);
    device::set_cuda_device(0);
    
    int64_t B = 16;
    int64_t T = 1024;
    int64_t C = 768;
    int64_t n_heads = 12;

    MHA mha(C, n_heads);
    mha.to(device);

    Tensor x = Tensor::randn(Shape({{B, T, C}}), TensorOptions().with_device(device).with_dtype(Dtype::Float32));
    x.set_requires_grad(true);

    for (int i = 0; i < 5; ++i) {
        Tensor out = mha.forward(x);
        Tensor loss = autograd::sum(out);
        Tensor grad_scale = Tensor::full(Shape{{1}}, TensorOptions().with_device(device), 1.0f);
        loss.backward(&grad_scale);
    }

    std::cout << "Done" << std::endl;
    return 0;
}
