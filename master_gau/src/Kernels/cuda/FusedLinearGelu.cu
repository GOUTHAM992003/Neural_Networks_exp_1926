// =============================================================================
// FusedLinearGelu.cu
//
// Fused Linear + GeLU using cuBLASLt epilogue API.
// Works on sm_80+ (Ampere, Ada, Hopper). No arch gating — uses heuristic.
//
// Forward:  output = GeLU(input × weight^T + bias)
// Backward: DGELU epilogue for grad computation
//
// Layout: all row-major. cuBLASLt uses col-major internally,
// we feed transposed problem via leading-dimension trick.
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <mutex>
#include <array>
#include <cstring>
#include <algorithm>

#include "ops/helpers/FusedLinearGelu.cuh"
#include "ops/helpers/ActivationKernels.h"
#include "dtype/Dtype.h"

namespace OwnTensor {
namespace cuda {

// =============================================================================
// cuBLASLt handle + workspace — lazy init, thread-safe
// =============================================================================

static std::mutex s_lt_mutex;
static std::array<cublasLtHandle_t, 8> s_lt_handles{};
static std::array<bool, 8> s_lt_ready{};

static cublasLtHandle_t get_lt(int dev) {
    if (!s_lt_ready[dev]) {
        std::lock_guard<std::mutex> lk(s_lt_mutex);
        if (!s_lt_ready[dev]) {
            cudaSetDevice(dev);
            cublasLtCreate(&s_lt_handles[dev]);
            s_lt_ready[dev] = true;
        }
    }
    return s_lt_handles[dev];
}

static constexpr size_t WS_SIZE = 32ull * 1024 * 1024;
static std::array<void*, 8> s_ws{};
static std::array<bool, 8> s_ws_ready{};
static std::mutex s_ws_mutex;

static void* get_ws(int dev) {
    if (!s_ws_ready[dev]) {
        std::lock_guard<std::mutex> lk(s_ws_mutex);
        if (!s_ws_ready[dev]) {
            cudaSetDevice(dev);
            cudaMalloc(&s_ws[dev], WS_SIZE);
            s_ws_ready[dev] = true;
        }
    }
    return s_ws[dev];
}

// cuBLAS handle for fallback
static std::mutex s_cb_mutex;
static std::array<cublasHandle_t, 8> s_cb_handles{};
static std::array<bool, 8> s_cb_ready{};

static cublasHandle_t get_cublas(int dev) {
    if (!s_cb_ready[dev]) {
        std::lock_guard<std::mutex> lk(s_cb_mutex);
        if (!s_cb_ready[dev]) {
            cudaSetDevice(dev);
            cublasCreate(&s_cb_handles[dev]);
            cublasSetMathMode(s_cb_handles[dev], CUBLAS_TF32_TENSOR_OP_MATH);
            s_cb_ready[dev] = true;
        }
    }
    return s_cb_handles[dev];
}

// =============================================================================
// Helper kernels
// =============================================================================

// Column reduction: dst[j] += sum over rows of src[row * N + j]
__global__ void col_reduce_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                  int64_t rows, int64_t cols) {
    int64_t j = blockIdx.x;
    if (j >= cols) return;

    __shared__ float smem[256];
    float partial = 0.0f;
    for (int64_t row = threadIdx.x; row < rows; row += blockDim.x)
        partial += src[row * cols + j];

    smem[threadIdx.x] = partial;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(&dst[j], smem[0]);
}

// Bias add: C[i] += bias[i % N]
__global__ void add_bias_kernel_lg(float* C, const float* bias, int64_t M, int64_t N) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = M * N;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total; i += stride)
        C[i] += bias[i % N];
}

// =============================================================================
// FORWARD: output = GeLU(input × weight^T + bias)
// =============================================================================

// Try cuBLASLt with GELU_BIAS epilogue. Returns true if succeeded.
static bool try_cublaslt_forward_f32(
    cublasLtHandle_t lt,
    const float* weight, const float* input, const float* bias, float* output,
    float* gelu_aux,  // optional: save pre-gelu for backward (GELU_AUX epilogue)
    int64_t M, int64_t N, int64_t K,
    void* ws, size_t ws_size)
{
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    cublasOperation_t op_n = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));

    // Choose epilogue
    cublasLtEpilogue_t epilogue;
    if (gelu_aux && bias) {
        epilogue = CUBLASLT_EPILOGUE_GELU_AUX_BIAS;
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &gelu_aux, sizeof(gelu_aux));
        int64_t aux_ld = N;
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &aux_ld, sizeof(aux_ld));
    } else if (bias) {
        epilogue = CUBLASLT_EPILOGUE_GELU_BIAS;
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    } else {
        epilogue = CUBLASLT_EPILOGUE_GELU;
    }
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));

    // Row-major → col-major trick: C[M,N] row = C^T[N,M] col
    cublasLtMatrixLayout_t lA = nullptr, lB = nullptr, lC = nullptr;
    cublasLtMatrixLayoutCreate(&lA, CUDA_R_32F, K, N, K);
    cublasLtMatrixLayoutCreate(&lB, CUDA_R_32F, K, M, K);
    cublasLtMatrixLayoutCreate(&lC, CUDA_R_32F, N, M, N);

    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof(ws_size));

    cublasLtMatmulHeuristicResult_t heur{};
    int found = 0;
    cublasLtMatmulAlgoGetHeuristic(lt, desc, lA, lB, lC, lC, pref, 1, &heur, &found);

    bool ok = false;
    if (found > 0) {
        const float alpha = 1.0f, beta = 0.0f;
        auto status = cublasLtMatmul(lt, desc, &alpha,
            weight, lA, input, lB, &beta, output, lC, output, lC,
            &heur.algo, ws, ws_size, nullptr);
        ok = (status == CUBLAS_STATUS_SUCCESS);
    }

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(lC);
    cublasLtMatrixLayoutDestroy(lB);
    cublasLtMatrixLayoutDestroy(lA);
    cublasLtMatmulDescDestroy(desc);
    return ok;
}

// Fallback: cublasSgemm + bias + gelu (3 kernels)
static void fallback_forward_f32(
    cublasHandle_t h,
    const float* weight, const float* input, const float* bias, float* output,
    int64_t M, int64_t N, int64_t K)
{
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N,
        (int)N, (int)M, (int)K, &alpha,
        weight, (int)K, input, (int)K, &beta, output, (int)N);

    if (bias) {
        int threads = 256;
        int64_t total = M * N;
        int blocks = (int)std::min((total + threads - 1) / threads, (int64_t)65535);
        add_bias_kernel_lg<<<blocks, threads>>>(output, bias, M, N);
    }

    fused_gelu_cuda(output, output, M * N);
}

// =============================================================================
// BACKWARD
// grad_gelu = grad_output * GeLU'(pre_gelu)
// grad_input = grad_gelu × weight          [M, K]
// grad_weight = input^T × grad_gelu        [N, K]  (accumulated)
// grad_bias = sum(grad_gelu, dim=0)         [N]
// =============================================================================

// Try cuBLASLt with DGELU epilogue for grad_input computation
static bool try_cublaslt_backward_f32(
    cublasLtHandle_t lt,
    const float* grad_output,  // [M, N]
    const float* gelu_aux,     // [M, N] pre-gelu values from forward
    const float* weight,       // [N, K]
    float* grad_input,         // [M, K]
    float* grad_bias,          // [N] or nullptr
    int64_t M, int64_t N, int64_t K,
    void* ws, size_t ws_size)
{
    // Step 1: grad_input = (grad_output ⊙ GeLU'(aux)) × weight
    // Using DGELU epilogue: the matmul computes grad_output × weight^T,
    // but with DGELU applied to grad_output using aux before the matmul.
    // Actually DGELU works differently — it's applied to the matmul INPUT.
    //
    // For now, use the simple approach:
    // 1. Compute grad_gelu = grad_output * GeLU'(pre_gelu) using our backward kernel
    // 2. grad_input = grad_gelu × weight (standard matmul)
    // 3. grad_weight = input^T × grad_gelu (standard matmul)
    // 4. grad_bias = sum(grad_gelu, dim=0)
    //
    // The DGELU epilogue fusion is complex and architecture-specific.
    // We use the proven 2-step approach.
    return false;  // signal caller to use fallback
}

// =============================================================================
// Public entry points
// =============================================================================

void fused_linear_gelu_forward(
    const void* input, const void* weight, const void* bias,
    void* output,
    int64_t M, int64_t N, int64_t K,
    Dtype dtype, int device_idx)
{
    if (device_idx < 0) cudaGetDevice(&device_idx);

    if (dtype == Dtype::Float32) {
        const float* W = static_cast<const float*>(weight);
        const float* X = static_cast<const float*>(input);
        const float* b = static_cast<const float*>(bias);
        float* Y = static_cast<float*>(output);

        // Try cuBLASLt fused path (no arch check — heuristic decides)
        bool ok = try_cublaslt_forward_f32(
            get_lt(device_idx), W, X, b, Y, nullptr,
            M, N, K, get_ws(device_idx), WS_SIZE);

        if (!ok) {
            fallback_forward_f32(get_cublas(device_idx), W, X, b, Y, M, N, K);
        }
    } else {
        throw std::runtime_error("fused_linear_gelu_forward: only fp32 for now");
    }
}

void fused_linear_gelu_backward(
    const void* grad_output,
    const void* input,
    const void* weight,
    const void* bias,
    const void* gelu_aux,
    void* grad_input,
    void* grad_weight,
    void* grad_bias,
    int64_t M, int64_t N, int64_t K,
    Dtype dtype, int device_idx)
{
    if (device_idx < 0) cudaGetDevice(&device_idx);

    if (dtype == Dtype::Float32) {
        const float* dY = static_cast<const float*>(grad_output);
        const float* X  = static_cast<const float*>(input);
        const float* W  = static_cast<const float*>(weight);
        const float* b  = static_cast<const float*>(bias);
        float* dX = static_cast<float*>(grad_input);
        float* dW = static_cast<float*>(grad_weight);
        float* db = static_cast<float*>(grad_bias);

        // Step 1: Compute grad_gelu = dY * GeLU'(X @ W^T + b)
        // We need pre_gelu = X @ W^T + b. If gelu_aux was saved, use it.
        // Otherwise recompute.
        int64_t MN = M * N;
        float* grad_gelu = nullptr;
        cudaMalloc(&grad_gelu, MN * sizeof(float));

        if (gelu_aux) {
            // Use saved pre-gelu values
            fused_gelu_backward_cuda(dY, static_cast<const float*>(gelu_aux),
                                     grad_gelu, MN);
        } else {
            // Recompute: pre_gelu = X @ W^T + b
            float* pre_gelu = nullptr;
            cudaMalloc(&pre_gelu, MN * sizeof(float));

            const float alpha = 1.0f, beta = 0.0f;
            cublasSgemm(get_cublas(device_idx),
                CUBLAS_OP_T, CUBLAS_OP_N,
                (int)N, (int)M, (int)K, &alpha,
                W, (int)K, X, (int)K, &beta, pre_gelu, (int)N);

            if (b) {
                int threads = 256;
                int blocks = (int)std::min((MN + threads - 1) / threads, (int64_t)65535);
                add_bias_kernel_lg<<<blocks, threads>>>(pre_gelu, b, M, N);
            }

            fused_gelu_backward_cuda(dY, pre_gelu, grad_gelu, MN);
            cudaFree(pre_gelu);
        }

        cublasHandle_t h = get_cublas(device_idx);
        const float alpha = 1.0f, beta = 0.0f;

        // Step 2: grad_input = grad_gelu × weight  [M,N] × [N,K] → [M,K]
        // Row-major: dX = grad_gelu × W
        // Col-major: dX^T[K,M] = W^T[K,N] × grad_gelu^T[N,M]
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
            (int)K, (int)M, (int)N, &alpha,
            W, (int)K, grad_gelu, (int)N, &beta, dX, (int)K);

        // Step 3: grad_weight = grad_gelu^T × input  [N,M] × [M,K] → [N,K]
        // Col-major: dW^T[K,N] = X^T[K,M] × grad_gelu[M,N] → wait, need [N,K]
        // Row-major dW[N,K] = grad_gelu^T[N,M] × X[M,K]
        // Col-major: dW^T[K,N] = X^T[K,M] × grad_gelu[N,M]^T
        const float beta_acc = 1.0f;  // accumulate into grad_weight
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T,
            (int)K, (int)N, (int)M, &alpha,
            X, (int)K, grad_gelu, (int)N, &beta_acc, dW, (int)K);

        // Step 4: grad_bias = sum(grad_gelu, dim=0) → [N]
        if (db) {
            cudaMemset(db, 0, N * sizeof(float));
            col_reduce_kernel<<<(int)N, 256>>>(grad_gelu, db, M, N);
        }

        cudaFree(grad_gelu);
    } else {
        throw std::runtime_error("fused_linear_gelu_backward: only fp32 for now");
    }
}

} // namespace cuda
} // namespace OwnTensor
