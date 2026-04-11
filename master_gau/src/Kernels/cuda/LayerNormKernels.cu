#include "ops/helpers/LayerNormKernels.h"
#include "dtype/Types.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace OwnTensor {
namespace cuda {

// =================================================================================
// Type conversion helpers
// =================================================================================
template<typename T> __device__ __forceinline__ float to_float(T val);
template<> __device__ __forceinline__ float to_float(float val) { return val; }
template<> __device__ __forceinline__ float to_float(__half val) { return __half2float(val); }
template<> __device__ __forceinline__ float to_float(__nv_bfloat16 val) { return __bfloat162float(val); }

template<typename T> __device__ __forceinline__ T from_float(float val);
template<> __device__ __forceinline__ float from_float(float val) { return val; }
template<> __device__ __forceinline__ __half from_float(float val) { return __float2half(val); }
template<> __device__ __forceinline__ __nv_bfloat16 from_float(float val) { return __float2bfloat16(val); }

// =================================================================================
// Warp Reduction
// =================================================================================
template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// =================================================================================
// Welford Helpers
// =================================================================================
template<typename AccT>
struct WelfordData {
    AccT n, mu, m2;
};

template<typename AccT>
__device__ __inline__ WelfordData<AccT> welford_merge(WelfordData<AccT> a, WelfordData<AccT> b) {
    if (a.n == 0) return b;
    if (b.n == 0) return a;
    WelfordData<AccT> res;
    res.n  = a.n + b.n;
    AccT delta = b.mu - a.mu;
    res.mu = a.mu + delta * (b.n / res.n);
    res.m2 = a.m2 + b.m2 + delta * delta * (a.n * b.n / res.n);
    return res;
}

// =================================================================================
// Fused Forward Kernel — LayerNorm + RMSNorm via bool template
//   rms_norm=false: standard LayerNorm (mean + var + normalize with beta)
//   rms_norm=true:  RMSNorm (no mean, just rms + normalize, no beta)
// The if constexpr branches compile to separate PTX — zero runtime overhead.
// =================================================================================
template<typename T, typename AccT, bool rms_norm>
__global__ void norm_forward_kernel(
    const T*    __restrict__ x,
    const T*    __restrict__ gamma,
    const T*    __restrict__ beta,   // ignored when rms_norm=true
    T*          __restrict__ y,
    AccT*       __restrict__ mean_out,  // nullptr when rms_norm=true
    AccT*       __restrict__ rstd_out,
    int cols,
    AccT eps)
{
    const int row  = blockIdx.x;
    const int tid  = threadIdx.x;
    const T* row_x = x + row * cols;
    T*       row_y = y + row * cols;

    // --- PHASE 1: VECTORIZED LOCAL ACCUMULATION ---
    AccT local_sum    = 0;
    AccT local_sq_sum = 0;
    int  local_count  = 0;

    if constexpr (std::is_same_v<T, float>) {
        const float4* x_vec = reinterpret_cast<const float4*>(row_x);
        const int vec_cols = cols / 4;
        #pragma unroll 4
        for (int i = tid; i < vec_cols; i += blockDim.x) {
            float4 v = x_vec[i];
            AccT vx = v.x, vy = v.y, vz = v.z, vw = v.w;
            if constexpr (!rms_norm) local_sum += vx + vy + vz + vw;
            local_sq_sum += vx*vx + vy*vy + vz*vz + vw*vw;
            local_count += 4;
        }
        for (int i = vec_cols * 4 + tid; i < cols; i += blockDim.x) {
            AccT val = row_x[i];
            if constexpr (!rms_norm) local_sum += val;
            local_sq_sum += val * val;
            local_count++;
        }
    } else if constexpr (std::is_same_v<T, __half>) {
        const float4* x_vec = reinterpret_cast<const float4*>(row_x);
        const int vec_cols = cols / 8;
        #pragma unroll 4
        for (int i = tid; i < vec_cols; i += blockDim.x) {
            float4 raw = x_vec[i];
            const __half2* h = reinterpret_cast<const __half2*>(&raw);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float2 f = __half22float2(h[k]);
                if constexpr (!rms_norm) local_sum += (AccT)f.x + (AccT)f.y;
                local_sq_sum += (AccT)f.x * (AccT)f.x + (AccT)f.y * (AccT)f.y;
            }
            local_count += 8;
        }
        for (int i = vec_cols * 8 + tid; i < cols; i += blockDim.x) {
            AccT val = (AccT)row_x[i];
            if constexpr (!rms_norm) local_sum += val;
            local_sq_sum += val * val;
            local_count++;
        }
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        const float4* x_vec = reinterpret_cast<const float4*>(row_x);
        const int vec_cols = cols / 8;
        #pragma unroll 4
        for (int i = tid; i < vec_cols; i += blockDim.x) {
            float4 raw = x_vec[i];
            const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float2 f = __bfloat1622float2(h[k]);
                if constexpr (!rms_norm) local_sum += (AccT)f.x + (AccT)f.y;
                local_sq_sum += (AccT)f.x * (AccT)f.x + (AccT)f.y * (AccT)f.y;
            }
            local_count += 8;
        }
        for (int i = vec_cols * 8 + tid; i < cols; i += blockDim.x) {
            AccT val = (AccT)row_x[i];
            if constexpr (!rms_norm) local_sum += val;
            local_sq_sum += val * val;
            local_count++;
        }
    }

    // --- PHASE 2: CONVERT TO WELFORD STATE ---
    AccT n = (AccT)local_count;
    WelfordData<AccT> state;
    if constexpr (!rms_norm) {
        state = { n,
                  (n > 0) ? local_sum / n : (AccT)0,
                  (n > 0) ? (local_sq_sum - local_sum * local_sum / n) : (AccT)0 };
    } else {
        // RMSNorm: no mean tracking, m2 = sum of squares
        state = { n, (AccT)0, local_sq_sum };
    }

    // --- PHASE 3: WARP-LEVEL REDUCTION ---
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        WelfordData<AccT> other;
        other.n  = __shfl_down_sync(0xffffffff, state.n,  offset);
        other.mu = __shfl_down_sync(0xffffffff, state.mu, offset);
        other.m2 = __shfl_down_sync(0xffffffff, state.m2, offset);
        if constexpr (!rms_norm) {
            state = welford_merge(state, other);
        } else {
            // RMSNorm: just sum m2 (sum of squares) and n
            state.m2 += other.m2;
            state.n  += other.n;
        }
    }

    // --- PHASE 4: BLOCK-LEVEL REDUCTION ---
    __shared__ WelfordData<AccT> s_welford[32];
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    if (lane_id == 0) s_welford[warp_id] = state;
    __syncthreads();

    if (tid == 0) {
        WelfordData<AccT> final_state = s_welford[0];
        const int num_warps = blockDim.x / 32;
        for (int i = 1; i < num_warps; ++i) {
            if constexpr (!rms_norm) {
                final_state = welford_merge(final_state, s_welford[i]);
            } else {
                final_state.m2 += s_welford[i].m2;
                final_state.n  += s_welford[i].n;
            }
        }
        s_welford[0] = final_state;
    }
    __syncthreads();

    // --- PHASE 5: FINAL STATISTICS ---
    AccT mu, rstd;
    if constexpr (!rms_norm) {
        mu   = s_welford[0].mu;
        rstd = rsqrtf(s_welford[0].m2 / cols + eps);
    } else {
        mu   = (AccT)0;
        rstd = rsqrtf(s_welford[0].m2 / cols + eps);
    }

    if (tid == 0) {
        if constexpr (!rms_norm) { if (mean_out) mean_out[row] = mu; }
        if (rstd_out) rstd_out[row] = rstd;
    }

    // --- PHASE 6: VECTORIZED NORMALIZE & WRITEOUT ---
    if constexpr (std::is_same_v<T, float>) {
        float4*       y_vec = reinterpret_cast<float4*>(row_y);
        const float4* x_vec = reinterpret_cast<const float4*>(row_x);
        const int vec_cols = cols / 4;
        #pragma unroll 4
        for (int i = tid; i < vec_cols; i += blockDim.x) {
            float4 xv = x_vec[i];
            float4 gv = gamma ? reinterpret_cast<const float4*>(gamma)[i] : make_float4(1,1,1,1);
            float4 res;
            if constexpr (!rms_norm) {
                float4 bv = beta ? reinterpret_cast<const float4*>(beta)[i] : make_float4(0,0,0,0);
                res.x = ((AccT)xv.x - mu) * rstd * gv.x + bv.x;
                res.y = ((AccT)xv.y - mu) * rstd * gv.y + bv.y;
                res.z = ((AccT)xv.z - mu) * rstd * gv.z + bv.z;
                res.w = ((AccT)xv.w - mu) * rstd * gv.w + bv.w;
            } else {
                res.x = (AccT)xv.x * rstd * gv.x;
                res.y = (AccT)xv.y * rstd * gv.y;
                res.z = (AccT)xv.z * rstd * gv.z;
                res.w = (AccT)xv.w * rstd * gv.w;
            }
            y_vec[i] = res;
        }
        for (int i = vec_cols * 4 + tid; i < cols; i += blockDim.x) {
            AccT g = gamma ? (AccT)gamma[i] : (AccT)1;
            if constexpr (!rms_norm) {
                AccT b = beta ? (AccT)beta[i] : (AccT)0;
                row_y[i] = (T)(((AccT)row_x[i] - mu) * rstd * g + b);
            } else {
                row_y[i] = (T)((AccT)row_x[i] * rstd * g);
            }
        }
    } else if constexpr (std::is_same_v<T, __half>) {
        float4*       y_vec = reinterpret_cast<float4*>(row_y);
        const float4* x_vec = reinterpret_cast<const float4*>(row_x);
        const int vec_cols = cols / 8;
        #pragma unroll 4
        for (int i = tid; i < vec_cols; i += blockDim.x) {
            float4 xraw = x_vec[i];
            const __half2* xh = reinterpret_cast<const __half2*>(&xraw);

            float4 graw;
            if (gamma) graw = reinterpret_cast<const float4*>(gamma)[i];
            const __half2* gh = gamma ? reinterpret_cast<const __half2*>(&graw) : nullptr;

            float4 braw;
            if constexpr (!rms_norm) { if (beta) braw = reinterpret_cast<const float4*>(beta)[i]; }
            const __half2* bh = (!rms_norm && beta) ? reinterpret_cast<const __half2*>(&braw) : nullptr;

            float4 yraw;
            __half2* yh = reinterpret_cast<__half2*>(&yraw);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float2 xf = __half22float2(xh[k]);
                if constexpr (!rms_norm) {
                    xf.x = (xf.x - mu) * rstd;
                    xf.y = (xf.y - mu) * rstd;
                } else {
                    xf.x = xf.x * rstd;
                    xf.y = xf.y * rstd;
                }
                if (gh) { float2 gf = __half22float2(gh[k]); xf.x *= gf.x; xf.y *= gf.y; }
                if constexpr (!rms_norm) {
                    if (bh) { float2 bf = __half22float2(bh[k]); xf.x += bf.x; xf.y += bf.y; }
                }
                yh[k] = __float22half2_rn(xf);
            }
            y_vec[i] = yraw;
        }
        for (int i = vec_cols * 8 + tid; i < cols; i += blockDim.x) {
            AccT g = gamma ? (AccT)gamma[i] : (AccT)1;
            if constexpr (!rms_norm) {
                AccT b = beta ? (AccT)beta[i] : (AccT)0;
                row_y[i] = (T)(((AccT)row_x[i] - mu) * rstd * g + b);
            } else {
                row_y[i] = (T)((AccT)row_x[i] * rstd * g);
            }
        }
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        float4*       y_vec = reinterpret_cast<float4*>(row_y);
        const float4* x_vec = reinterpret_cast<const float4*>(row_x);
        const int vec_cols = cols / 8;
        #pragma unroll 4
        for (int i = tid; i < vec_cols; i += blockDim.x) {
            float4 xraw = x_vec[i];
            const __nv_bfloat162* xh = reinterpret_cast<const __nv_bfloat162*>(&xraw);

            float4 graw;
            if (gamma) graw = reinterpret_cast<const float4*>(gamma)[i];
            const __nv_bfloat162* gh = gamma ? reinterpret_cast<const __nv_bfloat162*>(&graw) : nullptr;

            float4 braw;
            if constexpr (!rms_norm) { if (beta) braw = reinterpret_cast<const float4*>(beta)[i]; }
            const __nv_bfloat162* bh = (!rms_norm && beta) ? reinterpret_cast<const __nv_bfloat162*>(&braw) : nullptr;

            float4 yraw;
            __nv_bfloat162* yh = reinterpret_cast<__nv_bfloat162*>(&yraw);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float2 xf = __bfloat1622float2(xh[k]);
                if constexpr (!rms_norm) {
                    xf.x = (xf.x - mu) * rstd;
                    xf.y = (xf.y - mu) * rstd;
                } else {
                    xf.x = xf.x * rstd;
                    xf.y = xf.y * rstd;
                }
                if (gh) { float2 gf = __bfloat1622float2(gh[k]); xf.x *= gf.x; xf.y *= gf.y; }
                if constexpr (!rms_norm) {
                    if (bh) { float2 bf = __bfloat1622float2(bh[k]); xf.x += bf.x; xf.y += bf.y; }
                }
                yh[k] = __float22bfloat162_rn(xf);
            }
            y_vec[i] = yraw;
        }
        for (int i = vec_cols * 8 + tid; i < cols; i += blockDim.x) {
            AccT g = gamma ? (AccT)gamma[i] : (AccT)1;
            if constexpr (!rms_norm) {
                AccT b = beta ? (AccT)beta[i] : (AccT)0;
                row_y[i] = (T)(((AccT)row_x[i] - mu) * rstd * g + b);
            } else {
                row_y[i] = (T)((AccT)row_x[i] * rstd * g);
            }
        }
    }
}

// =================================================================================
// LayerNorm Forward Launchers
// =================================================================================
void layer_norm_forward_cuda(
    const float* x, const float* gamma, const float* beta,
    float* y, float* mean, float* rstd,
    int rows, int cols, float eps)
{
    int threads = 256;
    norm_forward_kernel<float, float, false><<<rows, threads>>>(x, gamma, beta, y, mean, rstd, cols, eps);
}

void layer_norm_forward_cuda(
    const float16_t* x, const float16_t* gamma, const float16_t* beta,
    float16_t* y, float* mean, float* rstd,
    int rows, int cols, float eps)
{
    int threads = 256;
    norm_forward_kernel<__half, float, false><<<rows, threads>>>(
        reinterpret_cast<const __half*>(x),
        reinterpret_cast<const __half*>(gamma),
        reinterpret_cast<const __half*>(beta),
        reinterpret_cast<__half*>(y),
        mean, rstd, cols, eps);
}

void layer_norm_forward_cuda(
    const bfloat16_t* x, const bfloat16_t* gamma, const bfloat16_t* beta,
    bfloat16_t* y, float* mean, float* rstd,
    int rows, int cols, float eps)
{
    int threads = 256;
    norm_forward_kernel<__nv_bfloat16, float, false><<<rows, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(gamma),
        reinterpret_cast<const __nv_bfloat16*>(beta),
        reinterpret_cast<__nv_bfloat16*>(y),
        mean, rstd, cols, eps);
}

// =================================================================================
// RMSNorm Forward Launchers — same kernel with rms_norm=true
// =================================================================================
void rms_norm_forward_cuda(
    const float* x, const float* gamma,
    float* y, float* rstd,
    int rows, int cols, float eps)
{
    int threads = 256;
    norm_forward_kernel<float, float, true><<<rows, threads>>>(
        x, gamma, nullptr, y, nullptr, rstd, cols, eps);
}

void rms_norm_forward_cuda(
    const float16_t* x, const float16_t* gamma,
    float16_t* y, float* rstd,
    int rows, int cols, float eps)
{
    int threads = 256;
    norm_forward_kernel<__half, float, true><<<rows, threads>>>(
        reinterpret_cast<const __half*>(x),
        reinterpret_cast<const __half*>(gamma),
        nullptr, reinterpret_cast<__half*>(y),
        nullptr, rstd, cols, eps);
}

void rms_norm_forward_cuda(
    const bfloat16_t* x, const bfloat16_t* gamma,
    bfloat16_t* y, float* rstd,
    int rows, int cols, float eps)
{
    int threads = 256;
    norm_forward_kernel<__nv_bfloat16, float, true><<<rows, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(gamma),
        nullptr, reinterpret_cast<__nv_bfloat16*>(y),
        nullptr, rstd, cols, eps);
}

// =================================================================================
// Backward Kernel A: Gamma/Beta gradients (column-wise reduction over all rows)
// =================================================================================
template<typename T>
__global__ void ln_backward_gamma_beta_kernel(
    const T* __restrict__ grad_y,
    const T* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_gamma,
    float* __restrict__ grad_beta,
    int rows, int cols)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float s_dgamma[8][32];
    __shared__ float s_dbeta[8][32];

    #pragma unroll 4
    for (int col_base = blockIdx.x * 32; col_base < cols; col_base += gridDim.x * 32) {
        int col = col_base + tx;
        float d_gamma_acc = 0.0f, d_beta_acc = 0.0f;

        if (col < cols) {
            for (int row = blockIdx.y * 8 + ty; row < rows; row += gridDim.y * 8) {
                float gy = to_float(grad_y[row * cols + col]);
                float input_val = to_float(x[row * cols + col]);
                float norm_x = (input_val - mean[row]) * rstd[row];
                d_beta_acc += gy;
                d_gamma_acc += gy * norm_x;
            }
        }

        s_dgamma[ty][tx] = d_gamma_acc;
        s_dbeta[ty][tx] = d_beta_acc;
        __syncthreads();

        if (ty == 0 && col < cols) {
            float fg = 0, fb = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) { fg += s_dgamma[i][tx]; fb += s_dbeta[i][tx]; }
            atomicAdd(&grad_gamma[col], fg);
            atomicAdd(&grad_beta[col], fb);
        }
        __syncthreads();
    }
}

// =================================================================================
// Backward Kernel B: Input gradients (per-row) — VECTORIZED with float4
// =================================================================================
template<typename T>
__global__ void ln_backward_input_kernel(
    const T* __restrict__ grad_y,
    const T* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    const T* __restrict__ gamma,
    T* __restrict__ grad_x,
    int cols)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const T* dy_row = grad_y + row * cols;
    const T* x_row  = x + row * cols;
    T* dx_row = grad_x + row * cols;

    float m = mean[row];
    float rs = rstd[row];

    float sum_dy_gamma = 0.0f;
    float sum_dy_gamma_norm = 0.0f;

    // ── Pass A: vectorized accumulation ──
    if constexpr (std::is_same_v<T, float>) {
        const float4* dy_vec = reinterpret_cast<const float4*>(dy_row);
        const float4* x_vec  = reinterpret_cast<const float4*>(x_row);
        const float4* g_vec  = gamma ? reinterpret_cast<const float4*>(gamma) : nullptr;
        const int vec_cols = cols / 4;

        #pragma unroll 4
        for (int i = tid; i < vec_cols; i += blockDim.x) {
            float4 dy4 = dy_vec[i];
            float4 x4  = x_vec[i];
            float4 g4  = g_vec ? g_vec[i] : make_float4(1,1,1,1);

            float nx = (x4.x - m) * rs, ny = (x4.y - m) * rs;
            float nz = (x4.z - m) * rs, nw = (x4.w - m) * rs;
            float dg_x = dy4.x * g4.x, dg_y = dy4.y * g4.y;
            float dg_z = dy4.z * g4.z, dg_w = dy4.w * g4.w;

            sum_dy_gamma += dg_x + dg_y + dg_z + dg_w;
            sum_dy_gamma_norm += dg_x*nx + dg_y*ny + dg_z*nz + dg_w*nw;
        }
        for (int i = vec_cols * 4 + tid; i < cols; i += blockDim.x) {
            float g = gamma ? gamma[i] : 1.0f;
            float dy = dy_row[i];
            float norm_x = (x_row[i] - m) * rs;
            sum_dy_gamma += dy * g;
            sum_dy_gamma_norm += dy * g * norm_x;
        }
    } else {
        // fp16/bf16: scalar with upcast (vectorization here is marginal gain)
        #pragma unroll 4
        for (int i = tid; i < cols; i += blockDim.x) {
            float g = gamma ? to_float(gamma[i]) : 1.0f;
            float dy = to_float(dy_row[i]);
            float val = to_float(x_row[i]);
            float norm_x = (val - m) * rs;
            sum_dy_gamma += dy * g;
            sum_dy_gamma_norm += dy * g * norm_x;
        }
    }

    // Warp + block reduction
    sum_dy_gamma = warpReduceSum(sum_dy_gamma);
    sum_dy_gamma_norm = warpReduceSum(sum_dy_gamma_norm);

    __shared__ float s_sum1, s_sum2;
    if (tid == 0) { s_sum1 = 0; s_sum2 = 0; }
    __syncthreads();
    if (tid % warpSize == 0) {
        atomicAdd(&s_sum1, sum_dy_gamma);
        atomicAdd(&s_sum2, sum_dy_gamma_norm);
    }
    __syncthreads();

    float total_sum1 = s_sum1;
    float total_sum2 = s_sum2;
    float inv_cols = 1.0f / cols;

    // ── Pass B: vectorized grad_x computation ──
    if constexpr (std::is_same_v<T, float>) {
        float4* dx_vec = reinterpret_cast<float4*>(dx_row);
        const float4* dy_vec = reinterpret_cast<const float4*>(dy_row);
        const float4* x_vec  = reinterpret_cast<const float4*>(x_row);
        const float4* g_vec  = gamma ? reinterpret_cast<const float4*>(gamma) : nullptr;
        const int vec_cols = cols / 4;

        #pragma unroll 4
        for (int i = tid; i < vec_cols; i += blockDim.x) {
            float4 dy4 = dy_vec[i];
            float4 x4  = x_vec[i];
            float4 g4  = g_vec ? g_vec[i] : make_float4(1,1,1,1);
            float4 res;

            float nx = (x4.x - m) * rs, ny = (x4.y - m) * rs;
            float nz = (x4.z - m) * rs, nw = (x4.w - m) * rs;

            res.x = rs * (dy4.x * g4.x - (total_sum1 + nx * total_sum2) * inv_cols);
            res.y = rs * (dy4.y * g4.y - (total_sum1 + ny * total_sum2) * inv_cols);
            res.z = rs * (dy4.z * g4.z - (total_sum1 + nz * total_sum2) * inv_cols);
            res.w = rs * (dy4.w * g4.w - (total_sum1 + nw * total_sum2) * inv_cols);
            dx_vec[i] = res;
        }
        for (int i = vec_cols * 4 + tid; i < cols; i += blockDim.x) {
            float g = gamma ? gamma[i] : 1.0f;
            float dy = dy_row[i];
            float norm_x = (x_row[i] - m) * rs;
            dx_row[i] = rs * (dy * g - (total_sum1 + norm_x * total_sum2) * inv_cols);
        }
    } else {
        #pragma unroll 4
        for (int i = tid; i < cols; i += blockDim.x) {
            float g = gamma ? to_float(gamma[i]) : 1.0f;
            float dy = to_float(dy_row[i]);
            float val = to_float(x_row[i]);
            float norm_x = (val - m) * rs;
            dx_row[i] = from_float<T>(rs * (dy * g - (total_sum1 + norm_x * total_sum2) * inv_cols));
        }
    }
}

// =================================================================================
// RMSNorm Backward Kernels
// =================================================================================

// Kernel A: grad_gamma for RMSNorm (column-wise reduction, no beta)
template<typename T>
__global__ void rms_backward_gamma_kernel(
    const T* __restrict__ grad_y,
    const T* __restrict__ x,
    const float* __restrict__ rstd,
    float* __restrict__ grad_gamma,
    int rows, int cols)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float s_dgamma[8][32];

    #pragma unroll 4
    for (int col_base = blockIdx.x * 32; col_base < cols; col_base += gridDim.x * 32) {
        int col = col_base + tx;
        float acc = 0.0f;

        if (col < cols) {
            for (int row = blockIdx.y * 8 + ty; row < rows; row += gridDim.y * 8) {
                float gy = to_float(grad_y[row * cols + col]);
                float v  = to_float(x[row * cols + col]);
                acc += gy * v * rstd[row];
            }
        }

        s_dgamma[ty][tx] = acc;
        __syncthreads();

        if (ty == 0 && col < cols) {
            float fg = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) fg += s_dgamma[i][tx];
            atomicAdd(&grad_gamma[col], fg);
        }
        __syncthreads();
    }
}

// Kernel B: grad_input for RMSNorm (per-row)
// dx = rstd * (dy*gamma - x * rstd^2 * sum(dy*gamma*x) / cols)
template<typename T>
__global__ void rms_backward_input_kernel(
    const T* __restrict__ grad_y,
    const T* __restrict__ x,
    const float* __restrict__ rstd,
    const T* __restrict__ gamma,
    T* __restrict__ grad_x,
    int cols)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const T* dy_row = grad_y + row * cols;
    const T* x_row  = x + row * cols;
    T* dx_row = grad_x + row * cols;
    float rs = rstd[row];

    // Pass A: dot = sum(dy * gamma * x)
    float dot = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < cols; i += blockDim.x) {
        float g = gamma ? to_float(gamma[i]) : 1.0f;
        float dy = to_float(dy_row[i]);
        float v  = to_float(x_row[i]);
        dot += dy * g * v;
    }
    dot = warpReduceSum(dot);

    __shared__ float s_dot;
    if (tid == 0) s_dot = 0;
    __syncthreads();
    if (tid % warpSize == 0) atomicAdd(&s_dot, dot);
    __syncthreads();

    float total_dot = s_dot;
    float inv_cols = 1.0f / cols;
    float rs3 = rs * rs * rs;

    // Pass B: grad_x
    #pragma unroll 4
    for (int i = tid; i < cols; i += blockDim.x) {
        float g = gamma ? to_float(gamma[i]) : 1.0f;
        float dy = to_float(dy_row[i]);
        float v  = to_float(x_row[i]);
        dx_row[i] = from_float<T>(rs * dy * g - rs3 * v * total_dot * inv_cols);
    }
}

// =================================================================================
// LayerNorm Backward Launch Helper
// =================================================================================
template<typename T>
static void launch_layer_norm_backward(
    const T* grad_y, const T* x,
    const float* mean, const float* rstd, const T* gamma,
    T* grad_x, float* grad_gamma, float* grad_beta,
    int rows, int cols)
{
    // Gamma/Beta gradients
    if (grad_gamma != nullptr || grad_beta != nullptr) {
        cudaMemset(grad_gamma, 0, cols * sizeof(float));
        cudaMemset(grad_beta, 0, cols * sizeof(float));

        dim3 threads(32, 8);
        int blocks_x = (cols + 31) / 32;
        int blocks_y = 128 / blocks_x;
        if (blocks_y < 1) blocks_y = 1;
        if (blocks_y > 32) blocks_y = 32;
        dim3 grid(blocks_x, blocks_y);

        ln_backward_gamma_beta_kernel<T><<<grid, threads>>>(
            grad_y, x, mean, rstd, grad_gamma, grad_beta, rows, cols);
    }

    // Input gradients
    if (grad_x != nullptr) {
        int threads = (cols > 256) ? 512 : 256;
        ln_backward_input_kernel<T><<<rows, threads>>>(
            grad_y, x, mean, rstd, gamma, grad_x, cols);
    }
}

// =================================================================================
// RMSNorm Backward Launch Helper
// =================================================================================
template<typename T>
static void launch_rms_norm_backward(
    const T* grad_y, const T* x,
    const float* rstd, const T* gamma,
    T* grad_x, float* grad_gamma,
    int rows, int cols)
{
    if (grad_gamma != nullptr) {
        cudaMemset(grad_gamma, 0, cols * sizeof(float));

        dim3 threads(32, 8);
        int blocks_x = (cols + 31) / 32;
        int blocks_y = 128 / blocks_x;
        if (blocks_y < 1) blocks_y = 1;
        if (blocks_y > 32) blocks_y = 32;
        dim3 grid(blocks_x, blocks_y);

        rms_backward_gamma_kernel<T><<<grid, threads>>>(
            grad_y, x, rstd, grad_gamma, rows, cols);
    }

    if (grad_x != nullptr) {
        int threads = (cols > 256) ? 512 : 256;
        rms_backward_input_kernel<T><<<rows, threads>>>(
            grad_y, x, rstd, gamma, grad_x, cols);
    }
}

// =================================================================================
// LayerNorm Backward Launchers
// =================================================================================
void layer_norm_backward_cuda(
    const float* grad_y, const float* x, const float* mean, const float* rstd, const float* gamma,
    float* grad_x, float* grad_gamma, float* grad_beta, int rows, int cols)
{
    launch_layer_norm_backward<float>(grad_y, x, mean, rstd, gamma, grad_x, grad_gamma, grad_beta, rows, cols);
}

void layer_norm_backward_cuda(
    const float16_t* grad_y, const float16_t* x, const float* mean, const float* rstd, const float16_t* gamma,
    float16_t* grad_x, float* grad_gamma, float* grad_beta, int rows, int cols)
{
    launch_layer_norm_backward<__half>(
        reinterpret_cast<const __half*>(grad_y), reinterpret_cast<const __half*>(x),
        mean, rstd, reinterpret_cast<const __half*>(gamma),
        reinterpret_cast<__half*>(grad_x), grad_gamma, grad_beta, rows, cols);
}

void layer_norm_backward_cuda(
    const bfloat16_t* grad_y, const bfloat16_t* x, const float* mean, const float* rstd, const bfloat16_t* gamma,
    bfloat16_t* grad_x, float* grad_gamma, float* grad_beta, int rows, int cols)
{
    launch_layer_norm_backward<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16*>(grad_y), reinterpret_cast<const __nv_bfloat16*>(x),
        mean, rstd, reinterpret_cast<const __nv_bfloat16*>(gamma),
        reinterpret_cast<__nv_bfloat16*>(grad_x), grad_gamma, grad_beta, rows, cols);
}

// =================================================================================
// RMSNorm Backward Launchers
// =================================================================================
void rms_norm_backward_cuda(
    const float* grad_y, const float* x, const float* rstd, const float* gamma,
    float* grad_x, float* grad_gamma, int rows, int cols)
{
    launch_rms_norm_backward<float>(grad_y, x, rstd, gamma, grad_x, grad_gamma, rows, cols);
}

void rms_norm_backward_cuda(
    const float16_t* grad_y, const float16_t* x, const float* rstd, const float16_t* gamma,
    float16_t* grad_x, float* grad_gamma, int rows, int cols)
{
    launch_rms_norm_backward<__half>(
        reinterpret_cast<const __half*>(grad_y), reinterpret_cast<const __half*>(x),
        rstd, reinterpret_cast<const __half*>(gamma),
        reinterpret_cast<__half*>(grad_x), grad_gamma, rows, cols);
}

void rms_norm_backward_cuda(
    const bfloat16_t* grad_y, const bfloat16_t* x, const float* rstd, const bfloat16_t* gamma,
    bfloat16_t* grad_x, float* grad_gamma, int rows, int cols)
{
    launch_rms_norm_backward<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16*>(grad_y), reinterpret_cast<const __nv_bfloat16*>(x),
        rstd, reinterpret_cast<const __nv_bfloat16*>(gamma),
        reinterpret_cast<__nv_bfloat16*>(grad_x), grad_gamma, rows, cols);
}

} // namespace cuda
} // namespace OwnTensor
