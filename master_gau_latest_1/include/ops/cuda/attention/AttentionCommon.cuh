#pragma once
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>
#include <stdio.h>

namespace OwnTensor {

static constexpr int MAX_HD = 256; 

// --- Reduction Primitives ---
static __inline__ __device__ float warpReduceMax(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

static __inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

static __inline__ __device__ float blockReduceMax(float val) {
    __shared__ float warp_vals[32];
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    val = warpReduceMax(val);
    if (laneId == 0) warp_vals[warpId] = val;
    __syncthreads();
    if (threadIdx.x == 0) {
        float bmax = -INFINITY;
        int nw = (blockDim.x + warpSize - 1) / warpSize;
        for (int w = 0; w < nw; ++w) bmax = fmaxf(bmax, warp_vals[w]);
        warp_vals[0] = bmax;
    }
    __syncthreads();
    return warp_vals[0];
}

static __inline__ __device__ float blockReduceSum(float val) {
    __shared__ float warp_vals[32];
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    val = warpReduceSum(val);
    if (laneId == 0) warp_vals[warpId] = val;
    __syncthreads();
    if (threadIdx.x == 0) {
        float bsum = 0.0f;
        int nw = (blockDim.x + warpSize - 1) / warpSize;
        for (int w = 0; w < nw; ++w) bsum += warp_vals[w];
        warp_vals[0] = bsum;
    }
    __syncthreads();
    return warp_vals[0];
}

struct MemEfficientBwdParams {
    const float* __restrict__ Q;
    const float* __restrict__ K;
    const float* __restrict__ V;
    const float* __restrict__ dO;
    const float* __restrict__ LSE;
    const float* __restrict__ D;
    float* __restrict__ dQ;
    float* __restrict__ dK;
    float* __restrict__ dV;
    int T;
    float scale;
    bool is_causal;
};

} // namespace OwnTensor