#include "ops/helpers/AttentionKernels.h"
#include "ops/cuda/attention/AttentionCommon.cuh"

namespace OwnTensor {

// ============================================================================
// Ada Lovelace SM89 Optimised Backward Kernel
// exp12: BM=32, BN=16, bank-conflict-free padding (+4 per dim)
// smem HD=64: 43.8 KB → 2 blocks/SM, 16 active warps, 50% theoretical occupancy
// ============================================================================

static constexpr int SM89_BWD_WARP_SZ      = 32;
static constexpr int SM89_BWD_NUM_THREADS   = 256;   // 8 warps × 32 lanes
static constexpr int SM89_BWD_BLOCK_M_D     = 8;

__inline__ __device__ float sm89_bwd_warp_sum(float val) {
    #pragma unroll
    for (int offset = SM89_BWD_WARP_SZ / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// ── Precompute D = rowsum(dO ⊙ O) ────────────────────────────────────────────
// Separate name from AttentionBackward.cu's copy to avoid linker symbol clash.
template<int HeadDim>
__global__ void mem_efficient_bwd_precompute_D_sm89(
    const float* __restrict__ dO,
    const float* __restrict__ O,
    float* __restrict__ D,
    int T)
{
    constexpr int LocalN = (HeadDim + SM89_BWD_WARP_SZ - 1) / SM89_BWD_WARP_SZ;

    const int bh      = blockIdx.y;
    const int warp_id = threadIdx.x / SM89_BWD_WARP_SZ;
    const int lane_id = threadIdx.x % SM89_BWD_WARP_SZ;

    const int base_row = blockIdx.x * (SM89_BWD_BLOCK_M_D * 2) + warp_id * 2;
    const int row0     = base_row;
    const int row1     = base_row + 1;
    const bool v0      = (row0 < T);
    const bool v1      = (row1 < T);

    const long long bh_base = (long long)bh * T * HeadDim;
    const long long off0    = bh_base + (long long)row0 * HeadDim;
    const long long off1    = bh_base + (long long)row1 * HeadDim;

    float sum0 = 0.f, sum1 = 0.f;
    #pragma unroll
    for (int i = 0; i < LocalN; ++i) {
        const int k = lane_id + i * SM89_BWD_WARP_SZ;
        if (k < HeadDim) {
            if (v0) sum0 += __ldg(&dO[off0 + k]) * __ldg(&O[off0 + k]);
            if (v1) sum1 += __ldg(&dO[off1 + k]) * __ldg(&O[off1 + k]);
        }
    }
    sum0 = sm89_bwd_warp_sum(sum0);
    sum1 = sm89_bwd_warp_sum(sum1);
    if (lane_id == 0) {
        if (v0) D[(long long)bh * T + row0] = sum0;
        if (v1) D[(long long)bh * T + row1] = sum1;
    }
}

// ============================================================================
// exp12: Q-tile-centric backward, TF32 WMMA, dQ atomic-free
// BM=32, BN=16 — sweet-spot for SM89 (43.8 KB, 2 blocks/SM, 16 active warps)
// Phase A 4/8 warps active; 4 dK/dV k-stages per iteration.
// ============================================================================

template <int HeadDim, bool Causal>
__launch_bounds__(256, 2)
__global__ void mem_efficient_bwd_unified_kernel_exp12(MemEfficientBwdParams params)
{
    using namespace nvcuda;

    constexpr int BlockN    = 16;
    constexpr int BM_WMMA   = 32;
    constexpr int BM_TILES  = BM_WMMA / 16;   // 2 — row groups
    constexpr int HD_CHUNKS = HeadDim / 16;    // 4 for HD=64
    constexpr int HD_PAD    = HeadDim + 4;     // 68 for HD=64  gcd(68,32)=4 → 2-way
    constexpr int BKN_PAD   = BlockN  + 4;     // 20            gcd(20,32)=4 → 2-way
    constexpr int BM_PAD    = BM_WMMA + 4;     // 36            gcd(36,32)=4 → 2-way
    constexpr float BWD_LOG2E = 1.4426950408889634074f;

    extern __shared__ float smem_f[];
    float* Q_sm    = smem_f;
    float* dO_sm   = Q_sm    + BM_WMMA * HD_PAD;   // [32×68]
    float* Ks      = dO_sm   + BM_WMMA * HD_PAD;   // [16×68]
    float* Vs      = Ks      + BlockN   * HD_PAD;
    float* ds_qd   = Vs      + BlockN   * HD_PAD;   // [32×20]
    float* DPV_sm  = ds_qd   + BM_WMMA  * BKN_PAD;
    float* ds_kd   = DPV_sm  + BM_WMMA  * BKN_PAD; // [16×36]  (BN×BM_PAD)
    float* p_kd    = ds_kd   + BlockN   * BM_PAD;
    float* tile_st = p_kd    + BlockN   * BM_PAD;   // [32×68] — final dQ + per-iter dK/dV
    float* LSE_sm  = tile_st + BM_WMMA  * HD_PAD;   // [32]
    float* D_sm    = LSE_sm  + BM_WMMA;             // [32]

    const int bh           = blockIdx.y;
    const int q_tile       = blockIdx.x;
    const int q_tile_start = q_tile * BM_WMMA;
    const int tile_size    = min(BM_WMMA, params.T - q_tile_start);
    if (q_tile_start >= params.T) return;

    const int warp_id = threadIdx.x / SM89_BWD_WARP_SZ;
    const int chunk   = warp_id % HD_CHUNKS;

    const long long bh_off = (long long)bh * params.T * HeadDim;
    const long long bh_T   = (long long)bh * params.T;
    const float* Q_bh   = params.Q   + bh_off;
    const float* K_bh   = params.K   + bh_off;
    const float* V_bh   = params.V   + bh_off;
    const float* dO_bh  = params.dO  + bh_off;
    const float* LSE_bh = params.LSE + bh_T;
    const float* D_bh   = params.D   + bh_T;
    float*       dQ_bh  = params.dQ  + bh_off;
    float*       dK_bh  = params.dK  + bh_off;
    float*       dV_bh  = params.dV  + bh_off;

    // Step 0: load 32 Q rows + dO rows + LSE + D into smem
    // 32×64 = 2048 elements; 256 threads → 8 iterations each
    for (int idx = threadIdx.x; idx < BM_WMMA * HeadDim; idx += blockDim.x) {
        const int r  = idx / HeadDim, k = idx % HeadDim;
        const int qi = q_tile_start + r;
        const bool vq = (qi < params.T);
        Q_sm [r * HD_PAD + k] = vq ? Q_bh [qi * HeadDim + k] : 0.f;
        dO_sm[r * HD_PAD + k] = vq ? dO_bh[qi * HeadDim + k] : 0.f;
    }
    // BM_WMMA=32 < 256, so all 32 LSE/D values loaded by first 32 threads
    if (threadIdx.x < BM_WMMA) {
        const int qi  = q_tile_start + threadIdx.x;
        const bool vq = (qi < params.T);
        LSE_sm[threadIdx.x] = vq ? LSE_bh[qi] : 0.f;
        D_sm  [threadIdx.x] = vq ? D_bh  [qi] : 0.f;
    }
    __syncthreads();

    // 2 persistent dQ accumulators per warp (warps 0..HD_CHUNKS-1)
    // Each covers one [16×16] row-group × HD-chunk tile of dQ[32×64]
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> dq_frag[BM_TILES];
    #pragma unroll
    for (int rg = 0; rg < BM_TILES; rg++) wmma::fill_fragment(dq_frag[rg], 0.0f);

    const int kv_loop_end = Causal ? (q_tile_start + BM_WMMA) : params.T;

    for (int kv_base = 0; kv_base < kv_loop_end; kv_base += BlockN) {
        __syncthreads();  // (a)
        const int kv_tile_size = min(BlockN, params.T - kv_base);

        for (int idx = threadIdx.x; idx < BlockN * HeadDim; idx += blockDim.x) {
            const int r = idx / HeadDim, k = idx % HeadDim;
            const int g = kv_base + r;
            Ks[r * HD_PAD + k] = (g < params.T) ? K_bh[g * HeadDim + k] : 0.f;
            Vs[r * HD_PAD + k] = (g < params.T) ? V_bh[g * HeadDim + k] : 0.f;
        }
        __syncthreads();  // (b)

        // ── Phase A: warps 0-3 active (4/8) ──────────────────────────────────
        // warps 0-1: QK^T,  warp j → ds_qd  [j*16:j*16+16][0:16]
        // warps 2-3: DPV,   warp j → DPV_sm [(j-2)*16:(j-2)*16+16][0:16]
        if (warp_id < 2 * BM_TILES) {
            const int  row_group = warp_id % BM_TILES;
            const bool is_qk     = (warp_id < BM_TILES);
            const float* src_sm  = (is_qk ? Q_sm  : dO_sm) + row_group * 16 * HD_PAD;
            const float* kv_sm   =  is_qk ? Ks    : Vs;
            float*       dst_sm  = (is_qk ? ds_qd : DPV_sm) + row_group * 16 * BKN_PAD;

            wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc_frag;
            wmma::fill_fragment(acc_frag, 0.0f);
            wmma::fragment<wmma::matrix_a, 16, 16, 8,
                           wmma::precision::tf32, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 8,
                           wmma::precision::tf32, wmma::col_major> b_frag;
            #pragma unroll
            for (int ks = 0; ks < 2 * HD_CHUNKS; ++ks) {
                wmma::load_matrix_sync(a_frag, src_sm + ks * 8, HD_PAD);
                wmma::load_matrix_sync(b_frag, kv_sm  + ks * 8, HD_PAD);
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
            wmma::store_matrix_sync(dst_sm, acc_frag, BKN_PAD, wmma::mem_row_major);
        }
        __syncthreads();  // (c1)

        // Scalar post-process: 32×16=512 elements, 256 threads → 2 each
        for (int elem = threadIdx.x; elem < BM_WMMA * BlockN; elem += blockDim.x) {
            const int qi_local  = elem / BlockN;
            const int j_local   = elem % BlockN;
            const float raw_s = ds_qd [qi_local * BKN_PAD + j_local];
            const float dpv   = DPV_sm[qi_local * BKN_PAD + j_local];
            const float L     = LSE_sm[qi_local];
            const float D_    = D_sm  [qi_local];
            const bool qi_ok  = ((q_tile_start + qi_local) < params.T);
            const bool j_ok   = (j_local < kv_tile_size);
            const bool cok    = !Causal || ((kv_base + j_local) <= (q_tile_start + qi_local));
            float p = 0.f;
            if (qi_ok && j_ok && cok)
                p = exp2f(BWD_LOG2E * (raw_s * params.scale - L));
            const float ds = p * (dpv - D_) * params.scale;
            ds_qd[qi_local * BKN_PAD + j_local]  = ds;
            ds_kd[j_local  * BM_PAD  + qi_local] = ds;
            p_kd [j_local  * BM_PAD  + qi_local] = p;
        }
        __syncthreads();  // (c2)

        // ── Phase B1: warps 0-3 → dQ (2 frags each), warps 4-7 → dK (4 stages)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 8,
                           wmma::precision::tf32, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 8,
                           wmma::precision::tf32, wmma::row_major> b_frag;
            if (warp_id < HD_CHUNKS) {
                const float* b_ptr = Ks + chunk * 16;
                #pragma unroll
                for (int rg = 0; rg < BM_TILES; rg++) {
                    const float* a_base = ds_qd + rg * 16 * BKN_PAD;
                    wmma::load_matrix_sync(a_frag, a_base,             BKN_PAD);
                    wmma::load_matrix_sync(b_frag, b_ptr,              HD_PAD);
                    wmma::mma_sync(dq_frag[rg], a_frag, b_frag, dq_frag[rg]);
                    wmma::load_matrix_sync(a_frag, a_base + 8,         BKN_PAD);
                    wmma::load_matrix_sync(b_frag, b_ptr + 8 * HD_PAD, HD_PAD);
                    wmma::mma_sync(dq_frag[rg], a_frag, b_frag, dq_frag[rg]);
                }
            } else {
                // dK: A=ds_kd[BN=16 × BM=32], k-dim=32 → 4 stages of 8
                wmma::fragment<wmma::accumulator, 16, 16, 8, float> dk_frag;
                wmma::fill_fragment(dk_frag, 0.0f);
                const float* b_ptr = Q_sm + chunk * 16;
                #pragma unroll
                for (int ks = 0; ks < BM_TILES * 2; ks++) {
                    wmma::load_matrix_sync(a_frag, ds_kd + ks * 8,          BM_PAD);
                    wmma::load_matrix_sync(b_frag, b_ptr + ks * 8 * HD_PAD, HD_PAD);
                    wmma::mma_sync(dk_frag, a_frag, b_frag, dk_frag);
                }
                wmma::store_matrix_sync(tile_st + chunk * 16, dk_frag,
                                        HD_PAD, wmma::mem_row_major);
            }
        }
        __syncthreads();  // (d)

        for (int idx = threadIdx.x; idx < kv_tile_size * HeadDim; idx += blockDim.x) {
            const int r = idx / HeadDim, k = idx % HeadDim;
            atomicAdd(&dK_bh[(kv_base + r) * HeadDim + k], tile_st[r * HD_PAD + k]);
        }
        __syncthreads();  // (e)

        // ── Phase B2: warps 4-7 → dV (4 k-stages) ───────────────────────────
        if (warp_id >= HD_CHUNKS) {
            wmma::fragment<wmma::matrix_a, 16, 16, 8,
                           wmma::precision::tf32, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 8,
                           wmma::precision::tf32, wmma::row_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 8, float> dv_frag;
            wmma::fill_fragment(dv_frag, 0.0f);
            const float* b_ptr = dO_sm + chunk * 16;
            #pragma unroll
            for (int ks = 0; ks < BM_TILES * 2; ks++) {
                wmma::load_matrix_sync(a_frag, p_kd + ks * 8,           BM_PAD);
                wmma::load_matrix_sync(b_frag, b_ptr + ks * 8 * HD_PAD, HD_PAD);
                wmma::mma_sync(dv_frag, a_frag, b_frag, dv_frag);
            }
            wmma::store_matrix_sync(tile_st + chunk * 16, dv_frag,
                                    HD_PAD, wmma::mem_row_major);
        }
        __syncthreads();  // (f)

        for (int idx = threadIdx.x; idx < kv_tile_size * HeadDim; idx += blockDim.x) {
            const int r = idx / HeadDim, k = idx % HeadDim;
            atomicAdd(&dV_bh[(kv_base + r) * HeadDim + k], tile_st[r * HD_PAD + k]);
        }
    }

    // ── Final dQ store: dq_frags → tile_st → global dQ (no atomicAdd) ────────
    __syncthreads();
    if (warp_id < HD_CHUNKS) {
        #pragma unroll
        for (int rg = 0; rg < BM_TILES; rg++) {
            wmma::store_matrix_sync(tile_st + rg * 16 * HD_PAD + chunk * 16,
                                    dq_frag[rg], HD_PAD, wmma::mem_row_major);
        }
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < tile_size * HeadDim; idx += blockDim.x) {
        const int r = idx / HeadDim, k = idx % HeadDim;
        dQ_bh[(q_tile_start + r) * HeadDim + k] = tile_st[r * HD_PAD + k];
    }
}

// ============================================================================
// SM89 public entry point — called by mem_efficient_attn_backward when on Ada.
// Caller guarantees hd % 16 == 0 (non-%16 cases are handled by exp7 in
// AttentionBackward.cu before this function is reached).
// ============================================================================

void mem_efficient_attn_backward_sm89_cuda(
    const float* query, const float* key, const float* value,
    const float* output, const float* grad_output, const float* lse,
    float* grad_query, float* grad_key, float* grad_value,
    float* D_buf,
    int64_t B, int64_t nh, int64_t T, int64_t hd,
    bool is_causal)
{
    float scale = 1.0f / sqrtf(static_cast<float>(hd));
    const int BH = (int)(B * nh);
    dim3 block_cfg(SM89_BWD_NUM_THREADS);
    dim3 grid_D(((int)T + (SM89_BWD_BLOCK_M_D * 2) - 1) / (SM89_BWD_BLOCK_M_D * 2), BH);

    MemEfficientBwdParams params;
    params.Q        = query;
    params.K        = key;
    params.V        = value;
    params.dO       = grad_output;
    params.LSE      = lse;
    params.D        = D_buf;
    params.dQ       = grad_query;
    params.dK       = grad_key;
    params.dV       = grad_value;
    params.T        = (int)T;
    params.scale    = scale;
    params.is_causal = is_causal;

    // exp12: BM=32, BN=16 — 43.8 KB smem, 2 blocks/SM on SM89
#define LAUNCH_MEM_BWD_EXP12(HD) \
    do { \
        constexpr int BN12 = 16, BM12 = 32; \
        const size_t shmem12 = \
            (2ULL * BM12 * ((HD) + 4)   /* Q_sm + dO_sm   */ \
           + 2ULL * BN12 * ((HD) + 4)   /* Ks + Vs        */ \
           + 2ULL * BM12 * ((BN12) + 4) /* ds_qd + DPV_sm */ \
           + 2ULL * BN12 * ((BM12) + 4) /* ds_kd + p_kd   */ \
           + 1ULL * BM12 * ((HD) + 4)   /* tile_st        */ \
           + 2ULL * BM12                /* LSE_sm + D_sm  */ \
            ) * sizeof(float); \
        const int q12 = ((int)T + BM12 - 1) / BM12; \
        dim3 grid_q12(q12, BH); \
        cudaFuncSetAttribute( \
            mem_efficient_bwd_unified_kernel_exp12<HD, false>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem12); \
        cudaFuncSetAttribute( \
            mem_efficient_bwd_unified_kernel_exp12<HD, true>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem12); \
        cudaMemsetAsync(params.dK, 0, (size_t)BH * (int)T * (HD) * sizeof(float)); \
        cudaMemsetAsync(params.dV, 0, (size_t)BH * (int)T * (HD) * sizeof(float)); \
        mem_efficient_bwd_precompute_D_sm89<HD><<<grid_D, block_cfg>>>( \
            grad_output, output, D_buf, (int)T); \
        if (is_causal) { \
            mem_efficient_bwd_unified_kernel_exp12<HD, true> \
                <<<grid_q12, block_cfg, shmem12>>>(params); \
        } else { \
            mem_efficient_bwd_unified_kernel_exp12<HD, false> \
                <<<grid_q12, block_cfg, shmem12>>>(params); \
        } \
    } while (0)

    switch ((int)hd) {
        case  16: LAUNCH_MEM_BWD_EXP12( 16); break;
        case  32: LAUNCH_MEM_BWD_EXP12( 32); break;
        case  48: LAUNCH_MEM_BWD_EXP12( 48); break;
        case  64: LAUNCH_MEM_BWD_EXP12( 64); break;
        case  80: LAUNCH_MEM_BWD_EXP12( 80); break;
        case  96: LAUNCH_MEM_BWD_EXP12( 96); break;
        case 128: LAUNCH_MEM_BWD_EXP12(128); break;
        case 160: LAUNCH_MEM_BWD_EXP12(160); break;
        case 192: LAUNCH_MEM_BWD_EXP12(192); break;
        case 256: LAUNCH_MEM_BWD_EXP12(256); break;
        default:
            printf("mem_efficient_attn_backward_sm89: unsupported head_dim %d\n", (int)hd);
            break;
    }
#undef LAUNCH_MEM_BWD_EXP12
}

} // namespace OwnTensor
