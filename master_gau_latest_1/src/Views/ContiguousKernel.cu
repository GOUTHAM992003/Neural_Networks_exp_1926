#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

namespace OwnTensor {

// =============================================================================
// FastDivmod — magic-number fast divisor
// Replaces 64-bit div/mod (~40 cycles) with multiply-high + shift (~6 cycles).
// =============================================================================
struct FastDivmod {
    uint32_t d;
    uint32_t magic;
    uint32_t shift;

    static FastDivmod make(uint32_t divisor) {
        FastDivmod f;
        f.d = divisor;
        f.shift = 0;
        if (divisor == 1) { f.magic = 0; return f; }
        f.magic = (uint32_t)(((uint64_t(1) << 32) + (uint64_t)divisor - 1) / divisor);
        return f;
    }

    __device__ __forceinline__
    void divmod(uint32_t n, uint32_t &q, uint32_t &r) const {
        if (d == 1) { q = n; r = 0; return; }
        q = __umulhi(n, magic);
        if (q * d > n) --q;
        r = n - q * d;
    }
};

struct ContiguousMeta {
    FastDivmod divmods[10];
    int64_t    strides[10];
    int32_t    ndim;
    int64_t    storage_offset_elems;
    int32_t    elem_size;
};

// =============================================================================
// PATH 1: TILED 2D TRANSPOSE (TensorFlow-style, ~2x speedup)
// When the operation reduces to a 2D transpose (dim0, dim1 swap), use shared
// memory tiling with 32x32 tiles. Both read and write are coalesced.
// Bank-conflict avoidance via +1 padding in shared memory.
// =============================================================================
template<typename T, int TileSize = 32, int BlockRows = 8>
__global__ void transpose_2d_tiled_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int rows, int cols)
{
    __shared__ T tile[TileSize][TileSize + 1];  // +1 avoids bank conflicts

    int x = blockIdx.x * TileSize + threadIdx.x;
    int y = blockIdx.y * TileSize + threadIdx.y;

    // Phase 1: coalesced read from src into shared memory
    #pragma unroll
    for (int j = 0; j < TileSize; j += BlockRows) {
        if (x < cols && (y + j) < rows) {
            tile[threadIdx.y + j][threadIdx.x] = src[(y + j) * cols + x];
        }
    }
    __syncthreads();

    // Phase 2: coalesced write from shared memory (transposed) to dst
    x = blockIdx.y * TileSize + threadIdx.x;
    y = blockIdx.x * TileSize + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TileSize; j += BlockRows) {
        if (x < rows && (y + j) < cols) {
            dst[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// 3D batched transpose: batched [B, rows, cols] -> [B, cols, rows]
template<typename T, int TileSize = 32, int BlockRows = 8>
__global__ void transpose_3d_tiled_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int batch, int rows, int cols)
{
    __shared__ T tile[TileSize][TileSize + 1];
    int b = blockIdx.z;
    if (b >= batch) return;

    const T* src_b = src + (int64_t)b * rows * cols;
    T* dst_b = dst + (int64_t)b * rows * cols;

    int x = blockIdx.x * TileSize + threadIdx.x;
    int y = blockIdx.y * TileSize + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TileSize; j += BlockRows) {
        if (x < cols && (y + j) < rows)
            tile[threadIdx.y + j][threadIdx.x] = src_b[(y + j) * cols + x];
    }
    __syncthreads();

    x = blockIdx.y * TileSize + threadIdx.x;
    y = blockIdx.x * TileSize + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TileSize; j += BlockRows) {
        if (x < rows && (y + j) < cols)
            dst_b[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

// =============================================================================
// PATH 2: VECTORIZED COPY (PyTorch-style, ~2x speedup)
// When inner dim is contiguous AND 16-byte aligned, use float4/int4 loads.
// =============================================================================
template<typename VecT, typename T>
__global__ void vectorized_contiguous_copy_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int64_t n_vec_elems)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    const VecT* src_v = reinterpret_cast<const VecT*>(src);
    VecT* dst_v = reinterpret_cast<VecT*>(dst);
    #pragma unroll 4
    for (int64_t i = idx; i < n_vec_elems; i += stride) {
        dst_v[i] = __ldg(&src_v[i]);
    }
}

// =============================================================================
// PATH 3: GENERIC STRIDED COPY (fallback — current kernel)
// Handles any shape/stride pattern with FastDivmod.
// =============================================================================
template<int MaxDims>
__global__ void generic_strided_copy_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int64_t total_elems,
    const __grid_constant__ ContiguousMeta meta)
{
    int64_t i_base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i_base >= total_elems) return;

    #pragma unroll
    for (int v = 0; v < 4; ++v) {
        int64_t i = i_base + v;
        if (i >= total_elems) break;

        uint32_t linear = (uint32_t)i;
        int64_t elem_off = meta.storage_offset_elems;

        #pragma unroll
        for (int d = meta.ndim - 1; d >= 0; --d) {
            uint32_t q, r;
            meta.divmods[d].divmod(linear, q, r);
            elem_off += (int64_t)r * meta.strides[d];
            linear = q;
        }

        int64_t src_byte = elem_off * meta.elem_size;
        int64_t dst_byte = i       * meta.elem_size;

        if (meta.elem_size == 4) {
            *reinterpret_cast<float*>(dst + dst_byte) =
                __ldg(reinterpret_cast<const float*>(src + src_byte));
        } else if (meta.elem_size == 8) {
            *reinterpret_cast<double*>(dst + dst_byte) =
                __ldg(reinterpret_cast<const double*>(src + src_byte));
        } else if (meta.elem_size == 2) {
            *reinterpret_cast<uint16_t*>(dst + dst_byte) =
                __ldg(reinterpret_cast<const uint16_t*>(src + src_byte));
        } else {
            for (int k = 0; k < meta.elem_size; ++k)
                dst[dst_byte + k] = src[src_byte + k];
        }
    }
}

// =============================================================================
// HOST-SIDE DISPATCHER WITH DIMENSION COALESCING
// =============================================================================

// Coalesce adjacent dimensions where strides are aligned.
// E.g. [2,3,4] with strides [12,4,1] → [24] with stride [1]
// Returns new ndim after coalescing.
static int coalesce_dimensions(
    int64_t* dims, int64_t* strides, int ndim)
{
    if (ndim <= 1) return ndim;
    int write = 0;
    for (int read = 1; read < ndim; ++read) {
        // If dims[write] has stride that matches dims[read] * strides[read]
        // (meaning they form a contiguous block), merge them.
        if (strides[write] == dims[read] * strides[read]) {
            dims[write] *= dims[read];
            // stride[write] stays the same (from the inner dim)
            strides[write] = strides[read];
        } else {
            ++write;
            dims[write] = dims[read];
            strides[write] = strides[read];
        }
    }
    return write + 1;
}

// Detect if the coalesced pattern is a 2D transpose: shape [a,b] with strides [1,a]
// Returns true if this is a transpose pattern; fills out rows/cols.
static bool is_2d_transpose(
    const int64_t* dims, const int64_t* strides, int ndim,
    int& rows_out, int& cols_out)
{
    if (ndim != 2) return false;
    // After transpose, outer dim has stride 1 (was inner), inner dim has stride = outer size
    if (strides[0] == 1 && strides[1] == dims[0]) {
        rows_out = (int)dims[1];  // output rows
        cols_out = (int)dims[0];  // output cols = input rows
        return true;
    }
    return false;
}

// Detect 3D batched transpose: shape [B,a,b] with strides for transpose pattern
static bool is_3d_batched_transpose(
    const int64_t* dims, const int64_t* strides, int ndim,
    int& batch_out, int& rows_out, int& cols_out)
{
    if (ndim != 3) return false;
    // Batched transpose: last two dims are transposed
    // Output [B, a, b] from input [B, b, a] with strides [a*b, 1, a]
    if (strides[2] == 1 && strides[1] == dims[2] && strides[0] == dims[1] * dims[2]) {
        batch_out = (int)dims[0];
        rows_out = (int)dims[1];
        cols_out = (int)dims[2];
        return true;
    }
    return false;
}

// Detect if everything is contiguous after coalescing (single dim with stride 1)
static bool is_fully_contiguous(
    const int64_t* strides, int ndim)
{
    return ndim == 1 && strides[0] == 1;
}

// Check if vectorization with N-byte loads is possible
// Requires: inner dim contiguous (stride=1), inner dim size divisible by vec_count
// Pointer alignment is checked at call site.
static bool can_vectorize(
    const int64_t* dims, const int64_t* strides, int ndim,
    int vec_count, int elem_size,
    const void* src, const void* dst)
{
    if (ndim == 0) return false;
    if (strides[ndim - 1] != 1) return false;  // inner not contiguous
    if (dims[ndim - 1] % vec_count != 0) return false;
    // Alignment check (16-byte for float4/int4)
    int align = vec_count * elem_size;
    if ((reinterpret_cast<uintptr_t>(src) % align) != 0) return false;
    if ((reinterpret_cast<uintptr_t>(dst) % align) != 0) return false;
    return true;
}

extern "C" void contiguous_strided_copy_cuda(
    const void* src, void* dst,
    int64_t total_elems,
    const int64_t* dims_in, const int64_t* strides_in, int32_t ndim_in,
    int64_t storage_offset, int32_t elem_size,
    cudaStream_t stream)
{
    constexpr int MaxDims = 12;
    constexpr int Threads = 256;

    // Step 1: Copy dims/strides locally for coalescing
    int64_t dims[12], strides[12];
    int ndim = ndim_in;
    for (int i = 0; i < ndim && i < 12; ++i) {
        dims[i] = dims_in[i];
        strides[i] = strides_in[i];
    }

    // Step 2: Dimension coalescing (merge contiguous runs)
    // Only safe when storage_offset == 0 (otherwise offset logic gets complicated).
    if (storage_offset == 0) {
        ndim = coalesce_dimensions(dims, strides, ndim);
    }

    const uint8_t* src_bytes = reinterpret_cast<const uint8_t*>(src);
    uint8_t* dst_bytes = reinterpret_cast<uint8_t*>(dst);

    // Step 3: DISPATCH — pick the fastest applicable path

    // ── 3a. Fully contiguous → cudaMemcpyAsync (hardware DMA, fastest) ──
    if (storage_offset == 0 && is_fully_contiguous(strides, ndim)) {
        cudaMemcpyAsync(dst_bytes, src_bytes,
                        total_elems * elem_size,
                        cudaMemcpyDeviceToDevice, stream);
        return;
    }

    // ── 3b. 2D transpose → TILED SHARED-MEM KERNEL (TensorFlow-style) ──
    // Common case: matrix transpose, attention head permute (after coalescing)
    {
        int rows = 0, cols = 0;
        if (storage_offset == 0 && is_2d_transpose(dims, strides, ndim, rows, cols)
            && rows >= 16 && cols >= 16) {
            dim3 block(32, 8);
            dim3 grid((cols + 31) / 32, (rows + 31) / 32);
            if (elem_size == 4) {
                transpose_2d_tiled_kernel<float><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const float*>(src), reinterpret_cast<float*>(dst), rows, cols);
                return;
            } else if (elem_size == 2) {
                transpose_2d_tiled_kernel<uint16_t><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const uint16_t*>(src), reinterpret_cast<uint16_t*>(dst), rows, cols);
                return;
            } else if (elem_size == 8) {
                transpose_2d_tiled_kernel<double><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const double*>(src), reinterpret_cast<double*>(dst), rows, cols);
                return;
            }
            // Otherwise fall through to generic
        }
    }

    // ── 3c. 3D batched transpose → TILED 3D KERNEL ──
    {
        int batch = 0, rows = 0, cols = 0;
        if (storage_offset == 0 && is_3d_batched_transpose(dims, strides, ndim, batch, rows, cols)
            && rows >= 16 && cols >= 16) {
            dim3 block(32, 8);
            dim3 grid((cols + 31) / 32, (rows + 31) / 32, batch);
            if (elem_size == 4) {
                transpose_3d_tiled_kernel<float><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const float*>(src), reinterpret_cast<float*>(dst), batch, rows, cols);
                return;
            } else if (elem_size == 2) {
                transpose_3d_tiled_kernel<uint16_t><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const uint16_t*>(src), reinterpret_cast<uint16_t*>(dst), batch, rows, cols);
                return;
            }
            // Otherwise fall through
        }
    }

    // ── 3d. GENERIC FALLBACK: strided copy with FastDivmod ──
    int64_t threads_needed = (total_elems + 3) / 4;
    int64_t blocks = (threads_needed + Threads - 1) / Threads;

    ContiguousMeta meta = {};
    meta.ndim = ndim;
    meta.storage_offset_elems = storage_offset;
    meta.elem_size = elem_size;
    for (int d = 0; d < ndim && d < 10; ++d) {
        meta.divmods[d] = FastDivmod::make((uint32_t)dims[d]);
        meta.strides[d] = strides[d];
    }

    generic_strided_copy_kernel<MaxDims><<<blocks, Threads, 0, stream>>>(
        src_bytes, dst_bytes, total_elems, meta);
}

} // namespace OwnTensor
