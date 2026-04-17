#include <cuda_runtime.h>
#include <cstdint>

namespace OwnTensor {

// Pass dims/strides by value to avoid cudaMallocAsync overhead
// struct ContiguousMeta {
//     int64_t dims[12];
//     int64_t strides[12];
//     int32_t ndim;
//     int64_t storage_offset_elems;
//     int32_t elem_size;
// };

// template<int MaxDims>
// __global__ void contiguous_strided_copy_kernel(
//     const uint8_t* __restrict__ src,
//     uint8_t* __restrict__ dst,
//     int64_t total_elems,
//     ContiguousMeta meta)
// {
//     int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= total_elems) return;

//     // Convert output linear index to multi-index
//     int64_t idx[MaxDims];
//     {
//         int64_t linear = i;
//         for (int d = meta.ndim - 1; d >= 0; --d) {
//             idx[d] = linear % meta.dims[d];
//             linear /= meta.dims[d];
//         }
//     }

//     // Compute source element offset using strides
//     int64_t elem_off = meta.storage_offset_elems;

//     #pragma unroll
//     for (int d = 0; d < meta.ndim; ++d) {
//         elem_off += idx[d] * meta.strides[d];
//     }

//     // Compute byte offsets
//     int64_t src_byte = elem_off * meta.elem_size;
//     int64_t dst_byte = i * meta.elem_size;

//     // Use typed copy for common element sizes to get coalesced memory access
//     if (meta.elem_size == 4) {
//         *reinterpret_cast<float*>(dst + dst_byte) =
//             *reinterpret_cast<const float*>(src + src_byte);
//     } else if (meta.elem_size == 2) {
//         *reinterpret_cast<uint16_t*>(dst + dst_byte) =
//             *reinterpret_cast<const uint16_t*>(src + src_byte);
//     } else if (meta.elem_size == 8) {
//         *reinterpret_cast<double*>(dst + dst_byte) =
//             *reinterpret_cast<const double*>(src + src_byte);
//     } else {
//         for (int k = 0; k < meta.elem_size; ++k) {
//             (dst + dst_byte)[k] = (src + src_byte)[k];
//         }
//     }
// }

// extern "C" void contiguous_strided_copy_cuda(
//     const void* src, void* dst,
//     int64_t total_elems,
//     const int64_t* dims, const int64_t* strides, int32_t ndim,
//     int64_t storage_offset, int32_t elem_size,
//     cudaStream_t stream)
// {
//     constexpr int MaxDims = 12;
//     constexpr int Threads = 256;
//     int64_t blocks = (total_elems + Threads - 1) / Threads;

//     // Build metadata on stack — passed by value to kernel (no device malloc)
//     ContiguousMeta meta = {};
//     meta.ndim = ndim;
//     meta.storage_offset_elems = storage_offset;
//     meta.elem_size = elem_size;
//     for (int d = 0; d < ndim && d < MaxDims; ++d) {
//         meta.dims[d] = dims[d];
//         meta.strides[d] = strides[d];
//     }

//     contiguous_strided_copy_kernel<MaxDims><<<blocks, Threads, 0, stream>>>(
//         reinterpret_cast<const uint8_t*>(src),
//         reinterpret_cast<uint8_t*>(dst),
//         total_elems,
//         meta
//     );
// }

struct ContiguousMeta {
    int64_t dims[10];
    int64_t strides[10];
    int32_t ndim;
    int64_t storage_offset_elems;
    int32_t elem_size;
};

// Optimization: Template on MaxDims to allow the compiler to 
// aggressively unroll loops and optimize index math [11, 12].
template<int MaxDims>
__global__ void contiguous_strided_copy_kernel_optimized(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int64_t total_elems,
    // Optimization: Use __grid_constant__ to place metadata in 
    // read-only constant cache for faster broadcast to all threads [3, 4].
    const __grid_constant__ ContiguousMeta meta)
{
    // Optimization: Thread coarsening. Each thread processes 4 elements (vectorization).
    // This reduces the number of expensive divisions/modulos performed across the grid [13, 14].
    int64_t i_base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (i_base >= total_elems) return;

    // Use a loop for vectorization (handling up to 4 elements per thread)
    #pragma unroll
    for (int v = 0; v < 4; ++v) {
        int64_t i = i_base + v;
        if (i >= total_elems) break;

        // Convert linear index to multi-index
        // Note: Sources suggest using bitwise shifts if dims are powers of 2 [9].
        int64_t idx[MaxDims];
        int64_t linear = i;
        
        #pragma unroll
        for (int d = meta.ndim - 1; d >= 0; --d) {
            idx[d] = linear % meta.dims[d];
            linear /= meta.dims[d];
        }

        // Compute source element offset
        int64_t elem_off = meta.storage_offset_elems;
        #pragma unroll
        for (int d = 0; d < meta.ndim; ++d) {
            elem_off += idx[d] * meta.strides[d];
        }

        int64_t src_byte = elem_off * meta.elem_size;
        int64_t dst_byte = i * meta.elem_size;

        // Optimization: Use wider memory transactions for common sizes [1, 2].
        // Vectorized loads/stores maximize global memory throughput.
        if (meta.elem_size == 4) {
            // Using __ldg() tells the compiler to use the read-only data cache [15, 16].
            *reinterpret_cast<float*>(dst + dst_byte) = 
                __ldg(reinterpret_cast<const float*>(src + src_byte));
        } else if (meta.elem_size == 8) {
            *reinterpret_cast<double*>(dst + dst_byte) = 
                __ldg(reinterpret_cast<const double*>(src + src_byte));
        } else {
            // Standard byte-wise copy for odd sizes
            for (int k = 0; k < meta.elem_size; ++k) {
                dst[dst_byte + k] = src[src_byte + k];
            }
        }
    }
}

extern "C" void contiguous_strided_copy_cuda(
    const void* src, void* dst,
    int64_t total_elems,
    const int64_t* dims, const int64_t* strides, int32_t ndim,
    int64_t storage_offset, int32_t elem_size,
    cudaStream_t stream)
{
    constexpr int MaxDims = 12;
    // Optimization: Block size as multiple of 32 for warp efficiency [17, 18].
    constexpr int Threads = 256;
    
    // Account for coarsening: 4 elements per thread
    int64_t threads_needed = (total_elems + 3) / 4;
    int64_t blocks = (threads_needed + Threads - 1) / Threads;

    ContiguousMeta meta = {};
    meta.ndim = ndim;
    meta.storage_offset_elems = storage_offset;
    meta.elem_size = elem_size;
    for (int d = 0; d < ndim && d < MaxDims; ++d) {
        meta.dims[d] = dims[d];
        meta.strides[d] = strides[d];
    }

    contiguous_strided_copy_kernel_optimized<MaxDims><<<blocks, Threads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(src),
        reinterpret_cast<uint8_t*>(dst),
        total_elems,
        meta
    );
}

} // namespace OwnTensor