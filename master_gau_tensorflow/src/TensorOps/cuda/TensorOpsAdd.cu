#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "ops/TensorOps.cuh"
#include "core/Tensor.h"
#include "core/TensorDispatch.h"

#include <stdio.h>

namespace OwnTensor
{
    // Helper to get number of SMs
    inline int get_num_sms() {
        int deviceId;
        cudaGetDevice(&deviceId);
        int numSMs;
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId);
        return numSMs;
    }

    template <typename T>
    __global__ __launch_bounds__(256)
    void add_kernel(const T *a, const T *b, T *output, size_t n)
    {
        size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
        size_t stride = (size_t)blockDim.x * gridDim.x;
        for (size_t i = idx; i < n; i += stride)
        {
            output[i] = a[i] + b[i];
        }
    }

    template <>
    __global__ void add_kernel<__half>(const __half *a, const __half *b, __half *output, size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;
        #pragma unroll 4
        for (size_t i = idx; i < n; i += stride)
        {
            output[i] = __hadd(a[i], b[i]);
        }
    }

    template <>
    __global__ void add_kernel<__nv_bfloat16>(const __nv_bfloat16 *a, const __nv_bfloat16 *b, __nv_bfloat16 *output, size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;
        #pragma unroll 4
        for (size_t i = idx; i < n; i += stride)
        {
            output[i] = __hadd(a[i], b[i]);
        }
    }

    template <typename T>
    __global__ void add_kernel_nd_broadcast(const T *a, const T *b, T *output,
                                            const size_t *a_shape, const size_t *b_shape, const size_t *out_shape,
                                            const size_t *a_strides, const size_t *b_strides, const size_t *out_strides,
                                            size_t a_ndim, size_t b_ndim, size_t out_ndim,
                                            size_t total_elems)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;

        #pragma unroll 4
        for (size_t linear_idx = idx; linear_idx < total_elems; linear_idx += stride)
        {
            size_t temp_idx = linear_idx;
            size_t a_idx = 0;
            size_t b_idx = 0;

            for (int dim = out_ndim - 1; dim >= 0; --dim)
            {
                size_t coord = temp_idx % out_shape[dim];
                temp_idx /= out_shape[dim];
                
                int a_dim_matched = (int)a_ndim - (int)out_ndim + dim;
                if (a_dim_matched >= 0 && a_shape[a_dim_matched] > 1) {
                    a_idx += coord * a_strides[a_dim_matched];
                }

                int b_dim_matched = (int)b_ndim - (int)out_ndim + dim;
                if (b_dim_matched >= 0 && b_shape[b_dim_matched] > 1) {
                     b_idx += coord * b_strides[b_dim_matched];
                }
            }

            output[linear_idx] = a[a_idx] + b[b_idx];
        }
    }

    template <>
    __global__ void add_kernel_nd_broadcast<__half>(const __half *a, const __half *b, __half *output,
                                                    const size_t *a_shape, const size_t *b_shape, const size_t *out_shape,
                                                    const size_t *a_strides, const size_t *b_strides, const size_t *out_strides,
                                                    size_t a_ndim, size_t b_ndim, size_t out_ndim,
                                                    size_t total_elems)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;

        #pragma unroll 4
        for (size_t linear_idx = idx; linear_idx < total_elems; linear_idx += stride)
        {
            size_t temp_idx = linear_idx;
            size_t a_idx = 0;
            size_t b_idx = 0;

            for (int dim = out_ndim - 1; dim >= 0; --dim)
            {
                size_t coord = temp_idx % out_shape[dim];
                temp_idx /= out_shape[dim];
                
                int a_dim_matched = (int)a_ndim - (int)out_ndim + dim;
                if (a_dim_matched >= 0 && a_shape[a_dim_matched] > 1) {
                    a_idx += coord * a_strides[a_dim_matched];
                }

                int b_dim_matched = (int)b_ndim - (int)out_ndim + dim;
                if (b_dim_matched >= 0 && b_shape[b_dim_matched] > 1) {
                     b_idx += coord * b_strides[b_dim_matched];
                }
            }

            output[linear_idx] = __hadd(a[a_idx], b[b_idx]);
        }
    }

    template <>
    __global__ void add_kernel_nd_broadcast<__nv_bfloat16>(const __nv_bfloat16 *a, const __nv_bfloat16 *b, __nv_bfloat16 *output,
                                                           const size_t *a_shape, const size_t *b_shape, const size_t *out_shape,
                                                           const size_t *a_strides, const size_t *b_strides, const size_t *out_strides,
                                                           size_t a_ndim, size_t b_ndim, size_t out_ndim,
                                                           size_t total_elems)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;

        for (size_t linear_idx = idx; linear_idx < total_elems; linear_idx += stride)
        {
            size_t temp_idx = linear_idx;
            size_t a_idx = 0;
            size_t b_idx = 0;

            for (int dim = out_ndim - 1; dim >= 0; --dim)
            {
                size_t coord = temp_idx % out_shape[dim];
                temp_idx /= out_shape[dim];
                
                int a_dim_matched = (int)a_ndim - (int)out_ndim + dim;
                if (a_dim_matched >= 0 && a_shape[a_dim_matched] > 1) {
                    a_idx += coord * a_strides[a_dim_matched];
                }

                int b_dim_matched = (int)b_ndim - (int)out_ndim + dim;
                if (b_dim_matched >= 0 && b_shape[b_dim_matched] > 1) {
                     b_idx += coord * b_strides[b_dim_matched];
                }
            }

            output[linear_idx] = __hadd(a[a_idx], b[b_idx]);
        }
    }

    void cuda_add_tensor(const Tensor &A, const Tensor &B, Tensor &output, cudaStream_t stream)
    {
        bool needs_broadcasting = (A.shape().dims != B.shape().dims);
        size_t total_elems = output.numel();
        size_t block_size = 256;
        
        static int num_sms = get_num_sms();
        size_t num_blocks = num_sms * 4;
        
        size_t max_blocks = (total_elems + block_size - 1) / block_size;
        size_t grid_size = std::min(num_blocks, max_blocks);
        if (grid_size == 0) grid_size = 1;

        dispatch_by_dtype(A.dtype(), [&](auto dummy)
                          {
        using T = decltype(dummy);
        const T* a_ptr = A.data<T>();
        const T* b_ptr = B.data<T>();
        T* output_ptr = output.data<T>();
        
        if (!needs_broadcasting) {
            add_kernel<<<grid_size, block_size, 0, stream>>>(a_ptr, b_ptr, output_ptr, total_elems);
        } else {
            const auto& a_shape = A.shape().dims;
            const auto& b_shape = B.shape().dims;
            const auto& out_shape = output.shape().dims;
            
            size_t a_ndim = a_shape.size();
            size_t b_ndim = b_shape.size();
            size_t out_ndim = out_shape.size();
            
            size_t *d_a_shape, *d_b_shape, *d_out_shape;
            size_t *d_a_strides, *d_b_strides, *d_out_strides;
            
            cudaMallocAsync(&d_a_shape, a_ndim * sizeof(size_t), stream);
            cudaMallocAsync(&d_b_shape, b_ndim * sizeof(size_t), stream);
            cudaMallocAsync(&d_out_shape, out_ndim * sizeof(size_t), stream);
            cudaMallocAsync(&d_a_strides, a_ndim * sizeof(size_t), stream);
            cudaMallocAsync(&d_b_strides, b_ndim * sizeof(size_t), stream);
            cudaMallocAsync(&d_out_strides, out_ndim * sizeof(size_t), stream);
            
            cudaMemcpyAsync(d_a_shape, a_shape.data(), a_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_b_shape, b_shape.data(), b_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_out_shape, out_shape.data(), out_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_a_strides, A.stride().strides.data(), a_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_b_strides, B.stride().strides.data(), b_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_out_strides, output.stride().strides.data(), out_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);
            
            add_kernel_nd_broadcast<<<grid_size, block_size, 0, stream>>>(
                a_ptr, b_ptr, output_ptr,
                d_a_shape, d_b_shape, d_out_shape,
                d_a_strides, d_b_strides, d_out_strides,
                a_ndim, b_ndim, out_ndim, total_elems
            );
            
            cudaFreeAsync(d_a_shape, stream);
            cudaFreeAsync(d_b_shape, stream);
            cudaFreeAsync(d_out_shape, stream);
            cudaFreeAsync(d_a_strides, stream);
            cudaFreeAsync(d_b_strides, stream);
            cudaFreeAsync(d_out_strides, stream);
        } });
    }

    /*########################################################
                TENSOR INPLACE CUDA KERNELS
    ##########################################################*/

    template <typename T>
    __global__ void add_inplace_kernel(T *lhs, const T *rhs, size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;
        #pragma unroll 4
        for (size_t i = idx; i < n; i += stride)
        {
            lhs[i] += rhs[i];
        }
    }

    template <>
    __global__ void add_inplace_kernel<__half>(__half *lhs, const __half *rhs, size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;
        #pragma unroll 4
        for (size_t i = idx; i < n; i += stride)
        {
            lhs[i] = __hadd(lhs[i], rhs[i]);
        }
    }

    template <>
    __global__ void add_inplace_kernel<__nv_bfloat16>(__nv_bfloat16 *lhs, const __nv_bfloat16 *rhs, size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;
        for (size_t i = idx; i < n; i += stride)
        {
            lhs[i] = __hadd(lhs[i], rhs[i]);
        }
    }

    template <typename T>
    __global__ void add_inplace_kernel_broadcast(T *lhs, const T *rhs,
                                                 size_t lhs_rows, size_t lhs_cols,
                                                 size_t rhs_rows, size_t rhs_cols,
                                                 size_t out_rows, size_t out_cols)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;
        size_t total_elems = out_rows * out_cols;

        #pragma unroll 4
        for (size_t i = idx; i < total_elems; i += stride)
        {
            size_t r = i / out_cols;
            size_t c = i % out_cols;

            size_t lhs_row_stride = (lhs_rows == 1) ? 0 : lhs_cols;
            size_t lhs_col_stride = (lhs_cols == 1) ? 0 : 1;
            size_t rhs_row_stride = (rhs_rows == 1) ? 0 : rhs_cols;
            size_t rhs_col_stride = (rhs_cols == 1) ? 0 : 1;

            size_t lhs_idx = (r * lhs_row_stride) + (c * lhs_col_stride);
            size_t rhs_idx = (r * rhs_row_stride) + (c * rhs_col_stride);

            lhs[lhs_idx] += rhs[rhs_idx];
        }
    }

    template <>
    __global__ void add_inplace_kernel_broadcast<__half>(__half *lhs, const __half *rhs,
                                                         size_t lhs_rows, size_t lhs_cols,
                                                         size_t rhs_rows, size_t rhs_cols,
                                                         size_t out_rows, size_t out_cols)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;
        size_t total_elems = out_rows * out_cols;

        #pragma unroll 4
        for (size_t i = idx; i < total_elems; i += stride)
        {
            size_t r = i / out_cols;
            size_t c = i % out_cols;

            size_t lhs_row_stride = (lhs_rows == 1) ? 0 : lhs_cols;
            size_t lhs_col_stride = (lhs_cols == 1) ? 0 : 1;
            size_t rhs_row_stride = (rhs_rows == 1) ? 0 : rhs_cols;
            size_t rhs_col_stride = (rhs_cols == 1) ? 0 : 1;

            size_t lhs_idx = (r * lhs_row_stride) + (c * lhs_col_stride);
            size_t rhs_idx = (r * rhs_row_stride) + (c * rhs_col_stride);

            lhs[lhs_idx] = __hadd(lhs[lhs_idx], rhs[rhs_idx]);
        }
    }

    template <>
    __global__ void add_inplace_kernel_broadcast<__nv_bfloat16>(__nv_bfloat16 *lhs, const __nv_bfloat16 *rhs,
                                                                size_t lhs_rows, size_t lhs_cols,
                                                                size_t rhs_rows, size_t rhs_cols,
                                                                size_t out_rows, size_t out_cols)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;
        size_t total_elems = out_rows * out_cols;

        for (size_t i = idx; i < total_elems; i += stride)
        {
            size_t r = i / out_cols;
            size_t c = i % out_cols;

            size_t lhs_row_stride = (lhs_rows == 1) ? 0 : lhs_cols;
            size_t lhs_col_stride = (lhs_cols == 1) ? 0 : 1;
            size_t rhs_row_stride = (rhs_rows == 1) ? 0 : rhs_cols;
            size_t rhs_col_stride = (rhs_cols == 1) ? 0 : 1;

            size_t lhs_idx = (r * lhs_row_stride) + (c * lhs_col_stride);
            size_t rhs_idx = (r * rhs_row_stride) + (c * rhs_col_stride);

            lhs[lhs_idx] = __hadd(lhs[lhs_idx], rhs[rhs_idx]);
        }
    }

    void cuda_add_tensor_inplace(Tensor &A, const Tensor &B, cudaStream_t stream)
    {
        bool needs_broadcasting = (A.shape().dims != B.shape().dims);
        size_t total_elems = A.numel();
        size_t block_size = 256;
        
        static int num_sms = get_num_sms();
        size_t num_blocks = num_sms * 4;
        
        size_t max_blocks = (total_elems + block_size - 1) / block_size;
        size_t grid_size = std::min(num_blocks, max_blocks);
        if (grid_size == 0) grid_size = 1;

        dispatch_by_dtype(A.dtype(), [&](auto dummy)
                          {
            using T = decltype(dummy);
            T* a_ptr = A.data<T>();
            const T* b_ptr = B.data<T>();
            
            if (!needs_broadcasting) {
                add_inplace_kernel<<<grid_size, block_size, 0, stream>>>(a_ptr, b_ptr, total_elems);
            } else {
                size_t a_rows = A.shape().dims[0];
                size_t a_cols = A.shape().dims[1];
                size_t b_rows = B.shape().dims[0];
                size_t b_cols = B.shape().dims[1];
                size_t out_rows = A.shape().dims[0];
                size_t out_cols = A.shape().dims[1];
                
                add_inplace_kernel_broadcast<<<grid_size, block_size, 0, stream>>>(
                    a_ptr, b_ptr, a_rows, a_cols, b_rows, b_cols, out_rows, out_cols
                );
            } });
    }

}
#endif