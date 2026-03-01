#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "ops/TensorOps.cuh"
#include "core/Tensor.h"
#include "core/TensorDispatch.h"
#include "ops/helpers/BroadcastUtils.h"

#include <stdio.h>

namespace OwnTensor {

    // ========================================================================
    // Device Functions for Modulo
    // ========================================================================

    template<typename T>
    __device__ __forceinline__ T mod_op(T a, T b) {
        return a % b;
    }

    template<> __device__ __forceinline__ float mod_op(float a, float b) {
        return fmodf(a, b);
    }

    template<> __device__ __forceinline__ double mod_op(double a, double b) {
        return fmod(a, b);
    }

    template<> __device__ __forceinline__ __half mod_op(__half a, __half b) {
        return __float2half(fmodf(__half2float(a), __half2float(b)));
    }

    template<> __device__ __forceinline__ __nv_bfloat16 mod_op(__nv_bfloat16 a, __nv_bfloat16 b) {
        return __float2bfloat16(fmodf(__bfloat162float(a), __bfloat162float(b)));
    }

    template<> __device__ __forceinline__ __nv_fp8_e4m3 mod_op(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b) {
        __nv_fp8_storage_t a_raw = reinterpret_cast<__nv_fp8_storage_t&>(a);
        __nv_fp8_storage_t b_raw = reinterpret_cast<__nv_fp8_storage_t&>(b);
        float a_f = __half2float(__nv_cvt_fp8_to_halfraw(a_raw, __NV_E4M3));
        float b_f = __half2float(__nv_cvt_fp8_to_halfraw(b_raw, __NV_E4M3));
        __nv_fp8_storage_t res = __nv_cvt_float_to_fp8(fmodf(a_f, b_f), __NV_SATFINITE, __NV_E4M3);
        return *reinterpret_cast<__nv_fp8_e4m3*>(&res);
    }

    template<> __device__ __forceinline__ __nv_fp8_e5m2 mod_op(__nv_fp8_e5m2 a, __nv_fp8_e5m2 b) {
        __nv_fp8_storage_t a_raw = reinterpret_cast<__nv_fp8_storage_t&>(a);
        __nv_fp8_storage_t b_raw = reinterpret_cast<__nv_fp8_storage_t&>(b);
        float a_f = __half2float(__nv_cvt_fp8_to_halfraw(a_raw, __NV_E5M2));
        float b_f = __half2float(__nv_cvt_fp8_to_halfraw(b_raw, __NV_E5M2));
        __nv_fp8_storage_t res = __nv_cvt_float_to_fp8(fmodf(a_f, b_f), __NV_SATFINITE, __NV_E5M2);
        return *reinterpret_cast<__nv_fp8_e5m2*>(&res);
    }

    // ========================================================================
    // Kernels
    // ========================================================================

    template <typename T>
    __global__ void mod_kernel(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ output, size_t numel) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < numel) {
            output[idx] = mod_op(A[idx], B[idx]);
        }
    }

    template <typename T>
    __global__ void mod_kernel_broadcast(
        const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ output, size_t numel,
        int ndim,
        const size_t* __restrict__ a_strides, const size_t* __restrict__ b_strides,
        const size_t* __restrict__ out_shape
    ) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < numel) {
            size_t temp_idx = idx;
            size_t a_idx = 0;
            size_t b_idx = 0;

            for (int dim = ndim - 1; dim >= 0; --dim) {
                size_t coord = temp_idx % out_shape[dim];
                temp_idx /= out_shape[dim];
                a_idx += coord * a_strides[dim];
                b_idx += coord * b_strides[dim];
            }

            output[idx] = mod_op(A[a_idx], B[b_idx]);
        }
    }

    template <typename T>
    __global__ void mod_inplace_kernel(T* __restrict__ A, const T* __restrict__ B, size_t numel) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < numel) {
            A[idx] = mod_op(A[idx], B[idx]);
        }
    }

    template <typename T>
    __global__ void mod_inplace_kernel_broadcast(
        T* __restrict__ A, const T* __restrict__ B, size_t numel,
        int ndim,
        const size_t* __restrict__ a_strides, const size_t* __restrict__ b_strides,
        const size_t* __restrict__ out_shape
    ) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < numel) {
            size_t temp_idx = idx;
            size_t a_idx = 0;
            size_t b_idx = 0;

            for (int dim = ndim - 1; dim >= 0; --dim) {
                size_t coord = temp_idx % out_shape[dim];
                temp_idx /= out_shape[dim];
                a_idx += coord * a_strides[dim];
                b_idx += coord * b_strides[dim];
            }

            A[a_idx] = mod_op(A[a_idx], B[b_idx]);
        }
    }

    void cuda_mod_tensor(const Tensor& A, const Tensor& B, Tensor& output, cudaStream_t stream) {
        bool needs_broadcasting = (A.shape().dims != B.shape().dims);
        size_t total_elems = output.numel();
        size_t block_size = 256;
        size_t grid_size = (total_elems + block_size - 1) / block_size;
        
        dispatch_by_dtype(A.dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            
            if constexpr (std::is_same_v<T, complex32_t> || std::is_same_v<T, complex64_t> || std::is_same_v<T, complex128_t>) {
                throw std::runtime_error("Modulo not implemented for complex types");
            } else {
                if (!needs_broadcasting) {
                    mod_kernel<T><<<grid_size, block_size, 0, stream>>>(
                        A.data<T>(), B.data<T>(), output.data<T>(), total_elems
                    );
                } else {
                    // Handle broadcasting
                    size_t out_ndim = output.shape().dims.size();
                    std::vector<size_t> a_bcast_strides(out_ndim);
                    std::vector<size_t> b_bcast_strides(out_ndim);
                    
                    const auto& a_shape = A.shape().dims;
                    const auto& b_shape = B.shape().dims;
                    const auto& out_shape = output.shape().dims;
                    const auto& a_strides_vec = A.stride().strides;
                    const auto& b_strides_vec = B.stride().strides;

                    size_t a_ndim = a_shape.size();
                    size_t b_ndim = b_shape.size();

                    for (size_t i = 0; i < out_ndim; ++i) {
                        size_t a_dim_idx = a_ndim - out_ndim + i;
                        if (a_dim_idx < a_ndim && a_shape[a_dim_idx] > 1) {
                            a_bcast_strides[i] = a_strides_vec[a_dim_idx];
                        } else {
                            a_bcast_strides[i] = 0;
                        }

                        size_t b_dim_idx = b_ndim - out_ndim + i;
                        if (b_dim_idx < b_ndim && b_shape[b_dim_idx] > 1) {
                            b_bcast_strides[i] = b_strides_vec[b_dim_idx];
                        } else {
                            b_bcast_strides[i] = 0;
                        }
                    }

                    // Allocate device memory for strides/shape
                    size_t* d_a_strides;
                    size_t* d_b_strides;
                    size_t* d_out_shape;
                    
                    cudaMallocAsync(&d_a_strides, out_ndim * sizeof(size_t), stream);
                    cudaMallocAsync(&d_b_strides, out_ndim * sizeof(size_t), stream);
                    cudaMallocAsync(&d_out_shape, out_ndim * sizeof(size_t), stream);

                    cudaMemcpyAsync(d_a_strides, a_bcast_strides.data(), out_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);
                    cudaMemcpyAsync(d_b_strides, b_bcast_strides.data(), out_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);
                    cudaMemcpyAsync(d_out_shape, out_shape.data(), out_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);

                    mod_kernel_broadcast<T><<<grid_size, block_size, 0, stream>>>(
                        A.data<T>(), B.data<T>(), output.data<T>(), total_elems,
                        out_ndim, d_a_strides, d_b_strides, d_out_shape
                    );

                    cudaFreeAsync(d_a_strides, stream);
                    cudaFreeAsync(d_b_strides, stream);
                    cudaFreeAsync(d_out_shape, stream);
                }
            }
        });
    }

    void cuda_mod_tensor_inplace(Tensor& A, const Tensor& B, cudaStream_t stream) {
        bool needs_broadcasting = (A.shape().dims != B.shape().dims);
        size_t total_elems = A.numel();
        size_t block_size = 256;
        size_t grid_size = (total_elems + block_size - 1) / block_size;

        dispatch_by_dtype(A.dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            
            if constexpr (std::is_same_v<T, complex32_t> || std::is_same_v<T, complex64_t> || std::is_same_v<T, complex128_t>) {
                throw std::runtime_error("Modulo not implemented for complex types");
            } else {
                if (!needs_broadcasting) {
                    mod_inplace_kernel<T><<<grid_size, block_size, 0, stream>>>(
                        A.data<T>(), B.data<T>(), total_elems
                    );
                } else {
                    // Handle broadcasting
                    size_t out_ndim = A.shape().dims.size();
                    std::vector<size_t> a_bcast_strides(out_ndim);
                    std::vector<size_t> b_bcast_strides(out_ndim);
                    
                    const auto& a_shape = A.shape().dims;
                    const auto& b_shape = B.shape().dims;
                    const auto& a_strides_vec = A.stride().strides;
                    const auto& b_strides_vec = B.stride().strides;

                    size_t a_ndim = a_shape.size();
                    size_t b_ndim = b_shape.size();

                    for (size_t i = 0; i < out_ndim; ++i) {
                        size_t a_dim_idx = a_ndim - out_ndim + i;
                        if (a_dim_idx < a_ndim && a_shape[a_dim_idx] > 1) {
                            a_bcast_strides[i] = a_strides_vec[a_dim_idx];
                        } else {
                            a_bcast_strides[i] = 0;
                        }

                        size_t b_dim_idx = b_ndim - out_ndim + i;
                        if (b_dim_idx < b_ndim && b_shape[b_dim_idx] > 1) {
                            b_bcast_strides[i] = b_strides_vec[b_dim_idx];
                        } else {
                            b_bcast_strides[i] = 0;
                        }
                    }

                    size_t* d_a_strides;
                    size_t* d_b_strides;
                    size_t* d_out_shape;
                    
                    cudaMallocAsync(&d_a_strides, out_ndim * sizeof(size_t), stream);
                    cudaMallocAsync(&d_b_strides, out_ndim * sizeof(size_t), stream);
                    cudaMallocAsync(&d_out_shape, out_ndim * sizeof(size_t), stream);

                    cudaMemcpyAsync(d_a_strides, a_bcast_strides.data(), out_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);
                    cudaMemcpyAsync(d_b_strides, b_bcast_strides.data(), out_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);
                    cudaMemcpyAsync(d_out_shape, a_shape.data(), out_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);

                    mod_inplace_kernel_broadcast<T><<<grid_size, block_size, 0, stream>>>(
                        A.data<T>(), B.data<T>(), total_elems,
                        out_ndim, d_a_strides, d_b_strides, d_out_shape
                    );

                    cudaFreeAsync(d_a_strides, stream);
                    cudaFreeAsync(d_b_strides, stream);
                    cudaFreeAsync(d_out_shape, stream);
                }
            }
        });
    }

} // namespace OwnTensor

#endif