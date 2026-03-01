#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "ops/TensorOps.cuh"
#include "core/Tensor.h"
#include "core/TensorDispatch.h"

namespace OwnTensor
{   

// ============================================================================
// OUT-OF-PLACE KERNELS
// ============================================================================

template<typename T>
__global__ void bool_not_kernel(const T* a, bool* output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        bool a_bool = (a[idx] != T(0.0f));
        output[idx] = !a_bool;
    }
}

template<>
__global__ void bool_not_kernel<__half>(const __half* a, bool* output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        __half zero = __float2half(0.0f);
        bool a_bool = !__heq(a[idx], zero);
        output[idx] = !a_bool;
    }
}

template<>
__global__ void bool_not_kernel<__nv_bfloat16>(const __nv_bfloat16* a, bool* output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        __nv_bfloat16 zero = __float2bfloat16(0.0f);
        bool a_bool = !__heq(a[idx], zero);
        output[idx] = !a_bool;
    }
}

void cuda_logical_not_outplace(const Tensor &A, Tensor &output, cudaStream_t stream)
{
    size_t total_elems = output.numel();
    size_t block_size = 256;
    size_t grid_size = (total_elems + block_size - 1) / block_size;

    dispatch_by_dtype(A.dtype(), [&](auto dummy)
    {
        using T = decltype(dummy);
        const T* a_ptr = A.data<T>();
        bool* output_ptr = output.data<bool>();
        
        bool_not_kernel<<<grid_size, block_size, 0, stream>>>(a_ptr, output_ptr, total_elems);
    });
}

// ============================================================================
// IN-PLACE KERNELS
// ============================================================================

template<typename T>
__global__ void bool_not_inplace_kernel(T* a, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        bool a_bool = (a[idx] != T(0.0f));
        // Write back as T (0.0f or 1.0f)
        a[idx] = static_cast<T>(!a_bool);
    }
}

template<>
__global__ void bool_not_inplace_kernel<__half>(__half* a, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        __half zero = __float2half(0.0f);
        __half one = __float2half(1.0f);
        bool a_bool = !__heq(a[idx], zero);
        a[idx] = (!a_bool) ? one : zero;
    }
}

template<>
__global__ void bool_not_inplace_kernel<__nv_bfloat16>(__nv_bfloat16* a, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        __nv_bfloat16 zero = __float2bfloat16(0.0f);
        __nv_bfloat16 one = __float2bfloat16(1.0f);
        bool a_bool = !__heq(a[idx], zero);
        a[idx] = (!a_bool) ? one : zero;
    }
}

void cuda_logical_not_inplace(Tensor &A, cudaStream_t stream)
{
    size_t total_elems = A.numel();
    size_t block_size = 256;
    size_t grid_size = (total_elems + block_size - 1) / block_size;

    dispatch_by_dtype(A.dtype(), [&](auto dummy)
    {
        using T = decltype(dummy);
        T* a_ptr = A.data<T>();
        
        bool_not_inplace_kernel<<<grid_size, block_size, 0, stream>>>(a_ptr, total_elems);
    });
}

}
#endif // WITH_CUDA