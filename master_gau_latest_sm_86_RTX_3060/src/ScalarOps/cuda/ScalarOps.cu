#if defined(WITH_CUDA)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "core/TensorDispatch.h"
#include <stdexcept>
#include "core/Tensor.h"
#include "ops/ScalarOps.h"
#include "dtype/Types.h"
#include "dtype/DtypeTraits.h"
#include "device/DeviceCore.h"

namespace OwnTensor {

// ============================================================================
// COMPLEX TRAIT
// ============================================================================
template <typename T> struct is_complex_t : std::false_type {};
template <> struct is_complex_t<complex32_t>  : std::true_type {};
template <> struct is_complex_t<complex64_t>  : std::true_type {};
template <> struct is_complex_t<complex128_t> : std::true_type {};

// Convenience alias: true when T is NOT a complex type
template <typename T>
static constexpr bool is_real_type_v = !is_complex_t<T>::value;

// ============================================================================
// HELPERS
// ============================================================================
static constexpr int UNROLL_FACTOR = 4;

inline __host__ __device__ size_t round_up(size_t n, size_t d) {
    return ((n + d - 1) / d) * d;
}

inline dim3 pick_grid(size_t n, dim3 b) {
    // Each thread handles UNROLL_FACTOR elements, so fewer blocks needed
    size_t blocks = (n + b.x * UNROLL_FACTOR - 1) / (b.x * UNROLL_FACTOR);
    if (blocks < 1) blocks = 1;
    if (blocks > 2147483647ULL) blocks = 2147483647ULL;
    return dim3(static_cast<unsigned int>(blocks));
}

inline Dtype get_division_output_dtype(Dtype input_dtype) {
    if (input_dtype == Dtype::Bool)   return Dtype::Float32;
    if (input_dtype == Dtype::Int16 ||
        input_dtype == Dtype::Int32 ||
        input_dtype == Dtype::Int64)  return Dtype::Float32;
    return input_dtype;
}

// ============================================================================
// KERNELS — In-place (rounded grid-stride + 4× unroll)
// ============================================================================
template<typename T>
__global__ void scalar_add_inplace(T* d, T s, size_t n) {
    const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    const size_t n_round = round_up(n, stride * UNROLL_FACTOR);
    for (size_t base = tid; base < n_round; base += stride * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            const size_t i = base + u * stride;
            if (i < n) d[i] = d[i] + s;
        }
    }
}
template<typename T>
__global__ void scalar_sub_inplace(T* d, T s, size_t n) {
    const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    const size_t n_round = round_up(n, stride * UNROLL_FACTOR);
    for (size_t base = tid; base < n_round; base += stride * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            const size_t i = base + u * stride;
            if (i < n) d[i] = d[i] - s;
        }
    }
}
template<typename T>
__global__ void scalar_mul_inplace(T* d, T s, size_t n) {
    const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    const size_t n_round = round_up(n, stride * UNROLL_FACTOR);
    for (size_t base = tid; base < n_round; base += stride * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            const size_t i = base + u * stride;
            if (i < n) d[i] = d[i] * s;
        }
    }
}
template<typename T>
__global__ void scalar_div_inplace(T* d, T s, size_t n) {
    const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    const size_t n_round = round_up(n, stride * UNROLL_FACTOR);
    for (size_t base = tid; base < n_round; base += stride * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            const size_t i = base + u * stride;
            if (i < n) d[i] = d[i] / s;
        }
    }
}

// ============================================================================
// KERNELS — Copy (SrcT → DstT via scalar S) (rounded grid-stride + 4× unroll)
// ============================================================================
template<typename T, typename S, typename U>
__global__ void scalar_add_copy(const T* a, S s, U* o, size_t n) {
    const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    const size_t n_round = round_up(n, stride * UNROLL_FACTOR);
    for (size_t base = tid; base < n_round; base += stride * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            const size_t i = base + u * stride;
            if (i < n) o[i] = static_cast<U>(a[i]) + static_cast<U>(s);
        }
    }
}
template<typename T, typename S, typename U>
__global__ void scalar_sub_copy(const T* a, S s, U* o, size_t n) {
    const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    const size_t n_round = round_up(n, stride * UNROLL_FACTOR);
    for (size_t base = tid; base < n_round; base += stride * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            const size_t i = base + u * stride;
            if (i < n) o[i] = static_cast<U>(a[i]) - static_cast<U>(s);
        }
    }
}
template<typename T, typename S, typename U>
__global__ void scalar_mul_copy(const T* a, S s, U* o, size_t n) {
    const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    const size_t n_round = round_up(n, stride * UNROLL_FACTOR);
    for (size_t base = tid; base < n_round; base += stride * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            const size_t i = base + u * stride;
            if (i < n) o[i] = static_cast<U>(a[i]) * static_cast<U>(s);
        }
    }
}
template<typename T, typename S, typename U>
__global__ void scalar_div_copy(const T* a, S s, U* o, size_t n) {
    const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    const size_t n_round = round_up(n, stride * UNROLL_FACTOR);
    for (size_t base = tid; base < n_round; base += stride * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            const size_t i = base + u * stride;
            if (i < n) o[i] = static_cast<U>(a[i]) / static_cast<U>(s);
        }
    }
}

template<typename T, typename S, typename U>
__global__ void scalar_sub_s_t(S s, const T* a, U* o, size_t n) {
    const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    const size_t n_round = round_up(n, stride * UNROLL_FACTOR);
    for (size_t base = tid; base < n_round; base += stride * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            const size_t i = base + u * stride;
            if (i < n) o[i] = static_cast<U>(s) - static_cast<U>(a[i]);
        }
    }
}

template<typename T, typename S, typename U>
__global__ void scalar_div_s_t(S s, const T* a, U* o, size_t n) {
    const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    const size_t n_round = round_up(n, stride * UNROLL_FACTOR);
    for (size_t base = tid; base < n_round; base += stride * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            const size_t i = base + u * stride;
            if (i < n) o[i] = static_cast<U>(s) / static_cast<U>(a[i]);
        }
    }
}
// ============================================================================
// KERNEL — Greater-than comparison (T > S → Bool)
// NOTE: operator> is intentionally absent for complex types; the guard below
//       prevents this kernel from ever being instantiated with complex T.
// ============================================================================
template<typename T, typename S>
__global__ void scalar_eq_copy(const T* a, S s, uint8_t* o, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = static_cast<uint8_t>(a[i] == static_cast<T>(s));
}

template<typename T, typename S>
__global__ void scalar_neq_copy(const T* a, S s, uint8_t* o, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = static_cast<uint8_t>(a[i] != static_cast<T>(s));
}

template<typename T, typename S>
__global__ void scalar_gt_t_s(const T* a, S s, uint8_t* o, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = static_cast<uint8_t>(a[i] > static_cast<T>(s));
}

template<typename T, typename S>
__global__ void scalar_lt_t_s(const T* a, S s, uint8_t* o, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = static_cast<uint8_t>(a[i] < static_cast<T>(s));
}

template<typename T, typename S>
__global__ void scalar_geq_t_s(const T* a, S s, uint8_t* o, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = static_cast<uint8_t>(a[i] >= static_cast<T>(s));
}

template<typename T, typename S>
__global__ void scalar_leq_t_s(const T* a, S s, uint8_t* o, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = static_cast<uint8_t>(a[i] <= static_cast<T>(s));
}

template<typename T, typename S>
__global__ void scalar_gt_s_t( S s,const T* a, uint8_t* o, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = static_cast<uint8_t>(static_cast<T>(s) > a[i]);
}

template<typename T, typename S>
__global__ void scalar_lt_s_t( S s,const T* a, uint8_t* o, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = static_cast<uint8_t>( static_cast<T>(s) < a[i]);
}

template<typename T, typename S>
__global__ void scalar_geq_s_t( S s,const T* a, uint8_t* o, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = static_cast<uint8_t>( static_cast<T>(s)  >= a[i]);
}

template<typename T, typename S>
__global__ void scalar_leq_s_t( S s,const T* a, uint8_t* o, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = static_cast<uint8_t>( static_cast<T>(s) <= a[i]);
}
// ============================================================================
// PUBLIC CUDA BACKEND — In-place
// ============================================================================
template<typename S>
void cuda_add_inplace(Tensor& t, S s, cudaStream_t stream) {
    dim3 block(256), grid = pick_grid(t.numel(), block);
    scalar_add_inplace<<<grid, block, 0, stream>>>(t.data<S>(), s, t.numel());
}
template<typename S>
void cuda_sub_inplace(Tensor& t, S s, cudaStream_t stream) {
    dim3 block(256), grid = pick_grid(t.numel(), block);
    scalar_sub_inplace<<<grid, block, 0, stream>>>(t.data<S>(), s, t.numel());
}
template<typename S>
void cuda_mul_inplace(Tensor& t, S s, cudaStream_t stream) {
    dim3 block(256), grid = pick_grid(t.numel(), block);
    scalar_mul_inplace<<<grid, block, 0, stream>>>(t.data<S>(), s, t.numel());
}
template<typename S>
void cuda_div_inplace(Tensor& t, S s, cudaStream_t stream) {
    if (s == static_cast<S>(0)) throw std::runtime_error("Division by zero");
    Dtype dt = t.dtype();
    Dtype promoted_dt = get_division_output_dtype(dt);
    if (promoted_dt != dt)
        throw std::runtime_error("In-place /= requires float dtype. Input is " +
            get_dtype_name(dt) + " but needs " + get_dtype_name(promoted_dt));
    dim3 block(256), grid = pick_grid(t.numel(), block);
    scalar_div_inplace<<<grid, block, 0, stream>>>(t.data<S>(), s, t.numel());
}

template<typename S>
void cuda_add_copy(const Tensor& a, S s, Tensor& b, Dtype promoted_dtype, cudaStream_t stream) {
    dim3 block(256), grid = pick_grid(a.numel(), block);
    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {                      // ← FIX
            dispatch_by_dtype(promoted_dtype, [&](auto d_dst) {
                using DstT = decltype(d_dst);
                if constexpr (is_real_type_v<DstT>) {              // ← FIX
                    scalar_add_copy<<<grid, block, 0, stream>>>(
                        a.data<SrcT>(), s, b.data<DstT>(), a.numel());
                }
            });
        }
    });
}

template<typename S>
void cuda_sub_copy(const Tensor& a, S s, Tensor& b, Dtype promoted_dtype, cudaStream_t stream) {
    dim3 block(256), grid = pick_grid(a.numel(), block);
    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {
            dispatch_by_dtype(promoted_dtype, [&](auto d_dst) {
                using DstT = decltype(d_dst);
                if constexpr (is_real_type_v<DstT>) {
                    scalar_sub_copy<SrcT, S, DstT><<<grid, block, 0, stream>>>(
                        a.data<SrcT>(), s, b.data<DstT>(), a.numel());
                }
            });
        }
    });
}

template<typename S>
void cuda_mul_copy(const Tensor& a, S s, Tensor& b, Dtype promoted_dtype, cudaStream_t stream) {
    dim3 block(256), grid = pick_grid(a.numel(), block);
    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {
            dispatch_by_dtype(promoted_dtype, [&](auto d_dst) {
                using DstT = decltype(d_dst);
                if constexpr (is_real_type_v<DstT>) {
                    scalar_mul_copy<SrcT, S, DstT><<<grid, block, 0, stream>>>(
                        a.data<SrcT>(), s, b.data<DstT>(), a.numel());
                }
            });
        }
    });
}

template<typename S>
void cuda_div_copy(const Tensor& a, S s, Tensor& out, Dtype promoted_dtype, cudaStream_t stream) {
    const Dtype output_dt = get_division_output_dtype(a.dtype());
    dim3 block(256), grid = pick_grid(a.numel(), block);
    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {
            dispatch_by_dtype(output_dt, [&](auto d_dst) {
                using DstT = decltype(d_dst);
                if constexpr (is_real_type_v<DstT>) {
                    scalar_div_copy<SrcT, S, DstT><<<grid, block, 0, stream>>>(
                        a.data<SrcT>(), s, out.data<DstT>(), a.numel());
                }
            });
        }
    });
}

template<typename S>
void cuda_sub_copy_scalar_tensor(S s, const Tensor& a, Tensor& out,
                                      Dtype promoted_dtype, cudaStream_t stream) {
    const Dtype output_dt = out.dtype();
    dim3 block(256), grid = pick_grid(a.numel(), block);
    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {
            dispatch_by_dtype(output_dt, [&](auto d_dst) {
                using DstT = decltype(d_dst);
                if constexpr (is_real_type_v<DstT>) {
                    scalar_sub_s_t<SrcT, S, DstT><<<grid, block, 0, stream>>>(
                        s, a.data<SrcT>(), out.data<DstT>(), a.numel());
                }
            });
        }
    });
}

template<typename S>
void cuda_div_copy_scalar_tensor(S s, const Tensor& a, Tensor& out, Dtype promoted_dtype, cudaStream_t stream) {
    const Dtype output_dt = out.dtype();
    dim3 block(256), grid = pick_grid(a.numel(), block);
    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {
            dispatch_by_dtype(output_dt, [&](auto d_dst) {
                using DstT = decltype(d_dst);
                if constexpr (is_real_type_v<DstT>) {
                    scalar_div_s_t<SrcT, S, DstT><<<grid, block, 0, stream>>>(
                         s,a.data<SrcT>(), out.data<DstT>(), a.numel());
                }
            });
        }
    });
}

// ============================================================================
template <typename S>
void cuda_eq_tensor_scalar(const Tensor& a, S s, Tensor& out,
                                    Dtype promoted_dtype, cudaStream_t stream)
{
    const size_t n = a.numel();                       // FIX 1: was undeclared
    dim3 block(256), grid = pick_grid(n, block);

    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {         // FIX 4: replaces broken guard
            // Output is always Bool (uint8_t) — no need for a second dispatch
            scalar_eq_copy<SrcT, S><<<grid, block, 0, stream>>>(
                a.data<SrcT>(), s,
                out.data<uint8_t>(),                  // FIX 3: was decltype(bool)
                n);
        }
    });
}

template <typename S>
void cuda_neq_tensor_scalar(const Tensor& a, S s, Tensor& out,
                                    Dtype promoted_dtype, cudaStream_t stream)
{
    const size_t n = a.numel();                       // FIX 1: was undeclared
    dim3 block(256), grid = pick_grid(n, block);

    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {     
            scalar_neq_copy<SrcT, S><<<grid, block, 0, stream>>>(
                a.data<SrcT>(), s,
                out.data<uint8_t>(),           
                n);
        }
    });
}

template <typename S>
void cuda_gt_tensor_scalar(const Tensor& a, S s, Tensor& out,
                                    Dtype promoted_dtype, cudaStream_t stream)
{
    const size_t n = a.numel();                       // FIX 1: was undeclared
    dim3 block(256), grid = pick_grid(n, block);

    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {         // FIX 4: replaces broken guard
            // Output is always Bool (uint8_t) — no need for a second dispatch
            scalar_gt_t_s<SrcT, S><<<grid, block, 0, stream>>>(
                a.data<SrcT>(), s,
                out.data<uint8_t>(),                  // FIX 3: was decltype(bool)
                n);
        }
    });
}

template <typename S>
void cuda_lt_tensor_scalar(const Tensor& a, S s, Tensor& out,
                                    Dtype promoted_dtype, cudaStream_t stream)
{
    const size_t n = a.numel();                       // FIX 1: was undeclared
    dim3 block(256), grid = pick_grid(n, block);

    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {         // FIX 4: replaces broken guard
            // Output is always Bool (uint8_t) — no need for a second dispatch
            scalar_lt_t_s<SrcT, S><<<grid, block, 0, stream>>>(
                a.data<SrcT>(), s,
                out.data<uint8_t>(),                  // FIX 3: was decltype(bool)
                n);
        }
    });
}

template <typename S>
void cuda_geq_tensor_scalar(const Tensor& a, S s, Tensor& out,
                                    Dtype promoted_dtype, cudaStream_t stream)
{
    const size_t n = a.numel();                       // FIX 1: was undeclared
    dim3 block(256), grid = pick_grid(n, block);

    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {         // FIX 4: replaces broken guard
            // Output is always Bool (uint8_t) — no need for a second dispatch
            scalar_geq_t_s<SrcT, S><<<grid, block, 0, stream>>>(
                a.data<SrcT>(), s,
                out.data<uint8_t>(),                  // FIX 3: was decltype(bool)
                n);
        }
    });
}

template <typename S>
void cuda_leq_tensor_scalar(const Tensor& a, S s, Tensor& out,
                                    Dtype promoted_dtype, cudaStream_t stream)
{
    const size_t n = a.numel();                       // FIX 1: was undeclared
    dim3 block(256), grid = pick_grid(n, block);

    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {         // FIX 4: replaces broken guard
            // Output is always Bool (uint8_t) — no need for a second dispatch
            scalar_leq_t_s<SrcT, S><<<grid, block, 0, stream>>>(
                a.data<SrcT>(), s,
                out.data<uint8_t>(),                  // FIX 3: was decltype(bool)
                n);
        }
    });
}


template <typename S>
void cuda_gt_scalar_tensor( S s, const Tensor& a,Tensor& out,
                                    Dtype promoted_dtype, cudaStream_t stream)
{
    const size_t n = a.numel();                       // FIX 1: was undeclared
    dim3 block(256), grid = pick_grid(n, block);

    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {         // FIX 4: replaces broken guard
            // Output is always Bool (uint8_t) — no need for a second dispatch
            scalar_gt_s_t<SrcT, S><<<grid, block, 0, stream>>>(
                s, a.data<SrcT>(),
                out.data<uint8_t>(),                  // FIX 3: was decltype(bool)
                n);
        }
    });
}

template <typename S>
void cuda_lt_scalar_tensor( S s, const Tensor& a,Tensor& out,
                                    Dtype promoted_dtype, cudaStream_t stream)
{
    const size_t n = a.numel();                       // FIX 1: was undeclared
    dim3 block(256), grid = pick_grid(n, block);

    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {         // FIX 4: replaces broken guard
            // Output is always Bool (uint8_t) — no need for a second dispatch
            scalar_lt_s_t<SrcT, S><<<grid, block, 0, stream>>>(
                s, a.data<SrcT>(),
                out.data<uint8_t>(),                  // FIX 3: was decltype(bool)
                n);
        }
    });
}

template <typename S>
void cuda_geq_scalar_tensor( S s, const Tensor& a,Tensor& out,
                                    Dtype promoted_dtype, cudaStream_t stream)
{
    const size_t n = a.numel();                       // FIX 1: was undeclared
    dim3 block(256), grid = pick_grid(n, block);

    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {         // FIX 4: replaces broken guard
            // Output is always Bool (uint8_t) — no need for a second dispatch
            scalar_geq_s_t<SrcT, S><<<grid, block, 0, stream>>>(
                s, a.data<SrcT>(),
                out.data<uint8_t>(),                  // FIX 3: was decltype(bool)
                n);
        }
    });
}

template <typename S>
void cuda_leq_scalar_tensor( S s, const Tensor& a,Tensor& out,
                                    Dtype promoted_dtype, cudaStream_t stream)
{
    const size_t n = a.numel();                       // FIX 1: was undeclared
    dim3 block(256), grid = pick_grid(n, block);

    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {         // FIX 4: replaces broken guard
            // Output is always Bool (uint8_t) — no need for a second dispatch
            scalar_leq_s_t<SrcT, S><<<grid, block, 0, stream>>>(
                s, a.data<SrcT>(),
                out.data<uint8_t>(),                  // FIX 3: was decltype(bool)
                n);
        }
    });
}
// ============================================================================
// EXPLICIT INSTANTIATIONS
// ============================================================================

// ---- cuda_add_inplace ----
template void cuda_add_inplace<int8_t>(Tensor&, int8_t, cudaStream_t);
template void cuda_add_inplace<int16_t>(Tensor&, int16_t, cudaStream_t);
template void cuda_add_inplace<int32_t>(Tensor&, int32_t, cudaStream_t);
template void cuda_add_inplace<int64_t>(Tensor&, int64_t, cudaStream_t);
template void cuda_add_inplace<uint8_t>(Tensor&, uint8_t, cudaStream_t);
template void cuda_add_inplace<uint16_t>(Tensor&, uint16_t, cudaStream_t);
template void cuda_add_inplace<uint32_t>(Tensor&, uint32_t, cudaStream_t);
template void cuda_add_inplace<uint64_t>(Tensor&, uint64_t, cudaStream_t);
template void cuda_add_inplace<float>(Tensor&, float, cudaStream_t);
template void cuda_add_inplace<double>(Tensor&, double, cudaStream_t);
template void cuda_add_inplace<float16_t>(Tensor&, float16_t, cudaStream_t);
template void cuda_add_inplace<bfloat16_t>(Tensor&, bfloat16_t, cudaStream_t);
template void cuda_add_inplace<bool>(Tensor&, bool, cudaStream_t);

// ---- cuda_sub_inplace ----
template void cuda_sub_inplace<int8_t>(Tensor&, int8_t, cudaStream_t);
template void cuda_sub_inplace<int16_t>(Tensor&, int16_t, cudaStream_t);
template void cuda_sub_inplace<int32_t>(Tensor&, int32_t, cudaStream_t);
template void cuda_sub_inplace<int64_t>(Tensor&, int64_t, cudaStream_t);
template void cuda_sub_inplace<uint8_t>(Tensor&, uint8_t, cudaStream_t);
template void cuda_sub_inplace<uint16_t>(Tensor&, uint16_t, cudaStream_t);
template void cuda_sub_inplace<uint32_t>(Tensor&, uint32_t, cudaStream_t);
template void cuda_sub_inplace<uint64_t>(Tensor&, uint64_t, cudaStream_t);
template void cuda_sub_inplace<float>(Tensor&, float, cudaStream_t);
template void cuda_sub_inplace<double>(Tensor&, double, cudaStream_t);
template void cuda_sub_inplace<float16_t>(Tensor&, float16_t, cudaStream_t);
template void cuda_sub_inplace<bfloat16_t>(Tensor&, bfloat16_t, cudaStream_t);
template void cuda_sub_inplace<bool>(Tensor&, bool, cudaStream_t);

// ---- cuda_mul_inplace ----
template void cuda_mul_inplace<int8_t>(Tensor&, int8_t, cudaStream_t);
template void cuda_mul_inplace<int16_t>(Tensor&, int16_t, cudaStream_t);
template void cuda_mul_inplace<int32_t>(Tensor&, int32_t, cudaStream_t);
template void cuda_mul_inplace<int64_t>(Tensor&, int64_t, cudaStream_t);
template void cuda_mul_inplace<uint8_t>(Tensor&, uint8_t, cudaStream_t);
template void cuda_mul_inplace<uint16_t>(Tensor&, uint16_t, cudaStream_t);
template void cuda_mul_inplace<uint32_t>(Tensor&, uint32_t, cudaStream_t);
template void cuda_mul_inplace<uint64_t>(Tensor&, uint64_t, cudaStream_t);
template void cuda_mul_inplace<float>(Tensor&, float, cudaStream_t);
template void cuda_mul_inplace<double>(Tensor&, double, cudaStream_t);
template void cuda_mul_inplace<float16_t>(Tensor&, float16_t, cudaStream_t);
template void cuda_mul_inplace<bfloat16_t>(Tensor&, bfloat16_t, cudaStream_t);
template void cuda_mul_inplace<bool>(Tensor&, bool, cudaStream_t);

// ---- cuda_div_inplace ----
template void cuda_div_inplace<int8_t>(Tensor&, int8_t, cudaStream_t);
template void cuda_div_inplace<int16_t>(Tensor&, int16_t, cudaStream_t);
template void cuda_div_inplace<int32_t>(Tensor&, int32_t, cudaStream_t);
template void cuda_div_inplace<int64_t>(Tensor&, int64_t, cudaStream_t);
template void cuda_div_inplace<uint8_t>(Tensor&, uint8_t, cudaStream_t);
template void cuda_div_inplace<uint16_t>(Tensor&, uint16_t, cudaStream_t);
template void cuda_div_inplace<uint32_t>(Tensor&, uint32_t, cudaStream_t);
template void cuda_div_inplace<uint64_t>(Tensor&, uint64_t, cudaStream_t);
template void cuda_div_inplace<float>(Tensor&, float, cudaStream_t);
template void cuda_div_inplace<double>(Tensor&, double, cudaStream_t);
template void cuda_div_inplace<float16_t>(Tensor&, float16_t, cudaStream_t);
template void cuda_div_inplace<bfloat16_t>(Tensor&, bfloat16_t, cudaStream_t);
template void cuda_div_inplace<bool>(Tensor&, bool, cudaStream_t);

// ---- cuda_add_copy ----
template void cuda_add_copy<int8_t>(const Tensor&, int8_t, Tensor&, Dtype, cudaStream_t);
template void cuda_add_copy<int16_t>(const Tensor&, int16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_add_copy<int32_t>(const Tensor&, int32_t, Tensor&, Dtype, cudaStream_t);
template void cuda_add_copy<int64_t>(const Tensor&, int64_t, Tensor&, Dtype, cudaStream_t);
template void cuda_add_copy<uint8_t>(const Tensor&, uint8_t, Tensor&, Dtype, cudaStream_t);
template void cuda_add_copy<uint16_t>(const Tensor&, uint16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_add_copy<uint32_t>(const Tensor&, uint32_t, Tensor&, Dtype, cudaStream_t);
template void cuda_add_copy<uint64_t>(const Tensor&, uint64_t, Tensor&, Dtype, cudaStream_t);
template void cuda_add_copy<float>(const Tensor&, float, Tensor&, Dtype, cudaStream_t);
template void cuda_add_copy<double>(const Tensor&, double, Tensor&, Dtype, cudaStream_t);
template void cuda_add_copy<float16_t>(const Tensor&, float16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_add_copy<bfloat16_t>(const Tensor&, bfloat16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_add_copy<bool>(const Tensor&, bool, Tensor&, Dtype, cudaStream_t);

// ---- cuda_sub_copy ----
template void cuda_sub_copy<int8_t>(const Tensor&, int8_t, Tensor&, Dtype, cudaStream_t);
template void cuda_sub_copy<int16_t>(const Tensor&, int16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_sub_copy<int32_t>(const Tensor&, int32_t, Tensor&, Dtype, cudaStream_t);
template void cuda_sub_copy<int64_t>(const Tensor&, int64_t, Tensor&, Dtype, cudaStream_t);
template void cuda_sub_copy<uint8_t>(const Tensor&, uint8_t, Tensor&, Dtype, cudaStream_t);
template void cuda_sub_copy<uint16_t>(const Tensor&, uint16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_sub_copy<uint32_t>(const Tensor&, uint32_t, Tensor&, Dtype, cudaStream_t);
template void cuda_sub_copy<uint64_t>(const Tensor&, uint64_t, Tensor&, Dtype, cudaStream_t);
template void cuda_sub_copy<float>(const Tensor&, float, Tensor&, Dtype, cudaStream_t);
template void cuda_sub_copy<double>(const Tensor&, double, Tensor&, Dtype, cudaStream_t);
template void cuda_sub_copy<float16_t>(const Tensor&, float16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_sub_copy<bfloat16_t>(const Tensor&, bfloat16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_sub_copy<bool>(const Tensor&, bool, Tensor&, Dtype, cudaStream_t);

// ---- cuda_mul_copy ----
template void cuda_mul_copy<int8_t>(const Tensor&, int8_t, Tensor&, Dtype, cudaStream_t);
template void cuda_mul_copy<int16_t>(const Tensor&, int16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_mul_copy<int32_t>(const Tensor&, int32_t, Tensor&, Dtype, cudaStream_t);
template void cuda_mul_copy<int64_t>(const Tensor&, int64_t, Tensor&, Dtype, cudaStream_t);
template void cuda_mul_copy<uint8_t>(const Tensor&, uint8_t, Tensor&, Dtype, cudaStream_t);
template void cuda_mul_copy<uint16_t>(const Tensor&, uint16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_mul_copy<uint32_t>(const Tensor&, uint32_t, Tensor&, Dtype, cudaStream_t);
template void cuda_mul_copy<uint64_t>(const Tensor&, uint64_t, Tensor&, Dtype, cudaStream_t);
template void cuda_mul_copy<float>(const Tensor&, float, Tensor&, Dtype, cudaStream_t);
template void cuda_mul_copy<double>(const Tensor&, double, Tensor&, Dtype, cudaStream_t);
template void cuda_mul_copy<float16_t>(const Tensor&, float16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_mul_copy<bfloat16_t>(const Tensor&, bfloat16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_mul_copy<bool>(const Tensor&, bool, Tensor&, Dtype, cudaStream_t);

// ---- cuda_div_copy ----
template void cuda_div_copy<int8_t>(const Tensor&, int8_t, Tensor&, Dtype, cudaStream_t);
template void cuda_div_copy<int16_t>(const Tensor&, int16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_div_copy<int32_t>(const Tensor&, int32_t, Tensor&, Dtype, cudaStream_t);
template void cuda_div_copy<int64_t>(const Tensor&, int64_t, Tensor&, Dtype, cudaStream_t);
template void cuda_div_copy<uint8_t>(const Tensor&, uint8_t, Tensor&, Dtype, cudaStream_t);
template void cuda_div_copy<uint16_t>(const Tensor&, uint16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_div_copy<uint32_t>(const Tensor&, uint32_t, Tensor&, Dtype, cudaStream_t);
template void cuda_div_copy<uint64_t>(const Tensor&, uint64_t, Tensor&, Dtype, cudaStream_t);
template void cuda_div_copy<float>(const Tensor&, float, Tensor&, Dtype, cudaStream_t);
template void cuda_div_copy<double>(const Tensor&, double, Tensor&, Dtype, cudaStream_t);
template void cuda_div_copy<float16_t>(const Tensor&, float16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_div_copy<bfloat16_t>(const Tensor&, bfloat16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_div_copy<bool>(const Tensor&, bool, Tensor&, Dtype, cudaStream_t);

// ---- cuda_sub_copy_scalar_tensor ----
template void cuda_sub_copy_scalar_tensor<int8_t>(int8_t, const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_sub_copy_scalar_tensor<int16_t>(int16_t, const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_sub_copy_scalar_tensor<int32_t>(int32_t, const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_sub_copy_scalar_tensor<int64_t>(int64_t, const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_sub_copy_scalar_tensor<uint8_t>(uint8_t, const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_sub_copy_scalar_tensor<uint16_t>(uint16_t, const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_sub_copy_scalar_tensor<uint32_t>(uint32_t, const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_sub_copy_scalar_tensor<uint64_t>(uint64_t, const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_sub_copy_scalar_tensor<float>(float, const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_sub_copy_scalar_tensor<double>(double, const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_sub_copy_scalar_tensor<float16_t>(float16_t, const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_sub_copy_scalar_tensor<bfloat16_t>(bfloat16_t, const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_sub_copy_scalar_tensor<bool>(bool, const Tensor&, Tensor&, Dtype, cudaStream_t);

// ---- cuda_div_copy_scalar_tensor ----
template void cuda_div_copy_scalar_tensor<int8_t>(int8_t, const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_div_copy_scalar_tensor<int16_t>(int16_t, const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_div_copy_scalar_tensor<int32_t>(int32_t, const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_div_copy_scalar_tensor<int64_t>(int64_t, const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_div_copy_scalar_tensor<uint8_t>(uint8_t, const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_div_copy_scalar_tensor<uint16_t>(uint16_t, const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_div_copy_scalar_tensor<uint32_t>(uint32_t, const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_div_copy_scalar_tensor<uint64_t>(uint64_t, const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_div_copy_scalar_tensor<float>(float, const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_div_copy_scalar_tensor<double>(double, const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_div_copy_scalar_tensor<float16_t>(float16_t, const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_div_copy_scalar_tensor<bfloat16_t>(bfloat16_t, const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_div_copy_scalar_tensor<bool>(bool, const Tensor&, Tensor&, Dtype, cudaStream_t);

// ---- cuda_eq_tensor_scalar_outplace ----
template void cuda_eq_tensor_scalar<int8_t>(const Tensor&, int8_t, Tensor&, Dtype, cudaStream_t);
template void cuda_eq_tensor_scalar<int16_t>(const Tensor&, int16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_eq_tensor_scalar<int32_t>(const Tensor&, int32_t, Tensor&, Dtype, cudaStream_t);
template void cuda_eq_tensor_scalar<int64_t>(const Tensor&, int64_t, Tensor&, Dtype, cudaStream_t);
template void cuda_eq_tensor_scalar<uint8_t>(const Tensor&, uint8_t, Tensor&, Dtype, cudaStream_t);
template void cuda_eq_tensor_scalar<uint16_t>(const Tensor&, uint16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_eq_tensor_scalar<uint32_t>(const Tensor&, uint32_t, Tensor&, Dtype, cudaStream_t);
template void cuda_eq_tensor_scalar<uint64_t>(const Tensor&, uint64_t, Tensor&, Dtype, cudaStream_t);
template void cuda_eq_tensor_scalar<float>(const Tensor&, float, Tensor&, Dtype, cudaStream_t);
template void cuda_eq_tensor_scalar<double>(const Tensor&, double, Tensor&, Dtype, cudaStream_t);
template void cuda_eq_tensor_scalar<float16_t>(const Tensor&, float16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_eq_tensor_scalar<bfloat16_t>(const Tensor&, bfloat16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_eq_tensor_scalar<bool>(const Tensor&, bool, Tensor&, Dtype, cudaStream_t);

// ---- cuda_neq_tensor_scalar ----
template void cuda_neq_tensor_scalar<int8_t>(const Tensor&, int8_t, Tensor&, Dtype, cudaStream_t);
template void cuda_neq_tensor_scalar<int16_t>(const Tensor&, int16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_neq_tensor_scalar<int32_t>(const Tensor&, int32_t, Tensor&, Dtype, cudaStream_t);
template void cuda_neq_tensor_scalar<int64_t>(const Tensor&, int64_t, Tensor&, Dtype, cudaStream_t);
template void cuda_neq_tensor_scalar<uint8_t>(const Tensor&, uint8_t, Tensor&, Dtype, cudaStream_t);
template void cuda_neq_tensor_scalar<uint16_t>(const Tensor&, uint16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_neq_tensor_scalar<uint32_t>(const Tensor&, uint32_t, Tensor&, Dtype, cudaStream_t);
template void cuda_neq_tensor_scalar<uint64_t>(const Tensor&, uint64_t, Tensor&, Dtype, cudaStream_t);
template void cuda_neq_tensor_scalar<float>(const Tensor&, float, Tensor&, Dtype, cudaStream_t);
template void cuda_neq_tensor_scalar<double>(const Tensor&, double, Tensor&, Dtype, cudaStream_t);
template void cuda_neq_tensor_scalar<float16_t>(const Tensor&, float16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_neq_tensor_scalar<bfloat16_t>(const Tensor&, bfloat16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_neq_tensor_scalar<bool>(const Tensor&, bool, Tensor&, Dtype, cudaStream_t);

// ---- cuda_gt_tensor_scalar ----
template void cuda_gt_tensor_scalar<int8_t>(const Tensor&, int8_t, Tensor&, Dtype, cudaStream_t);
template void cuda_gt_tensor_scalar<int16_t>(const Tensor&, int16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_gt_tensor_scalar<int32_t>(const Tensor&, int32_t, Tensor&, Dtype, cudaStream_t);
template void cuda_gt_tensor_scalar<int64_t>(const Tensor&, int64_t, Tensor&, Dtype, cudaStream_t);
template void cuda_gt_tensor_scalar<uint8_t>(const Tensor&, uint8_t, Tensor&, Dtype, cudaStream_t);
template void cuda_gt_tensor_scalar<uint16_t>(const Tensor&, uint16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_gt_tensor_scalar<uint32_t>(const Tensor&, uint32_t, Tensor&, Dtype, cudaStream_t);
template void cuda_gt_tensor_scalar<uint64_t>(const Tensor&, uint64_t, Tensor&, Dtype, cudaStream_t);
template void cuda_gt_tensor_scalar<float>(const Tensor&, float, Tensor&, Dtype, cudaStream_t);
template void cuda_gt_tensor_scalar<double>(const Tensor&, double, Tensor&, Dtype, cudaStream_t);
template void cuda_gt_tensor_scalar<float16_t>(const Tensor&, float16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_gt_tensor_scalar<bfloat16_t>(const Tensor&, bfloat16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_gt_tensor_scalar<bool>(const Tensor&, bool, Tensor&, Dtype, cudaStream_t);

// ---- cuda_lt_tensor_scalar ----
template void cuda_lt_tensor_scalar<int8_t>(const Tensor&, int8_t, Tensor&, Dtype, cudaStream_t);
template void cuda_lt_tensor_scalar<int16_t>(const Tensor&, int16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_lt_tensor_scalar<int32_t>(const Tensor&, int32_t, Tensor&, Dtype, cudaStream_t);
template void cuda_lt_tensor_scalar<int64_t>(const Tensor&, int64_t, Tensor&, Dtype, cudaStream_t);
template void cuda_lt_tensor_scalar<uint8_t>(const Tensor&, uint8_t, Tensor&, Dtype, cudaStream_t);
template void cuda_lt_tensor_scalar<uint16_t>(const Tensor&, uint16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_lt_tensor_scalar<uint32_t>(const Tensor&, uint32_t, Tensor&, Dtype, cudaStream_t);
template void cuda_lt_tensor_scalar<uint64_t>(const Tensor&, uint64_t, Tensor&, Dtype, cudaStream_t);
template void cuda_lt_tensor_scalar<float>(const Tensor&, float, Tensor&, Dtype, cudaStream_t);
template void cuda_lt_tensor_scalar<double>(const Tensor&, double, Tensor&, Dtype, cudaStream_t);
template void cuda_lt_tensor_scalar<float16_t>(const Tensor&, float16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_lt_tensor_scalar<bfloat16_t>(const Tensor&, bfloat16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_lt_tensor_scalar<bool>(const Tensor&, bool, Tensor&, Dtype, cudaStream_t);

// ---- cuda_geq_tensor_scalar ----
template void cuda_geq_tensor_scalar<int8_t>(const Tensor&, int8_t, Tensor&, Dtype, cudaStream_t);
template void cuda_geq_tensor_scalar<int16_t>(const Tensor&, int16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_geq_tensor_scalar<int32_t>(const Tensor&, int32_t, Tensor&, Dtype, cudaStream_t);
template void cuda_geq_tensor_scalar<int64_t>(const Tensor&, int64_t, Tensor&, Dtype, cudaStream_t);
template void cuda_geq_tensor_scalar<uint8_t>(const Tensor&, uint8_t, Tensor&, Dtype, cudaStream_t);
template void cuda_geq_tensor_scalar<uint16_t>(const Tensor&, uint16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_geq_tensor_scalar<uint32_t>(const Tensor&, uint32_t, Tensor&, Dtype, cudaStream_t);
template void cuda_geq_tensor_scalar<uint64_t>(const Tensor&, uint64_t, Tensor&, Dtype, cudaStream_t);
template void cuda_geq_tensor_scalar<float>(const Tensor&, float, Tensor&, Dtype, cudaStream_t);
template void cuda_geq_tensor_scalar<double>(const Tensor&, double, Tensor&, Dtype, cudaStream_t);
template void cuda_geq_tensor_scalar<float16_t>(const Tensor&, float16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_geq_tensor_scalar<bfloat16_t>(const Tensor&, bfloat16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_geq_tensor_scalar<bool>(const Tensor&, bool, Tensor&, Dtype, cudaStream_t);

// ---- cuda_leq_tensor_scalar ----
template void cuda_leq_tensor_scalar<int8_t>(const Tensor&, int8_t, Tensor&, Dtype, cudaStream_t);
template void cuda_leq_tensor_scalar<int16_t>(const Tensor&, int16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_leq_tensor_scalar<int32_t>(const Tensor&, int32_t, Tensor&, Dtype, cudaStream_t);
template void cuda_leq_tensor_scalar<int64_t>(const Tensor&, int64_t, Tensor&, Dtype, cudaStream_t);
template void cuda_leq_tensor_scalar<uint8_t>(const Tensor&, uint8_t, Tensor&, Dtype, cudaStream_t);
template void cuda_leq_tensor_scalar<uint16_t>(const Tensor&, uint16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_leq_tensor_scalar<uint32_t>(const Tensor&, uint32_t, Tensor&, Dtype, cudaStream_t);
template void cuda_leq_tensor_scalar<uint64_t>(const Tensor&, uint64_t, Tensor&, Dtype, cudaStream_t);
template void cuda_leq_tensor_scalar<float>(const Tensor&, float, Tensor&, Dtype, cudaStream_t);
template void cuda_leq_tensor_scalar<double>(const Tensor&, double, Tensor&, Dtype, cudaStream_t);
template void cuda_leq_tensor_scalar<float16_t>(const Tensor&, float16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_leq_tensor_scalar<bfloat16_t>(const Tensor&, bfloat16_t, Tensor&, Dtype, cudaStream_t);
template void cuda_leq_tensor_scalar<bool>(const Tensor&, bool, Tensor&, Dtype, cudaStream_t);

// ---- cuda_gt_scalar_tensor ----
template void cuda_gt_scalar_tensor<int8_t>(  int8_t, const Tensor&,Tensor&, Dtype, cudaStream_t);
template void cuda_gt_scalar_tensor<int16_t>( int16_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_gt_scalar_tensor<int32_t>( int32_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_gt_scalar_tensor<int64_t>( int64_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_gt_scalar_tensor<uint8_t>( uint8_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_gt_scalar_tensor<uint16_t>(uint16_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_gt_scalar_tensor<uint32_t>(uint32_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_gt_scalar_tensor<uint64_t>(uint64_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_gt_scalar_tensor<float>(float,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_gt_scalar_tensor<double>(double,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_gt_scalar_tensor<float16_t>(float16_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_gt_scalar_tensor<bfloat16_t>(bfloat16_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_gt_scalar_tensor<bool>(bool,const Tensor&, Tensor&, Dtype, cudaStream_t);

// ---- cuda_lt_scalar_tensor ----
template void cuda_lt_scalar_tensor<int8_t>(  int8_t, const Tensor&,Tensor&, Dtype, cudaStream_t);
template void cuda_lt_scalar_tensor<int16_t>( int16_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_lt_scalar_tensor<int32_t>( int32_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_lt_scalar_tensor<int64_t>( int64_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_lt_scalar_tensor<uint8_t>( uint8_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_lt_scalar_tensor<uint16_t>(uint16_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_lt_scalar_tensor<uint32_t>(uint32_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_lt_scalar_tensor<uint64_t>(uint64_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_lt_scalar_tensor<float>(float,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_lt_scalar_tensor<double>(double,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_lt_scalar_tensor<float16_t>(float16_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_lt_scalar_tensor<bfloat16_t>(bfloat16_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_lt_scalar_tensor<bool>(bool,const Tensor&, Tensor&, Dtype, cudaStream_t);

// ---- cuda_geq_scalar_tensor ----
template void cuda_geq_scalar_tensor<int8_t>(  int8_t, const Tensor&,Tensor&, Dtype, cudaStream_t);
template void cuda_geq_scalar_tensor<int16_t>( int16_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_geq_scalar_tensor<int32_t>( int32_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_geq_scalar_tensor<int64_t>( int64_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_geq_scalar_tensor<uint8_t>( uint8_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_geq_scalar_tensor<uint16_t>( uint16_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_geq_scalar_tensor<uint32_t>( uint32_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_geq_scalar_tensor<uint64_t>( uint64_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_geq_scalar_tensor<float>( float,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_geq_scalar_tensor<double>(double,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_geq_scalar_tensor<float16_t>(float16_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_geq_scalar_tensor<bfloat16_t>(bfloat16_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_geq_scalar_tensor<bool>(bool,const Tensor&, Tensor&, Dtype, cudaStream_t);

// ---- cuda_leq_scalar_tensor ----
template void cuda_leq_scalar_tensor<int8_t>(  int8_t, const Tensor&,Tensor&, Dtype, cudaStream_t);
template void cuda_leq_scalar_tensor<int16_t>( int16_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_leq_scalar_tensor<int32_t>( int32_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_leq_scalar_tensor<int64_t>( int64_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_leq_scalar_tensor<uint8_t>( uint8_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_leq_scalar_tensor<uint16_t>(uint16_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_leq_scalar_tensor<uint32_t>(uint32_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_leq_scalar_tensor<uint64_t>(uint64_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_leq_scalar_tensor<float>(float,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_leq_scalar_tensor<double>(double,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_leq_scalar_tensor<float16_t>(float16_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_leq_scalar_tensor<bfloat16_t>(bfloat16_t,const Tensor&, Tensor&, Dtype, cudaStream_t);
template void cuda_leq_scalar_tensor<bool>(bool,const Tensor&, Tensor&, Dtype, cudaStream_t);


} // namespace OwnTensor
#endif // WITH_CUDA

