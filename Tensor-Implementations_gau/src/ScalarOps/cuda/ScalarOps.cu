// ScalarOps.cu - CUDA scalar operations
// Clean implementation using dispatch_by_dtype
#if defined(WITH_CUDA)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "core/TensorDispatch.h"
#include <stdexcept>
#include "core/Tensor.h"
#include "ops/ScalarOps.h"
#include "dtype/Types.h"
#include "dtype/DtypeTraits.h"
#include "dtype/DtypeCastUtils.h"

namespace OwnTensor {
namespace { // anonymous namespace for file-local helpers

inline dim3 pick_grid(size_t n, dim3 b) {
    size_t blocks = (n + b.x - 1) / b.x;
    if (blocks > 2147483647ULL) blocks = 2147483647ULL;
    return dim3(static_cast<unsigned int>(blocks));
}

inline void ckerr(const char* where) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(e));
}

// Convert scalar to target type - handles ALL type combinations
// This is needed because dispatch_by_dtype instantiates ALL type combinations at compile time
template<typename T, typename S>
__device__ inline T to_target_type(S scalar) {
    if constexpr (is_complex_type_v<T> && is_complex_type_v<S>) {
        // Both complex: convert real and imag parts
        using RealT = decltype(T().real());
        return T(static_cast<RealT>(scalar.real()), static_cast<RealT>(scalar.imag()));
    } else if constexpr (is_complex_type_v<T> && !is_complex_type_v<S>) {
        // T is complex, S is real: create complex with zero imag
        using RealT = decltype(T().real());
        return T(static_cast<RealT>(scalar), RealT(0));
    } else if constexpr (!is_complex_type_v<T> && is_complex_type_v<S>) {
        // T is real, S is complex: use real part (shouldn't happen at runtime due to promotion)
        return static_cast<T>(scalar.real());
    } else {
        // Both real
        return static_cast<T>(scalar);
    }
}

// ============================================================================
// KERNELS
// ============================================================================

// Arithmetic in-place
template<typename T, typename S>
__global__ void k_add_inplace(T* d, S scalar, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        d[i] = d[i] + to_target_type<T>(scalar);
    }
}
template<typename T, typename S>
__global__ void k_sub_inplace(T* d, S scalar, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        d[i] = d[i] - to_target_type<T>(scalar);
    }
}
template<typename T, typename S>
__global__ void k_mul_inplace(T* d, S scalar, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        d[i] = d[i] * to_target_type<T>(scalar);
    }
}
template<typename T, typename S>
__global__ void k_div_inplace(T* d, S scalar, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        d[i] = d[i] / to_target_type<T>(scalar);
    }
}

// Arithmetic copy
template<typename T, typename S>
__global__ void k_add_copy(const T* a, T* o, S scalar, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        o[i] = a[i] + to_target_type<T>(scalar);
    }
}
template<typename T, typename S>
__global__ void k_sub_copy(const T* a, T* o, S scalar, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        o[i] = a[i] - to_target_type<T>(scalar);
    }
}
template<typename T, typename S>
__global__ void k_mul_copy(const T* a, T* o, S scalar, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        o[i] = a[i] * to_target_type<T>(scalar);
    }
}
template<typename T, typename S>
__global__ void k_div_copy(const T* a, T* o, S scalar, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        o[i] = a[i] / to_target_type<T>(scalar);
    }
}

// Scalar - Tensor
template<typename T, typename S>
__global__ void k_sub_scalar_tensor(const T* a, T* o, S scalar, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        o[i] = to_target_type<T>(scalar) - a[i];
    }
}
template<typename T, typename S>
__global__ void k_div_scalar_tensor(const T* a, T* o, S scalar, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        o[i] = to_target_type<T>(scalar) / a[i];
    }
}

// Modulo operations (using fmod for floats, % for integers)
template<typename T, typename S>
__global__ void k_mod_inplace(T* d, S scalar, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        T val = to_target_type<T>(scalar);
        if constexpr (std::is_integral_v<T>) {
            d[i] = d[i] % val;
        } else if constexpr (std::is_floating_point_v<T>) {
            d[i] = static_cast<T>(fmod(static_cast<double>(d[i]), static_cast<double>(val)));
        }
    }
}
template<typename T, typename S>
__global__ void k_mod_copy(const T* a, T* o, S scalar, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        T val = to_target_type<T>(scalar);
        if constexpr (std::is_integral_v<T>) {
            o[i] = a[i] % val;
        } else if constexpr (std::is_floating_point_v<T>) {
            o[i] = static_cast<T>(fmod(static_cast<double>(a[i]), static_cast<double>(val)));
        }
    }
}
template<typename T, typename S>
__global__ void k_mod_scalar_tensor(const T* a, T* o, S scalar, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        T val = to_target_type<T>(scalar);
        if constexpr (std::is_integral_v<T>) {
            o[i] = val % a[i];
        } else if constexpr (std::is_floating_point_v<T>) {
            o[i] = static_cast<T>(fmod(static_cast<double>(val), static_cast<double>(a[i])));
        }
    }
}

// Comparison
template<typename T, typename S>
__global__ void k_eq(const T* a, uint8_t* o, S scalar, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        o[i] = (a[i] == to_target_type<T>(scalar)) ? 1 : 0;
    }
}
template<typename T, typename S>
__global__ void k_neq(const T* a, uint8_t* o, S scalar, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        o[i] = (a[i] != to_target_type<T>(scalar)) ? 1 : 0;
    }
}
template<typename T, typename S>
__global__ void k_leq(const T* a, uint8_t* o, S scalar, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        if constexpr (!is_complex_type_v<T> && !is_complex_type_v<S>) o[i] = (a[i] <= to_target_type<T>(scalar)) ? 1 : 0;
    }
}
template<typename T, typename S>
__global__ void k_geq(const T* a, uint8_t* o, S scalar, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        if constexpr (!is_complex_type_v<T> && !is_complex_type_v<S>) o[i] = (a[i] >= to_target_type<T>(scalar)) ? 1 : 0;
    }
}
template<typename T, typename S>
__global__ void k_lt(const T* a, uint8_t* o, S scalar, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        if constexpr (!is_complex_type_v<T> && !is_complex_type_v<S>) o[i] = (a[i] < to_target_type<T>(scalar)) ? 1 : 0;
    }
}
template<typename T, typename S>
__global__ void k_gt(const T* a, uint8_t* o, S scalar, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        if constexpr (!is_complex_type_v<T> && !is_complex_type_v<S>) o[i] = (a[i] > to_target_type<T>(scalar)) ? 1 : 0;
    }
}

// Scalar-Tensor comparison (reversed)
template<typename T, typename S>
__global__ void k_s_leq(const T* a, uint8_t* o, S scalar, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        if constexpr (!is_complex_type_v<T> && !is_complex_type_v<S>) o[i] = (to_target_type<T>(scalar) <= a[i]) ? 1 : 0;
    }
}
template<typename T, typename S>
__global__ void k_s_geq(const T* a, uint8_t* o, S scalar, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        if constexpr (!is_complex_type_v<T> && !is_complex_type_v<S>) o[i] = (to_target_type<T>(scalar) >= a[i]) ? 1 : 0;
    }
}
template<typename T, typename S>
__global__ void k_s_lt(const T* a, uint8_t* o, S scalar, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        if constexpr (!is_complex_type_v<T> && !is_complex_type_v<S>) o[i] = (to_target_type<T>(scalar) < a[i]) ? 1 : 0;
    }
}
template<typename T, typename S>
__global__ void k_s_gt(const T* a, uint8_t* o, S scalar, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        if constexpr (!is_complex_type_v<T> && !is_complex_type_v<S>) o[i] = (to_target_type<T>(scalar) > a[i]) ? 1 : 0;
    }
}

} // anon namespace

// ============================================================================
// PUBLIC IMPL FUNCTIONS
// ============================================================================

template<typename S>
void cuda_add_inplace_impl(Tensor& t, S scalar, cudaStream_t stream) {
    const size_t n = t.numel();
    const dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(t.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        k_add_inplace<T, S><<<grid, block, 0, stream>>>(t.data<T>(), scalar, n);
    });
    ckerr("cuda_add_inplace");
}

template<typename S>
void cuda_sub_inplace_impl(Tensor& t, S scalar, cudaStream_t stream) {
    const size_t n = t.numel();
    const dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(t.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        k_sub_inplace<T, S><<<grid, block, 0, stream>>>(t.data<T>(), scalar, n);
    });
    ckerr("cuda_sub_inplace");
}

template<typename S>
void cuda_mul_inplace_impl(Tensor& t, S scalar, cudaStream_t stream) {
    const size_t n = t.numel();
    const dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(t.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        k_mul_inplace<T, S><<<grid, block, 0, stream>>>(t.data<T>(), scalar, n);
    });
    ckerr("cuda_mul_inplace");
}

template<typename S>
void cuda_div_inplace_impl(Tensor& t, S scalar, cudaStream_t stream) {
    const size_t n = t.numel();
    const dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(t.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        k_div_inplace<T, S><<<grid, block, 0, stream>>>(t.data<T>(), scalar, n);
    });
    ckerr("cuda_div_inplace");
}

template<typename S>
Tensor cuda_add_copy_impl(const Tensor& a, S scalar, cudaStream_t stream) {
    Tensor out(a.shape(), a.dtype(), a.device(), a.requires_grad());
    const size_t n = a.numel();
    const dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(a.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        k_add_copy<T, S><<<grid, block, 0, stream>>>(a.data<T>(), out.data<T>(), scalar, n);
    });
    ckerr("cuda_add_copy");
    return out;
}

template<typename S>
Tensor cuda_sub_copy_impl(const Tensor& a, S scalar, cudaStream_t stream) {
    Tensor out(a.shape(), a.dtype(), a.device(), a.requires_grad());
    const size_t n = a.numel();
    const dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(a.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        k_sub_copy<T, S><<<grid, block, 0, stream>>>(a.data<T>(), out.data<T>(), scalar, n);
    });
    ckerr("cuda_sub_copy");
    return out;
}

template<typename S>
Tensor cuda_mul_copy_impl(const Tensor& a, S scalar, cudaStream_t stream) {
    Tensor out(a.shape(), a.dtype(), a.device(), a.requires_grad());
    const size_t n = a.numel();
    const dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(a.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        k_mul_copy<T, S><<<grid, block, 0, stream>>>(a.data<T>(), out.data<T>(), scalar, n);
    });
    ckerr("cuda_mul_copy");
    return out;
}

template<typename S>
Tensor cuda_div_copy_impl(const Tensor& a, S scalar, cudaStream_t stream) {
    const Dtype result_dt = is_complex(a.dtype()) ? a.dtype() : get_promoted_dtype(a.dtype());
    const Tensor& src = (result_dt == a.dtype()) ? a : a.as_type(result_dt);
    Tensor out(src.shape(), result_dt, src.device(), src.requires_grad());
    const size_t n = src.numel();
    const dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(result_dt, [&](auto dummy) {
        using T = decltype(dummy);
        k_div_copy<T, S><<<grid, block, 0, stream>>>(src.data<T>(), out.data<T>(), scalar, n);
    });
    ckerr("cuda_div_copy");
    return out;
}

template<typename S>
Tensor cuda_sub_copy_scalar_tensor_impl(S scalar, const Tensor& a, cudaStream_t stream) {
    Tensor out(a.shape(), a.dtype(), a.device(), a.requires_grad());
    const size_t n = a.numel();
    const dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(a.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        k_sub_scalar_tensor<T, S><<<grid, block, 0, stream>>>(a.data<T>(), out.data<T>(), scalar, n);
    });
    ckerr("cuda_sub_copy_scalar_tensor");
    return out;
}

template<typename S>
Tensor cuda_div_copy_scalar_tensor_impl(S scalar, const Tensor& a, cudaStream_t stream) {
    const Dtype result_dt = is_complex(a.dtype()) ? a.dtype() : get_promoted_dtype(a.dtype());
    const Tensor& src = (result_dt == a.dtype()) ? a : a.as_type(result_dt);
    Tensor out(src.shape(), result_dt, src.device(), src.requires_grad());
    const size_t n = src.numel();
    const dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(result_dt, [&](auto dummy) {
        using T = decltype(dummy);
        k_div_scalar_tensor<T, S><<<grid, block, 0, stream>>>(src.data<T>(), out.data<T>(), scalar, n);
    });
    ckerr("cuda_div_copy_scalar_tensor");
    return out;
}

// Modulo operations
template<typename S>
void cuda_mod_inplace_impl(Tensor& t, S scalar, cudaStream_t stream) {
    const size_t n = t.numel();
    const dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(t.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T>) {
            k_mod_inplace<T, S><<<grid, block, 0, stream>>>(t.data<T>(), scalar, n);
        }
    });
    ckerr("cuda_mod_inplace");
}

template<typename S>
Tensor cuda_mod_copy_impl(const Tensor& a, S scalar, cudaStream_t stream) {
    Tensor out(a.shape(), a.dtype(), a.device(), a.requires_grad());
    const size_t n = a.numel();
    const dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(a.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T>) {
            k_mod_copy<T, S><<<grid, block, 0, stream>>>(a.data<T>(), out.data<T>(), scalar, n);
        }
    });
    ckerr("cuda_mod_copy");
    return out;
}

template<typename S>
Tensor cuda_mod_copy_scalar_tensor_impl(S scalar, const Tensor& a, cudaStream_t stream) {
    Tensor out(a.shape(), a.dtype(), a.device(), a.requires_grad());
    const size_t n = a.numel();
    const dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(a.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T>) {
            k_mod_scalar_tensor<T, S><<<grid, block, 0, stream>>>(a.data<T>(), out.data<T>(), scalar, n);
        }
    });
    ckerr("cuda_mod_copy_scalar_tensor");
    return out;
}

// Comparison operations
template<typename S>
Tensor cuda_eq_copy_impl(const Tensor& a, S scalar, cudaStream_t stream) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), false);
    const size_t n = a.numel();
    const dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(a.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        k_eq<T, S><<<grid, block, 0, stream>>>(a.data<T>(), reinterpret_cast<uint8_t*>(out.data()), scalar, n);
    });
    ckerr("cuda_eq_copy");
    return out;
}

template<typename S>
Tensor cuda_neq_copy_impl(const Tensor& a, S scalar, cudaStream_t stream) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), false);
    const size_t n = a.numel();
    const dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(a.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        k_neq<T, S><<<grid, block, 0, stream>>>(a.data<T>(), reinterpret_cast<uint8_t*>(out.data()), scalar, n);
    });
    ckerr("cuda_neq_copy");
    return out;
}

template<typename S>
Tensor cuda_leq_copy_impl(const Tensor& a, S scalar, cudaStream_t stream) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), false);
    const size_t n = a.numel();
    const dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(a.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        if constexpr (!is_complex_type_v<T>) {
            k_leq<T, S><<<grid, block, 0, stream>>>(a.data<T>(), reinterpret_cast<uint8_t*>(out.data()), scalar, n);
        }
    });
    ckerr("cuda_leq_copy");
    return out;
}

template<typename S>
Tensor cuda_geq_copy_impl(const Tensor& a, S scalar, cudaStream_t stream) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), false);
    const size_t n = a.numel();
    const dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(a.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        if constexpr (!is_complex_type_v<T>) {
            k_geq<T, S><<<grid, block, 0, stream>>>(a.data<T>(), reinterpret_cast<uint8_t*>(out.data()), scalar, n);
        }
    });
    ckerr("cuda_geq_copy");
    return out;
}

template<typename S>
Tensor cuda_lt_copy_impl(const Tensor& a, S scalar, cudaStream_t stream) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), false);
    const size_t n = a.numel();
    const dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(a.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        if constexpr (!is_complex_type_v<T>) {
            k_lt<T, S><<<grid, block, 0, stream>>>(a.data<T>(), reinterpret_cast<uint8_t*>(out.data()), scalar, n);
        }
    });
    ckerr("cuda_lt_copy");
    return out;
}

template<typename S>
Tensor cuda_gt_copy_impl(const Tensor& a, S scalar, cudaStream_t stream) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), false);
    const size_t n = a.numel();
    const dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(a.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        if constexpr (!is_complex_type_v<T>) {
            k_gt<T, S><<<grid, block, 0, stream>>>(a.data<T>(), reinterpret_cast<uint8_t*>(out.data()), scalar, n);
        }
    });
    ckerr("cuda_gt_copy");
    return out;
}

template<typename S>
Tensor cuda_s_leq_copy_impl(S scalar, const Tensor& a, cudaStream_t stream) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), false);
    const size_t n = a.numel();
    const dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(a.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        if constexpr (!is_complex_type_v<T>) {
            k_s_leq<T, S><<<grid, block, 0, stream>>>(a.data<T>(), reinterpret_cast<uint8_t*>(out.data()), scalar, n);
        }
    });
    ckerr("cuda_s_leq_copy");
    return out;
}

template<typename S>
Tensor cuda_s_geq_copy_impl(S scalar, const Tensor& a, cudaStream_t stream) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), false);
    const size_t n = a.numel();
    const dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(a.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        if constexpr (!is_complex_type_v<T>) {
            k_s_geq<T, S><<<grid, block, 0, stream>>>(a.data<T>(), reinterpret_cast<uint8_t*>(out.data()), scalar, n);
        }
    });
    ckerr("cuda_s_geq_copy");
    return out;
}

template<typename S>
Tensor cuda_s_lt_copy_impl(S scalar, const Tensor& a, cudaStream_t stream) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), false);
    const size_t n = a.numel();
    const dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(a.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        if constexpr (!is_complex_type_v<T>) {
            k_s_lt<T, S><<<grid, block, 0, stream>>>(a.data<T>(), reinterpret_cast<uint8_t*>(out.data()), scalar, n);
        }
    });
    ckerr("cuda_s_lt_copy");
    return out;
}

template<typename S>
Tensor cuda_s_gt_copy_impl(S scalar, const Tensor& a, cudaStream_t stream) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), false);
    const size_t n = a.numel();
    const dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(a.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        if constexpr (!is_complex_type_v<T>) {
            k_s_gt<T, S><<<grid, block, 0, stream>>>(a.data<T>(), reinterpret_cast<uint8_t*>(out.data()), scalar, n);
        }
    });
    ckerr("cuda_s_gt_copy");
    return out;
}

// ============================================================================
// EXPLICIT TEMPLATE INSTANTIATIONS
// ============================================================================
#define INSTANTIATE_SCALAR_OPS(S) \
    template void cuda_add_inplace_impl<S>(Tensor&, S, cudaStream_t); \
    template void cuda_sub_inplace_impl<S>(Tensor&, S, cudaStream_t); \
    template void cuda_mul_inplace_impl<S>(Tensor&, S, cudaStream_t); \
    template void cuda_div_inplace_impl<S>(Tensor&, S, cudaStream_t); \
    template Tensor cuda_add_copy_impl<S>(const Tensor&, S, cudaStream_t); \
    template Tensor cuda_sub_copy_impl<S>(const Tensor&, S, cudaStream_t); \
    template Tensor cuda_mul_copy_impl<S>(const Tensor&, S, cudaStream_t); \
    template Tensor cuda_div_copy_impl<S>(const Tensor&, S, cudaStream_t); \
    template Tensor cuda_mod_copy_impl<S>(const Tensor&, S, cudaStream_t); \
    template Tensor cuda_sub_copy_scalar_tensor_impl<S>(S, const Tensor&, cudaStream_t); \
    template Tensor cuda_div_copy_scalar_tensor_impl<S>(S, const Tensor&, cudaStream_t); \
    template Tensor cuda_mod_copy_scalar_tensor_impl<S>(S, const Tensor&, cudaStream_t); \
    template void cuda_mod_inplace_impl<S>(Tensor&, S, cudaStream_t); \
    template Tensor cuda_eq_copy_impl<S>(const Tensor&, S, cudaStream_t); \
    template Tensor cuda_neq_copy_impl<S>(const Tensor&, S, cudaStream_t); \
    template Tensor cuda_leq_copy_impl<S>(const Tensor&, S, cudaStream_t); \
    template Tensor cuda_geq_copy_impl<S>(const Tensor&, S, cudaStream_t); \
    template Tensor cuda_lt_copy_impl<S>(const Tensor&, S, cudaStream_t); \
    template Tensor cuda_gt_copy_impl<S>(const Tensor&, S, cudaStream_t); \
    template Tensor cuda_s_leq_copy_impl<S>(S, const Tensor&, cudaStream_t); \
    template Tensor cuda_s_geq_copy_impl<S>(S, const Tensor&, cudaStream_t); \
    template Tensor cuda_s_lt_copy_impl<S>(S, const Tensor&, cudaStream_t); \
    template Tensor cuda_s_gt_copy_impl<S>(S, const Tensor&, cudaStream_t);\

// Standard C++ integer types
INSTANTIATE_SCALAR_OPS(int8_t)          // signed char
INSTANTIATE_SCALAR_OPS(int16_t)         // short
INSTANTIATE_SCALAR_OPS(int)             // int32_t
INSTANTIATE_SCALAR_OPS(long)            // platform-dependent
INSTANTIATE_SCALAR_OPS(long long)       // int64_t

// Unsigned integer types
INSTANTIATE_SCALAR_OPS(uint8_t)
INSTANTIATE_SCALAR_OPS(uint16_t)
INSTANTIATE_SCALAR_OPS(uint32_t)
INSTANTIATE_SCALAR_OPS(uint64_t)

// Standard floating point types
INSTANTIATE_SCALAR_OPS(float)
INSTANTIATE_SCALAR_OPS(double)

// Boolean
INSTANTIATE_SCALAR_OPS(bool)

// Half-precision and custom float types
INSTANTIATE_SCALAR_OPS(float16_t)
INSTANTIATE_SCALAR_OPS(bfloat16_t)
INSTANTIATE_SCALAR_OPS(float8_e4m3fn_t)
INSTANTIATE_SCALAR_OPS(float8_e5m2_t)

// Complex scalar types
INSTANTIATE_SCALAR_OPS(complex32_t)
INSTANTIATE_SCALAR_OPS(complex64_t)
INSTANTIATE_SCALAR_OPS(complex128_t)

} // namespace OwnTensor
#endif // WITH_CUDA