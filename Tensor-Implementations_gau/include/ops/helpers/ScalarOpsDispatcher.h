// ScalarOpsDispatcher.h - Template implementations for scalar operators
// Handles scalars directly with templates for both CPU and CUDA
#pragma once
#include <cstdint>
#include <stdexcept>
#include "ops/ScalarOps.h"
#include "dtype/Types.h"
#include "dtype/DtypeTraits.h"
#include "dtype/DtypeCastUtils.h"
#include "core/TensorDispatch.h"
#include <driver_types.h>
#include "device/DeviceCore.h"

namespace OwnTensor {

// ═══════════════════════════════════════════════════════════════════════════
// SCALAR CONVERSION HELPER
// ═══════════════════════════════════════════════════════════════════════════
// This helper is needed because dispatch_by_dtype instantiates ALL type combinations
// at compile time, even if they won't be called at runtime. We use if constexpr
// to make all type combinations valid at compile time.

template<typename T, typename S>
inline T convert_scalar(S s) {
    if constexpr (is_complex_type_v<T> && is_complex_type_v<S>) {
        // Both are complex - convert real and imag parts
        using RealT = decltype(T().real());
        return T(static_cast<RealT>(s.real()), static_cast<RealT>(s.imag()));
    } else if constexpr (is_complex_type_v<T> && !is_complex_type_v<S>) {
        // T is complex, S is real - create complex with zero imaginary
        using RealT = decltype(T().real());
        return T(static_cast<RealT>(s), RealT(0));
    } else if constexpr (!is_complex_type_v<T> && is_complex_type_v<S>) {
        // T is real, S is complex - this shouldn't happen at runtime due to promotion
        // but we need valid code for compile time. Use real part.
        return static_cast<T>(s.real());
    } else {
        // Both are real types
        return static_cast<T>(s);
    }
}

// CUDA backend declarations
template<typename S> void   cuda_add_inplace_impl(Tensor&, S, cudaStream_t);
template<typename S> void   cuda_sub_inplace_impl(Tensor&, S, cudaStream_t);
template<typename S> void   cuda_mul_inplace_impl(Tensor&, S, cudaStream_t);
template<typename S> void   cuda_div_inplace_impl(Tensor&, S, cudaStream_t);

template<typename S> Tensor cuda_add_copy_impl(const Tensor&, S, cudaStream_t);
template<typename S> Tensor cuda_sub_copy_impl(const Tensor&, S, cudaStream_t);
template<typename S> Tensor cuda_mul_copy_impl(const Tensor&, S, cudaStream_t);
template<typename S> Tensor cuda_div_copy_impl(const Tensor&, S, cudaStream_t);
template<typename S> Tensor cuda_mod_copy_impl(const Tensor&, S, cudaStream_t);
template<typename S> Tensor cuda_sub_copy_scalar_tensor_impl(S, const Tensor&, cudaStream_t);
template<typename S> Tensor cuda_div_copy_scalar_tensor_impl(S, const Tensor&, cudaStream_t);
template<typename S> Tensor cuda_mod_copy_scalar_tensor_impl(S, const Tensor&, cudaStream_t);
template<typename S> void   cuda_mod_inplace_impl(Tensor&, S, cudaStream_t);

template<typename S> Tensor cuda_eq_copy_impl(const Tensor&, S, cudaStream_t);
template<typename S> Tensor cuda_neq_copy_impl(const Tensor&, S, cudaStream_t);
template<typename S> Tensor cuda_leq_copy_impl(const Tensor&, S, cudaStream_t);
template<typename S> Tensor cuda_geq_copy_impl(const Tensor&, S, cudaStream_t);
template<typename S> Tensor cuda_lt_copy_impl(const Tensor&, S, cudaStream_t);
template<typename S> Tensor cuda_gt_copy_impl(const Tensor&, S, cudaStream_t);
template<typename S> Tensor cuda_s_leq_copy_impl(S, const Tensor&, cudaStream_t);
template<typename S> Tensor cuda_s_geq_copy_impl(S, const Tensor&, cudaStream_t);
template<typename S> Tensor cuda_s_lt_copy_impl(S, const Tensor&, cudaStream_t);
template<typename S> Tensor cuda_s_gt_copy_impl(S, const Tensor&, cudaStream_t);

// In-Place Operations
template<typename S>
Tensor& operator+=(Tensor& t, S s) {
    constexpr Dtype scalar_dt = type_to_dtype<S>();
    const Dtype tensor_dt = t.dtype();
    const Dtype promoted_dt = promote_scalar_ops(tensor_dt, scalar_dt);
    if (promoted_dt != tensor_dt) {
        throw std::runtime_error("In-place +=: type mismatch. Use outplace +.");
    }
    if (t.device().is_cuda()) {
#ifdef WITH_CUDA
        cuda_add_inplace_impl<S>(t, s, OwnTensor::cuda_1926::getCurrentStream());
#else
        throw std::runtime_error("CUDA not supported");
#endif
    } else {
        dispatch_by_dtype(tensor_dt, [&](auto dummy) {
            using T = decltype(dummy);
            T* data = t.data<T>();
            size_t n = t.numel();
        T val = convert_scalar<T>(s);
            for (size_t i = 0; i < n; ++i) data[i] = data[i] + val;
        });
    }
    return t;
}

template<typename S>
Tensor& operator-=(Tensor& t, S s) {
    constexpr Dtype scalar_dt = type_to_dtype<S>();
    const Dtype tensor_dt = t.dtype();
    const Dtype promoted_dt = promote_scalar_ops(tensor_dt, scalar_dt);
    if (promoted_dt != tensor_dt) {
        throw std::runtime_error("In-place -=: type mismatch. Use outplace -.");
    }
    if (t.device().is_cuda()) {
#ifdef WITH_CUDA
        cuda_sub_inplace_impl<S>(t, s, OwnTensor::cuda_1926::getCurrentStream());
#else
        throw std::runtime_error("CUDA not supported");
#endif
    } else {
        dispatch_by_dtype(tensor_dt, [&](auto dummy) {
            using T = decltype(dummy);
            T* data = t.data<T>();
            size_t n = t.numel();
        T val = convert_scalar<T>(s);
            for (size_t i = 0; i < n; ++i) data[i] = data[i] - val;
        });
    }
    return t;
}

template<typename S>
Tensor& operator*=(Tensor& t, S s) {
    constexpr Dtype scalar_dt = type_to_dtype<S>();
    const Dtype tensor_dt = t.dtype();
    const Dtype promoted_dt = promote_scalar_ops(tensor_dt, scalar_dt);
    if (promoted_dt != tensor_dt) {
        throw std::runtime_error("In-place *=: type mismatch. Use outplace *.");
    }
    if (t.device().is_cuda()) {
#ifdef WITH_CUDA
        cuda_mul_inplace_impl<S>(t, s, OwnTensor::cuda_1926::getCurrentStream());
#else
        throw std::runtime_error("CUDA not supported");
#endif
    } else {
        dispatch_by_dtype(tensor_dt, [&](auto dummy) {
            using T = decltype(dummy);
            T* data = t.data<T>();
            size_t n = t.numel();
        T val = convert_scalar<T>(s);
            for (size_t i = 0; i < n; ++i) data[i] = data[i] * val;
        });
    }
    return t;
}

template<typename S>
Tensor& operator/=(Tensor& t, S s) {
    constexpr Dtype scalar_dt = type_to_dtype<S>();
    const Dtype tensor_dt = t.dtype();
    const Dtype promoted_dt = promote_dtypes_division(tensor_dt, scalar_dt);
    if (promoted_dt != tensor_dt) {
        throw std::runtime_error("In-place /=: type mismatch. Use outplace /.");
    }
    if (t.device().is_cuda()) {
#ifdef WITH_CUDA
        cuda_div_inplace_impl<S>(t, s, OwnTensor::cuda_1926::getCurrentStream());
#else
        throw std::runtime_error("CUDA not supported");
#endif
    } else {
        dispatch_by_dtype(tensor_dt, [&](auto dummy) {
            using T = decltype(dummy);
            T* data = t.data<T>();
            size_t n = t.numel();
        T val = convert_scalar<T>(s);
            for (size_t i = 0; i < n; ++i) data[i] = data[i] / val;
        });
    }
    return t;
}

// Modulo in-place (works for integers and floats via fmod)
template<typename S>
Tensor& operator%=(Tensor& t, S s) {
    constexpr Dtype scalar_dt = type_to_dtype<S>();
    const Dtype tensor_dt = t.dtype();
    const Dtype promoted_dt = promote_scalar_ops(tensor_dt, scalar_dt);
    if (promoted_dt != tensor_dt) {
        throw std::runtime_error("In-place %=: type mismatch. Use outplace %.");
    }
    if (t.device().is_cuda()) {
#ifdef WITH_CUDA
        cuda_mod_inplace_impl<S>(t, s, OwnTensor::cuda_1926::getCurrentStream());
#else
        throw std::runtime_error("CUDA not supported");
#endif
    } else {
        dispatch_by_dtype(tensor_dt, [&](auto dummy) {
            using T = decltype(dummy);
            T* data = t.data<T>();
            size_t n = t.numel();
            T val = convert_scalar<T>(s);
            for (size_t i = 0; i < n; ++i) {
                if constexpr (std::is_integral_v<T>) {
                    if (val == T(0)) throw std::runtime_error("Modulo by zero");
                    data[i] = data[i] % val;
                } else if constexpr (std::is_floating_point_v<T>) {
                    data[i] = static_cast<T>(std::fmod(static_cast<double>(data[i]), static_cast<double>(val)));
                } else {
                    throw std::runtime_error("Modulo not supported for this dtype");
                }
            }
        });
    }
    return t;
}

// Copy Operations (Tensor op Scalar)
template<typename S>
Tensor operator+(const Tensor& a, S s) {
    constexpr Dtype scalar_dt = type_to_dtype<S>();
    const Dtype tensor_dt = a.dtype();
    const Dtype promoted_dt = promote_scalar_ops(tensor_dt, scalar_dt);
    const Tensor& src = (promoted_dt == tensor_dt) ? a : a.as_type(promoted_dt);
    if (src.device().is_cuda()) {
#ifdef WITH_CUDA
        return cuda_add_copy_impl<S>(src, s, OwnTensor::cuda_1926::getCurrentStream());
#else
        throw std::runtime_error("CUDA not supported");
#endif
    }
    Tensor result(src.shape(), promoted_dt, src.device(), src.requires_grad());
    dispatch_by_dtype(promoted_dt, [&](auto dummy) {
        using T = decltype(dummy);
        const T* in = src.data<T>();
        T* out = result.data<T>();
        size_t n = src.numel();
        T val = convert_scalar<T>(s);
        for (size_t i = 0; i < n; ++i) out[i] = in[i] + val;
    });
    return result;
}

template<typename S>
Tensor operator-(const Tensor& a, S s) {
    constexpr Dtype scalar_dt = type_to_dtype<S>();
    const Dtype tensor_dt = a.dtype();
    const Dtype promoted_dt = promote_scalar_ops(tensor_dt, scalar_dt);
    const Tensor& src = (promoted_dt == tensor_dt) ? a : a.as_type(promoted_dt);
    if (src.device().is_cuda()) {
#ifdef WITH_CUDA
        return cuda_sub_copy_impl<S>(src, s, OwnTensor::cuda_1926::getCurrentStream());
#else
        throw std::runtime_error("CUDA not supported");
#endif
    }
    Tensor result(src.shape(), promoted_dt, src.device(), src.requires_grad());
    dispatch_by_dtype(promoted_dt, [&](auto dummy) {
        using T = decltype(dummy);
        const T* in = src.data<T>();
        T* out = result.data<T>();
        size_t n = src.numel();
        T val = convert_scalar<T>(s);
        for (size_t i = 0; i < n; ++i) out[i] = in[i] - val;
    });
    return result;
}

template<typename S>
Tensor operator*(const Tensor& a, S s) {
    constexpr Dtype scalar_dt = type_to_dtype<S>();
    const Dtype tensor_dt = a.dtype();
    const Dtype promoted_dt = promote_scalar_ops(tensor_dt, scalar_dt);
    const Tensor& src = (promoted_dt == tensor_dt) ? a : a.as_type(promoted_dt);
    if (src.device().is_cuda()) {
#ifdef WITH_CUDA
        return cuda_mul_copy_impl<S>(src, s, OwnTensor::cuda_1926::getCurrentStream());
#else
        throw std::runtime_error("CUDA not supported");
#endif
    }
    Tensor result(src.shape(), promoted_dt, src.device(), src.requires_grad());
    dispatch_by_dtype(promoted_dt, [&](auto dummy) {
        using T = decltype(dummy);
        const T* in = src.data<T>();
        T* out = result.data<T>();
        size_t n = src.numel();
        T val = convert_scalar<T>(s);
        for (size_t i = 0; i < n; ++i) out[i] = in[i] * val;
    });
    return result;
}

template<typename S>
Tensor operator/(const Tensor& a, S s) {
    constexpr Dtype scalar_dt = type_to_dtype<S>();
    const Dtype tensor_dt = a.dtype();
    const Dtype promoted_dt = promote_dtypes_division(tensor_dt, scalar_dt);
    const Tensor& src = (promoted_dt == tensor_dt) ? a : a.as_type(promoted_dt);
    if (src.device().is_cuda()) {
#ifdef WITH_CUDA
        return cuda_div_copy_impl<S>(src, s, OwnTensor::cuda_1926::getCurrentStream());
#else
        throw std::runtime_error("CUDA not supported");
#endif
    }
    Tensor result(src.shape(), promoted_dt, src.device(), src.requires_grad());
    dispatch_by_dtype(promoted_dt, [&](auto dummy) {
        using T = decltype(dummy);
        const T* in = src.data<T>();
        T* out = result.data<T>();
        size_t n = src.numel();
        T val = convert_scalar<T>(s);
        for (size_t i = 0; i < n; ++i) out[i] = in[i] / val;
    });
    return result;
}

// Scalar op Tensor
template<typename S>
Tensor operator-(S s, const Tensor& a) {
    constexpr Dtype scalar_dt = type_to_dtype<S>();
    const Dtype tensor_dt = a.dtype();
    const Dtype promoted_dt = promote_scalar_ops(tensor_dt, scalar_dt);
    const Tensor& src = (promoted_dt == tensor_dt) ? a : a.as_type(promoted_dt);
    if (src.device().is_cuda()) {
#ifdef WITH_CUDA
        return cuda_sub_copy_scalar_tensor_impl<S>(s, src, OwnTensor::cuda_1926::getCurrentStream());
#else
        throw std::runtime_error("CUDA not supported");
#endif
    }
    Tensor result(src.shape(), promoted_dt, src.device(), src.requires_grad());
    dispatch_by_dtype(promoted_dt, [&](auto dummy) {
        using T = decltype(dummy);
        const T* in = src.data<T>();
        T* out = result.data<T>();
        size_t n = src.numel();
        T val = convert_scalar<T>(s);
        for (size_t i = 0; i < n; ++i) out[i] = val - in[i];
    });
    return result;
}

template<typename S>
Tensor operator/(S s, const Tensor& a) {
    constexpr Dtype scalar_dt = type_to_dtype<S>();
    const Dtype tensor_dt = a.dtype();
    const Dtype promoted_dt = promote_dtypes_division(tensor_dt, scalar_dt);
    const Tensor& src = (promoted_dt == tensor_dt) ? a : a.as_type(promoted_dt);
    if (src.device().is_cuda()) {
#ifdef WITH_CUDA
        return cuda_div_copy_scalar_tensor_impl<S>(s, src, OwnTensor::cuda_1926::getCurrentStream());
#else
        throw std::runtime_error("CUDA not supported");
#endif
    }
    Tensor result(src.shape(), promoted_dt, src.device(), src.requires_grad());
    dispatch_by_dtype(promoted_dt, [&](auto dummy) {
        using T = decltype(dummy);
        const T* in = src.data<T>();
        T* out = result.data<T>();
        size_t n = src.numel();
        T val = convert_scalar<T>(s);
        for (size_t i = 0; i < n; ++i) out[i] = val / in[i];
    });
    return result;
}

// Modulo copy (tensor % scalar)
template<typename S>
Tensor operator%(const Tensor& a, S s) {
    constexpr Dtype scalar_dt = type_to_dtype<S>();
    const Dtype tensor_dt = a.dtype();
    const Dtype promoted_dt = promote_scalar_ops(tensor_dt, scalar_dt);
    const Tensor& src = (promoted_dt == tensor_dt) ? a : a.as_type(promoted_dt);
    
    if (src.device().is_cuda()) {
#ifdef WITH_CUDA
        return cuda_mod_copy_impl<S>(src, s, OwnTensor::cuda_1926::getCurrentStream());
#else
        throw std::runtime_error("CUDA not supported");
#endif
    }
    
    Tensor result(src.shape(), promoted_dt, src.device(), src.requires_grad());
    dispatch_by_dtype(promoted_dt, [&](auto dummy) {
        using T = decltype(dummy);
        const T* in = src.data<T>();
        T* out = result.data<T>();
        size_t n = src.numel();
        T val = convert_scalar<T>(s);
        for (size_t i = 0; i < n; ++i) {
            if constexpr (std::is_integral_v<T>) {
                if (val == T(0)) throw std::runtime_error("Modulo by zero");
                out[i] = in[i] % val;
            } else if constexpr (std::is_floating_point_v<T>) {
                out[i] = static_cast<T>(std::fmod(static_cast<double>(in[i]), static_cast<double>(val)));
            } else {
                throw std::runtime_error("Modulo not supported for this dtype");
            }
        }
    });
    return result;
}

// Modulo copy (scalar % tensor)
template<typename S>
Tensor operator%(S s, const Tensor& a) {
    constexpr Dtype scalar_dt = type_to_dtype<S>();
    const Dtype tensor_dt = a.dtype();
    const Dtype promoted_dt = promote_scalar_ops(tensor_dt, scalar_dt);
    const Tensor& src = (promoted_dt == tensor_dt) ? a : a.as_type(promoted_dt);
    
    if (src.device().is_cuda()) {
#ifdef WITH_CUDA
        return cuda_mod_copy_scalar_tensor_impl<S>(s, src, OwnTensor::cuda_1926::getCurrentStream());
#else
        throw std::runtime_error("CUDA not supported");
#endif
    }
    
    Tensor result(src.shape(), promoted_dt, src.device(), src.requires_grad());
    dispatch_by_dtype(promoted_dt, [&](auto dummy) {
        using T = decltype(dummy);
        const T* in = src.data<T>();
        T* out = result.data<T>();
        size_t n = src.numel();
        T val = convert_scalar<T>(s);
        for (size_t i = 0; i < n; ++i) {
            if constexpr (std::is_integral_v<T>) {
                if (in[i] == T(0)) throw std::runtime_error("Modulo by zero");
                out[i] = val % in[i];
            } else if constexpr (std::is_floating_point_v<T>) {
                out[i] = static_cast<T>(std::fmod(static_cast<double>(val), static_cast<double>(in[i])));
            } else {
                throw std::runtime_error("Modulo not supported for this dtype");
            }
        }
    });
    return result;
}

// Comparison Operations
template<typename S>
Tensor operator==(const Tensor& a, S s) {
    if (a.device().is_cuda()) {
#ifdef WITH_CUDA
        return cuda_eq_copy_impl<S>(a, s, OwnTensor::cuda_1926::getCurrentStream());
#else
        throw std::runtime_error("CUDA not supported");
#endif
    }
    Tensor result(a.shape(), Dtype::Bool, a.device(), false);
    dispatch_by_dtype(a.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        const T* in = a.data<T>();
        bool* out = result.data<bool>();
        size_t n = a.numel();
        T val = convert_scalar<T>(s);
        for (size_t i = 0; i < n; ++i) out[i] = (in[i] == val);
    });
    return result;
}

template<typename S>
Tensor operator!=(const Tensor& a, S s) {
    if (a.device().is_cuda()) {
#ifdef WITH_CUDA
        return cuda_neq_copy_impl<S>(a, s, OwnTensor::cuda_1926::getCurrentStream());
#else
        throw std::runtime_error("CUDA not supported");
#endif
    }
    Tensor result(a.shape(), Dtype::Bool, a.device(), false);
    dispatch_by_dtype(a.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        const T* in = a.data<T>();
        bool* out = result.data<bool>();
        size_t n = a.numel();
        T val = convert_scalar<T>(s);
        for (size_t i = 0; i < n; ++i) out[i] = (in[i] != val);
    });
    return result;
}

template<typename S>
Tensor operator<=(const Tensor& a, S s) {
    if (is_complex(a.dtype())) throw std::runtime_error("<= not defined for complex");
    if (a.device().is_cuda()) {
#ifdef WITH_CUDA
        return cuda_leq_copy_impl<S>(a, s, OwnTensor::cuda_1926::getCurrentStream());
#else
        throw std::runtime_error("CUDA not supported");
#endif
    }
    Tensor result(a.shape(), Dtype::Bool, a.device(), false);
    dispatch_by_dtype(a.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        if constexpr (!is_complex_type_v<T>) {
            const T* in = a.data<T>();
            bool* out = result.data<bool>();
            T val = static_cast<T>(s);
            for (size_t i = 0; i < a.numel(); ++i) out[i] = (in[i] <= val);
        }
    });
    return result;
}

template<typename S>
Tensor operator>=(const Tensor& a, S s) {
    if (is_complex(a.dtype())) throw std::runtime_error(">= not defined for complex");
    if (a.device().is_cuda()) {
#ifdef WITH_CUDA
        return cuda_geq_copy_impl<S>(a, s, OwnTensor::cuda_1926::getCurrentStream());
#else
        throw std::runtime_error("CUDA not supported");
#endif
    }
    Tensor result(a.shape(), Dtype::Bool, a.device(), false);
    dispatch_by_dtype(a.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        if constexpr (!is_complex_type_v<T>) {
            const T* in = a.data<T>();
            bool* out = result.data<bool>();
            T val = static_cast<T>(s);
            for (size_t i = 0; i < a.numel(); ++i) out[i] = (in[i] >= val);
        }
    });
    return result;
}

template<typename S>
Tensor operator>(const Tensor& a, S s) {
    if (is_complex(a.dtype())) throw std::runtime_error("> not defined for complex");
    if (a.device().is_cuda()) {
#ifdef WITH_CUDA
        return cuda_gt_copy_impl<S>(a, s, OwnTensor::cuda_1926::getCurrentStream());
#else
        throw std::runtime_error("CUDA not supported");
#endif
    }
    Tensor result(a.shape(), Dtype::Bool, a.device(), false);
    dispatch_by_dtype(a.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        if constexpr (!is_complex_type_v<T>) {
            const T* in = a.data<T>();
            bool* out = result.data<bool>();
            T val = static_cast<T>(s);
            for (size_t i = 0; i < a.numel(); ++i) out[i] = (in[i] > val);
        }
    });
    return result;
}

template<typename S>
Tensor operator<(const Tensor& a, S s) {
    if (is_complex(a.dtype())) throw std::runtime_error("< not defined for complex");
    if (a.device().is_cuda()) {
#ifdef WITH_CUDA
        return cuda_lt_copy_impl<S>(a, s, OwnTensor::cuda_1926::getCurrentStream());
#else
        throw std::runtime_error("CUDA not supported");
#endif
    }
    Tensor result(a.shape(), Dtype::Bool, a.device(), false);
    dispatch_by_dtype(a.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        if constexpr (!is_complex_type_v<T>) {
            const T* in = a.data<T>();
            bool* out = result.data<bool>();
            T val = static_cast<T>(s);
            for (size_t i = 0; i < a.numel(); ++i) out[i] = (in[i] < val);
        }
    });
    return result;
}

// Commutative versions
template<typename S> Tensor operator+(S s, const Tensor& a) { return a + s; }
template<typename S> Tensor operator*(S s, const Tensor& a) { return a * s; }
template<typename S> Tensor operator==(S s, const Tensor& a) { return a == s; }
template<typename S> Tensor operator!=(S s, const Tensor& a) { return a != s; }

// Reversed comparison
template<typename S> Tensor operator<=(S s, const Tensor& a) { return a >= s; }
template<typename S> Tensor operator>=(S s, const Tensor& a) { return a <= s; }
template<typename S> Tensor operator<(S s, const Tensor& a) { return a > s; }
template<typename S> Tensor operator>(S s, const Tensor& a) { return a < s; }

} // namespace OwnTensor
