// ScalarOps.cpp - FIXED DIVISION OPERATORS
#include <cstdint>
#include <stdexcept>
#include "core/Tensor.h"
#include "core/TensorDispatch.h"
#include "dtype/Types.h"
#include "dtype/DtypeTraits.h"  //  For promote_dtypes_division

namespace OwnTensor {

// Helper trait for complex types (compile-time guard)
template <typename T> struct is_complex_t : std::false_type {};
template <> struct is_complex_t<complex32_t> : std::true_type {};
template <> struct is_complex_t<complex64_t> : std::true_type {};
template <> struct is_complex_t<complex128_t> : std::true_type {};

namespace { // file-local helpers

inline bool is_integer_dtype(Dtype dt) {
    return dt == Dtype::Int16 || dt == Dtype::Int32 || dt == Dtype::Int64;
}

inline float load_u16_as_f32(uint16_t bits, Dtype dt) {
    if (dt == Dtype::Float16)  return detail::float16_to_float(bits);
    if (dt == Dtype::Bfloat16) return detail::bfloat16_to_float(bits);
    return static_cast<float>(bits);
}

inline bool is_complex_dtype(Dtype dt) {
    return dt == Dtype::Complex32 || dt == Dtype::Complex64 || dt == Dtype::Complex128;
}

inline uint16_t store_f32_to_u16(float v, Dtype dt) {
    if (dt == Dtype::Float16)  return detail::float_to_float16(v);
    if (dt == Dtype::Bfloat16) return detail::float_to_bfloat16(v);
    return static_cast<uint16_t>(v);
}

// template <typename T>
// inline double ld(const T* p, size_t i, Dtype) { return static_cast<double>(p[i]); }

// // Specializations for complex types (can't convert complex to double)
// template <>
// inline double ld<complex32_t>(const complex32_t*, size_t, Dtype) {
//    throw std::runtime_error("Cannot perform scalar operations on complex32 types");
// }

// template <>
// inline double ld<complex64_t>(const complex64_t*, size_t, Dtype) {
//     throw std::runtime_error("Cannot perform scalar operations on complex64 types");
// }

// template <>
// inline double ld<complex128_t>(const complex128_t*, size_t, Dtype) {
//     throw std::runtime_error("Cannot perform scalar operations on complex128 types");
// }

// template <>
// inline double ld<uint16_t>(const uint16_t* p, size_t i, Dtype dt) {
//     return static_cast<double>(load_u16_as_f32(p[i], dt));
// }

// template <>
// inline double ld<float4_e2m1_2x_t>(const float4_e2m1_2x_t*, size_t, Dtype) {
//     throw std::runtime_error("Cannot perform scalar operations on packed FP4 types");
// }

// template <typename T>
// inline void st(T* p, size_t i, double v, Dtype) { p[i] = static_cast<T>(v); }

// template <>
// inline void st<float4_e2m1_2x_t>(float4_e2m1_2x_t*, size_t, double, Dtype) {
//     throw std::runtime_error("Cannot perform scalar operations on packed FP4 types");
// }


// template <>
// inline void st<uint16_t>(uint16_t* p, size_t i, double v, Dtype dt) {
//     p[i] = store_f32_to_u16(static_cast<float>(v), dt);
// }

// template <typename T, typename F>
// inline void apply_inplace(T* data, size_t n, Dtype dt, F&& f) {
//     for (size_t i = 0; i < n; ++i) st<T>(data, i, f(ld<T>(data, i, dt)), dt);
// }


// template <typename T, typename F>
// inline void apply_copy(const T* src, T* dst, size_t n, Dtype dt, F&& f) {
//     for (size_t i = 0; i < n; ++i) st<T>(dst, i, f(ld<T>(src, i, dt)), dt);
// }

// // Special version for comparison ops that write bool
// template <typename T, typename F>
// inline void apply_copy_to_bool(const T* src, uint8_t* dst, size_t n, Dtype dt, F&& f) {
//     for (size_t i = 0; i < n; ++i) {
//         dst[i] = f(ld<T>(src, i, dt)) ? 1 : 0;
//     }
// }

//  NEW: Helper to determine promoted dtype for division
inline Dtype get_division_output_dtype(Dtype input_dtype) {
    // Bool → Float32
    if (input_dtype == Dtype::Bool) return Dtype::Float32;
    
    // Integer types → Float32
    if (is_integer_dtype(input_dtype)) return Dtype::Float32;
    
    // Float types → Keep same (Float16/BFloat16/Float32/Float64)
    return input_dtype;
}

} // anon


// --------- Arithmetic ops (unchanged) ---------
template<typename S>
void cpu_add_inplace(Tensor& t, S s) {
    S* data_ptr = t.data<S>();
    for(size_t i=0;i<t.numel();i++)
    {
        data_ptr[i] = data_ptr[i] + s;
    }
}

template<typename S>
void cpu_sub_inplace(Tensor& t, S s) {
    S* data_ptr = t.data<S>();
    for(size_t i=0;i<t.numel();i++)
    {
        data_ptr[i] = data_ptr[i] - s;
    }
}

template<typename S>
void cpu_mul_inplace(Tensor& t, S s) {
    S* data_ptr = t.data<S>();
    for(size_t i=0;i<t.numel();i++)
    {
        data_ptr[i] = data_ptr[i] * s;
    }
}

//  FIXED: Division in-place (check if promotion needed)
template<typename S>
void cpu_div_inplace(Tensor& t, S s) {
    const Dtype dt = t.dtype();
    if (s == static_cast<S>(0)) throw std::runtime_error("Division by zero");
    //  Check if this would require promotion
    Dtype promoted_dt = get_division_output_dtype(dt);
    if (promoted_dt != dt) {
        throw std::runtime_error(
            "In-place division /= requires float dtype. "
            "Input is " + get_dtype_name(dt) + " but division needs " + 
            get_dtype_name(promoted_dt) + ". Use regular division (/) instead."
        );
    }  
    S* data_ptr = t.data<S>();
    for(size_t i=0;i<t.numel();i++)
    {
        data_ptr[i] = data_ptr[i] / s;
    }
}

template<typename S>
void cpu_add_copy(const Tensor& a, S s, Tensor& output, Dtype promote_dtype) {
    Dtype dt = a.dtype();
    if(is_complex_dtype(dt) || is_complex_dtype(output.dtype())) {
        throw std::runtime_error("Scalar division is not supported for complex dtypes");
    }
    Tensor b = a;
    if(dt != promote_dtype) {
        b=a.as_type(promote_dtype);
    }
    dispatch_by_dtype(promote_dtype, [&](auto d){
        using T = decltype(d);
        if constexpr (is_complex_t<T>::value == is_complex_t<S>::value) {
            T* data_ptr = b.data<T>();
            T* out_ptr = output.data<T>();
            for(size_t i=0;i<b.numel();i++)
            {
                out_ptr[i] = data_ptr[i] + static_cast<T>(s);
            }
        }
    });
}

template<typename S>
void cpu_sub_copy(const Tensor& a, S s, Tensor& output, Dtype promote_dtype) {
    Dtype dt = a.dtype();
    if(is_complex_dtype(dt) || is_complex_dtype(output.dtype())) {
        throw std::runtime_error("Scalar division is not supported for complex dtypes");
    }
    Tensor b;
    if(dt != promote_dtype) {
        b=a.as_type(promote_dtype);
    } else {
        b = a;
    }
    dispatch_by_dtype(promote_dtype, [&](auto d){
        using T = decltype(d);
        if constexpr (is_complex_t<T>::value == is_complex_t<S>::value) {
            T* data_ptr = b.data<T>();
            T* out_ptr = output.data<T>();
            for(size_t i=0;i<b.numel();i++)
            {
                out_ptr[i] = data_ptr[i] - static_cast<T>(s);
            }
        }
    });
}

template<typename S>
void cpu_mul_copy(const Tensor& a, S s, Tensor& output, Dtype promote_dtype) {
    Dtype dt = a.dtype();
    if(is_complex_dtype(dt) || is_complex_dtype(output.dtype())) {
        throw std::runtime_error("Scalar division is not supported for complex dtypes");
    }
    Tensor b;
    if(dt != promote_dtype) {
        b=a.as_type(promote_dtype);
    } else {
        b = a;
    }
    dispatch_by_dtype(promote_dtype, [&](auto d){
        using T = decltype(d);
        if constexpr (is_complex_t<T>::value == is_complex_t<S>::value) {
            T* data_ptr = b.data<T>();
            T* out_ptr = output.data<T>();
            for(size_t i=0;i<b.numel();i++)
            {
                out_ptr[i] = data_ptr[i] * static_cast<T>(s);
            }
        }
    });
}

// //  FIXED: Division creates Float32 output for integers/bool
template<typename S>
void cpu_div_copy(const Tensor& a, S s, Tensor& output, Dtype promote_dtype) {
    const Dtype input_dt = a.dtype();
    const Dtype output_dt = get_division_output_dtype(input_dt);
    if(is_complex_dtype(input_dt) || is_complex_dtype(output_dt)) {
        throw std::runtime_error("Scalar division is not supported for complex dtypes");
    }
    if ( s == static_cast<S>(0)) {
        throw std::runtime_error("Division by zero");
    }
    
    // If types match, use same-type path
    if (input_dt == output_dt) {
        dispatch_by_dtype(input_dt, [&](auto d){
            using T = decltype(d);
            if constexpr (!std::is_same_v<T, complex32_t> && !std::is_same_v<T, complex64_t> && !std::is_same_v<T, complex128_t>) {
                const T* src_ptr = a.data<T>();
                T* dst_ptr = output.data<T>();
                for (size_t i = 0; i < a.numel(); ++i) {
                    dst_ptr[i] = src_ptr[i] / s;
                }
            }
        });
    } else {
        // Cross-type conversion (Int16/Bool → Float32)
        dispatch_by_dtype(input_dt, [&](auto d_in){
            using SrcT = decltype(d_in);
            if constexpr (!std::is_same_v<SrcT, complex32_t> && !std::is_same_v<SrcT, complex64_t> && !std::is_same_v<SrcT, complex128_t>) {
                dispatch_by_dtype(output_dt, [&](auto d_out){
                    using DstT = decltype(d_out);
                    if constexpr (!std::is_same_v<DstT, complex32_t> && !std::is_same_v<DstT, complex64_t> && !std::is_same_v<DstT, complex128_t>) {
                        const SrcT* src_ptr = a.data<SrcT>();
                        DstT* dst_ptr = output.data<DstT>();
                        for (size_t i = 0; i < a.numel(); ++i) {
                           dst_ptr[i] = static_cast<DstT>(src_ptr[i]/ s);
                        }
                    }
                });
            }
            });
        }
}

template<typename S>
void cpu_sub_copy_scalar_tensor(S s, const Tensor& a, Tensor& output, Dtype promote_dtype) {
    const Dtype dt = a.dtype();
    Tensor b;
    if(dt != promote_dtype) {
        b=a.as_type(promote_dtype);
    } else {
        b = a;
    }
    dispatch_by_dtype(promote_dtype, [&](auto d){
        using T = decltype(d);
        if constexpr (is_complex_t<T>::value == is_complex_t<S>::value) {
            T* data_ptr = b.data<T>();
            T* out_ptr = output.data<T>();
            for(size_t i=0;i<b.numel();i++)
            {
                out_ptr[i] = data_ptr[i] - static_cast<T>(s);
            }
        }
    });
}

// //  FIXED: Scalar / Tensor also promotes to float
template<typename S>
void cpu_div_copy_scalar_tensor( S s, const Tensor& a, Tensor& output, Dtype promote_dtype) {
    const Dtype input_dt = a.dtype();
    const Dtype output_dt = output.dtype();
    Tensor b;
    if(input_dt == output_dt) {
        b=a;
    }
    else{
        b=a.as_type(promote_dtype); 
    }
    // Check for division by zero in integer tensors
   
        dispatch_by_dtype(input_dt, [&](auto d){
            using T = decltype(d);
            if constexpr (is_complex_t<T>::value == is_complex_t<S>::value) {
                const T* data_ptr = b.data<T>();
                T* out_ptr = output.data<T>();
                for (size_t i = 0, n = a.numel(); i < n; ++i)
                {
                    out_ptr[i] = static_cast<T>(s) / data_ptr[i];
                }
            }
        });
}

// // --------- Comparison ops (unchanged) ---------
template<typename S>
void cpu_eq_tensor_scalar(const Tensor& a, S s, Tensor& output, Dtype promote_dtype) {
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(output.data());
    dispatch_by_dtype(a.dtype(), [&](auto d) {
        using T = decltype(d);
        const T* data_ptr = a.data<T>();
        auto s1 = static_cast<T>(s);
        for (size_t i = 0, n = a.numel(); i < n; ++i) {
            out_ptr[i] = static_cast<uint8_t>(data_ptr[i] == s1);
        }
    });
}

template<typename S>
void  cpu_neq_tensor_scalar(const Tensor& a,S s, Tensor& output, Dtype promote_dtype) {
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(output.data());
    dispatch_by_dtype(a.dtype(), [&](auto d) {
        using T = decltype(d);
        const T* data_ptr = a.data<T>();
        auto s1 = static_cast<T>(s);
        for (size_t i = 0, n = a.numel(); i < n; ++i) {
            out_ptr[i] = static_cast<uint8_t>(data_ptr[i] == s1);
        }
    });
}

template<typename S>
void  cpu_lt_tensor_scalar(const Tensor& a,S s, Tensor& output, Dtype promote_dtype) {
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(output.data());
    dispatch_by_dtype(a.dtype(), [&](auto d) {
        using T = decltype(d);
        if constexpr (is_complex_t<T>::value) {
            throw std::runtime_error("Less-than comparison is not supported for complex types");
        } else {
            const T* data_ptr = a.data<T>();
            auto s1 = static_cast<T>(s);
            for (size_t i = 0, n = a.numel(); i < n; ++i) {
                out_ptr[i] = static_cast<uint8_t>(data_ptr[i] < s1);
            }
        }
    });
}

template<typename S>
void cpu_gt_tensor_scalar(const Tensor& a, S s, Tensor& output, Dtype promote_dtype) {
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(output.data());
    dispatch_by_dtype(a.dtype(), [&](auto d)
    {
        using T = decltype(d);
        if constexpr (is_complex_t<T>::value) {
            throw std::runtime_error("Greater-than comparison is not supported for complex types");
        } else {
            const T* data_ptr = a.data<T>();
            auto s1 = static_cast<T>(s);
            for (size_t i = 0, n = a.numel(); i < n; ++i)
            {
                out_ptr[i] = static_cast<uint8_t>(data_ptr[i] > s1);
            }
        }
    });
}

template<typename S>
void  cpu_leq_tensor_scalar(const Tensor& a,S s, Tensor& output, Dtype promote_dtype) {
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(output.data());
    dispatch_by_dtype(a.dtype(), [&](auto d) {
        using T = decltype(d);
        if constexpr (is_complex_t<T>::value) {
            throw std::runtime_error("Less-than-or-equal comparison is not supported for complex types");
        } else {
            const T* data_ptr = a.data<T>();
            auto s1 = static_cast<T>(s);
            for (size_t i = 0, n = a.numel(); i < n; ++i) {
                out_ptr[i] = static_cast<uint8_t>(data_ptr[i] <= s1);
            }
        }
    });
}

template<typename S>
void  cpu_geq_tensor_scalar(const Tensor& a,S s, Tensor& output, Dtype promote_dtype) {
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(output.data());
    dispatch_by_dtype(a.dtype(), [&](auto d) {
        using T = decltype(d);
        if constexpr (is_complex_t<T>::value) {
            throw std::runtime_error("Greater-than-or-equal comparison is not supported for complex types");
        } else {
            const T* data_ptr = a.data<T>();
            auto s1 = static_cast<T>(s);
            for (size_t i = 0, n = a.numel(); i < n; ++i) {
                out_ptr[i] = static_cast<uint8_t>(data_ptr[i] >= s1);
            }
        }
    });
}

template<typename S>
void  cpu_lt_scalar_tensor(S s,const Tensor& a, Tensor& output, Dtype promote_dtype) {
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(output.data());
    dispatch_by_dtype(a.dtype(), [&](auto d) {
        using T = decltype(d);
        if constexpr (is_complex_t<T>::value) {
            throw std::runtime_error("Less-than comparison is not supported for complex types");
        } else {
            const T* data_ptr = a.data<T>();
            auto s1 = static_cast<T>(s);
            for (size_t i = 0, n = a.numel(); i < n; ++i) {
                out_ptr[i] = static_cast<uint8_t>(s1 < data_ptr[i]);
            }
        }
    });
}

template<typename S>
void cpu_gt_scalar_tensor( S s,const Tensor& a, Tensor& output, Dtype promote_dtype) {
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(output.data());
    dispatch_by_dtype(a.dtype(), [&](auto d)
    {
        using T = decltype(d);
        if constexpr (is_complex_t<T>::value) {
            throw std::runtime_error("Greater-than comparison is not supported for complex types");
        } else {
            const T* data_ptr = a.data<T>();
            auto s1 = static_cast<T>(s);
            for (size_t i = 0, n = a.numel(); i < n; ++i)
            {
                out_ptr[i] = static_cast<uint8_t>( s1 > data_ptr[i]);
            }
        }
    });
}

template<typename S>
void  cpu_leq_scalar_tensor(S s,const Tensor& a, Tensor& output, Dtype promote_dtype) {
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(output.data());
    dispatch_by_dtype(a.dtype(), [&](auto d) {
        using T = decltype(d);
        if constexpr (is_complex_t<T>::value) {
            throw std::runtime_error("Less-than-or-equal comparison is not supported for complex types");
        } else {
            const T* data_ptr = a.data<T>();
            auto s1 = static_cast<T>(s);
            for (size_t i = 0, n = a.numel(); i < n; ++i) {
                out_ptr[i] = static_cast<uint8_t>(s1 <= data_ptr[i]);
            }
        }
    });
}

template<typename S>
void  cpu_geq_scalar_tensor(S s,const Tensor& a, Tensor& output, Dtype promote_dtype) {
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(output.data());
    dispatch_by_dtype(a.dtype(), [&](auto d) {
        using T = decltype(d);
        if constexpr (is_complex_t<T>::value) {
            throw std::runtime_error("Greater-than-or-equal comparison is not supported for complex types");
        } else {
            const T* data_ptr = a.data<T>();
            auto s1 = static_cast<T>(s);
            for (size_t i = 0, n = a.numel(); i < n; ++i) {
                out_ptr[i] = static_cast<uint8_t>(s1 >= data_ptr[i]);
            }
        }
    });
}
// --------- User-facing operators ---------
//CPU in-place
template void cpu_add_inplace<int8_t>( Tensor&, int8_t);
template void cpu_add_inplace<int16_t>( Tensor&, int16_t);
template void cpu_add_inplace<int32_t>( Tensor&, int32_t);
template void cpu_add_inplace<int64_t>( Tensor&, int64_t);
template void cpu_add_inplace<float>( Tensor&, float);
template void cpu_add_inplace<double>( Tensor&, double);
template void cpu_add_inplace<bool>( Tensor&, bool);
template void cpu_add_inplace<float16_t>( Tensor&, float16_t);
template void cpu_add_inplace<bfloat16_t>( Tensor&, bfloat16_t); 
template void cpu_add_inplace<uint8_t>( Tensor&, uint8_t);
template void cpu_add_inplace<uint16_t>( Tensor&, uint16_t);
template void cpu_add_inplace<uint32_t>( Tensor&, uint32_t);
template void cpu_add_inplace<uint64_t>( Tensor&, uint64_t);
// template void cpu_add_inplace<complex32_t>( Tensor&, complex32_t);
// template void cpu_add_inplace<complex64_t>( Tensor&, complex64_t);
// template void cpu_add_inplace<complex128_t>( Tensor&, complex128_t);

template void cpu_sub_inplace<int8_t>( Tensor&, int8_t);
template void cpu_sub_inplace<int16_t>( Tensor&, int16_t);
template void cpu_sub_inplace<int32_t>( Tensor&, int32_t);
template void cpu_sub_inplace<int64_t>( Tensor&, int64_t);
template void cpu_sub_inplace<float>( Tensor&, float);
template void cpu_sub_inplace<double>( Tensor&, double);
template void cpu_sub_inplace<bool>( Tensor&, bool);
template void cpu_sub_inplace<float16_t>( Tensor&, float16_t);
template void cpu_sub_inplace<bfloat16_t>( Tensor&, bfloat16_t); 
template void cpu_sub_inplace<uint8_t>( Tensor&, uint8_t);
template void cpu_sub_inplace<uint16_t>( Tensor&, uint16_t);
template void cpu_sub_inplace<uint32_t>( Tensor&, uint32_t);
template void cpu_sub_inplace<uint64_t>( Tensor&, uint64_t);
// template void cpu_sub_inplace<complex32_t>( Tensor&, complex32_t);
// template void cpu_sub_inplace<complex64_t>( Tensor&, complex64_t);
// template void cpu_sub_inplace<complex128_t>( Tensor&, complex128_t);

template void cpu_mul_inplace<int8_t>( Tensor&, int8_t);
template void cpu_mul_inplace<int16_t>( Tensor&, int16_t);
template void cpu_mul_inplace<int32_t>( Tensor&, int32_t);
template void cpu_mul_inplace<int64_t>( Tensor&, int64_t);
template void cpu_mul_inplace<float>( Tensor&, float);
template void cpu_mul_inplace<double>( Tensor&, double);
template void cpu_mul_inplace<bool>( Tensor&, bool);
template void cpu_mul_inplace<float16_t>( Tensor&, float16_t);
template void cpu_mul_inplace<bfloat16_t>( Tensor&, bfloat16_t); 
template void cpu_mul_inplace<uint8_t>( Tensor&, uint8_t);
template void cpu_mul_inplace<uint16_t>( Tensor&, uint16_t);
template void cpu_mul_inplace<uint32_t>( Tensor&, uint32_t);
template void cpu_mul_inplace<uint64_t>( Tensor&, uint64_t);
// template void cpu_mul_inplace<complex32_t>( Tensor&, complex32_t);
// template void cpu_mul_inplace<complex64_t>( Tensor&, complex64_t);
// template void cpu_mul_inplace<complex128_t>( Tensor&, complex128_t);

template void cpu_div_inplace<int8_t>( Tensor&, int8_t);
template void cpu_div_inplace<int16_t>( Tensor&, int16_t);
template void cpu_div_inplace<int32_t>( Tensor&, int32_t);
template void cpu_div_inplace<int64_t>( Tensor&, int64_t);
template void cpu_div_inplace<float>( Tensor&, float);
template void cpu_div_inplace<double>( Tensor&, double);
template void cpu_div_inplace<bool>( Tensor&, bool);
template void cpu_div_inplace<float16_t>( Tensor&, float16_t);
template void cpu_div_inplace<bfloat16_t>( Tensor&, bfloat16_t); 
template void cpu_div_inplace<uint8_t>( Tensor&, uint8_t);
template void cpu_div_inplace<uint16_t>( Tensor&, uint16_t);
template void cpu_div_inplace<uint32_t>( Tensor&, uint32_t);
template void cpu_div_inplace<uint64_t>( Tensor&, uint64_t);
// template void cpu_div_inplace<complex32_t>( Tensor&, complex32_t);
// template void cpu_div_inplace<complex64_t>( Tensor&, complex64_t);
// template void cpu_div_inplace<complex128_t>( Tensor&, complex128_t);

template void cpu_add_copy<int8_t>(const Tensor&, int8_t, Tensor&, Dtype );
template void cpu_add_copy<int16_t>(const Tensor&, int16_t, Tensor&, Dtype );
template void cpu_add_copy<int32_t>(const Tensor&, int32_t, Tensor&, Dtype );
template void cpu_add_copy<int64_t>(const Tensor&, int64_t, Tensor&, Dtype );
template void cpu_add_copy<float>(const Tensor&, float, Tensor&, Dtype );
template void cpu_add_copy<double>(const Tensor&, double, Tensor&, Dtype );
template void cpu_add_copy<bool>(const Tensor&, bool, Tensor&, Dtype );
template void cpu_add_copy<float16_t>(const Tensor&, float16_t, Tensor&, Dtype );
template void cpu_add_copy<bfloat16_t>(const Tensor&, bfloat16_t, Tensor&, Dtype ); 
template void cpu_add_copy<uint8_t>(const Tensor&, uint8_t, Tensor&, Dtype );
template void cpu_add_copy<uint16_t>(const Tensor&, uint16_t, Tensor&, Dtype );
template void cpu_add_copy<uint32_t>(const Tensor&, uint32_t, Tensor&, Dtype );
template void cpu_add_copy<uint64_t>(const Tensor&, uint64_t, Tensor&, Dtype );
// template void cpu_add_copy<complex32_t>(const Tensor&, complex32_t, Tensor&, Dtype );
// template void cpu_add_copy<complex64_t>(const Tensor&, complex64_t, Tensor&, Dtype );
// template void cpu_add_copy<complex128_t>(const Tensor&, complex128_t, Tensor&, Dtype );

template void cpu_sub_copy<int8_t>(const Tensor&, int8_t, Tensor&, Dtype );
template void cpu_sub_copy<int16_t>(const Tensor&, int16_t, Tensor&, Dtype );
template void cpu_sub_copy<int32_t>(const Tensor&, int32_t, Tensor&, Dtype );
template void cpu_sub_copy<int64_t>(const Tensor&, int64_t, Tensor&, Dtype );
template void cpu_sub_copy<float>(const Tensor&, float, Tensor&, Dtype );
template void cpu_sub_copy<double>(const Tensor&, double, Tensor&, Dtype );
template void cpu_sub_copy<bool>(const Tensor&, bool, Tensor&, Dtype );
template void cpu_sub_copy<float16_t>(const Tensor&, float16_t, Tensor&, Dtype );
template void cpu_sub_copy<bfloat16_t>(const Tensor&, bfloat16_t, Tensor&, Dtype ); 
template void cpu_sub_copy<uint8_t>(const Tensor&, uint8_t, Tensor&, Dtype );
template void cpu_sub_copy<uint16_t>(const Tensor&, uint16_t, Tensor&, Dtype );
template void cpu_sub_copy<uint32_t>(const Tensor&, uint32_t, Tensor&, Dtype );
template void cpu_sub_copy<uint64_t>(const Tensor&, uint64_t, Tensor&, Dtype );
// template void cpu_sub_copy<complex32_t>(const Tensor&, complex32_t, Tensor&, Dtype );
// template void cpu_sub_copy<complex64_t>(const Tensor&, complex64_t, Tensor&, Dtype );
// template void cpu_sub_copy<complex128_t>(const Tensor&, complex128_t, Tensor&, Dtype );

template void cpu_mul_copy<int8_t>(const Tensor&, int8_t, Tensor&, Dtype );
template void cpu_mul_copy<int16_t>(const Tensor&, int16_t, Tensor&, Dtype );
template void cpu_mul_copy<int32_t>(const Tensor&, int32_t, Tensor&, Dtype );
template void cpu_mul_copy<int64_t>(const Tensor&, int64_t, Tensor&, Dtype );
template void cpu_mul_copy<float>(const Tensor&, float, Tensor&, Dtype );
template void cpu_mul_copy<double>(const Tensor&, double, Tensor&, Dtype );
template void cpu_mul_copy<bool>(const Tensor&, bool, Tensor&, Dtype );
template void cpu_mul_copy<float16_t>(const Tensor&, float16_t, Tensor&, Dtype );
template void cpu_mul_copy<bfloat16_t>(const Tensor&, bfloat16_t, Tensor&, Dtype ); 
template void cpu_mul_copy<uint8_t>(const Tensor&, uint8_t, Tensor&, Dtype );
template void cpu_mul_copy<uint16_t>(const Tensor&, uint16_t, Tensor&, Dtype );
template void cpu_mul_copy<uint32_t>(const Tensor&, uint32_t, Tensor&, Dtype );
template void cpu_mul_copy<uint64_t>(const Tensor&, uint64_t, Tensor&, Dtype );
// template void cpu_mul_copy<complex32_t>(const Tensor&, complex32_t, Tensor&, Dtype );
// template void cpu_mul_copy<complex64_t>(const Tensor&, complex64_t, Tensor&, Dtype );
// template void cpu_mul_copy<complex128_t>(const Tensor&, complex128_t, Tensor&, Dtype );

template void cpu_div_copy<int8_t>(const Tensor&, int8_t, Tensor&, Dtype );
template void cpu_div_copy<int16_t>(const Tensor&, int16_t, Tensor&, Dtype );
template void cpu_div_copy<int32_t>(const Tensor&, int32_t, Tensor&, Dtype );
template void cpu_div_copy<int64_t>(const Tensor&, int64_t, Tensor&, Dtype );
template void cpu_div_copy<float>(const Tensor&, float, Tensor&, Dtype );
template void cpu_div_copy<double>(const Tensor&, double, Tensor&, Dtype );
template void cpu_div_copy<bool>(const Tensor&, bool, Tensor&, Dtype );
template void cpu_div_copy<float16_t>(const Tensor&, float16_t, Tensor&, Dtype );
template void cpu_div_copy<bfloat16_t>(const Tensor&, bfloat16_t, Tensor&, Dtype ); 
template void cpu_div_copy<uint8_t>(const Tensor&, uint8_t, Tensor&, Dtype );
template void cpu_div_copy<uint16_t>(const Tensor&, uint16_t, Tensor&, Dtype );
template void cpu_div_copy<uint32_t>(const Tensor&, uint32_t, Tensor&, Dtype );
template void cpu_div_copy<uint64_t>(const Tensor&, uint64_t, Tensor&, Dtype );
// template void cpu_div_copy<complex32_t>(const Tensor&, complex32_t, Tensor&, Dtype );
// template void cpu_div_copy<complex64_t>(const Tensor&, complex64_t, Tensor&, Dtype );
// template void cpu_div_copy<complex128_t>(const Tensor&, complex128_t, Tensor&, Dtype );

template void cpu_sub_copy_scalar_tensor<int8_t>(int8_t, const Tensor&, Tensor&, Dtype);
template void cpu_sub_copy_scalar_tensor<int16_t>(int16_t, const Tensor&, Tensor&, Dtype);
template void cpu_sub_copy_scalar_tensor<int32_t>(int32_t, const Tensor&, Tensor&, Dtype);
template void cpu_sub_copy_scalar_tensor<int64_t>(int64_t, const Tensor&, Tensor&, Dtype);
template void cpu_sub_copy_scalar_tensor<float>(float, const Tensor&, Tensor&, Dtype);
template void cpu_sub_copy_scalar_tensor<double>(double, const Tensor&, Tensor&, Dtype);
template void cpu_sub_copy_scalar_tensor<bool>(bool, const Tensor&, Tensor&, Dtype);
template void cpu_sub_copy_scalar_tensor<float16_t>(float16_t, const Tensor&, Tensor&, Dtype);
template void cpu_sub_copy_scalar_tensor<bfloat16_t>(bfloat16_t, const Tensor&, Tensor&, Dtype);
template void cpu_sub_copy_scalar_tensor<uint8_t>(uint8_t, const Tensor&, Tensor&, Dtype);
template void cpu_sub_copy_scalar_tensor<uint16_t>(uint16_t, const Tensor&, Tensor&, Dtype);
template void cpu_sub_copy_scalar_tensor<uint32_t>(uint32_t, const Tensor&, Tensor&, Dtype);
template void cpu_sub_copy_scalar_tensor<uint64_t>(uint64_t, const Tensor&, Tensor&, Dtype);

template void cpu_div_copy_scalar_tensor<int8_t>(int8_t, const Tensor&, Tensor&, Dtype);
template void cpu_div_copy_scalar_tensor<int16_t>(int16_t, const Tensor&, Tensor&, Dtype);
template void cpu_div_copy_scalar_tensor<int32_t>(int32_t, const Tensor&, Tensor&, Dtype);
template void cpu_div_copy_scalar_tensor<int64_t>(int64_t, const Tensor&, Tensor&, Dtype);
template void cpu_div_copy_scalar_tensor<float>(float, const Tensor&, Tensor&, Dtype);
template void cpu_div_copy_scalar_tensor<double>(double, const Tensor&, Tensor&, Dtype);
template void cpu_div_copy_scalar_tensor<bool>(bool, const Tensor&, Tensor&, Dtype);
template void cpu_div_copy_scalar_tensor<float16_t>(float16_t, const Tensor&, Tensor&, Dtype);
template void cpu_div_copy_scalar_tensor<bfloat16_t>(bfloat16_t, const Tensor&, Tensor&, Dtype);
template void cpu_div_copy_scalar_tensor<uint8_t>(uint8_t, const Tensor&, Tensor&, Dtype);
template void cpu_div_copy_scalar_tensor<uint16_t>(uint16_t, const Tensor&, Tensor&, Dtype);
template void cpu_div_copy_scalar_tensor<uint32_t>(uint32_t, const Tensor&, Tensor&, Dtype);
template void cpu_div_copy_scalar_tensor<uint64_t>(uint64_t, const Tensor&, Tensor&, Dtype);

template void cpu_eq_tensor_scalar<int8_t>(const Tensor&, int8_t, Tensor&, Dtype);
template void cpu_eq_tensor_scalar<int16_t>(const Tensor&, int16_t, Tensor&, Dtype);
template void cpu_eq_tensor_scalar<int32_t>(const Tensor&, int32_t, Tensor&, Dtype);
template void cpu_eq_tensor_scalar<int64_t>(const Tensor&, int64_t, Tensor&, Dtype);
template void cpu_eq_tensor_scalar<float>(const Tensor&, float, Tensor&, Dtype);
template void cpu_eq_tensor_scalar<double>(const Tensor&, double, Tensor&, Dtype);
template void cpu_eq_tensor_scalar<bool>(const Tensor&, bool, Tensor&, Dtype);
template void cpu_eq_tensor_scalar<float16_t>(const Tensor&, float16_t, Tensor&, Dtype);
template void cpu_eq_tensor_scalar<bfloat16_t>(const Tensor&, bfloat16_t, Tensor&, Dtype);
template void cpu_eq_tensor_scalar<uint8_t>(const Tensor&, uint8_t, Tensor&, Dtype);
template void cpu_eq_tensor_scalar<uint16_t>(const Tensor&, uint16_t, Tensor&, Dtype);
template void cpu_eq_tensor_scalar<uint32_t>(const Tensor&, uint32_t, Tensor&, Dtype);
template void cpu_eq_tensor_scalar<uint64_t>(const Tensor&, uint64_t, Tensor&, Dtype);

template void cpu_neq_tensor_scalar<int8_t>(const Tensor&, int8_t, Tensor&, Dtype);
template void cpu_neq_tensor_scalar<int16_t>(const Tensor&, int16_t, Tensor&, Dtype);
template void cpu_neq_tensor_scalar<int32_t>(const Tensor&, int32_t, Tensor&, Dtype);
template void cpu_neq_tensor_scalar<int64_t>(const Tensor&, int64_t, Tensor&, Dtype);
template void cpu_neq_tensor_scalar<float>(const Tensor&, float, Tensor&, Dtype);
template void cpu_neq_tensor_scalar<double>(const Tensor&, double, Tensor&, Dtype);
template void cpu_neq_tensor_scalar<bool>(const Tensor&, bool, Tensor&, Dtype);
template void cpu_neq_tensor_scalar<float16_t>(const Tensor&, float16_t, Tensor&, Dtype);
template void cpu_neq_tensor_scalar<bfloat16_t>(const Tensor&, bfloat16_t, Tensor&, Dtype);
template void cpu_neq_tensor_scalar<uint8_t>(const Tensor&, uint8_t, Tensor&, Dtype);
template void cpu_neq_tensor_scalar<uint16_t>(const Tensor&, uint16_t, Tensor&, Dtype);
template void cpu_neq_tensor_scalar<uint32_t>(const Tensor&, uint32_t, Tensor&, Dtype);
template void cpu_neq_tensor_scalar<uint64_t>(const Tensor&, uint64_t, Tensor&, Dtype);


template void cpu_gt_tensor_scalar<int8_t>(const Tensor&, int8_t, Tensor&, Dtype);
template void cpu_gt_tensor_scalar<int16_t>(const Tensor&, int16_t, Tensor&, Dtype);
template void cpu_gt_tensor_scalar<int32_t>(const Tensor&, int32_t, Tensor&, Dtype);
template void cpu_gt_tensor_scalar<int64_t>(const Tensor&, int64_t, Tensor&, Dtype);
template void cpu_gt_tensor_scalar<float>(const Tensor&, float, Tensor&, Dtype);
template void cpu_gt_tensor_scalar<double>(const Tensor&, double, Tensor&, Dtype);
template void cpu_gt_tensor_scalar<bool>(const Tensor&, bool, Tensor&, Dtype);
template void cpu_gt_tensor_scalar<float16_t>(const Tensor&, float16_t, Tensor&, Dtype);
template void cpu_gt_tensor_scalar<bfloat16_t>(const Tensor&, bfloat16_t, Tensor&, Dtype);
template void cpu_gt_tensor_scalar<uint8_t>(const Tensor&, uint8_t, Tensor&, Dtype);
template void cpu_gt_tensor_scalar<uint16_t>(const Tensor&, uint16_t, Tensor&, Dtype);
template void cpu_gt_tensor_scalar<uint32_t>(const Tensor&, uint32_t, Tensor&, Dtype);
template void cpu_gt_tensor_scalar<uint64_t>(const Tensor&, uint64_t, Tensor&, Dtype);

template void cpu_lt_tensor_scalar<int8_t>(const Tensor&, int8_t, Tensor&, Dtype);
template void cpu_lt_tensor_scalar<int16_t>(const Tensor&, int16_t, Tensor&, Dtype);
template void cpu_lt_tensor_scalar<int32_t>(const Tensor&, int32_t, Tensor&, Dtype);
template void cpu_lt_tensor_scalar<int64_t>(const Tensor&, int64_t, Tensor&, Dtype);
template void cpu_lt_tensor_scalar<float>(const Tensor&, float, Tensor&, Dtype);
template void cpu_lt_tensor_scalar<double>(const Tensor&, double, Tensor&, Dtype);
template void cpu_lt_tensor_scalar<bool>(const Tensor&, bool, Tensor&, Dtype);
template void cpu_lt_tensor_scalar<float16_t>(const Tensor&, float16_t, Tensor&, Dtype);
template void cpu_lt_tensor_scalar<bfloat16_t>(const Tensor&, bfloat16_t, Tensor&, Dtype);
template void cpu_lt_tensor_scalar<uint8_t>(const Tensor&, uint8_t, Tensor&, Dtype);
template void cpu_lt_tensor_scalar<uint16_t>(const Tensor&, uint16_t, Tensor&, Dtype);
template void cpu_lt_tensor_scalar<uint32_t>(const Tensor&, uint32_t, Tensor&, Dtype);
template void cpu_lt_tensor_scalar<uint64_t>(const Tensor&, uint64_t, Tensor&, Dtype);

template void cpu_geq_tensor_scalar<int8_t>(const Tensor&, int8_t, Tensor&, Dtype);
template void cpu_geq_tensor_scalar<int16_t>(const Tensor&, int16_t, Tensor&, Dtype);
template void cpu_geq_tensor_scalar<int32_t>(const Tensor&, int32_t, Tensor&, Dtype);
template void cpu_geq_tensor_scalar<int64_t>(const Tensor&, int64_t, Tensor&, Dtype);
template void cpu_geq_tensor_scalar<float>(const Tensor&, float, Tensor&, Dtype);
template void cpu_geq_tensor_scalar<double>(const Tensor&, double, Tensor&, Dtype);
template void cpu_geq_tensor_scalar<bool>(const Tensor&, bool, Tensor&, Dtype);
template void cpu_geq_tensor_scalar<float16_t>(const Tensor&, float16_t, Tensor&, Dtype);
template void cpu_geq_tensor_scalar<bfloat16_t>(const Tensor&, bfloat16_t, Tensor&, Dtype);
template void cpu_geq_tensor_scalar<uint8_t>(const Tensor&, uint8_t, Tensor&, Dtype);
template void cpu_geq_tensor_scalar<uint16_t>(const Tensor&, uint16_t, Tensor&, Dtype);
template void cpu_geq_tensor_scalar<uint32_t>(const Tensor&, uint32_t, Tensor&, Dtype);
template void cpu_geq_tensor_scalar<uint64_t>(const Tensor&, uint64_t, Tensor&, Dtype);

template void cpu_leq_tensor_scalar<int8_t>(const Tensor&, int8_t, Tensor&, Dtype);
template void cpu_leq_tensor_scalar<int16_t>(const Tensor&, int16_t, Tensor&, Dtype);
template void cpu_leq_tensor_scalar<int32_t>(const Tensor&, int32_t, Tensor&, Dtype);
template void cpu_leq_tensor_scalar<int64_t>(const Tensor&, int64_t, Tensor&, Dtype);
template void cpu_leq_tensor_scalar<float>(const Tensor&, float, Tensor&, Dtype);
template void cpu_leq_tensor_scalar<double>(const Tensor&, double, Tensor&, Dtype);
template void cpu_leq_tensor_scalar<bool>(const Tensor&, bool, Tensor&, Dtype);
template void cpu_leq_tensor_scalar<float16_t>(const Tensor&, float16_t, Tensor&, Dtype);
template void cpu_leq_tensor_scalar<bfloat16_t>(const Tensor&, bfloat16_t, Tensor&, Dtype);
template void cpu_leq_tensor_scalar<uint8_t>(const Tensor&, uint8_t, Tensor&, Dtype);
template void cpu_leq_tensor_scalar<uint16_t>(const Tensor&, uint16_t, Tensor&, Dtype);
template void cpu_leq_tensor_scalar<uint32_t>(const Tensor&, uint32_t, Tensor&, Dtype);
template void cpu_leq_tensor_scalar<uint64_t>(const Tensor&, uint64_t, Tensor&, Dtype);

template void cpu_gt_scalar_tensor<int8_t>(  int8_t, const Tensor&, Tensor&, Dtype);
template void cpu_gt_scalar_tensor<int16_t>( int16_t,const Tensor&, Tensor&, Dtype);
template void cpu_gt_scalar_tensor<int32_t>( int32_t,const Tensor&, Tensor&, Dtype);
template void cpu_gt_scalar_tensor<int64_t>( int64_t,const Tensor&, Tensor&, Dtype);
template void cpu_gt_scalar_tensor<float>(   float,const Tensor&, Tensor&, Dtype);
template void cpu_gt_scalar_tensor<double>(  double, const Tensor&, Tensor&, Dtype);
template void cpu_gt_scalar_tensor<bool>(    bool, const Tensor&, Tensor&, Dtype);
template void cpu_gt_scalar_tensor<float16_t>( float16_t,const Tensor&, Tensor&, Dtype);
template void cpu_gt_scalar_tensor<bfloat16_t>( bfloat16_t,const Tensor&, Tensor&, Dtype);
template void cpu_gt_scalar_tensor<uint8_t>(  uint8_t, const Tensor&,Tensor&, Dtype);
template void cpu_gt_scalar_tensor<uint16_t>( uint16_t, const Tensor&,Tensor&, Dtype);
template void cpu_gt_scalar_tensor<uint32_t>( uint32_t, const Tensor&,Tensor&, Dtype);
template void cpu_gt_scalar_tensor<uint64_t>( uint64_t,const  Tensor&,Tensor&, Dtype);

template void cpu_lt_scalar_tensor<int8_t>(  int8_t, const Tensor&,Tensor&, Dtype);
template void cpu_lt_scalar_tensor<int16_t>( int16_t,const Tensor&, Tensor&, Dtype);
template void cpu_lt_scalar_tensor<int32_t>( int32_t,const Tensor&, Tensor&, Dtype);
template void cpu_lt_scalar_tensor<int64_t>( int64_t,const Tensor&, Tensor&, Dtype);
template void cpu_lt_scalar_tensor<float>(  float, const Tensor&,Tensor&, Dtype);
template void cpu_lt_scalar_tensor<double>( double,const Tensor&, Tensor&, Dtype);
template void cpu_lt_scalar_tensor<bool>(  bool, const Tensor&, Tensor&, Dtype);
template void cpu_lt_scalar_tensor<float16_t>( float16_t,const Tensor&, Tensor&, Dtype);
template void cpu_lt_scalar_tensor<bfloat16_t>( bfloat16_t,const Tensor&, Tensor&, Dtype);
template void cpu_lt_scalar_tensor<uint8_t>( uint8_t, const Tensor&,Tensor&, Dtype);
template void cpu_lt_scalar_tensor<uint16_t>(uint16_t, const Tensor&,Tensor&, Dtype);
template void cpu_lt_scalar_tensor<uint32_t>(uint32_t, const Tensor&,Tensor&, Dtype);
template void cpu_lt_scalar_tensor<uint64_t>(uint64_t,const  Tensor&,Tensor&, Dtype);

template void cpu_geq_scalar_tensor<int8_t>(int8_t, const Tensor&, Tensor&, Dtype);
template void cpu_geq_scalar_tensor<int16_t>(int16_t, const Tensor&, Tensor&, Dtype);
template void cpu_geq_scalar_tensor<int32_t>(int32_t, const Tensor&, Tensor&, Dtype);
template void cpu_geq_scalar_tensor<int64_t>(int64_t, const Tensor&, Tensor&, Dtype);
template void cpu_geq_scalar_tensor<float>(float, const Tensor&, Tensor&, Dtype);
template void cpu_geq_scalar_tensor<double>(double, const Tensor&, Tensor&, Dtype);
template void cpu_geq_scalar_tensor<bool>(bool, const Tensor&, Tensor&, Dtype);
template void cpu_geq_scalar_tensor<float16_t>(float16_t, const Tensor&, Tensor&, Dtype);
template void cpu_geq_scalar_tensor<bfloat16_t>(bfloat16_t, const Tensor&, Tensor&, Dtype);
template void cpu_geq_scalar_tensor<uint8_t>(uint8_t, const Tensor&, Tensor&, Dtype);
template void cpu_geq_scalar_tensor<uint16_t>(uint16_t, const Tensor&, Tensor&, Dtype);
template void cpu_geq_scalar_tensor<uint32_t>(uint32_t, const Tensor&, Tensor&, Dtype);
template void cpu_geq_scalar_tensor<uint64_t>(uint64_t, const Tensor&, Tensor&, Dtype);

template void cpu_leq_scalar_tensor<int8_t>(int8_t, const Tensor&, Tensor&, Dtype);
template void cpu_leq_scalar_tensor<int16_t>(int16_t, const Tensor&, Tensor&, Dtype);
template void cpu_leq_scalar_tensor<int32_t>(int32_t, const Tensor&, Tensor&, Dtype);
template void cpu_leq_scalar_tensor<int64_t>(int64_t, const Tensor&, Tensor&, Dtype);
template void cpu_leq_scalar_tensor<float>(float, const Tensor&, Tensor&, Dtype);
template void cpu_leq_scalar_tensor<double>(double, const Tensor&, Tensor&, Dtype);
template void cpu_leq_scalar_tensor<bool>(bool, const Tensor&, Tensor&, Dtype);
template void cpu_leq_scalar_tensor<float16_t>(float16_t, const Tensor&, Tensor&, Dtype);
template void cpu_leq_scalar_tensor<bfloat16_t>(bfloat16_t, const Tensor&, Tensor&, Dtype);
template void cpu_leq_scalar_tensor<uint8_t>(uint8_t, const Tensor&, Tensor&, Dtype);
template void cpu_leq_scalar_tensor<uint16_t>(uint16_t, const Tensor&, Tensor&, Dtype);
template void cpu_leq_scalar_tensor<uint32_t>(uint32_t, const Tensor&, Tensor&, Dtype);
template void cpu_leq_scalar_tensor<uint64_t>(uint64_t, const Tensor&, Tensor&, Dtype);


} // namespace OwnTensor