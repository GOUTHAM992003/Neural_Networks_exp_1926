// include/ops/helpers/GPUIntrinsics.cuh
// Centralized GPU intrinsic overloads for all low-precision types
// Namespace: gpu_1926

#pragma once

#ifndef GPU_INTRINSICS_CUH
#define GPU_INTRINSICS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "dtype/Types.h"

namespace OwnTensor {
namespace gpu_1926 {

// ═══════════════════════════════════════════════════════════════════════════
// TYPE CONVERSION - to_float / from_float
// ═══════════════════════════════════════════════════════════════════════════

// ---- to_float: Convert any type to float ----
template<typename T> __device__ __forceinline__ float to_float(T val) { 
    return static_cast<float>(val); 
}

template<> __device__ __forceinline__ float to_float(__half val) { 
    return __half2float(val); 
}

template<> __device__ __forceinline__ float to_float(__nv_bfloat16 val) { 
    return __bfloat162float(val); 
}

template<> __device__ __forceinline__ float to_float(__nv_fp8_e4m3 val) { 
    return static_cast<float>(val); 
}

template<> __device__ __forceinline__ float to_float(__nv_fp8_e5m2 val) { 
    return static_cast<float>(val); 
}

// ---- from_float: Convert float to target type ----
template<typename T> __device__ __forceinline__ T from_float(float val) { 
    return static_cast<T>(val); 
}

template<> __device__ __forceinline__ __half from_float(float val) { 
    return __float2half(val); 
}

template<> __device__ __forceinline__ __nv_bfloat16 from_float(float val) { 
    return __float2bfloat16(val); 
}

template<> __device__ __forceinline__ __nv_fp8_e4m3 from_float(float val) { 
    return __nv_fp8_e4m3(val); 
}

template<> __device__ __forceinline__ __nv_fp8_e5m2 from_float(float val) { 
    return __nv_fp8_e5m2(val); 
}

// ═══════════════════════════════════════════════════════════════════════════
// ARITHMETIC OPERATIONS - gpu_add, gpu_sub, gpu_mul, gpu_div, gpu_neg
// ═══════════════════════════════════════════════════════════════════════════

// ---- gpu_add ----
template<typename T> __device__ __forceinline__ T gpu_add(T a, T b) { 
    return a + b; 
}

template<> __device__ __forceinline__ __half gpu_add(__half a, __half b) { 
    return __hadd(a, b); 
}

template<> __device__ __forceinline__ __nv_bfloat16 gpu_add(__nv_bfloat16 a, __nv_bfloat16 b) { 
    return __hadd(a, b); 
}

// ---- gpu_sub ----
template<typename T> __device__ __forceinline__ T gpu_sub(T a, T b) { 
    return a - b; 
}

template<> __device__ __forceinline__ __half gpu_sub(__half a, __half b) { 
    return __hsub(a, b); 
}

template<> __device__ __forceinline__ __nv_bfloat16 gpu_sub(__nv_bfloat16 a, __nv_bfloat16 b) { 
    return __hsub(a, b); 
}

// ---- gpu_mul ----
template<typename T> __device__ __forceinline__ T gpu_mul(T a, T b) { 
    return a * b; 
}

template<> __device__ __forceinline__ __half gpu_mul(__half a, __half b) { 
    return __hmul(a, b); 
}

template<> __device__ __forceinline__ __nv_bfloat16 gpu_mul(__nv_bfloat16 a, __nv_bfloat16 b) { 
    return __hmul(a, b); 
}

// ---- gpu_div ----
template<typename T> __device__ __forceinline__ T gpu_div(T a, T b) { 
    return a / b; 
}

template<> __device__ __forceinline__ __half gpu_div(__half a, __half b) { 
    return __hdiv(a, b); 
}

template<> __device__ __forceinline__ __nv_bfloat16 gpu_div(__nv_bfloat16 a, __nv_bfloat16 b) { 
    return __hdiv(a, b); 
}

// ---- gpu_neg ----
template<typename T> __device__ __forceinline__ T gpu_neg(T a) { 
    return -a; 
}

template<> __device__ __forceinline__ __half gpu_neg(__half a) { 
    return __hneg(a); 
}

template<> __device__ __forceinline__ __nv_bfloat16 gpu_neg(__nv_bfloat16 a) { 
    return __hneg(a); 
}

// ---- gpu_abs ----
template<typename T> __device__ __forceinline__ T gpu_abs(T a) { 
    return (a < T(0)) ? -a : a; 
}

template<> __device__ __forceinline__ __half gpu_abs(__half a) { 
    return __habs(a); 
}

template<> __device__ __forceinline__ __nv_bfloat16 gpu_abs(__nv_bfloat16 a) { 
    return __habs(a); 
}

// ---- gpu_fma (fused multiply-add: a*b + c) ----
template<typename T> __device__ __forceinline__ T gpu_fma(T a, T b, T c) { 
    return a * b + c; 
}

template<> __device__ __forceinline__ __half gpu_fma(__half a, __half b, __half c) { 
    return __hfma(a, b, c); 
}

template<> __device__ __forceinline__ __nv_bfloat16 gpu_fma(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) { 
    return __hfma(a, b, c); 
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPARISON OPERATIONS - gpu_lt, gpu_le, gpu_gt, gpu_ge, gpu_eq, gpu_ne
// ═══════════════════════════════════════════════════════════════════════════

// ---- gpu_lt (less than) ----
template<typename T> __device__ __forceinline__ bool gpu_lt(T a, T b) { 
    return a < b; 
}

template<> __device__ __forceinline__ bool gpu_lt(__half a, __half b) { 
    return __hlt(a, b); 
}

template<> __device__ __forceinline__ bool gpu_lt(__nv_bfloat16 a, __nv_bfloat16 b) { 
    return __hlt(a, b); 
}

// ---- gpu_le (less than or equal) ----
template<typename T> __device__ __forceinline__ bool gpu_le(T a, T b) { 
    return a <= b; 
}

template<> __device__ __forceinline__ bool gpu_le(__half a, __half b) { 
    return __hle(a, b); 
}

template<> __device__ __forceinline__ bool gpu_le(__nv_bfloat16 a, __nv_bfloat16 b) { 
    return __hle(a, b); 
}

// ---- gpu_gt (greater than) ----
template<typename T> __device__ __forceinline__ bool gpu_gt(T a, T b) { 
    return a > b; 
}

template<> __device__ __forceinline__ bool gpu_gt(__half a, __half b) { 
    return __hgt(a, b); 
}

template<> __device__ __forceinline__ bool gpu_gt(__nv_bfloat16 a, __nv_bfloat16 b) { 
    return __hgt(a, b); 
}

// ---- gpu_ge (greater than or equal) ----
template<typename T> __device__ __forceinline__ bool gpu_ge(T a, T b) { 
    return a >= b; 
}

template<> __device__ __forceinline__ bool gpu_ge(__half a, __half b) { 
    return __hge(a, b); 
}

template<> __device__ __forceinline__ bool gpu_ge(__nv_bfloat16 a, __nv_bfloat16 b) { 
    return __hge(a, b); 
}

// ---- gpu_eq (equal) ----
template<typename T> __device__ __forceinline__ bool gpu_eq(T a, T b) { 
    return a == b; 
}

template<> __device__ __forceinline__ bool gpu_eq(__half a, __half b) { 
    return __heq(a, b); 
}

template<> __device__ __forceinline__ bool gpu_eq(__nv_bfloat16 a, __nv_bfloat16 b) { 
    return __heq(a, b); 
}

// ---- gpu_ne (not equal) ----
template<typename T> __device__ __forceinline__ bool gpu_ne(T a, T b) { 
    return a != b; 
}

template<> __device__ __forceinline__ bool gpu_ne(__half a, __half b) { 
    return __hne(a, b); 
}

template<> __device__ __forceinline__ bool gpu_ne(__nv_bfloat16 a, __nv_bfloat16 b) { 
    return __hne(a, b); 
}

// ═══════════════════════════════════════════════════════════════════════════
// MATH OPERATIONS - gpu_isnan
// ═══════════════════════════════════════════════════════════════════════════

template<typename T> __device__ __forceinline__ bool gpu_isnan(T val) { 
    if constexpr (std::is_same_v<T, complex32_t> || 
                  std::is_same_v<T, complex64_t> || 
                  std::is_same_v<T, complex128_t>) {
        return isnan(val);  // Use OwnTensor::isnan for complex
    } else if constexpr (std::is_floating_point_v<T>) {
        return ::isnan(val);
    } else {
        return false;  // Non-floating types can't be NaN
    }
}

template<> __device__ __forceinline__ bool gpu_isnan(__half val) { 
    return __hisnan(val); 
}

template<> __device__ __forceinline__ bool gpu_isnan(__nv_bfloat16 val) { 
    return __hisnan(val); 
}

// ═══════════════════════════════════════════════════════════════════════════
// WARP SHUFFLE - shfl_down
// ═══════════════════════════════════════════════════════════════════════════

// ---- Generic shfl_down ----
template<typename T>
__device__ __forceinline__ T shfl_down(T val, unsigned int delta) {
    return __shfl_down_sync(0xffffffff, val, delta, 32);
}

// ---- Specialization for __half ----
__device__ __forceinline__ __half shfl_down(__half val, unsigned int delta) {
    return __shfl_down_sync(0xffffffff, val, delta, 32);
}

// ---- Specialization for __nv_bfloat16 ----
__device__ __forceinline__ __nv_bfloat16 shfl_down(__nv_bfloat16 val, unsigned int delta) {
    return __shfl_down_sync(0xffffffff, val, delta, 32);
}

// ---- Specialization for __nv_fp8_e4m3 (convert to float, shuffle, convert back) ----
__device__ __forceinline__ __nv_fp8_e4m3 shfl_down(__nv_fp8_e4m3 val, unsigned int delta) {
    float f_val = static_cast<float>(val);
    float shuffled = __shfl_down_sync(0xffffffff, f_val, delta, 32);
    return __nv_fp8_e4m3(shuffled);
}

// ---- Specialization for __nv_fp8_e5m2 ----
__device__ __forceinline__ __nv_fp8_e5m2 shfl_down(__nv_fp8_e5m2 val, unsigned int delta) {
    float f_val = static_cast<float>(val);
    float shuffled = __shfl_down_sync(0xffffffff, f_val, delta, 32);
    return __nv_fp8_e5m2(shuffled);
}

// ---- Specialization for float8_e4m3fn_t (CPU type - shuffle raw bits) ----
__device__ __forceinline__ float8_e4m3fn_t shfl_down(float8_e4m3fn_t val, unsigned int delta) {
    uint8_t raw = __shfl_down_sync(0xffffffff, val.raw_bits, delta, 32);
    float8_e4m3fn_t result;
    result.raw_bits = raw;
    return result;
}

// ---- Specialization for float8_e5m2_t (CPU type - shuffle raw bits) ----
__device__ __forceinline__ float8_e5m2_t shfl_down(float8_e5m2_t val, unsigned int delta) {
    uint8_t raw = __shfl_down_sync(0xffffffff, val.raw_bits, delta, 32);
    float8_e5m2_t result;
    result.raw_bits = raw;
    return result;
}

// ---- Specialization for complex32_t (32-bit, cast to int) ----
__device__ __forceinline__ complex32_t shfl_down(complex32_t val, unsigned int delta) {
    int* ptr = reinterpret_cast<int*>(&val);
    int res = __shfl_down_sync(0xffffffff, *ptr, delta, 32);
    return *reinterpret_cast<complex32_t*>(&res);
}

// ---- Specialization for complex64_t (64-bit, cast to unsigned long long) ----
__device__ __forceinline__ complex64_t shfl_down(complex64_t val, unsigned int delta) {
    unsigned long long* ptr = reinterpret_cast<unsigned long long*>(&val);
    unsigned long long res = __shfl_down_sync(0xffffffff, *ptr, delta, 32);
    return *reinterpret_cast<complex64_t*>(&res);
}

// ---- Specialization for complex128_t (128-bit, shuffle components separately) ----
__device__ __forceinline__ complex128_t shfl_down(complex128_t val, unsigned int delta) {
    double r = __shfl_down_sync(0xffffffff, val.real_, delta, 32);
    double i = __shfl_down_sync(0xffffffff, val.imag_, delta, 32);
    return complex128_t(r, i);
}

} // namespace gpu_1926
} // namespace OwnTensor

#endif // GPU_INTRINSICS_CUH
