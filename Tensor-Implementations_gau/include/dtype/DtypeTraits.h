// include/dtype/DtypeTraits.h - MERGED VERSION (Compatible with both versions)
#pragma once

#ifndef DTYPE_TRAIT_H
#define DTYPE_TRAIT_H

#include <cstdint>
#include <complex>
#include <type_traits>
#include <string>
#include "core/Tensor.h"
#include <cassert>
// ✅ ALWAYS use custom structs (both CPU and GPU compilation)
#include "dtype/Types.h"

// ═══════════════════════════════════════════════════════════
// DTYPE TRAITS
// ═══════════════════════════════════════════════════════════

namespace OwnTensor {
    enum class Dtype;

    template <Dtype dt>
    struct dtype_traits {
        using type = void;
        static constexpr size_t size = 0;
        static constexpr const char* name = "invalid";
        static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = false;
        static constexpr bool is_unsigned = false;
    };

    // Integer Types
    template <> struct dtype_traits<Dtype::Int8> {
        using type = int8_t;
        static constexpr size_t size = sizeof(int8_t);
        static constexpr const char* name = "int8";
        static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = true;
        static constexpr bool is_unsigned = false;
    };
    template <> struct dtype_traits<Dtype::Int16> {
        using type = int16_t;
        static constexpr size_t size = sizeof(int16_t);
        static constexpr const char* name = "int16";
        static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = true;
        static constexpr bool is_unsigned = false;
    };

    template <> struct dtype_traits<Dtype::Int32> {
        using type = int32_t;
        static constexpr size_t size = sizeof(int32_t);
        static constexpr const char* name = "int32";
        static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = true;
        static constexpr bool is_unsigned = false;
    };

    template <> struct dtype_traits<Dtype::Int64> {
        using type = int64_t;
        static constexpr size_t size = sizeof(int64_t);
        static constexpr const char* name = "int64";
        static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = true;
        static constexpr bool is_unsigned = false;
    };
//Unsigned int types
template<> struct dtype_traits<Dtype::UInt8> {
    using type = uint8_t;
    static constexpr size_t size = sizeof(uint8_t);
    static constexpr const char* name = "UInt8";
    static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = true;
        static constexpr bool is_unsigned = true;
        
};

template<> struct dtype_traits<Dtype::UInt16> {
    using type = uint16_t;
    static constexpr size_t size = sizeof(uint16_t);
    static constexpr const char* name = "UInt16";
    static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = true;
        static constexpr bool is_unsigned = true;
};

template<> struct dtype_traits<Dtype::UInt32> {
    using type = uint32_t;
    static constexpr size_t size = sizeof(uint32_t);
    static constexpr const char* name = "UInt32";
    static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = true;
        static constexpr bool is_unsigned = true;
};

template<> struct dtype_traits<Dtype::UInt64> {
    using type = uint64_t;
    static constexpr size_t size = sizeof(uint64_t);
    static constexpr const char* name = "UInt64";
    static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = true;
        static constexpr bool is_unsigned = true;
};

    // Floating Point Types -  used custom structs for custom floating point types
    template <> struct dtype_traits<Dtype::Float16> {
        using type = float16_t;  // Custom struct
        static constexpr size_t size = sizeof(float16_t);
        static constexpr const char* name = "fp16";
        static constexpr bool is_floating_point = true;
        static constexpr bool is_integral = false;
        static constexpr bool is_unsigned = false;
    };

    template <> struct dtype_traits<Dtype::Bfloat16> {   
        using type = bfloat16_t;  // Custom struct
        static constexpr size_t size = sizeof(bfloat16_t);
        static constexpr const char* name = "bf16";
        static constexpr bool is_floating_point = true;
        static constexpr bool is_integral = false;
        static constexpr bool is_unsigned = false;
    };

    template <> struct dtype_traits<Dtype::Float32> {
        using type = float;
        static constexpr size_t size = sizeof(float);
        static constexpr const char* name = "float32";
        static constexpr bool is_floating_point = true;
        static constexpr bool is_integral = false;
        static constexpr bool is_unsigned = false;
    };

    template <> struct dtype_traits<Dtype::Float64> {
        using type = double;
        static constexpr size_t size = sizeof(double);
        static constexpr const char* name = "float64";
        static constexpr bool is_floating_point = true;
        static constexpr bool is_integral = false;
        static constexpr bool is_unsigned = false;
    };

    // bool specialization
    template<> struct dtype_traits<Dtype::Bool> {
        static constexpr const char* name = "bool";
        static constexpr Dtype dtype = Dtype::Bool;
        static constexpr size_t size = sizeof(bool);  // Usually 1 byte
        using type = bool;
        // using storage_type = uint8_t;
    };

    // Complex specification - used custom struct for custom complex struct (chalf)
    
    template<> struct dtype_traits<Dtype::Complex32>{
        using type = complex32_t;
        static constexpr Dtype dtype = Dtype::Complex32;
        static constexpr size_t size= sizeof(complex32_t);
        static constexpr const char* name = "complex32";
        static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = false;
        static constexpr bool is_unsigned =false;
        static constexpr bool is_complex = true;
    };

    template<> struct dtype_traits<Dtype::Complex64>{
        using type = complex64_t;
        static constexpr Dtype dtype = Dtype::Complex64;
        static constexpr size_t size = sizeof(complex64_t);
        static constexpr const char* name = "complex64";
        static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = false;
        static constexpr bool is_unsigned = false;
        static constexpr bool is_complex = true;
    };
 
    template<> struct dtype_traits<Dtype::Complex128>{
        using type = complex128_t;
        static constexpr Dtype dtype = Dtype::Complex128;
        static constexpr size_t size = sizeof(complex128_t);
        static constexpr const char* name = "complex128";
        static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = false;
        static constexpr bool is_unsigned = false;
        static constexpr bool is_complex = true;
    };

    // FP8 Types (8-bit Floating Point for Deep Learning)
    template<> struct dtype_traits<Dtype::Float8_E4M3FN> {
        using type = float8_e4m3fn_t;  // Will define this struct in Types.h
        static constexpr size_t size = sizeof(uint8_t);  // 1 byte
        static constexpr const char* name = "float8_e4m3fn";
        static constexpr bool is_floating_point = true;
        static constexpr bool is_integral = false;
        static constexpr bool is_unsigned = false;
        static constexpr bool is_complex = false;
    };

    template<> struct dtype_traits<Dtype::Float8_E5M2> {
        using type = float8_e5m2_t;  // Will define this struct in Types.h
        static constexpr size_t size = sizeof(uint8_t);  // 1 byte
        static constexpr const char* name = "float8_e5m2";
        static constexpr bool is_floating_point = true;
        static constexpr bool is_integral = false;
        static constexpr bool is_unsigned = false;
        static constexpr bool is_complex = false;
    };

    // Helper function
    template<typename T>
    bool is_same_type(Dtype dtype) {
        if constexpr (std::is_same_v<T, int32_t>) {
            return dtype == Dtype::Int32;
        }else if constexpr(std::is_same_v<T,int8_t>){
            return dtype == Dtype::Int8;
        }
         else if constexpr (std::is_same_v<T, float>) {
            return dtype == Dtype::Float32;
        } else if constexpr (std::is_same_v<T, double>) {
            return dtype == Dtype::Float64;
        } else if constexpr (std::is_same_v<T, int16_t>) {
            return dtype == Dtype::Int16;
        } else if constexpr (std::is_same_v<T, int64_t>) {
            return dtype == Dtype::Int64;
        } 
        else if constexpr (std::is_same_v<T, uint8_t>) {
            return dtype == Dtype::UInt8;
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            return dtype == Dtype::UInt16;
        } else if constexpr (std::is_same_v<T, uint32_t>) {
            return dtype == Dtype::UInt32;
        } else if constexpr (std::is_same_v<T, uint64_t>) {
            return dtype == Dtype::UInt64;
        }
        else if constexpr (std::is_same_v<T, float16_t>) {
            return dtype == Dtype::Float16; 
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            return dtype == Dtype::Bfloat16;
        }
        else if constexpr (std::is_same_v<T, float8_e4m3fn_t>) {
            return dtype == Dtype::Float8_E4M3FN;
        }
        else if constexpr (std::is_same_v<T, float8_e5m2_t>) {
            return dtype == Dtype::Float8_E5M2;
        }
        else if constexpr (std::is_same_v<T, bool>) {
            return dtype == Dtype::Bool;
        }else if constexpr (std::is_same_v<T, complex32_t>){
            return dtype == Dtype::Complex32;
        }else if constexpr (std::is_same_v<T,complex64_t>){
            return dtype == Dtype::Complex64;
        }else if constexpr (std::is_same_v<T, complex128_t>){
            return dtype == Dtype::Complex128;
        }
        return false;
    }

    // ═══════════════════════════════════════════════════════════
    // TYPE TO DTYPE CONVERSION
    // ═══════════════════════════════════════════════════════════

    template<typename T>
    constexpr Dtype type_to_dtype() {
        // Integer types
        if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, short>) {
            return Dtype::Int16;
        }
        else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, signed char>){
            return Dtype::Int8;
        }
        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int>) {
            return Dtype::Int32;
        }
        else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, long> || 
                           std::is_same_v<T, long long>) {
            return Dtype::Int64;
        }
        // Unsigned integer types
    else if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, unsigned char>) {
        return Dtype::UInt8;
    } else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, unsigned short>) {
        return Dtype::UInt16;
    } else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, unsigned int>) {
        return Dtype::UInt32;
    } else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, unsigned long> || std::is_same_v<T, unsigned long long>) {
        return Dtype::UInt64;
    }
        // Floating point types - custom structs
        else if constexpr (std::is_same_v<T, float16_t>) {
            return Dtype::Float16;
        }
        else if constexpr (std::is_same_v<T, bfloat16_t>) {
            return Dtype::Bfloat16;
        }
        // FP8 types
        else if constexpr (std::is_same_v<T, float8_e4m3fn_t>) {
            return Dtype::Float8_E4M3FN;
        }
        else if constexpr (std::is_same_v<T, float8_e5m2_t>) {
            return Dtype::Float8_E5M2;
        }
        else if constexpr (std::is_same_v<T, float>) {
            return Dtype::Float32;
        }
        else if constexpr (std::is_same_v<T, double>) {
            return Dtype::Float64;
        }
        else if constexpr (std::is_same_v<T, bool>) {
            return Dtype::Bool;
        }
        else if constexpr (std::is_same_v<T, complex32_t>){
            return Dtype::Complex32;
       }
        else if constexpr (std::is_same_v<T, complex64_t> ){
            return Dtype::Complex64;
        }else if constexpr (std::is_same_v<T,complex128_t> ){
            return Dtype::Complex128;
        }
        else {
            static_assert(!std::is_same_v<T, T>, "Unsupported type");
        }
    }

    // ═══════════════════════════════════════════════════════════
    // TYPE PREDICATES 
    // ═══════════════════════════════════════════════════════════

    constexpr bool is_float(Dtype dt) {
        switch (dt) {
        case Dtype::Float16:
        case Dtype::Bfloat16:
        case Dtype::Float32:
        case Dtype::Float64:
        case Dtype::Float8_E4M3FN:
        case Dtype::Float8_E5M2:
            return true;
        default:
            return false;
        }
    }

    constexpr bool is_int(Dtype dt) {
        switch (dt) {
        case Dtype::Int8:
        case Dtype::Int16:
        case Dtype::Int32:
        case Dtype::Int64:
       
            return true;
        default:
            return false;
        }
    }
    constexpr bool is_unsigned(Dtype dt){
        switch (dt){
            case Dtype::UInt8:
            case Dtype::UInt16:
            case Dtype::UInt32:
            case Dtype::UInt64:
                return true;
            default :
                return false;
        }
    }
    constexpr bool is_bool(Dtype dt) {
        return dt == Dtype::Bool;
    }
    constexpr bool is_complex(Dtype dt) {
        switch (dt){
            case Dtype::Complex32:
            case Dtype::Complex64:
            case Dtype::Complex128:
                return true;
            default :
                return false;
        }
    }

    // ═══════════════════════════════════════════════════════════
    // COMPILE-TIME COMPLEX TYPE CHECK (for template code)
    // ═══════════════════════════════════════════════════════════
    template<typename T>
    inline constexpr bool is_complex_type_v = 
        std::is_same_v<T, complex32_t> || 
        std::is_same_v<T, complex64_t> || 
        std::is_same_v<T, complex128_t>;
    // ═══════════════════════════════════════════════════════════
    // DTYPE NAME HELPER ( Returns std::string )
    // ═══════════════════════════════════════════════════════════

    inline std::string get_dtype_name(Dtype dtype) {
        switch(dtype) {
            case Dtype::Int8:     return "int8";
            case Dtype::Int16:    return "int16";
            case Dtype::Int32:    return "int32";
            case Dtype::Int64:    return "int64";
            case Dtype::UInt8:    return "uint8";
            case Dtype::UInt16:   return "uint16";
            case Dtype::UInt32:   return "uint32";
            case Dtype::UInt64:   return "uint64";
            case Dtype::Float16:  return "float16";  
            case Dtype::Bfloat16: return "bfloat16";
            case Dtype::Float8_E4M3FN: return "float8_e4m3fn";
            case Dtype::Float8_E5M2:   return "float8_e5m2";
            case Dtype::Float32:  return "float32";
            case Dtype::Float64:  return "float64";
            case Dtype::Bool:     return "bool";
            case Dtype::Complex32: return "complex32";
            case Dtype::Complex64: return "complex64";
            case Dtype::Complex128: return "complex128";
            default:              return "Unknown";  
        }
    }

// ═══════════════════════════════════════════════════════════════════════════
// DTYPE PROMOTION SYSTEM OVERVIEW
// ═══════════════════════════════════════════════════════════════════════════
//
// This library implements TWO promotion functions:
//
// 1. promote_tensor_ops(a, b)     → Tensor + Tensor operations
// 2. promote_scalar_ops(t, s)   → Tensor + C++ Scalar operations
//
// ═══════════════════════════════════════════════════════════════════════════
// DESIGN DECISION: 0-dim Tensors vs Regular Tensors
// ═══════════════════════════════════════════════════════════════════════════
//
// CURRENT DESIGN: 0-dim tensors are treated SAME as regular tensors.
// Both use promote_tensor_ops() for equal precedence.
//
// PYTORCH DESIGN: PyTorch has 3 categories with different precedence:
//   - dimResult (dim > 0)    → STRONGEST
//   - zeroResult (dim == 0)  → MIDDLE  
//   - wrappedResult (scalar) → WEAKEST
//
// IMPACT: For integer ops like int8[3] + int32[]:
//   - Our library:  int32 (normal promotion)
//   - PyTorch:      int8  (dim tensor wins)
//
// FUTURE: To match/extend  PyTorch or other deeplearning libraries, add promote_zerodim_tensor() and
// check ndim() before choosing which promotion function to use.
//promote_zerodim_tensor() function behaviour is written in comments at the end of this file .(based on pytorch dtypes behaviour)
//
// ═══════════════════════════════════════════════════════════════════════════
// LOOKUP TABLE FOR TENSOR-TENSOR DTYPE PROMOTION
// ═══════════════════════════════════════════════════════════════════════════
// 18x18 table indexed by Dtype enum values for O(1) lookup
// 
// Dtype enum order (0-17):
//   0:Int8, 1:Int16, 2:Int32, 3:Int64, 4:UInt8, 5:UInt16, 6:UInt32, 7:UInt64,
//   8:Bfloat16, 9:Float16, 10:Float32, 11:Float64, 12:Bool, 
//   13:Complex32, 14:Complex64, 15:Complex128, 16:Float8_E4M3FN, 17:Float8_E5M2
//
// Special values: ERROR_* markers indicate runtime errors should be thrown
// ═══════════════════════════════════════════════════════════════════════════

constexpr int DTYPE_COUNT = 18;

// Error markers (negative enum values not used by Dtype)
constexpr int ERR_MIXED_FP8 = -1;        // E4M3 + E5M2 not allowed
constexpr int ERR_FP8_HIGHER = -2;       // FP8 + Float16/32/64/BFloat16 not allowed
constexpr int ERR_UINT_SIGNED = -3;      // UInt16/32/64 + signed int not allowed

// Helper to convert Dtype to int index
constexpr int dtype_to_idx(Dtype d) {
    return static_cast<int>(d);
}

// Shorthand for readability
constexpr int I8 = 0, I16 = 1, I32 = 2, I64 = 3;
constexpr int U8 = 4, U16 = 5, U32 = 6, U64 = 7;
constexpr int BF16 = 8, F16 = 9, F32 = 10, F64 = 11;
constexpr int BOOL = 12;
constexpr int C32 = 13, C64 = 14, C128 = 15;
constexpr int FP8_E4 = 16, FP8_E5 = 17;

// ═══════════════════════════════════════════════════════════════════════════
// THE LOOKUP TABLE
// ═══════════════════════════════════════════════════════════════════════════
// promotion_table[a][b] = resulting dtype index
// Symmetric: promotion_table[a][b] == promotion_table[b][a]
//
// RULES ENCODED:
// 1. Same type: return that type (diagonal)
// 2. Complex wins: Complex128 > Complex64 > Complex32
// 3. Float + Complex: promote based on float precision
// 4. Float wins: Float64 > Float32 > Float16/BFloat16
// 5. Float16 + BFloat16 → Float32 (different formats)
// 6. Int: Int64 > Int32 > Int16 > Int8
// 7. UInt + UInt: larger wins (UInt64 > UInt32 > UInt16 > UInt8)
// 8. UInt8 + Int* → Int16/Int32/Int64 (safe promotion to larger signed type)
// 9. UInt16/32/64 + Int → NumPy-style (except for uint64 + int8/16/32/64 -> fp32)
// 10. Bool: weakest type
// 11. FP8: special handling (errors for mixed types)
//
// ═══════════════════════════════════════════════════════════════════════════
// NUMPY-STYLE UINT + INT PROMOTION
// ═══════════════════════════════════════════════════════════════════════════
// UInt8 + Int8   → Int16 (need 16 bits to hold both ranges)
// UInt8 + Int16  → Int16
// UInt8 + Int32  → Int32
// UInt8 + Int64  → Int64
//
// UInt16 + Int8  → Int32 (need > 16 bits)
// UInt16 + Int16 → Int32 (need > 16 bits)
// UInt16 + Int32 → Int32
// UInt16 + Int64 → Int64
//
// UInt32 + Int8  → Int64 (need > 32 bits)
// UInt32 + Int16 → Int64
// UInt32 + Int32 → Int64
// UInt32 + Int64 → Int64
//
// UInt64 + Int*  → Float32 (no integer can hold both ranges, using Float32 for DL efficiency)
// ═══════════════════════════════════════════════════════════════════════════

constexpr int promotion_table[DTYPE_COUNT][DTYPE_COUNT] = {
//        I8   I16  I32  I64  U8   U16  U32  U64  BF16 F16  F32  F64  BOOL C32  C64  C128 E4   E5
/*I8  */ {I8,  I16, I32, I64, I16, I32, I64, F32, BF16, F16, F32, F64, I8,  C32, C64, C128, FP8_E4, FP8_E5},
/*I16 */ {I16, I16, I32, I64, I16, I32, I64, F32, BF16, F16, F32, F64, I16, C32, C64, C128, FP8_E4, FP8_E5},
/*I32 */ {I32, I32, I32, I64, I32, I32, I64, F32, BF16, F16, F32, F64, I32, C64, C64, C128, FP8_E4, FP8_E5},
/*I64 */ {I64, I64, I64, I64, I64, I64, I64, F32, BF16, F16, F32, F64, I64, C64, C64, C128, FP8_E4, FP8_E5},
/*U8  */ {I16, I16, I32, I64, U8,  U16, U32, U64, BF16, F16, F32, F64, U8,  C32, C64, C128, FP8_E4, FP8_E5},
/*U16 */ {I32, I32, I32, I64, U16, U16, U32, U64, BF16, F16, F32, F64, U16, C64, C64, C128, FP8_E4, FP8_E5},
/*U32 */ {I64, I64, I64, I64, U32, U32, U32, U64, BF16, F16, F32, F64, U32, C64, C64, C128, FP8_E4, FP8_E5},
/*U64 */ {F32, F32, F32, F32, U64, U64, U64, U64, BF16, F16, F32, F64, U64, C64, C64, C128, FP8_E4, FP8_E5},
/*BF16*/ {BF16, BF16, BF16, BF16, BF16, BF16, BF16, BF16, BF16, F32, F32, F64, BF16, C64, C64, C128, ERR_FP8_HIGHER, ERR_FP8_HIGHER},
/*F16 */ {F16, F16, F16, F16, F16, F16, F16, F16, F32, F16, F32, F64, F16, C32, C64, C128, ERR_FP8_HIGHER, ERR_FP8_HIGHER},
/*F32 */ {F32, F32, F32, F32, F32, F32, F32, F32, F32, F32, F32, F64, F32, C64, C64, C128, ERR_FP8_HIGHER, ERR_FP8_HIGHER},
/*F64 */ {F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, C128, C128, C128, ERR_FP8_HIGHER, ERR_FP8_HIGHER},
/*BOOL*/ {I8,  I16, I32, I64, U8,  U16, U32, U64, BF16, F16, F32, F64, BOOL, C32, C64, C128, FP8_E4, FP8_E5},
/*C32 */ {C32, C32, C64, C64, C32, C64, C64, C64, C64, C32, C64, C128, C32, C32, C64, C128, ERR_FP8_HIGHER, ERR_FP8_HIGHER},
/*C64 */ {C64, C64, C64, C64, C64, C64, C64, C64, C64, C64, C64, C128, C64, C64, C64, C128, ERR_FP8_HIGHER, ERR_FP8_HIGHER},
/*C128*/ {C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, ERR_FP8_HIGHER, ERR_FP8_HIGHER},
/*E4  */ {FP8_E4, FP8_E4, FP8_E4, FP8_E4, FP8_E4, FP8_E4, FP8_E4, FP8_E4, ERR_FP8_HIGHER, ERR_FP8_HIGHER, ERR_FP8_HIGHER, ERR_FP8_HIGHER, FP8_E4, ERR_FP8_HIGHER, ERR_FP8_HIGHER, ERR_FP8_HIGHER, FP8_E4, ERR_MIXED_FP8},
/*E5  */ {FP8_E5, FP8_E5, FP8_E5, FP8_E5, FP8_E5, FP8_E5, FP8_E5, FP8_E5, ERR_FP8_HIGHER, ERR_FP8_HIGHER, ERR_FP8_HIGHER, ERR_FP8_HIGHER, FP8_E5, ERR_FP8_HIGHER, ERR_FP8_HIGHER, ERR_FP8_HIGHER, ERR_MIXED_FP8, FP8_E5}
};

// ═══════════════════════════════════════════════════════════════════════════
// TYPE PROMOTION RULES (Tensor + Tensor) - All dimensions including 0-dim
// ═══════════════════════════════════════════════════════════════════════════
// Uses O(1) lookup table with error handling for special cases.
// Preserves original if-else documentation in comments for clarity.

inline Dtype promote_tensor_ops(Dtype a, Dtype b) {
    // ═══════════════════════════════════════════════════════════════════════════
    // O(1) LOOKUP TABLE IMPLEMENTATION
    // ═══════════════════════════════════════════════════════════════════════════
    // Uses the pre-computed promotion_table for fast dtype resolution.
    // Error markers in the table trigger runtime exceptions for invalid combinations.
    //
    // RULES SUMMARY (encoded in promotion_table above):
    // 1. Same type → return that type (diagonal)
    // 2. Complex wins: Complex128 > Complex64 > Complex32
    // 3. Float + Complex → promote based on float precision
    // 4. Float wins: Float64 > Float32 > (Float16/BFloat16)
    // 5. Float16 + BFloat16 → Float32 (different formats)
    // 6. Int promotion: Int64 > Int32 > Int16 > Int8
    // 7. UInt + UInt → larger wins
    // 8. UInt8 + Int* → Int16/Int32/Int64 (safe promotion to signed)
    // 9. UInt16 + Int* → Int32/Int64 (NumPy-style)
    // 10. UInt32 + Int* → Int64 (NumPy-style)
    // 11. UInt64 + Int* → Float64 (NumPy-style, no integer can hold both ranges)
    // 12. Bool → weakest type (promotes to anything)
    // 13. FP8 + FP8 (different) → ERROR
    // 14. FP8 + Float16/32/64/BFloat16/Complex → ERROR
    // 15. FP8 + Int/Bool → FP8 wins
    // ═══════════════════════════════════════════════════════════════════════════
    
    int idx_a = dtype_to_idx(a);
    int idx_b = dtype_to_idx(b);
    int result = promotion_table[idx_a][idx_b];
    
    // Handle error cases (FP8 only now - UInt+Int uses NumPy-style promotion)
    if (result == ERR_MIXED_FP8) {
        throw std::runtime_error("Input dtypes ('float8_e5m2', 'float8_e4m3fn') have no available implicit dtype promotion path.");
    }
    if (result == ERR_FP8_HIGHER) {
        throw std::runtime_error("8-bit floats do not support implicit promotion with higher precision floats or complex types.");
    }
    
    return static_cast<Dtype>(result);
}

//  NEW: Special promotion for division (always promotes to float)
inline Dtype promote_dtypes_division(Dtype a, Dtype b) {
    //  CRITICAL: Division always promotes to float to avoid integer division issues
    
    // RULE 0: Complex types - division with complex uses normal promotion
    // Complex64 / Int32 → Complex64, Float64 / Complex64 → Complex128, etc.
    if (is_complex(a) || is_complex(b)) {
        return promote_tensor_ops(a, b);
    }
    
    // If either is already float, use highest precision float
    if (a == Dtype::Float64 || b == Dtype::Float64) return Dtype::Float64;
    if (a == Dtype::Float32 || b == Dtype::Float32) return Dtype::Float32;
    if (a == Dtype::Float16 || b == Dtype::Float16) return Dtype::Float16;
    if (a == Dtype::Bfloat16 || b == Dtype::Bfloat16) return Dtype::Bfloat16;
    
    // FP8 handling for division
    bool is_fp8_a = (a == Dtype::Float8_E4M3FN || a == Dtype::Float8_E5M2);
    bool is_fp8_b = (b == Dtype::Float8_E4M3FN || b == Dtype::Float8_E5M2);

    if (is_fp8_a || is_fp8_b) {
        // Both are FP8
        if (is_fp8_a && is_fp8_b) {
            // Same FP8 type -> return that type
            if (a == b) return a;
            // Mixed FP8 types (E4M3 vs E5M2) -> throw error
            throw std::runtime_error("Input dtypes ('float8_e5m2', 'float8_e4m3fn') have no available implicit dtype promotion path. Use explicit x.astype('float32').");
        }
        // FP8 + higher float types -> already handled above (Float64/32/16/BF16 checks)
        // If we reach here, it's FP8 + Int/Bool -> return the FP8 type
        return is_fp8_a ? a : b;
    }
    
    // Otherwise, promote integers and bool to Float32
    // This matches PyTorch's behavior: Int16 / Bool → Float32
    return Dtype::Float32;
}

// ═══════════════════════════════════════════════════════════════════════════
// SCALAR-TENSOR TYPE PROMOTION (PyTorch-style, scalars are weak)
// ═══════════════════════════════════════════════════════════════════════════
// Unlike tensor-tensor promotion, scalars don't upgrade floating tensors
// EXCEPT when the scalar is complex (preserves data).
//
// Verified against PyTorch 2.x behavior:
//   Float16 tensor + float64 scalar → Float16 (tensor wins)
//   Float16 tensor + complex scalar → Complex32 (promotes to preserve complex)
//   Float32 tensor + complex scalar → Complex64 (promotes to preserve complex)
//   Float64 tensor + complex scalar → Complex128 (promotes to preserve complex)
//   Int32 tensor + float scalar → Float32 (default float)
//   Bool tensor + int scalar → scalar_dt (bool is weakest)
//
// ═══════════════════════════════════════════════════════════════════════════
// SCALAR-TENSOR PROMOTION LOOKUP TABLE
// ═══════════════════════════════════════════════════════════════════════════
// scalar_tensor_table[tensor_idx][scalar_idx] = result dtype
// 
// Error markers:
//   ERR_S_UINT_INT = -4   : UInt tensor + Int scalar (signed/unsigned mismatch)
//   ERR_S_UINT_LARGER = -5: UInt tensor + larger UInt scalar
//   ERR_S_INT_LARGER = -6 : Int tensor + larger Int/UInt scalar
// ═══════════════════════════════════════════════════════════════════════════

constexpr int ERR_S_UINT_INT = -4;
constexpr int ERR_S_UINT_LARGER = -5;
constexpr int ERR_S_INT_LARGER = -6;

// Row = Tensor dtype, Col = Scalar dtype
// Tensor:  I8   I16  I32  I64  U8   U16  U32  U64  BF16 F16  F32  F64  BOOL C32  C64  C128 E4   E5
// Scalar:  I8   I16  I32  I64  U8   U16  U32  U64  BF16 F16  F32  F64  BOOL C32  C64  C128 E4   E5

constexpr int scalar_tensor_table[DTYPE_COUNT][DTYPE_COUNT] = {
//              I8          I16          I32          I64         U8           U16          U32          U64         BF16 F16  F32  F64  BOOL C32  C64  C128 E4   E5
/*T:I8  */ {I8,          ERR_S_INT_LARGER, ERR_S_INT_LARGER, ERR_S_INT_LARGER, ERR_S_INT_LARGER, ERR_S_INT_LARGER, ERR_S_INT_LARGER, ERR_S_INT_LARGER, BF16, F16, F32, F32, I8,  C32, C64, C64, FP8_E4,  FP8_E5},
/*T:I16 */ {I16,         I16,          ERR_S_INT_LARGER, ERR_S_INT_LARGER, I16,          ERR_S_INT_LARGER, ERR_S_INT_LARGER, ERR_S_INT_LARGER, BF16, F16, F32, F32, I16, C32, C64, C64, FP8_E4, FP8_E5},
/*T:I32 */ {I32,         I32,          I32,          ERR_S_INT_LARGER, I32,          I32,          ERR_S_INT_LARGER, ERR_S_INT_LARGER, BF16, F32, F32, F32, I32, C64, C64, C64, FP8_E4, FP8_E5},
/*T:I64 */ {I64,         I64,          I64,          I64,          I64,          I64,          I64,          ERR_S_INT_LARGER, F32, F32, F32, F32, I64, C64, C64, C64, FP8_E4, FP8_E5},
/*T:U8  */ {ERR_S_UINT_INT, ERR_S_UINT_INT, ERR_S_UINT_INT, ERR_S_UINT_INT, U8,  ERR_S_UINT_LARGER, ERR_S_UINT_LARGER, ERR_S_UINT_LARGER, BF16, F16, F32, F32, U8,  C32, C64, C64, FP8_E4, FP8_E5},
/*T:U16 */ {ERR_S_UINT_INT, ERR_S_UINT_INT, ERR_S_UINT_INT, ERR_S_UINT_INT, U16, U16,  ERR_S_UINT_LARGER, ERR_S_UINT_LARGER, BF16, F16, F32, F32, U16, C32, C64, C64, FP8_E4, FP8_E5},
/*T:U32 */ {ERR_S_UINT_INT, ERR_S_UINT_INT, ERR_S_UINT_INT, ERR_S_UINT_INT, U32, U32,  U32,  ERR_S_UINT_LARGER, BF16, F32, F32, F32, U32, C64, C64, C64,FP8_E4,FP8_E5},
/*T:U64 */ {ERR_S_UINT_INT, ERR_S_UINT_INT, ERR_S_UINT_INT, ERR_S_UINT_INT, U64, U64,  U64,  U64,  F32, F32, F32, F32, U64, C64, C64, C64, FP8_E4, FP8_E5},
/*T:BF16*/ {BF16, BF16, BF16, BF16, BF16, BF16, BF16, BF16, BF16, BF16, BF16, BF16, BF16, C64, C64, C64, BF16, BF16},
/*T:F16 */ {F16, F16, F16, F16, F16, F16, F16, F16, F16, F16, F16, F16, F16, C32, C64, C64, F16, F16},
/*T:F32 */ {F32, F32, F32, F32, F32, F32, F32, F32, F32, F32, F32, F32, F32, C64, C64, C64, F32, F32},
/*T:F64 */ {F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, C128, C128, C128, F64, F64},
/*T:BOOL*/ {I8,  I16, I32, I64, U8,  U16, U32, U64, BF16, F16, F32, F32, BOOL, C32, C64, C64, FP8_E4, FP8_E5},
/*T:C32 */ {C32, C32, C32, C32, C32, C32, C32, C32, C32, C32, C32, C32, C32, C32, C32, C32, C32, C32},
/*T:C64 */ {C64, C64, C64, C64, C64, C64, C64, C64, C64, C64, C64, C64, C64, C64, C64, C64, C64, C64},
/*T:C128*/ {C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128},
/*T:E4  */ {FP8_E4, FP8_E4, FP8_E4, FP8_E4, FP8_E4, FP8_E4, FP8_E4, FP8_E4, FP8_E4, FP8_E4, FP8_E4, FP8_E4, FP8_E4, C32, C64, C64, FP8_E4, FP8_E4},
/*T:E5  */ {FP8_E5, FP8_E5, FP8_E5, FP8_E5, FP8_E5, FP8_E5, FP8_E5, FP8_E5, FP8_E5, FP8_E5, FP8_E5, FP8_E5, FP8_E5, C32, C64, C64, FP8_E5, FP8_E5}
};

inline Dtype promote_scalar_ops(Dtype tensor_dt, Dtype scalar_dt) {
    // ═══════════════════════════════════════════════════════════════════════════
    // O(1) LOOKUP TABLE IMPLEMENTATION FOR SCALAR-TENSOR PROMOTION
    // ═══════════════════════════════════════════════════════════════════════════
    // Uses scalar_tensor_table[tensor_idx][scalar_idx] for fast lookup.
    // Error markers trigger runtime exceptions for invalid combinations.
    //
    // KEY RULES:
    // 1. Complex tensor → tensor wins (always)
    // 2. Float tensor + complex scalar → promote to Complex32/64/128
    // 3. Float tensor + non-complex → tensor wins (scalar is weak)
    // 4. UInt tensor + Int scalar → ERROR (signed/unsigned mismatch)
    // 5. UInt tensor + larger UInt scalar → ERROR
    // 6. Int tensor + larger Int/UInt scalar → ERROR
    // 7. Bool tensor → scalar_dt wins (bool is weakest type)
    // ═══════════════════════════════════════════════════════════════════════════
    
    int t_idx = dtype_to_idx(tensor_dt);
    int s_idx = dtype_to_idx(scalar_dt);
    int result = scalar_tensor_table[t_idx][s_idx];
    
    // Handle error cases
    if (result == ERR_S_UINT_INT) {
        throw std::runtime_error("UInt tensor + int scalar: not supported (signed/unsigned mismatch). Cast tensor first.");
    }
    if (result == ERR_S_UINT_LARGER) {
        throw std::runtime_error("UInt tensor + larger uint scalar: not supported. Cast tensor first.");
    }
    if (result == ERR_S_INT_LARGER) {
        throw std::runtime_error("Int tensor + larger int/uint scalar: not supported. Cast tensor first.");
    }
    
    return static_cast<Dtype>(result);
}

} // namespace OwnTensor

#endif // DTYPE_TRAIT_H


// ============================================================
// Tensor + 0-dim Tensor
// ============================================================
// -------------------------------------------------------
// Tensor dtype    | 0-dim dtype  | Result dtype   
// -------------------------------------------------------
// bool            | int8         | int8
// bool            | int16        | int16
// bool            | int32        | int32
// bool            | int64        | int64
// bool            | float16      | float16
// bool            | float32      | float32
// bool            | float64      | float64
// bool            | complex32    | complex32
// bool            | complex64    | complex64
// bool            | complex128   | complex128
// int8            | int8         | int8
// int8            | int16        | int8
// int8            | int32        | int8
// int8            | int64        | int8
// int8            | float16      | float16
// int8            | float32      | float32
// int8            | float64      | float64
// int8            | complex32    | complex32
// int8            | complex64    | complex64
// int8            | complex128   | complex128
// int16           | int8         | int16
// int16           | int16        | int16
// int16           | int32        | int16
// int16           | int64        | int16
// int16           | float16      | float16
// int16           | float32      | float32
// int16           | float64      | float64
// int16           | complex32    | complex32
// int16           | complex64    | complex64
// int16           | complex128   | complex128
// int32           | int8         | int32
// int32           | int16        | int32
// int32           | int32        | int32
// int32           | int64        | int32
// int32           | float16      | float16
// int32           | float32      | float32
// int32           | float64      | float64
// int32           | complex32    | complex32
// int32           | complex64    | complex64
// int32           | complex128   | complex128
// int64           | int8         | int64
// int64           | int16        | int64
// int64           | int32        | int64
// int64           | int64        | int64
// int64           | float16      | float16
// int64           | float32      | float32
// int64           | float64      | float64
// int64           | complex32    | complex32
// int64           | complex64    | complex64
// int64           | complex128   | complex128
// float16         | int8         | float16
// float16         | int16        | float16
// float16         | int32        | float16
// float16         | int64        | float16
// float16         | float16      | float16
// float16         | float32      | float16
// float16         | float64      | float16
// float16         | complex32    | complex32
// float16         | complex64    | complex32
// float16         | complex128   | complex32
// bfloat16        | int8         | bfloat16
// bfloat16        | int16        | bfloat16
// bfloat16        | int32        | bfloat16
// bfloat16        | int64        | bfloat16
// bfloat16        | float16      | bfloat16
// bfloat16        | float32      | bfloat16
// bfloat16        | float64      | bfloat16
// bfloat16        | complex32    | complex64
// bfloat16        | complex64    | complex64
// bfloat16        | complex128   | complex64
// float32         | int8         | float32
// float32         | int16        | float32
// float32         | int32        | float32
// float32         | int64        | float32
// float32         | float16      | float32
// float32         | float32      | float32
// float32         | float64      | float32
// float32         | complex32    | complex64
// float32         | complex64    | complex64
// float32         | complex128   | complex64
// float64         | int8         | float64
// float64         | int16        | float64
// float64         | int32        | float64
// float64         | int64        | float64
// float64         | float16      | float64
// float64         | float32      | float64
// float64         | float64      | float64
// float64         | complex32    | complex128
// float64         | complex64    | complex128
// float64         | complex128   | complex128
// complex32       | int8         | complex32
// complex32       | int16        | complex32
// complex32       | int32        | complex32
// complex32       | int64        | complex32
// complex32       | float16      | complex32
// complex32       | float32      | complex32
// complex32       | float64      | complex32
// complex32       | complex32    | complex32
// complex32       | complex64    | complex32
// complex32       | complex128   | complex32
// complex64       | int8         | complex64
// complex64       | int16        | complex64
// complex64       | int32        | complex64
// complex64       | int64        | complex64
// complex64       | float16      | complex64
// complex64       | float32      | complex64
// complex64       | float64      | complex64
// complex64       | complex32    | complex64
// complex64       | complex64    | complex64
// complex64       | complex128   | complex64
// complex128      | int8         | complex128
// complex128      | int16        | complex128
// complex128      | int32        | complex128
// complex128      | int64        | complex128
// complex128      | float16      | complex128
// complex128      | float32      | complex128
// complex128      | float64      | complex128
// complex128      | complex32    | complex128
// complex128      | complex64    | complex128
// complex128      | complex128   | complex128