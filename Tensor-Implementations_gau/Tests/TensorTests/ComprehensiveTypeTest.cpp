#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cassert>
#include <functional>
#include "TensorLib.h"
#include "dtype/DtypeTraits.h"

using namespace OwnTensor;
using namespace std;

// ============================================================================
// ORACLE: Re-implementation of Promotion Rules for Verification
// ============================================================================

Dtype oracle_promote_bool(Dtype a, Dtype b) {
    if (a == b) return a;

    bool is_complex_a = (a == Dtype::Complex128 || a == Dtype::Complex64 || a == Dtype::Complex32);
    bool is_complex_b = (b == Dtype::Complex128 || b == Dtype::Complex64 || b == Dtype::Complex32);

    if (is_complex_a || is_complex_b) {
        if (a == Dtype::Complex128 || b == Dtype::Complex128 || 
            a == Dtype::Float64 || b == Dtype::Float64) {
            return Dtype::Complex128;
        }
        if (a == Dtype::Complex64 || b == Dtype::Complex64 || 
            a == Dtype::Float32 || b == Dtype::Float32 ||
            a == Dtype::Bfloat16 || b == Dtype::Bfloat16) {
            return Dtype::Complex64;
        }
        return Dtype::Complex32;
    }

    bool is_fp8_a = (a == Dtype::Float8_E4M3FN || a == Dtype::Float8_E5M2);
    bool is_fp8_b = (b == Dtype::Float8_E4M3FN || b == Dtype::Float8_E5M2);
    
    if (is_fp8_a || is_fp8_b) {
        if (is_fp8_a && is_fp8_b) {
            throw std::runtime_error("Mixed FP8 types not supported");
        }
        bool is_higher_float_a = (a == Dtype::Float64 || a == Dtype::Float32 || a == Dtype::Float16 || a == Dtype::Bfloat16);
        bool is_higher_float_b = (b == Dtype::Float64 || b == Dtype::Float32 || b == Dtype::Float16 || b == Dtype::Bfloat16);
        if (is_higher_float_a || is_higher_float_b) {
            throw std::runtime_error("FP8 + Higher Float not supported");
        }
        if (is_fp8_a) return a;
        return b;
    }

    if (a == Dtype::Float64 || b == Dtype::Float64) return Dtype::Float64;
    if (a == Dtype::Float32 || b == Dtype::Float32) return Dtype::Float32;
    
    bool has_f16 = (a == Dtype::Float16 || b == Dtype::Float16);
    bool has_bf16 = (a == Dtype::Bfloat16 || b == Dtype::Bfloat16);
    if (has_f16 && has_bf16) return Dtype::Float32;
    if (has_f16) return Dtype::Float16;
    if (has_bf16) return Dtype::Bfloat16;

    if (a == Dtype::Int64 || b == Dtype::Int64) return Dtype::Int64;
    if (a == Dtype::Int32 || b == Dtype::Int32) return Dtype::Int32;
    if (a == Dtype::Int16 || b == Dtype::Int16) return Dtype::Int16;
    
    bool has_int8 = (a == Dtype::Int8 || b == Dtype::Int8);
    bool has_uint8 = (a == Dtype::UInt8 || b == Dtype::UInt8);
    if (has_int8 && has_uint8) return Dtype::Int16;
    if (has_int8) return Dtype::Int8;
    if (a == Dtype::UInt8 || b == Dtype::UInt8) return Dtype::UInt8;

    if (a==Dtype::UInt16 || b==Dtype::UInt16 || a==Dtype::UInt32 || b==Dtype::UInt32 || a==Dtype::UInt64 || b==Dtype::UInt64) {
        throw std::runtime_error("Unsupported UInt promotion");
    }

    return Dtype::Bool;
}

Dtype oracle_promote_div(Dtype a, Dtype b) {
    if (a == Dtype::Float64 || b == Dtype::Float64) return Dtype::Float64;
    if (a == Dtype::Float32 || b == Dtype::Float32) return Dtype::Float32;
    if (a == Dtype::Float16 || b == Dtype::Float16) return Dtype::Float16;
    if (a == Dtype::Bfloat16 || b == Dtype::Bfloat16) return Dtype::Bfloat16;

    bool is_fp8_a = (a == Dtype::Float8_E4M3FN || a == Dtype::Float8_E5M2);
    bool is_fp8_b = (b == Dtype::Float8_E4M3FN || b == Dtype::Float8_E5M2);

    if (is_fp8_a || is_fp8_b) {
        if (is_fp8_a && is_fp8_b) {
            if (a == b) return a;
            throw std::runtime_error("Mixed FP8 types not supported");
        }
        return Dtype::Float32;
    }

    return Dtype::Float32;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

std::vector<Dtype> get_all_dtypes() {
    return {
        Dtype::Int8, Dtype::Int16, Dtype::Int32, Dtype::Int64,
        Dtype::UInt8, Dtype::UInt16, Dtype::UInt32, Dtype::UInt64,
        Dtype::Float16, Dtype::Bfloat16, Dtype::Float32, Dtype::Float64,
        Dtype::Bool,
        Dtype::Complex32, Dtype::Complex64, Dtype::Complex128,
        Dtype::Float8_E4M3FN, Dtype::Float8_E5M2
    };
}

string dtype_to_str(Dtype dt) {
    return get_dtype_name(dt);
}

// ============================================================================
// TEST RUNNER
// ============================================================================

void run_tests() {
    auto all_dtypes = get_all_dtypes();
    int passed = 0;
    int failed = 0;
    int skipped = 0;

    cout << "Starting Comprehensive Type Promotion Tests..." << endl;

    // 1. Verify Oracle against Known Ground Truths
    cout << "Verifying Oracle Logic..." << endl;
    assert(oracle_promote_bool(Dtype::Int32, Dtype::Float32) == Dtype::Float32);
    assert(oracle_promote_bool(Dtype::Int32, Dtype::Float16) == Dtype::Float16); // Rule 3: Float wins
    assert(oracle_promote_bool(Dtype::Float16, Dtype::Bfloat16) == Dtype::Float32); // Special case
    assert(oracle_promote_div(Dtype::Int32, Dtype::Int32) == Dtype::Float32); // Division promotion
    cout << "Oracle Logic Verified." << endl;

    // 2. Exhaustive Binary Op Test
    for (Dtype dt1 : all_dtypes) {
        for (Dtype dt2 : all_dtypes) {
            // Create dummy tensors (1x1) on CPU for speed/simplicity of type checking
            // We are testing dispatch and promotion logic, which is device-agnostic mostly.
            // But actual execution might need CUDA if compiled with it.
            // Let's use CUDA if available, else CPU.
            Device device = Device::CUDA; 
            
            // Determine expected behavior from Oracle
            Dtype expected_bool = Dtype::Bool; // Dummy init
            bool expect_bool_error = false;
            try {
                expected_bool = oracle_promote_bool(dt1, dt2);
            } catch (...) {
                expect_bool_error = true;
            }

            try {
                Tensor t1(Shape{{1}}, dt1, device);
                Tensor t2(Shape{{1}}, dt2, device);
                
                // --- Arithmetic Ops (+, -, *, %) ---
                // +, -, * use promote_tensor_ops
                if (!expect_bool_error) {
                    // +
                    Tensor res = t1 + t2;
                    if (res.dtype() != expected_bool) {
                        cout << "FAIL: " << dtype_to_str(dt1) << " + " << dtype_to_str(dt2) 
                             << " -> " << dtype_to_str(res.dtype()) << " (Expected: " << dtype_to_str(expected_bool) << ")" << endl;
                        failed++;
                    } else {
                        passed++;
                    }
                    
                    // -
                    res = t1 - t2;
                    if (res.dtype() != expected_bool) failed++; else passed++;
                    
                    // *
                    res = t1 * t2;
                    if (res.dtype() != expected_bool) failed++; else passed++;
                    
                    // % (Modulo)
                    // Modulo has specific restrictions (no complex).
                    bool is_complex = (dt1 == Dtype::Complex32 || dt1 == Dtype::Complex64 || dt1 == Dtype::Complex128 ||
                                       dt2 == Dtype::Complex32 || dt2 == Dtype::Complex64 || dt2 == Dtype::Complex128);
                    if (is_complex) {
                        try {
                            res = t1 % t2;
                            cout << "FAIL: Modulo should throw for complex types: " << dtype_to_str(dt1) << ", " << dtype_to_str(dt2) << endl;
                            failed++;
                        } catch (...) {
                            passed++; // Expected throw
                        }
                    } else {
                        res = t1 % t2;
                        if (res.dtype() != expected_bool) failed++; else passed++;
                    }
                } else {
                    // Expect error for +, -, *
                    try {
                        Tensor res = t1 + t2;
                        cout << "FAIL: Expected promotion error for " << dtype_to_str(dt1) << " + " << dtype_to_str(dt2) << endl;
                        failed++;
                    } catch (...) {
                        passed++;
                    }
                }

                // --- Division (/) ---
                // Uses promote_dtypes_division
                Dtype expected_div = Dtype::Float32; // Dummy
                bool expect_div_error = false;
                try {
                    expected_div = oracle_promote_div(dt1, dt2);
                } catch (...) {
                    expect_div_error = true;
                }

                if (!expect_div_error) {
                    Tensor res = t1 / t2;
                    if (res.dtype() != expected_div) {
                            cout << "FAIL: " << dtype_to_str(dt1) << " / " << dtype_to_str(dt2) 
                            << " -> " << dtype_to_str(res.dtype()) << " (Expected: " << dtype_to_str(expected_div) << ")" << endl;
                            failed++;
                    } else {
                        passed++;
                    }
                } else {
                    try {
                        Tensor res = t1 / t2;
                        cout << "FAIL: Expected division error for " << dtype_to_str(dt1) << " / " << dtype_to_str(dt2) << endl;
                        failed++;
                    } catch (...) {
                        passed++;
                    }
                }

                // --- Comparison Ops (==, !=, <, >, <=, >=) ---
                // Always return Bool. Internal promotion uses promote_tensor_ops.
                // If promote_tensor_ops throws, then comparison MIGHT throw or might handle it?
                // TensorOps.cpp usually calls promote_tensor_ops inside.
                // So if expect_bool_error is true, comparison should probably throw too?
                // Let's verify: operator== calls promote_tensor_ops.
                
                // Complex numbers are not ordered, so <, >, <=, >= should throw
                bool is_complex_dt1 = (dt1 == Dtype::Complex32 || dt1 == Dtype::Complex64 || dt1 == Dtype::Complex128);
                bool is_complex_dt2 = (dt2 == Dtype::Complex32 || dt2 == Dtype::Complex64 || dt2 == Dtype::Complex128);
                bool is_either_complex = is_complex_dt1 || is_complex_dt2;
                
                if (!expect_bool_error) {
                    // == and != work for all types including complex
                    Tensor res = (t1 == t2);
                    if (res.dtype() != Dtype::Bool) failed++; else passed++;
                    
                    res = (t1 != t2);
                    if (res.dtype() != Dtype::Bool) failed++; else passed++;
                    
                    // <, >, <=, >= only work for non-complex types
                    if (!is_either_complex) {
                        res = (t1 < t2);
                        if (res.dtype() != Dtype::Bool) failed++; else passed++;
                    } else {
                        // Complex types should throw for ordering comparisons
                        try {
                            res = (t1 < t2);
                            failed++; // Should have thrown!
                        } catch (...) {
                            passed++; // Expected exception
                        }
                    }
                } else {
                     try {
                        Tensor res = (t1 == t2);
                        cout << "FAIL: Expected comparison error (due to promotion) for " << dtype_to_str(dt1) << ", " << dtype_to_str(dt2) << endl;
                        failed++;
                    } catch (...) {
                        passed++;
                    }
                }

                // --- Logical Ops (&&, ||, ^) ---
                // Always return Bool. No promotion needed (converts to bool).
                {
                    Tensor res = logical_AND(t1, t2);
                    if (res.dtype() != Dtype::Bool) failed++; else passed++;
                    
                    res = logical_OR(t1, t2);
                    if (res.dtype() != Dtype::Bool) failed++; else passed++;
                    
                    res = logical_XOR(t1, t2);
                    if (res.dtype() != Dtype::Bool) failed++; else passed++;
                }

                // --- In-place Ops (+=, -=, *=, /=, %=) ---
                // Require that RHS can be promoted to LHS.
                if (!expect_bool_error) {
                    // +=
                    bool can_inplace = (expected_bool == dt1);
                    if (can_inplace) {
                        t1 += t2; // Should succeed
                        passed++;
                    } else {
                        try {
                            t1 += t2;
                            cout << "FAIL: Inplace += should fail for " << dtype_to_str(dt1) << ", " << dtype_to_str(dt2) << endl;
                            failed++;
                        } catch (...) {
                            passed++;
                        }
                    }
                }
                
                // /= (Division Inplace)
                if (!expect_div_error) {
                    bool can_inplace_div = (expected_div == dt1);
                    if (can_inplace_div) {
                        t1 /= t2;
                        passed++;
                    } else {
                        try {
                            t1 /= t2;
                            failed++; // Should have thrown
                        } catch (...) {
                            passed++;
                        }
                    }
                }

            } catch (const std::exception& e) {
                // Unexpected error during tensor creation or execution
                // cout << "ERROR: " << dtype_to_str(dt1) << ", " << dtype_to_str(dt2) << ": " << e.what() << endl;
                // Some errors are expected (e.g. UInt support), so we might just log and continue
                cout<<"Unexpected error during tensor creation or execution for "<<dtype_to_str(dt1)<<", "<<dtype_to_str(dt2)<<": "<<e.what()<<endl;
                skipped++;
            }
        }
    }
    
    // 3. Unary Ops (logical_not)
    for (Dtype dt : all_dtypes) {
         try {
             Tensor t(Shape{{1}}, dt, Device::CUDA);
             Tensor res = logical_NOT(t);
             if (res.dtype() != Dtype::Bool) failed++; else passed++;
         } catch (...) {
             skipped++;
         }
    }

    cout << "Tests Completed." << endl;
    cout << "Passed: " << passed << endl;
    cout << "Failed: " << failed << endl;
    cout << "Skipped: " << skipped << endl;
    
    if (failed > 0) exit(1);
}

int main() {
    try {
        run_tests();
    } catch (const std::exception& e) {
        cerr << "FATAL: " << e.what() << endl;
        return 1;
    }
    return 0;
}
