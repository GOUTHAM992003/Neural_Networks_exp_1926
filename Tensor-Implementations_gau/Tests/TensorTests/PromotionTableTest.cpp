// ═══════════════════════════════════════════════════════════════════════════
// COMPREHENSIVE TENSOR DTYPE PROMOTION TEST
// ═══════════════════════════════════════════════════════════════════════════
// Tests all 18x18 = 324 combinations of dtype promotion
// Validates that the lookup table returns expected results
// ═══════════════════════════════════════════════════════════════════════════

#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include "dtype/DtypeTraits.h"

using namespace OwnTensor;
using namespace std;

// ═══════════════════════════════════════════════════════════════════════════
// EXPECTED RESULTS TABLE (for validation)
// ═══════════════════════════════════════════════════════════════════════════
// -1 = ERR_MIXED_FP8, -2 = ERR_FP8_HIGHER, -3 = ERR_UINT_SIGNED

const int EXPECTED[18][18] = {
//        I8   I16  I32  I64  U8   U16  U32  U64  BF16 F16  F32  F64  BOOL C32  C64  C128 E4   E5
/*I8  */ {0,   1,   2,   3,   1,   2,   3,   10,  8,   9,   10,  11,  0,   13,  14,  15,  16,  17},
/*I16 */ {1,   1,   2,   3,   1,   2,   3,   10,  8,   9,   10,  11,  1,   13,  14,  15,  16,  17},
/*I32 */ {2,   2,   2,   3,   2,   2,   3,   10,  8,   9,   10,  11,  2,   14,  14,  15,  16,  17},
/*I64 */ {3,   3,   3,   3,   3,   3,   3,   10,  8,   9,   10,  11,  3,   14,  14,  15,  16,  17},
/*U8  */ {1,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  4,   13,  14,  15,  16,  17},
/*U16 */ {2,   2,   2,   3,   5,   5,   6,   7,   8,   9,   10,  11,  5,   14,  14,  15,  16,  17},
/*U32 */ {3,   3,   3,   3,   6,   6,   6,   7,   8,   9,   10,  11,  6,   14,  14,  15,  16,  17},
/*U64 */ {10,  10,  10,  10,  7,   7,   7,   7,   8,   9,   10,  11,  7,   14,  14,  15,  16,  17},
/*BF16*/ {8,   8,   8,   8,   8,   8,   8,   8,   8,   10,  10,  11,  8,   14,  14,  15,  -2,  -2},
/*F16 */ {9,   9,   9,   9,   9,   9,   9,   9,   10,  9,   10,  11,  9,   13,  14,  15,  -2,  -2},
/*F32 */ {10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  11,  10,  14,  14,  15,  -2,  -2},
/*F64 */ {11,  11,  11,  11,  11,  11,  11,  11,  11,  11,  11,  11,  11,  15,  15,  15,  -2,  -2},
/*BOOL*/ {0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17},
/*C32 */ {13,  13,  14,  14,  13,  14,  14,  14,  14,  13,  14,  15,  13,  13,  14,  15,  -2,  -2},
/*C64 */ {14,  14,  14,  14,  14,  14,  14,  14,  14,  14,  14,  15,  14,  14,  14,  15,  -2,  -2},
/*C128*/ {15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  -2,  -2},
/*E4  */ {16,  16,  16,  16,  16,  16,  16,  16,  -2,  -2,  -2,  -2,  16,  -2,  -2,  -2,  16,  -1},
/*E5  */ {17,  17,  17,  17,  17,  17,  17,  17,  -2,  -2,  -2,  -2,  17,  -2,  -2,  -2,  -1,  17}
};

// Dtype names for printing
const char* DTYPE_NAMES[18] = {
    "Int8", "Int16", "Int32", "Int64",
    "UInt8", "UInt16", "UInt32", "UInt64",
    "BFloat16", "Float16", "Float32", "Float64",
    "Bool",
    "Complex32", "Complex64", "Complex128",
    "Float8_E4M3FN", "Float8_E5M2"
};

// All dtypes in order
const Dtype ALL_DTYPES[18] = {
    Dtype::Int8, Dtype::Int16, Dtype::Int32, Dtype::Int64,
    Dtype::UInt8, Dtype::UInt16, Dtype::UInt32, Dtype::UInt64,
    Dtype::Bfloat16, Dtype::Float16, Dtype::Float32, Dtype::Float64,
    Dtype::Bool,
    Dtype::Complex32, Dtype::Complex64, Dtype::Complex128,
    Dtype::Float8_E4M3FN, Dtype::Float8_E5M2
};

int main() {
    cout << "═══════════════════════════════════════════════════════════════════" << endl;
    cout << "  DTYPE PROMOTION TABLE TEST - Testing all 324 combinations" << endl;
    cout << "═══════════════════════════════════════════════════════════════════" << endl << endl;
    
    int passed = 0;
    int failed = 0;
    vector<string> failures;
    
    for (int i = 0; i < 18; i++) {
        for (int j = 0; j < 18; j++) {
            Dtype a = ALL_DTYPES[i];
            Dtype b = ALL_DTYPES[j];
            int expected = EXPECTED[i][j];
            
            bool threw_error = false;
            int actual = -999;
            string error_type = "";
            
            try {
                Dtype result = promote_tensor_ops(a, b);
                actual = static_cast<int>(result);
            } catch (const std::runtime_error& e) {
                threw_error = true;
                string msg = e.what();
                if (msg.find("float8_e5m2") != string::npos || msg.find("float8_e4m3fn") != string::npos) {
                    actual = -1;  // ERR_MIXED_FP8
                    error_type = "ERR_MIXED_FP8";
                } else if (msg.find("8-bit floats") != string::npos) {
                    actual = -2;  // ERR_FP8_HIGHER
                    error_type = "ERR_FP8_HIGHER";
                } else if (msg.find("signed/unsigned") != string::npos) {
                    actual = -3;  // ERR_UINT_SIGNED
                    error_type = "ERR_UINT_SIGNED";
                } else {
                    actual = -999;  // Unknown error
                    error_type = "UNKNOWN";
                }
            }
            
            bool test_passed = (actual == expected);
            
            if (test_passed) {
                passed++;
            } else {
                failed++;
                string failure_msg = string(DTYPE_NAMES[i]) + " + " + DTYPE_NAMES[j] + 
                    " : expected " + to_string(expected) + 
                    ", got " + to_string(actual);
                if (threw_error) {
                    failure_msg += " (" + error_type + ")";
                }
                failures.push_back(failure_msg);
            }
        }
    }
    
    // Print summary
    cout << "Results: " << passed << " passed, " << failed << " failed" << endl << endl;
    
    if (failed > 0) {
        cout << "═══════════════════════════════════════════════════════════════════" << endl;
        cout << "  FAILURES:" << endl;
        cout << "═══════════════════════════════════════════════════════════════════" << endl;
        for (const auto& f : failures) {
            cout << "  ❌ " << f << endl;
        }
    } else {
        cout << "✅ ALL TESTS PASSED!" << endl;
    }
    
    // Print the full promotion matrix for reference
    cout << endl;
    cout << "═══════════════════════════════════════════════════════════════════" << endl;
    cout << "  PROMOTION MATRIX (actual results):" << endl;
    cout << "═══════════════════════════════════════════════════════════════════" << endl;
    cout << "         ";
    for (int j = 0; j < 18; j++) {
        cout << setw(6) << DTYPE_NAMES[j] << " ";
    }
    cout << endl;
    
    for (int i = 0; i < 18; i++) {
        cout << setw(8) << DTYPE_NAMES[i] << " ";
        for (int j = 0; j < 18; j++) {
            Dtype a = ALL_DTYPES[i];
            Dtype b = ALL_DTYPES[j];
            try {
                Dtype result = promote_tensor_ops(a, b);
                int idx = static_cast<int>(result);
                cout << setw(6) << idx << " ";
            } catch (...) {
                cout << setw(6) << "ERR" << " ";
            }
        }
        cout << endl;
    }
    
    return failed > 0 ? 1 : 0;
}
