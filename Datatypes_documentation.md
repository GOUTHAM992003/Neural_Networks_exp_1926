# Datatypes — Complete Documentation
## UInt8, Bool, FP4 (E2M1), Complex Numbers & The Dtype Infrastructure

---

## 0. The Dtype System — Introduction

The OwnTensor library uses a unified type system built on a C++ `enum class` that defines **18 data types**. Every tensor has exactly one `Dtype`, which determines how its raw memory bytes are interpreted.

### The Dtype Enum

```cpp
enum class Dtype {
    // Signed Integers (4)
    Int8, Int16, Int32, Int64,
    // Unsigned Integers (4)
    UInt8, UInt16, UInt32, UInt64,
    // Floating Point (6)
    Bfloat16, Float16, Float32, Float64, Float4_e2m1, Float4_e2m1_2x,
    // Boolean (1)
    Bool,
    // Complex (3)
    Complex32, Complex64, Complex128
};
```

### Complete Type Reference Table

| Dtype | C++ Type | Size (bytes) | Category | Signed | Range |
|-------|----------|:---:|----------|:---:|-------|
| `Int8` | `int8_t` | 1 | Integer | ✅ | -128 to 127 |
| `Int16` | `int16_t` | 2 | Integer | ✅ | -32,768 to 32,767 |
| `Int32` | `int32_t` | 4 | Integer | ✅ | -2.1×10⁹ to 2.1×10⁹ |
| `Int64` | `int64_t` | 8 | Integer | ✅ | -9.2×10¹⁸ to 9.2×10¹⁸ |
| `UInt8` | `uint8_t` | 1 | Unsigned Integer | ❌ | 0 to 255 |
| `UInt16` | `uint16_t` | 2 | Unsigned Integer | ❌ | 0 to 65,535 |
| `UInt32` | `uint32_t` | 4 | Unsigned Integer | ❌ | 0 to ~4.29×10⁹ |
| `UInt64` | `uint64_t` | 8 | Unsigned Integer | ❌ | 0 to ~1.84×10¹⁹ |
| `Bfloat16` | `bfloat16_t` | 2 | Float | ✅ | ±3.39×10³⁸ (~2-3 digits) |
| `Float16` | `float16_t` | 2 | Float | ✅ | ±65,504 (~3-4 digits) |
| `Float32` | `float` | 4 | Float | ✅ | ±3.4×10³⁸ (~7 digits) |
| `Float64` | `double` | 8 | Float | ✅ | ±1.8×10³⁰⁸ (~15 digits) |
| `Float4_e2m1` | `float4_e2m1_t` | 1 | Float (4-bit) | ✅ | {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6} |
| `Float4_e2m1_2x` | `float4_e2m1_2x_t` | 1 | Float (packed) | ✅ | 2 × FP4 values in 1 byte |
| `Bool` | `bool` | 1 | Boolean | — | true / false |
| `Complex32` | `complex32_t` | 4 | Complex | ✅ | float16 real + float16 imag |
| `Complex64` | `complex64_t` | 8 | Complex | ✅ | float32 real + float32 imag |
| `Complex128` | `complex128_t` | 16 | Complex | ✅ | float64 real + float64 imag |

---

## 1. Algorithm Design

### 1.1 Bool Type

**Storage:** `bool` uses C++'s native `bool` type — 1 byte per element (not 1 bit!).

**How Bool integrates with operations:**
- **Creation:** `Tensor({3,3}, Dtype::Bool)` creates a tensor of booleans
- **`reduce_all` / `reduce_any`:** Use `AllOp` / `AnyOp` operator structs
  - `AllOp::identity()` = `true`, `AllOp::reduce(a,b)` = `a && b`
  - `AnyOp::identity()` = `false`, `AnyOp::reduce(a,b)` = `a || b`
- **Non-bool input handling:** If a non-bool tensor is passed to `reduce_all`/`reduce_any`, it is auto-converted to bool first (any nonzero value → `true`)
- **Arithmetic operations on Bool:** When bool tensors are used in arithmetic (sum, mean, etc.), bool values are treated as `0`/`1` integers. Sum of a bool tensor counts the number of `true` values.

### 1.2 FP4 E2M1 — Lookup Table Conversion (NOT Bit Manipulation!)

Unlike FP16/BF16 which use bit manipulation, FP4 uses a **fixed lookup table** with only 16 possible values. The format is E2M1: 2 exponent bits + 1 mantissa bit + 1 sign bit = 4 bits total.

**The complete value table:**

| 4-bit Code | Binary | Value | | 4-bit Code | Binary | Value |
|:---:|:---:|:---:|---|:---:|:---:|:---:|
| 0 | `0000` | 0.0 | | 8 | `1000` | -0.0 |
| 1 | `0001` | 0.5 | | 9 | `1001` | -0.5 |
| 2 | `0010` | 1.0 | | 10 | `1010` | -1.0 |
| 3 | `0011` | 1.5 | | 11 | `1011` | -1.5 |
| 4 | `0100` | 2.0 | | 12 | `1100` | -2.0 |
| 5 | `0101` | 3.0 | | 13 | `1101` | -3.0 |
| 6 | `0110` | 4.0 | | 14 | `1110` | -4.0 |
| 7 | `0111` | 6.0 | | 15 | `1111` | -6.0 |

> **No Infinity, No NaN** — FP4 has no special values. NaN inputs are mapped to the maximum value (±6.0).

**Float → FP4 conversion (`float_to_fp4_e2m1`):**

Uses **nearest-neighbor rounding** with midpoint thresholds — NOT bit manipulation:
```
Input: f (float)

Step 1: Extract sign → sign_bit = (f < 0) ? 8 : 0

Step 2: Take abs(f) and find nearest FP4 value using thresholds:
        if abs_f < 0.25 → code 0 (maps to 0.0)
        if abs_f < 0.75 → code 1 (maps to 0.5)     ← midpoint of 0.0 and 0.5 = 0.25
        if abs_f < 1.25 → code 2 (maps to 1.0)     ← midpoint of 0.5 and 1.0 = 0.75
        if abs_f < 1.75 → code 3 (maps to 1.5)     ← midpoint of 1.0 and 1.5 = 1.25
        if abs_f < 2.25 → code 4 (maps to 2.0)     ← midpoint of 1.5 and 2.0 = 1.75
        if abs_f < 3.50 → code 5 (maps to 3.0)     ← midpoint of 2.0 and 3.0 = 2.5
        if abs_f < 5.00 → code 6 (maps to 4.0)     ← midpoint of 3.0 and 4.0 = 3.5
        else            → code 7 (maps to 6.0)      ← midpoint of 4.0 and 6.0 = 5.0

Step 3: return sign_bit | code

Note: Values > 6.0 are clamped to 6.0 (no overflow to infinity)
```

**FP4 → Float conversion (`fp4_e2m1_to_float`):**

Simple switch statement — the 4-bit code is used directly as an index:
```
sign = (val >= 8) ? -1.0 : 1.0
magnitude = val & 7   (strip sign bit)
result = lookup[magnitude] * sign
```

### 1.3 FP4 Packed Format (`float4_e2m1_2x_t`)

Two FP4 values are packed into a single byte for memory efficiency:

```
Byte layout:
┌────────────┬────────────┐
│ High Nibble│ Low Nibble  │
│  Value 1   │  Value 0    │
│  Bits 7-4  │  Bits 3-0   │
└────────────┴────────────┘
```

**Accessors:**
```cpp
get_low()  → raw_bits & 0x0F              // Extract lower 4 bits
get_high() → (raw_bits >> 4) & 0x0F       // Extract upper 4 bits

set_low(v)  → raw_bits = (raw_bits & 0xF0) | (v.raw_bits & 0xF)
set_high(v) → raw_bits = (raw_bits & 0x0F) | ((v.raw_bits & 0xF) << 4)
```

> **Important:** FP4 is **display/storage only** — NO arithmetic operators are defined. All computation must go through promotion to FP32 first.

### 1.4 Complex Number Types

Three complex types with increasing precision:

| Type | Components | Component Type | Total Size |
|------|-----------|---------------|:---:|
| `complex32_t` | `real_` + `imag_` | `float16_t` | 4 bytes |
| `complex64_t` | `real_` + `imag_` | `float` | 8 bytes |
| `complex128_t` | `real_` + `imag_` | `double` | 16 bytes |

**Complex Arithmetic:**

All three types implement the standard complex number operations:

**Addition/Subtraction:**
```
(a + bi) + (c + di) = (a+c) + (b+d)i
(a + bi) - (c + di) = (a-c) + (b-d)i
```

**Multiplication:**
```
(a + bi) × (c + di) = (ac - bd) + (ad + bc)i
```
```cpp
complex32_t operator*(const complex32_t& other) const {
    float16_t r = real_ * other.real_ - imag_ * other.imag_;
    float16_t i = real_ * other.imag_ + imag_ * other.real_;
    return complex32_t(r, i);
}
```

**Division:**
```
(a + bi) / (c + di) = [(ac + bd) + (bc - ad)i] / (c² + d²)
```
```cpp
complex32_t operator/(const complex32_t& other) const {
    float16_t denom = other.real_ * other.real_ + other.imag_ * other.imag_;
    float16_t r = (real_ * other.real_ + imag_ * other.imag_) / denom;
    float16_t i = (imag_ * other.real_ - real_ * other.imag_) / denom;
    return complex32_t(r, i);
}
```

**Math functions for each complex type:**

| Function | Formula | Return Type |
|----------|---------|-------------|
| `abs(z)` | `√(real² + imag²)` | Component type (float16/float/double) |
| `conj(z)` | `real - imag·i` | Same complex type |
| `arg(z)` | `atan2(imag, real)` | Component type |
| `norm(z)` | `real² + imag²` | Component type |
| `polar(ρ, θ)` | `ρ·cos(θ) + ρ·sin(θ)·i` | Same complex type |
| `isnan(z)` | `isnan(real) \|\| isnan(imag)` | `bool` |

**Cross-type conversions:**
```cpp
to_complex64(complex128_t c)  → complex64_t   // Narrowing
to_complex128(complex64_t c)  → complex128_t  // Widening

convert_complex32_to_complex64(Tensor)  → Tensor  // Element-wise tensor conversion
convert_complex64_to_complex32(Tensor)  → Tensor  // Element-wise tensor conversion
```

**Mixed scalar-complex operators:**
Both `float` and `double` scalars can be combined with any complex type:
```cpp
// Examples:
2.0f + complex32_t(1,2)    → complex32_t(3, 2)
complex64_t(3,4) * 0.5     → complex64_t(1.5, 2.0)
3.0 == complex128_t(3,0)   → true
```

### 1.5 Cross-Type Conversion Functions (`FP4Converters.h`)

A complete set of converter functions between FP4 and all other float types:

| From | To | Function |
|------|-----|----------|
| FP4 | FP32 | `fp4_to_fp32()` |
| FP4 | FP64 | `fp4_to_fp64()` |
| FP4 | FP16 | `fp4_to_fp16()` |
| FP4 | BF16 | `fp4_to_bf16()` |
| FP32 | FP4 | `fp32_to_fp4()` |
| FP64 | FP4 | `fp64_to_fp4()` |
| FP16 | FP4 | `fp16_to_fp4()` |
| BF16 | FP4 | `bf16_to_fp4()` |
| FP4 packed | FP32 pair | `packed_fp4_to_fp32()` |
| FP4 packed | FP16 pair | `packed_fp4_to_fp16()` |
| FP4 packed | FP4 pair | `packed_fp4_to_fp4()` |

All conversions go through `float` as the intermediate representation.

### 1.6 Half ↔ Float32 Tensor Conversion (`DtypeCastUtils.h`)

For CPU operations that don't support FP16/BF16 natively, tensors are promoted element-wise:

```cpp
// Promote: Half → Float32
Tensor convert_half_to_float32(const Tensor& input);     // element-wise cast
// Demote: Float32 → Half (writes into pre-allocated output)
void convert_float32_to_half(const Tensor& float_tensor, Tensor& output);
```

---

## 2. System Design

### 2.1 The `dtype_traits<>` Compile-Time Trait System

Every dtype has a compile-time specialization that provides metadata:

```cpp
template<> struct dtype_traits<Dtype::UInt8> {
    using type = uint8_t;                          // C++ type
    static constexpr size_t size = sizeof(uint8_t); // Size in bytes
    static constexpr const char* name = "UInt8";    // String name
    static constexpr bool is_floating_point = false;
    static constexpr bool is_integral = true;
    static constexpr bool is_unsigned = true;
};
```

**Usage in generic code:**
```cpp
using T = dtype_traits<Dtype::UInt8>::type;   // T = uint8_t
constexpr size_t sz = dtype_traits<Dtype::Complex64>::size;  // sz = 8
```

### 2.2 Reverse Mapping: `type_to_dtype<T>()`

Converts a C++ type back to the Dtype enum at compile time:
```cpp
Dtype d = type_to_dtype<uint8_t>();      // returns Dtype::UInt8
Dtype d = type_to_dtype<complex64_t>();  // returns Dtype::Complex64
Dtype d = type_to_dtype<bool>();         // returns Dtype::Bool
```

Uses `if constexpr` chains for zero-runtime-cost resolution.

### 2.3 Runtime Type Predicates

Five predicate functions for runtime type checking:
```cpp
is_float(Dtype dt)    → true for Float16, Bfloat16, Float32, Float64, Float4_*
is_int(Dtype dt)      → true for Int16..Int64, UInt8..UInt64
is_unsigned(Dtype dt) → true for UInt8, UInt16, UInt32, UInt64
is_bool(Dtype dt)     → true only for Bool
is_complex(Dtype dt)  → true for Complex32, Complex64, Complex128
```

### 2.4 Type Promotion Rules (`promote_dtypes_bool`)

When two tensors with different dtypes interact (e.g., `add(int32_tensor, float32_tensor)`), the library promotes both to a common type:

**Priority order (highest to lowest):**
```
Complex128 > Complex64 > Complex32 > Float64 > Float32 > Float16 > 
Bfloat16 > FP4 > Int64 > Int32 > Int16 > Int8 > UInt8 > Bool
```

**Special case — Division promotion (`promote_dtypes_division`):**
```
Int + Int division → always promotes to Float32 (not integer division!)
Bool + Bool division → promotes to Float32
Float + anything → normal promotion
```
This matches PyTorch's behavior: `torch.tensor([5]) / torch.tensor([2])` returns `2.5`, not `2`.

### 2.5 Integer Promotion for Operations (`get_promoted_dtype`)

Certain operations (like unary math ops) promote integer types to float:
```
Int16, Int32, Bool, UInt8..UInt32 → Float32
Int64, UInt64                     → Float64
Complex types                     → No change
FP4 types                        → Float32
Float types                      → No change
```

### 2.6 File Architecture

| File | Role |
|------|------|
| [`Dtype.h`](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensor_centric_tensorlib/include/dtype/Dtype.h) | Enum definition — 18 types |
| [`Types.h`](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensor_centric_tensorlib/include/dtype/Types.h) | BF16/FP16 structs + Complex structs + math overloads |
| [`fp4.h`](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensor_centric_tensorlib/include/dtype/fp4.h) | FP4 E2M1 structs (unpacked + packed) + conversion functions |
| [`DtypeTraits.h`](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensor_centric_tensorlib/include/dtype/DtypeTraits.h) | Compile-time traits, predicates, promotion rules |
| [`DtypeCastUtils.h`](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensor_centric_tensorlib/include/dtype/DtypeCastUtils.h) | Runtime conversion helpers: half↔float32, complex32↔64, safe_pow |
| [`FP4Converters.h`](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensor_centric_tensorlib/include/dtype/FP4Converters.h) | FP4 ↔ all-type converter functions |
| [`CudaTraits.h`](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensor_centric_tensorlib/include/dtype/CudaTraits.h) | GPU type bridging: custom structs → CUDA native types |

---

## 3. Numerical Stability

### 3.1 FP4 Extreme Quantization

FP4 has only **8 positive values**: `{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}`.

**Quantization error is massive:**
```
Input: 2.7 → Nearest FP4: 3.0 → Error: 11%
Input: 5.5 → Nearest FP4: 6.0 → Error: 9%
Input: 0.3 → Nearest FP4: 0.5 → Error: 67%!
```

**Why FP4 still exists:** Used exclusively for quantized model weights in inference. The idea is that neural network weights, when properly calibrated, can tolerate extreme quantization with minimal accuracy loss. FP4 is **never** used for activation values, gradients, or intermediate computations.

> **No arithmetic operators** are defined on FP4 — you MUST convert to FP32 first. This is by design: computing in FP4 would be meaningless.

### 3.2 Complex32 Low Precision

`complex32_t` uses `float16_t` components (10-bit mantissa each). This means:
- Complex multiplication involves 4 multiplications and 2 additions of FP16 values
- Each step loses precision → final result has even fewer significant digits
- **Recommendation:** Use `complex64_t` (float components) for any real computation

### 3.3 `safe_pow()` Edge Case Handling

```
safe_pow(base, exponent):
    NaN input    → NaN output
    0^0          → 1 (mathematical convention)
    0^(negative) → +Infinity
    0^(positive) → 0
    negative^(non-integer) → NaN (complex result not representable)
    Normal case  → std::pow(base, exponent)
```

### 3.4 Bool Division Promotion

Division always promotes to float to prevent integer truncation:
```
Bool / Bool      → Float32  (true/true = 1.0, not 1)
Int32 / Int32    → Float32  (5/2 = 2.5, not 2)
Int64 / Int64    → Float64
Float32 / Int32  → Float32   (normal promotion applies)
```

---

## 4. Memory Layout

### 4.1 Sizes and Alignment

| Type | Memory per Element | Notes |
|------|:--:|------|
| `bool` | **1 byte** | Not 1 bit! C++ `sizeof(bool)` = 1 |
| `uint8_t` | **1 byte** | Standard unsigned byte |
| `float4_e2m1_t` | **1 byte** | Only lower 4 bits are meaningful; upper 4 bits unused |
| `float4_e2m1_2x_t` | **1 byte** | Both nibbles used — 2 values packed into 1 byte |
| `complex32_t` | **4 bytes** | = 2 × sizeof(float16_t) = 2 × 2 |
| `complex64_t` | **8 bytes** | = 2 × sizeof(float) = 2 × 4 |
| `complex128_t` | **16 bytes** | = 2 × sizeof(double) = 2 × 8 |

### 4.2 Bool Memory Layout

```
Bool tensor of shape [4]:
Memory: [0x01] [0x00] [0x01] [0x01]
         true   false  true   true   ← 4 bytes for 4 booleans
```

> **Why not bit-packing?** While packing 8 bools per byte would save memory, it makes random access 8× slower (requires shift + mask per element) and breaks alignment for SIMD/GPU operations. The extra memory cost (8×) is acceptable for the simplicity and speed gains.

### 4.3 FP4 Packed vs Unpacked Memory

```
Unpacked FP4 [4 values]: 4 bytes
[____XXXX] [____XXXX] [____XXXX] [____XXXX]
 val 0      val 1      val 2      val 3
 ↑ upper 4 bits wasted

Packed FP4 [4 values]: 2 bytes
[XXXXYYYY] [XXXXYYYY]
 val1 val0  val3 val2
 ↑ zero waste — 2× memory efficiency
```

### 4.4 Complex Memory Layout

```
complex64_t tensor of shape [3]:
Memory: [re0][im0] [re1][im1] [re2][im2]
         4B   4B    4B   4B    4B   4B    = 24 bytes total
         └─ 8B ─┘   └─ 8B ─┘  └─ 8B ─┘
```

Components are stored **interleaved** (real, imag, real, imag...) — each complex element is contiguous in memory.

---

## 5. Pseudocode / Program Logic

### 5.1 Float → FP4 Conversion (Nearest Neighbor)

```
FUNCTION float_to_fp4(f: float) → uint8:
    abs_f = |f|
    sign_bit = (f < 0) ? 8 : 0     // Bit 3 = sign

    IF isnan(abs_f): RETURN sign_bit | 7    // NaN → max value (6.0)
    IF abs_f > 6.0:  RETURN sign_bit | 7    // Clamp overflow

    // Threshold-based nearest-neighbor lookup
    IF abs_f < 0.25: RETURN sign_bit | 0    // → 0.0
    IF abs_f < 0.75: RETURN sign_bit | 1    // → 0.5
    IF abs_f < 1.25: RETURN sign_bit | 2    // → 1.0
    IF abs_f < 1.75: RETURN sign_bit | 3    // → 1.5
    IF abs_f < 2.25: RETURN sign_bit | 4    // → 2.0  (note: NOT 2.5!)
    IF abs_f < 3.50: RETURN sign_bit | 5    // → 3.0
    IF abs_f < 5.00: RETURN sign_bit | 6    // → 4.0
    RETURN sign_bit | 7                     // → 6.0
```

### 5.2 FP4 → Float Conversion (Switch Lookup)

```
FUNCTION fp4_to_float(val: uint8) → float:
    val = val & 0xF                        // Mask to 4 bits
    sign = (val >= 8) ? -1.0 : 1.0
    magnitude = val & 7                    // Strip sign bit

    LOOKUP[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
    RETURN LOOKUP[magnitude] × sign
```

### 5.3 FP4 Packed Operations

```
FUNCTION get_low(packed: uint8) → fp4:
    RETURN fp4(packed & 0x0F)

FUNCTION get_high(packed: uint8) → fp4:
    RETURN fp4((packed >> 4) & 0x0F)

FUNCTION pack(v0: fp4, v1: fp4) → uint8:
    RETURN (v0.bits & 0xF) << 4 | (v1.bits & 0xF)
```

### 5.4 Complex Multiplication

```
FUNCTION complex_mul(a: complex, b: complex) → complex:
    // (a.re + a.im·i) × (b.re + b.im·i)
    // = (a.re × b.re - a.im × b.im) + (a.re × b.im + a.im × b.re)·i
    real = a.re × b.re - a.im × b.im
    imag = a.re × b.im + a.im × b.re
    RETURN complex(real, imag)
```

### 5.5 Complex Division

```
FUNCTION complex_div(a: complex, b: complex) → complex:
    // Multiply numerator and denominator by conjugate of b
    denom = b.re² + b.im²                    // |b|²
    real = (a.re × b.re + a.im × b.im) / denom
    imag = (a.im × b.re - a.re × b.im) / denom
    RETURN complex(real, imag)
```

### 5.6 Type Promotion Logic

```
FUNCTION promote_dtypes(a: Dtype, b: Dtype) → Dtype:
    IF a == b: RETURN a                        // Same type → no promotion

    // Priority 1: Complex (highest)
    IF either is Complex128: RETURN Complex128
    IF either is Complex64:  RETURN Complex64
    IF either is Complex32:  RETURN Complex32

    // Priority 2: Floating point
    IF either is Float64:  RETURN Float64
    IF either is Float32:  RETURN Float32
    IF either is Float16:  RETURN Float16
    IF either is Bfloat16: RETURN Bfloat16

    // Priority 3: Integer (largest wins)
    IF either is Int64: RETURN Int64
    IF either is Int32: RETURN Int32
    IF either is Int16: RETURN Int16

    // Fallback
    RETURN Bool
```

---

## 6. Research Material

### 6.1 Standards and Specifications

| Reference | Relevance |
|-----------|-----------|
| **OCP Microscaling (MX) Format** | Defines the FP4 E2M1 format used by NVIDIA and AMD GPUs |
| **IEEE 754-2008** | Defines complex number representation conventions |
| **C++ `<complex>`** | Standard library complex number operations that our custom types mirror |

### 6.2 Comparison with PyTorch

| Feature | OwnTensor | PyTorch |
|---------|-----------|---------|
| Unsigned integers | ✅ UInt8, UInt16, UInt32, UInt64 | ✅ `torch.uint8` only (limited) |
| Bool type | ✅ `Dtype::Bool` | ✅ `torch.bool` |
| FP4 support | ✅ E2M1 (unpacked + packed) | ❌ (via external libs only) |
| Complex types | ✅ complex32, complex64, complex128 | ✅ `torch.complex32/64/128` |
| Type promotion | ✅ `promote_dtypes_bool()` | ✅ `torch.result_type()` |
| Division promotion | ✅ Int→Float32 always | ✅ Same behavior |
| Bool storage | 1 byte per element | 1 byte per element |
| Compile-time traits | ✅ `dtype_traits<>` | ✅ `c10::ScalarType` |

### 6.3 Operation Support by Type

| Operation | Bool | UInt | Int | Float | FP4 | Complex |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| Arithmetic (+,-,*,/) | via promotion | ✅ | ✅ | ✅ | ❌ | ✅ |
| Reduction (sum/min/max) | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ (sum only) |
| `all` / `any` | ✅ | via cast | via cast | via cast | ❌ | ❌ |
| NaN-aware ops | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Variance/Std | ❌ | ✅→F64 | ✅→F64 | ✅ | ❌ | ✅ |
| ArgMin/ArgMax | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Autograd | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
