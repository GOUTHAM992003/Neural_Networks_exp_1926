# FP16 & BF16 Implementations — Complete Documentation

---

## 0. What are FP16 and BF16?

**Float16 (FP16)** and **BFloat16 (BF16)** are 16-bit floating-point formats that use **half the memory** of standard 32-bit floats. They are essential in deep learning for reducing memory bandwidth, increasing throughput, and enabling larger batch sizes on GPUs.

### Bit Layout Comparison

```
FP32 (IEEE 754 Single Precision) — 32 bits
┌───┬──────────┬───────────────────────────┐
│ S │ EEEEEEEE │ MMMMMMMMMMMMMMMMMMMMMMM   │
│ 1 │    8     │          23               │
└───┴──────────┴───────────────────────────┘

BF16 (Brain Floating Point 16) — 16 bits
┌───┬──────────┬─────────┐
│ S │ EEEEEEEE │ MMMMMMM │
│ 1 │    8     │    7    │
└───┴──────────┴─────────┘

FP16 (IEEE 754 Half Precision) — 16 bits
┌───┬───────┬────────────┐
│ S │ EEEEE │ MMMMMMMMMM │
│ 1 │   5   │     10     │
└───┴───────┴────────────┘
```

Where: **S** = Sign bit, **E** = Exponent bits, **M** = Mantissa (fraction) bits.

### Format Comparison Table

| Property | FP32 | BF16 | FP16 |
|----------|------|------|------|
| **Total bits** | 32 | 16 | 16 |
| **Sign bits** | 1 | 1 | 1 |
| **Exponent bits** | 8 | 8 | 5 |
| **Mantissa bits** | 23 | 7 | 10 |
| **Exponent bias** | 127 | 127 | 15 |
| **Max value** | ~3.4×10³⁸ | ~3.39×10³⁸ | 65,504 |
| **Min positive normal** | ~1.18×10⁻³⁸ | ~1.18×10⁻³⁸ | ~6.10×10⁻⁵ |
| **Decimal precision** | ~7 digits | ~2-3 digits | ~3-4 digits |
| **Memory per element** | 4 bytes | 2 bytes | 2 bytes |
| **Primary use case** | General compute | Deep learning (training) | Graphics, inference |

### Key Insight: BF16 vs FP16 Trade-off

- **BF16** has the **same range as FP32** (8-bit exponent) but **less precision** (7-bit mantissa). This makes it ideal for **training** neural networks, where the range of gradient values matters more than precision.
- **FP16** has **more precision** than BF16 (10-bit mantissa) but a **much smaller range** (5-bit exponent, max ±65504). It's better for **inference** and **graphics** where values are bounded.

### Why Custom Structs Instead of Using `__half` / `__nv_bfloat16` Directly?

CUDA provides native `__half` and `__nv_bfloat16` types, but they:
1. **Only exist during GPU compilation** (`nvcc`) — they cause errors in CPU-only (host) code compiled with `g++` or `clang++`
2. **Cannot be used as template parameters** in some contexts on the host side
3. **Don't integrate with `std::numeric_limits`**, `std::is_arithmetic`, etc.

The solution: custom structs (`float16_t`, `bfloat16_t`) that:
- Store the raw 16-bit pattern in a `uint16_t raw_bits` field
- Work on **both CPU and GPU** (all methods marked `__device__ __host__`)
- Are **bitwise compatible** with `__half` / `__nv_bfloat16` — enabling zero-cost `reinterpret_cast` at GPU kernel launch boundaries via `CudaTraits.h`

---

## 1. Algorithm Design — Bit Manipulation Conversion Functions

All conversions live in `OwnTensor::detail` namespace and are marked `__device__ __host__ inline` to work on both CPU and GPU.

### 1.1 `bfloat16_to_float(uint16_t b) → float`

**Core idea:** BF16 is simply the upper 16 bits of an FP32 number. Conversion is mostly a left-shift by 16 bits, with special handling for Inf/NaN.

**Step-by-step:**

```
Input:  b = 0x4120  (BF16 representation of 10.0)

Step 1: Extract sign, exponent, mantissa from 16-bit BF16
        sign = (b & 0x8000) << 16    = 0x00000000  (positive)
        exp  = (b & 0x7F80) >> 7     = 0x82 = 130  (biased exponent)
        frac = (b & 0x007F)          = 0x20 = 32   (7-bit mantissa)

Step 2: Check for special values
        IF exp == 0xFF (255):
            IF frac == 0 → Infinity:  u = sign | 0x7F800000
            ELSE         → NaN:       u = sign | 0x7F800000 | (frac << 16)
        ELSE:
            Normal/subnormal: u = b << 16   (just shift the bits up)
            u = 0x41200000

Step 3: Reinterpret u as float via memcpy
        Result: 10.0f ✓
```

**Why the simple shift works for normal numbers:** BF16 shares the same exponent range and bias as FP32. The upper 16 bits of an FP32 number ARE a BF16 number (with the lower 16 mantissa bits truncated). So going back just requires padding those lower bits with zeros.

### 1.2 `float_to_bfloat16(float f) → uint16_t`

**Core idea:** Truncate the lower 16 bits of the FP32 representation, with Round-to-Nearest-Even (RNE) to minimize conversion error.

**Step-by-step:**

```
Input: f = 10.0f → binary: 0x41200000

Step 1: Reinterpret float as uint32_t via memcpy
        u = 0x41200000

Step 2: Extract components
        sign     = u & 0x80000000  = 0x00000000
        exponent = (u >> 23) & 0xFF = 0x82 = 130
        mantissa = u & 0x7FFFFF     = 0x200000

Step 3: Handle special cases
        IF exponent == 0xFF:
            IF mantissa == 0 → return Infinity: (sign >> 16) | 0x7F80
            ELSE             → return NaN:      (sign >> 16) | 0x7FC1
        IF exponent > 0x8E (142):
            → Overflow, clamp to Infinity: (sign >> 16) | 0x7F80

Step 4: Round-to-Nearest-Even (RNE)
        lsb = (u >> 16) & 1          ← the bit that will become the LSB of BF16
        IF lsb == 1:
            rounding_bias = 0x8000    ← round UP (towards even)
        ELSE:
            rounding_bias = 0x7FFF    ← round DOWN (towards even)
        u += rounding_bias

Step 5: Truncate — shift right 16 bits
        result = u >> 16 = 0x4120
```

**Why Round-to-Nearest-Even (Banker's Rounding)?**
Standard "round-half-up" (0.5 always rounds up) introduces a systematic positive bias over many operations. RNE alternates: when the value is exactly halfway, it rounds to the nearest EVEN number. This eliminates the statistical bias, which is critical when converting millions of neural network weights.

### 1.3 `float16_to_float(uint16_t h) → float`

**Core idea:** More complex than BF16 because FP16 has a different exponent bias (15 vs 127). Must re-bias the exponent and handle denormal numbers.

**Step-by-step:**

```
Input: h = 0x4900  (FP16 representation of 10.0)

Step 1: Extract sign, exponent, mantissa from 16-bit FP16
        sign = (h & 0x8000) << 16       = 0x00000000  (positive)
        exp  = (h & 0x7C00) >> 10       = 0x12 = 18   (biased exponent, bias=15)
        frac = (h & 0x03FF)             = 0x100 = 256  (10-bit mantissa)

Step 2: Check for special values
        CASE exp == 0 (Zero or Denormal):
            IF frac == 0 → Zero: u = sign (preserve sign for +0/-0)
            ELSE → Denormal: f = (frac / 1024.0) × 2⁻¹⁴
                   (compute as float, apply sign, then memcpy to uint32)

        CASE exp == 0x1F (31) → Infinity or NaN:
            u = sign | 0x7F800000 | (frac << 13)

        CASE normal number:
            exp32 = exp + (127 - 15)           ← Re-bias: subtract FP16 bias, add FP32 bias
                  = 18 + 112 = 130 = 0x82
            u = sign | (exp32 << 23) | (frac << 13)
              = 0x00000000 | 0x41000000 | 0x00200000
              = 0x41200000

Step 3: Reinterpret u as float via memcpy
        Result: 10.0f ✓
```

**Exponent re-biasing explained:**
- FP16 exponent is stored with bias 15 → stored value = actual_exponent + 15
- FP32 exponent is stored with bias 127 → stored value = actual_exponent + 127
- To convert: `exp32 = exp16 + (127 - 15) = exp16 + 112`

**Mantissa widening:** FP16 has 10 mantissa bits, FP32 has 23. The 10 bits are shifted left by 13 positions (`frac << 13`) and the remaining 13 bits are implicitly zero.

### 1.4 `float_to_float16(float f) → uint16_t`

**Core idea:** Re-bias the exponent (127→15), truncate the mantissa (23→10 bits), handle overflow (FP32 values > 65504 clamp to FP16 infinity) and underflow (very small values become denormals or zero).

**Step-by-step:**

```
Input: f = 10.0f → binary: 0x41200000

Step 1: Reinterpret float as uint32_t
        x = 0x41200000

Step 2: Extract components
        sign    = (x >> 16) & 0x8000     = 0x0000
        exp_32  = (x >> 23) & 0xFF       = 0x82 = 130
        mant_32 = x & 0x007FFFFF         = 0x200000

Step 3: Handle special cases
        IF exp_32 == 0xFF:
            IF mant_32 == 0 → Infinity: sign | 0x7C00
            ELSE            → NaN: sign | 0x7C00 | (mant_32 >> 13)

Step 4: Re-bias exponent
        exp_16 = exp_32 - 127 + 15 = 130 - 112 = 18

Step 5: Handle range limits
        IF exp_16 <= 0:
            IF exp_16 < -10 → Underflow to zero: return sign
            ELSE → Generate denormal:
                mant_32 |= 0x00800000   ← add implicit leading 1
                shift = 1 - exp_16
                half_mant = mant_32 >> (shift + 13)
                return sign | half_mant

        IF exp_16 >= 31 → Overflow to infinity: return sign | 0x7C00

Step 6: Normal number — round and pack
        half_exp  = exp_16 = 18
        half_mant = (mant_32 + 0x00001000) >> 13    ← round bit at position 12
                  = (0x200000 + 0x1000) >> 13
                  = 0x201000 >> 13
                  = 0x100
        result = sign | (half_exp << 10) | half_mant
               = 0x0000 | 0x4800 | 0x0100
               = 0x4900 ✓
```

---

## 2. System Design — Struct Architecture

### 2.1 The `bfloat16_t` and `float16_t` Structs

Both structs follow an identical design pattern — a thin wrapper around a `uint16_t`:

```cpp
struct bfloat16_t {         // (or float16_t)
    uint16_t raw_bits;      // The ONLY data member — exactly 2 bytes

    // Constructors
    bfloat16_t();                        // default: raw_bits = 0
    explicit bfloat16_t(float val);      // convert float → BF16 bits
    bfloat16_t(const bfloat16_t& other); // copy
    template<U> explicit bfloat16_t(U);  // from any arithmetic type (int, double...)

    // Implicit conversion to float
    operator float() const;              // BF16 bits → float

    // Operators: =, >, <, >=, <=, ==, !=, +, -, *, /, +=, -=, *=, /=
};
```

### 2.2 The "Promote-Compute-Demote" Pattern

**Every single arithmetic operation** works the same way:

```
BF16 value A, BF16 value B
    │                │
    ▼                ▼
  float(A)       float(B)         ← PROMOTE to 32-bit
    │                │
    └──── op(·) ─────┘             ← COMPUTE in full precision
           │
           ▼
      bfloat16_t(result)           ← DEMOTE back to 16-bit
```

**Example — Addition:**
```cpp
bfloat16_t operator+(const bfloat16_t& other) const {
    return bfloat16_t(static_cast<float>(*this) + static_cast<float>(other));
    //                ─── promote ───────   op   ─── promote ───────
    //  ──────────── demote (constructor) ──────────────────────────
}
```

**Why not compute directly in 16-bit?**
FP16/BF16 have extremely limited precision (3-4 decimal digits). Intermediate results would lose too much accuracy. By promoting to FP32, computing there, and then demoting, we get the most accurate 16-bit result possible.

**Comparison operators** use the same promote pattern:
```cpp
bool operator>(const bfloat16_t& other) const {
    return static_cast<float>(*this) > static_cast<float>(other);
}
```

**Exception — equality (`==`):**
```cpp
bool operator==(const bfloat16_t& other) const {
    return raw_bits == other.raw_bits;   // Direct bit comparison! No conversion.
}
```
This is correct because two BF16 values are equal if and only if their bit patterns are identical (with the caveat that +0.0 and -0.0 are different bit patterns).

### 2.3 Templated Constructor for Any Arithmetic Type

```cpp
template <typename U, typename = std::enable_if_t<
    std::is_arithmetic_v<U> && !std::is_same_v<std::decay_t<U>, float>
>>
explicit bfloat16_t(U val) {
    raw_bits = detail::float_to_bfloat16(static_cast<float>(val));
}
```

This allows constructing from `int`, `double`, `long`, etc. — the value is first cast to `float`, then converted to BF16 bits. The `enable_if_t` guard prevents ambiguity with the `explicit bfloat16_t(float val)` constructor.

### 2.4 Math Function Overloads

Both types provide overloads for standard math functions. All follow the promote-compute-demote pattern:

```cpp
inline float16_t sqrt(float16_t a) {
    return float16_t(std::sqrt(static_cast<float>(a)));
}
```

**Full list of overloaded math functions:**

| Function | BF16 | FP16 |
|----------|:----:|:----:|
| `abs` | ✅ | ✅ |
| `sqrt` | ✅ | ✅ |
| `exp` | ✅ | ✅ |
| `log` | ✅ | ❌ |
| `sin` | ✅ | ✅ |
| `cos` | ✅ | ✅ |
| `tan` | ✅ | ✅ |
| `tanh` | ✅ | ✅ |
| `floor` | ✅ | ✅ |
| `ceil` | ✅ | ✅ |
| `round` | ✅ | ✅ |
| `pow` | ✅ | ✅ |
| `hypot` | ✅ | ✅ |

### 2.5 `std::numeric_limits` Specializations

The library provides `std::numeric_limits` specializations so that generic code (like reduction kernels with `std::numeric_limits<T>::max()`) works correctly:

```cpp
namespace std {
    template<> struct numeric_limits<OwnTensor::bfloat16_t> {
        static constexpr bool is_specialized = true;
        static constexpr bool is_signed = true;
        static constexpr bool has_infinity = true;
        static constexpr bool has_quiet_NaN = true;
        static bfloat16_t lowest() { return bfloat16_t(-3.38953e38f); }
        static bfloat16_t max()    { return bfloat16_t(3.38953e38f); }
        static bfloat16_t infinity() { return bfloat16_t(INFINITY); }
        static bfloat16_t quiet_NaN() { return bfloat16_t(NAN); }
    };

    template<> struct numeric_limits<OwnTensor::float16_t> {
        // Same structure...
        static float16_t lowest() { return float16_t(-65504.0f); }
        static float16_t max()    { return float16_t(65504.0f); }
    };
}
```

### 2.6 GPU Type Bridging — `CudaTraits.h`

When launching CUDA kernels, the custom structs must be converted to CUDA's native types. Since they are **bitwise identical** (both store `uint16_t` internally), this is a zero-cost `reinterpret_cast`:

```cpp
// CudaTraits.h - Compile-time type mapping
template<typename T> struct ToCudaNative { using type = T; };       // Default: keep same type
template<> struct ToCudaNative<float16_t>  { using type = __half; };
template<> struct ToCudaNative<bfloat16_t> { using type = __nv_bfloat16; };

template<typename T>
using CudaNativeType = typename ToCudaNative<T>::type;

// Usage in GPU dispatcher (ReductionImplGPU.cu):
using CudaT = CudaNativeType<T>;  // float16_t → __half
const CudaT* input_data = reinterpret_cast<const CudaT*>(input.data<T>());
// ↑ Zero-cost conversion — no data copying, just reinterpreting the same memory
```

**Why this works:** Both `float16_t` and `__half` are exactly 2 bytes containing the same IEEE 754 bit pattern. `reinterpret_cast` just tells the compiler "treat this memory as a different type" without moving any data.

---

## 3. Numerical Stability

### 3.1 Precision Loss

| Type | Mantissa Bits | Decimal Digits | Example: π |
|------|:---:|:---:|---|
| FP32 | 23 | ~7 | 3.141593 |
| FP16 | 10 | ~3-4 | 3.141 |
| BF16 | 7 | ~2-3 | 3.12 |

**Practical impact:** When summing 1000 BF16 values of `1.0`, naïve accumulation gives `992.0` instead of `1000.0` — the low mantissa precision causes rounding errors that compound. This is why the reduction kernels use **double accumulation** for FP16/BF16 inputs (see ReductionOps documentation, Section 3.2).

### 3.2 Range Overflow

```
FP16 max: 65,504
BF16 max: ~3.39 × 10³⁸ (same as FP32!)
```

**FP16 is dangerous for gradients:** In deep learning, gradient values can easily exceed 65504 during backpropagation (especially in early training or with high learning rates), causing overflow to infinity. This is why **loss scaling** is required for FP16 mixed-precision training.

**BF16 avoids this problem** because it shares FP32's exponent range. Gradients almost never overflow BF16.

### 3.3 Round-to-Nearest-Even (Banker's Rounding)

The `float_to_bfloat16` function uses RNE rounding:

```
Value  = 1.5  → rounds to 2 (nearest even)
Value  = 2.5  → rounds to 2 (nearest even)
Value  = 3.5  → rounds to 4 (nearest even)
Value  = 4.5  → rounds to 4 (nearest even)
```

Standard "round-half-up" would always round 0.5 upward, creating a **systematic positive bias**. Over millions of weight updates, this bias would cause the model to drift. RNE eliminates this by alternating the rounding direction at the halfway point.

### 3.4 Special Value Handling Summary

| Input | `float_to_bfloat16` | `float_to_float16` |
|-------|---------------------|---------------------|
| +0.0 | 0x0000 | 0x0000 |
| -0.0 | 0x8000 | 0x8000 |
| +Infinity | 0x7F80 | 0x7C00 |
| -Infinity | 0xFF80 | 0xFC00 |
| NaN | 0x7FC1 (quiet NaN) | 0x7C01+ (preserves mantissa bits) |
| Overflow (>max) | Clamp to ±Infinity | Clamp to ±Infinity |
| Very small underflow | Flush to ±0 | Generate denormal or ±0 |

---

## 4. Memory Layout

### 4.1 16-Bit Storage

Each `float16_t` and `bfloat16_t` occupies exactly **2 bytes** in memory:

```cpp
struct bfloat16_t {
    uint16_t raw_bits;    // 2 bytes, no padding
};
static_assert(sizeof(bfloat16_t) == 2);
```

### 4.2 Bit Field Layouts

**BFloat16 (BF16):**
```
Bit:  15  14  13  12  11  10   9   8   7   6   5   4   3   2   1   0
      ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
      │ S │      Exponent (8 bits)       │    Mantissa (7 bits)        │
      └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
      
Masks: sign = 0x8000, exponent = 0x7F80, mantissa = 0x007F
Value = (-1)^S × 2^(E-127) × (1 + M/128)
```

**Float16 (FP16):**
```
Bit:  15  14  13  12  11  10   9   8   7   6   5   4   3   2   1   0
      ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
      │ S │   Exp (5b)    │          Mantissa (10 bits)                │
      └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
      
Masks: sign = 0x8000, exponent = 0x7C00, mantissa = 0x03FF
Value = (-1)^S × 2^(E-15) × (1 + M/1024)
```

### 4.3 Memory Bandwidth Impact

| Scenario | FP32 | FP16/BF16 | Savings |
|----------|------|-----------|---------|
| 1M parameters | 4 MB | 2 MB | 2× less memory |
| Batch of 256 × 768 activations | 786 KB | 393 KB | 2× less bandwidth |
| GPU memory-bound kernel | 900 GB/s throughput | Effectively 2× more elements/sec | 2× faster |

**The key benefit is bandwidth, not compute.** Modern GPUs like A100 have dedicated Tensor Cores that compute FP16/BF16 matrix operations at 2-4× the speed of FP32 in addition to the bandwidth savings.

---

## 5. Pseudocode / Program Logic

### 5.1 BF16 → Float32 Conversion

```
FUNCTION bfloat16_to_float(b: uint16):
    sign = (b AND 0x8000) << 16          // Move sign to bit 31
    exp  = (b AND 0x7F80) >> 7           // Extract 8-bit exponent
    frac = (b AND 0x007F)                // Extract 7-bit mantissa

    IF exp == 255:                        // Special values
        IF frac == 0:
            result_bits = sign OR 0x7F800000      // ±Infinity
        ELSE:
            result_bits = sign OR 0x7F800000 OR (frac << 16)   // NaN
    ELSE:
        result_bits = b << 16             // Normal: just shift left 16

    RETURN reinterpret_as_float(result_bits)
```

### 5.2 Float32 → BF16 Conversion

```
FUNCTION float_to_bfloat16(f: float):
    u = reinterpret_as_uint32(f)
    sign     = u AND 0x80000000
    exponent = (u >> 23) AND 0xFF
    mantissa = u AND 0x7FFFFF

    // Special cases
    IF exponent == 255:
        IF mantissa == 0: RETURN (sign >> 16) OR 0x7F80     // Infinity
        ELSE:             RETURN (sign >> 16) OR 0x7FC1     // Quiet NaN

    IF exponent > 142:    RETURN (sign >> 16) OR 0x7F80     // Overflow → Inf

    // Round-to-Nearest-Even
    lsb = (u >> 16) AND 1
    IF lsb == 1:  rounding_bias = 0x8000    // Round up (to even)
    ELSE:         rounding_bias = 0x7FFF    // Round down (to even)
    u = u + rounding_bias

    RETURN u >> 16                          // Truncate lower 16 bits
```

### 5.3 FP16 → Float32 Conversion

```
FUNCTION float16_to_float(h: uint16):
    sign = (h AND 0x8000) << 16
    exp  = (h AND 0x7C00) >> 10
    frac = (h AND 0x03FF)

    IF exp == 0:                              // Zero or Denormal
        IF frac == 0:
            result_bits = sign                // ±Zero
        ELSE:
            f = (frac / 1024.0) × 2^(-14)    // Denormal: no implicit 1
            IF sign: f = -f
            RETURN f

    IF exp == 31:                             // Infinity or NaN
        result_bits = sign OR 0x7F800000 OR (frac << 13)

    ELSE:                                     // Normal number
        exp32 = exp + 112                     // Re-bias: 127 - 15 = 112
        result_bits = sign OR (exp32 << 23) OR (frac << 13)

    RETURN reinterpret_as_float(result_bits)
```

### 5.4 Float32 → FP16 Conversion

```
FUNCTION float_to_float16(f: float):
    x = reinterpret_as_uint32(f)
    sign    = (x >> 16) AND 0x8000
    exp_32  = (x >> 23) AND 0xFF
    mant_32 = x AND 0x007FFFFF

    // Special cases
    IF exp_32 == 255:
        IF mant_32 == 0: RETURN sign OR 0x7C00              // ±Infinity
        ELSE:            RETURN sign OR 0x7C00 OR (mant >> 13)  // NaN

    exp_16 = exp_32 - 112                                     // Re-bias

    IF exp_16 <= 0:                                           // Underflow
        IF exp_16 < -10: RETURN sign                          // Too small → ±0
        ELSE:                                                 // Denormal
            mant_32 = mant_32 OR 0x00800000                   // Add implicit 1
            shift = 1 - exp_16
            RETURN sign OR (mant_32 >> (shift + 13))

    IF exp_16 >= 31: RETURN sign OR 0x7C00                    // Overflow → ±Inf

    // Normal: round mantissa and pack
    half_mant = (mant_32 + 0x00001000) >> 13                  // Round at bit 12
    RETURN sign OR (exp_16 << 10) OR half_mant
```

---

## 6. Research Material

### 6.1 Standards and Specifications

| Reference | Relevance |
|-----------|-----------|
| **IEEE 754-2008** | Defines the FP16 half-precision format (binary16) |
| **Google Brain BFloat16** | Introduced BF16 for TPU training (2018). Key insight: exponent range matters more than mantissa precision for DL |
| **NVIDIA Ampere Architecture** | A100 GPU with native BF16 Tensor Core support (2020) |

### 6.2 Comparison with Other Frameworks

| Feature | OwnTensor | PyTorch | TensorFlow |
|---------|-----------|---------|------------|
| FP16 type name | `float16_t` | `torch.float16` / `torch.half` | `tf.float16` |
| BF16 type name | `bfloat16_t` | `torch.bfloat16` | `tf.bfloat16` |
| Implementation | Custom struct + bit manipulation | Custom C++ class (similar) | Eigen library |
| CPU arithmetic | Promote→float32→compute→demote | Same pattern | Same pattern |
| GPU native ops | Via `CudaTraits` `reinterpret_cast` | Via `at::Half` → `__half` | Via XLA compilation |
| `numeric_limits` | ✅ Custom specialization | ✅ `c10::Half` specialization | N/A (Python only) |
| Rounding mode | Round-to-Nearest-Even | Implementation-dependent | Round-to-Nearest-Even |

### 6.3 File Architecture Summary

| File | Role |
|------|------|
| [`Types.h`](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensor_centric_tensorlib/include/dtype/Types.h) | All type definitions: conversion functions, structs, math overloads, numeric_limits |
| [`CudaTraits.h`](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensor_centric_tensorlib/include/dtype/CudaTraits.h) | GPU type bridging: `CudaNativeType<float16_t>` → `__half` |
| [`DtypeTraits.h`](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensor_centric_tensorlib/include/dtype/DtypeTraits.h) | Type trait system connecting dtypes to C++ types |
| [`Dtype.h`](file:///home/blu-bridge016/Desktop/Neural_Networks_exp_1926/tensor_centric_tensorlib/include/dtype/Dtype.h) | Dtype enum definitions (`Dtype::Float16`, `Dtype::BFloat16`) |
