# Datatypes — Complete Documentation
## Int8, UInt8/16/32/64, Bool, FP8 (E4M3FN & E5M2), Complex32/64/128 — Implementation & Integration

> **Note:** FP16 (`float16_t`) and BF16 (`bfloat16_t`) are documented separately in `FP16_BF16_documentation.md`.

---

## 0. History — How the Type System Evolved

### Phase 1: The Original 7 Types

The library initially supported only **7 data types**:
```
Int16, Int32, Int64, Float32, Float64, BFloat16, Float16
```
- `Int16/32/64` and `Float32/64` used native C++ types (`int16_t`, `float`, `double`, etc.)
- `BFloat16` and `Float16` required custom structs (`bfloat16_t`, `float16_t`) with bitwise conversion functions (documented in `FP16_BF16_documentation.md`)

### Phase 2: The Types We Added (This Component)

We expanded the type system from **7 → 18 types** by implementing **11 new types**:

| New Type | C++ Type | What We Built |
|----------|----------|---------------|
| `Int8` | `int8_t` | Native C++ type, just needed enum + traits + integration |
| `UInt8` | `uint8_t` | Native C++ type, full integration |
| `UInt16` | `uint16_t` | Native C++ type, full integration |
| `UInt32` | `uint32_t` | Native C++ type, full integration |
| `UInt64` | `uint64_t` | Native C++ type, full integration |
| `Bool` | `bool` | Native C++ type, special promotion rules |
| `Float8_E4M3FN` | `float8_e4m3fn_t` | **Custom struct** with bitwise FP32↔FP8 conversion |
| `Float8_E5M2` | `float8_e5m2_t` | **Custom struct** with clever FP16 shortcut conversion |
| `Complex32` | `complex32_t` | **Custom struct** (float16 real + float16 imag) |
| `Complex64` | `complex64_t` | **Custom struct** (float32 real + float32 imag) |
| `Complex128` | `complex128_t` | **Custom struct** (float64 real + float64 imag) |

---

## 1. Algorithm Design

### 1.1 Int8 and Unsigned Integer Types (UInt8/16/32/64)

These use **native C++ types** — no custom structs or conversion functions needed. The implementation work was purely in **integration**: adding them to the enum, traits, predicates, promotion tables, and operation dispatchers.

**Why UInt matters for deep learning:**
- `UInt8`: Image pixel data (0–255), quantized model weights, boolean masks
- `UInt16/32`: Index tensors, segmentation masks
- `UInt64`: Large dataset indices, hash values

### 1.2 Bool Type

Uses C++'s native `bool` type (1 byte, not 1 bit). Bool is the **weakest type** in promotion — it yields to everything else:
```
Bool + Int32  → Int32
Bool + Float16 → Float16
Bool + Complex64 → Complex64
Bool / Bool → Float32 (division always promotes to float!)
```

### 1.3 Float8 E4M3FN — Custom Struct with Bitwise Conversion

**Format:** `[Sign:1][Exponent:4][Mantissa:3]`, Bias=7, Max ≈ 448
**Key property:** No Infinity representation — overflow saturates to NaN (`0x7F`)
**Use case:** Forward pass (weights, activations) in training

#### Float → E4M3FN Conversion (`float_to_e4m3fn`)

```cpp
__device__ __host__ inline uint8_t float_to_e4m3fn(float f) {
    constexpr uint32_t fp8_max = UINT32_C(1087) << 20;      // 480.0f binary
    constexpr uint32_t denorm_mask = UINT32_C(141) << 23;    // Magic: (127-7)+(23-3)+1 = 141
    
    uint32_t f_bits;
    memcpy(&f_bits, &f, sizeof(f));
    
    const uint32_t sign = f_bits & 0x80000000u;
    f_bits ^= sign;  // Remove sign for processing
    
    if (f_bits >= fp8_max) {
        result = 0x7F;  // Overflow → NaN (NO infinity in E4M3FN!)
    } else if (f_bits < (121u << 23)) {
        // Denormal: use magic float addition trick
        float f_tmp = *(float*)&f_bits;
        f_tmp += *(float*)&denorm_mask;   // Magic denorm conversion
        memcpy(&f_bits_tmp, &f_tmp, sizeof(f_tmp));
        result = (uint8_t)(f_bits_tmp - denorm_mask);
    } else {
        // Normal: Round-to-Nearest-Even
        uint8_t mant_odd = (f_bits >> 20) & 1;
        f_bits += ((7 - 127) << 23) + 0x7FFFF;  // Rebias exponent + round
        f_bits += mant_odd;                       // RNE tie-break
        result = (uint8_t)(f_bits >> 20);
    }
    
    result |= (uint8_t)(sign >> 24);
    return result;
}
```

**Key algorithm points:**
1. **Magic number 141:** `(127 - 7) + (23 - 3) + 1` — adjusts FP32 bias (127) to E4M3 bias (7), accounts for mantissa width difference (23→3), plus 1 for normalization
2. **Denormal hack:** Adding a "magic float" forces the FPU to naturally shift bits into the correct denormal position — avoids manual bit shifting
3. **Round-to-Nearest-Even:** `mant_odd` checks the LSB of the mantissa; if it's 1 (odd), the rounding bias is adjusted to round towards even

#### E4M3FN → Float Conversion (`e4m3fn_to_float`)

Uses **leading zero count** (`__clz` on GPU, `__builtin_clz` on CPU) for fast denormal detection:
```cpp
__device__ __host__ inline float e4m3fn_to_float(uint8_t input) {
    const uint32_t w = (uint32_t)input << 24;           // Shift to upper byte
    const uint32_t sign = w & 0x80000000u;
    const uint32_t nonsign = w & 0x7FFFFFFFu;
    
    uint32_t renorm_shift = __builtin_clz(nonsign);     // Count leading zeros
    renorm_shift = renorm_shift > 4 ? renorm_shift - 4 : 0;
    
    const int32_t inf_nan_mask = ((int32_t)(nonsign + 0x01000000) >> 8) & 0x7F800000;
    const int32_t zero_mask = (int32_t)(nonsign - 1) >> 31;
    
    // Normalize, adjust bias (0x78 = 120 = 127 - 7), handle NaN and zero
    uint32_t result = sign |
        ((((nonsign << renorm_shift >> 4) + ((0x78 - renorm_shift) << 23)) |
          inf_nan_mask) & ~zero_mask);
    
    float f;
    memcpy(&f, &result, sizeof(f));
    return f;
}
```

### 1.4 Float8 E5M2 — The Clever FP16 Shortcut

**Format:** `[Sign:1][Exponent:5][Mantissa:2]`, Bias=15, Max ≈ 57344
**Key property:** Supports ±Infinity (unlike E4M3FN)
**Use case:** Backward pass (gradients) — wider range needed for gradient magnitudes

#### E5M2 → Float: The "Shift-and-Reuse" Trick

E5M2 has **exactly the same 5-bit exponent** as FP16. So conversion is trivially:
```cpp
__device__ __host__ inline float e5m2_to_float(uint8_t input) {
    // E5M2:  [S:1][EEEEE:5][MM:2]
    // FP16:  [S:1][EEEEE:5][MMMMMMMMMM:10]
    // Just shift left 8 bits → valid FP16 with zero-padded mantissa!
    uint16_t fp16_bits = static_cast<uint16_t>(input) << 8;
    return float16_to_float(fp16_bits);  // Reuse existing FP16→FP32 converter!
}
```

This is **zero additional code** — pure reuse of the FP16 converter we already built in Component 2!

#### Float → E5M2 Conversion

Similar to E4M3FN but with different constants:
- Denorm magic: `((127 - 15) + (23 - 2) + 1) = 134`
- Overflow threshold: 65536.0f (vs 480 for E4M3)
- Mantissa odd bit at position 21 (vs 20 for E4M3)
- **Supports Infinity:** Overflow → `0x7C` (Inf), not NaN

### 1.5 Complex Number Types

Three precision levels built as **custom structs**:

| Type | Components | Component Type | Total Size |
|------|-----------|---------------|:---:|
| `complex32_t` | `real_` + `imag_` | `float16_t` | 4 bytes |
| `complex64_t` | `real_` + `imag_` | `float` | 8 bytes |
| `complex128_t` | `real_` + `imag_` | `double` | 16 bytes |

**Each struct implements:**
- Constructors (default, from components, copy, from scalar, from other complex types)
- Arithmetic operators: `+`, `-`, `*`, `/` with full complex math formulas
- Compound assignment: `+=`, `-=`, `*=`, `/=`
- Comparison: `==`, `!=`
- Mixed scalar-complex operators (friend functions for `float`, `double`)
- Conversion operators: `to_complex64()`, `to_complex128()`

**Complex Multiplication:**
```
(a + bi) × (c + di) = (ac - bd) + (ad + bc)i
```

**Complex Division (conjugate method):**
```
(a + bi) / (c + di) = [(ac + bd) + (bc - ad)i] / (c² + d²)
```

**Math function overloads per complex type:**
`abs`, `conj`, `arg`, `norm`, `polar`, `isnan` — all implemented as free functions.

---

## 2. System Design — Step-by-Step Implementation Guide

### 2.1 How to Add a New Datatype: The 7-File Checklist

When adding a new type (e.g., `UInt8`), here are **all the files and functions** that must be modified:

---

#### Step 1: Add to `Dtype.h` — The Enum

```cpp
enum class Dtype {
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,  // ← ADD HERE
    ...
};
```

---

#### Step 2: Add to `Types.h` — Struct Definition (if custom type)

For native types (`uint8_t`, `bool`): **Nothing needed** — C++ already has these.

For custom types (`float8_e4m3fn_t`, `complex32_t`): Write the full struct:
```cpp
struct float8_e4m3fn_t {
    uint8_t raw_bits;
    // Constructors, operator float(), operators, etc.
};
```

Also add in `Types.h`:
- **Conversion functions** in `detail` namespace (e.g., `float_to_e4m3fn`, `e4m3fn_to_float`)
- **Math overloads:** `abs()`, `sqrt()`, `exp()`, `log()`, `sin()`, `cos()`, `tan()`, `tanh()`, `floor()`, `ceil()`, `round()`, `pow()`
- **`std::numeric_limits` specialization** (if the type needs `min()`, `max()`, `lowest()`, `infinity()`, `quiet_NaN()`)
- **CUDA native operator overloads** (inside `#ifdef __CUDACC__` block) — because CUDA's `__nv_fp8_e4m3` type has NO built-in arithmetic operators

---

#### Step 3: Add to `DtypeTraits.h` — 6 Functions Must Be Updated

**Function 1: `dtype_traits<>` specialization**
```cpp
template<> struct dtype_traits<Dtype::UInt8> {
    using type = uint8_t;
    static constexpr size_t size = sizeof(uint8_t);
    static constexpr const char* name = "UInt8";
    static constexpr bool is_floating_point = false;
    static constexpr bool is_integral = true;
    static constexpr bool is_unsigned = true;
};
```

**Function 2: `is_same_type<T>()`** — Add branch:
```cpp
else if constexpr (std::is_same_v<T, uint8_t>) {
    return dtype == Dtype::UInt8;
}
```

**Function 3: `type_to_dtype<T>()`** — Add reverse mapping:
```cpp
else if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, unsigned char>) {
    return Dtype::UInt8;
}
```

**Function 4: Type predicates** — Add to the correct predicate:
```cpp
constexpr bool is_unsigned(Dtype dt) {
    switch (dt) {
        case Dtype::UInt8:   // ← ADD
        case Dtype::UInt16:  // ← ADD
            return true;
    }
}
```

**Function 5: `get_dtype_name()`** — Add string representation:
```cpp
case Dtype::UInt8: return "uint8";
```

**Function 6: Promotion tables** — This is the BIG one. Two 18×18 tables must be updated:
- `promotion_table[18][18]` — Tensor + Tensor promotion
- `scalar_tensor_table[18][18]` — Scalar + Tensor promotion

Every row AND column for the new type must be filled in.

---

#### Step 4: Add to `DtypeCastUtils.h` — Conversion Helpers

For half-precision or complex types, add element-wise tensor converters:
```cpp
inline Tensor convert_complex32_to_complex64(const Tensor& input);
inline void convert_complex64_to_complex32(const Tensor& input, Tensor& output);
```

Also add to `get_promoted_dtype()` — the unary operation promotion function:
```cpp
case Dtype::UInt8:
case Dtype::UInt16:
case Dtype::UInt32:
    return Dtype::Float32;  // Promote unsigned ints to float for math ops
```

---

#### Step 5: Add to Operation Dispatch Macros

Every operation (`TensorOps.h`, `ScalarOps.h`, `Reduction.h`, etc.) uses dispatch macros to route runtime `Dtype` values to compile-time template instantiations. Example pattern:

```cpp
// BEFORE (7 types):
switch(dtype) {
    case Dtype::Int16: op<int16_t>(); break;
    case Dtype::Int32: op<int32_t>(); break;
    case Dtype::Float32: op<float>(); break;
    // ... 7 cases
}

// AFTER (18 types):
switch(dtype) {
    case Dtype::Int8: op<int8_t>(); break;      // ← NEW
    case Dtype::Int16: op<int16_t>(); break;
    case Dtype::Int32: op<int32_t>(); break;
    case Dtype::UInt8: op<uint8_t>(); break;     // ← NEW
    case Dtype::UInt16: op<uint16_t>(); break;   // ← NEW
    case Dtype::Bool: op<bool>(); break;         // ← NEW
    case Dtype::Float8_E4M3FN: op<float8_e4m3fn_t>(); break;  // ← NEW
    case Dtype::Complex64: op<complex64_t>(); break;           // ← NEW
    // ... 18 cases total
}
```

**Files with dispatch switches that need updating:**
- `include/ops/TensorOps.h` — Binary tensor operations (add, sub, mul, div)
- `include/ops/ScalarOps.h` — Scalar-tensor operations
- `include/ops/UnaryOps/Arithmetics.h` — Unary math ops (neg, abs, square)
- `include/ops/UnaryOps/Exponents.h` — exp, log, pow
- `include/ops/UnaryOps/Trigonometry.h` — sin, cos, tan
- `include/ops/UnaryOps/Conjugate.h` — Complex conjugate
- `include/ops/UnaryOps/Reduction.h` — Sum, mean, min, max, etc.
- `include/ops/helpers/ReductionOps.h` — Reduction operator structs
- `include/ops/helpers/ConditionalOps.h` — Where function
- `include/core/TensorDispatch.h` — Core dispatch infrastructure
- `src/core/TensorFactory.cpp` — tensor creation, zeros, ones, rand
- `src/core/AsTypeTensor.cpp` — `.astype()` casting

---

#### Step 6: GPU Kernel Files (`.cu` files)

CUDA kernel instantiations must include the new types:
```cpp
// In each .cu file, add template instantiation:
template __global__ void kernel<uint8_t>(...);
template __global__ void kernel<float8_e4m3fn_t>(...);
template __global__ void kernel<complex64_t>(...);
```

**GPU-specific files:**
- `src/TensorOps/cuda/TensorOpsAdd.cu` (and Sub, Mul, Div)
- `src/UnaryOps/cuda/Arithmetics.cu`
- `src/UnaryOps/cuda/Exponents.cu`
- `src/UnaryOps/cuda/Trigonometry.cu`
- `src/UnaryOps/cuda/ReductionImplGPU.cu`
- `src/Kernels/cuda/ConversionKernels.cu` — dtype casting on GPU
- `src/compiler/cuda/ConditionalOps.cu`

---

#### Step 7: Add to `.astype()` Conversion

The `AsTypeTensor.cpp` handles runtime type casting (`tensor.astype(Dtype::UInt8)`). Must add conversion cases for every type pair.

---

### 2.2 Integration Difficulty by Type

| Type | Difficulty | Why |
|------|:---:|------|
| `Int8` | Easy | Native C++ type. Just enum + traits + dispatch |
| `UInt8/16/32/64` | Easy | Native C++ types. Same pattern ×4 |
| `Bool` | Medium | Special case: weakest in promotion, division always → Float32 |
| `Float8_E4M3FN` | Hard | Custom struct + bitwise conversion + RNE rounding + denormal magic + CUDA native operator overloads |
| `Float8_E5M2` | Medium | Custom struct, but leverages FP16 shortcut for one conversion direction |
| `Complex32/64/128` | Hard | Custom structs ×3 + complex arithmetic (mul/div formulas) + math overloads + `is_complex` predicate + special promotion rules |

---

## 3. Numerical Stability

### 3.1 FP8 Precision Limitations

| Comparison | E4M3FN | E5M2 |
|------------|--------|------|
| Exponent bits | 4 | 5 |
| Mantissa bits | 3 | 2 |
| Max value | ~448 | ~57344 |
| Precision (decimal digits) | ~1 | < 1 |
| Infinity? | ❌ No | ✅ Yes |
| NaN? | ✅ (all bits = 1) | ✅ (standard IEEE) |
| Best for | Forward pass (weights/activations) | Backward pass (gradients) |

**Why two FP8 formats?**
- Forward pass needs precision for weights → E4M3FN (3 mantissa bits)
- Backward pass needs range for gradients → E5M2 (5 exponent bits, wider range)
- Mixing them (E4M3 + E5M2) throws a runtime error — they can't be implicitly promoted

### 3.2 CUDA Native FP8 Types Have No Operators

NVIDIA's `__nv_fp8_e4m3` and `__nv_fp8_e5m2` types (from `<cuda_fp8.h>`) provide conversion intrinsics but **NO arithmetic or comparison operators**. We had to manually provide:
- Arithmetic: `+`, `-`, `*`, `/`, unary `-`
- Compound: `+=`, `-=`, `*=`, `/=`
- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`

All using the promote-compute-demote pattern via `float`.

### 3.3 Complex32 Precision Warning

`complex32_t` uses `float16_t` components (10-bit mantissa). Complex multiplication does 4 multiplications + 2 additions in FP16, causing error accumulation. Use `complex64_t` (float components) for any serious computation.

### 3.4 `safe_pow()` Edge Cases

```
safe_pow(base, exponent):
    NaN input         → NaN output
    0^0               → 1 (convention)
    0^(negative)      → +Infinity
    0^(positive)      → 0
    negative^(non-int) → NaN (complex result not representable in real)
```

---

## 4. Memory Layout

### 4.1 Storage Sizes and Bandwidth Savings

| Type | Bytes/Element | Bandwidth vs FP32 |
|------|:---:|:---:|
| `bool` | 1 | 4× savings |
| `uint8_t` | 1 | 4× savings |
| `float8_e4m3fn_t` | 1 | 4× savings |
| `float8_e5m2_t` | 1 | 4× savings |
| `complex32_t` | 4 (2×FP16) | Same as FP32 |
| `complex64_t` | 8 (2×FP32) | 2× cost |
| `complex128_t` | 16 (2×FP64) | 4× cost |

### 4.2 FP8 Bit Layout

```
E4M3FN (8 bits):
┌───┬────────┬───────┐
│ S │  EEEE  │  MMM  │
│ 1 │   4    │   3   │
└───┴────────┴───────┘
Bias = 7

E5M2 (8 bits):
┌───┬─────────┬──────┐
│ S │  EEEEE  │  MM  │
│ 1 │    5    │   2  │
└───┴─────────┴──────┘
Bias = 15
```

### 4.3 Complex Memory Layout (Interleaved)

```
complex64_t tensor of shape [3]:
Memory: [re0:4B][im0:4B] [re1:4B][im1:4B] [re2:4B][im2:4B]
         └── 8B ────┘     └── 8B ────┘     └── 8B ────┘
Total = 24 bytes
```

### 4.4 Why Bool is 1 Byte, Not 1 Bit

C++ `sizeof(bool) == 1`. Bit-packing would require shift+mask on every access, destroying cache performance and breaking GPU coalesced memory access patterns. The 8× memory overhead is worth the performance.

---

## 5. Pseudocode / Program Logic

### 5.1 Adding a New Dtype (Step-by-Step Pseudocode)

```
TO ADD NEW TYPE "NewType":

1. Dtype.h:
   Add "NewType" to enum class Dtype { ... }

2. Types.h (if custom struct needed):
   Write struct new_type_t { uint_bits raw_bits; ... }
   Write conversion functions: float_to_new(), new_to_float()
   Write math overloads: abs(), sqrt(), exp(), log(), etc.
   Write std::numeric_limits<new_type_t> specialization
   Write CUDA native operator overloads (if GPU type exists)

3. DtypeTraits.h:
   Add dtype_traits<Dtype::NewType> specialization
   Add branch to is_same_type<T>()
   Add branch to type_to_dtype<T>()
   Add to appropriate predicate (is_float/is_int/is_unsigned/is_complex/is_bool)
   Add to get_dtype_name()
   Add ROW + COLUMN to promotion_table[18][18]      → becomes [19][19]
   Add ROW + COLUMN to scalar_tensor_table[18][18]  → becomes [19][19]

4. DtypeCastUtils.h:
   Add to get_promoted_dtype() switch
   Add conversion helper functions if needed

5. Operation dispatch (ALL switch statements):
   Add case Dtype::NewType: op<new_type_t>(); break;
   → In EVERY .h and .cpp/.cu file that dispatches by dtype

6. GPU kernels:
   Add template instantiation for new type in every .cu file

7. AsTypeTensor.cpp:
   Add conversion cases to/from new type
```

### 5.2 E5M2 → Float (The Clever Shortcut)

```
FUNCTION e5m2_to_float(input: uint8) → float32:
    // E5M2 has same exponent layout as FP16!
    fp16_bits = input << 8       // Pad mantissa with 8 zeros
    RETURN float16_to_float(fp16_bits)  // Reuse existing converter
```

### 5.3 Complex Multiplication

```
FUNCTION complex_mul(a: complex, b: complex) → complex:
    real = a.re × b.re - a.im × b.im
    imag = a.re × b.im + a.im × b.re
    RETURN complex(real, imag)
```

### 5.4 Complex Division

```
FUNCTION complex_div(a: complex, b: complex) → complex:
    denom = b.re² + b.im²
    real = (a.re × b.re + a.im × b.im) / denom
    imag = (a.im × b.re - a.re × b.im) / denom
    RETURN complex(real, imag)
```

---

## 6. Research Material

### 6.1 Implementation References

| Source | What We Used It For |
|--------|-------------------|
| **PyTorch `c10/util/Float8_e4m3fn.h`** | Direct reference for FP8 E4M3FN conversion algorithm (magic numbers, denormal handling) |
| **PyTorch `c10/util/Float8_e5m2.h`** | E5M2 conversion — inspired the FP16 shortcut trick |
| **NVIDIA `<cuda_fp8.h>`** | Discovered that native CUDA FP8 types lack operators; we wrote them ourselves |
| **IEEE 754 / C++ `<complex>`** | Standards for complex number arithmetic and math functions |
| **NumPy promotion rules** | Basis for UInt + Int promotion (e.g., UInt8 + Int8 → Int16) |
| **PyTorch dtype promotion** | Division always promotes to float; Bool is weakest type |

### 6.2 Comparison with PyTorch

| Capability | OwnTensor Library | PyTorch |
|---------|-----------|---------|
| FP8 support | ✅ First-class custom structs (E4M3, E5M2) | ✅ `torch.float8_e5m2`, `torch.float8_e4m3fn` |
| Unsigned integers | ✅ Full set: UInt8–UInt64 | ⚠️ Only `torch.uint8` (limited support above) |
| Complex types | ✅ complex32/64/128 with full arithmetic | ✅ `torch.complex32/64/128` |
| Promotion tables | ✅ O(1) lookup (18×18 precomputed) | ✅ `torch.result_type()` |
| CUDA FP8 operators | ✅ Manually written (NVIDIA doesn't provide them!) | ✅ Same approach |
| FP8 mixing (E4M3 + E5M2) | ❌ Runtime error (by design) | ❌ Same — no implicit promotion |
| Division promotion | ✅ Int→Float32 always | ✅ Same behavior |

### 6.3 Source Files Reference

| File | Lines | What's Inside |
|------|:---:|------|
| `Dtype.h` | 15 | Enum with 18 types |
| `Types.h` | 1661 | All custom structs + conversion functions + math overloads + numeric_limits + CUDA operator overloads |
| `DtypeTraits.h` | 842 | Traits, predicates, type mappings, 18×18 promotion tables (tensor-tensor + scalar-tensor) |
| `DtypeCastUtils.h` | 163 | Runtime conversion helpers, safe_pow |
