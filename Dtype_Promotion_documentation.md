# Dtype Promotion — Complete Documentation
## Implementation & Integration of the Type Promotion System

---

## 0. What is Dtype Promotion?

When two tensors of **different types** meet in an operation, the library must decide: **what type should the result be?**

```
int16_tensor + float32_tensor = ???_tensor
```

This decision is called **dtype promotion**. We implement three promotion functions, each for a different scenario:

| Function | When It's Called | Example |
|----------|-----------------|---------|
| `promote_tensor_ops(a, b)` | Tensor + Tensor binary ops (add, sub, mul) | `Int16 + Float32 → Float32` |
| `promote_scalar_ops(t, s)` | Tensor + C++ Scalar ops | `Float16_tensor + 3.14 → Float16` |
| `promote_dtypes_division(a, b)` | Any division operation | `Int32 / Int32 → Float32` |

Plus two unary promotion functions:

| Function | When It's Called | Example |
|----------|-----------------|---------|
| `get_promoted_dtype(dt)` | Unary math ops (exp, log, sqrt) | `exp(Int32_tensor) → Float32` |
| `get_promoted_dtype_square(dt)` | Square operation specifically | `square(Int16_tensor) → Float64` |

---

## 1. Algorithm Design

### 1.1 Tensor + Tensor Promotion (`promote_tensor_ops`)

Uses an **18×18 precomputed lookup table** for O(1) resolution. The rules encoded:

**Rule hierarchy (highest priority first):**

1. **Same type → return that type** (diagonal of the table)
2. **Complex wins everything:** `Complex128 > Complex64 > Complex32`
3. **Float + Complex → promote based on float precision:**
   - `Float16 + Complex → Complex32`
   - `Float32 + Complex → Complex64`
   - `Float64 + Complex → Complex128`
4. **Float beats Int:** `Float64 > Float32 > Float16/BFloat16`
5. **Float16 + BFloat16 → Float32** (different underlying formats, can't mix)
6. **Int promotion:** `Int64 > Int32 > Int16 > Int8`
7. **UInt + UInt:** larger wins (`UInt64 > UInt32 > UInt16 > UInt8`)
8. **UInt + Int (NumPy-style):**
   ```
   UInt8  + Int8   → Int16   (need >8 bits signed)
   UInt8  + Int16  → Int16
   UInt8  + Int32  → Int32
   UInt16 + Int8   → Int32   (need >16 bits signed)
   UInt16 + Int16  → Int32
   UInt32 + Int8   → Int64   (need >32 bits signed)
   UInt32 + Int32  → Int64
   UInt64 + Int*   → Float32 (no integer can hold both ranges!)
   ```
9. **Bool is weakest:** promotes to whatever the other type is
10. **FP8 special handling:**
    - `FP8 + Int/Bool → FP8 wins`
    - `FP8 + Float16/32/64/BFloat16 → ERROR` (no implicit promotion)
    - `E4M3FN + E5M2 → ERROR` (mixing FP8 formats forbidden)

### 1.2 Tensor + Scalar Promotion (`promote_scalar_ops`)

**Key difference from tensor-tensor:** Scalars are **weak** — they don't upgrade floating-point tensors.

```
Float16_tensor + float64_scalar → Float16  (tensor wins!)
```

But complex scalars always promote (to preserve data):
```
Float16_tensor + complex_scalar → Complex32
Float32_tensor + complex_scalar → Complex64
Float64_tensor + complex_scalar → Complex128
```

**Why this difference?** In deep learning, when you write `tensor * 0.5`, you don't want your carefully chosen FP16 tensor to suddenly blow up to Float64. The scalar `0.5` is "just a number" — the tensor's type should dominate.

Also uses an **18×18 lookup table** (`scalar_tensor_table`) but with different rules:
- Float tensor + any non-complex scalar → tensor wins
- Complex tensor + anything → tensor wins (always)
- Bool tensor + any scalar → scalar wins (bool is weakest)
- UInt tensor + signed Int scalar → **ERROR** (ambiguous, user must cast explicitly)
- Int tensor + larger Int scalar → **ERROR** (prevents silent overflow)

### 1.3 Division Promotion (`promote_dtypes_division`)

**Critical rule: Division ALWAYS promotes to float.** Prevents destructive integer truncation.

```
Int32 / Int32 → Float32  (NOT Int32! Python-style true division)
Int64 / Int64 → Float32
Bool  / Bool  → Float32
```

**Logic flow:**
```
1. If either operand is Complex → use normal promote_tensor_ops()
2. If either is Float64 → Float64
3. If either is Float32 → Float32
4. If either is Float16 → Float16
5. If either is BFloat16 → BFloat16
6. FP8 handling (same as tensor-tensor for errors)
7. Otherwise (both are Int/Bool) → Float32
```

### 1.4 Unary Promotion (`get_promoted_dtype`)

For single-tensor math operations (exp, log, sqrt, sin, cos...):

```
get_promoted_dtype(Int16)      → Float32   (math needs float!)
get_promoted_dtype(Int32)      → Float32
get_promoted_dtype(Int64)      → Float64   (preserve range)
get_promoted_dtype(UInt8)      → Float32
get_promoted_dtype(UInt64)     → Float64
get_promoted_dtype(Bool)       → Float32
get_promoted_dtype(Float32)    → Float32   (no change)
get_promoted_dtype(Complex64)  → Complex64 (no change)
```

**Why?** You can't compute `exp(5)` in integer — the result `148.41` isn't an integer. So integers are promoted to float before the operation.

### 1.5 Square Promotion (`get_promoted_dtype_square`)

Specifically for the `square()` operation — promotes integers to **Float64** (not Float32):
```
get_promoted_dtype_square(Int16) → Float64
get_promoted_dtype_square(Int32) → Float64
get_promoted_dtype_square(UInt8) → Float64
```

**Why Float64 for square?** `square(INT32_MAX)` = `(2^31)^2 = 2^62`, which overflows Float32's 24-bit mantissa. Float64 has 53 bits of mantissa, enough to hold it precisely.

---

## 2. System Design

### 2.1 The Lookup Table Architecture

Instead of nested if-else chains (slow, error-prone), we use **compile-time constant arrays**:

```cpp
constexpr int DTYPE_COUNT = 18;

// Shorthand indices for readability:
constexpr int I8 = 0, I16 = 1, I32 = 2, I64 = 3;
constexpr int U8 = 4, U16 = 5, U32 = 6, U64 = 7;
constexpr int BF16 = 8, F16 = 9, F32 = 10, F64 = 11;
constexpr int BOOL = 12;
constexpr int C32 = 13, C64 = 14, C128 = 15;
constexpr int FP8_E4 = 16, FP8_E5 = 17;

constexpr int promotion_table[18][18] = {
    // 18 rows × 18 columns — every possible type pair
    // Example row for Int8:
    //       I8  I16 I32 I64 U8  U16 U32 U64 BF16 F16 F32 F64 BOOL C32 C64 C128 E4  E5
    /*I8*/ { I8, I16,I32,I64,I16,I32,I64,F32,BF16,F16,F32,F64, I8, C32,C64,C128,FP8_E4,FP8_E5},
    // ... 17 more rows
};
```

**Lookup is O(1):**
```cpp
inline Dtype promote_tensor_ops(Dtype a, Dtype b) {
    int result = promotion_table[static_cast<int>(a)][static_cast<int>(b)];
    
    // Handle special error markers
    if (result == ERR_MIXED_FP8) throw "Can't mix E4M3 + E5M2";
    if (result == ERR_FP8_HIGHER) throw "FP8 can't promote with higher floats";
    
    return static_cast<Dtype>(result);
}
```

**Error markers** (negative values that can't be valid Dtype indices):
```cpp
constexpr int ERR_MIXED_FP8    = -1;  // E4M3 + E5M2
constexpr int ERR_FP8_HIGHER   = -2;  // FP8 + Float16/32/64
constexpr int ERR_S_UINT_INT   = -4;  // UInt tensor + Int scalar
constexpr int ERR_S_UINT_LARGER = -5; // UInt tensor + larger UInt scalar
constexpr int ERR_S_INT_LARGER = -6;  // Int tensor + larger Int scalar
```

### 2.2 How Promotion Integrates Into Operations

#### Pattern 1: Tensor + Tensor (in `TensorOps.cpp`)

```cpp
Tensor add(const Tensor& lhs, const Tensor& rhs) {
    // STEP 1: Determine result dtype
    Dtype promoted_dtype = promote_tensor_ops(lhs.dtype(), rhs.dtype());
    
    // STEP 2: Convert both inputs to promoted dtype (if needed)
    Tensor lhs_promoted = promote_if_needed(lhs, promoted_dtype);
    Tensor rhs_promoted = promote_if_needed(rhs, promoted_dtype);
    
    // STEP 3: Create output tensor with promoted dtype
    Tensor output(output_shape, promoted_dtype, lhs.device(), ...);
    
    // STEP 4: Execute the operation in the promoted type
    apply_binary_operation(lhs_promoted, rhs_promoted, output, 
        [](auto a, auto b) { return a + b; });
    
    return output;
}
```

`promote_if_needed()` helper:
```cpp
static Tensor promote_if_needed(const Tensor& input, Dtype target_dtype) {
    if (input.dtype() == target_dtype) return input;     // No conversion needed
    return input.as_type(target_dtype);                   // Convert via .astype()
}
```

#### Pattern 2: Tensor + Scalar (in `ScalarOpsDispatcher.h`)

```cpp
template<typename S>
Tensor operator+(const Tensor& a, S s) {
    // STEP 1: Get scalar's dtype at COMPILE TIME
    constexpr Dtype scalar_dt = type_to_dtype<S>();
    
    // STEP 2: Determine result dtype (scalar is weak!)
    const Dtype promoted_dt = promote_scalar_ops(a.dtype(), scalar_dt);
    
    // STEP 3: If tensor needs promotion, convert it
    const Tensor& src = (promoted_dt == a.dtype()) ? a : a.as_type(promoted_dt);
    
    // STEP 4: Convert scalar to match promoted type and compute
    dispatch_by_dtype(promoted_dt, [&](auto dummy) {
        using T = decltype(dummy);
        T val = convert_scalar<T>(s);      // ← Scalar conversion helper
        // ... element-wise loop: out[i] = in[i] + val
    });
}
```

#### Pattern 3: Division (special case in `TensorOps.cpp`)

```cpp
Tensor div(const Tensor& lhs, const Tensor& rhs) {
    // Uses promote_dtypes_division() instead of promote_tensor_ops()
    Dtype promoted_dtype = promote_dtypes_division(lhs.dtype(), rhs.dtype());
    // Int32 / Int32 → promoted_dtype is Float32!
    // ... rest follows same pattern
}
```

#### Pattern 4: In-Place Operations (error if promotion needed)

```cpp
template<typename S>
Tensor& operator+=(Tensor& t, S s) {
    const Dtype promoted_dt = promote_scalar_ops(t.dtype(), scalar_dt);
    
    // In-place can't change dtype! If promotion is different, ERROR.
    if (promoted_dt != t.dtype()) {
        throw std::runtime_error("In-place +=: type mismatch. Use outplace +.");
    }
    // ... proceed only if types already match
}
```

**Why?** In-place operations modify the tensor's data buffer directly. You can't change a `Float16` buffer into a `Float32` buffer in-place — you'd need a completely new allocation. So if the scalar would force promotion, we throw an error telling the user to use the out-of-place version instead.

### 2.3 The `convert_scalar<T>(s)` Helper

When `dispatch_by_dtype` instantiates ALL 18 type combinations at compile time, some combinations don't make sense (e.g., converting a `complex64_t` scalar to `uint8_t`). The `convert_scalar` helper uses `if constexpr` to make all combinations compile-safe:

```cpp
template<typename T, typename S>
inline T convert_scalar(S s) {
    if constexpr (is_complex_type_v<T> && is_complex_type_v<S>) {
        // Both complex: convert each component
        return T(static_cast<RealT>(s.real()), static_cast<RealT>(s.imag()));
    } else if constexpr (is_complex_type_v<T> && !is_complex_type_v<S>) {
        // Real scalar → complex: set imaginary = 0
        return T(static_cast<RealT>(s), RealT(0));
    } else if constexpr (!is_complex_type_v<T> && is_complex_type_v<S>) {
        // Complex to real: take real part (shouldn't happen at runtime due to promotion)
        return static_cast<T>(s.real());
    } else {
        // Both real: simple cast
        return static_cast<T>(s);
    }
}
```

### 2.4 The `convert_half_to_float32()` CPU Path

For half-precision types (FP16, BF16), many math operations need promotion to Float32 for computation. This utility converts element-by-element:

```cpp
inline Tensor convert_half_to_float32(const Tensor& input) {
    Tensor temp(input.shape(), Dtype::Float32, ...);
    float* temp_ptr = temp.data<float>();
    
    if (input.dtype() == Dtype::Float16) {
        const float16_t* in = input.data<float16_t>();
        for (size_t i = 0; i < input.numel(); ++i)
            temp_ptr[i] = static_cast<float>(in[i]);  // uses float16_t::operator float()
    } else { // BFloat16
        const bfloat16_t* in = input.data<bfloat16_t>();
        for (size_t i = 0; i < input.numel(); ++i)
            temp_ptr[i] = static_cast<float>(in[i]);  // uses bfloat16_t::operator float()
    }
    return temp;
}
```

And the reverse: `convert_float32_to_half()` to convert results back.

Similarly for complex: `convert_complex32_to_complex64()` and `convert_complex64_to_complex32()`.

---

## 3. Numerical Stability

### 3.1 Why Division MUST Promote to Float

```
int a = 7, b = 3;
int result = a / b;     // = 2  (WRONG! Real answer is 2.333...)
float result = (float)a / (float)b;  // = 2.333  (CORRECT)
```

In neural network training, integer division would silently destroy gradient information: `gradient / batch_size` rounding to 0 would kill training. Our `promote_dtypes_division()` guarantees this never happens.

### 3.2 UInt64 + Int64 → Float32 (Lossy but Necessary)

No integer type can hold both `uint64` (0 to 2^64-1) and `int64` (-2^63 to 2^63-1) ranges. We promote to `Float32` as a practical deep learning compromise — though `Float32` only has 24-bit mantissa precision. For exact arithmetic, user should explicitly cast to `Float64`.

### 3.3 FP8 Mixing Is Forbidden

E4M3FN and E5M2 have incompatible precision/range tradeoffs. Automatic promotion would be ambiguous — which format "wins"? Neither can represent the other's full range without loss. We throw a runtime error with a helpful message:

```
"Input dtypes ('float8_e5m2', 'float8_e4m3fn') have no available implicit dtype 
 promotion path. Use explicit x.astype('float32')."
```

### 3.4 In-Place Type Mismatch Safety

In-place operations (`+=`, `-=`, etc.) refuse to silently change tensor dtype. This prevents subtle bugs:
```cpp
float16_tensor += complex_scalar;  
// Without check: would silently lose imaginary data!
// With check: throws error telling user to use out-of-place version
```

---

## 4. Memory Layout

### 4.1 Promotion's Impact on Memory

Every promotion creates a **new tensor** with potentially larger element size:

```
Int16 tensor (1000 elements):  2 KB
  → promoted to Float32:       4 KB  (2× memory)
  → promoted to Complex128:    16 KB (8× memory!)
```

For the `promote_if_needed()` path, if `input.dtype() == target_dtype`, **no copy happens** — the input tensor is returned by reference.

### 4.2 Lookup Table Memory

Both promotion tables are `constexpr` — placed in **read-only data segment** at compile time:
```
promotion_table[18][18]:      324 × 4 bytes = 1,296 bytes
scalar_tensor_table[18][18]:  324 × 4 bytes = 1,296 bytes
Total: ~2.6 KB (negligible)
```

---

## 5. Pseudocode / Program Logic

### 5.1 Complete Tensor Binary Operation Flow

```
FUNCTION tensor_add(lhs: Tensor, rhs: Tensor) → Tensor:
    
    // ── PHASE 1: Dtype Resolution ──
    promoted_dtype = promotion_table[lhs.dtype][rhs.dtype]
    IF promoted_dtype is ERROR_MARKER:
        THROW appropriate error message
    
    // ── PHASE 2: Input Promotion ──
    IF lhs.dtype != promoted_dtype:
        lhs_promoted = lhs.astype(promoted_dtype)  // Creates new tensor
    ELSE:
        lhs_promoted = lhs  // No copy, same tensor
    
    // Same for rhs...
    
    // ── PHASE 3: Shape Resolution ──
    output_shape = broadcast_shape(lhs_promoted.shape, rhs_promoted.shape)
    
    // ── PHASE 4: Allocate Output ──
    output = new Tensor(output_shape, promoted_dtype, device)
    
    // ── PHASE 5: Dispatch + Compute ──
    dispatch_by_dtype(promoted_dtype):
        FOR i in 0..numel:
            output[i] = lhs_promoted[i] + rhs_promoted[i]
    
    RETURN output
```

### 5.2 Complete Scalar Operation Flow

```
FUNCTION tensor_add_scalar<S>(tensor: Tensor, scalar: S) → Tensor:
    
    // ── PHASE 1: Compile-time scalar dtype ──
    scalar_dt = type_to_dtype<S>()  // Known at compile time!
    
    // ── PHASE 2: Scalar-Tensor Promotion ──
    promoted_dt = scalar_tensor_table[tensor.dtype][scalar_dt]
    IF promoted_dt is ERROR_MARKER:
        THROW error
    
    // ── PHASE 3: Promote tensor if needed ──
    IF promoted_dt != tensor.dtype:
        src = tensor.astype(promoted_dt)
    ELSE:
        src = tensor
    
    // ── PHASE 4: Convert scalar to match ──
    val = convert_scalar<PromotedType>(scalar)
    
    // ── PHASE 5: Compute ──
    output = new Tensor(src.shape, promoted_dt)
    FOR i in 0..numel:
        output[i] = src[i] + val
    
    RETURN output
```

---

## 6. The Full 18×18 Tensor-Tensor Promotion Table

```
Legend: I8=Int8, I16=Int16, I32=Int32, I64=Int64
        U8=UInt8, U16=UInt16, U32=UInt32, U64=UInt64
        B16=BFloat16, F16=Float16, F32=Float32, F64=Float64
        Bo=Bool, C32=Complex32, C64=Complex64, C128=Complex128
        E4=Float8_E4M3FN, E5=Float8_E5M2, X=ERROR

     I8  I16 I32 I64 U8  U16 U32 U64 B16 F16 F32 F64 Bo  C32 C64 C128 E4  E5
I8 [ I8  I16 I32 I64 I16 I32 I64 F32 B16 F16 F32 F64 I8  C32 C64 C128 E4  E5 ]
I16[ I16 I16 I32 I64 I16 I32 I64 F32 B16 F16 F32 F64 I16 C32 C64 C128 E4  E5 ]
I32[ I32 I32 I32 I64 I32 I32 I64 F32 B16 F16 F32 F64 I32 C64 C64 C128 E4  E5 ]
I64[ I64 I64 I64 I64 I64 I64 I64 F32 B16 F16 F32 F64 I64 C64 C64 C128 E4  E5 ]
U8 [ I16 I16 I32 I64 U8  U16 U32 U64 B16 F16 F32 F64 U8  C32 C64 C128 E4  E5 ]
U16[ I32 I32 I32 I64 U16 U16 U32 U64 B16 F16 F32 F64 U16 C64 C64 C128 E4  E5 ]
U32[ I64 I64 I64 I64 U32 U32 U32 U64 B16 F16 F32 F64 U32 C64 C64 C128 E4  E5 ]
U64[ F32 F32 F32 F32 U64 U64 U64 U64 B16 F16 F32 F64 U64 C64 C64 C128 E4  E5 ]
B16[ B16 B16 B16 B16 B16 B16 B16 B16 B16 F32 F32 F64 B16 C64 C64 C128 X   X  ]
F16[ F16 F16 F16 F16 F16 F16 F16 F16 F32 F16 F32 F64 F16 C32 C64 C128 X   X  ]
F32[ F32 F32 F32 F32 F32 F32 F32 F32 F32 F32 F32 F64 F32 C64 C64 C128 X   X  ]
F64[ F64 F64 F64 F64 F64 F64 F64 F64 F64 F64 F64 F64 F64 C128C128C128 X   X  ]
Bo [ I8  I16 I32 I64 U8  U16 U32 U64 B16 F16 F32 F64 Bo  C32 C64 C128 E4  E5 ]
C32[ C32 C32 C64 C64 C32 C64 C64 C64 C64 C32 C64 C128C32 C32 C64 C128 X   X  ]
C64[ C64 C64 C64 C64 C64 C64 C64 C64 C64 C64 C64 C128C64 C64 C64 C128 X   X  ]
C128[C128C128C128C128C128C128C128C128C128C128C128C128C128C128C128C128 X   X  ]
E4 [ E4  E4  E4  E4  E4  E4  E4  E4  X   X   X   X   E4  X   X   X   E4   X  ]
E5 [ E5  E5  E5  E5  E5  E5  E5  E5  X   X   X   X   E5  X   X   X   X   E5  ]
```

---

## 7. Research Material

### 7.1 Design References

| Source | What We Derived |
|--------|----------------|
| **PyTorch `torch.result_type()`** | Overall promotion hierarchy: Complex > Float > Int > Bool |
| **PyTorch true division** | Int / Int → Float (Python 3 semantics) |
| **NumPy uint+int rules** | UInt8 + Int8 → Int16 (safe signed widening) |
| **PyTorch scalar weakness** | Scalars don't upgrade float tensor types |
| **FP8 standards (OCP)** | E4M3 and E5M2 must not be implicitly mixed |

### 7.2 Source Files Reference

| File | What's Inside |
|------|------|
| `DtypeTraits.h` (lines 414–710) | Both 18×18 lookup tables, `promote_tensor_ops()`, `promote_scalar_ops()`, `promote_dtypes_division()`, error markers |
| `DtypeCastUtils.h` (160 lines) | `get_promoted_dtype()`, `get_promoted_dtype_square()`, `convert_half_to_float32()`, `convert_float32_to_half()`, `convert_complex32_to_complex64()`, `safe_pow()` |
| `ScalarOpsDispatcher.h` (585 lines) | `convert_scalar<T>()`, all scalar operator implementations (`+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `>`, `<=`, `>=`), both in-place and copy versions, with full promotion integration |
| `TensorOps.cpp` (lines 15–220) | `promote_if_needed()`, tensor-tensor add/sub/mul/div/mod with promotion |

### 7.3 Our Design vs PyTorch

| Aspect | Our Implementation | PyTorch |
|--------|-------------------|---------|
| Lookup complexity | O(1) table lookup | O(1) (similar internal dispatch) |
| 0-dim tensor handling | Treated same as regular tensors | 3-tier priority (dim > 0 strongest, 0-dim middle, scalar weakest) |
| UInt support | Full: UInt8–UInt64 with NumPy-style rules | Limited: mostly UInt8 only |
| FP8 promotion | Error on mixing E4M3/E5M2 | Same behavior |
| In-place type mismatch | Runtime error with helpful message | Same — throws error |
| Division promotion | Always to float | Same (Python 3 semantics) |
