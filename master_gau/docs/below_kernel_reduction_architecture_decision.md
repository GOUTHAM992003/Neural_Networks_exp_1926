# Below-Kernel Reduction Architecture Decision
## PyTorch vs Eigen Approach — Which Suits Our Library?

---

## PyTorch vs Eigen Approach

### What Eigen's compile-time dispatch actually does

```cpp
// Eigen: reduction axes are a COMPILE-TIME template parameter
tensor.reduce(Eigen::array<int, 2>{0, 1});
//                          ^^^^^^^^^^^^^ known at compile time

// So this resolves at compile time:
static constexpr bool ReducingInnerMostDims = are_inner_most_dims<Dims, N, Layout>::value;
// → completely different code path compiled in, zero runtime branch
```

### What PyTorch's runtime dispatch does

```cpp
// After reorder_dimensions() + coalesce_dimensions():
// strides are now simple — just read 3 integers:
if (strides[0]==0 && strides[1]==sizeof(T))
    → vectorized_inner_reduction   // horizontal SIMD
else if (strides[0]==0 && strides[2]==sizeof(T) && strides[3]==sizeof(T))
    → vectorized_outer_reduction   // vertical SIMD
else
    → basic_loop                   // scalar fallback
```

Cost of that runtime check: **3 integer comparisons, ~1ns**. The kernel itself runs for microseconds to milliseconds. The dispatch overhead is completely negligible.

---

## Can We Do Eigen-Style Templates with PyTorch Logic? The Problem

**Our reduction axes come in at runtime** — user calls `reduce({axis_0=0, axis_1=2})` via a `std::vector<int64_t>`. You cannot template on a runtime value.

```cpp
// What Eigen needs (compile-time):
reduce<Eigen::array<int,2>{0,1}>(tensor)   // axis is a TYPE

// What our API looks like (runtime):
reduce(tensor, {0, 1})                     // axis is a VALUE at runtime
```

Eigen's compile-time dispatch only fires when the reduction axes are baked into the type system. When TensorFlow uses Eigen with **runtime** axes (which it often does), Eigen internally falls to `GenericDimReducer` — the scalar fallback, not the SIMD paths.

---

## The Critical Issue: Our Library is a Pre-Compiled `.so`

Eigen is **header-only** — it compiles all template specializations into the **user's binary** at the user's build time. The user's compiler sees the actual types and generates exactly the right code.

Our library is a pre-compiled `.so`. To do Eigen-style compile-time dispatch, we'd need to explicitly pre-instantiate every combination:

```
types × axes_combinations × op_types × inner/outer/full =
8 dtypes × 2^8 axis combos × 8 ops × 3 cases
= thousands of explicit instantiations
```

This causes:
- **Massive binary bloat** — tens of MB just for reduction kernels
- **Extremely long compile times** when building the library
- **No actual runtime benefit** over PyTorch's 3-integer runtime check

---

## What Eigen-Style Templates DO Give (Worth Keeping)

The **SIMD operations within a kernel** can and should be compile-time via templates:

```cpp
// This IS compile-time — resolved based on build flags (AVX2/AVX-512):
template<typename T>
void vectorized_inner_reduction(T* in, T* out, int64_t n) {
    using Vec = Vectorized<T>;         // maps to __m256 or __m512 at compile time
    Vec acc[4];                        // 4 compile-time-typed SIMD registers
    // loop body is fully templated, no runtime SIMD dispatch
}
```

The **case selection** (which of the 3 paths to run) is runtime — cheap. The **SIMD operations inside** the selected path are compile-time — fast. This is exactly PyTorch's actual design.

---

## Verdict: PyTorch's Approach

| Concern | Eigen compile-time | PyTorch runtime |
|---|---|---|
| Axes known at runtime? | ✗ — needs compile-time axes | ✓ — works with any axes |
| Pre-compiled `.so` compatible? | ✗ — needs user's compiler | ✓ — runtime strides work |
| Dispatch overhead? | Zero | ~1ns (3 integer compares) |
| SIMD still compile-time? | Yes | Yes (same — within kernel) |
| Binary size | Explosion (thousands of instantiations) | Manageable |
| Matches our API design? | No — would need to change caller API | Yes |

**The 1ns runtime dispatch overhead is completely irrelevant.**
After the kernel runs for even 10µs on a 1000-element tensor, that is 0.01% overhead. For million-element tensors it is essentially 0%.

---

## Implementation Plan (PyTorch approach applied to our library)

```
Step 1 — reorder_dimensions() + coalesce_dimensions()          [BIGGEST WIN]
    Eliminates division/modulo per element in the inner loop.
    From ~40 cycles per element to ~3 cycles (stride multiply replaces mod+div).
    A [1000, 1, 500] tensor reduces to [500000] — single flat loop.

Step 2 — Runtime stride check (3 integer compares — essentially free)
    is_inner_reduction?   → horizontal SIMD path
    is_outer_reduction?   → vertical SIMD path
    else                  → scalar fallback (stride multiply, no division)

Step 3 — vectorized_inner_reduction with 4×Vec unroll (horizontal SIMD)
    For contiguous inner dim (row-reduce: reduce [N, M] along axis=1).
    AVX2 float: processes 32 floats per iteration with 4 independent accumulators.
    4 accumulators break FP dependency chain — uses all 4 FP execution units.

Step 4 — vectorized_outer_reduction with 4×Vec vertical (vertical SIMD)
    For contiguous outer dim (column-reduce: reduce [N, M] along axis=0).
    Each SIMD lane accumulates one column independently.
    No horizontal reduction at end — direct store to output.

Step 5 — Scalar fallback for non-contiguous
    Same as current approach but WITHOUT division — uses precomputed stride multiply
    after coalesce_dimensions() simplifies the layout.
```

**Step 1 alone (coalesce + reorder) is probably a 5–10x speedup** on many common shapes
just from eliminating the division/modulo in the hot loop.
Steps 3+4 then add another 4–8x on top from SIMD.

---

## Why Each Step Matters

### Step 1: coalesce_dimensions() + reorder_dimensions() — the foundation

Current inner loop per element (our code):
```cpp
// ~40 cycles per element per dimension:
int64_t tmp = i;
for (int d = ndim - 1; d >= 0; --d) {
    full_input_coords[d] = tmp % reduce_shape[d];   // division: ~20 cycles
    tmp /= reduce_shape[d];                          // division: ~20 cycles
}
int64_t input_lin_idx = 0;
for (int d = 0; d < ndim; ++d)
    input_lin_idx += full_input_coords[d] * input_strides[d];  // multiply: ~3 cycles
```

After coalesce + reorder (PyTorch approach — no more modulo/division):
```cpp
// ~3 cycles per element — just stride multiplies:
// [N, M] reduce along axis=1 → coalesced to flat [M] inner loop
// strides precomputed once, pointer just advances by sizeof(T) each step
in_ptr += sizeof(T);   // or just: acc += *in_ptr++
```

The coalescing converts N-dimensional index arithmetic into a simple pointer advance
for the common cases (contiguous inner or outer reduction).

### Step 2: Runtime stride check — essentially free

After coalesce_dimensions(), the shape is ≤2D in most common cases.
Checking 3 stride values costs <1ns. The kernel runs for microseconds.
Ratio: dispatch cost / kernel cost < 0.01%.

### Steps 3+4: SIMD — compile-time templated, runtime dispatched

```
AVX2  (256-bit): Vec<float>::size() = 8  → 4 accumulators × 8 = 32 floats/iter
AVX-512 (512-bit): Vec<float>::size() = 16 → 4 accumulators × 16 = 64 floats/iter
SSE4.2 (128-bit): Vec<float>::size() = 4  → 4 accumulators × 4  = 16 floats/iter
```

The SIMD width is resolved at compile time via build flags.
Which path (inner/outer/scalar) is resolved at runtime via stride check.
This combination gives Eigen-level performance with PyTorch-level flexibility.

---

## Summary of the Decision

| What | How | Why |
|---|---|---|
| Case selection (inner/outer/generic) | **Runtime** (stride check after coalescing) | Axes are runtime values; `.so` can't pre-instantiate all combos |
| SIMD operations within each case | **Compile-time** (templated `Vectorized<T>`) | Zero runtime overhead, picks AVX2/AVX-512/NEON at build time |
| Numerical accuracy (tree reduction) | **Compile-time** optional (for sum/mean only) | Eigen-style tree, only for non-stateful associative ops |
| Parallelism | **Runtime** (OpenMP outer loop, grain-size aware) | Matches our current OpenMP model |

**Conclusion: PyTorch's runtime approach is the correct architecture for our pre-compiled library.
Eigen's compile-time approach cannot apply because our API takes runtime axes and we ship a `.so`.**
The runtime dispatch cost (3 integer compares, ~1ns) is irrelevant vs kernel execution time.
SIMD operations within the kernel are still compile-time — we get full performance without binary explosion.
