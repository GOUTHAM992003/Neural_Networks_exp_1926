# Can We Write Our Own Fused Matmul+Bias+GeLU Kernel?

## Short Answer

**Yes, you can.** It's not a hardware limitation — the GPU hardware on sm_86 is perfectly capable of computing matmul+bias+GeLU in a single kernel. NVIDIA just didn't ship that specific pre-compiled kernel in cuBLASLt. You can write your own.

**But should you?** Probably not. Here's why.

---

## Understanding What "Fused" Actually Means

### Your current fallback (3 separate kernels):

```
┌─────────────────────────────┐
│ Kernel 1: cublasSgemm       │  ← Reads A,B from GMEM, writes C to GMEM
│ C = A × B                   │     GMEM Write: M×N×4 bytes
└──────────────┬──────────────┘
               │ C sits in GMEM (DRAM)
               ▼
┌─────────────────────────────┐
│ Kernel 2: add_bias_kernel   │  ← Reads C from GMEM, reads bias, writes C back
│ C[i] += bias[i % N]        │     GMEM Read + Write: 2 × M×N×4 bytes
└──────────────┬──────────────┘
               │ C sits in GMEM again
               ▼
┌─────────────────────────────┐
│ Kernel 3: fused_gelu_cuda   │  ← Reads C from GMEM, writes C back
│ C[i] = GeLU(C[i])          │     GMEM Read + Write: 2 × M×N×4 bytes
└─────────────────────────────┘

Total GMEM writes for C: 3 times  (5 × M×N×4 bytes total traffic for C alone)
```

### A truly fused kernel (1 kernel):

```
┌─────────────────────────────────────────────┐
│ Single kernel: matmul_bias_gelu             │
│                                             │
│ for each output tile:                       │
│   1. Load A,B tiles into shared memory      │
│   2. Compute C_tile in REGISTERS            │
│   3. C_tile[i] += bias[j]          ← in registers, no GMEM!
│   4. C_tile[i] = GeLU(C_tile[i])   ← in registers, no GMEM!
│   5. Write final result to GMEM    ← ONE write
│                                             │
└─────────────────────────────────────────────┘

Total GMEM writes for C: 1 time   (1 × M×N×4 bytes)
```

> [!IMPORTANT]
> **The fusion happens at the register level.** After the matmul computes each output element in a register, you add bias and apply GeLU *right there in the register*, before writing to global memory. This is fundamentally different from the 3-kernel fallback.

---

## Three Approaches to Write Your Own

### Approach 1: Modify Your Existing Matmul Kernel

You already have `matmul_fp32_optimized` in [GenMatmul.cu](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/master_gau/src/Kernels/cuda/GenMatmul.cu#L567-L628). Look at the store loop (lines 624-627):

```cpp
// CURRENT: just write results to GMEM
for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {
    int r = by*BM + tRow*4 + i, c = bx*BN + tCol*4 + j;
    if (r < M && c < N) Cp[r*s_cm + c*s_cn] = results[i*4+j];
}
```

**To fuse bias+GeLU, you change ONLY this store loop:**

```cpp
// FUSED: add bias + GeLU in registers, THEN write to GMEM
for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {
    int r = by*BM + tRow*4 + i, c = bx*BN + tCol*4 + j;
    if (r < M && c < N) {
        float val = results[i*4+j];
        
        // ── Bias addition (in register) ──
        val += bias[c];   // bias is [N], broadcast across rows
        
        // ── GeLU (in register) ──
        float x3 = val * val * val;
        float inner = 0.7978845608f * (val + 0.044715f * x3);
        float t;
        asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(inner));
        val = 0.5f * val * (1.0f + t);
        
        Cp[r*s_cm + c*s_cn] = val;  // ONE write to GMEM
    }
}
```

**That's it.** 3 extra lines of math, zero extra GMEM traffic.

### The problem with Approach 1

Your `matmul_fp32_optimized` gets **~5 TFLOPS**.  
cuBLAS's `cublasSgemm` on the same GPU gets **~19 TFLOPS**.

So your fused kernel would do:
- Matmul: **~5 TFLOPS** (3.8x slower than cuBLAS)
- + Bias+GeLU: essentially free (happens in registers)

The 3-kernel fallback does:
- Matmul: **~19 TFLOPS** (cuBLAS, highly optimized)
- + Bias: ~1 extra GMEM round-trip
- + GeLU: ~1 extra GMEM round-trip

**For large matrices (like GPT-2's `[batch×seq, 3072] × [3072, 768]`), the matmul dominates.** The cuBLAS matmul is so much faster that even with 2 extra kernel launches, it beats your fused kernel.

```
Your fused kernel:  matmul takes 200μs + bias+GeLU adds 0μs  = ~200μs total
3-kernel fallback:  matmul takes  53μs + bias takes 15μs + GeLU takes 15μs = ~83μs total
```

> [!CAUTION]
> **Your fused kernel would be ~2.4x SLOWER than the unfused fallback** because the matmul portion is so much slower. The memory savings from fusion don't compensate for the matmul performance loss.

---

### Approach 2: NVIDIA CUTLASS (Best of Both Worlds)

[CUTLASS](https://github.com/NVIDIA/cutlass) is NVIDIA's open-source matmul library. It provides **near-cuBLAS performance** with **customizable epilogues**.

```cpp
// CUTLASS approach: you define a custom epilogue functor
struct BiasGeluEpilogue {
    __device__ float operator()(float matmul_result, float bias_val) {
        float x = matmul_result + bias_val;
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        float t;
        asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(inner));
        return 0.5f * x * (1.0f + t);
    }
};

// Then use CUTLASS GEMM with your custom epilogue
using Gemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,    // A
    float, cutlass::layout::RowMajor,    // B
    float, cutlass::layout::RowMajor,    // C
    float,                                // accumulator
    cutlass::arch::OpClassTensorOp,       // use Tensor Cores
    cutlass::arch::Sm86,                  // target sm_86
    // ... tile sizes ...
    BiasGeluEpilogue                      // YOUR custom epilogue
>;
```

**Performance**: ~17-18 TFLOPS matmul + zero-cost bias+GeLU in epilogue.

**Downside**: CUTLASS is a heavy header-only C++ template library. Adds significant compile time and complexity.

---

### Approach 3: Keep Current Architecture (Recommended)

Your current approach: `cublasSgemm` + `add_bias_kernel` + `fused_gelu_cuda`

```
Matmul perf:     ~19 TFLOPS (best possible)
Extra overhead:  ~2 kernel launches + ~2 GMEM round-trips for C
```

For GPT-2 training with typical shapes (M=batch×seq, N=3072, K=768):

| Shape (M×N×K) | cuBLAS matmul | + bias + GeLU overhead | Total |
|---|---|---|---|
| 256×3072×768 | ~53μs | ~30μs | ~83μs |
| 1024×3072×768 | ~180μs | ~90μs | ~270μs |

The overhead is ~35-40% of the matmul — significant but not catastrophic. And you get the **best possible matmul performance**.

---

## Summary: Decision Table

| Approach | Matmul Perf | Fusion | Total Perf | Complexity |
|---|---|---|---|---|
| **Your tiled kernel + bias+GeLU** | ~5 TFLOPS | ✅ True fusion | ❌ Slowest | Low |
| **CUTLASS + custom epilogue** | ~17 TFLOPS | ✅ True fusion | ✅ Fastest | 🔴 Very High |
| **cuBLAS + separate kernels (current)** | ~19 TFLOPS | ❌ 3 kernels | ✅ Good enough | ✅ Low |

> [!TIP]
> **Recommended**: Keep your current approach. The 3-kernel path with cuBLAS is the pragmatic sweet spot. The matmul is the bottleneck, and cuBLAS is unbeatable at it.
>
> If you *really* want fusion, the path is **CUTLASS** — not hand-writing the matmul. But that's a serious engineering investment for marginal gains on FP32.
>
> The real win would be switching to **FP16/BF16 mixed precision**, where cuBLASLt's built-in GELU epilogue DOES work on sm_86, giving you fusion for free at 2x the throughput.
