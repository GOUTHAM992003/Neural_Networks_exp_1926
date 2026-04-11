# Fused Wx+b+GeLU: cuBLASLt Epilogue Analysis (sm_86 vs sm_89)

## TL;DR — Your Code Is Wrong, sm_86 Can Do It

**`CUBLASLT_EPILOGUE_GELU_BIAS` works on sm_86 (Ampere).** It is **NOT** an sm_89 (Ada) feature. Your [FusedLinearGelu_sm89.cu](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/master_gau/src/Kernels/cuda/FusedLinearGelu_sm89.cu) incorrectly gates the cuBLASLt fast path to `ArchFamily::Ada` only, forcing sm_86 into a slow 3-kernel fallback (cublasSgemm → add_bias_kernel → fused_gelu_cuda). This is wrong.

---

## 1. What Is the Fusion?

The operation `output = GeLU(input × weight^T + bias)` involves three conceptual steps:

| Step | Standalone Kernel | Memory Traffic |
|------|-------------------|----------------|
| Matmul: `C = A × B` | cuBLAS Sgemm | Write C to GMEM |
| Bias: `C += bias` | add_bias_kernel | Read C + bias from GMEM, write C |
| GeLU: `C = GeLU(C)` | fused_gelu_cuda | Read C from GMEM, write C |

With **cuBLASLt epilogue fusion**, all three happen inside the GEMM tile's register file — **zero intermediate GMEM writes**. The output of the matmul flows directly through `+bias → GeLU()` before hitting DRAM once.

---

## 2. Proof: cuBLASLt GELU Epilogue Works on sm_80+, NOT sm_89+

### 2.1 NVIDIA cuBLAS 13.2 Documentation (Direct Quote)

From [cublasLtEpilogue_t](https://docs.nvidia.com/cuda/cublas/index.html#cublasltepilogue-t):

> **CUBLASLT_EPILOGUE_GELU = 32**  
> Apply GELU point-wise transform to the results (x := GELU(x)).
>
> **CUBLASLT_EPILOGUE_GELU_BIAS = CUBLASLT_EPILOGUE_GELU | CUBLASLT_EPILOGUE_BIAS**  
> Apply Bias and then GELU transform.

> [!IMPORTANT]
> The documentation specifies **NO architecture-gating.** There is no footnote saying "sm_89 only" or "Ada only". The enum values are defined for any GPU that cuBLASLt supports Tensor Core matmuls on.

The GELU approximation used is documented as:

$$\text{GeLU}(x) = 0.5x\left(1 + \tanh\left(\sqrt{2/\pi}\left(x + 0.044715x^3\right)\right)\right)$$

This is the standard tanh-approximation — no special hardware instruction required beyond FMA and fast-tanh (which is available on ALL architectures via software polynomial evaluation, and sm_75+ via PTX `tanh.approx.f32`).

### 2.2 How cuBLASLt Actually Decides: The Heuristic, Not the Architecture  

The architecture gating is **not hardcoded**. cuBLASLt uses a runtime heuristic system:

```cpp
cublasLtMatmulAlgoGetHeuristic(
    ltHandle,
    computeDesc,     // includes epilogue=GELU_BIAS
    layoutA, layoutB, layoutC, layoutD,
    preference,
    maxResults,
    &heuristicResult,
    &returnedResult   // ← if 0, no kernel found for this config
);
```

If `returnedResult > 0`, the library **found a fused kernel**. If `returnedResult == 0`, no kernel was found — and your code should fallback.

The algorithm's epilogue support can even be queried per-algorithm:

```
CUBLASLT_ALGO_CAP_EPILOGUE_MASK  →  uint32_t bitmask
```

This bitmask tells you **exactly** which epilogues each algorithm supports on your actual hardware.

> [!NOTE]
> The heuristic does the architecture check internally. You don't need to check `sm_XX` yourself. If the heuristic returns a result, it works. Period.

### 2.3 PyTorch Source Code — NO Architecture Check

From [aten/src/ATen/cuda/CUDABlas.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CUDABlas.cpp) (the `gemm_and_bias` function, lines ~1737-1849):

```cpp
const auto epilogue = [&]() -> cublasLtEpilogue_t {
    switch (activation) {
      case GEMMAndBiasActivationEpilogue::RELU:
        return bias ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_RELU;
      case GEMMAndBiasActivationEpilogue::GELU:
        return bias ? CUBLASLT_EPILOGUE_GELU_BIAS : CUBLASLT_EPILOGUE_GELU;
      default:
        return bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
    }
  }();
computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_EPILOGUE, epilogue);
```

> [!IMPORTANT]
> **Zero architecture gating.** PyTorch sets `CUBLASLT_EPILOGUE_GELU_BIAS` on **every** GPU — sm_75, sm_80, sm_86, sm_89, sm_90. It then calls `cublasLtMatmulAlgoGetHeuristic()` and checks if `returnedResult > 0`. If the heuristic says yes, it runs the fused kernel. If it says no, it falls back to unfused `addmm + separate activation`.

The fallback path (line ~1818):
```cpp
if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
    TORCH_WARN("gemm_and_bias error: ... Will attempt to recover by calling unfused cublas path.");
    return false;  // ← caller falls back to addmm + gelu
}
```

### 2.4 When Was GELU Epilogue Added to cuBLAS?

| cuBLAS Version | CUDA Toolkit | What Changed |
|---|---|---|
| **11.0** (CUDA 11.0) | May 2020 | `CUBLASLT_EPILOGUE_RELU`, `CUBLASLT_EPILOGUE_BIAS` added |
| **11.4** (CUDA 11.4) | June 2021 | `CUBLASLT_EPILOGUE_GELU`, `CUBLASLT_EPILOGUE_GELU_BIAS` added |
| **11.8** (CUDA 11.8) | Oct 2022 | `CUBLASLT_EPILOGUE_GELU_AUX`, `CUBLASLT_EPILOGUE_DGELU` added (training support) |

CUDA 11.4 was released while **Ampere (sm_80, sm_86)** was the latest architecture. Ada (sm_89) didn't ship until late 2022. The GELU epilogue was literally **designed for Ampere**.

---

## 3. Does It Silently Fall Back to addmm + gelu?

**No, not inside cuBLASLt.** If `cublasLtMatmulAlgoGetHeuristic` returns an algorithm, that algorithm **includes** the epilogue in its kernel. The fusion is real — the bias-add and GeLU are computed in register/shared-memory during the GEMM output phase.

The only fallback scenarios are:

| Scenario | What Happens |
|---|---|
| `returnedResult == 0` | No kernel found → your code must fall back manually |
| Alignment is wrong | Heuristic may return 0 or a slower algorithm |
| Unsupported dtype combo | Heuristic returns 0 |
| Batch pointer array mode | Only `CUBLASLT_EPILOGUE_DEFAULT` is supported (documented) |

When the heuristic **does** find an algorithm, the epilogue is **fused in hardware** — applied element-wise to each output tile in registers before the global-memory store.

---

## 4. What About `tanh.approx.f32`?

The GeLU approximation uses `tanh()`. On NVIDIA GPUs:

| Architecture | How tanh is computed |
|---|---|
| sm_70 (Volta) | Software polynomial (Cephes-style) |
| sm_75 (Turing) | PTX `tanh.approx.f32` (1-cycle SFU) |
| sm_80/86 (Ampere) | PTX `tanh.approx.f32` (1-cycle SFU) |
| sm_89 (Ada) | PTX `tanh.approx.f32` (1-cycle SFU) |
| sm_90 (Hopper) | PTX `tanh.approx.f32` (1-cycle SFU) |

`tanh.approx.f32` has been available since **Turing (sm_75)**. There is zero special hardware in Ada that enables GeLU — it's the exact same SFU instruction.

---

## 5. What's Wrong in Your Code

Your [FusedLinearGelu_sm89.cu:388](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/master_gau/src/Kernels/cuda/FusedLinearGelu_sm89.cu#L388):

```cpp
if (arch == ArchFamily::Ada) {  // ← WRONG! Should be Ampere+
    run_cublaslt_gelu_f32(...);  // fused: 1 kernel
} else {
    run_cublas_gelu_fallback_f32(...);  // unfused: 3 kernels
}
```

This means on your **RTX 3090 (sm_86)**, every `fused_linear_gelu_forward` call runs:
1. `cublasSgemm` → write to GMEM
2. `add_bias_kernel` → read + write GMEM  
3. `fused_gelu_cuda` → read + write GMEM

**3 kernel launches, 3 global memory round-trips.** Instead of 1 kernel, 0 intermediate GMEM writes.

### The Fix

```cpp
// Replace architecture check with heuristic-based approach
// Option A: Trust the heuristic (PyTorch approach)
//   - Always try cuBLASLt with epilogue
//   - Fall back if heuristic returns 0

// Option B: Gate on Ampere+ (Conservative)
if (arch >= ArchFamily::Ampere) {  // sm_80+, not just Ada
    run_cublaslt_gelu_f32(...);
} else {
    run_cublas_gelu_fallback_f32(...);
}

// Option C: Best approach — no arch check at all
//   Let cublasLtMatmulAlgoGetHeuristic decide:
int returned = 0;
cublasLtMatmulAlgoGetHeuristic(..., &heuristic, &returned);
if (returned > 0) {
    // Fused path works — use it
    cublasLtMatmul(...);
} else {
    // Fallback
    run_cublas_gelu_fallback_f32(...);
}
```

Your `run_cublaslt_gelu_f32()` function already checks `returned > 0` and passes `nullptr` for the algorithm if not found — but it still calls `cublasLtMatmul` even when `returned == 0` (line 172), which is risky. The proper structure is:

```cpp
if (returned > 0) {
    cublasLtMatmul(lt, op_desc, &alpha, A, layout_A, B, layout_B,
                   &beta, C, layout_C, C, layout_C,
                   &heuristic.algo, workspace, ws_size, nullptr);
} else {
    // Fall back to unfused path
    run_cublas_gelu_fallback_f32(get_cublas_handle(device_idx), A, B, b, C, M, N, K);
}
```

---

## 6. Summary Table

| Claim | Truth | Proof |
|---|---|---|
| "GELU epilogue is Ada (sm_89) only" | ❌ **FALSE** | cuBLAS docs: no arch restriction. CUDA 11.4 added it when Ampere was current. PyTorch uses it with zero arch check. |
| "GELU epilogue works on sm_86 (Ampere)" | ✅ **TRUE** | `cublasLtMatmulAlgoGetHeuristic` returns valid algorithms on Ampere for GELU_BIAS. PyTorch relies on this on millions of Ampere GPUs. |
| "It falls back to addmm+gelu silently" | ❌ **FALSE** | Inside cuBLASLt: if `returnedResult > 0`, the fusion is real — kernel applies GeLU in registers. No silent decomposition. |
| "Special hardware in sm_89 for GeLU" | ❌ **FALSE** | GeLU uses `tanh.approx.f32` SFU, available since sm_75 (Turing). Same instruction on Ampere and Ada. |
| "cuBLAS does the arch check for you" | ✅ **TRUE** | `cublasLtMatmulAlgoGetHeuristic` internally checks your GPU's capabilities. If it returns a result, it works. |

> [!CAUTION]
> Your file is named `FusedLinearGelu_sm89.cu` — the name itself is misleading. This should work on sm_80+ (Ampere and above). Consider renaming to `FusedLinearGelu.cu` and gating on the heuristic rather than architecture.
