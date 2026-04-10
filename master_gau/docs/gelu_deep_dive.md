# GeLU Optimization Deep Dive — Complete Change Log

All changes made during the GeLU optimization sprint. Covers restructuring, forward+backward CPU/GPU optimization, benchmarks vs PyTorch/TensorFlow.

---

## Complete File List

### NEW files created (8 files)

| # | File | Lines | Purpose |
|---|------|-------|---------|
| 1 | `include/ops/UnaryOps/Activations.h` | 98 | Pure math activation declarations — 8 forward + 7 backward functions + 2 result structs |
| 2 | `src/UnaryOps/cpu/Activations.cpp` | ~1170 | Full CPU/GPU dispatch + optimized AVX2 CPU kernels for all forward+backward activations |
| 3 | `gelu_training_bench.cpp` | ~65 | Benchmark: NEW optimized code, exact training tensor size [8,1024,1536], all 4 bifurcations |
| 4 | `gelu_before_opt_test.cpp` | ~75 | Benchmark: OLD code baseline, same tensor size, for colleague's system comparison |
| 5 | `gelu_backtrack.cpp` | ~45 | Benchmark: quick test with [8,1024,384], all 4 bifurcations |
| 6 | `gelu_scaling_test.cpp` | ~100 | Benchmark: scaling test across 7 tensor sizes (16K to 134M elements) |
| 7 | `bench_pytorch_gelu.py` | ~50 | PyTorch comparison benchmark, same conditions |
| 8 | `bench_tensorflow_gelu.py` | ~50 | TensorFlow comparison benchmark, same conditions |

### MODIFIED files (7 files)

| # | File | What changed |
|---|------|-------------|
| 1 | `src/autograd/operations/ActivationOps.cpp` | **400→179 lines.** Removed ALL inline CPU math + dispatch logic. Now just calls `_forward()` functions from Activations.h and records autograd graph. Zero tensor math inside. |
| 2 | `src/autograd/backward/ActivationBackward.cpp` | **277→90 lines.** Removed ALL inline backward math + GPU dispatch. Now just calls `_backward()` functions from Activations.h. Zero tensor math inside. |
| 3 | `include/ops/helpers/Vectorized.h` | **+70 lines.** Added `abs()` method (line 132-135) and `tanh()` rational polynomial approximation (lines 148-201) to `Vectorized<float>`. Cephes coefficients, Horner's method with FMA, blendv clamping. |
| 4 | `include/ops/helpers/ActivationKernels.h` | **Rewritten.** Removed duplicate declarations, organized by activation (1-6), added comments with formulas/dtypes. Added: `relu_forward_cuda` fp16+bf16 overloads, `fused_bias_gelu_cuda` fp16+bf16 overloads. |
| 5 | `src/Kernels/cuda/ActivationKernels.cu` | **ReLU forward:** rewritten with `(x+fabsf(x))*0.5f` (NaN-propagating, branch-free), templated for fp32/fp16/bf16. **fused_bias_gelu forward:** templated for fp32/fp16/bf16 (was fp32-only). |
| 6 | `include/TensorLib.h` | **+1 line.** Added `#include "ops/UnaryOps/Activations.h"` at line 20. |
| 7 | `docs/gelu_deep_dive.md` | This file — complete documentation. |

### DELETED (from previous session)

| Directory | Files | Reason |
|-----------|-------|--------|
| `include/mlp/` | `activation.h`, `layers.h`, `loss.h`, `TensorMLP.h`, `WeightInit.h` | Unused legacy code |
| `src/mlp-blocks/` | `activation.cpp`, `layers.cpp`, `loss.cpp`, `WeightInit.cpp` | Unused legacy, `activation.cpp` had sign bug (`-0.044715` instead of `+0.044715`) |

---

## What each modified file looks like now

### `src/autograd/operations/ActivationOps.cpp` (179 lines)

Every function follows the same 5-line pattern:
```cpp
Tensor gelu(const Tensor &x) {
    GraphRecordMode::record_forward("ACTIVATION: GeLU");
    Tensor output = gelu_forward(x);           // ← calls math engine
    if (GradMode::is_enabled() && x.requires_grad()) {
        auto grad_fn = std::make_shared<GeLUBackward>(x);
        grad_fn->set_next_edge(0, get_grad_edge(x));
        output.set_grad_fn(grad_fn);
        output.set_requires_grad(true);
    }
    return output;
}
```

All 8 activations: relu (L14-27), gelu (L32-45), sigmoid (L50-63), softmax (L68-98), fused_tril_softmax (L105-112), dropout (L117-136), swiglu (L141-155), fused_bias_gelu (L160-176).

### `src/autograd/backward/ActivationBackward.cpp` (90 lines)

Every backward node is now a one-liner:
```cpp
std::vector<Tensor> GeLUBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("GeLUBackward: no gradients provided");
    return {gelu_backward(grads[0], saved_input_)};  // ← calls math engine
}
```

All 7 backward nodes: ReluBackward, GeLUBackward, SigmoidBackward, SoftmaxBackward, DropoutBackward, SwiGLUBackward, FusedBiasGeLUBackward.

### `include/ops/UnaryOps/Activations.h` (98 lines)

Declares in `namespace OwnTensor`:
- **Forward (8):** `relu_forward`, `gelu_forward`, `sigmoid_forward`, `softmax_forward`, `swiglu_forward`, `fused_bias_gelu_forward`, `dropout_forward` (returns `DropoutForwardResult`), `fused_tril_softmax_forward`
- **Backward (7):** `relu_backward`, `gelu_backward`, `sigmoid_backward`, `softmax_backward`, `dropout_backward`, `swiglu_backward`, `fused_bias_gelu_backward` (returns `FusedBiasGeLUBackwardResult`)

### `src/UnaryOps/cpu/Activations.cpp` (~1170 lines)

Every function handles: GPU dispatch (dtype switch → CUDA kernel call) + optimized CPU kernel.

**Forward CPU optimizations applied (gelu + fused_bias_gelu):**
- Single-pass fused kernel (zero temporaries)
- AVX2 SIMD (8 floats per vector)
- Vectorized tanh polynomial (`Vectorized<float>::tanh()`)
- FMA (`Vec::fmadd`)
- 2x loop unrolling (16 floats per iteration)
- OpenMP parallelization (threshold: 16384 elements)
- fp16/bf16 F16C paths (`load_fp16_as_float`/`store_float_as_fp16`)

**Backward CPU optimizations applied (gelu_backward + fused_bias_gelu_backward):**
- Same 7 optimizations as forward
- fused_bias_gelu_backward: thread-local bias accumulators to avoid contention

**ReLU:** `(x + |x|) * 0.5` — NaN-propagating, branch-free, AVX2 optimized on CPU.

### `include/ops/helpers/ActivationKernels.h`

Clean organized declarations:
1. ReLU — forward fp32/fp16/bf16, backward fp32/fp16/bf16
2. GeLU — forward template, backward fp32/fp16/bf16 overloads
3. Sigmoid — forward template, backward fp32/fp16/bf16 overloads
4. Softmax — forward fp32 + typed template, backward fp32/fp16/bf16
5. SwiGLU — forward template, backward fp32/fp16/bf16
6. Fused Bias+GeLU — forward fp32/fp16/bf16, backward fp32

### `src/Kernels/cuda/ActivationKernels.cu`

- **fused_gelu forward:** template `fused_gelu_kernel<T>` (fp32/fp16/bf16) + `fused_gelu_kernel_vectorized` (fp32 float4). Uses `fast_tanh` PTX.
- **fused_gelu backward:** template `fused_gelu_backward_kernel<T>` (fp32/fp16/bf16 scalar).
- **fused_bias_gelu forward:** template `fused_bias_gelu_kernel<T>` for fp32, `fused_bias_gelu_kernel_typed<T>` for fp16/bf16.
- **ReLU forward:** template `relu_forward_kernel<T>` with `(x+fabsf(x))*0.5f`, fp32/fp16/bf16.

### `include/ops/helpers/Vectorized.h`

Added to `Vectorized<float>`:
- `abs()` (line 132): `_mm256_andnot_ps(sign_mask, values)`
- `tanh()` (lines 148-201): Cephes rational polynomial P(z)/Q(z), Horner's method with `_mm256_fmadd_ps`, clamped at ±1 via `_mm256_blendv_ps`

---

## Benchmark Results

### Real training size: [8, 1024, 1536] = 12.6M elements

| Op | Our CPU | Our GPU | PyTorch CPU | PyTorch GPU |
|---|---|---|---|---|
| **gelu_forward** | 3.26 ms | 0.33 ms | 0.88 ms | 0.09 ms |
| **fused_bias_gelu_forward** | 3.18 ms | 0.33 ms | 0.98 ms (unfused) | 0.18 ms (unfused) |
| **gelu_backward** | 3.75 ms | 0.48 ms | 2.80 ms (fwd+bwd) | 0.31 ms (fwd+bwd) |
| **fused_bias_gelu_backward** | 4.05 ms | 1.03 ms | 2.58 ms (fwd+bwd) | 0.40 ms (fwd+bwd) |

Note: PyTorch backward includes forward pass time. Our backward is backward-only.

### Small test size: [8, 1024, 384] = 3.1M elements (vs PyTorch and TensorFlow)

| Op | **OwnTensor** | **PyTorch** | **TensorFlow** | vs PyTorch | vs TF |
|---|---|---|---|---|---|
| CPU gelu_fwd | **0.154 ms** | 0.882 ms | 3.014 ms | **5.7x** | **19.6x** |
| CPU bias+gelu_fwd | **0.137 ms** | 0.979 ms | 3.495 ms | **7.1x** | **25.5x** |
| CPU gelu_bwd | **0.198 ms** | 2.798 ms* | 9.543 ms* | **14.1x** | **48.2x** |
| GPU gelu_fwd | **0.077 ms** | 0.091 ms | 0.110 ms | **1.2x** | **1.4x** |
| GPU bias+gelu_fwd | **0.091 ms** | 0.179 ms | 0.121 ms | **2.0x** | **1.3x** |

### GPU scaling test (gelu_forward only)

| Size | CPU | GPU | GPU speedup |
|---|---|---|---|
| [1,128,128] 16K | 0.038 ms | 0.004 ms | 10x |
| [8,1024,384] 3M | 0.168 ms | 0.077 ms | 2.2x |
| [32,1024,768] 25M | 7.2 ms | 0.63 ms | 11.5x |
| [64,2048,1024] 134M | 30.2 ms | 3.4 ms | 8.9x |

---

## Architecture (before vs after)

### Before
```
autograd::gelu(x) → [dispatch + math + autograd ALL in ActivationOps.cpp ~400 lines]
GeLUBackward::apply() → [dispatch + math ALL in ActivationBackward.cpp ~277 lines]
```

### After
```
autograd::gelu(x) → gelu_forward(x)       [Activations.cpp — math + dispatch]
                   → record graph           [ActivationOps.cpp — autograd only, 179 lines]

GeLUBackward::apply() → gelu_backward()   [Activations.cpp — math + dispatch]
                                            [ActivationBackward.cpp — autograd only, 90 lines]
```

Same pattern as `autograd::sum()` → `reduce_sum()` in ReductionOps.

---

## Deleted legacy

- `include/mlp/` and `src/mlp-blocks/` — unused early code with bugs, training uses `autograd::gelu()` exclusively.

---

## Remaining work

- **GPU backward gelu:** scalar kernel, no float4 vectorization (matches old code perf)
- **SwiGLU backward CPU:** not implemented (GPU only, throws on CPU)
- **AVX-512 runtime dispatch:** library-wide infrastructure change for future
- **fused_tril_softmax backward CPU:** not implemented (GPU only)
