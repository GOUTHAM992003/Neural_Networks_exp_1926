# Attention Kernels — Strided-Input Refactor (option 1, PyTorch-matching)

## The problem we were solving

Profiling `report12.sqlite` showed **45,600 invocations** of `generic_strided_copy_kernel`
in a 10-step training run. Every single call had the same shape:

- `gridX = 12,288`, `gridY = 1`, `gridZ = 1`
- `elements_copied = 12,582,912`  (= B × T × n_embd = 16 × 1024 × 768)
- `avg_dur_ns = 118,117`
- `total_ms = 5,386`

That's **~538 ms of GPU time per step** spent just moving 50 MB chunks around
inside device memory — essentially a zero-compute operation that was blocking
the attention kernel from starting.

### Where those copies came from — the call-chain

```
gpt2_attn_navin.cpp:173-175         (the GPT-2 attention block)
  q = autograd::transpose(autograd::reshape(q, {B,T,H,HD}), 1, 2);
  k = autograd::transpose(autograd::reshape(k, {B,T,H,HD}), 1, 2);
  v = autograd::transpose(autograd::reshape(v, {B,T,H,HD}), 1, 2);
```
After `reshape`, Q/K/V are contiguous `[B, T, H, HD]`. After `transpose(1, 2)`
they are **non-contiguous strided views** with shape `[B, H, T, HD]`
(no data moved yet, it's just a view).

```
AttentionOps.cpp:278-281            (our attention dispatch)
  // Kernel requires contiguous (B*nh, T, hd) layout — make contiguous if needed
  Tensor q_contig = query.is_contiguous() ? query : query.contiguous();   // 50 MB memcpy
  Tensor k_contig = key.is_contiguous()   ? key   : key.contiguous();     // 50 MB memcpy
  Tensor v_contig = value.is_contiguous() ? value : value.contiguous();   // 50 MB memcpy
```

### The root cause — the kernel's memory layout assumption

```
AttentionForward_sm89.cu:123-127
  const float* Q_bnh   = Q   + bnh * T * HeadDim;   // ← assumes Q is contiguous [B*H, T, HD]
  const float* K_bnh   = K   + bnh * T * HeadDim;
  const float* V_bnh   = V   + bnh * T * HeadDim;
  float*       O_bnh   = O   + bnh * T * HeadDim;
  float*       LSE_bnh = LSE + bnh * T;
```
The kernel computes every element address with a single multiply on
`bnh = blockIdx.y` (which iterates over B·H combinations). That only works if
the tensor is physically laid out in `[B·H, T, HD]` contiguous memory. Any
other layout gives wrong addresses → silent memory corruption.

That's why the caller has to force contiguity.

## What PyTorch does differently

Reading `pytorch_source/aten/src/ATen/native/transformers/cuda/attention.cu`:

1. **Pass non-contiguous views directly to the kernel** (line 1053-1075):
   ```cpp
   Tensor q_t = q_chunk.transpose(1, 2);   // non-contiguous view, no copy
   Tensor k_t = k_chunk.transpose(1, 2);
   Tensor v_t = v_chunk.transpose(1, 2);
   auto [attention, ...] = at::_efficient_attention_forward(q_t, k_t, v_t, ...);
   ```
   No `.contiguous()` call.

2. **Only require the last dim (HeadDim) to be contiguous** (line 1376-1378):
   ```cpp
   CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(query);
   CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(key);
   CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(value);
   ```
   Other dims (B, seqlen, num_heads) can have any strides.

3. **Pass three strides per tensor to the kernel** (line 1732-1741):
   ```cpp
   p.q_strideB = query.stride(0);
   p.q_strideM = query.stride(1);   // sequence stride
   p.q_strideH = query.stride(2);   // head stride
   // same for K, V
   ```
   The kernel reads each element with
   `Q + b*strideB + t*strideM + h*strideH + d*1`
   where `d*1` = last dim stride is always 1.

Cost inside the kernel: three extra multiplies per thread when computing the
base pointer. That's essentially free on a memory-bound kernel.

Gain: zero `.contiguous()` copies at the caller. PyTorch pays nothing for
the strided layout because the kernel absorbs the generality.

## What we're changing

We match PyTorch's design: kernels become **stride-aware**, callers stop
forcing contiguity.

### 1. `include/ops/cuda/attention/AttentionCommon.cuh`
Add strides + `B, nh` fields to the existing `MemEfficientBwdParams`.
Add a parallel `MemEfficientFwdParams` struct so the forward kernel can also
take a single struct argument (previously it took ~10 loose args).

### 2. Forward kernels
Change the kernel signature from a long list of raw args to
`(MemEfficientFwdParams params)`. Inside the kernel:

Before:
```cuda
const float* Q_bnh = Q + bnh * T * HeadDim;
... Q_bnh[(qi + q) * HeadDim + d] ...    // row access
```
After:
```cuda
const int b = bnh / nh;
const int h = bnh - b * nh;
const float* Q_bnh = Q + b * params.q_strideB + h * params.q_strideH;
... Q_bnh[(qi + q) * params.q_strideM + d] ...
```
Head-dim stride is always 1, so the innermost `+ d` is unchanged.

Files:
- `src/Kernels/cuda/attention/arch/AttentionForward_sm89.cu` (the sm89 path, our hot kernel)
- `src/Kernels/cuda/attention/AttentionForward.cu`
  (`fused_attn_forward_kernel` scalar + `fused_attn_forward_kernel_tc` generic WMMA — fallback paths for non-Ada GPUs)

### 3. Backward kernels
Same treatment on the existing `MemEfficientBwdParams` — use the new stride
fields. Files:
- `src/Kernels/cuda/attention/arch/AttentionBackward_sm89.cu`
  (exp12 unified bwd + precompute_D_sm89)
- `src/Kernels/cuda/attention/AttentionBackward.cu`
  (exp11 bwd, exp7 scalar bwd, precompute_D)

### 4. Public entry points
Headers in `include/ops/helpers/AttentionKernels.h` gain stride args:
```
void mem_efficient_attn_forward_tc(
    const float* query,  int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
    const float* key,    int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
    const float* value,  int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
    float* output,       int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
    float* lse,          int64_t lse_strideB, int64_t lse_strideH,
    int64_t B, int64_t nh, int64_t T, int64_t hd,
    bool is_causal, float dropout_p, const float* dropout_mask);
```
(and same pattern for `mem_efficient_attn_backward`).

### 5. Callers (the copies disappear here)
- `src/autograd/operations/AttentionOps.cpp` (forward)
  Remove three `.contiguous()` calls on Q/K/V. Extract strides from input
  tensors and pass them through. Output is still allocated as contiguous
  `[B, nh, T, hd]`.
- `src/autograd/backward/AttentionBackward.cpp` (backward)
  Remove six `.contiguous()` calls (grad_output + saved Q/K/V/O/LSE).
  Extract strides and pass through.

## Expected impact

From the sqlite counts:
- **4,560 `generic_strided_copy_kernel` calls per step eliminated** (= 45,600 / 10 steps)
- **~538 ms of GPU memset/memcpy activity per step gone** (5% of step time)
- **~115 ms per step of real memory-bandwidth time recovered** (the 3 × 50 MB per attention block that are on the critical path between matmul and attention)
- Per-step launch count drops by roughly 4,560, which also lightens the
  queue-submission pressure the profile has been showing
  (`cudaLaunchKernel` avg = 290 µs).

## Correctness criteria

- Loss at steps 0-5 must match the current baseline to within floating-point
  precision (same input batches, same seed, so should be bit-identical or
  near-identical modulo kernel ordering).
- `norm` values at each step should likewise match.
- Training should run for at least 10 steps without NaN/Inf.

## Rollback plan

If the refactor is wrong:
- Revert with `git checkout -- <file list>`. The change is contained to:
  `AttentionCommon.cuh`, both attention .cu files, both arch .cu files,
  `AttentionOps.cpp`, `AttentionBackward.cpp`, and the header
  `AttentionKernels.h`. No model code changes. No autograd graph changes.

## References

- PyTorch SDPA mem-efficient: `pytorch_source/aten/src/ATen/native/transformers/cuda/attention.cu:1053-1075, 1732-1741`
- Our current sm89 forward: `master_gau_latest_1/src/Kernels/cuda/attention/arch/AttentionForward_sm89.cu:69-127`
- Our current sm89 backward: `master_gau_latest_1/src/Kernels/cuda/attention/arch/AttentionBackward_sm89.cu:71-130`
- Profile that found the issue: `master_gau_latest_1/report12.sqlite`
