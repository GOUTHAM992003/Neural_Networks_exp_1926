#!/usr/bin/env python3
"""
PyTorch Benchmark: sum, nansum, mean, nanmean
All dtypes × All paths × All DL shapes × Edge cases

Run: python3 Tests/pytorch_sum_mean_bench.py
Output: CSV-style for direct comparison with our C++ benchmark
"""
import torch
import time
import sys
import os

def bench(fn, warmup=10, iters=50):
    """Median of trimmed timings in microseconds"""
    for _ in range(warmup): fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    times = []
    for _ in range(iters):
        s = time.perf_counter_ns()
        fn()
        e = time.perf_counter_ns()
        times.append((e - s) / 1000.0)  # ns → μs
    times.sort()
    return times[len(times)//2]  # median

def inject_nans(t, pct):
    """Inject NaN into pct fraction of elements"""
    if pct <= 0: return t
    mask = torch.rand_like(t, dtype=torch.float32) < pct
    t[mask] = float('nan')
    return t

# ─── Configuration ───────────────────────────────────────────────
NTHREADS = int(os.environ.get("OMP_NUM_THREADS", "28"))
torch.set_num_threads(NTHREADS)

W = 110
SEP = "=" * W

print(SEP)
print(f"  PYTORCH BENCHMARK: sum / nansum / mean / nanmean")
print(f"  PyTorch {torch.__version__}, Threads: {torch.get_num_threads()}")
print(f"  CPU: {NTHREADS} threads")
print(SEP)

# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════
def fmt_shape(shape):
    return "(" + ",".join(str(s) for s in shape) + ")"

def fmt_dims(dims):
    if dims is None: return "None"
    if isinstance(dims, int): return str(dims)
    return "(" + ",".join(str(d) for d in dims) + ")"

def run_op(op_name, shape, dims, dtype, nan_pct=0.0, warmup=10, iters=50):
    """Run a single benchmark and return (time_us, result_sample)"""
    t = torch.randn(shape, dtype=dtype)
    if nan_pct > 0:
        t = inject_nans(t, nan_pct)
    
    if op_name == "sum":
        if dims is None: fn = lambda: torch.sum(t)
        else: fn = lambda: torch.sum(t, dim=dims)
    elif op_name == "nansum":
        if dims is None: fn = lambda: torch.nansum(t)
        else: fn = lambda: torch.nansum(t, dim=dims)
    elif op_name == "mean":
        if dims is None: fn = lambda: torch.mean(t)
        else: fn = lambda: torch.mean(t, dim=dims)
    elif op_name == "nanmean":
        if dims is None: fn = lambda: torch.nanmean(t)
        else: fn = lambda: torch.nanmean(t, dim=dims)
    else:
        raise ValueError(f"Unknown op: {op_name}")
    
    us = bench(fn, warmup, iters)
    return us

# ═══════════════════════════════════════════════════════════════════
# SECTION 1: INNERCONTIGUOUS (reduce last dim)
# ═══════════════════════════════════════════════════════════════════
def section_inner():
    print(f"\n{'─'*W}")
    print("  SECTION 1: INNERCONTIGUOUS (reduce last dim)")
    print(f"{'─'*W}")
    
    shapes = [
        ((8, 16), "tiny"),
        ((32, 64), "tiny"),
        ((10, 100), "small"),
        ((32, 768), "LayerNorm"),
        ((32, 4096), "large-LN"),
        ((256, 4096), "batch-feat"),
        ((4096, 768), "Seq-LN"),
        ((1000, 10000), "medium"),
        ((2048, 50176), "spatial"),
        ((49152, 128), "many-out"),
        ((10, 1000000), "wide"),
        ((1000000, 10), "tall"),
    ]
    
    print(f"{'Shape':>22} {'Label':>12} {'Dtype':>6} | {'sum':>10} {'nansum':>10} {'mean':>10} {'nanmean':>10}")
    print("-" * 95)
    
    for shape, label in shapes:
        numel = shape[0] * shape[1]
        if numel > 200_000_000: continue
        for dtype in [torch.float32, torch.float64]:
            dt = "fp32" if dtype == torch.float32 else "fp64"
            t_sum = run_op("sum", shape, -1, dtype)
            t_nsum = run_op("nansum", shape, -1, dtype, nan_pct=0.1)
            t_mean = run_op("mean", shape, -1, dtype)
            t_nmean = run_op("nanmean", shape, -1, dtype, nan_pct=0.1)
            print(f"  {fmt_shape(shape):>20} {label:>12} {dt:>6} | {t_sum:>9.0f}μ {t_nsum:>9.0f}μ {t_mean:>9.0f}μ {t_nmean:>9.0f}μ")

# ═══════════════════════════════════════════════════════════════════
# SECTION 2: OUTERCONTIGUOUS (reduce first dim)
# ═══════════════════════════════════════════════════════════════════
def section_outer():
    print(f"\n{'─'*W}")
    print("  SECTION 2: OUTERCONTIGUOUS (reduce first dim)")
    print(f"{'─'*W}")
    
    shapes = [
        ((16, 64), "tiny"),
        ((100, 256), "small"),
        ((1000, 256), "medium"),
        ((10000, 100), "tall"),
        ((100, 10000), "wide"),
        ((50176, 2048), "spatial"),
    ]
    
    print(f"{'Shape':>22} {'Label':>12} {'Dtype':>6} | {'sum':>10} {'nansum':>10} {'mean':>10} {'nanmean':>10}")
    print("-" * 95)
    
    for shape, label in shapes:
        numel = shape[0] * shape[1]
        if numel > 200_000_000: continue
        for dtype in [torch.float32, torch.float64]:
            dt = "fp32" if dtype == torch.float32 else "fp64"
            t_sum = run_op("sum", shape, 0, dtype)
            t_nsum = run_op("nansum", shape, 0, dtype, nan_pct=0.1)
            t_mean = run_op("mean", shape, 0, dtype)
            t_nmean = run_op("nanmean", shape, 0, dtype, nan_pct=0.1)
            print(f"  {fmt_shape(shape):>20} {label:>12} {dt:>6} | {t_sum:>9.0f}μ {t_nsum:>9.0f}μ {t_mean:>9.0f}μ {t_nmean:>9.0f}μ")

# ═══════════════════════════════════════════════════════════════════
# SECTION 3: GENERIC (reduce mixed/non-contiguous dims)
# ═══════════════════════════════════════════════════════════════════
def section_generic():
    print(f"\n{'─'*W}")
    print("  SECTION 3: GENERIC (reduce mixed/non-contiguous dims)")
    print(f"{'─'*W}")
    
    configs = [
        ((100, 200, 50), (0, 2), "3D-XZ"),
        ((100, 200, 50), (0,), "3D-X"),
        ((100, 200, 50), (1, 2), "3D-YZ"),
        ((32, 128, 768), (0, 2), "transformer"),
        ((32, 128, 768), (1, 2), "batch"),
        ((16, 64, 32, 32), (0, 2), "4D-XH"),
        ((16, 64, 32, 32), (2, 3), "4D-spatial"),
    ]
    
    print(f"{'Shape':>22} {'Dims':>12} {'Label':>12} {'Dtype':>6} | {'sum':>10} {'mean':>10} {'nanmean':>10}")
    print("-" * 95)
    
    for shape, dims, label in configs:
        for dtype in [torch.float32, torch.float64]:
            dt = "fp32" if dtype == torch.float32 else "fp64"
            t_sum = run_op("sum", shape, dims, dtype)
            t_mean = run_op("mean", shape, dims, dtype)
            t_nmean = run_op("nanmean", shape, dims, dtype, nan_pct=0.1)
            print(f"  {fmt_shape(shape):>20} {fmt_dims(dims):>12} {label:>12} {dt:>6} | {t_sum:>9.0f}μ {t_mean:>9.0f}μ {t_nmean:>9.0f}μ")

# ═══════════════════════════════════════════════════════════════════
# SECTION 4: FULL REDUCTION (all dims)
# ═══════════════════════════════════════════════════════════════════
def section_full():
    print(f"\n{'─'*W}")
    print("  SECTION 4: FULL REDUCTION (reduce all dims)")
    print(f"{'─'*W}")
    
    sizes = [100, 1000, 10000, 100000, 1000000, 10000000, 50000000]
    
    print(f"{'Size':>12} {'Dtype':>6} | {'sum':>10} {'nansum':>10} {'mean':>10} {'nanmean':>10}")
    print("-" * 70)
    
    for sz in sizes:
        for dtype in [torch.float32, torch.float64]:
            dt = "fp32" if dtype == torch.float32 else "fp64"
            t_sum = run_op("sum", (sz,), None, dtype)
            t_nsum = run_op("nansum", (sz,), None, dtype, nan_pct=0.1)
            t_mean = run_op("mean", (sz,), None, dtype)
            t_nmean = run_op("nanmean", (sz,), None, dtype, nan_pct=0.1)
            print(f"  {sz:>10,} {dt:>6} | {t_sum:>9.0f}μ {t_nsum:>9.0f}μ {t_mean:>9.0f}μ {t_nmean:>9.0f}μ")

# ═══════════════════════════════════════════════════════════════════
# SECTION 5: DL APPLICATION SHAPES
# ═══════════════════════════════════════════════════════════════════
def section_dl():
    print(f"\n{'─'*W}")
    print("  SECTION 5: DEEP LEARNING APPLICATION SHAPES (fp32)")
    print(f"{'─'*W}")
    
    apps = [
        ("LayerNorm-small",    (32, 768),          (-1,), ),
        ("LayerNorm-large",    (32, 4096),         (-1,), ),
        ("BatchNorm-2D",       (32, 256, 14, 14),  (0, 2, 3), ),
        ("BatchNorm-1D",       (32, 256, 100),     (0, 2), ),
        ("Attention-QK",       (32, 12, 128, 128), (-1,), ),
        ("Attention-head-mean",(32, 12, 128, 128), (1,), ),
        ("Seq-LayerNorm",      (32, 512, 768),     (-1,), ),
        ("Feature-mean",       (64, 2048),         (0,), ),
        ("Spatial-pool",       (32, 512, 7, 7),    (2, 3), ),
        ("Global-avg-pool",    (32, 2048, 7, 7),   (2, 3), ),
        ("Loss-reduction",     (32, 10000),        (-1,), ),
        ("Loss-full-red",      (32, 10000),        None, ),
        ("Token-mean",         (32, 512, 768),      (1,), ),
        ("Channel-mean",       (32, 256, 56, 56),  (1,), ),
        ("Embedding-mean",     (32, 128, 300),     (-1,), ),
        ("BERT-pool",          (16, 512, 1024),    (1,), ),
        ("ViT-patch",          (32, 197, 768),     (1,), ),
        ("ResNet-feat",        (64, 2048),         (-1,), ),
    ]
    
    print(f"{'Application':>22} {'Shape':>25} {'Dim':>12} | {'sum':>10} {'mean':>10} {'nanmean':>10}")
    print("-" * 100)
    
    for name, shape, dims in apps:
        t_sum = run_op("sum", shape, dims, torch.float32)
        t_mean = run_op("mean", shape, dims, torch.float32)
        t_nmean = run_op("nanmean", shape, dims, torch.float32, nan_pct=0.1)
        dim_str = fmt_dims(dims)
        print(f"  {name:>20} {fmt_shape(shape):>25} {dim_str:>12} | {t_sum:>9.0f}μ {t_mean:>9.0f}μ {t_nmean:>9.0f}μ")

# ═══════════════════════════════════════════════════════════════════
# SECTION 6: INTEGER DTYPES
# ═══════════════════════════════════════════════════════════════════
def section_int():
    print(f"\n{'─'*W}")
    print("  SECTION 6: INTEGER DTYPES (sum only — PyTorch rejects int mean)")
    print(f"{'─'*W}")
    
    sizes = [10000, 100000, 1000000, 10000000]
    dtypes = [
        (torch.int8, "int8"),
        (torch.int16, "int16"),
        (torch.int32, "int32"),
        (torch.int64, "int64"),
    ]
    
    print(f"{'Size':>12} {'Dtype':>8} | {'sum-full':>10} {'sum-last':>10}")
    print("-" * 50)
    
    for sz in sizes:
        for dtype, name in dtypes:
            t = torch.randint(-100, 100, (sz,), dtype=dtype)
            t_full = bench(lambda: torch.sum(t))
            # 2D for last dim
            rows = max(1, sz // 100)
            cols = 100
            t2 = torch.randint(-100, 100, (rows, cols), dtype=dtype)
            t_last = bench(lambda: torch.sum(t2, dim=-1))
            print(f"  {sz:>10,} {name:>8} | {t_full:>9.0f}μ {t_last:>9.0f}μ")

# ═══════════════════════════════════════════════════════════════════
# SECTION 7: NaN% SWEEP (nanmean sensitivity)
# ═══════════════════════════════════════════════════════════════════
def section_nan_sweep():
    print(f"\n{'─'*W}")
    print("  SECTION 7: NaN PERCENTAGE SWEEP — nanmean (1000,10000) fp32")
    print(f"{'─'*W}")
    
    pcts = [0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    print(f"{'NaN%':>8} | {'nanmean':>10} {'nansum':>10}")
    print("-" * 35)
    
    for pct in pcts:
        t = torch.randn(1000, 10000)
        t = inject_nans(t, pct)
        t_nm = bench(lambda: torch.nanmean(t, dim=-1))
        t_ns = bench(lambda: torch.nansum(t, dim=-1))
        print(f"  {pct*100:>5.1f}% | {t_nm:>9.0f}μ {t_ns:>9.0f}μ")

# ═══════════════════════════════════════════════════════════════════
# RUN ALL SECTIONS
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    section_inner()
    section_outer()
    section_generic()
    section_full()
    section_dl()
    section_int()
    section_nan_sweep()
    
    print(f"\n{SEP}")
    print("  PYTORCH BENCHMARK COMPLETE")
    print(SEP)
