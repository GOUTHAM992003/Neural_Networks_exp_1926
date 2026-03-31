#!/usr/bin/env python3
"""
Precision + Speed test for Mean/NanMean across ALL dtypes
Compare PyTorch vs NumPy ground truth
"""
import torch
import numpy as np
import time

torch.set_num_threads(28)

def bench(fn, warmup=5, iters=30):
    for _ in range(warmup): fn()
    times = []
    for _ in range(iters):
        s = time.perf_counter_ns(); r = fn(); e = time.perf_counter_ns()
        times.append((e-s)/1000)
    times.sort()
    return times[len(times)//2], r

def rel_err(got, expected):
    if expected == 0: return 0 if got == 0 else float('inf')
    return abs(got - expected) / abs(expected)

print("="*120)
print("  MEAN PRECISION + SPEED: PyTorch — ALL dtypes, ALL reduction paths, ALL sizes")
print("="*120)
print(f"PyTorch {torch.__version__}, Threads: {torch.get_num_threads()}\n")

# ============================================================
# 1. Regular Mean — ALL dtypes × ALL sizes × InnerContiguous
# ============================================================
print("--- Regular Mean: InnerContiguous (reduce dim=-1) ---")
print(f"{'Shape':>20s} {'Dtype':>12s} {'Time(μs)':>10s} {'rel_err':>12s} {'Precision':>12s}")
print("-"*70)

for shape in [(32, 768), (256, 4096), (4096, 768), (1000, 10000), (10, 1000000)]:
    for dt, np_dt in [(torch.float16, np.float16), (torch.bfloat16, np.float32),
                       (torch.float32, np.float32), (torch.float64, np.float64)]:
        np.random.seed(42)
        data = np.random.randn(*shape).astype(np_dt)
        gt = np.float64(np.sum(data.astype(np.float64), axis=-1) / shape[-1])

        t = torch.from_numpy(data.astype(np_dt)).to(dt)
        us, result = bench(lambda: torch.mean(t, dim=-1))

        # Compare first element
        r0 = float(result[0])
        g0 = float(gt[0]) if gt.ndim > 0 else float(gt)
        err = rel_err(r0, g0)
        prec = "PERFECT" if err == 0 else f"{err:.2e}"
        print(f"{str(shape):>20s} {str(dt):>12s} {us:>10.0f} {err:>12.2e} {prec:>12s}")

# ============================================================
# 2. Regular Mean — OuterContiguous (reduce dim=0)
# ============================================================
print("\n--- Regular Mean: OuterContiguous (reduce dim=0) ---")
print(f"{'Shape':>20s} {'Dtype':>12s} {'Time(μs)':>10s} {'rel_err':>12s}")
print("-"*70)

for shape in [(1000, 256), (10000, 100), (50176, 2048)]:
    for dt in [torch.float32, torch.float64]:
        np.random.seed(42)
        t = torch.randn(*shape, dtype=dt)
        us, result = bench(lambda: torch.mean(t, dim=0))
        # Ground truth
        data_np = t.numpy().astype(np.float64)
        gt = np.mean(data_np, axis=0)
        err = rel_err(float(result[0]), float(gt[0]))
        print(f"{str(shape):>20s} {str(dt):>12s} {us:>10.0f} {err:>12.2e}")

# ============================================================
# 3. Regular Mean — Generic (mixed dims)
# ============================================================
print("\n--- Regular Mean: Generic (reduce mixed dims) ---")
print(f"{'Shape':>20s} {'Dims':>10s} {'Dtype':>12s} {'Time(μs)':>10s} {'rel_err':>12s}")
print("-"*80)

for shape, dims in [((100,200,50), (0,2)), ((32,128,768), (0,2)), ((32,128,768), (1,2)),
                     ((16,64,32,32), (0,2)), ((16,64,32,32), (1,3))]:
    for dt in [torch.float32, torch.float64]:
        t = torch.randn(*shape, dtype=dt)
        us, result = bench(lambda: torch.mean(t, dim=dims))
        gt_np = np.mean(t.numpy().astype(np.float64), axis=dims)
        err = rel_err(float(result.flatten()[0]), float(gt_np.flatten()[0]))
        print(f"{str(shape):>20s} {str(dims):>10s} {str(dt):>12s} {us:>10.0f} {err:>12.2e}")

# ============================================================
# 4. Full Reduction — ALL dtypes
# ============================================================
print("\n--- Regular Mean: Full Reduction (all dtypes) ---")
print(f"{'Size':>12s} {'Dtype':>12s} {'Time(μs)':>10s} {'rel_err':>12s} {'Value':>15s}")
print("-"*70)

for size in [1000, 100000, 10000000]:
    for dt in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        np.random.seed(42)
        data = np.random.randn(size).astype(np.float32)
        gt = np.float64(np.sum(data.astype(np.float64)) / size)
        t = torch.from_numpy(data).to(dt)
        us, result = bench(lambda: torch.mean(t))
        r = float(result)
        err = rel_err(r, gt)
        print(f"{size:>12d} {str(dt):>12s} {us:>10.0f} {err:>12.2e} {r:>15.8f}")

# ============================================================
# 5. NanMean — ALL dtypes × varying NaN%
# ============================================================
print("\n--- NanMean: InnerContiguous (reduce dim=-1) ---")
print(f"{'Shape':>20s} {'NaN%':>6s} {'Dtype':>12s} {'Time(μs)':>10s} {'rel_err':>12s}")
print("-"*70)

for shape in [(32, 768), (1000, 10000), (4096, 768), (10, 1000000)]:
    for nan_pct in [0.0, 0.1, 0.5, 0.9]:
        for dt in [torch.float32, torch.float64]:
            np.random.seed(42)
            data = np.random.randn(*shape).astype(np.float64 if dt == torch.float64 else np.float32)
            if nan_pct > 0:
                mask = np.random.rand(*shape) < nan_pct
                data[mask] = np.nan
            gt = np.nanmean(data.astype(np.float64), axis=-1)

            t = torch.from_numpy(data.copy()).to(dt)
            us, result = bench(lambda: torch.nanmean(t, dim=-1))
            err = rel_err(float(result[0]), float(gt[0]))
            print(f"{str(shape):>20s} {nan_pct*100:>5.0f}% {str(dt):>12s} {us:>10.0f} {err:>12.2e}")

# ============================================================
# 6. DL Application Shapes
# ============================================================
print("\n--- DL Application Shapes ---")
print(f"{'Application':>25s} {'Shape':>25s} {'Dim':>6s} {'Time(μs)':>10s}")
print("-"*75)

dl_cases = [
    ("LayerNorm", (32, 768), -1),
    ("LayerNorm-large", (32, 4096), -1),
    ("BatchNorm-channel", (32, 256, 14, 14), (0, 2, 3)),
    ("Attention-scores", (32, 12, 128, 128), -1),
    ("Seq-LayerNorm", (32, 512, 768), -1),
    ("Feature-mean", (64, 2048), 0),
    ("Spatial-mean", (32, 512, 7, 7), (2, 3)),
    ("Loss-reduction", (32, 10000), -1),
    ("Global-mean", (32, 3, 224, 224), (1, 2, 3)),
    ("Token-mean", (32, 512, 768), 1),
]

for name, shape, dim in dl_cases:
    t = torch.randn(*shape, dtype=torch.float32)
    us, _ = bench(lambda: torch.mean(t, dim=dim))
    print(f"{name:>25s} {str(shape):>25s} {str(dim):>6s} {us:>10.0f}")

# ============================================================
# 7. Edge Cases
# ============================================================
print("\n--- Edge Cases ---")

# Near overflow
t = torch.full((10000,), 3e38, dtype=torch.float32)
r = torch.mean(t)
print(f"  mean([3e38]*10K) = {r.item()} (expected: 3e38, overflow? {r.item() == float('inf')})")

# Subnormals
t = torch.full((10000,), 1e-40, dtype=torch.float32)
r = torch.mean(t)
print(f"  mean([1e-40]*10K) = {r.item():.6e} (expected: 1e-40)")

# Catastrophic cancellation
t = torch.zeros(1000000, dtype=torch.float32)
t[0] = 1e8; t[1] = -1e8
for i in range(2, 1000000): t[i] = 1.0
gt = (1e8 - 1e8 + 999998.0) / 1000000
print(f"  mean([1e8, -1e8, 1, 1, ...]) = {torch.mean(t).item():.10f} (expected: {gt:.10f})")

# Integer mean (PyTorch doesn't support)
try:
    t = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
    print(f"  mean(int64) = {torch.mean(t).item()}")
except Exception as e:
    print(f"  mean(int64) = ERROR: {e}")

print(f"\n{'='*120}")
print("COMPLETE")
