#!/usr/bin/env python3
"""
Comprehensive Mean/NanMean Benchmark: Our Library vs PyTorch
Tests: speed, precision, all dtypes, all reduction paths, edge cases
"""
import torch
import subprocess
import time
import numpy as np
import sys
import os

torch.set_num_threads(28)

def bench_pytorch(fn, warmup=5, iters=30):
    for _ in range(warmup): fn()
    times = []
    for _ in range(iters):
        s = time.perf_counter_ns()
        result = fn()
        e = time.perf_counter_ns()
        times.append((e - s) / 1000)  # microseconds
    times.sort()
    median = times[len(times)//2]
    return median, result

def print_header(title):
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")

def print_subheader(title):
    print(f"\n--- {title} ---")

# ============================================================
# SPEED BENCHMARKS
# ============================================================
print_header("SPEED BENCHMARK: Mean/NanMean — PyTorch (28 threads)")
print(f"PyTorch version: {torch.__version__}")
print(f"Threads: {torch.get_num_threads()}")

# --- Regular Mean ---
print_subheader("Regular Mean — InnerContiguous (reduce last dim)")
for shape in [(32, 768), (32, 4096), (4096, 768), (256, 4096), (2048, 50176),
              (49152, 128), (1000, 100), (1000, 10000), (1000, 100000),
              (10, 1000000), (1000000, 10), (100, 10000000)]:
    for dt in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
        t = torch.randn(*shape, dtype=torch.float32).to(dt)
        us, result = bench_pytorch(lambda: torch.mean(t, dim=-1))
        print(f"  {str(shape):20s} {str(dt):18s} → {us:10.0f} μs  (output shape: {tuple(result.shape)})")

print_subheader("Regular Mean — OuterContiguous (reduce first dim)")
for shape in [(1000, 256), (10000, 100), (100, 10000), (50176, 2048)]:
    for dt in [torch.float32, torch.float64]:
        t = torch.randn(*shape, dtype=dt)
        us, result = bench_pytorch(lambda: torch.mean(t, dim=0))
        print(f"  {str(shape):20s} {str(dt):18s} → {us:10.0f} μs  (output shape: {tuple(result.shape)})")

print_subheader("Regular Mean — Generic (reduce mixed dims)")
for shape in [(100, 200, 50), (32, 128, 768), (16, 64, 32, 32)]:
    for dims in [(0, 2), (0,), (1, 2)]:
        dt = torch.float32
        t = torch.randn(*shape, dtype=dt)
        try:
            us, result = bench_pytorch(lambda: torch.mean(t, dim=dims))
            print(f"  {str(shape):20s} dims={str(dims):10s} → {us:10.0f} μs  (output: {tuple(result.shape)})")
        except:
            pass

print_subheader("Regular Mean — Full Reduction (1 output)")
for size in [1000, 10000, 100000, 1000000, 10000000, 50000000]:
    for dt in [torch.float32, torch.float64]:
        t = torch.randn(size, dtype=dt)
        us, result = bench_pytorch(lambda: torch.mean(t))
        print(f"  ({size:>10d},) {str(dt):18s} → {us:10.0f} μs")

# --- NanMean ---
print_subheader("NanMean — InnerContiguous (10% NaN)")
for shape in [(32, 768), (4096, 768), (256, 4096), (2048, 50176),
              (1000, 10000), (10, 1000000), (1000000, 10)]:
    for dt in [torch.float32, torch.float64]:
        t = torch.randn(*shape, dtype=dt)
        mask = torch.rand(*shape) < 0.1
        t[mask] = float('nan')
        us, result = bench_pytorch(lambda: torch.nanmean(t, dim=-1))
        print(f"  {str(shape):20s} {str(dt):18s} → {us:10.0f} μs")

print_subheader("NanMean — Full Reduction (10% NaN)")
for size in [10000, 100000, 1000000, 10000000, 50000000]:
    t = torch.randn(size, dtype=torch.float32)
    mask = torch.rand(size) < 0.1
    t[mask] = float('nan')
    us, result = bench_pytorch(lambda: torch.nanmean(t))
    print(f"  ({size:>10d},) float32 → {us:10.0f} μs")

print_subheader("NanMean — Varying NaN percentage (1000, 10000)")
for nan_pct in [0.0, 0.01, 0.1, 0.5, 0.9, 0.99]:
    t = torch.randn(1000, 10000, dtype=torch.float32)
    if nan_pct > 0:
        mask = torch.rand(1000, 10000) < nan_pct
        t[mask] = float('nan')
    us, result = bench_pytorch(lambda: torch.nanmean(t, dim=-1))
    print(f"  {nan_pct*100:5.1f}% NaN → {us:10.0f} μs")

# ============================================================
# PRECISION BENCHMARKS
# ============================================================
print_header("PRECISION BENCHMARK: Mean/NanMean vs NumPy (ground truth)")

print_subheader("Regular Mean Precision")
np.random.seed(42)
for desc, data_fn in [
    ("Uniform [-1,1]", lambda n: np.random.uniform(-1, 1, n).astype(np.float32)),
    ("Gaussian N(0,1)", lambda n: np.random.randn(n).astype(np.float32)),
    ("Large mean + tiny var", lambda n: (1e6 + np.random.randn(n) * 1e-3).astype(np.float32)),
    ("Mixed scale [1e-6, 1e6]", lambda n: np.random.uniform(1e-6, 1e6, n).astype(np.float32)),
    ("Near FLT_MAX", lambda n: np.random.uniform(1e37, 3e38, n).astype(np.float32)),
    ("Subnormals", lambda n: np.random.uniform(1e-45, 1e-38, n).astype(np.float32)),
]:
    for N in [1000, 100000, 10000000]:
        data_np = data_fn(N)
        gt = np.float64(np.sum(data_np.astype(np.float64)) / N)  # fp64 ground truth
        pt = torch.mean(torch.from_numpy(data_np)).item()
        rel_err_pt = abs(pt - gt) / abs(gt) if gt != 0 else 0
        print(f"  {desc:25s} N={N:>10d}  PyTorch rel_err={rel_err_pt:.2e}")

print_subheader("NanMean Precision (10% NaN)")
for N in [1000, 100000, 10000000]:
    data_np = np.random.randn(N).astype(np.float32)
    nan_mask = np.random.rand(N) < 0.1
    data_np[nan_mask] = np.nan
    gt = np.float64(np.nansum(data_np.astype(np.float64)) / np.sum(~nan_mask))
    pt = torch.nanmean(torch.from_numpy(data_np.copy())).item()
    rel_err_pt = abs(pt - gt) / abs(gt) if gt != 0 else 0
    print(f"  N={N:>10d}  PyTorch rel_err={rel_err_pt:.2e}")

# ============================================================
# EDGE CASES
# ============================================================
print_header("EDGE CASES")

print_subheader("All NaN tensor")
t = torch.full((100,), float('nan'), dtype=torch.float32)
print(f"  nanmean([NaN]*100) = {torch.nanmean(t).item()}")
print(f"  mean([NaN]*100) = {torch.mean(t).item()}")

print_subheader("Single element")
t = torch.tensor([42.0])
print(f"  mean([42.0]) = {torch.mean(t).item()}")
print(f"  nanmean([42.0]) = {torch.nanmean(t).item()}")

print_subheader("Mixed NaN positions")
t = torch.tensor([1.0, float('nan'), 3.0, float('nan'), 5.0])
print(f"  nanmean([1, NaN, 3, NaN, 5]) = {torch.nanmean(t).item()} (expected: 3.0)")

print_subheader("Large reduction with precision check")
t = torch.ones(100000000, dtype=torch.float32)
print(f"  mean(ones(100M)) = {torch.mean(t).item()} (expected: 1.0)")
t[0] = 1e8
print(f"  mean(ones(100M) + [1e8 at pos 0]) = {torch.mean(t).item():.10f} (expected: ~2.0)")

print(f"\n{'='*100}")
print("BENCHMARK COMPLETE")
print(f"{'='*100}")
