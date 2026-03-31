#!/usr/bin/env python3
"""
PyTorch benchmark: Full AND Partial reduction argmax/argmin
Compare with our library's results from our_argmax_bench.cpp
"""
import torch, time

def bench(fn, warmup=5, iters=50):
    for _ in range(warmup): fn()
    times = []
    for _ in range(iters):
        s = time.perf_counter_ns(); fn(); e = time.perf_counter_ns()
        times.append((e - s) / 1000.0)
    times.sort()
    trim = max(1, len(times) // 10)
    return sum(times[trim:-trim]) / (len(times) - 2*trim)

torch.set_num_threads(28)
print(f"PyTorch {torch.__version__}, Threads: {torch.get_num_threads()}")
print()

# FULL REDUCTION
print("=" * 80)
print("FULL REDUCTION: tensor.argmax() — no dim (scalar output)")
print("=" * 80)
print(f"{'Size':>12} | {'Dtype':>8} | {'argmax (μs)':>14} | {'argmin (μs)':>14}")
print("-" * 60)
for dtype in [torch.float32, torch.float64]:
    for size in [10000, 100000, 1000000, 10000000, 50000000, 100000000]:
        t = torch.randn(size, dtype=dtype)
        am = bench(lambda: torch.argmax(t))
        ai = bench(lambda: torch.argmin(t))
        dn = "f32" if dtype == torch.float32 else "f64"
        print(f"{size:>12,} | {dn:>8} | {am:>12.1f}μs | {ai:>12.1f}μs")
    print()

# PARTIAL REDUCTION — reduce last dim
print("=" * 80)
print("PARTIAL REDUCTION: tensor.argmax(dim=-1) — reduce last dimension")
print("=" * 80)
print(f"{'Shape':>20} | {'Dtype':>8} | {'argmax (μs)':>14} | {'argmin (μs)':>14}")
print("-" * 70)
shapes = [(100, 100000), (1000, 10000), (10000, 1000), (64, 1000000), (256, 100000)]
for rows, cols in shapes:
    for dtype in [torch.float32]:
        t = torch.randn(rows, cols, dtype=dtype)
        am = bench(lambda: torch.argmax(t, dim=-1))
        ai = bench(lambda: torch.argmin(t, dim=-1))
        print(f"({rows:>6},{cols:>8}) | {'f32':>8} | {am:>12.1f}μs | {ai:>12.1f}μs")
print()

# PARTIAL REDUCTION — reduce first dim (outer)
print("=" * 80)
print("PARTIAL REDUCTION: tensor.argmax(dim=0) — reduce first dimension (outer)")
print("=" * 80)
print(f"{'Shape':>20} | {'Dtype':>8} | {'argmax (μs)':>14} | {'argmin (μs)':>14}")
print("-" * 70)
for rows, cols in [(100000, 100), (10000, 1000), (1000, 10000), (1000000, 64), (100000, 256)]:
    for dtype in [torch.float32]:
        t = torch.randn(rows, cols, dtype=dtype)
        am = bench(lambda: torch.argmax(t, dim=0))
        ai = bench(lambda: torch.argmin(t, dim=0))
        print(f"({rows:>6},{cols:>8}) | {'f32':>8} | {am:>12.1f}μs | {ai:>12.1f}μs")
