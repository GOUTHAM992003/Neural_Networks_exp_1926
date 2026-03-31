#!/usr/bin/env python3
"""
Compare PyTorch argmax timings at same sizes as our benchmark.
Same sizes: 10K, 100K, 1M, 10M, 50M, 100M — float32 and float64
"""
import torch
import time

def bench(fn, warmup=5, iters=50):
    for _ in range(warmup): fn()
    times = []
    for _ in range(iters):
        s = time.perf_counter_ns()
        fn()
        e = time.perf_counter_ns()
        times.append((e - s) / 1000.0)
    times.sort()
    trim = max(1, len(times) // 10)
    return sum(times[trim:-trim]) / (len(times) - 2*trim)

torch.set_num_threads(28)
print(f"PyTorch {torch.__version__}, Threads: {torch.get_num_threads()}")
print()
print(f"{'Size':>12} | {'Dtype':>10} | {'PyTorch argmax':>16} | {'PyTorch argmin':>16}")
print("-" * 70)

sizes = [10000, 100000, 1000000, 10000000, 50000000, 100000000]
for dtype in [torch.float32, torch.float64]:
    for size in sizes:
        t = torch.randn(size, dtype=dtype)
        t_max = bench(lambda: torch.argmax(t))
        t_min = bench(lambda: torch.argmin(t))
        print(f"{size:>12,} | {str(dtype):>10} | {t_max:>13.1f} μs | {t_min:>13.1f} μs")
    print()
