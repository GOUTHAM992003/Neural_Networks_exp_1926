#!/usr/bin/env python3
"""
Test: Force PyTorch to use binary_kernel_reduce (non-lastdim) path
by making the tensor non-contiguous, then compare with lastdim path.
"""
import torch
import time

def bench(fn, warmup=5, iters=50):
    for _ in range(warmup):
        fn()
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
print(f"Threads: {torch.get_num_threads()}")
print()

for size in [1_000_000, 10_000_000, 50_000_000]:
    # Path A: 1D contiguous → lastdim (1 thread)
    t_contig = torch.randn(size)
    time_lastdim = bench(lambda: torch.argmax(t_contig))

    # Path B: Make it non-contiguous by adding a dummy dim and transposing
    # Shape (1, size) transposed to (size, 1) → stride is not lastdim
    t_2d = torch.randn(2, size)
    # Reduce along dim=1 (lastdim) → this should use lastdim path
    time_partial_lastdim = bench(lambda: torch.argmax(t_2d, dim=1))

    # Path C: Reduce along dim=0 (NOT lastdim) → forces binary_kernel_reduce
    t_2d_outer = torch.randn(size, 2)
    time_partial_outer = bench(lambda: torch.argmax(t_2d_outer, dim=0))

    # Path D: Reshape to (1, size) and reduce dim=0 → NOT lastdim, full-ish reduction
    # This forces binary_kernel_reduce path for near-full reduction
    t_reshaped = torch.randn(1, size)
    time_dim0 = bench(lambda: torch.argmax(t_reshaped, dim=1))

    # Path E: Use torch.max which returns (values, indices) - different code path
    time_max = bench(lambda: torch.max(t_contig))

    print(f"Size: {size:>12,}")
    print(f"  argmax (1D contiguous, lastdim path):    {time_lastdim:>10.1f} μs")
    print(f"  argmax (2×N, dim=1, lastdim partial):    {time_partial_lastdim:>10.1f} μs")
    print(f"  argmax (N×2, dim=0, outer/generic):      {time_partial_outer:>10.1f} μs")
    print(f"  argmax (1×N, dim=1, lastdim but 1 out):  {time_dim0:>10.1f} μs")
    print(f"  max (1D, value only, uses reduce_vec):    {time_max:>10.1f} μs")
    print()
