#!/usr/bin/env python3
"""
Verify if binary_kernel_reduce's parallel_for is being called
by checking if different thread counts change timing for non-lastdim argmax
"""
import torch
import time

def bench(fn, iters=30):
    for _ in range(3): fn()
    times = []
    for _ in range(iters):
        s = time.perf_counter_ns()
        fn()
        e = time.perf_counter_ns()
        times.append((e - s) / 1000.0)
    times.sort()
    trim = max(1, len(times) // 10)
    return sum(times[trim:-trim]) / (len(times) - 2*trim)

size = 10_000_000

print("=== 1D contiguous argmax (lastdim path) ===")
t = torch.randn(size)
for nt in [1, 4, 14, 28]:
    torch.set_num_threads(nt)
    us = bench(lambda: torch.argmax(t))
    print(f"  {nt:>2} threads: {us:>10.1f} μs")

print()
print("=== N×2 argmax dim=0 (generic/binary_kernel_reduce path) ===")
t2 = torch.randn(size, 2)
for nt in [1, 4, 14, 28]:
    torch.set_num_threads(nt)
    us = bench(lambda: torch.argmax(t2, dim=0))
    print(f"  {nt:>2} threads: {us:>10.1f} μs")

print()
print("=== 1D contiguous max (binary_kernel_reduce_vec / parallel_reduce path) ===")
t3 = torch.randn(size)
for nt in [1, 4, 14, 28]:
    torch.set_num_threads(nt)
    us = bench(lambda: torch.max(t3))
    print(f"  {nt:>2} threads: {us:>10.1f} μs")
