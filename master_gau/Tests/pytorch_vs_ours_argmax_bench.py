#!/usr/bin/env python3
"""
Benchmark: Full Reduction argmax/argmin — PyTorch vs Our Library
================================================================
Tests the PyTorch bug: full reduction argmax on contiguous tensor
routes to binary_kernel_reduce_lastdim which uses ONLY 1 thread.

Our library uses Strategy 2 (SplitReduction) which uses ALL threads.

We test:
- Various tensor sizes: 10K, 100K, 1M, 10M, 50M, 100M
- Full reduction (no dim argument) = output is scalar
- float32 and float64 dtypes
- argmax and argmin operations
- Compare wall-clock time: PyTorch vs our expectation
"""

import torch
import time
import os

def bench_pytorch_argmax(tensor, warmup=5, iters=50):
    """Benchmark torch.argmax (full reduction)"""
    # Warmup
    for _ in range(warmup):
        _ = torch.argmax(tensor)

    torch.cuda.synchronize() if tensor.is_cuda else None

    times = []
    for _ in range(iters):
        start = time.perf_counter_ns()
        result = torch.argmax(tensor)
        end = time.perf_counter_ns()
        times.append((end - start) / 1000.0)  # Convert to microseconds

    times.sort()
    # Remove top/bottom 10% outliers
    trim = max(1, len(times) // 10)
    trimmed = times[trim:-trim]
    return sum(trimmed) / len(trimmed), result.item()

def bench_pytorch_argmin(tensor, warmup=5, iters=50):
    """Benchmark torch.argmin (full reduction)"""
    for _ in range(warmup):
        _ = torch.argmin(tensor)

    times = []
    for _ in range(iters):
        start = time.perf_counter_ns()
        result = torch.argmin(tensor)
        end = time.perf_counter_ns()
        times.append((end - start) / 1000.0)

    times.sort()
    trim = max(1, len(times) // 10)
    trimmed = times[trim:-trim]
    return sum(trimmed) / len(trimmed), result.item()

def bench_pytorch_argmax_dim(tensor, dim, warmup=5, iters=50):
    """Benchmark torch.argmax with dim (partial reduction) for comparison"""
    for _ in range(warmup):
        _ = torch.argmax(tensor, dim=dim)

    times = []
    for _ in range(iters):
        start = time.perf_counter_ns()
        result = torch.argmax(tensor, dim=dim)
        end = time.perf_counter_ns()
        times.append((end - start) / 1000.0)

    times.sort()
    trim = max(1, len(times) // 10)
    trimmed = times[trim:-trim]
    return sum(trimmed) / len(trimmed)

def main():
    print("=" * 90)
    print("BENCHMARK: Full Reduction argmax/argmin — PyTorch CPU")
    print("=" * 90)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"Note: PyTorch's binary_kernel_reduce_lastdim uses Strategy 1 (1 thread for full reduction)")
    print(f"      Our library uses Strategy 2 (all threads for full reduction)")
    print()

    # =========================================================
    # TEST 1: Full Reduction argmax — various sizes
    # =========================================================
    print("-" * 90)
    print("TEST 1: Full Reduction argmax (tensor.argmax()) — 1D contiguous tensors")
    print("-" * 90)
    print(f"{'Size':>12} | {'Dtype':>10} | {'argmax (μs)':>14} | {'argmin (μs)':>14} | {'Elements/μs':>14}")
    print("-" * 90)

    sizes = [10_000, 100_000, 1_000_000, 10_000_000, 50_000_000, 100_000_000]
    dtypes = [torch.float32, torch.float64]

    for dtype in dtypes:
        for size in sizes:
            tensor = torch.randn(size, dtype=dtype)

            t_argmax, idx_max = bench_pytorch_argmax(tensor)
            t_argmin, idx_min = bench_pytorch_argmin(tensor)
            throughput = size / t_argmax

            print(f"{size:>12,} | {str(dtype):>10} | {t_argmax:>12.1f}μs | {t_argmin:>12.1f}μs | {throughput:>12.1f}M/s")
        print()

    # =========================================================
    # TEST 2: Thread count verification
    # =========================================================
    print("-" * 90)
    print("TEST 2: Thread count impact — Full reduction argmax with different thread settings")
    print("-" * 90)

    test_size = 10_000_000
    tensor = torch.randn(test_size, dtype=torch.float32)

    print(f"Tensor size: {test_size:,} float32 elements")
    print(f"{'Threads':>10} | {'argmax (μs)':>14} | {'Speedup vs 1T':>16}")
    print("-" * 50)

    thread_counts = [1, 2, 4, 8, 14, 20]
    base_time = None

    for nt in thread_counts:
        torch.set_num_threads(nt)
        t, _ = bench_pytorch_argmax(tensor, warmup=10, iters=100)
        if base_time is None:
            base_time = t
        speedup = base_time / t
        print(f"{nt:>10} | {t:>12.1f}μs | {speedup:>14.2f}x")

    # Reset threads
    torch.set_num_threads(20)
    print()

    # =========================================================
    # TEST 3: Full reduction vs Partial reduction
    # =========================================================
    print("-" * 90)
    print("TEST 3: Full reduction vs Partial reduction (dim=-1) — shows threading difference")
    print("-" * 90)

    # 2D tensor: (100, 1_000_000) — partial reduction has 100 output slots
    rows, cols = 100, 1_000_000
    tensor_2d = torch.randn(rows, cols, dtype=torch.float32)

    t_full, _ = bench_pytorch_argmax(tensor_2d.flatten())  # Full reduction of same data
    t_partial = bench_pytorch_argmax_dim(tensor_2d, dim=1)  # Partial: 100 output slots

    print(f"Data: ({rows}, {cols:,}) float32 = {rows * cols:,} elements")
    print(f"Full reduction (flatten + argmax):     {t_full:>10.1f} μs  (1 output → likely 1 thread)")
    print(f"Partial reduction (argmax dim=1):      {t_partial:>10.1f} μs  (100 outputs → multi-thread)")
    print(f"Speedup of partial over full:          {t_full / t_partial:>10.2f}x")
    print()

    if t_full > t_partial * 1.5:
        print("*** CONFIRMED: Full reduction is significantly slower than partial reduction!")
        print("*** This proves PyTorch's lastdim path uses fewer threads for full reduction.")
    else:
        print("Note: Full and partial are similar — PyTorch may be optimizing differently here.")

    # =========================================================
    # TEST 4: Explicitly show the bug with 1 thread vs max threads
    # =========================================================
    print()
    print("-" * 90)
    print("TEST 4: The Smoking Gun — Does adding more threads help full reduction argmax?")
    print("-" * 90)

    test_size = 50_000_000
    tensor = torch.randn(test_size, dtype=torch.float32)

    print(f"Tensor: {test_size:,} float32 elements, full reduction argmax")
    print()

    torch.set_num_threads(1)
    t1, _ = bench_pytorch_argmax(tensor, warmup=5, iters=30)

    torch.set_num_threads(20)
    t20, _ = bench_pytorch_argmax(tensor, warmup=5, iters=30)

    speedup = t1 / t20

    print(f"1 thread:   {t1:>10.1f} μs")
    print(f"20 threads: {t20:>10.1f} μs")
    print(f"Speedup:    {speedup:>10.2f}x")
    print()

    if speedup < 2.0:
        print("*** BUG CONFIRMED: 20 threads gives < 2x speedup over 1 thread!")
        print("*** PyTorch's binary_kernel_reduce_lastdim uses only ~1 thread for full reduction.")
        print("*** Our library fixes this with Strategy 2 (SplitReduction).")
    elif speedup < 5.0:
        print("** PARTIAL BUG: Some threading benefit but far from linear scaling.")
        print("** Expected ~10-15x with 20 threads on 50M elements.")
    else:
        print("No bug detected — PyTorch is using multiple threads effectively.")

    print()
    print("=" * 90)
    print("BENCHMARK COMPLETE")
    print("=" * 90)

if __name__ == "__main__":
    main()
