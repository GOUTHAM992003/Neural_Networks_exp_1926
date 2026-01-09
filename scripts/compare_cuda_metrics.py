"""
CUDA GPU Performance Comparison
Generates SEPARATE comparison CSVs for GPU benchmarks
"""

import pandas as pd
import os

print("="*80)
print("CUDA GPU PERFORMANCE COMPARISON")
print("="*80)

# CUDA libraries to compare (NumPy excluded - CPU only)
CUDA_LIBRARIES = ['pytorch_cuda', 'libtorch_cuda', 'tensorlib_cuda']
METRICS = ['timings', 'throughput', 'bandwidth', 'flops']

# Create comparison directory
os.makedirs("benchmark_results/comparison_cuda", exist_ok=True)

# Function to load metric CSV for a library
def load_metric(library, metric):
    """Load a specific metric CSV for a CUDA library"""
    try:
        filepath = f"benchmark_results/{library}/{library}_{metric}.csv"
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        return None

# Check which CUDA libraries have results
available_libs = []
for lib in CUDA_LIBRARIES:
    if load_metric(lib, 'timings') is not None:
        available_libs.append(lib)
        print(f"✓ Found {lib} CUDA results")

if len(available_libs) == 0:
    print("\n⚠ No CUDA benchmark results found!")
    print("Run CUDA benchmarks first:")
    print("  python3 pytorch_tests/pytorch_cuda_benchmark.py")
    exit(0)

print(f"\n✓ Found {len(available_libs)} CUDA library/libraries\n")

# ============================================================================
# 1. TIMING COMPARISON
# ============================================================================
print("1. Creating CUDA timing comparison...")
timing_data = {}
for lib in available_libs:
    df = load_metric(lib, 'timings')
    if df is not None:
        timing_data[lib] = df.set_index('operation')

if len(timing_data) >= 1:
    all_ops = set()
    for lib_df in timing_data.values():
        all_ops.update(lib_df.index)
    
    comparison_rows = []
    for op in sorted(all_ops):
        row = {'operation': op}
        for lib in CUDA_LIBRARIES:
            if lib in timing_data and op in timing_data[lib].index:
                row[f'{lib}_mean_ms'] = timing_data[lib].loc[op, 'mean_ms']
                row[f'{lib}_std_ms'] = timing_data[lib].loc[op, 'std_ms']
            else:
                row[f'{lib}_mean_ms'] = None
                row[f'{lib}_std_ms'] = None
        comparison_rows.append(row)
    
    timing_comp_df = pd.DataFrame(comparison_rows)
    timing_comp_df.to_csv("benchmark_results/comparison_cuda/cuda_timing_comparison.csv", index=False)
    print("   ✓ Saved: benchmark_results/comparison_cuda/cuda_timing_comparison.csv")
else:
    print("   ⚠ No timing data available")

# ============================================================================
# 2. THROUGHPUT COMPARISON
# ============================================================================
print("\n2. Creating CUDA throughput comparison...")
throughput_data = {}
for lib in available_libs:
    df = load_metric(lib, 'throughput')
    if df is not None:
        throughput_data[lib] = df.set_index('operation')

if len(throughput_data) >= 1:
    all_ops = set()
    for lib_df in throughput_data.values():
        all_ops.update(lib_df.index)
    
    comparison_rows = []
    for op in sorted(all_ops):
        row = {'operation': op}
        for lib in CUDA_LIBRARIES:
            if lib in throughput_data and op in throughput_data[lib].index:
                row[f'{lib}_throughput_elem_per_sec'] = throughput_data[lib].loc[op, 'throughput_elem_per_sec']
            else:
                row[f'{lib}_throughput_elem_per_sec'] = None
        comparison_rows.append(row)
    
    throughput_comp_df = pd.DataFrame(comparison_rows)
    throughput_comp_df.to_csv("benchmark_results/comparison_cuda/cuda_throughput_comparison.csv", index=False)
    print("   ✓ Saved: benchmark_results/comparison_cuda/cuda_throughput_comparison.csv")
else:
    print("   ⚠ No throughput data available")

# ============================================================================
# 3. BANDWIDTH COMPARISON
# ============================================================================
print("\n3. Creating CUDA bandwidth comparison...")
bandwidth_data = {}
for lib in available_libs:
    df = load_metric(lib, 'bandwidth')
    if df is not None:
        bandwidth_data[lib] = df.set_index('operation')

if len(bandwidth_data) >= 1:
    all_ops = set()
    for lib_df in bandwidth_data.values():
        all_ops.update(lib_df.index)
    
    comparison_rows = []
    for op in sorted(all_ops):
        row = {'operation': op}
        for lib in CUDA_LIBRARIES:
            if lib in bandwidth_data and op in bandwidth_data[lib].index:
                row[f'{lib}_bandwidth_gb_per_sec'] = bandwidth_data[lib].loc[op, 'memory_bandwidth_gb_per_sec']
            else:
                row[f'{lib}_bandwidth_gb_per_sec'] = None
        comparison_rows.append(row)
    
    bandwidth_comp_df = pd.DataFrame(comparison_rows)
    bandwidth_comp_df.to_csv("benchmark_results/comparison_cuda/cuda_bandwidth_comparison.csv", index=False)
    print("   ✓ Saved: benchmark_results/comparison_cuda/cuda_bandwidth_comparison.csv")
else:
    print("   ⚠ No bandwidth data available")

# ============================================================================
# 4. FLOPS COMPARISON
# ============================================================================
print("\n4. Creating CUDA FLOPS comparison...")
flops_data = {}
for lib in available_libs:
    df = load_metric(lib, 'flops')
    if df is not None:
        flops_data[lib] = df.set_index('operation')

if len(flops_data) >= 1:
    all_ops = set()
    for lib_df in flops_data.values():
        all_ops.update(lib_df.index)
    
    comparison_rows = []
    for op in sorted(all_ops):
        row = {'operation': op}
        for lib in CUDA_LIBRARIES:
            if lib in flops_data and op in flops_data[lib].index:
                row[f'{lib}_gflops'] = flops_data[lib].loc[op, 'gflops']
            else:
                row[f'{lib}_gflops'] = None
        comparison_rows.append(row)
    
    flops_comp_df = pd.DataFrame(comparison_rows)
    flops_comp_df.to_csv("benchmark_results/comparison_cuda/cuda_flops_comparison.csv", index=False)
    print("   ✓ Saved: benchmark_results/comparison_cuda/cuda_flops_comparison.csv")
else:
    print("   ⚠ No FLOPS data available")

# ============================================================================
# PRINT SUMMARY
# ============================================================================
print("\n" + "="*80)
print("CUDA COMPARISON SUMMARY")
print("="*80)

print(f"\nCUDA Libraries Available: {', '.join(available_libs)}")
print(f"\nGenerated CUDA Comparison Files:")
print(f"  1. cuda_timing_comparison.csv     - GPU execution time comparison")
print(f"  2. cuda_throughput_comparison.csv - GPU data throughput comparison")
print(f"  3. cuda_bandwidth_comparison.csv  - GPU memory bandwidth comparison")
print(f"  4. cuda_flops_comparison.csv      - GPU GFLOPS comparison")

# Quick preview of timing comparison
if len(timing_data) >= 1:
    print(f"\n" + "="*80)
    print("CUDA TIMING COMPARISON PREVIEW (First 10 operations)")
    print("="*80)
    print(f"\n{'Operation':<20}", end="")
    for lib in available_libs:
        if lib in timing_data:
            print(f"{lib:>20}", end="")
    print()
    print("-"*80)
    
    for _, row in timing_comp_df.head(10).iterrows():
        op = row['operation']
        print(f"{op:<20}", end="")
        
        for lib in available_libs:
            col = f'{lib}_mean_ms'
            if col in row and pd.notna(row[col]):
                print(f"{row[col]:>18.4f}ms", end="")
            else:
                print(f"{'N/A':>20}", end="")
        print()

print("\n" + "="*80)
print("✅ CUDA COMPARISON COMPLETE!")
print("="*80 + "\n")
