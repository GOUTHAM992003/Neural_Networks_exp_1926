"""
Multi-Library Performance Comparison
Generates SEPARATE comparison CSVs for each metric across all libraries
"""

import pandas as pd
import os

print("="*80)
print("MULTI-LIBRARY PERFORMANCE COMPARISON")
print("="*80)

# Libraries to compare
LIBRARIES = ['numpy', 'pytorch', 'libtorch', 'tensorlib', 'pytorch_cuda', 'libtorch_cuda', 'tensorlib_cuda']
METRICS = ['values', 'timings', 'throughput', 'bandwidth', 'flops']

# Create comparison directory
os.makedirs("benchmark_results/comparison", exist_ok=True)

# Function to load metric CSV for a library
def load_metric(library, metric):
    """Load a specific metric CSV for a library"""
    try:
        filepath = f"benchmark_results/{library}/{library}_{metric}.csv"
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        return None

# ============================================================================
# 1. TIMING COMPARISON
# ============================================================================
print("\n1. Creating timing comparison...")
timing_data = {}
for lib in LIBRARIES:
    df = load_metric(lib, 'timings')
    if df is not None:
        timing_data[lib] = df.set_index('operation')
        print(f"   ✓ Loaded {lib} timings")

if len(timing_data) >= 2:
    # Get all unique operations
    all_ops = set()
    for lib_df in timing_data.values():
        all_ops.update(lib_df.index)
    
    # Create comparison
    comparison_rows = []
    for op in sorted(all_ops):
        row = {'operation': op}
        for lib in LIBRARIES:
            if lib in timing_data and op in timing_data[lib].index:
                row[f'{lib}_mean_ms'] = timing_data[lib].loc[op, 'mean_ms']
                row[f'{lib}_std_ms'] = timing_data[lib].loc[op, 'std_ms']
            else:
                row[f'{lib}_mean_ms'] = None
                row[f'{lib}_std_ms'] = None
        comparison_rows.append(row)
    
    timing_comp_df = pd.DataFrame(comparison_rows)
    timing_comp_df.to_csv("benchmark_results/comparison/timing_comparison.csv", index=False)
    print("   ✓ Saved: benchmark_results/comparison/timing_comparison.csv")
else:
    print("   ⚠ Need at least 2 libraries for timing comparison")

# ============================================================================
# 2. THROUGHPUT COMPARISON
# ============================================================================
print("\n2. Creating throughput comparison...")
throughput_data = {}
for lib in LIBRARIES:
    df = load_metric(lib, 'throughput')
    if df is not None:
        throughput_data[lib] = df.set_index('operation')
        print(f"   ✓ Loaded {lib} throughput")

if len(throughput_data) >= 2:
    all_ops = set()
    for lib_df in throughput_data.values():
        all_ops.update(lib_df.index)
    
    comparison_rows = []
    for op in sorted(all_ops):
        row = {'operation': op}
        for lib in LIBRARIES:
            if lib in throughput_data and op in throughput_data[lib].index:
                row[f'{lib}_throughput_elem_per_sec'] = throughput_data[lib].loc[op, 'throughput_elem_per_sec']
            else:
                row[f'{lib}_throughput_elem_per_sec'] = None
        comparison_rows.append(row)
    
    throughput_comp_df = pd.DataFrame(comparison_rows)
    throughput_comp_df.to_csv("benchmark_results/comparison/throughput_comparison.csv", index=False)
    print("   ✓ Saved: benchmark_results/comparison/throughput_comparison.csv")
else:
    print("   ⚠ Need at least 2 libraries for throughput comparison")

# ============================================================================
# 3. BANDWIDTH COMPARISON
# ============================================================================
print("\n3. Creating bandwidth comparison...")
bandwidth_data = {}
for lib in LIBRARIES:
    df = load_metric(lib, 'bandwidth')
    if df is not None:
        bandwidth_data[lib] = df.set_index('operation')
        print(f"   ✓ Loaded {lib} bandwidth")

if len(bandwidth_data) >= 2:
    all_ops = set()
    for lib_df in bandwidth_data.values():
        all_ops.update(lib_df.index)
    
    comparison_rows = []
    for op in sorted(all_ops):
        row = {'operation': op}
        for lib in LIBRARIES:
            if lib in bandwidth_data and op in bandwidth_data[lib].index:
                row[f'{lib}_bandwidth_gb_per_sec'] = bandwidth_data[lib].loc[op, 'memory_bandwidth_gb_per_sec']
            else:
                row[f'{lib}_bandwidth_gb_per_sec'] = None
        comparison_rows.append(row)
    
    bandwidth_comp_df = pd.DataFrame(comparison_rows)
    bandwidth_comp_df.to_csv("benchmark_results/comparison/bandwidth_comparison.csv", index=False)
    print("   ✓ Saved: benchmark_results/comparison/bandwidth_comparison.csv")
else:
    print("   ⚠ Need at least 2 libraries for bandwidth comparison")

# ============================================================================
# 4. FLOPS COMPARISON
# ============================================================================
print("\n4. Creating FLOPS comparison...")
flops_data = {}
for lib in LIBRARIES:
    df = load_metric(lib, 'flops')
    if df is not None:
        flops_data[lib] = df.set_index('operation')
        print(f"   ✓ Loaded {lib} FLOPS")

if len(flops_data) >= 2:
    all_ops = set()
    for lib_df in flops_data.values():
        all_ops.update(lib_df.index)
    
    comparison_rows = []
    for op in sorted(all_ops):
        row = {'operation': op}
        for lib in LIBRARIES:
            if lib in flops_data and op in flops_data[lib].index:
                row[f'{lib}_gflops'] = flops_data[lib].loc[op, 'gflops']
            else:
                row[f'{lib}_gflops'] = None
        comparison_rows.append(row)
    
    flops_comp_df = pd.DataFrame(comparison_rows)
    flops_comp_df.to_csv("benchmark_results/comparison/flops_comparison.csv", index=False)
    print("   ✓ Saved: benchmark_results/comparison/flops_comparison.csv")
else:
    print("   ⚠ Need at least 2 libraries for FLOPS comparison")

# ============================================================================
# 5. VALUES COMPARISON (Precision Check)
# ============================================================================
print("\n5. Creating values/precision comparison...")
values_data = {}
for lib in LIBRARIES:
    df = load_metric(lib, 'values')
    if df is not None:
        values_data[lib] = df
        print(f"   ✓ Loaded {lib} values ({len(df)} rows)")

if len(values_data) >= 2:
    # Use first library as baseline
    baseline_lib = list(values_data.keys())[0]
    baseline_df = values_data[baseline_lib]
    
    # Get operation columns (exclude index columns)
    exclude_cols = ['Idx', 'Batch', 'Row', 'Column', 'TENSOR a', 'TENSOR b']
    op_columns = [col for col in baseline_df.columns if col not in exclude_cols]
    
    precision_summary = []
    
    for compare_lib in values_data.keys():
        if compare_lib == baseline_lib:
            continue
        
        compare_df = values_data[compare_lib]
        
        for op_col in op_columns:
            if op_col in compare_df.columns:
                try:
                    baseline_vals = pd.to_numeric(baseline_df[op_col], errors='coerce').values
                    compare_vals = pd.to_numeric(compare_df[op_col], errors='coerce').values
                    
                    # Skip if not numeric
                    if baseline_vals is None or compare_vals is None:
                        continue
                    
                    # Calculate error metrics
                    abs_diff = abs(baseline_vals - compare_vals)
                    rel_diff = abs_diff / (abs(baseline_vals) + 1e-10)
                    
                    precision_summary.append({
                        'operation': op_col,
                        'baseline': baseline_lib,
                        'compared': compare_lib,
                        'max_abs_error': abs_diff.max(),
                        'mean_abs_error': abs_diff.mean(),
                        'max_rel_error': rel_diff.max(),
                        'mean_rel_error': rel_diff.mean(),
                        'rmse': ((abs_diff ** 2).mean() ** 0.5)
                    })
                except (ValueError, TypeError):
                    # Skip non-numeric columns
                    continue
    
    precision_df = pd.DataFrame(precision_summary)
    precision_df.to_csv("benchmark_results/comparison/precision_comparison.csv", index=False)
    print(f"   ✓ Saved: benchmark_results/comparison/precision_comparison.csv")
else:
    print("   ⚠ Need at least 2 libraries for precision comparison")

# ============================================================================
# PRINT SUMMARY
# ============================================================================
print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)

available_libs = []
for lib in LIBRARIES:
    if any(load_metric(lib, metric) is not None for metric in METRICS):
        available_libs.append(lib)

print(f"\nLibraries Available: {', '.join(available_libs)}")
print(f"\nGenerated Comparison Files:")
print(f"  1. timing_comparison.csv     - Execution time comparison")
print(f"  2. throughput_comparison.csv - Data throughput comparison")
print(f"  3. bandwidth_comparison.csv  - Memory bandwidth comparison")
print(f"  4. flops_comparison.csv      - GFLOPS comparison")
print(f"  5. precision_comparison.csv  - Numerical accuracy comparison")

# Quick preview of timing comparison
if len(timing_data) >= 2:
    print(f"\n" + "="*80)
    print("TIMING COMPARISON PREVIEW (First 10 operations)")
    print("="*80)
    print(f"\n{'Operation':<20}", end="")
    for lib in available_libs:
        if lib in timing_data:
            print(f"{lib:>15}", end="")
    print()
    print("-"*80)
    
    for _, row in timing_comp_df.head(10).iterrows():
        op = row['operation']
        print(f"{op:<20}", end="")
        
        for lib in available_libs:
            col = f'{lib}_mean_ms'
            if col in row and pd.notna(row[col]):
                print(f"{row[col]:>13.4f}ms", end="")
            else:
                print(f"{'N/A':>15}", end="")
        print()

print("\n" + "="*80)
print(" COMPARISON COMPLETE!")
print("="*80 + "\n")
