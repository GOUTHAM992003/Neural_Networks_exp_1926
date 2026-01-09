#!/usr/bin/env python3
"""
This file is for reference only,You can run it manually  and not auto-called ,for any queries ,ask @Tensor&ops team
CSV Comparison Script for TensorLib vs LibTorch Benchmarks
Compares numerical results with configurable tolerance
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

def compare_csvs(tensorlib_csv, libtorch_csv, tolerance):
    """
    Compare two CSV files numerically
    
    Args:
        tensorlib_csv: Path to TensorLib output CSV
        libtorch_csv: Path to LibTorch output CSV
        tolerance: Maximum allowed absolute difference
    
    Returns:
        bool: True if all columns match within tolerance
    """
    
    print("═" * 60)
    print("CSV Comparison - TensorLib vs LibTorch")
    print("═" * 60)
    print(f"TensorLib: {tensorlib_csv}")
    print(f"LibTorch:  {libtorch_csv}")
    print(f"Tolerance: {tolerance:.2e}")
    print("═" * 60)
    print()
    
    # Read CSV files
    try:
        df_tensorlib = pd.read_csv(tensorlib_csv)
        df_libtorch = pd.read_csv(libtorch_csv)
    except Exception as e:
        print(f" ERROR: Failed to read CSV files: {e}")
        return False
    
    # Check shapes
    if df_tensorlib.shape != df_libtorch.shape:
        print(f" Shape mismatch:")
        print(f"   TensorLib: {df_tensorlib.shape}")
        print(f"   LibTorch:  {df_libtorch.shape}")
        return False
    
    print(f"  Shape matches: {df_tensorlib.shape[0]} rows × {df_tensorlib.shape[1]} columns")
    print()
    
    # Get numeric columns (exclude index, depth, row, col)
    numeric_cols = df_tensorlib.select_dtypes(include=[np.number]).columns.tolist()
    metadata_cols = ['index', 'depth', 'row', 'col']
    comparison_cols = [col for col in numeric_cols if col not in metadata_cols]
    
    print(f"Comparing {len(comparison_cols)} numerical columns...")
    print("─" * 60)
    
    all_match = True
    mismatches = []
    
    for col in comparison_cols:
        if col not in df_libtorch.columns:
            print(f" Column '{col}' missing in LibTorch output")
            all_match = False
            continue
        
        # Calculate absolute difference
        diff = np.abs(df_tensorlib[col] - df_libtorch[col])
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        # Check for NaN/Inf differences
        tensorlib_nan = df_tensorlib[col].isna().sum()
        libtorch_nan = df_libtorch[col].isna().sum()
        
        tensorlib_inf = np.isinf(df_tensorlib[col]).sum()
        libtorch_inf = np.isinf(df_libtorch[col]).sum()
        
        if tensorlib_nan != libtorch_nan:
            print(f"  Column '{col}': NaN count mismatch (TensorLib: {tensorlib_nan}, LibTorch: {libtorch_nan})")
            all_match = False
            mismatches.append(col)
            continue
        
        if tensorlib_inf != libtorch_inf:
            print(f"  Column '{col}': Inf count mismatch (TensorLib: {tensorlib_inf}, LibTorch: {libtorch_inf})")
            all_match = False
            mismatches.append(col)
            continue
        
        if max_diff > tolerance:
            print(f" Column '{col:<15}': max diff = {max_diff:.6e} (mean = {mean_diff:.6e})")
            all_match = False
            mismatches.append(col)
        else:
            status = "✓ "
            if max_diff > tolerance / 10:
                status = " "  # Warning if close to tolerance
            print(f"{status} Column '{col:<15}': max diff = {max_diff:.6e} (mean = {mean_diff:.6e})")
    
    print("─" * 60)
    print()
    
    if all_match:
        print("┌" + "─" * 58 + "┐")
        print("│" + " " * 58 + "│")
        print("│" + "   ALL RESULTS MATCH WITHIN TOLERANCE  ".center(58) + "│")
        print("│" + " " * 58 + "│")
        print("└" + "─" * 58 + "┘")
    else:
        print("┌" + "─" * 58 + "┐")
        print("│" + " " * 58 + "│")
        print("│" + f"   {len(mismatches)} COLUMNS EXCEED TOLERANCE  ".center(58) + "│")
        print("│" + " " * 58 + "│")
        print("└" + "─" * 58 + "┘")
        print()
        print("Columns with issues:")
        for col in mismatches:
            print(f"  - {col}")
    
    print()
    return all_match


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 compare_csvs.py <tensorlib.csv> <libtorch.csv> <tolerance>")
        print("Example: python3 compare_csvs.py tensorlib_out.csv libtorch_out.csv 1e-6")
        sys.exit(1)
    
    tensorlib_csv = Path(sys.argv[1])
    libtorch_csv = Path(sys.argv[2])
    tolerance = float(sys.argv[3])
    
    if not tensorlib_csv.exists():
        print(f" ERROR: TensorLib CSV not found: {tensorlib_csv}")
        sys.exit(1)
    
    if not libtorch_csv.exists():
        print(f" ERROR: LibTorch CSV not found: {libtorch_csv}")
        sys.exit(1)
    
    success = compare_csvs(tensorlib_csv, libtorch_csv, tolerance)
    sys.exit(0 if success else 1)
