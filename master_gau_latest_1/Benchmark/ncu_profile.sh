#!/bin/bash
# NCU Profiling Script for Custom vs PyTorch Kernels
#
# This script profiles KernelBenchmarks using Nsight Compute (ncu) to capture
# kernel-level metrics. It filters for specific kernel names and exports CSV
# data for post-processing.
#
# Usage:
#   sudo bash ncu_profile.sh              # Full profiling
#   sudo bash ncu_profile.sh --quick      # Quick run (fewer iterations)
#
# Requirements: NVIDIA Nsight Compute (ncu), root/admin permissions

set -e

PROJ_ROOT="/home/blubridge-028/Desktop/new_struct/tensor"
OUTPUT_DIR="${PROJ_ROOT}/Benchmark/ncu_results"
BINARY="${PROJ_ROOT}/Benchmark/Benchmark_obj/KernelBenchmarks"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUTPUT_DIR"

if [ ! -f "$BINARY" ]; then
    echo "Error: KernelBenchmarks binary not found. Run 'bash run_kernel_bench.sh' first."
    exit 1
fi

# Check for ncu (handle sudo PATH resetting)
NCU_BIN=$(command -v ncu || which ncu || echo "/usr/local/cuda/bin/ncu")

if [ ! -x "$NCU_BIN" ]; then
    echo "Error: ncu (Nsight Compute) not found at [ $NCU_BIN ]."
    echo "Please ensure CUDA toolkit is installed and ncu is in your PATH or at /usr/local/cuda/bin/ncu."
    exit 1
fi

echo "Using NCU at: $NCU_BIN"

echo "================================================================"
echo "       NCU PROFILING: Custom vs PyTorch Kernels"
echo "================================================================"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

echo "--- Profiling all launched kernels ---"
echo ""

# Skip warmup iterations (20 warmups * number of benchmarks ~ 120 kernel launches)
# Capture 100 launches of each kernel
LAUNCH_SKIP=0
LAUNCH_COUNT=999999  # Capture all

$NCU_BIN -o ${OUTPUT_DIR}/report_${TIMESTAMP} \
    --target-processes all \
    --metrics gpu__time_duration.sum,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed \
    $BINARY

echo ""
echo "================================================================"
echo "  NCU profiling complete!"
echo "  Raw CSV: ${OUTPUT_DIR}/csv/ncu_raw_${TIMESTAMP}.csv"
echo ""
echo "  To analyze results:"
echo "    python3 analyze_ncu.py ${OUTPUT_DIR}/csv/ncu_raw_${TIMESTAMP}.csv"
echo "================================================================"
