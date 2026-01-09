#!/bin/bash
# Master Benchmark Script
# Runs all library benchmarks and generates comprehensive comparisons

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "================================================================================"
echo "                    COMPREHENSIVE BENCHMARK SUITE"
echo "================================================================================"
echo ""
echo "This will benchmark: NumPy, PyTorch, LibTorch, and your TensorLib"
echo "Generating 5 metric CSVs per library + 5 comparison CSVs"
echo ""

# Step 1: TensorLib (FIRST - generates shared input CSV for all other benchmarks)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. Running TensorLib Benchmark (generates shared inputs)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd "$PROJECT_ROOT"
if [ -f "scripts/run_tensorlib_benchmark.sh" ]; then
    echo "Running TensorLib benchmark..."
    ./scripts/run_tensorlib_benchmark.sh
else
    # Fallback: run directly
    echo "Running TensorLib directly..."
    (cd Tensor_Implementations_kota/Tensor_Implementations && make run-snippet FILE=Tests/benchmark/benchmark_all.cpp)
fi
echo ""

# Step 2: NumPy
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. Running NumPy Benchmark"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 numpy_tests/numpy_dk_ops.py
echo ""

# Step 3: PyTorch
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. Running PyTorch Benchmark"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 pytorch_tests/pytorch_fun_tests.py
echo ""

# Step 4: LibTorch
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. Running LibTorch Benchmark"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -d "libtorch_tests/benchmark" ]; then
    echo "Building and running LibTorch benchmarks..."
    (cd libtorch_tests/benchmark && make clean && make -j$(nproc) && ./libtorch_fun_tests)
   
else
    echo "⚠ LibTorch benchmark directory not found - skipping"
fi
echo ""


# Step 5: Generate Comparisons
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5. Generating CPU Metric Comparisons"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 scripts/compare_all_metrics.py
echo ""

# Step 6: CUDA Benchmarks (if GPU available)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6. Running CUDA GPU Benchmarks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check CUDA availability
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "✓ CUDA available - running GPU benchmarks..."
    
    # TensorLib CUDA
    echo "  Running TensorLib CUDA benchmark..."
    if (cd Tensor_Implementations_kota/Tensor_Implementations && make run-snippet FILE=Tests/benchmark/benchmark_cuda.cpp); then
        echo "  ✓ TensorLib CUDA benchmark completed"
    else
        echo "  ⚠ TensorLib CUDA benchmark failed"
    fi
    
    # PyTorch CUDA
    echo "  Running PyTorch CUDA benchmark..."
    if python3 pytorch_tests/pytorch_cuda_benchmark.py; then
        echo "  ✓ PyTorch CUDA benchmark completed"
    else
        echo "  ⚠ PyTorch CUDA benchmark failed"
    fi
    
    # LibTorch CUDA
    echo "  Running LibTorch CUDA benchmark..."
    if (cd libtorch_tests/benchmark && make clean && make cuda_bench_all_3d && ./cuda_bench_all_3d); then
        echo "  ✓ LibTorch CUDA benchmark completed"
    else
        echo "  ⚠ LibTorch CUDA benchmark failed"
    fi
    
    # Generate CUDA comparisons
    echo "  Generating CUDA metric comparisons..."
    python3 scripts/compare_cuda_metrics.py
else
    echo "⚠ CUDA not available - skipping GPU benchmarks"
    echo "  To enable CUDA benchmarks:"
    echo "    - Install CUDA toolkit"
    echo "    - Install PyTorch with CUDA support"
    echo "    - Ensure GPU is available"
fi
echo ""

echo "================================================================================"
echo "                           BENCHMARK COMPLETE!"
echo "================================================================================"
echo ""
echo "📊 Generated Files:"
echo ""
echo "CPU Per-Library Metrics (each library has 5 CSVs):"
echo "  • benchmark_results/numpy/       - numpy_values.csv, timings, throughput, bandwidth, flops"
echo "  • benchmark_results/pytorch/     - pytorch_values.csv, timings, throughput, bandwidth, flops"
echo "  • benchmark_results/libtorch/    - (same 5 files if available)"
echo "  • benchmark_results/tensorlib/   - (same 5 files if available)"
echo ""
echo "📈 CPU Comparison Files (all libraries side-by-side):"
echo "  • benchmark_results/comparison/"
echo "      - timing_comparison.csv      - CPU execution time comparison"
echo "      - throughput_comparison.csv  - CPU data throughput comparison"
echo "      - bandwidth_comparison.csv   - CPU memory bandwidth comparison"
echo "      - flops_comparison.csv       - CPU GFLOPS comparison"
echo "      - precision_comparison.csv   - Numerical accuracy comparison"
echo ""

# Check if CUDA results were generated
if [ -d "benchmark_results/pytorch_cuda" ] || [ -d "benchmark_results/comparison_cuda" ]; then
    echo "🚀 GPU (CUDA) Results:"
    echo "  • benchmark_results/pytorch_cuda/     - PyTorch GPU metrics (5 files)"
    echo "  • benchmark_results/libtorch_cuda/    - LibTorch GPU metrics (if available)"
    echo "  • benchmark_results/tensorlib_cuda/   - TensorLib GPU metrics (if available)"
    echo ""
    echo "🎮 GPU Comparison Files:"
    echo "  • benchmark_results/comparison_cuda/"
    echo "      - cuda_timing_comparison.csv      - GPU execution time comparison"
    echo "      - cuda_throughput_comparison.csv  - GPU data throughput comparison"
    echo "      - cuda_bandwidth_comparison.csv   - GPU memory bandwidth comparison"
    echo "      - cuda_flops_comparison.csv       - GPU GFLOPS comparison"
    echo ""
fi

echo "✅ All benchmarks complete!"
echo "================================================================================"
