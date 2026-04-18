#!/bin/bash
# Build and run KernelBenchmarks.cu
# Uses the same pattern as run_sparse_ce_bench.sh

set -e

# Update these paths as needed

# LIBTORCH_PATH="/home/blubridge-028/Desktop/new_struct/libtorch"
# PROJ_ROOT="/home/blubridge-028/Desktop/new_struct/tensor"
LIBTORCH_PATH="/home/blubridge-036/pytorch_test_local/LIBTORCH_CUDA/libtorch"
PROJ_ROOT="/home/blubridge-036/Madhu_folder/Optimizations_tensor/tensor"

if [ ! -d "$LIBTORCH_PATH" ]; then
    echo "Error: LibTorch not found at $LIBTORCH_PATH"
    exit 1
fi

echo "=== Updating libtensor... ==="
make -C "$PROJ_ROOT" all -j$(nproc)

echo ""
echo "=== Compiling KernelBenchmarks... ==="
nvcc -std=c++20 -O3 -arch=sm_86 --extended-lambda --expt-relaxed-constexpr \
    -Iinclude \
    -IBenchmark \
    -I${LIBTORCH_PATH}/include \
    -I${LIBTORCH_PATH}/include/torch/csrc/api/include \
    -L${LIBTORCH_PATH}/lib \
    -Llib \
    -Xlinker -rpath -Xlinker ${LIBTORCH_PATH}/lib \
    -Xlinker -rpath -Xlinker lib \
    -Xlinker --no-as-needed -ltorch_cuda -ltorch_cpu -ltorch -lc10 -lc10_cuda \
    -ltensor -lcudart -ltbb -lcurand -lcublas -lnvidia-ml -ldl \
    Benchmark/KernelBenchmarks.cu \
    -o Benchmark/Benchmark_obj/KernelBenchmarks 

echo ""
echo "=== Compilation successful. Running benchmark... ==="
echo ""
./Benchmark/Benchmark_obj/KernelBenchmarks 
